"""Cross-provider LLM fallback (spec §34; ARCHITECTURE.md "Cross-provider
LLM fallback").

The gap this closes, confirmed live: this deployment's default provider
(Gemini) exhausted its free-tier daily quota — `RESOURCE_EXHAUSTED`, 20
requests/day/model, across every one of Gemini's *own* internal fallback
models in turn (gemini-3.5-flash → 3.6-flash → 3.7-flash → 3.8-flash →
3.1-pro-preview → 3-flash-preview → 3.1-flash-lite; see
`gemini_provider._MODEL_FALLBACKS` and its `gemini_model_unavailable_
switching` log line). Everything below the provider boundary worked
exactly as designed: the SDK retried, `StructuredLLMProvider._call_with_
retry` retried, the provider walked its own model list, and then the DBA
got "The gemini service is temporarily unavailable — please try again in a
moment." What nobody could do was step *sideways*: real ANTHROPIC_API_KEY,
OPENAI_API_KEY and DEEPSEEK_API_KEY were all sitting configured in the same
`.env`, completely unused, while the agent told a DBA it was stuck.

`CrossProviderFallbackLLM` is that sideways step. It wraps the provider a
conversation actually resolved to (its `/model` choice, or the deployment
default) and, only when that provider proves *unreachable* for one specific
call, spends the remaining time budget on the next configured provider
family from `Settings.configured_llm_providers()` — the same
preference-ordered, key-filtered list `/models` and the default resolution
already use, so a provider with no API key is never even considered.

Three properties this layer is deliberately built around:

1. **It is never silent.** Every call that only succeeded because a
   different vendor answered records a `FallbackEvent` on the caller's
   sink, and `fallback_notice_text` turns those into one plain sentence
   appended to the DBA's reply ("The gemini service was unavailable, so I
   used anthropic instead."). A locked `LLM_PROVIDER` or an explicit
   `/model` choice does NOT disable the fallback — an answer beats
   "temporarily unavailable" with three unused keys on disk — but the DBA
   is always told which service actually produced the reply, because a
   silently substituted model is a different answer than the one they
   asked for.

2. **It fits inside the existing latency ceiling, it does not stack on
   top of it.** See `per_attempt_budget_seconds` for the full arithmetic.

3. **It cannot touch the offline path.** The wrapper is only ever built by
   `LLMRegistry.resilient_for_conversation`, which returns the bare
   provider untouched when the resolved provider is the deterministic
   `MockLLMProvider` or when there is nothing else configured to fall back
   to — so `llm_provider="mock"` (the whole default test suite) and the
   single-provider deployments that are today's common case both run
   exactly the code they ran before this module existed. Even if one were
   wrapped, `LLMProvider.*_or_raise` only raises from
   `StructuredLLMProvider`, so the mock could never start a walk.

Tested in tests/unit/test_cross_provider_fallback.py — entirely with stub
providers, no network, no real sleeps.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from numi.agent.llm.base import (
    OVERALL_DEADLINE_SECONDS,
    LLMProvider,
    LLMProviderUnavailableError,
)
from numi.agent.planner.actions import (
    AgentAction,
    AskClarification,
    CritiqueVerdict,
    IntentExtraction,
)
from numi.common.observability import get_logger

logger = get_logger(__name__)

# The smallest slice of the overall deadline worth handing to one provider.
# A cross-provider attempt costs a fresh TLS handshake plus a full
# completion; below a few seconds it would reliably time out and only burn
# budget the *next* provider could have used.
#
# This floor does not bind for any deployment this codebase can currently
# have: `LLM_PROVIDER_PREFERENCE` has exactly four entries, and
# 20.0 / 4 == 5.0. It exists so that adding a fifth provider family can
# never silently shrink every attempt into uselessness — the arithmetic in
# `per_attempt_budget_seconds` would then hand out four viable 5s attempts
# and let the overall clamp cut the walk short, rather than five
# guaranteed-to-fail 4s ones.
MIN_PER_ATTEMPT_BUDGET_SECONDS = 5.0


def per_attempt_budget_seconds(total_seconds: float, attempts: int) -> float:
    """How long ONE provider in the walk may take.

    The whole point is that cross-provider fallback must not add a second
    ~20s ceiling per provider on top of the existing one. POLICY_MODEL.md
    and ARCHITECTURE.md's "Latency ceiling on a single LLM decision" state
    the guarantee as a property of *a decision*, not of a provider: "a
    single investigative decision is never worse than ~20s late no matter
    how many retries or model fallbacks happen underneath". Four providers
    at the full ceiling each would make that up to 80s and break it.

    So the ceiling is divided, not repeated:

        attempts | per attempt | worst-case total
        ---------+-------------+------------------
               1 |       20.0s | 20s  (identical to a deployment with one
                 |             |       configured provider today)
               2 |       10.0s | 20s
               3 |        6.7s | 20s
               4 |        5.0s | 20s

    `attempts` counts the primary itself plus every configured fallback,
    because the primary's own attempt comes out of the same budget — it is
    the one most likely to answer, but it is not entitled to spend the
    entire allowance and leave nothing for the escape hatch.

    The caller additionally clamps each attempt to the time actually
    remaining, so even a floored budget (see
    `MIN_PER_ATTEMPT_BUDGET_SECONDS`) can never push the sum past
    `total_seconds` — and the floor itself is clamped to the total here, so
    that a deliberately small ceiling (a test, or a future deployment that
    tightens it) degrades honestly to "the primary gets everything and
    there was never room for a second attempt" rather than pretending to
    hand out slices the clock cannot pay for.
    """
    if attempts <= 1:
        return total_seconds
    return min(total_seconds, max(MIN_PER_ATTEMPT_BUDGET_SECONDS, total_seconds / attempts))


@dataclasses.dataclass
class FallbackEvent:
    """One call that only succeeded because a different vendor answered.

    Recorded on the sink list the caller passed in (the orchestrator keeps
    one per inbound message on `ConversationState.llm_fallback_notices`), so
    the DBA-facing reply can disclose the substitution. Never recorded when
    the primary answered normally, and never recorded when *everything*
    failed — in that case no provider was used and the reply is the
    unchanged "temporarily unavailable" message naming the primary.
    """

    failed_providers: list[str]
    used_provider: str


def fallback_notice_text(events: Sequence[FallbackEvent]) -> str:
    """One plain sentence for the DBA, or "" when nothing was substituted.

    Deduplicated and flattened across however many LLM calls one inbound
    message made (an investigation turn can call `decide_next_action`
    several times), because "the gemini service was unavailable" is one
    fact about this reply, not one fact per call. Matches the house voice
    of `_HELP_TEXT` / `_APPROVAL_MODEL_TEXT` in `agent.orchestrator`: plain,
    first-person, no vendor marketing, no apology loop.
    """
    if not events:
        return ""
    failed: list[str] = []
    used: list[str] = []
    for event in events:
        for name in event.failed_providers:
            if name not in failed:
                failed.append(name)
        if event.used_provider not in used:
            used.append(event.used_provider)
    if not failed or not used:
        return ""
    failed_phrase = _join(failed)
    was_were = "was" if len(failed) == 1 else "were"
    service_s = "service" if len(failed) == 1 else "services"
    return (
        f"(The {failed_phrase} {service_s} {was_were} unavailable, so I used "
        f"{_join(used)} instead.)"
    )


def _join(names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


class CrossProviderFallbackLLM(LLMProvider):
    """A vendor-neutral `LLMProvider` that walks a chain of real ones.

    Constructed by `LLMRegistry.resilient_for_conversation`, once per
    inbound message rather than cached, so `_notices` is plain per-message
    state with no sharing between concurrent conversations (the underlying
    provider objects are still the registry's cached, reused ones — only
    this thin wrapper is per-message).
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Sequence[tuple[str, Callable[[], LLMProvider]]],
        *,
        notices: list[FallbackEvent] | None = None,
        total_deadline_seconds: float = OVERALL_DEADLINE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        timeout_runner: Callable[[Awaitable[Any], float], Awaitable[Any]] = asyncio.wait_for,
    ) -> None:
        super().__init__(primary.model)
        self.provider_name = primary.provider_name
        self._primary = primary
        # (name, factory) rather than built instances: a provider whose key
        # is present but whose SDK constructor rejects it must not break
        # the primary path at construction time — it is skipped mid-walk
        # instead, with its name still available to log.
        self._fallbacks = list(fallbacks)
        self._notices = notices
        self._total_deadline_seconds = total_deadline_seconds
        self._clock = clock
        # Injected so tests can assert the per-attempt budgets exactly
        # without ever sleeping; production always uses asyncio.wait_for.
        self._timeout_runner = timeout_runner

    # -- the walk ---------------------------------------------------------

    async def _walk(
        self,
        *,
        what: str,
        call: Callable[[LLMProvider], Awaitable[Any]],
    ) -> Any:
        """Try the primary, then each configured fallback, inside one shared
        deadline. Raises `LLMProviderUnavailableError` naming the PRIMARY if
        every provider in the chain failed — so the caller's degraded reply
        is exactly the one it always produced, naming the provider the DBA
        actually chose, not whichever vendor happened to be last in the
        chain."""
        chain: list[tuple[str, Callable[[], LLMProvider]]] = [
            (self._primary.provider_name, lambda: self._primary),
            *self._fallbacks,
        ]
        budget = per_attempt_budget_seconds(self._total_deadline_seconds, len(chain))
        started = self._clock()
        failed: list[str] = []
        last_error: Exception | None = None

        for name, factory in chain:
            remaining = self._total_deadline_seconds - (self._clock() - started)
            if remaining <= 0:
                logger.warning(
                    "llm_cross_provider_budget_exhausted",
                    what=what,
                    primary=self._primary.provider_name,
                    tried=failed,
                    not_tried=name,
                )
                break
            try:
                provider = factory()
            except Exception as exc:  # noqa: BLE001 — a provider that can't even be
                # built (key present but rejected by its SDK) is skipped like
                # any other failure; it must never take down the walk.
                logger.warning(
                    "llm_cross_provider_build_failed", provider=name, what=what, error=str(exc)
                )
                failed.append(name)
                last_error = exc
                continue
            try:
                result = await self._timeout_runner(call(provider), min(budget, remaining))
            except (LLMProviderUnavailableError, TimeoutError) as exc:
                logger.warning(
                    "llm_cross_provider_attempt_failed",
                    provider=name,
                    what=what,
                    budget_seconds=min(budget, remaining),
                    error=str(exc),
                )
                failed.append(name)
                last_error = exc
                continue
            if failed:
                # Only reached when a DIFFERENT vendor answered — this is
                # what the DBA is told about. `failed` is the providers
                # tried and lost, in the order they were tried.
                logger.info(
                    "llm_cross_provider_fallback_used",
                    what=what,
                    failed=failed,
                    used=name,
                )
                if self._notices is not None:
                    self._notices.append(
                        FallbackEvent(failed_providers=list(failed), used_provider=name)
                    )
            return result

        raise LLMProviderUnavailableError(
            self._primary.provider_name,
            what,
            last_error or RuntimeError("no configured LLM provider could be reached"),
        )

    # -- LLMProvider ------------------------------------------------------

    async def extract_intent(
        self,
        message: str,
        known_database_names: list[str],
        known_server_hints: list[str] | None = None,
    ) -> IntentExtraction:
        try:
            return await self.extract_intent_or_raise(
                message, known_database_names, known_server_hints
            )
        except LLMProviderUnavailableError:
            # Identical degradation to a bare provider's (see
            # `StructuredLLMProvider.extract_intent`): treat the raw message
            # as a DBA task and let `decide_next_action` be the place that
            # actually reports the outage to the DBA.
            return IntentExtraction(is_dba_task=True, problem_summary=message.strip())

    async def extract_intent_or_raise(
        self,
        message: str,
        known_database_names: list[str],
        known_server_hints: list[str] | None = None,
    ) -> IntentExtraction:
        return await self._walk(
            what="extract_intent",
            call=lambda provider: provider.extract_intent_or_raise(
                message, known_database_names, known_server_hints
            ),
        )

    async def decide_next_action(
        self,
        *,
        problem_statement: str,
        available_tool_ids: list[str],
        transcript: list[dict[str, Any]],
        turn_count: int,
        tool_requirements: dict[str, list[str]] | None = None,
    ) -> AgentAction:
        try:
            return await self.decide_next_action_or_raise(
                problem_statement=problem_statement,
                available_tool_ids=available_tool_ids,
                transcript=transcript,
                turn_count=turn_count,
                tool_requirements=tool_requirements,
            )
        except LLMProviderUnavailableError:
            # Every configured provider failed. The DBA gets the exact
            # message they always got, naming the provider they chose —
            # this path is a strict no-regression of the pre-fallback
            # behaviour, not a new one.
            return AskClarification(question=self._primary._unavailable_question())

    async def decide_next_action_or_raise(
        self,
        *,
        problem_statement: str,
        available_tool_ids: list[str],
        transcript: list[dict[str, Any]],
        turn_count: int,
        tool_requirements: dict[str, list[str]] | None = None,
    ) -> AgentAction:
        return await self._walk(
            what="decide_next_action",
            call=lambda provider: provider.decide_next_action_or_raise(
                problem_statement=problem_statement,
                available_tool_ids=available_tool_ids,
                transcript=transcript,
                turn_count=turn_count,
                tool_requirements=tool_requirements,
            ),
        )

    async def summarize_for_human(
        self, *, problem_statement: str, transcript: list[dict[str, Any]]
    ) -> str:
        """Deliberately NOT part of the fallback walk. A summary is a
        nice-to-have restatement of material the investigation already
        gathered and already reported structurally (see
        `orchestrator._format_report`) — it is never the step that leaves a
        DBA stuck, so it is not worth another vendor's slice of a deadline
        that exists to bound *decisions*."""
        return await self._primary.summarize_for_human(
            problem_statement=problem_statement, transcript=transcript
        )

    async def list_models(self) -> list[str]:
        return await self._primary.list_models()

    async def critique_conclusion(self, **kwargs) -> CritiqueVerdict:
        """Also deliberately NOT part of the fallback walk, for the same
        reason as `summarize_for_human` just above: a critique failing is
        "skip it, accept the conclusion" (see
        `orchestrator._self_critique_conclude`), never "the DBA gets
        stuck" — not worth another vendor's slice of the decision-bounding
        deadline. `orchestrator.py` catches whatever this raises anyway."""
        return await self._primary.critique_conclusion(**kwargs)
