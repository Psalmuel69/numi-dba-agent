"""Cross-provider LLM fallback (agent/llm/fallback.py; ARCHITECTURE.md
"Cross-provider LLM fallback").

Reproduces the live failure this exists for: the default provider (Gemini)
exhausted its free-tier daily quota across every one of its OWN internal
fallback models, so `StructuredLLMProvider`'s retry loop and the provider's
model walk both ran out, and the DBA got "temporarily unavailable" while
real ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY sat configured
and unused in the same `.env`.

Entirely test-double-driven, matching tests/unit/test_llm_registry.py and
tests/unit/test_structured_llm_resilience.py: no API key is ever real, no
network call is ever made, and nothing here sleeps for longer than the
deadline test's own 50ms.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from numi.agent.context_manager import ContextManager
from numi.agent.llm import fallback
from numi.agent.llm.base import (
    OVERALL_DEADLINE_SECONDS,
    LLMProviderUnavailableError,
    StructuredLLMProvider,
)
from numi.agent.llm.fallback import (
    MIN_PER_ATTEMPT_BUDGET_SECONDS,
    CrossProviderFallbackLLM,
    FallbackEvent,
    fallback_notice_text,
    per_attempt_budget_seconds,
)
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import AskClarification, IntentExtraction, ProposeToolCall
from numi.common.config import Settings

_HEALTH = {
    "action": "propose_tool_call",
    "tool_id": "database.get_health",
    "reason": "Baseline check.",
}


class _StubProvider(StructuredLLMProvider):
    """A real `StructuredLLMProvider` subclass (so it goes through the same
    retry/deadline/validation plumbing every live provider does) whose only
    fake part is the SDK primitive itself."""

    _CALL_RETRY_DELAY_SECONDS = 0  # never sleep in a test

    def __init__(self, name: str, *, result: dict[str, Any] | None = None, error: Exception | None = None):
        super().__init__(f"{name}-model")
        self.provider_name = name
        self._result = result
        self._error = error
        self.call_count = 0

    async def _call_tool(self, *, system: str, user: str, schema: dict, tool_name: str) -> dict:
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return dict(self._result or {})

    async def _call_text(self, *, system: str, user: str) -> str:
        return "summary"


def _exhausted(name: str) -> _StubProvider:
    """The exact live failure: RESOURCE_EXHAUSTED on every model."""
    return _StubProvider(
        name,
        error=RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded for all models"),
    )


def _wrap(primary: _StubProvider, *fallbacks: _StubProvider, **kwargs) -> CrossProviderFallbackLLM:
    return CrossProviderFallbackLLM(
        primary, [(p.provider_name, (lambda p=p: p)) for p in fallbacks], **kwargs
    )


def _decide(llm, **overrides):
    kwargs = dict(
        problem_statement="check health",
        available_tool_ids=["database.get_health"],
        transcript=[],
        turn_count=0,
    )
    kwargs.update(overrides)
    return llm.decide_next_action(**kwargs)


# --- (a) a dead primary escapes to the next configured provider ----------


@pytest.mark.asyncio
async def test_a_total_primary_outage_falls_through_to_the_next_provider_and_succeeds():
    gemini = _exhausted("gemini")
    anthropic = _StubProvider("anthropic", result=_HEALTH)
    llm = _wrap(gemini, anthropic)

    action = await _decide(llm)

    assert isinstance(action, ProposeToolCall)
    assert action.tool_id == "database.get_health"
    # The primary really was given its full shot first (initial attempt +
    # its own retry) before anything escalated.
    assert gemini.call_count == gemini._CALL_RETRIES + 1
    assert anthropic.call_count == 1


@pytest.mark.asyncio
async def test_the_walk_keeps_going_past_a_second_dead_provider():
    llm = _wrap(_exhausted("gemini"), _exhausted("openai"), _StubProvider("deepseek", result=_HEALTH))
    assert isinstance(await _decide(llm), ProposeToolCall)


@pytest.mark.asyncio
async def test_extract_intent_falls_back_too_not_just_decide_next_action():
    gemini = _exhausted("gemini")
    anthropic = _StubProvider(
        "anthropic", result={"is_dba_task": True, "problem_summary": "CoreBanking is slow"}
    )
    llm = _wrap(gemini, anthropic)

    intent = await llm.extract_intent("CoreBanking is slow", known_database_names=["CoreBanking"])

    assert isinstance(intent, IntentExtraction)
    assert intent.problem_summary == "CoreBanking is slow"  # the real extraction, not the
    # best-effort "treat the raw message as a task" degradation
    assert anthropic.call_count == 1


@pytest.mark.asyncio
async def test_a_provider_that_cannot_even_be_built_is_skipped_not_fatal():
    """A key that is present but rejected by its own SDK constructor must
    behave like any other failed provider — skipped mid-walk — never an
    exception that takes down a call the next provider could have served."""

    def _explode() -> Any:
        raise ValueError("No API key configured for provider 'openai'.")

    working = _StubProvider("deepseek", result=_HEALTH)
    llm = CrossProviderFallbackLLM(
        _exhausted("gemini"),
        [("openai", _explode), ("deepseek", (lambda: working))],
    )
    assert isinstance(await _decide(llm), ProposeToolCall)


# --- (b) everything down: the pre-existing degraded reply, unchanged -----


@pytest.mark.asyncio
async def test_when_every_provider_fails_the_existing_degraded_reply_still_fires():
    """No-regression pin: the DBA-facing wording when nothing can be
    reached is exactly what a bare provider produced before this layer
    existed, and it names the provider the DBA actually chose — not
    whichever vendor happened to be last in the chain."""
    bare = _exhausted("gemini")
    expected = await _decide(bare)
    assert isinstance(expected, AskClarification)

    llm = _wrap(_exhausted("gemini"), _exhausted("anthropic"), _exhausted("openai"))
    action = await _decide(llm)

    assert isinstance(action, AskClarification)
    assert action.question == expected.question
    assert "gemini" in action.question and "temporarily unavailable" in action.question
    assert "anthropic" not in action.question


@pytest.mark.asyncio
async def test_extract_intent_still_degrades_to_best_effort_when_everything_is_down():
    llm = _wrap(_exhausted("gemini"), _exhausted("anthropic"))
    intent = await llm.extract_intent("  CoreBanking is slow  ", known_database_names=[])
    assert intent.is_dba_task is True
    assert intent.problem_summary == "CoreBanking is slow"


@pytest.mark.asyncio
async def test_a_malformed_completion_is_not_escalated_to_another_vendor():
    """A provider that ANSWERED and merely failed validation is a
    prompt/schema problem, not an outage — a different vendor is no more
    likely to get it right, and it already has its own distinct degraded
    reply (see tests/unit/test_structured_llm_resilience.py). Spending a
    second vendor's slice of the deadline on it would only add latency."""
    primary = _StubProvider("gemini", result={"action": "propose_tool_call", "tool_id": "x"})  # no reason
    never = _StubProvider("anthropic", result=_HEALTH)
    llm = _wrap(primary, never)

    action = await _decide(llm)

    assert isinstance(action, AskClarification)
    assert "couldn't work out a safe next step" in action.question
    assert never.call_count == 0


# --- (c) the time budget ------------------------------------------------


def test_the_per_attempt_budget_divides_the_existing_ceiling_never_multiplies_it():
    """The guarantee in ARCHITECTURE.md's "Latency ceiling on a single LLM
    decision" is a property of a DECISION, not of a provider: one decision
    is never worse than ~20s late. Four providers at the full ceiling each
    would make that 80s."""
    assert per_attempt_budget_seconds(OVERALL_DEADLINE_SECONDS, 1) == OVERALL_DEADLINE_SECONDS
    assert per_attempt_budget_seconds(OVERALL_DEADLINE_SECONDS, 2) == 10.0
    assert per_attempt_budget_seconds(OVERALL_DEADLINE_SECONDS, 4) == 5.0
    for attempts in range(1, 5):  # the whole range this codebase can produce
        assert per_attempt_budget_seconds(OVERALL_DEADLINE_SECONDS, attempts) * attempts <= (
            OVERALL_DEADLINE_SECONDS + 1e-9
        )


def test_the_floor_stops_a_fifth_provider_from_shrinking_every_attempt_to_uselessness():
    assert per_attempt_budget_seconds(OVERALL_DEADLINE_SECONDS, 8) == MIN_PER_ATTEMPT_BUDGET_SECONDS


def test_the_floor_itself_never_exceeds_the_total_it_is_dividing():
    """A deliberately small ceiling degrades honestly — the primary gets
    everything and there was never room for a second attempt — rather than
    promising a 5s slice out of a 3s budget."""
    assert per_attempt_budget_seconds(3.0, 2) == 3.0


@pytest.mark.asyncio
async def test_each_attempt_is_given_its_divided_share_of_the_one_shared_deadline():
    """Asserted on the timeouts actually handed to each attempt (via an
    injected runner), so this proves the real budgeting without waiting on
    any of it."""
    budgets: list[float] = []

    async def _recording_runner(awaitable, timeout):
        budgets.append(timeout)
        return await awaitable

    llm = _wrap(
        _exhausted("gemini"),
        _exhausted("anthropic"),
        _exhausted("openai"),
        timeout_runner=_recording_runner,
    )
    await _decide(llm)

    assert len(budgets) == 3
    assert all(b == pytest.approx(OVERALL_DEADLINE_SECONDS / 3) for b in budgets)
    assert sum(budgets) <= OVERALL_DEADLINE_SECONDS + 1e-9


@pytest.mark.asyncio
async def test_an_attempt_is_clamped_to_the_time_actually_left_and_the_walk_stops_at_zero():
    """A fake clock, so no test ever waits out a real budget: the primary
    "takes" 19.5s, leaving the second attempt only the 0.5s that remains
    (not its nominal 5s share), and the fourth provider is never tried at
    all because the shared deadline is gone."""
    ticks = iter([0.0, 0.0, 19.5, 21.0])
    budgets: list[float] = []

    async def _recording_runner(awaitable, timeout):
        budgets.append(timeout)
        return await awaitable

    llm = _wrap(
        _exhausted("gemini"),
        _exhausted("anthropic"),
        _exhausted("openai"),
        _exhausted("deepseek"),
        clock=(lambda: next(ticks)),
        timeout_runner=_recording_runner,
    )
    action = await _decide(llm)

    assert budgets == [5.0, 0.5]  # nominal share, then whatever was left
    assert isinstance(action, AskClarification)  # and then the unchanged degraded reply


@pytest.mark.asyncio
async def test_a_hanging_provider_really_is_cut_off_and_the_next_one_answers(monkeypatch):
    """The one test that exercises the real `asyncio.wait_for` path rather
    than an injected runner — a provider that hangs must not be able to eat
    the whole deadline. The ceiling and the per-attempt floor are scaled
    down together (0.2s / 0.05s, keeping their real 4:1 shape) so this runs
    in milliseconds; scaling only the ceiling would hit the honest clamp in
    `per_attempt_budget_seconds` and give the primary everything."""
    monkeypatch.setattr(fallback, "MIN_PER_ATTEMPT_BUDGET_SECONDS", 0.05)

    class _Hangs(_StubProvider):
        async def _call_tool(self, *, system, user, schema, tool_name):
            self.call_count += 1
            await asyncio.sleep(10)  # never actually reached
            return {}

    hangs = _Hangs("gemini")
    anthropic = _StubProvider("anthropic", result=_HEALTH)
    llm = _wrap(hangs, anthropic, total_deadline_seconds=0.2)

    start = time.monotonic()
    action = await _decide(llm)
    elapsed = time.monotonic() - start

    assert isinstance(action, ProposeToolCall)  # the fallback answered
    assert elapsed < 2.0  # nowhere near the 10s hang or a 20s ceiling


# --- (d) disclosure ------------------------------------------------------


def test_the_notice_names_the_failed_service_and_the_one_actually_used():
    assert fallback_notice_text([]) == ""
    text = fallback_notice_text([FallbackEvent(failed_providers=["gemini"], used_provider="anthropic")])
    assert text == "(The gemini service was unavailable, so I used anthropic instead.)"


def test_the_notice_is_deduplicated_across_the_several_calls_one_message_makes():
    """An investigation turn calls decide_next_action repeatedly; "gemini
    was unavailable" is one fact about the reply, not one per call."""
    events = [
        FallbackEvent(failed_providers=["gemini"], used_provider="anthropic"),
        FallbackEvent(failed_providers=["gemini"], used_provider="anthropic"),
    ]
    assert fallback_notice_text(events) == (
        "(The gemini service was unavailable, so I used anthropic instead.)"
    )


def test_the_notice_reads_correctly_when_several_services_were_down():
    text = fallback_notice_text(
        [FallbackEvent(failed_providers=["gemini", "openai"], used_provider="anthropic")]
    )
    assert text == "(The gemini and openai services were unavailable, so I used anthropic instead.)"


@pytest.mark.asyncio
async def test_a_successful_primary_discloses_nothing():
    notices: list[FallbackEvent] = []
    llm = _wrap(_StubProvider("gemini", result=_HEALTH), _StubProvider("anthropic"), notices=notices)
    await _decide(llm)
    assert notices == []
    assert fallback_notice_text(notices) == ""


@pytest.mark.asyncio
async def test_a_total_outage_discloses_nothing_either_because_nothing_was_substituted():
    notices: list[FallbackEvent] = []
    llm = _wrap(_exhausted("gemini"), _exhausted("anthropic"), notices=notices)
    await _decide(llm)
    assert notices == []


# --- registry wiring -----------------------------------------------------


def _registry(**overrides) -> LLMRegistry:
    return LLMRegistry(Settings(_env_file=None, **overrides))


def _stubbed_registry(stubs: dict[str, _StubProvider], **overrides) -> LLMRegistry:
    """A real registry (real `configured_llm_providers` ordering, real
    default resolution, real lock handling) whose only fake part is the SDK
    construction — so nothing here can reach a network."""
    registry = _registry(**{f"{name}_api_key": f"key-{name}" for name in stubs}, **overrides)

    def _construct(provider: str, model: str):
        if provider not in stubs:
            # Same failure the real `_construct` raises for a provider with
            # no key — the stale-selection path below depends on it.
            raise ValueError(f"No API key configured for provider '{provider}'.")
        return stubs[provider]

    registry._construct = _construct  # type: ignore[method-assign]
    return registry


def test_a_single_configured_provider_is_not_wrapped_at_all():
    """(e) Today's actual common case must behave identically to before
    this layer existed — same object, same single full-ceiling budget, no
    walk to take."""
    reg = _registry(gemini_api_key="g")
    llm = reg.resilient_for_conversation(provider=None, model=None)
    assert not isinstance(llm, CrossProviderFallbackLLM)
    assert llm is reg.for_conversation(provider=None, model=None)


def test_the_offline_mock_planner_is_never_wrapped_even_with_every_key_configured():
    """(4) `llm_provider="mock"` is what the whole default test suite runs
    on: it must never attempt fallback logic or reach real provider code."""
    reg = _registry(
        llm_provider="mock",
        anthropic_api_key="a",
        openai_api_key="o",
        gemini_api_key="g",
        deepseek_api_key="d",
    )
    llm = reg.resilient_for_conversation(provider=None, model=None)
    assert isinstance(llm, MockLLMProvider)
    assert not isinstance(llm, CrossProviderFallbackLLM)


def test_no_keys_at_all_still_lands_on_the_offline_planner():
    llm = _registry().resilient_for_conversation(provider=None, model=None)
    assert isinstance(llm, MockLLMProvider)


async def test_a_pinned_test_provider_is_never_wrapped():
    """`for_testing` pins one provider for every conversation — nothing to
    fall back to, and the pinned object is often a bare duck-typed double
    rather than an `LLMProvider` subclass. Mirrors
    test_llm_registry.py::test_for_testing_pins_one_provider_and_disables_selection."""
    pinned = MockLLMProvider()
    reg = LLMRegistry.for_testing(pinned)
    assert reg.resilient_for_conversation(provider="openai", model="x") is pinned


def test_the_chain_is_preference_ordered_and_skips_the_primary_and_unconfigured_keys():
    stubs = {n: _StubProvider(n) for n in ("anthropic", "gemini")}  # openai/deepseek unconfigured
    reg = _stubbed_registry(stubs, llm_provider="gemini")
    llm = reg.resilient_for_conversation(provider=None, model=None)
    assert isinstance(llm, CrossProviderFallbackLLM)
    assert [name for name, _ in llm._fallbacks] == ["anthropic"]


def test_a_fallback_provider_uses_its_own_default_model_not_the_primarys():
    """A model id is provider-specific — "gemini-3.5-flash" means nothing
    to Anthropic, so a conversation's `/model` choice must not be carried
    across the vendor boundary."""
    built: list[tuple[str, str | None]] = []
    reg = _registry(anthropic_api_key="a", gemini_api_key="g", llm_provider="gemini")
    stubs = {"anthropic": _StubProvider("anthropic"), "gemini": _StubProvider("gemini")}

    def _construct(provider, model):
        built.append((provider, model))
        return stubs[provider]

    reg._construct = _construct  # type: ignore[method-assign]
    llm = reg.resilient_for_conversation(provider="gemini", model="gemini-3.5-flash")
    assert isinstance(llm, CrossProviderFallbackLLM)
    [factory() for _, factory in llm._fallbacks]
    assert built == [("gemini", "gemini-3.5-flash"), ("anthropic", "")]


def test_a_stale_conversation_selection_still_gets_a_fallback_chain():
    """Extends (rather than duplicates) test_llm_registry.py::
    test_for_conversation_falls_back_to_default_on_a_stale_choice: a
    conversation pinned to a provider whose key has since been removed
    resolves to the default, and that default must still get a chain built
    around the providers that ARE configured — the stale-selection path was
    the one place `for_conversation` could return a provider other than the
    one asked for."""
    stubs = {n: _StubProvider(n) for n in ("anthropic", "gemini")}
    reg = _stubbed_registry(stubs)
    llm = reg.resilient_for_conversation(provider="openai", model="gpt-4o")  # key since removed
    assert isinstance(llm, CrossProviderFallbackLLM)
    assert llm.provider_name == "anthropic"  # the default, per LLM_PROVIDER_PREFERENCE
    assert [name for name, _ in llm._fallbacks] == ["gemini"]


def test_a_locked_provider_still_gets_a_fallback_chain():
    """A locked `LLM_PROVIDER` disables *DBA-driven* switching, not the
    outage escape hatch: an answer from another configured vendor beats
    telling a DBA the agent is stuck while a usable key sits idle. The
    disclosure below is what keeps that honest."""
    stubs = {n: _StubProvider(n) for n in ("anthropic", "gemini")}
    reg = _stubbed_registry(stubs, llm_provider="gemini")
    assert reg.selection_enabled() is False
    llm = reg.resilient_for_conversation(provider=None, model=None)
    assert isinstance(llm, CrossProviderFallbackLLM)
    assert llm.provider_name == "gemini"


# --- end to end through the real orchestrator ---------------------------


class _FakeToolClient:
    async def list_servers(self):
        return []

    async def available_tools(self, channel, channel_account_id):
        return []

    async def create_investigation(self, request):
        pass

    async def update_investigation(self, investigation_id, request):
        pass

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        return []

    async def log_decision_event(self, request):
        pass


@pytest.mark.asyncio
async def test_the_dba_facing_reply_says_which_provider_actually_answered():
    """(d) end to end: a locked-to-gemini deployment whose gemini quota is
    exhausted still answers — and the reply says plainly that anthropic
    produced it. Never a silent substitution."""
    stubs = {
        "gemini": _exhausted("gemini"),
        "anthropic": _StubProvider("anthropic", result={"is_dba_task": False, "meta_command": "help"}),
    }
    orchestrator = AgentOrchestrator(
        llm_registry=_stubbed_registry(stubs, llm_provider="gemini"),
        tool_client=_FakeToolClient(),
        context=ContextManager(),
    )

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U1",
        conversation_id="conv-fallback",
        channel_thread_id="",
        message="what can you do",
    )

    assert "I'm Numi, your AI DBA assistant." in reply.text  # the real answer still arrived
    assert reply.text.endswith("(The gemini service was unavailable, so I used anthropic instead.)")


@pytest.mark.asyncio
async def test_a_healthy_primary_reply_carries_no_notice_and_the_sink_does_not_leak():
    """The notice is per-inbound-message scratch space: a message answered
    by the primary must carry nothing, including right after one that did
    fall back."""
    gemini = _StubProvider("gemini", result={"is_dba_task": False, "meta_command": "help"})
    orchestrator = AgentOrchestrator(
        llm_registry=_stubbed_registry(
            {"gemini": gemini, "anthropic": _StubProvider("anthropic")}, llm_provider="gemini"
        ),
        tool_client=_FakeToolClient(),
        context=ContextManager(),
    )

    async def _send() -> str:
        reply = await orchestrator.handle_message(
            channel="slack",
            channel_account_id="U1",
            conversation_id="conv-healthy",
            channel_thread_id="",
            message="what can you do",
        )
        return reply.text

    assert "unavailable" not in await _send()
    assert "unavailable" not in await _send()


@pytest.mark.asyncio
async def test_the_orchestrator_still_reports_a_total_outage_plainly():
    """No-regression through the full stack: when nothing can be reached,
    the DBA gets the same "try again in a moment" they always did, with no
    fallback notice bolted onto it."""
    context = ContextManager()
    orchestrator = AgentOrchestrator(
        llm_registry=_stubbed_registry(
            {"gemini": _exhausted("gemini"), "anthropic": _exhausted("anthropic")},
            llm_provider="gemini",
        ),
        tool_client=_FakeToolClient(),
        context=context,
    )
    # An investigation already under way with its environment settled, so
    # this lands on `decide_next_action` — the call that actually reports an
    # outage to the DBA — rather than on the environment gate that runs
    # first for a brand-new request.
    state = context.get_or_create("conv-down", "slack", "", "U1")
    state.database_context["environment"] = "development"
    context.start_investigation(state, "CoreBanking is slow")

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U1",
        conversation_id="conv-down",
        channel_thread_id="",
        message="any update on that",
    )

    assert "temporarily unavailable" in reply.text
    assert "gemini" in reply.text
    assert "so I used" not in reply.text


# --- the raising variants the walk is built on ---------------------------


@pytest.mark.asyncio
async def test_or_raise_distinguishes_an_unreachable_provider_from_a_bad_answer():
    with pytest.raises(LLMProviderUnavailableError):
        await _exhausted("gemini").decide_next_action_or_raise(
            problem_statement="check health",
            available_tool_ids=["database.get_health"],
            transcript=[],
            turn_count=0,
        )
    # Answered, but unusable: NOT an availability failure.
    malformed = _StubProvider("gemini", result={"action": "propose_tool_call", "tool_id": "x"})
    action = await malformed.decide_next_action_or_raise(
        problem_statement="check health",
        available_tool_ids=["database.get_health"],
        transcript=[],
        turn_count=0,
    )
    assert isinstance(action, AskClarification)


@pytest.mark.asyncio
async def test_the_offline_planner_can_never_start_a_walk():
    """(4) again, at the interface level: the mock's `*_or_raise` variants
    are the inherited pass-throughs, so even a mock wrapped by hand could
    not trigger a fallback."""
    mock = MockLLMProvider()
    action = await mock.decide_next_action_or_raise(
        problem_statement="check health",
        available_tool_ids=["database.get_health"],
        transcript=[],
        turn_count=0,
    )
    assert action is not None
    never = _StubProvider("anthropic", result=_HEALTH)
    llm = CrossProviderFallbackLLM(mock, [("anthropic", (lambda: never))])
    await _decide(llm)
    assert never.call_count == 0


@pytest.mark.asyncio
async def test_critique_conclusion_delegates_to_the_primary_only():
    """Deliberately NOT part of the fallback walk (see the method's own
    docstring) — a fallback provider configured alongside the primary must
    never be touched by a critique call."""
    primary = _StubProvider("gemini", result={"sound": False, "issue": "no evidence for this"})
    fallback_provider = _StubProvider("anthropic", result={"sound": True})
    llm = _wrap(primary, fallback_provider)

    verdict = await llm.critique_conclusion(
        problem_statement="p",
        transcript=[],
        proposed_summary="s",
        proposed_root_cause=None,
        proposed_confidence="unable_to_confirm",
        proposed_recommendation=None,
    )

    assert verdict.sound is False
    assert verdict.issue == "no evidence for this"
    assert primary.call_count == 1
    assert fallback_provider.call_count == 0
