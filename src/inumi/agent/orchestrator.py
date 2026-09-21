"""Agent orchestrator (spec §6, §7, §53, §54).

Ties the LLM's *proposals* to the Gateway's *decisions*. This module never
decides authorization, policy, risk, or approval itself — it only ever
forwards a `ToolCallRequest` and relays back exactly what the Gateway
decided. The investigation loop (spec §7's
understand -> identify target -> investigate -> collect evidence ->
correlate -> diagnose -> recommend -> assess action -> approve -> execute ->
verify -> report) lives here, bounded to a small number of steps so a
confused model can't loop forever.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Literal

from inumi.agent.context_manager import (
    ContextManager,
    ConversationState,
    InvestigationState,
    PendingApproval,
)
from inumi.agent.llm.base import LLMProvider
from inumi.agent.llm.fallback import fallback_notice_text
from inumi.agent.llm.registry import LLMRegistry
from inumi.agent.planner.actions import (
    AskClarification,
    Conclude,
    IntentExtraction,
    ProposeToolCall,
    RecordObservation,
)
from inumi.agent.playbooks.library import PLAYBOOKS, get_playbook, match_playbook
from inumi.agent.reply import AgentReply, ApprovalCard
from inumi.agent.tool_client import ToolClient
from inumi.common.config import get_settings
from inumi.common.ids import new_id
from inumi.common.models.catalog import LeastPrivilegeFinding
from inumi.common.models.decision_event import DecisionEventCreateRequest
from inumi.common.models.investigation import (
    InvestigationCreateRequest,
    InvestigationUpdateRequest,
)
from inumi.common.models.tool import OperationType, ToolCallRequest, ToolCallResponse, ToolCallStatus
from inumi.common.observability import get_logger
from inumi.common.server_reference import normalize_server_reference

logger = get_logger(__name__)

_MAX_INVESTIGATION_TURNS = 6

# How many record_observation actions in a row (never interrupted by a new
# tool call or a conclude) are tolerated before giving up on asking the LLM
# to conclude and using whatever evidence already exists instead. Verified
# live: a real model can get stuck restating the same finding as one
# observation after another rather than ever emitting action=conclude —
# reproduced on the replication playbook specifically, where the real
# answer ("no replica configured") was already clear after the very first
# observation, yet the model spent its remaining turns re-recording that
# same conclusion as evidence and hit the full turn cap without ever
# reaching the DBA. Small and deliberate: this is meant to catch a stuck
# pattern fast, not to second-guess a model that's still making real
# progress (any other action resets the count to 0).
_MAX_CONSECUTIVE_RECORD_OBSERVATIONS = 2

# Bounds a run of *consecutive* AskClarification turns independently of
# _MAX_INVESTIGATION_TURNS (see the loop's own comment for why they're
# counted separately) — an unresolved back-and-forth (environment, then
# server, then database, ...) still can't run forever across many separate
# requests, since turn_count alone never catches that.
_MAX_CLARIFICATION_TURNS = 4

# A DENIED response whose failure_code means "the proposed call was shaped
# wrong" (not "this is not allowed") — the LLM can plausibly fix it given
# the specific reason, verified live: a real model that omitted a target
# field self-corrected immediately once the Gateway's exact error was fed
# back as an observation. Anything else (UNAUTHORIZED, POLICY_DENIED,
# TOOL_NOT_AVAILABLE, an approval-state problem, RATE_LIMITED, ...) is a
# permissions/policy fact no retry with different arguments changes, so
# those still end the turn immediately rather than burn the turn budget
# (and, with a real provider, API quota) retrying something that can only
# ever fail the same way.
_SELF_CORRECTABLE_DENIAL_CODES = {"INVALID_ARGUMENTS", "INVALID_TARGET", "TOOL_NOT_FOUND"}

# Same heuristic agent.llm.mock already uses to spot a table/database name in
# free text ("a CamelCase word is probably a schema object"). Used here to
# catch the *other* direction: a conclusion's free-text fields naming an
# object that was never actually seen anywhere in this investigation —
# verified live: a real conclusion named three such identifiers
# (AccountBalanceOutstandings, AccountBalances, TransactionPostingHistory_2)
# that don't exist in the database at all, instead of the real table names
# (Branch, production.location, ...) its own tool call had actually
# returned — likely primed by "CoreBanking"-style example names used
# throughout this file's own system prompts/help text.
_CAMEL_CASE_NAME_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")

# A bare answer to "which environment should I investigate" — matched
# word-boundary, case-insensitive, so it still finds "development" inside
# something like "development <@U0BOTID>" (a Slack mention appended after
# the actual answer) without needing an LLM call at all. See its one call
# site in handle_message for why this exists as a fast-path.
_ENVIRONMENT_ANSWER_RE = re.compile(r"\b(development|uat|production)\b", re.IGNORECASE)

# Slack renders an @-mention as a literal `<@U0BOTID>` token in the message
# text delivered to the webhook (see channels/api/app.py's slack_webhook:
# `message=event.get("text", "")`, forwarded completely unmodified — nothing
# anywhere in the channels layer resolves or strips it). Verified live:
# sending "@Inumi DBA Agent /models" — the ordinary way of addressing the
# bot in a shared channel, often the only way to get its attention there —
# arrived here as "<@U0BOTID> /models", which matched neither
# `_handle_command_if_any`'s exact-match check nor any known free-form
# phrasing, and fell through into the normal DBA-task path: reported live as
# the exact, already-advertised `/models` command misrouting into "Which
# environment should I investigate?" instead of listing models. Stripped
# once, centrally, at the very top of handle_message (never by loosening
# each individual exact-match check) so every literal slash command, every
# free-form meta_command phrase, and every downstream problem_summary/
# last_message all see the DBA's actual words, never this delivery
# artifact. Deliberately only strips a LEADING or TRAILING mention token,
# never one in the middle of a message — a mid-message mention can
# meaningfully name a different person (e.g. "check with <@U999> about
# approving this"), and that must never be silently discarded.
_MENTION_TOKEN = r"<@[^>]+>"
_LEADING_MENTIONS_RE = re.compile(rf"^(?:\s*{_MENTION_TOKEN}\s*)+")
_TRAILING_MENTIONS_RE = re.compile(rf"(?:\s*{_MENTION_TOKEN}\s*)+$")


def _strip_bot_mention_noise(message: str) -> str:
    stripped = _LEADING_MENTIONS_RE.sub("", message)
    stripped = _TRAILING_MENTIONS_RE.sub("", stripped)
    return stripped.strip()


def _ungrounded_identifiers(conclusion: Conclude, investigation) -> list[str]:
    """CamelCase-looking identifiers the conclusion's summary/root-cause/
    recommendation text claims, that don't appear anywhere this
    investigation's transcript, running evidence, or the DBA's own problem
    statement — the last one specifically so a term the DBA themselves used
    (e.g. "AlwaysOn") is never flagged just because it isn't a tool call's
    own output. This is a heuristic, not a proof of hallucination — it only
    ever causes one more self-correction turn (see the Conclude branch
    below), never blocks a conclusion from ever landing."""
    # Excludes this same check's own past rejection notes — they necessarily
    # quote the rejected name back (for a human reading the transcript), and
    # without this exclusion that quoting would "ground" the name for every
    # later attempt in the same investigation, defeating the whole check on
    # a second try. Caught by test_repeated_ungrounded_conclusions_fall_
    # through_to_the_safe_fallback before this ever shipped.
    real_transcript = [t for t in investigation.transcript if t.get("tool_id") != "internal.grounding_check"]
    real_evidence = [e for e in investigation.evidence if not e.startswith("(a draft conclusion naming")]
    haystack = " ".join(
        [json.dumps(real_transcript, default=str), investigation.problem, " ".join(real_evidence)]
    ).lower()
    claimed = " ".join(
        filter(None, [conclusion.summary, conclusion.likely_root_cause, conclusion.recommendation])
    )
    seen: set[str] = set()
    ungrounded: list[str] = []
    for match in _CAMEL_CASE_NAME_RE.finditer(claimed):
        name = match.group(0)
        if name in seen:
            continue
        seen.add(name)
        if name.lower() not in haystack:
            ungrounded.append(name)
    return ungrounded


# Write tools with an obvious, cheap, correlated read-only re-check —
# scoped deliberately to session-termination-shaped writes first (the case
# this project has repeatedly hit live and gotten wrong: a real model has
# voluntarily said things like "Subsequent session and blocking checks
# confirmed that session X has been successfully terminated", but nothing
# ever forced that check, so a DBA could just as easily get a clean-
# sounding "Completed" when the session was never actually gone). A write
# like update_statistics/create_index/modify_configuration has no such
# cheap, obvious re-check (confirming it actually helped needs a follow-up
# performance observation, not one more tool call), so those deliberately
# stay out of this mapping for now — this is written as a lookup table,
# not a single hardcoded tool_id check, specifically so extending it to a
# future write tool is just one more entry, not a new mechanism.
_VERIFICATION_TOOLS_BY_WRITE_TOOL: dict[str, tuple[str, ...]] = {
    "database.kill_session": ("database.get_blocking_sessions", "database.get_sessions"),
    "database.cancel_query": (
        "database.get_blocking_sessions",
        "database.get_running_queries",
        "database.get_sessions",
    ),
}


# The synthetic transcript `tool_id` recorded when the read-only gate below
# drops a proposal. Deliberately shaped like the existing
# `internal.grounding_check` / `internal.verification_check` entries: an
# `internal.*` transcript row is this codebase's established way of making a
# decision the orchestrator took *about* the model's output visible in the
# same place a human (and the model's own next turn) reads everything else,
# without inventing a parallel log. `ScheduledSummary.dropped_proposals`
# reads these back out, which is also what makes the guard directly
# assertable in a test instead of only observable as an absence.
_READ_ONLY_GUARD_TOOL_ID = "internal.readonly_guard"

# The playbook the scheduled daily digest runs, once per server. Named here
# rather than inlined so a rename in `playbooks.library` breaks loudly at
# this one reference (see `run_comprehensive_summary`, which refuses to run
# at all if this id no longer resolves) instead of silently degrading a
# scheduled run into a fully freeform investigation — which is exactly the
# shape of failure that would matter most here, since a freeform run is the
# one that would actually want to propose remediation.
_SCHEDULED_SUMMARY_PLAYBOOK_ID = "comprehensive_summary"

# The problem statement an unattended run opens with. This is layer zero of
# the read-only guarantee — the only layer that shapes what the model *wants*
# to do, rather than blocking what it tried to do — and it is here, not in a
# provider's system prompt, because it is specific to this one triggering
# path: the same model, in a live DBA conversation, absolutely should
# propose a remediation when one is warranted (spec §7's assess action ->
# approve -> execute), and nothing here changes that. It also carries the
# server/environment in the DBA's own vocabulary so the final concluding
# call reads the same way an interactive `comprehensive_summary` does.
_SCHEDULED_SUMMARY_PROBLEM = (
    "Scheduled daily health sweep of the {server_id} server ({environment}) — "
    "an unattended, proactive run with no DBA in the conversation to answer a "
    "question or approve anything. Report findings and recommendations as "
    "text only: only read-only diagnostics are available to you here, and "
    "nothing you propose will be executed, approved, or acted on "
    "automatically. If something needs remediation, describe what you would "
    "recommend and why, for a DBA to decide on — never as an action you are "
    "taking. Do not ask a clarifying question; nobody is there to answer it."
)


def _is_confirmed_read_tool(
    tool_id: str, tool_operation_types: dict[str, OperationType] | None
) -> bool:
    """Whether `tool_id` is *confirmed*, by the Gateway's own live tool
    catalog, to be a read-only diagnostic — the predicate the read-only gate
    in `_submit_and_relay` fails closed on.

    Deliberately not a naming-convention check (`tool_id.startswith(
    "database.get_")`) and deliberately not a hardcoded allow-list in this
    file: `operation_type` is the tool catalog's own classification
    (`gateway/domain/tool_catalog.py`), fetched fresh for this very
    investigation in `_continue_investigation`, so a tool reclassified
    WRITE tomorrow is treated as a write here tomorrow, with no second copy
    of that judgment in the Agent to drift out of sync. Same reasoning
    `_update_pending_verification` already gives for consulting
    `tool_operation_types` rather than trusting a table baked into this
    module.

    Unknown means no: a `tool_id` missing from the map (a tool that wasn't
    in the available list at all), or no map supplied, returns False. That
    is the safe direction *for this gate specifically* — the worst case is
    a read-only diagnostic being skipped and the server reported as
    "nothing gathered" in the digest, whereas the opposite default would
    let an unclassifiable proposal through in precisely the unattended
    context where nobody is watching. Note this is the inverse of
    `_update_pending_verification`'s own None handling, and intentionally
    so: there, None means "trust the curated write list"; here, None means
    "refuse", because that one errs toward *more* verification and this one
    errs toward *less* action."""
    operation_type = (tool_operation_types or {}).get(tool_id)
    return operation_type == OperationType.READ


@dataclasses.dataclass(frozen=True)
class ScheduledSummary:
    """One server's outcome from an unattended investigation — what both
    `run_comprehensive_summary` (the daily digest's fixed-playbook sweep)
    and `run_triggered_investigation` (an alert-triggered freeform
    investigation) hand back, via the shared `_unattended_summary`.

    Richer than the `AgentReply` an interactive turn returns, because a
    caller here has to make two decisions no channel adapter ever has to:
    whether this server is worth its own block in the digest at all
    (`is_clean` — see `agent.scheduled_report.build_digest`, which applies
    `comprehensive_summary`'s own "report ONLY deviations" discipline a
    second time, at the multi-server level), and whether the run actually
    produced an answer or merely produced *something* (`ok`). Both are
    derived structurally — from the typed `Conclude` action's own
    root-cause/recommendation fields, via `investigation.findings`/
    `recommendations` — never by pattern-matching the model's prose, which
    neither caller has an honest way to parse."""

    server_id: str
    environment: str
    investigation_id: str
    # AgentReply.status: "ok" | "denied" | "error" | "clarification".
    status: str
    text: str
    findings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    # Every tool_id the read-only gate refused to submit during this run.
    # Empty in the normal case; non-empty means the model tried to propose
    # an action and was structurally prevented from doing so — surfaced in
    # the digest (see `build_digest`) rather than swallowed, because "the
    # agent wanted to do something and wasn't allowed to" is information a
    # DBA should see, not an implementation detail to hide.
    dropped_proposals: tuple[str, ...] = ()
    # Why this server could not be checked, when `ok` is False. Never a raw
    # stack trace — same no-raw-error invariant the rest of this pipeline
    # holds (see ARCHITECTURE.md's "The no-raw-error invariant").
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether this run actually reached a reported conclusion. A
        non-"ok" reply status (the investigation was denied, errored, or
        ended still asking a clarifying question nobody was there to
        answer) is a server the digest must report as NOT checked — a DBA
        reading "6 servers checked, all healthy" when two never responded
        is worse than useless."""
        return self.status == "ok" and not self.error

    @property
    def is_clean(self) -> bool:
        """Nothing was flagged: the concluding action named no root cause
        and recommended nothing. Only meaningful when `ok`."""
        return not self.findings and not self.recommendations


def _verification_still_shows_condition(tool_id: str, session_id: str | None, result: dict | None) -> bool:
    """True if a post-write re-check's OWN result rows still show the
    exact session_id the write targeted — i.e. the write did not actually
    take effect. Deliberately narrow: only ever inspects the specific
    field(s) each of these read tools is known to populate (verified
    against `execution/adapters/*.py` — `session_id` for
    get_sessions/get_running_queries, `blocked_session_id`/
    `blocking_session_id` for get_blocking_sessions), and never guesses
    when session_id is unknown (e.g. a write whose arguments were stripped
    before this ever ran) or the result is missing/oddly shaped — those
    cases fall through to "not shown as still present", which is the safe
    direction: this function's only job is to catch a confirmed-not-
    resolved case, never to manufacture one from ambiguous data."""
    if not session_id:
        return False
    rows = (result or {}).get("rows") or []
    if not rows:
        return False
    keys = (
        ("blocked_session_id", "blocking_session_id")
        if tool_id == "database.get_blocking_sessions"
        else ("session_id",)
    )
    return any(str(row.get(key)) == str(session_id) for row in rows for key in keys)


def _affected_summary(result: dict | None) -> str:
    """A write tool's actual outcome (`execution.ExecutionResult.affected`
    — e.g. `{"terminated": False, "session_id": "13400"}` for kill_session)
    is real information the Gateway already returns, but the response's own
    `message` is a hardcoded "Completed." for every EXECUTED call
    regardless of what `affected` actually says (see
    `gateway.domain.tool_call_handler`) — a DBA reading only "Completed."
    has no way to tell a kill that returned `terminated: false` (nothing
    was actually there to kill — see the double kill_session call this
    surfaced live: a session already gone by the second attempt) from one
    that actually terminated something. Read tools never populate
    `affected` (they populate `rows`/`row_count` instead), so this is a
    no-op for them — only ever adds detail for a write."""
    affected = (result or {}).get("affected")
    if not affected:
        return ""
    return " (" + ", ".join(f"{k}={v}" for k, v in affected.items()) + ")"


_HELP_TEXT = (
    "I'm Inumi, your AI DBA assistant. I can investigate database health, "
    "performance, blocking, deadlocks, replication, backups, and more, and — "
    "with your role's approval where required — take controlled remediation "
    "actions. Try: \"Why is CoreBanking slow?\" or \"Check blocking on "
    "CoreBanking production.\"\n\nCommands: /help, /status, /approve <id>, "
    "/reject <id>, /models, /model <provider> <model>, /servers, /catalog <id>, "
    "/discover, /playbooks, /approvers\n\nYou never need the exact slash "
    "syntax for any of these — plain language works too (e.g. \"what "
    "servers do you have\" instead of /servers)."
)

# A general, honest answer to "who can approve requests from you?" and
# similar RBAC/approval-policy questions — deliberately NOT a dump of
# config/policy.yaml (that file is loaded only by the Gateway process, see
# POLICY_MODEL.md, and its exact per-role/per-environment grants are never
# meant to live in an agent prompt or reply) and deliberately not a per-role
# lookup this layer has no way to compute correctly anyway (the Agent's own
# ToolClient only ever sees a coarse allowed_roles list per tool, filtered
# to the asking DBA's own role — see tool_client.available_tools — never
# the full ALLOW/DENY/REQUIRES_APPROVAL table for every role). What follows
# is the general shape of the model as documented in POLICY_MODEL.md and
# implemented in gateway/domain/approval.py — true for every deployment,
# never a specific grant — plus a pointer to where a DBA gets the exact
# answer for their own request.
_APPROVAL_MODEL_TEXT = (
    "Approval requirements are decided by the DBA Control Gateway's policy "
    "engine, not by me — they depend on your own DBA role, the specific "
    "action, and the target environment, so there's no single fixed answer "
    "I can give in the abstract. In general: routine read-only checks never "
    "need approval; higher-risk write actions typically require a more "
    "senior DBA role or a separate, independent approver (you can never "
    "approve your own request); and the most critical actions (an instance "
    "restart or a failover) require two different qualified approvers, not "
    "just one. Whenever one of your own requests actually needs approval, "
    "I'll show you exactly what's required at that moment — for your "
    "role's specific permissions ahead of time, check with your "
    "organization's RBAC documentation or a DBA_MANAGER."
)


class AgentOrchestrator:
    def __init__(
        self, llm_registry: LLMRegistry, tool_client: ToolClient, context: ContextManager
    ):
        self._llm_registry = llm_registry
        self._tool_client = tool_client
        self._context = context

    def _llm_for(
        self, state: ConversationState, call_type: Literal["fast", "strong"] = "strong"
    ) -> LLMProvider:
        """The provider this conversation's next LLM call should use.

        `resilient_for_conversation` (not `for_conversation`) so that a
        total outage of the chosen provider — every retry and every one of
        its own internal model fallbacks exhausted, the exact shape of the
        live Gemini free-tier quota exhaustion this was built for — escapes
        to the next configured provider family instead of dead-ending in
        "temporarily unavailable" with other vendors' keys sitting unused.
        See agent/llm/fallback.py for the mechanism and its time-budget
        arithmetic; the registry returns the bare provider unchanged for
        the offline mock planner and for single-provider deployments, so
        neither pays anything for this.

        `call_type` proactively picks a model tier (see
        `LLMRegistry.tier_model`) — cheap for `extract_intent`-style calls,
        strong for `decide_next_action`/critique — but ONLY when the DBA
        hasn't made an explicit `/model` choice (`state.llm_provider is
        None and state.llm_model is None`, the exact existing lock
        condition): an explicit choice is never silently overridden by a
        tier default. A deployment-level provider lock (`Settings.
        llm_provider`) doesn't disable this either — that only constrains
        which vendor is used; `state.llm_provider`/`llm_model` stay `None`
        in that case, so tiering still applies within the locked vendor.
        Both tier settings are empty by default, so a zero-config
        deployment resolves the exact same model as before this existed.

        `state.llm_fallback_notices` is the sink: anything recorded there
        while answering the current message is disclosed to the DBA by
        `handle_message` below. A fresh wrapper per call is deliberate —
        it keeps that sink per-message rather than shared across
        concurrent conversations (the real provider objects underneath are
        still the registry's cached ones)."""
        model = state.llm_model
        if state.llm_provider is None and state.llm_model is None:
            model = self._llm_registry.tier_model(call_type)
        return self._llm_registry.resilient_for_conversation(
            provider=state.llm_provider,
            model=model,
            notices=state.llm_fallback_notices,
        )

    async def _list_servers_cached(self) -> list[dict]:
        try:
            return await self._tool_client.list_servers()
        except Exception:  # noqa: BLE001
            return []

    async def list_registered_servers(self) -> list[dict]:
        """Every registered server, straight from the Gateway's
        `/v1/catalog/servers` — the same payload every other server-aware
        path in this file reads, so the scheduled digest
        (`agent.scheduled_report.DailyDigestRunner`) never grows a second,
        separately-maintained idea of what the estate contains.

        Deliberately NOT `_list_servers_cached`: that one swallows every
        failure into `[]` because its callers are conveniences (offering the
        planner known database names, auto-filling an environment) where a
        missing list only costs a clarification round-trip. For the digest
        that same `[]` would be indistinguishable from "no servers are
        registered", and the difference is exactly what separates "nothing
        to check this morning" from "we checked nothing and don't know
        why" — so this one lets the failure propagate and leaves it to
        `run_once` to report the gap plainly."""
        return await self._tool_client.list_servers()

    async def _known_database_names(self) -> list[str]:
        """Every discovered database name — helps the planner resolve
        "check CoreBanking" to a real target. Server ids/aliases are NOT
        included (they're resolved separately, via `instance_hint`)."""
        names: list[str] = []
        for s in await self._list_servers_cached():
            names.extend((s.get("catalog") or {}).get("databases", []))
        return [n for n in dict.fromkeys(names) if n]

    async def _known_server_hints(self) -> list[str]:
        """Every registered server id + alias — lets the planner recognize
        "on postgres-local" and narrow an otherwise-ambiguous target. The
        Gateway still independently re-resolves and validates whatever
        comes back; this only saves the DBA a disambiguation round-trip."""
        hints: list[str] = []
        for s in await self._list_servers_cached():
            hints.append(s["id"])
            hints.extend(s.get("aliases") or [])
        return [h for h in dict.fromkeys(hints) if h]

    async def handle_message(
        self,
        *,
        channel: str,
        channel_account_id: str,
        conversation_id: str,
        channel_thread_id: str,
        message: str,
    ) -> AgentReply:
        """Public entry point. Thin wrapper around `_handle_message` whose
        only job is the cross-provider LLM fallback disclosure.

        A fallback is allowed to happen even when a DBA explicitly locked a
        provider (`LLM_PROVIDER=...` or `/model <provider>`) — an answer
        beats "temporarily unavailable" while other configured keys sit
        unused — but it is never allowed to happen *silently*: whichever
        vendor actually produced this reply is stated plainly at the end of
        it. Done here, at the single outermost return point, rather than at
        each of the dozen-odd places a reply is constructed below, so no
        future reply path can forget to disclose it.

        `handle_approval_decision` (the other public entry point) makes no
        LLM call at all, so it needs none of this.
        """
        state = self._context.get_or_create(
            conversation_id, channel, channel_thread_id, channel_account_id
        )
        # Per-message scratch space — anything left over from the previous
        # message on this conversation was already disclosed with it.
        state.llm_fallback_notices.clear()
        reply = await self._handle_message(
            channel=channel,
            channel_account_id=channel_account_id,
            conversation_id=conversation_id,
            channel_thread_id=channel_thread_id,
            message=message,
        )
        notice = fallback_notice_text(state.llm_fallback_notices)
        for event in state.llm_fallback_notices:
            # Pure telemetry, fired without blocking the reply — a fallback
            # substitution is already disclosed to the DBA via `notice`
            # below; this is the durable, reviewable copy of the same fact
            # (see gateway.domain.decision_events).
            await self._tool_client.log_decision_event(
                DecisionEventCreateRequest(
                    event_type="llm_cross_provider_fallback_used",
                    conversation_id=conversation_id,
                    provider=event.used_provider,
                    payload={"failed_providers": event.failed_providers},
                )
            )
        state.llm_fallback_notices.clear()
        if notice:
            reply.text = f"{reply.text}\n\n{notice}" if reply.text else notice
        return reply

    async def _handle_message(
        self,
        *,
        channel: str,
        channel_account_id: str,
        conversation_id: str,
        channel_thread_id: str,
        message: str,
    ) -> AgentReply:
        message = _strip_bot_mention_noise(message)
        state = self._context.get_or_create(
            conversation_id, channel, channel_thread_id, channel_account_id
        )
        self._context.touch(state)

        command_reply = await self._handle_command_if_any(state, message, channel, channel_account_id)
        if command_reply is not None:
            return command_reply

        # Resuming an in-progress investigation — the DBA's reply is part of
        # THIS conversation, not a fresh, standalone utterance to classify
        # from scratch. Verified live (twice, two different clarification
        # kinds): a short reply like "development" (answering the hardcoded
        # environment gate) or a bare server name (answering a freeform
        # AskClarification the LLM itself asked) both got reclassified by
        # extract_intent as chitchat/non-DBA and silently dropped the whole
        # investigation. decide_next_action already has the full transcript
        # and knows exactly what it just asked, so hand it the raw reply
        # directly — via investigation.last_message, see
        # _problem_statement_for_llm — instead of ever risking that
        # misclassification again. The one thing still enforced here, not
        # left to the LLM: never guessing an environment (spec: "for
        # production targets I won't guess") — a bare answer to that
        # specific question is recognized deterministically; anything else
        # while it's still missing re-asks rather than guessing.
        if state.investigation is not None and not state.investigation.is_concluded:
            investigation = state.investigation
            if "environment" not in state.database_context:
                match = _ENVIRONMENT_ANSWER_RE.search(message)
                if match:
                    state.database_context["environment"] = match.group(1).lower()
                else:
                    return AgentReply(
                        text=(
                            "Which environment should I investigate — development, uat, "
                            "or production? For production targets I won't guess."
                        ),
                        status="clarification",
                    )

            # A narrow exception to "always resume" above, added after a
            # separate live finding: an investigation stuck in
            # AWAITING_CLARIFICATION (an unanswered freeform question — the
            # DBA never has to answer it; there's no timeout) silently
            # swallowed every message that arrived afterward, forever,
            # framing each one to decide_next_action as "the DBA just
            # replied" to that old, unrelated question. Reproduced live:
            # "check blah on the thing pls fix asap!!!" (a deliberate
            # gibberish test) asked what "blah"/"the thing" meant; 32
            # minutes and several unrelated exchanges later, "Drop the test
            # database on postgres-local, it's no longer needed" got a
            # reply that rambled about "blah" and "the thing" instead of
            # addressing the actual request — the old, stale problem text
            # was still being prepended verbatim (see
            # `_problem_statement_for_llm`). Scoped as tightly as possible
            # to avoid resurrecting either of the two bugs already fixed
            # above (a bare "development" or a bare server name silently
            # dropping the investigation): only checked in this one status,
            # never when an approval is outstanding, and
            # `_classify_potential_topic_shift` itself demands multiple
            # positive signals, not just `is_dba_task`, before treating
            # anything as a topic shift.
            if investigation.status == "AWAITING_CLARIFICATION" and state.pending_approval is None:
                fresh_intent = await self._classify_potential_topic_shift(state, message)
                if fresh_intent is not None:
                    investigation.status = "CONCLUDED_UNRESOLVED"
                    return await self._start_fresh_investigation(
                        state, fresh_intent, message, channel, channel_account_id
                    )

            investigation.last_message = message
            return await self._continue_investigation(state, investigation, channel, channel_account_id)

        llm = self._llm_for(state, call_type="fast")
        intent = await llm.extract_intent(
            message,
            known_database_names=await self._known_database_names(),
            known_server_hints=await self._known_server_hints(),
        )
        return await self._start_fresh_investigation(state, intent, message, channel, channel_account_id)

    async def _classify_potential_topic_shift(
        self, state: ConversationState, message: str
    ) -> IntentExtraction | None:
        """Called only from the one narrow spot in `handle_message` above —
        an investigation is stuck in AWAITING_CLARIFICATION and a new
        message just arrived. Returns the classified intent when this reads
        as a genuinely fresh, self-contained instruction worth abandoning
        the stale investigation for; `None` when it's more likely a reply
        to the pending question (or anything else too ambiguous to act on),
        in which case the caller falls through to the existing resume path
        unchanged.

        Deliberately NOT the same trust level the pre-fix code gave
        `extract_intent` (see `handle_message`'s own docstring on the
        resume branch): `is_dba_task`/`is_greeting_or_chitchat` alone were
        exactly what silently dropped a bare, legitimate clarification
        answer before ("development", a bare server name). Both of those
        name no target of their own and are one or two words — so this
        additionally requires the message to name its own concrete target
        (instance, database, or environment) AND run longer than a bare
        answer plausibly would. A short reply to "which server did you
        mean?" can trivially set `instance_hint` too, so the word-count
        floor is doing real work here, not padding: a real fresh
        instruction ("restart the postgres-local instance", "drop the test
        database on postgres-local") always reads as a full sentence, never
        as a single named entity on its own.

        Checks the free word-count floor FIRST, before ever calling
        `extract_intent` — every legitimate short clarification answer
        ("development", a bare server name, "yes") fails it immediately, at
        zero LLM-call cost, so the added latency/API spend this whole check
        introduces only ever lands on messages already long enough to
        plausibly be a fresh instruction, not on the common case."""
        if len(message.split()) < 4:
            return None
        llm = self._llm_for(state, call_type="fast")
        intent = await llm.extract_intent(
            message,
            known_database_names=await self._known_database_names(),
            known_server_hints=await self._known_server_hints(),
        )
        if not intent.is_dba_task or intent.is_greeting_or_chitchat:
            return None
        names_own_target = bool(intent.instance_hint or intent.database_hint or intent.environment_hint)
        if not names_own_target:
            return None
        return intent

    async def _start_fresh_investigation(
        self,
        state: ConversationState,
        intent: IntentExtraction,
        message: str,
        channel: str,
        channel_account_id: str,
    ) -> AgentReply:
        """Everything `handle_message` does with a classified `intent` for a
        brand-new investigation — extracted so
        `_classify_potential_topic_shift`'s pivot case (an old,
        AWAITING_CLARIFICATION investigation abandoned for a fresh one) and
        the normal "no active investigation" case share the exact same
        meta-command/chitchat/target-resolution handling, instead of a
        second, independent copy that could quietly drift out of sync."""
        if intent.meta_command and not (
            intent.meta_command in ("approve", "reject") and state.pending_approval is None
        ):
            # A free-text equivalent of one of the exact slash commands
            # below (spec: the DBA should never be bound to a fixed
            # message structure) — "list my servers" works exactly like
            # "/servers", "what playbooks do you have" like "/playbooks",
            # etc. Checked before the greeting/chitchat gate since these
            # are a distinct, actionable third category, never either of
            # those. The approve/reject exclusion above is deliberate —
            # verified live: "go ahead and terminate session 19860"
            # matched the same phrasing as reacting to a shown approval
            # card ("go ahead" -> approve), but there was no pending
            # approval to react to, and it dead-ended with "there is no
            # pending approval" instead of acting. extract_intent has no
            # visibility into whether one actually exists (it classifies
            # from the raw message alone), so that specific combination
            # falls through to the normal DBA-task path below instead —
            # a message naming a specific session/server/action is far
            # more likely a fresh instruction than a reaction to nothing.
            return await self._handle_meta_command(intent, state, channel, channel_account_id)
        if (
            intent.meta_command in ("approve", "reject")
            and state.pending_approval is None
            and not intent.is_dba_task
        ):
            # See above — falling through, so this must still be treated
            # as the actionable DBA task it obviously is, not dismissed by
            # the classifier's own (now-irrelevant) is_dba_task verdict.
            intent.is_dba_task = True
        if intent.is_greeting_or_chitchat:
            return AgentReply(text=_HELP_TEXT)
        if not intent.is_dba_task:
            return AgentReply(
                text=(
                    "I can help with database health, performance, and operational "
                    "investigations. Could you tell me more about what you'd like me "
                    "to check?"
                ),
                status="clarification",
            )

        # Reachable only for a brand-new investigation (None) or a
        # previously-concluded one starting fresh — an active one already
        # returned above, before ever reaching extract_intent. Falls back
        # to the raw message if problem_summary is blank — verified live:
        # reachable via the approve/reject-with-no-pending-approval
        # fallback above, where a real model classifying the message as
        # meta_command left problem_summary empty since nothing told it to
        # fill that in for that path too.
        investigation = self._context.start_investigation(state, intent.problem_summary or message)
        # Deterministic, zero-LLM-call keyword match against a small
        # library of known scenarios (slow queries, high CPU, blocking,
        # ...) — see agent.playbooks.library for the rationale. None
        # means no known scenario matched; the loop below falls back to
        # the original fully-freeform behavior, unchanged.
        playbook = match_playbook(intent.problem_summary or message)
        if playbook is not None:
            investigation.playbook_id = playbook.playbook_id
        if intent.environment_hint:
            state.database_context["environment"] = intent.environment_hint
        if intent.database_hint:
            state.database_context["database"] = intent.database_hint
        if intent.instance_hint:
            switched_instance = state.database_context.get("instance") != intent.instance_hint
            if not intent.database_hint and switched_instance:
                # A remembered database (see _submit_and_relay) belongs to
                # whatever instance was previously in play — moving to a
                # different one makes it stale, and it's never safe to
                # assume the new server even has a database by that name.
                state.database_context.pop("database", None)
            state.database_context["instance"] = intent.instance_hint
            if not intent.environment_hint:
                # A specific, registered server was named but no
                # environment was — verified live: naming "postgres-local"
                # explicitly still triggered "which environment should I
                # investigate?" even though a registered server has
                # exactly one environment in config/servers.yaml. Asking
                # again for something already implied by the instance is
                # never necessary; only ever fills a gap, never overrides
                # an environment the DBA actually stated.
                auto_environment = await self._environment_for_instance(intent.instance_hint)
                if auto_environment:
                    state.database_context["environment"] = auto_environment
                elif switched_instance:
                    # Mirrors the database-goes-stale-on-switch logic just
                    # above, for the same reason: a remembered environment
                    # belongs to whatever instance was previously in play
                    # too. Moving to a different, unregistered/ambiguous
                    # server whose environment can't be auto-resolved must
                    # not silently keep asserting the OLD instance's
                    # environment for this new, unnamed-environment one —
                    # that's exactly the guess the spec forbids ("for
                    # production targets I won't guess"). Drop it so the
                    # check below asks instead of carrying over a value
                    # that may now simply be wrong.
                    state.database_context.pop("environment", None)

        # A fresh investigation only ever needs to ask for the environment
        # when it is genuinely unknown anywhere in this conversation — never
        # just because *this* message alone didn't repeat it. Verified live:
        # once the DBA had already established development/postgres-local a
        # few messages earlier, a later plain follow-up ("so what database
        # is the copy activity happening on?") with no environment/instance
        # wording of its own still asked "which environment should I
        # investigate?" again. state.database_context is the real source of
        # truth for "environment" — carried across investigations in this
        # same conversation exactly like instance/database already are
        # above — so this checks it directly rather than re-deriving the
        # answer from intent.environment_hint alone, which only ever
        # reflects this one message.
        if "environment" not in state.database_context:
            return AgentReply(
                text=(
                    "Which environment should I investigate — development, uat, or "
                    "production? For production targets I won't guess."
                ),
                status="clarification",
            )

        return await self._continue_investigation(state, investigation, channel, channel_account_id)

    async def _find_matching_servers(self, instance_hint: str) -> list[dict]:
        """Mirrors `ServerRegistry.find_candidates`'s own matching (exact,
        substring, host, and normalized-for-spacing/punctuation/padding)
        — the one shared implementation `_environment_for_instance` and
        `_canonical_server_id` both resolve a raw instance_hint through,
        so they can never drift into two different matching rules."""
        hint = instance_hint.strip().lower()
        hint_normalized = normalize_server_reference(hint)
        matches: list[dict] = []
        for s in await self._list_servers_cached():
            names = {s["id"].lower(), *(a.lower() for a in (s.get("aliases") or []))}
            host = (s.get("host") or "").lower()
            normalized_names = {normalize_server_reference(n) for n in names}
            if (
                hint in names
                or hint == host
                or hint_normalized in normalized_names
                or any(hint in n for n in names)
                or (host and hint in host)
                or any(hint_normalized in n for n in normalized_names)
            ):
                matches.append(s)
        return matches

    async def _environment_for_instance(self, instance_hint: str) -> str | None:
        """Auto-fills the environment for a server the Gateway would
        actually have resolved unambiguously anyway — but only ever when
        it's unambiguous (more than one match here just means no
        auto-fill, never a guess; the Gateway still separately,
        independently re-resolves and validates whatever ends up in the
        actual tool call target regardless)."""
        environments = [
            s["environment"] for s in await self._find_matching_servers(instance_hint) if s.get("environment")
        ]
        unique = set(environments)
        return environments[0] if len(unique) == 1 else None

    async def _canonical_server_id(self, instance_hint: str) -> str:
        """Resolves a raw, model-extracted `instance_hint` to the actual
        registered server id, for anything that persists or looks up by
        server identity (investigation memory keying, cross-server
        correlation) rather than just resolving one live tool call's
        target. Verified live: a real model extracted "Postgres dev 02"
        for the registered server `postgres-dev-02` — a real tool call
        still resolves that correctly via the Gateway's own independent
        fuzzy matching, but storing the raw hint as `server_id` would
        have silently fragmented memory recall/correlation for the same
        physical server across conversations that happened to phrase its
        name differently. Falls back to the raw hint when it matches no
        registered server (an ad-hoc/unknown name) or more than one
        (ambiguous) — never guesses which one it meant."""
        ids = {s["id"] for s in await self._find_matching_servers(instance_hint)}
        return next(iter(ids)) if len(ids) == 1 else instance_hint

    async def _continue_investigation(
        self, state: ConversationState, investigation, channel: str, channel_account_id: str
    ) -> AgentReply:
        llm = self._llm_for(state)
        await self._bootstrap_investigation_memory(state, investigation, channel_account_id)
        available = await self._tool_client.available_tools(channel, channel_account_id)
        if investigation.read_only:
            # First of the three layers enforcing "a scheduled run never
            # writes" (see `InvestigationState.read_only` and
            # ARCHITECTURE.md's "Scheduled daily digest" section): the
            # model is simply never *offered* anything but reads, so the
            # ordinary case is that it has nothing to propose. Cheapest and
            # least surprising layer — `StructuredLLMProvider
            # .decide_next_action`'s own post-validation already refuses a
            # `ProposeToolCall` naming a tool outside `available_tool_ids`
            # and turns it into an AskClarification (llm/base.py, the
            # `action.tool_id not in available_tool_ids` check), so the
            # write never even becomes a proposal this loop sees.
            #
            # This layer alone is NOT the guarantee, which is why the hard
            # gate in `_submit_and_relay` exists underneath it: a filtered
            # menu and a prompt instruction both depend on a provider
            # implementation behaving, and `LLMProvider` is an interface
            # anyone can implement (the offline mock planner does, and so
            # would a future provider whose post-validation this file knows
            # nothing about). The gate below depends on nothing but this
            # process's own control flow.
            available = [t for t in available if t.operation_type == OperationType.READ]
        available_ids = [t.tool_id for t in available]
        # Each tool's *actual* required arguments (its real Pydantic schema,
        # already alias-correct — e.g. "schema"/"table", not "schema_name"/
        # "table_name") — verified live: without this, the LLM has nothing
        # but the tool_id string to go on and reliably guesses wrong for any
        # schema/table-scoped write tool.
        #
        # Deliberately includes a tool whose required list is EMPTY (e.g.
        # every NoArgs read tool), rather than omitting it — this used to
        # filter those out (`if (reqs := ...)`, false for an empty list),
        # which was itself a live, repeated finding: get_blocking_sessions
        # and get_sessions (both NoArgs) kept getting called with an extra
        # `reason` or `session_id`/`database_name` folded into `arguments`,
        # because the model was never actually told those specific tools
        # need nothing there — it only ever saw entries for tools that DO
        # need something, and reasonably (if wrongly) generalized from the
        # `arguments` schema's superset of possible keys (`session_id`,
        # `reason`, ... — real requirements for OTHER tools). An explicit
        # `[]` entry is a positive "this tool needs nothing", not silence
        # the model has to interpret on its own.
        tool_requirements = {t.tool_id: t.argument_schema.get("required", []) for t in available}
        # Defense-in-depth backstop for the same finding: every available
        # tool's complete set of allowed `arguments` keys (not just the
        # required ones — e.g. get_top_queries' optional order_by/limit
        # still need to survive this), used by `_submit_and_relay` to
        # silently drop any key the model adds anyway despite the prompt
        # guidance above, before a request ever reaches the Gateway. Belt-
        # and-suspenders: the prompt fix alone was verified live to still
        # occasionally slip (a smaller/weaker model, or a fresh provider
        # this prompt hasn't been tuned against), and stripping here is
        # always safe — a key the tool's own schema wouldn't accept can
        # never have been meant for it.
        tool_allowed_arguments = {
            t.tool_id: set(t.argument_schema.get("properties", {})) for t in available
        }
        # Each tool's operation_type, straight from the same `/v1/tools`
        # response already fetched above — nothing new to thread from the
        # Gateway, this info was already here. Used by `_submit_and_relay`
        # as a defense-in-depth confirmation (alongside
        # `_VERIFICATION_TOOLS_BY_WRITE_TOOL`'s own tool_id membership) that
        # a tool this is about to treat as "a write needing a post-hoc
        # check" is actually still classified as OperationType.WRITE by the
        # tool catalog right now, not stale knowledge baked into this file.
        tool_operation_types = {t.tool_id: t.operation_type for t in available}

        reply = await self._run_investigation_loop(
            state,
            investigation,
            available_ids,
            channel,
            channel_account_id,
            llm,
            tool_requirements,
            tool_allowed_arguments,
            tool_operation_types,
        )
        raw_instance = state.database_context.get("instance")
        await self._tool_client.update_investigation(
            investigation.investigation_id,
            InvestigationUpdateRequest(
                server_id=await self._canonical_server_id(raw_instance) if raw_instance else None,
                playbook_id=investigation.playbook_id,
                environment=state.database_context.get("environment"),
                target=dict(state.database_context),
                status=investigation.effective_status,
                evidence=investigation.evidence,
                hypotheses=investigation.hypotheses,
                findings=investigation.findings,
                recommendations=investigation.recommendations,
                actions=investigation.actions,
            ),
        )
        return reply

    async def _bootstrap_investigation_memory(
        self, state: ConversationState, investigation, channel_account_id: str
    ) -> None:
        """One-time, best-effort hook covering every path into
        `_continue_investigation` (interactive, resumed, the scheduled
        digest, and an alert-triggered run): tells the Gateway this
        investigation exists, and folds two kinds of background into
        `investigation.memory_context` (see that field's own docstring for
        why it's kept separate from `evidence`) — recent, concluded
        findings for the SAME server (Phase 1), and, when a playbook
        matched, similar findings on OTHER servers (Phase 5 cross-server
        correlation, tagged with their own `server_id` so the model/DBA can
        tell the two apart). Guarded by `remote_bootstrap_done` since this
        must run exactly once per investigation, not on every resumed
        turn."""
        if investigation.remote_bootstrap_done:
            return
        investigation.remote_bootstrap_done = True
        raw_instance = state.database_context.get("instance")
        server_id = await self._canonical_server_id(raw_instance) if raw_instance else None
        environment = state.database_context.get("environment")
        await self._tool_client.create_investigation(
            InvestigationCreateRequest(
                investigation_id=investigation.investigation_id,
                conversation_id=state.conversation_id,
                user_subject_id=channel_account_id,
                server_id=server_id,
                playbook_id=investigation.playbook_id,
                environment=environment,
                target=dict(state.database_context),
                problem=investigation.problem,
                status=investigation.status,
            )
        )
        memory_context: list[dict] = []
        lookback = get_settings().investigation_memory_lookback
        if server_id and lookback > 0:
            entries = await self._tool_client.get_investigation_memory(
                server_id,
                exclude_investigation_id=investigation.investigation_id,
                limit=lookback,
            )
            memory_context.extend(e.model_dump(mode="json") for e in entries)
        if investigation.playbook_id:
            patterns = await self._tool_client.get_cross_server_patterns(
                playbook_id=investigation.playbook_id,
                environment=environment,
                exclude_server_id=server_id,
            )
            memory_context.extend(e.model_dump(mode="json") for e in patterns)
        investigation.memory_context = memory_context

    async def _run_investigation_loop(
        self,
        state: ConversationState,
        investigation,
        available_ids: list[str],
        channel: str,
        channel_account_id: str,
        llm: LLMProvider,
        tool_requirements: dict[str, list[str]] | None = None,
        tool_allowed_arguments: dict[str, set[str]] | None = None,
        tool_operation_types: dict[str, OperationType] | None = None,
    ) -> AgentReply:
        while investigation.turn_count < _MAX_INVESTIGATION_TURNS:
            # Every fresh pass through the loop means the investigation is
            # actively investigating again, not still blocked on whatever
            # it last asked — reached either because a playbook step or a
            # freeform tool call is about to run, or because this is a
            # brand-new call into the loop after the DBA answered a
            # clarification from an earlier, separate request (see
            # `handle_message`'s resume path). A CONCLUDED_* stage is never
            # clobbered by this: every place below that sets one also
            # returns immediately afterward, so this line never runs again
            # in the same investigation once that's happened. Resetting
            # here — rather than only where AskClarification is handled —
            # also covers a playbook's own deterministic steps, which never
            # pass through that branch at all.
            investigation.status = "INVESTIGATING"
            step_action = self._next_playbook_action(investigation, available_ids)
            if step_action is not None:
                # Deterministic step from a matched playbook — propose it
                # directly, skipping the LLM call entirely for this turn.
                # This is the actual point of a playbook: for a *known*
                # scenario, the sequence of diagnostics to run is already
                # decided, so there's nothing for the LLM to figure out here
                # — asking it anyway would only add latency and a chance of
                # a malformed completion for a call whose shape was never in
                # question. The LLM still gets one full turn at the end (once
                # playbook_step exhausts the step list, below falls through
                # to the normal decide_next_action call) to interpret
                # everything gathered and conclude.
                investigation.turn_count += 1
                reply = await self._submit_and_relay(
                    state,
                    investigation,
                    step_action,
                    channel,
                    channel_account_id,
                    tool_allowed_arguments,
                    tool_operation_types,
                )
                if reply is not None:
                    return reply
                continue  # executed (or failed-but-logged) — advance to the next step

            if investigation.consecutive_record_observations >= _MAX_CONSECUTIVE_RECORD_OBSERVATIONS:
                # Stop asking rather than wait out the rest of the turn
                # budget on a model that's already shown it isn't going to
                # conclude on its own — same safe fallback the turn cap
                # itself produces below, just reached sooner and without
                # spending further LLM calls on a pattern already proven
                # stuck.
                investigation.status = self._conclusion_stage(investigation)
                return self._no_root_cause_reply(investigation)

            action = await llm.decide_next_action(
                problem_statement=self._problem_statement_for_llm(investigation),
                available_tool_ids=available_ids,
                transcript=investigation.transcript,
                turn_count=investigation.turn_count,
                tool_requirements=tool_requirements,
            )
            investigation.last_message = ""  # consumed — see the field's own docstring

            if isinstance(action, AskClarification):
                # Does NOT consume _MAX_INVESTIGATION_TURNS — a clarification
                # is the DBA narrowing down a target, not the model looping
                # on diagnostics, which is what that budget exists to bound.
                # Verified live: a multi-step targeting dialogue (environment
                # -> server -> database) burned most of the shared budget
                # before real diagnostics even started, then hit the cap
                # right as they began succeeding. Bounded independently
                # instead, so an unresolved back-and-forth still can't run
                # forever across many separate requests (turn_count alone
                # wouldn't catch that, since it's never incremented here).
                investigation.consecutive_record_observations = 0
                investigation.clarification_count += 1
                if investigation.clarification_count > _MAX_CLARIFICATION_TURNS:
                    investigation.status = self._conclusion_stage(investigation)
                    return AgentReply(
                        text=(
                            "I still don't have enough information to proceed — "
                            "please restate what you'd like me to check, including "
                            "the environment, server, and database if relevant, in "
                            "one message."
                        ),
                        investigation_id=investigation.investigation_id,
                    )
                # The loop genuinely cannot proceed until the DBA answers
                # this — reflected in `status` (not just `clarification_
                # count`) so `/status` can report it plainly instead of
                # just "investigating". Reset back to INVESTIGATING at the
                # top of this same loop, the moment a real reply lets the
                # loop run again (see that reset's own comment).
                investigation.status = "AWAITING_CLARIFICATION"
                return AgentReply(
                    text=action.question,
                    status="clarification",
                    investigation_id=investigation.investigation_id,
                )

            investigation.turn_count += 1
            investigation.clarification_count = 0

            if isinstance(action, RecordObservation):
                investigation.consecutive_record_observations += 1
                investigation.evidence.append(action.text)
                continue

            if isinstance(action, ProposeToolCall):
                investigation.consecutive_record_observations = 0
                reply = await self._submit_and_relay(
                    state,
                    investigation,
                    action,
                    channel,
                    channel_account_id,
                    tool_allowed_arguments,
                    tool_operation_types,
                )
                if reply is not None:
                    return reply
                continue  # executed successfully — loop for the next step

            if isinstance(action, Conclude):
                investigation.consecutive_record_observations = 0
                reply = await self._finalize_conclude(investigation, action, llm=llm, state=state)
                if reply is not None:
                    return reply
                continue  # rejected as ungrounded — logged inside, try again

        # Turn budget exhausted without ever reaching action=conclude.
        # Verified live: a real investigation used its final 2 turns on
        # legitimate remediation attempts (kill_session, then cancel_query
        # as a fallback) that both turned up nothing to act on — genuinely
        # useful information — but hit the cap with zero turns left to
        # report it, and the DBA got the generic fallback below instead of
        # that. One last, bounded call gives the model a real chance to
        # explain the outcome first. No further tool calls are offered
        # (available_tool_ids=[]) — decide_next_action's own post-
        # validation turns any attempted one into an AskClarification,
        # which (like any non-Conclude response here) just falls through
        # to the same safe fallback as before; this can only ever add one
        # bounded call, never another loop.
        final_action = await llm.decide_next_action(
            problem_statement=self._problem_statement_for_llm(investigation)
            + "\n\nYou are out of further diagnostic or action turns. You MUST respond "
            "with action=conclude now, summarizing what was found and done so far — "
            "even if the root cause isn't fully confirmed, an honest \"here's what I "
            "found and tried\" is far more useful than nothing.",
            available_tool_ids=[],
            transcript=investigation.transcript,
            turn_count=investigation.turn_count,
            tool_requirements=None,
        )
        if isinstance(final_action, Conclude):
            # final_chance=True: no tool calls were even offered for this
            # last attempt (available_tool_ids=[] above), so a still-
            # pending post-write verification can no longer be fixed by
            # asking again — accept the conclusion but let `_format_report`
            # state plainly that it was never independently re-checked,
            # rather than rejecting into the generic no-root-cause
            # fallback and losing everything the investigation actually did.
            reply = await self._finalize_conclude(
                investigation, final_action, llm=llm, state=state, final_chance=True
            )
            if reply is not None:
                return reply

        investigation.status = self._conclusion_stage(investigation)
        return self._no_root_cause_reply(investigation)

    @staticmethod
    def _append_evidence_once(investigation, text: str) -> None:
        """Append to `investigation.evidence` unless it's a byte-identical
        repeat of the entry already at the end. Found live: a deterministic
        planner (the mock provider — it has no way to read the freeform
        guidance appended to `investigation.transcript` on a rejected
        Conclude, so it proposes the exact same conclusion again every turn)
        can retry the identical rejected Conclude turn after turn until the
        budget runs out, and both rejection notes below (ungrounded names,
        pending verification) are worded identically on every retry — the
        DBA-facing evidence list ended up with the same sentence repeated
        3+ times in a row, adding noise, not information. The transcript
        entry (what a real model actually sees to course-correct) is still
        appended every time, unchanged — only the human-facing summary
        de-duplicates consecutive repeats."""
        if not investigation.evidence or investigation.evidence[-1] != text:
            investigation.evidence.append(text)

    async def _self_critique_conclude(self, investigation, action: Conclude, llm: LLMProvider, state):
        """A second LLM opinion on a draft Conclude — see `CritiqueVerdict`'s
        docstring for what this catches that `_ungrounded_identifiers` and
        `pending_verification` don't. Returns `None` (meaning "accept," the
        same as a `sound=True` verdict) on ANY exception — a critique call
        failing (a timeout, a provider outage, a malformed response after
        retries) must never be worse than not having critiqued at all; only
        an actual working `sound=False` verdict rejects. Gated by
        `Settings.self_critique_enabled` as a cheaper-than-a-redeploy kill
        switch if this misbehaves in production.

        Deliberately reuses the ambient `llm` from the caller rather than
        re-resolving its own — `_continue_investigation` always resolves
        `llm` at the default `call_type="strong"` (it's only ever
        overridden to `"fast"` at the two `extract_intent` call sites, well
        before an investigation loop or a Conclude exists), so the
        provider critiquing a conclusion is already the same strong tier
        that proposed it; a second resolution here would be redundant, not
        safer, and would break the many existing tests that inject one
        `llm` test double directly into `_run_investigation_loop` without
        going through `LLMRegistry` at all."""
        if not get_settings().self_critique_enabled:
            return None
        try:
            verdict = await llm.critique_conclusion(
                problem_statement=investigation.problem,
                transcript=investigation.transcript,
                proposed_summary=action.summary,
                proposed_root_cause=action.likely_root_cause,
                proposed_confidence=action.confidence,
                proposed_recommendation=action.recommendation,
            )
        except Exception as exc:  # noqa: BLE001 — fail open, see docstring above
            logger.warning(
                "self_critique_call_failed",
                investigation_id=investigation.investigation_id,
                error=str(exc),
            )
            await self._tool_client.log_decision_event(
                DecisionEventCreateRequest(
                    event_type="self_critique_call_failed",
                    conversation_id=state.conversation_id,
                    investigation_id=investigation.investigation_id,
                    provider=llm.provider_name,
                    model=llm.model,
                    payload={"error": str(exc)},
                )
            )
            return None
        return verdict

    async def _finalize_conclude(
        self,
        investigation,
        action: Conclude,
        *,
        llm: LLMProvider,
        state,
        final_chance: bool = False,
    ) -> AgentReply | None:
        """Builds the final reply for a Conclude action, or returns None
        if it's rejected — the caller decides what happens next (loop back
        for a retry mid-investigation, or fall through to the safe generic
        fallback if this was the one bounded last-chance call after the
        turn budget ran out). Three independent things can reject a
        conclusion: it names something ungrounded (see
        `_ungrounded_identifiers`), it's trying to conclude with a write's
        real-world effect still unverified (see
        `investigation.pending_verification`'s own docstring for why this
        exists), or a second LLM opinion judges it doesn't actually follow
        from the evidence (see `_self_critique_conclude`) — the latter two
        only apply mid-investigation (`final_chance=False`): the one
        bounded last-chance call offers no tool calls at all
        (`available_tool_ids=[]`), so there is no way left for the model to
        act on new guidance, and rejecting it there would only throw away
        everything the investigation found in favor of the generic
        no-root-cause fallback. `_format_report` is what states the true,
        structurally-derived verification outcome in that case instead of
        trusting the model's own wording."""
        ungrounded = _ungrounded_identifiers(action, investigation)
        if ungrounded:
            # Don't accept an unverified claim at face value — the same
            # self-correction pattern used for a fixable DENIED response
            # above: feed back exactly what's wrong. Mid-investigation the
            # outer while loop's own check is what stops this if the model
            # keeps insisting; after the turn budget, the caller simply
            # doesn't retry and falls through to the safe fallback instead.
            note = (
                f"Your conclusion named {', '.join(ungrounded)}, which does not "
                "appear anywhere in this investigation's evidence or the "
                "DBA's own message — never state something as a finding "
                "unless it actually came from a tool result or what the "
                "DBA said. Revise your conclusion using only that."
            )
            investigation.transcript.append(
                {
                    "tool_id": "internal.grounding_check",
                    "reason": "Verifying the conclusion before reporting it.",
                    "result": {"rejected": ungrounded, "message": note},
                }
            )
            self._append_evidence_once(
                investigation,
                f"(a draft conclusion naming {', '.join(ungrounded)} was rejected — "
                "not found in any evidence gathered)",
            )
            logger.warning(
                "conclusion_rejected_ungrounded_identifiers",
                investigation_id=investigation.investigation_id,
                names=ungrounded,
            )
            await self._tool_client.log_decision_event(
                DecisionEventCreateRequest(
                    event_type="conclusion_rejected_ungrounded_identifiers",
                    conversation_id=state.conversation_id,
                    investigation_id=investigation.investigation_id,
                    provider=llm.provider_name,
                    model=llm.model,
                    payload={"names": ungrounded},
                )
            )
            return None

        if not final_chance and investigation.pending_verification is not None:
            # Structural enforcement of the project's own "SYSTEM VERIFIES
            # -> INUMI REPORTS THE ACTUAL OUTCOME" principle (README.md's
            # lifecycle diagram): a write that just executed must not be
            # treated as license to conclude "Completed" without an
            # independent re-check — never mark an incident resolved
            # merely because an action was submitted. Same self-correction
            # shape as the grounding check just above: reject, feed back
            # exactly what's missing, and let the loop's own turn budget
            # bound how many times this can happen — never an unbounded
            # wait for the model to eventually decide to check.
            pending = investigation.pending_verification
            verification_tools = " or ".join(pending["verification_tools"])
            note = (
                f"{pending['tool_id']} executed, but nothing has independently "
                f"re-checked yet whether it actually took effect — never mark an "
                f"action as resolved merely because it was submitted. Call "
                f"{verification_tools} to confirm the real-world outcome before "
                "concluding."
            )
            investigation.transcript.append(
                {
                    "tool_id": "internal.verification_check",
                    "reason": "Verifying the remediation's real-world effect before reporting it.",
                    "result": {"pending_tool_id": pending["tool_id"], "message": note},
                }
            )
            self._append_evidence_once(
                investigation,
                f"(a draft conclusion after {pending['tool_id']} was rejected — "
                "not yet independently verified)",
            )
            logger.warning(
                "conclusion_rejected_pending_verification",
                investigation_id=investigation.investigation_id,
                tool_id=pending["tool_id"],
            )
            await self._tool_client.log_decision_event(
                DecisionEventCreateRequest(
                    event_type="conclusion_rejected_pending_verification",
                    conversation_id=state.conversation_id,
                    investigation_id=investigation.investigation_id,
                    provider=llm.provider_name,
                    model=llm.model,
                    payload={"tool_id": pending["tool_id"]},
                )
            )
            return None

        if not final_chance:
            verdict = await self._self_critique_conclude(investigation, action, llm, state)
            if verdict is not None and not verdict.sound:
                note = (
                    f"A second review of your draft conclusion found a problem: "
                    f"{verdict.issue or 'it does not follow from the evidence gathered.'} "
                    "Revise your conclusion so it only states what the evidence actually "
                    "supports."
                )
                investigation.transcript.append(
                    {
                        "tool_id": "internal.self_critique_check",
                        "reason": "A second review of the conclusion before reporting it.",
                        "result": {"issue": verdict.issue, "message": note},
                    }
                )
                issue_text = verdict.issue or "did not follow from the evidence"
                self._append_evidence_once(
                    investigation, f"(a draft conclusion was rejected on review — {issue_text})"
                )
                logger.warning(
                    "conclusion_rejected_self_critique",
                    investigation_id=investigation.investigation_id,
                    issue=verdict.issue,
                )
                await self._tool_client.log_decision_event(
                    DecisionEventCreateRequest(
                        event_type="conclusion_rejected_self_critique",
                        conversation_id=state.conversation_id,
                        investigation_id=investigation.investigation_id,
                        provider=llm.provider_name,
                        model=llm.model,
                        payload={"issue": verdict.issue},
                    )
                )
                return None

        investigation.status = self._conclusion_stage(investigation)
        if action.likely_root_cause:
            investigation.findings.append(action.likely_root_cause)
        if action.recommendation:
            investigation.recommendations.append(action.recommendation)
        return AgentReply(
            text=self._format_report(investigation, action),
            status="ok",
            investigation_id=investigation.investigation_id,
        )

    @staticmethod
    def _no_root_cause_reply(investigation) -> AgentReply:
        """Shared by both places an investigation ends without ever
        reaching action=conclude: the turn cap itself, and the earlier,
        faster exit above once the model has shown it's stuck restating
        observations instead of concluding."""
        return AgentReply(
            text="I've run several diagnostic steps without reaching a confirmed root "
            "cause. Here's what I found:\n" + "\n".join(f"- {e}" for e in investigation.evidence),
            investigation_id=investigation.investigation_id,
        )

    def _next_playbook_action(self, investigation, available_ids: list[str]) -> ProposeToolCall | None:
        """The next step of this investigation's matched playbook, as a
        ready-to-submit ProposeToolCall — or None if there's no active
        playbook, or its steps are exhausted (falls back to the LLM either
        way). `investigation.playbook_step` always advances, even for a
        step that turns out to be unavailable — a fixed-argument step would
        fail the same way every time, so retrying it is never useful, and
        an unavailable tool is skipped silently (no turn spent, no Gateway
        round-trip) rather than surfaced as a denial for something the DBA
        never asked for by name."""
        playbook = get_playbook(investigation.playbook_id)
        if playbook is None:
            return None
        while investigation.playbook_step < len(playbook.steps):
            step = playbook.steps[investigation.playbook_step]
            investigation.playbook_step += 1
            if step.tool_id not in available_ids:
                continue
            return ProposeToolCall(
                tool_id=step.tool_id,
                arguments=dict(step.arguments),
                target={},
                reason=f"[{playbook.name} playbook] {step.purpose}",
            )
        return None

    @staticmethod
    def _problem_statement_for_llm(investigation) -> str:
        """The problem text handed to decide_next_action — unchanged for a
        freeform investigation with no observations yet. Once a matched
        playbook's steps are all used up, append its conclusion guidance so
        the one LLM call that follows (interpreting everything the playbook
        gathered) knows what "done" looks like for this specific scenario,
        and is nudged to conclude now if the evidence already supports it.
        That nudge deliberately states both directions, not just the
        "conclude now" one: `_next_playbook_action` returning None here
        does NOT mean the model is limited to conclude-only from this point
        — `available_tool_ids` handed to this same decide_next_action call
        is still the full, unrestricted tool menu (see
        `_continue_investigation`/`_run_investigation_loop`), so an
        inconclusive playbook is explicitly told it may propose one or more
        further freeform diagnostic tool calls, exactly like the original
        fully-freeform path, before ever concluding — a playbook only ever
        pre-decides a *known* scenario's fixed opening sequence, never a
        ceiling on what can be investigated afterward. Any such extra call
        still spends from the same shared `_MAX_INVESTIGATION_TURNS` budget
        as everything else (see the loop itself), so this can never let an
        investigation run longer than a fully freeform one could. Separately,
        once the model has already
        recorded an observation without concluding, add an escalating nudge
        before `_MAX_CONSECUTIVE_RECORD_OBSERVATIONS` cuts it off entirely
        — verified live: a real model can restate the same finding as one
        observation after another instead of ever calling conclude, even
        when a plain, complete answer (including "nothing wrong was found"
        or "X isn't configured") was already available. And separately,
        when resuming after the DBA sent a new reply (rather than this
        being the investigation's first turn), that raw reply is included
        verbatim — see investigation.last_message's own docstring for why:
        decide_next_action needs to actually see what was just said to
        interpret a short answer to whatever it last asked, instead of that
        answer only ever being visible to (and often misclassified by) a
        fresh, context-free intent-extraction call."""
        playbook = get_playbook(investigation.playbook_id)
        problem = investigation.problem
        if investigation.memory_context:
            # Background only — never grounding evidence. A prior
            # investigation's findings are the *previous* investigation's
            # confirmed facts, not this one's; `_ungrounded_identifiers`
            # only ever checks this investigation's own `evidence`/
            # `transcript`, so citing something from here without actually
            # re-confirming it this time will still get rejected as
            # ungrounded, which is deliberate. Each entry names the server
            # it actually happened on (same-server recall and cross-server
            # correlation are folded into the same list — see
            # `_bootstrap_investigation_memory`), so a DBA/model reading
            # this can tell "this happened here before" apart from "this
            # happened elsewhere too".
            lines = "\n".join(
                f"- [server: {m.get('server_id') or 'unknown'}] ({m['status']}) "
                f"{m['problem']!r} — findings: {m['findings']}; "
                f"recommendations: {m['recommendations']}"
                for m in investigation.memory_context
            )
            problem += (
                "\n\nFor background only, not verified findings for THIS "
                "investigation — related prior investigations:\n"
                f"{lines}"
            )
        if investigation.last_message:
            problem += f"\n\nThe DBA just replied: {investigation.last_message!r}"
        if playbook is not None and investigation.playbook_step >= len(playbook.steps):
            problem = (
                f"{problem}\n\nYou just followed the '{playbook.name}' "
                f"playbook — see the transcript for what was checked and found. "
                f"{playbook.conclusion_guidance} If the evidence gathered is enough "
                "to conclude, conclude now rather than proposing further tool calls. "
                "If it is NOT enough, propose one or more additional read-only "
                "diagnostic tool calls that would specifically fill the gap, before "
                "concluding — you are not limited to this playbook's fixed steps; "
                "any tool in the available list is yours to use."
            )
        if investigation.consecutive_record_observations:
            problem += (
                "\n\nYou have already recorded an observation without concluding. Do "
                "not record another restatement of the same finding — if you have "
                "enough information to answer the DBA's question (\"nothing wrong "
                "was found\" or \"X isn't configured\" both count as complete "
                "answers), you MUST use action=conclude now instead."
            )
        return problem

    @staticmethod
    def _strip_unschematized_arguments(
        action: ProposeToolCall, tool_allowed_arguments: dict[str, set[str]] | None
    ) -> dict:
        """Defense-in-depth backstop for a live, repeated finding: even with
        the prompt telling the model each tool's real schema (see
        `_continue_investigation`'s `tool_requirements`/`tool_allowed_arguments`
        and `StructuredLLMProvider._ACTION_SYSTEM`), a real model sometimes
        still folds an extra key into `arguments` that tool's schema
        forbids — most often `reason` (confused with `ProposeToolCall`'s own
        top-level `reason`) or a `session_id`/`database_name` it decided was
        relevant. The Gateway's own argument model is `extra="forbid"`, so
        an unstripped extra key is rejected outright as INVALID_ARGUMENTS —
        self-correctable (see `_SELF_CORRECTABLE_DENIAL_CODES`), but always
        at the cost of one wasted turn and Gateway round-trip.
        Silently dropping a key the tool's own schema never declared is
        always safe: it can never have been a value that tool would have
        accepted anyway. `tool_allowed_arguments` being None (the tool
        wasn't in the available list, or no schema info was supplied — see
        the unit tests in test_orchestrator_self_correction.py that call
        this without it) skips stripping entirely rather than guessing."""
        allowed = (tool_allowed_arguments or {}).get(action.tool_id)
        if allowed is None:
            return action.arguments
        extra = set(action.arguments) - allowed
        if not extra:
            return action.arguments
        logger.info(
            "stripped_unschematized_tool_arguments",
            tool_id=action.tool_id,
            dropped=sorted(extra),
        )
        return {k: v for k, v in action.arguments.items() if k in allowed}

    async def _submit_and_relay(
        self,
        state: ConversationState,
        investigation,
        action: ProposeToolCall,
        channel: str,
        channel_account_id: str,
        tool_allowed_arguments: dict[str, set[str]] | None = None,
        tool_operation_types: dict[str, OperationType] | None = None,
    ) -> AgentReply | None:
        if investigation.read_only and not _is_confirmed_read_tool(
            action.tool_id, tool_operation_types
        ):
            # THE guarantee, for an unattended run: nothing that isn't a
            # confirmed read-only diagnostic is ever submitted. This is the
            # second of three layers (see `_continue_investigation`'s
            # filtered tool menu above and the APPROVAL_REQUIRED branch
            # below), and the only one that depends on nothing outside this
            # process's own control flow — no prompt wording, no provider
            # implementation's post-validation, no Gateway decision.
            #
            # Placed HERE, before `ToolCallRequest` is even constructed,
            # rather than after the Gateway responds, and that ordering is
            # the whole point: a write that reaches the Gateway has already
            # had a policy decision made about it, and depending on the
            # tool and the configured identity's role that decision can be
            # APPROVAL_REQUIRED — which creates a real, live approval
            # record with a real TTL sitting in a channel. "Proactive
            # output is text + recommendation only" has to mean the write
            # never left this process, not that it was stopped somewhere
            # further down; anything later is already too late to honor
            # that. So a dropped proposal costs exactly one Agent-local
            # decision and zero network hops.
            #
            # Deliberately NOT an error and NOT the end of the turn: the
            # finding the model was reacting to is real and belongs in the
            # digest. Returning None continues the loop exactly like a
            # FAILED diagnostic does, with the drop recorded as evidence —
            # so the model's own next turn can see that its proposal went
            # nowhere and write it up as a recommendation for a DBA
            # instead, which is precisely the output this feature is
            # supposed to produce.
            logger.warning(
                "readonly_run_dropped_write_proposal",
                investigation_id=investigation.investigation_id,
                tool_id=action.tool_id,
                reason=action.reason,
            )
            investigation.transcript.append(
                {
                    "tool_id": _READ_ONLY_GUARD_TOOL_ID,
                    "reason": "Enforcing read-only mode on an unattended, scheduled run.",
                    "result": {
                        "dropped_tool_id": action.tool_id,
                        "dropped_reason": action.reason,
                        "message": (
                            f"{action.tool_id} was NOT submitted. This is an unattended "
                            "scheduled health sweep: it reports findings and "
                            "recommendations as text only and never takes an action. "
                            "Describe what you would recommend and why, and conclude — "
                            "a DBA will decide whether to act on it."
                        ),
                    },
                }
            )
            investigation.evidence.append(
                f"(a proposed {action.tool_id} was not submitted — this scheduled run "
                "is read-only and never executes an action)"
            )
            return None

        arguments = self._strip_unschematized_arguments(action, tool_allowed_arguments)
        request = ToolCallRequest(
            tool_id=action.tool_id,
            arguments=arguments,
            target={**state.database_context, **action.target},
            reason=action.reason,
            conversation_id=state.conversation_id,
            investigation_id=investigation.investigation_id,
            request_id=new_id("req"),
            channel=channel,
            channel_account_id=channel_account_id,
        )
        try:
            response = await self._tool_client.submit(request)
        except Exception as exc:  # noqa: BLE001 — a network-level failure reaching the
            # Gateway (verified live: httpcore.ReadTimeout when the database itself was
            # overloaded) — distinct from ToolCallStatus.FAILED below, which means the
            # Gateway/Execution pipeline DID respond with a structured "this call
            # failed" decision. Here we never got a response to relay at all, and
            # previously nothing caught that — it surfaced as an unhandled 500 instead
            # of a message. Ends the turn immediately rather than burning the rest of
            # the turn budget on further calls to the same likely-still-overloaded
            # target (see _SUBMIT_TIMEOUT_SECONDS for why this is bounded quickly).
            logger.warning(
                "tool_call_submit_failed", tool_id=action.tool_id, reason=action.reason, error=str(exc)
            )
            investigation.transcript.append(
                {"tool_id": action.tool_id, "reason": action.reason, "result": {"error": str(exc)}}
            )
            investigation.evidence.append(f"{action.tool_id} did not respond in time: {exc}")
            return AgentReply(
                text=(
                    f"I couldn't get a response for {action.tool_id} in time — the "
                    "database or Gateway may be under heavy load right now, which "
                    "could itself be relevant to what you're investigating. Try "
                    "again in a moment."
                ),
                status="error",
                investigation_id=investigation.investigation_id,
            )

        # Remember a database once it's actually been resolved — by the
        # DBA naming it, or (verified live, the recurring complaint this
        # exists for) by the model self-correcting an INVALID_TARGET
        # rejection within this same investigation — so a later message in
        # this conversation never has to re-supply it. Mirrors how
        # environment/instance already persist in state.database_context.
        # Safe to persist on any status except a DENIED specifically
        # *about* the target (INVALID_TARGET): the Gateway's own target
        # resolution runs before authorization/policy/risk/rate-limiting
        # in its pipeline, so EXECUTED, APPROVAL_REQUIRED, FAILED (an
        # adapter-level problem, unrelated to target correctness), or a
        # DENIED for any other reason (RBAC, rate limit, ...) all still
        # mean this exact database name was independently accepted as
        # valid for this server — never a guess of our own.
        database = request.target.get("database")
        if database and not (
            response.status == ToolCallStatus.DENIED and response.failure_code == "INVALID_TARGET"
        ):
            state.database_context["database"] = database

        if response.status == ToolCallStatus.APPROVAL_REQUIRED and investigation.read_only:
            # Third layer. Reachable only for a tool the live catalog
            # classifies READ that a deployment's `config/policy.yaml`
            # nonetheless puts behind approval for this identity/
            # environment — the gate above has already made a write
            # impossible, so this is not a second chance to catch one. It
            # exists because an unattended run must not leave an approval
            # card in a channel either: nobody is in the conversation that
            # produced it, so a DBA scrolling past would be asked to
            # rubber-stamp an action with none of the context a live
            # investigation would have given them, and `state` here is an
            # ephemeral object (see `run_comprehensive_summary`) that no
            # later `/approve` could ever resolve against anyway.
            #
            # The Gateway has already recorded its own approval request by
            # this point and that is left alone on purpose — it expires on
            # its own TTL (see OPERATIONS.md's "Approval queue hygiene"),
            # and the Agent has no authority to cancel a Gateway decision.
            # What this refuses is the Agent's half: no `PendingApproval`
            # is stored and no `ApprovalCard` is ever returned, so nothing
            # actionable reaches a human. Recorded as evidence and the loop
            # continues, same as a dropped proposal.
            logger.warning(
                "readonly_run_declined_approval_card",
                investigation_id=investigation.investigation_id,
                tool_id=action.tool_id,
                approval_id=response.approval_id,
            )
            investigation.transcript.append(
                {
                    "tool_id": _READ_ONLY_GUARD_TOOL_ID,
                    "reason": "Declining an approval card on an unattended, scheduled run.",
                    "result": {
                        "dropped_tool_id": action.tool_id,
                        "message": (
                            f"{action.tool_id} requires DBA approval, which an "
                            "unattended scheduled run must never request. Report this "
                            "as something a DBA needs to look at, and conclude."
                        ),
                    },
                }
            )
            investigation.evidence.append(
                f"({action.tool_id} requires DBA approval — not requested, because this "
                "scheduled run never asks for one)"
            )
            return None

        if response.status == ToolCallStatus.APPROVAL_REQUIRED:
            state.pending_approval = PendingApproval(
                approval_id=response.approval_id,
                tool_id=action.tool_id,
                summary=action.reason,
                request=request.model_dump(mode="json"),
            )
            risk = response.risk or {}
            card = ApprovalCard(
                approval_id=response.approval_id,
                tool_id=action.tool_id,
                target_summary=str(request.target),
                reason=action.reason,
                risk_level=risk.get("risk_level", "UNKNOWN"),
                blast_radius=risk.get("blast_radius", "UNKNOWN"),
            )
            return AgentReply(
                text=(
                    f"Recommended action: {action.tool_id} — {action.reason}\n"
                    f"Risk: {card.risk_level}\n\nThis requires DBA approval before I run it."
                ),
                status="approval_required",
                approval_card=card,
                investigation_id=investigation.investigation_id,
            )

        if response.status == ToolCallStatus.DENIED:
            self_correctable_code = response.failure_code in _SELF_CORRECTABLE_DENIAL_CODES
            correctable = self_correctable_code and investigation.turn_count < _MAX_INVESTIGATION_TURNS
            if correctable:
                # Feed the exact rejection back as an observation and let
                # the loop continue — the LLM gets a concrete next chance to
                # fix the specific problem instead of the whole turn ending
                # on a malformed-but-fixable call.
                investigation.transcript.append(
                    {
                        "tool_id": action.tool_id,
                        "reason": action.reason,
                        "result": {"error": response.message, "failure_code": response.failure_code},
                    }
                )
                investigation.evidence.append(
                    f"{action.tool_id} was rejected ({response.failure_code}): {response.message}"
                )
                return None
            if self_correctable_code:
                # Ran out of turns to self-correct — not a genuine policy/
                # permission fact. `response.message` here is deliberately
                # the raw, technical validation detail (see
                # tool_call_handler.py's INVALID_ARGUMENTS construction:
                # str(a Pydantic ValidationError)[:3]) — useful as feedback
                # for the LLM's own self-correction above, never meant for
                # a human. Verified live: a real DBA got that raw dump
                # (field names, "extra_forbidden", a pydantic.dev docs URL)
                # as the agent's entire reply, once the turn budget ran out
                # right on a self-correctable failure. Record the real
                # detail for audit, but show something a DBA can act on.
                logger.warning(
                    "self_correctable_denial_exhausted_turn_budget",
                    tool_id=action.tool_id,
                    failure_code=response.failure_code,
                    detail=response.message,
                )
                return AgentReply(
                    text=(
                        f"I ran out of attempts trying to get the request format "
                        f"right for {action.tool_id} — the last attempt failed "
                        f"validation ({response.failure_code}). Try rephrasing your "
                        "request, or ask again more specifically."
                    ),
                    status="denied",
                    investigation_id=investigation.investigation_id,
                )
            return AgentReply(
                text=f"I can't do that: {response.message}",
                status="denied",
                investigation_id=investigation.investigation_id,
            )

        if response.status == ToolCallStatus.FAILED:
            # An adapter-level failure (e.g. a diagnostic not implemented
            # for this engine, a transient connection error) — distinct
            # from DENIED (a policy fact) and previously unhandled here,
            # which meant it fell through to the "EXECUTED" branch below and
            # got logged as if the call had actually succeeded. That's a
            # real correctness gap a playbook makes more likely to surface
            # (it proactively calls diagnostics like get_replication_status
            # that a given engine/topology may not implement) — record it
            # plainly as a failed step and keep going; a single failed
            # diagnostic shouldn't abort the rest of the investigation.
            investigation.transcript.append(
                {"tool_id": action.tool_id, "reason": action.reason, "result": {"error": response.message}}
            )
            investigation.evidence.append(f"{action.tool_id} failed: {response.message}")
            return None

        # EXECUTED
        investigation.transcript.append(
            {"tool_id": action.tool_id, "reason": action.reason, "result": response.result or {}}
        )
        evidence_line = f"{action.tool_id}: {response.message}{_affected_summary(response.result)}"
        investigation.evidence.append(evidence_line)
        investigation.actions.append({"tool_id": action.tool_id, "result": response.result})
        self._update_pending_verification(investigation, action, arguments, response, tool_operation_types)
        return None

    @staticmethod
    def _update_pending_verification(
        investigation,
        action: ProposeToolCall,
        arguments: dict,
        response: ToolCallResponse,
        tool_operation_types: dict[str, OperationType] | None,
    ) -> None:
        """Sets or clears `investigation.pending_verification`/
        `last_verification` (see their own docstrings on
        `InvestigationState`) — the structural half of the post-remediation
        verification mechanism; `_finalize_conclude`/`_format_report` are
        what act on what this records. Called for every EXECUTED tool call,
        not just writes — most calls match neither branch below and this
        is a no-op for them.

        `tool_operation_types.get(action.tool_id) == OperationType.WRITE`
        is checked as a defense-in-depth confirmation (mirroring
        `_strip_unschematized_arguments`'s own belt-and-suspenders
        reasoning) that the tool catalog *currently* still classifies this
        tool_id as a write, not stale assumption baked into this file. When
        `tool_operation_types` is None (older/direct test call sites, or a
        code path that never fetched the catalog), this falls back to
        trusting `_VERIFICATION_TOOLS_BY_WRITE_TOOL`'s own tool_id
        membership alone rather than skipping the check outright — that
        mapping only ever lists tools declared WRITE in
        `gateway/domain/tool_catalog.py` to begin with."""
        if investigation.pending_verification is not None and action.tool_id in (
            investigation.pending_verification["verification_tools"]
        ):
            still_present = _verification_still_shows_condition(
                action.tool_id, investigation.pending_verification.get("session_id"), response.result
            )
            investigation.last_verification = "UNRESOLVED" if still_present else "RESOLVED"
            investigation.pending_verification = None
            return

        verification_tools = _VERIFICATION_TOOLS_BY_WRITE_TOOL.get(action.tool_id)
        if verification_tools is None:
            return
        is_write = tool_operation_types is None or (
            tool_operation_types.get(action.tool_id) == OperationType.WRITE
        )
        if not is_write:
            return
        investigation.pending_verification = {
            "tool_id": action.tool_id,
            "verification_tools": verification_tools,
            "session_id": arguments.get("session_id"),
            "reason": action.reason,
        }
        investigation.last_verification = None

    async def handle_approval_decision(
        self, *, conversation_id: str, decision: str, channel: str, channel_account_id: str
    ) -> AgentReply:
        state = self._context.get_or_create(conversation_id, channel, "", channel_account_id)
        pending = state.pending_approval
        if pending is None:
            return AgentReply(text="There is no pending approval on this conversation.", status="error")

        if decision == "reject":
            await self._tool_client.reject(pending.approval_id, channel, channel_account_id)
            state.pending_approval = None
            return AgentReply(text="Understood — action rejected and will not run.", status="ok")

        result = await self._tool_client.approve(pending.approval_id, channel, channel_account_id)
        if result.get("status") == "ERROR":
            # `pending` deliberately NOT cleared here (see field above) --
            # e.g. a separation-of-duties rejection means THIS identity
            # can't approve their own request, not that the request itself
            # is dead. A different, eligible DBA must still be able to act
            # on the same card.
            return AgentReply(
                text=f"Approval failed: {result.get('detail')}",
                status="error",
                approval_still_pending=True,
            )
        if result.get("status") == "AWAITING_SECOND_APPROVAL":
            # Also not cleared -- this leg succeeded, but a second,
            # different approver still needs to act on the same card.
            return AgentReply(
                text="Recorded — this critical action also needs a second approver.",
                status="ok",
                approval_still_pending=True,
            )

        # Fully approved — resubmit the exact original request with the approval_id.
        request = ToolCallRequest.model_validate({**pending.request, "approval_id": pending.approval_id})
        state.pending_approval = None  # cleared regardless — the Gateway already
        # recorded the approval; re-approving on a resubmit failure isn't meaningful.
        try:
            response = await self._tool_client.submit(request)
        except Exception as exc:  # noqa: BLE001 — same network-level-failure case as
            # _submit_and_relay above, but here the approval was already granted
            # server-side before this call — tell the DBA plainly what to check
            # rather than leaving them wondering whether the approved action ran.
            logger.warning(
                "approved_action_resubmit_failed", approval_id=pending.approval_id, error=str(exc)
            )
            return AgentReply(
                text=(
                    f"Your approval was recorded, but I couldn't confirm {pending.tool_id} "
                    f"executed — the Gateway didn't respond in time. Check the audit trail "
                    f"for approval_id {pending.approval_id} before retrying (see "
                    'OPERATIONS.md\'s "a DBA reports \'I approved it but nothing happened\'" runbook).'
                ),
                status="error",
            )

        investigation = state.investigation
        if response.status == ToolCallStatus.EXECUTED:
            if investigation is not None:
                investigation.actions.append({"tool_id": pending.tool_id, "result": response.result})
            return AgentReply(
                text=(
                    f"Action approved.\n\n{pending.tool_id} completed successfully.\n\n"
                    f"Result: {response.result}\n\nIncident status: MITIGATED."
                ),
                status="ok",
                investigation_id=investigation.investigation_id if investigation else None,
            )
        return AgentReply(
            text=f"Approved, but execution did not complete: {response.message}", status="error"
        )

    async def run_comprehensive_summary(
        self, *, server_id: str, environment: str, channel: str, channel_account_id: str
    ) -> ScheduledSummary:
        """Run the `comprehensive_summary` playbook once against one server
        and hand back its report — the per-server building block
        `agent.scheduled_report`'s daily digest calls once per registered
        server. The third public entry point on this class, alongside
        `handle_message` and `handle_approval_decision`, and the only one
        that isn't a human talking.

        Reuses the existing machinery wholesale rather than reimplementing
        any of it: this constructs the same `ConversationState` +
        `InvestigationState` pair `handle_message` would have, sets the same
        `playbook_id` `match_playbook` would have matched from a DBA typing
        "daily summary", and then calls the very same
        `_continue_investigation` — so every fixed step in
        `playbooks.library.comprehensive_summary`, the shared
        `_MAX_INVESTIGATION_TURNS` budget, the argument stripping, the
        DENIED self-correction, the FAILED-step-doesn't-abort behavior, the
        conclusion grounding check and the playbook's own
        `conclusion_guidance` all apply here identically and for free. The
        deliberate consequence is that this path can never drift from what a
        DBA gets when they ask for the same thing interactively; pinned by
        `test_scheduled_summary_reuses_the_real_playbook.py`, which asserts
        the exact tool sequence matches the playbook's own steps.

        What is *not* shared, on purpose:

        - **The state is ephemeral and never registered with the
          `ContextManager`.** A scheduled run must be invisible to the
          conversation layer: registering it would let an unattended sweep
          collide with (or clobber) a real DBA's live `database_context`,
          in-progress investigation, or pending approval on whatever
          `conversation_id` it happened to reuse, and would leave a
          concluded investigation sitting in a conversation nobody started.
          The `conversation_id` is synthetic and unique per run purely so
          the Gateway's own audit trail can correlate this run's calls with
          each other.
        - **`read_only=True`.** See `InvestigationState.read_only` and the
          three enforcement points it drives. This is the constraint the
          whole feature is built around: proactive output is text and a
          recommendation, never an action.
        - **The environment is supplied, never asked for.** `handle_message`
          refuses to guess an environment and asks the DBA (spec: "for
          production targets I won't guess"). There is nobody to ask here,
          so the caller passes the environment the server registry itself
          declares for this server id (`/v1/catalog/servers` — see
          `scheduled_report.DailyDigestRunner`), which is the registry's own
          fact rather than a guess of ours, and the Gateway independently
          re-resolves and validates the whole target regardless.
        """
        if get_playbook(_SCHEDULED_SUMMARY_PLAYBOOK_ID) is None:
            # Fail loudly rather than silently running a *freeform*
            # investigation against every server in the estate — see
            # `_SCHEDULED_SUMMARY_PLAYBOOK_ID`'s own comment. Unreachable
            # unless someone renames/removes the playbook, which is exactly
            # when a silent degradation would be hardest to notice.
            raise RuntimeError(
                f"Playbook '{_SCHEDULED_SUMMARY_PLAYBOOK_ID}' is not registered in "
                "agent.playbooks.library — the scheduled daily digest has no "
                "playbook to run."
            )

        state = ConversationState(
            conversation_id=f"scheduled-digest:{server_id}:{new_id('run')}",
            channel=channel,
            channel_thread_id="",
            channel_account_id=channel_account_id,
        )
        state.database_context["instance"] = server_id
        state.database_context["environment"] = environment
        investigation = InvestigationState(
            investigation_id=new_id("inv"),
            problem=_SCHEDULED_SUMMARY_PROBLEM.format(
                server_id=server_id, environment=environment
            ),
            playbook_id=_SCHEDULED_SUMMARY_PLAYBOOK_ID,
            read_only=True,
        )
        state.investigation = investigation

        reply = await self._continue_investigation(
            state, investigation, channel, channel_account_id
        )
        return self._unattended_summary(server_id, environment, investigation, reply)

    def _unattended_summary(
        self, server_id: str, environment: str, investigation, reply: AgentReply
    ) -> ScheduledSummary:
        """Shared by `run_comprehensive_summary` (the daily digest) and
        `run_triggered_investigation` (an alert-triggered investigation) —
        both hand this a just-concluded, nobody-watching investigation and
        get back the same structurally-derived `ScheduledSummary` a digest
        or an alert notification renders.

        "Did anything actually answer?" is deliberately structural, not a
        reading of the reply text: `investigation.actions` is appended to
        only in `_submit_and_relay`'s EXECUTED branch, so an empty list
        means not one diagnostic call succeeded — an unreachable server, a
        never-completed discovery, every step denied. The model will still
        happily write a fluent paragraph about having found nothing
        concerning in that situation, and neither caller must print that as
        a clean bill of health. See `scheduled_report.build_digest`: this is
        what makes "6 attempted, 2 failed to even respond" reportable
        instead of invisible.
        """
        error = ""
        if not investigation.actions:
            error = (
                "no diagnostic call succeeded — the server may be unreachable, its "
                "discovery may never have completed, or every check was denied"
            )
        elif reply.status != "ok":
            error = f"the investigation ended with status '{reply.status}'"

        if error:
            logger.warning(
                "unattended_investigation_incomplete",
                server_id=server_id,
                investigation_id=investigation.investigation_id,
                status=reply.status,
                executed_calls=len(investigation.actions),
            )

        return ScheduledSummary(
            server_id=server_id,
            environment=environment,
            investigation_id=investigation.investigation_id,
            status=reply.status,
            text=reply.text,
            findings=tuple(investigation.findings),
            recommendations=tuple(investigation.recommendations),
            dropped_proposals=tuple(
                str(entry.get("result", {}).get("dropped_tool_id", ""))
                for entry in investigation.transcript
                if entry.get("tool_id") == _READ_ONLY_GUARD_TOOL_ID
            ),
            error=error,
        )

    async def run_triggered_investigation(
        self, *, server_id: str, environment: str, channel: str, channel_account_id: str, problem: str
    ) -> ScheduledSummary:
        """Run one freeform, read-only investigation against one server,
        seeded with a problem statement this call already knows — the
        building block an event-driven trigger (a monitoring webhook firing
        because a threshold was breached — see `agent.alert_trigger`) calls
        once per alert. The fourth public entry point on this class,
        alongside `handle_message`, `handle_approval_decision`, and
        `run_comprehensive_summary`.

        Deliberately freeform (`playbook_id` left `None`), unlike
        `run_comprehensive_summary`'s fixed `comprehensive_summary`
        playbook: a digest sweep asks "how is this server doing" and always
        runs the same checklist, but an alert already names a specific
        symptom (e.g. "replication_lag_seconds is 340, threshold 120") that
        should drive what gets checked next — the same LLM-planned,
        turn-by-turn investigation a DBA typing that symptom into chat would
        get, via the same `_continue_investigation` loop, not a second
        planner. Every other property `run_comprehensive_summary` documents
        still holds identically here and for the same reasons: the state is
        ephemeral and never registered with the `ContextManager`, the
        environment is supplied (the server registry's own fact) rather
        than guessed, and — the constraint this whole feature is built
        around — `read_only=True`, enforced at the same three points, so an
        alert can only ever produce a report and a recommendation, never an
        executed action.
        """
        state = ConversationState(
            conversation_id=f"triggered-investigation:{server_id}:{new_id('run')}",
            channel=channel,
            channel_thread_id="",
            channel_account_id=channel_account_id,
        )
        state.database_context["instance"] = server_id
        state.database_context["environment"] = environment
        investigation = InvestigationState(
            investigation_id=new_id("inv"),
            problem=problem,
            read_only=True,
        )
        state.investigation = investigation

        reply = await self._continue_investigation(
            state, investigation, channel, channel_account_id
        )
        return self._unattended_summary(server_id, environment, investigation, reply)

    async def _handle_command_if_any(
        self, state: ConversationState, message: str, channel: str, channel_account_id: str
    ) -> AgentReply | None:
        stripped = message.strip()
        if stripped in ("/help",):
            return AgentReply(text=_HELP_TEXT)
        if stripped == "/status":
            return self._status_reply(state)
        if stripped == "/playbooks":
            return self._handle_playbooks_command()
        if stripped.startswith("/approve "):
            approval_id = stripped.split(" ", 1)[1].strip()
            if state.pending_approval and state.pending_approval.approval_id != approval_id:
                return AgentReply(
                    text="That approval id doesn't match the pending action on this conversation.",
                    status="error",
                )
            return await self.handle_approval_decision(
                conversation_id=state.conversation_id,
                decision="approve",
                channel=channel,
                channel_account_id=channel_account_id,
            )
        if stripped.startswith("/reject "):
            approval_id = stripped.split(" ", 1)[1].strip()
            if state.pending_approval and state.pending_approval.approval_id != approval_id:
                return AgentReply(
                    text="That approval id doesn't match the pending action on this conversation.",
                    status="error",
                )
            return await self.handle_approval_decision(
                conversation_id=state.conversation_id,
                decision="reject",
                channel=channel,
                channel_account_id=channel_account_id,
            )
        if stripped in ("/models", "/model"):
            return await self._handle_model_command(state, stripped)
        if stripped.startswith("/model "):
            return await self._handle_model_command(state, stripped)
        if stripped == "/approvers":
            return self._handle_approvers_command()
        if stripped == "/servers":
            return await self._handle_servers_command()
        if stripped == "/catalog" or stripped.startswith("/catalog "):
            parts = stripped.split(maxsplit=1)
            return await self._handle_catalog_command(parts[1].strip() if len(parts) > 1 else None)
        if stripped == "/discover" or stripped.startswith("/discover "):
            parts = stripped.split(maxsplit=1)
            server_id = parts[1].strip() if len(parts) > 1 else None
            return await self._handle_discover_command(server_id, channel, channel_account_id)
        return None

    # Plain-language phrasing for every stage `effective_status` can return
    # except AWAITING_VERIFICATION, which needs the specific tool_id it's
    # waiting on — see `_stage_phrase` below, the one place that reads this.
    _STAGE_PHRASES: dict[str, str] = {
        "INVESTIGATING": "Investigating.",
        "AWAITING_CLARIFICATION": "Awaiting your answer to a clarifying question.",
        "CONCLUDED_VERIFIED": (
            "Concluded — the remediation was independently re-checked and confirmed resolved."
        ),
        "CONCLUDED_UNRESOLVED": (
            "Concluded — the remediation was independently re-checked and did NOT resolve the condition."
        ),
        "CONCLUDED_UNVERIFIED": "Concluded — a remediation ran but was never independently verified.",
        "CONCLUDED_NO_ACTION": "Concluded.",
    }

    @staticmethod
    def _stage_phrase(inv) -> str:
        """The plain-language phrasing `/status` shows for `inv`'s current
        stage — reads `effective_status` (see its own docstring on
        `InvestigationState`), never `status` directly, so a write still
        awaiting its post-execution re-check is reported as
        AWAITING_VERIFICATION here even though `status` itself was never
        actually set to that value."""
        stage = inv.effective_status
        if stage == "AWAITING_VERIFICATION":
            tool_id = (inv.pending_verification or {}).get("tool_id", "the last action")
            return f"Awaiting independent verification of {tool_id}."
        return AgentOrchestrator._STAGE_PHRASES[stage]

    @staticmethod
    def _status_reply(state: ConversationState) -> AgentReply:
        inv = state.investigation
        if inv is None:
            return AgentReply(text="No active investigation on this conversation.")
        playbook = get_playbook(inv.playbook_id)
        playbook_note = (
            f" Following the '{playbook.name}' playbook (step "
            f"{min(inv.playbook_step, len(playbook.steps))}/{len(playbook.steps)})."
            if playbook is not None
            else ""
        )
        return AgentReply(
            text=f"Investigation {inv.investigation_id}: {AgentOrchestrator._stage_phrase(inv)}"
            f"{playbook_note} {len(inv.evidence)} observations so far.",
            investigation_id=inv.investigation_id,
        )

    async def _handle_meta_command(
        self, intent, state: ConversationState, channel: str, channel_account_id: str
    ) -> AgentReply:
        """Free-text equivalent of the exact slash commands above — never
        require the literal syntax (spec: the DBA should never be bound to
        a fixed message structure). `intent.instance_hint` doubles as the
        target server id for catalog/discover when one was named."""
        command = intent.meta_command
        if command == "help":
            return AgentReply(text=_HELP_TEXT)
        if command == "status":
            return self._status_reply(state)
        if command == "playbooks":
            return self._handle_playbooks_command()
        if command == "servers":
            return await self._handle_servers_command()
        if command == "catalog":
            return await self._handle_catalog_command(intent.instance_hint)
        if command == "discover":
            return await self._handle_discover_command(intent.instance_hint, channel, channel_account_id)
        if command == "models":
            return await self._handle_model_command(state, "/models")
        if command == "approvers":
            return self._handle_approvers_command()
        if command in ("approve", "reject"):
            # Only ever acts on the one pending approval this conversation
            # already has (handle_approval_decision itself replies clearly
            # if there isn't one) — never a guess at *which* action, since
            # there is only ever the single one already shown to the DBA
            # via its approval card.
            return await self.handle_approval_decision(
                conversation_id=state.conversation_id,
                decision=command,
                channel=channel,
                channel_account_id=channel_account_id,
            )
        return AgentReply(text=_HELP_TEXT)  # unreachable given IntentExtraction's own enum

    def _handle_playbooks_command(self) -> AgentReply:
        lines = [f"- {p.name}: {p.description}" for p in PLAYBOOKS]
        return AgentReply(
            text="I automatically follow one of these fixed diagnostic sequences "
            "when your message matches its scenario, instead of investigating "
            "fully freeform:\n" + "\n".join(lines)
        )

    async def _handle_servers_command(self) -> AgentReply:
        servers = await self._tool_client.list_servers()
        if not servers:
            return AgentReply(text="No servers are registered.")
        lines = []
        for s in servers:
            cat = s.get("catalog")
            if cat:
                # `databases` is already the full list of discovered database
                # NAMES on this same /v1/catalog/servers response (see
                # gateway/api/routers/catalog.py's list_servers — the same
                # field `_known_database_names` already reads elsewhere) —
                # free to include here too, no extra discovery call or new
                # tool needed. Answers "what databases do you have access
                # to?" directly instead of only a bare count, which was the
                # one real content gap in this reply. Truncated defensively;
                # the full per-database detail still lives behind
                # /catalog <id>.
                names = cat.get("databases") or []
                preview = ", ".join(names[:8]) + (", …" if len(names) > 8 else "")
                db_part = f" ({preview})" if preview else ""
                summary = (
                    f"{cat['database_count']} databases{db_part}, discovered "
                    f"{(cat['discovered_at'] or '')[:16]}"
                )
            else:
                summary = "not yet discovered — run /discover"
            lines.append(
                f"- {s['id']}  [{s['environment']}/{s['platform']}, {s['criticality']}]  {summary}"
            )
        return AgentReply(text="Registered servers:\n" + "\n".join(lines))

    @staticmethod
    def _handle_approvers_command() -> AgentReply:
        return AgentReply(text=_APPROVAL_MODEL_TEXT)

    async def _handle_catalog_command(self, server_id: str | None) -> AgentReply:
        if not server_id:
            return AgentReply(text="Which server's catalog would you like to see? (see /servers)")
        data = await self._tool_client.get_server_catalog(server_id)
        cat = (data or {}).get("catalog")
        if not cat:
            return AgentReply(text=f"No catalog for '{server_id}' yet — run /discover {server_id}")
        lines = [f"{data['server']['id']} — {cat['engine_edition']} {cat['engine_version']}"]
        for db in cat["databases"][:40]:
            kinds: dict[str, int] = {}
            for o in db["objects"]:
                kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
            size = f"{db['size_bytes'] / 1e6:.0f}MB" if db.get("size_bytes") else "?"
            exts = f", {len(db['extensions'])} extensions" if db.get("extensions") else ""
            lines.append(f"  {db['name']} ({db['state']}, {size}) — {dict(kinds)}{exts}")
        if cat.get("warnings"):
            lines.append(f"  warnings: {cat['warnings'][:3]}")
        # Least-privilege finding (see common/models/catalog.py's
        # LeastPrivilegeFinding): a diagnostic login that can SELECT user
        # data contradicts this system's core premise, so it is surfaced
        # plainly here rather than buried in the catalog JSON. Rendered from
        # `warning_text()` — the same single source the Gateway's own
        # WARNING log uses, so the two can never drift apart. `.get` rather
        # than `[...]` throughout: a catalog discovered before this field
        # existed round-trips as `None` and must still render.
        finding = cat.get("least_privilege") or {}
        warning = LeastPrivilegeFinding.model_validate(finding).warning_text() if finding else None
        if warning:
            lines.append(f"  {warning}")
            if finding.get("scope_note"):
                lines.append(f"    scope: {finding['scope_note']}")
        return AgentReply(text="\n".join(lines))

    async def _handle_discover_command(
        self, server_id: str | None, channel: str, channel_account_id: str
    ) -> AgentReply:
        result = await self._tool_client.refresh_catalog(channel, channel_account_id, server_id)
        if result.get("status") == "ERROR":
            return AgentReply(text=f"Discovery failed: {result.get('detail')}", status="error")
        if server_id is not None:
            # Single-server refresh: {"server_id", "databases", "warnings", "error"?}
            # (gateway's POST /v1/catalog/refresh/{id} — see
            # gateway/api/routers/catalog.py:refresh_one).
            reported_id = result.get("server_id", server_id)
            if "error" in result:
                return AgentReply(
                    text=f"Discovery failed for {reported_id}: {result['error']}",
                    status="error",
                )
            db_count = len(result.get("databases") or [])
            return AgentReply(text=f"Discovery complete: {reported_id} ({db_count} databases).")
        # Refresh-all: {server_id: "ok (N databases)" | "failed: <reason>"} — one
        # clean line per server, same style as _handle_servers_command above,
        # never the raw dict repr.
        lines = [f"- {sid}: {status}" for sid, status in result.items()]
        return AgentReply(text="Discovery complete:\n" + "\n".join(lines))

    async def _handle_model_command(self, state: ConversationState, stripped: str) -> AgentReply:
        registry = self._llm_registry
        parts = stripped.split()

        # `/models` — list what's available.
        if parts[0] == "/models":
            return AgentReply(text=await registry.describe_available())

        # `/model` — show the current selection.
        if len(parts) == 1:
            if state.llm_provider:
                current = state.llm_provider + (f" / {state.llm_model}" if state.llm_model else "")
                source = "this conversation"
            else:
                dp, dm = registry.default()
                current = dp + (f" / {dm}" if dm else " (provider default)")
                source = "deployment default"
            hint = "" if registry.selection_enabled() else " (switching is disabled here)"
            return AgentReply(
                text=f"Current model: {current} — {source}.{hint}\n"
                "Use `/model <provider> <model>` to switch, or `/models` to list options."
            )

        # `/model <provider> [<model>]` — switch.
        provider = parts[1].lower()
        model = parts[2] if len(parts) >= 3 else None
        error = registry.validate_selection(provider, model)
        if error:
            return AgentReply(text=error, status="error")
        if provider != "mock" and model is not None:
            available_models = await registry.list_models(provider)
            if available_models and model not in available_models:
                preview = ", ".join(available_models[:10])
                return AgentReply(
                    text=f"'{model}' isn't in {provider}'s available models. Options: {preview}",
                    status="error",
                )
        state.llm_provider = provider
        state.llm_model = model
        chosen = provider + (f" / {model}" if model else " (provider default)")
        return AgentReply(text=f"Model for this conversation set to {chosen}.")

    @staticmethod
    def _format_report(investigation, conclusion: Conclude) -> str:
        lines = []
        playbook = get_playbook(investigation.playbook_id)
        if playbook is not None:
            lines.append(f"Followed the '{playbook.name}' playbook.")
        lines.append(f"Summary: {conclusion.summary}")
        if investigation.evidence:
            lines.append("Evidence: " + "; ".join(investigation.evidence))
        if conclusion.likely_root_cause:
            prefix = {"confirmed": "Confirmed", "likely": "Likely", "unable_to_confirm": "Unable to confirm"}[
                conclusion.confidence
            ]
            lines.append(f"Root Cause ({prefix}): {conclusion.likely_root_cause}")
        if conclusion.recommendation:
            lines.append(f"Recommended Action: {conclusion.recommendation}")
        verification_note = AgentOrchestrator._verification_note(investigation)
        if verification_note:
            lines.append(verification_note)
        return "\n".join(lines)

    @staticmethod
    def _conclusion_stage(investigation) -> str:
        """Which of the four CONCLUDED_* stages (see `InvestigationStage`
        on `context_manager.InvestigationState`) this investigation is
        ending in — derived from exactly the same two signals,
        `last_verification`/`pending_verification`, that `_verification_note`
        below reads to build the DBA-facing text. Deliberately the ONE
        place either of them reads those signals: `_verification_note`
        maps this same stage to its wording rather than re-deriving its
        own, separate judgment — so the stage recorded in
        `investigation.status` can never disagree with what the reply
        itself actually says happened. Called at every point this
        investigation transitions into a CONCLUDED_* stage — not just the
        primary action=conclude path in `_finalize_conclude` below, but
        also the turn-cap, stuck-observation, and clarification-exhausted
        fallbacks in `_run_investigation_loop`, none of which ever build a
        `Conclude` action of their own to pass through here."""
        if investigation.last_verification == "RESOLVED":
            return "CONCLUDED_VERIFIED"
        if investigation.last_verification == "UNRESOLVED":
            return "CONCLUDED_UNRESOLVED"
        if investigation.pending_verification is not None:
            return "CONCLUDED_UNVERIFIED"
        return "CONCLUDED_NO_ACTION"

    @staticmethod
    def _verification_note(investigation) -> str | None:
        """States the real, independently-checked outcome of a write this
        investigation performed — structurally derived (via
        `_conclusion_stage` above) from `investigation.last_verification`/
        `pending_verification`, never from the model's own free-text
        Conclude, so a DBA is never left trusting a bare "Completed" for a
        write whose real-world effect was never actually checked. Three
        distinct, deliberately-worded outcomes (never collapsed into one
        generic "done"): independently confirmed resolved, independently
        confirmed NOT resolved, or executed but never independently
        checked at all (reachable only via the last-chance Conclude call —
        see `_finalize_conclude`'s `final_chance`). CONCLUDED_NO_ACTION
        (no write/verification involved at all) has nothing to add here —
        returns None exactly like before this stage existed."""
        stage = AgentOrchestrator._conclusion_stage(investigation)
        if stage == "CONCLUDED_VERIFIED":
            return "Verification: independently re-checked afterward and confirmed resolved."
        if stage == "CONCLUDED_UNRESOLVED":
            return (
                "Verification: independently re-checked afterward — this did NOT "
                "actually resolve the condition; further action is likely still needed."
            )
        if stage == "CONCLUDED_UNVERIFIED":
            pending = investigation.pending_verification
            return (
                f"Verification: {pending['tool_id']} executed, but I ran out of turns "
                "before independently re-checking whether it actually took effect — "
                "treat this as executed but NOT independently verified."
            )
        return None
