"""Conversation / investigation state (spec §25, §26, §37, §56).

Kept entirely in the Agent process (never a database credential or policy
decision in sight). Nothing stored here is ever treated as an authorization
grant — `database_context` is a convenience so "check it again" resolves to
the right target, but every tool call is still independently authorized by
the Gateway from scratch on every single request (spec §37).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Literal

from numi.agent.llm.fallback import FallbackEvent
from numi.common.ids import new_id

# The stages this codebase's control flow can actually and accurately
# observe an investigation transitioning through. Deliberately a small,
# bounded subset of a much richer 17-state investigation lifecycle from an
# external playbook spec (NEW -> TRIAGED -> ... -> AWAITING_APPROVAL ->
# APPROVED -> EXECUTING -> ... -> RESOLVED/CLOSED): most of those states
# aren't independently observable here (this Agent process has no distinct
# moment where an investigation becomes "TRIAGED" or "EVIDENCE_COLLECTED"
# as opposed to just "still investigating"), and the approval/execution
# richness that spec bakes into its own state machine already lives
# correctly elsewhere in this codebase — `PendingApproval` and the
# Gateway's own audit trail — so duplicating it into a second, parallel
# copy here would only ever be something that could drift out of sync with
# the real decision-maker, never a source of truth of its own.
#
# "AWAITING_VERIFICATION" is deliberately never a value `status` itself is
# ever assigned — see `InvestigationState.effective_status` below.
InvestigationStage = Literal[
    "INVESTIGATING",
    "AWAITING_CLARIFICATION",
    "AWAITING_VERIFICATION",
    "CONCLUDED_VERIFIED",
    "CONCLUDED_UNRESOLVED",
    "CONCLUDED_UNVERIFIED",
    "CONCLUDED_NO_ACTION",
]

# The four terminal stages — used by `InvestigationState.is_concluded` so a
# call site that only cares about "concluded vs. not" never has to
# enumerate all four itself (and can't silently miss one if a fifth were
# ever added later).
_CONCLUDED_STAGES: frozenset[str] = frozenset(
    {"CONCLUDED_VERIFIED", "CONCLUDED_UNRESOLVED", "CONCLUDED_UNVERIFIED", "CONCLUDED_NO_ACTION"}
)


@dataclasses.dataclass
class PendingApproval:
    approval_id: str
    tool_id: str
    summary: str
    # The exact ToolCallRequest (as a plain dict) that produced this
    # approval requirement — resubmitted verbatim (with approval_id filled
    # in) once approved, so the Gateway's action-hash check always matches.
    request: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class InvestigationState:
    investigation_id: str
    problem: str
    target: dict[str, Any] = dataclasses.field(default_factory=dict)
    status: InvestigationStage = "INVESTIGATING"
    evidence: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    hypotheses: list[str] = dataclasses.field(default_factory=list)
    findings: list[str] = dataclasses.field(default_factory=list)
    recommendations: list[str] = dataclasses.field(default_factory=list)
    actions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    transcript: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    turn_count: int = 0
    # Set once, when the investigation starts, from a deterministic keyword
    # match on the problem text (see agent.playbooks.library.match_playbook)
    # — None means this investigation is freeform, exactly as before
    # playbooks existed. Only the id is kept here (not the Playbook object
    # itself) so this state stays a plain, easily-inspectable/serializable
    # dataclass; look the object up via `get_playbook` when needed.
    playbook_id: str | None = None
    playbook_step: int = 0
    # True only for an investigation nobody is sitting in front of — today
    # that is exactly the scheduled daily digest (see
    # `orchestrator.run_comprehensive_summary` and `agent.scheduled_report`).
    # It means: this investigation may propose read-only diagnostics and
    # nothing else, ever. A write-shaped proposal is dropped before a
    # `ToolCallRequest` is even constructed, and an APPROVAL_REQUIRED
    # response is never turned into a `PendingApproval` — so an unattended
    # run can neither execute a write nor leave an approval card sitting in
    # a channel for a DBA to rubber-stamp without the context that produced
    # it.
    #
    # Why this lives on the investigation rather than being a parameter
    # threaded through the loop: every enforcement point
    # (`_continue_investigation`'s tool-menu filter, `_submit_and_relay`'s
    # hard gate and its APPROVAL_REQUIRED branch) already receives the
    # investigation, and a single field that travels with it cannot be
    # accidentally dropped by one call site the way a defaulted keyword
    # argument can. Defaults to False, so every pre-existing, DBA-driven
    # investigation behaves exactly as it did before this flag existed —
    # a DBA asking for a remediation turn-by-turn still gets the normal
    # LLM-proposes / Gateway-approves flow, unchanged.
    #
    # Pinned by tests/unit/test_scheduled_digest_never_writes.py, which is
    # the single most important test of the daily-digest feature.
    read_only: bool = False
    # How many record_observation actions the LLM has proposed *in a row*
    # (reset by any other action) — see orchestrator._MAX_CONSECUTIVE_
    # RECORD_OBSERVATIONS: a real model can get stuck restating the same
    # finding as one observation after another instead of ever calling
    # conclude, burning the whole turn budget on a case that was already
    # answerable after the first one.
    consecutive_record_observations: int = 0
    # Consecutive AskClarification turns — reset by any other action. Kept
    # separate from turn_count: a clarification is the DBA narrowing down a
    # target, not the model looping on diagnostics, so it doesn't spend the
    # shared diagnostic turn budget, but still needs its own bound (see
    # orchestrator._MAX_CLARIFICATION_TURNS) so an unresolved back-and-forth
    # can't run forever across many separate requests.
    clarification_count: int = 0
    # The DBA's raw reply when resuming an in-progress investigation — see
    # orchestrator.handle_message's resume path and _problem_statement_for_
    # llm. Set right before the next decide_next_action call, consumed
    # (cleared) by that same call so it doesn't linger and get repeated on
    # a later turn within the same request that has nothing to do with it.
    last_message: str = ""
    # Set the moment a write tool with a known, cheap, correlated read-only
    # re-check (see orchestrator._VERIFICATION_TOOLS_BY_WRITE_TOOL —
    # currently kill_session/cancel_query against get_blocking_sessions/
    # get_sessions/get_running_queries) executes, and cleared the moment one
    # of those re-check tools is itself proposed and executes — regardless
    # of what it finds; see `last_verification` for the actual verdict.
    # None means either no such write has happened yet this investigation,
    # or the one that did has already been followed by a re-check. Exists
    # because whether a remediation actually gets independently re-checked
    # was previously left entirely to the model's own discretion within its
    # turn budget — verified live, it sometimes does this unprompted, but
    # nothing forced it, so a DBA could get a clean "Completed" summary when
    # the underlying condition never actually cleared. See
    # `_finalize_conclude` (the check that acts on this) and
    # `_submit_and_relay` (where this is set/cleared).
    pending_verification: dict[str, Any] | None = None
    # The outcome of the most recently completed write-then-recheck
    # sequence this investigation has seen: "RESOLVED" (the recheck no
    # longer shows the condition the write targeted), "UNRESOLVED" (it
    # still does — the write executed but did not actually take effect), or
    # None (no such sequence has completed yet). Never set directly by the
    # model — derived structurally from a read tool's own result rows in
    # `_submit_and_relay`. Read by `_format_report` to state the real,
    # independently-checked outcome in the DBA-facing reply rather than
    # trusting the model's own free-text claim of success.
    last_verification: str | None = None
    # Set True the first time this investigation fires its one-time Gateway
    # persistence + memory-recall bootstrap (see
    # orchestrator._continue_investigation) — guards against redoing it on
    # every resumed turn, since the investigation object itself lives only
    # in this process and has no other way to know "have I already told the
    # Gateway about myself."
    remote_bootstrap_done: bool = False
    # Recalled findings from past, concluded investigations on the same
    # server (see gateway.domain.investigation_memory). Deliberately kept
    # separate from `evidence`, never merged into it: `evidence` is what
    # `_ungrounded_identifiers` treats as things *this* investigation
    # actually confirmed via a real tool call — folding a prior
    # investigation's unverified claim in there would let the model cite it
    # as if it had just confirmed it itself. This is prompt background only.
    memory_context: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def is_concluded(self) -> bool:
        """True once this investigation has reached any of the four
        CONCLUDED_* stages — the one place that answers "is this
        investigation concluded or not" so a call site that only cares
        about that (e.g. deciding whether a DBA's reply resumes an
        existing investigation or starts a fresh one) never has to
        enumerate all four itself."""
        return self.status in _CONCLUDED_STAGES

    @property
    def effective_status(self) -> InvestigationStage:
        """`status` as stored, with one addition: a write still awaiting
        its independent post-execution re-check (see
        `pending_verification`'s own docstring) reports as
        AWAITING_VERIFICATION here, even though `status` itself is never
        actually written to that value anywhere. Deliberate:
        `pending_verification` is already the one authoritative record of
        whether a re-check is outstanding — set and cleared in exactly one
        place, `orchestrator._update_pending_verification` — so mirroring
        it into a second, independently-written value on `status` would
        just be two things that could drift out of sync. Only ever
        overrides a plain INVESTIGATING: a stage the DBA is actively
        blocking on (AWAITING_CLARIFICATION) or a stage that's already
        final (any CONCLUDED_*) takes precedence over a re-check that can
        simply wait for the loop to get back to it."""
        if self.status == "INVESTIGATING" and self.pending_verification is not None:
            return "AWAITING_VERIFICATION"
        return self.status


@dataclasses.dataclass
class ConversationState:
    conversation_id: str
    channel: str
    channel_thread_id: str
    channel_account_id: str
    database_context: dict[str, Any] = dataclasses.field(default_factory=dict)
    investigation: InvestigationState | None = None
    pending_approval: PendingApproval | None = None
    # Per-conversation LLM choice (set via the `/model` command). None -> use
    # the deployment's configured default. Never an authorization input.
    llm_provider: str | None = None
    llm_model: str | None = None
    # Transient, per-INBOUND-MESSAGE scratch space, not conversation state:
    # every cross-provider LLM fallback that happened while handling the
    # message currently being answered (see agent.llm.fallback). Cleared at
    # the top of `AgentOrchestrator.handle_message` and drained by that same
    # method into the reply text, so a DBA is always told when a different
    # vendor produced their answer. Lives here purely because the
    # ConversationState is the one object already threaded through every
    # layer that can make an LLM call — nothing ever reads it across
    # messages, and it is never an authorization input.
    llm_fallback_notices: list[FallbackEvent] = dataclasses.field(default_factory=list)
    updated_at: dt.datetime = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.UTC)
    )


class ContextManager:
    """Process-local store. Swappable for a Redis-backed implementation
    behind the same interface for multi-instance deployments (spec §37's
    session continuity requirement doesn't require this to be durable across
    an Agent restart — a lost session simply starts a fresh investigation,
    which is the fail-closed choice over guessing stale state)."""

    def __init__(self) -> None:
        self._conversations: dict[str, ConversationState] = {}

    def get_or_create(
        self, conversation_id: str, channel: str, channel_thread_id: str, channel_account_id: str
    ) -> ConversationState:
        state = self._conversations.get(conversation_id)
        if state is None:
            state = ConversationState(
                conversation_id=conversation_id,
                channel=channel,
                channel_thread_id=channel_thread_id,
                channel_account_id=channel_account_id,
            )
            self._conversations[conversation_id] = state
        return state

    def start_investigation(self, state: ConversationState, problem: str) -> InvestigationState:
        state.investigation = InvestigationState(investigation_id=new_id("inv"), problem=problem)
        return state.investigation

    def touch(self, state: ConversationState) -> None:
        state.updated_at = dt.datetime.now(dt.UTC)
