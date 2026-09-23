"""THE test for the scheduled daily digest.

Why this file matters more than the rest of the feature put together: every
other part of the digest, if it breaks, produces a bad *report* — a missing
server, an ugly format, a duplicated line. This one is the difference
between "Numi told a DBA about a blocking chain at 6am" and "Numi killed a
session on a production database at 6am because nobody was watching and the
model thought it was a good idea". Proactive output is text and a
recommendation, never an action (the non-negotiable constraint this whole
feature was scoped inside); these tests are what actually holds that.

Three independent layers enforce it, and they are tested independently
rather than only in combination — a guarantee that only holds when all three
agree is really no guarantee at all, because the first one to silently stop
working would be invisible:

  1. `_continue_investigation` never even offers a non-READ tool to the
     model on a read-only run.
  2. `_submit_and_relay` refuses to submit anything not *confirmed* READ by
     the Gateway's own live tool catalog — before a `ToolCallRequest` is
     constructed, so the write never leaves the Agent process at all.
  3. `_submit_and_relay` never converts an APPROVAL_REQUIRED response into a
     `PendingApproval` or an approval card on a read-only run.

Layer 2 is the load-bearing one, and every test here that exercises it does
so with an LLM double that proposes a write *anyway*, ignoring the filtered
menu from layer 1 — i.e. deliberately simulating the case where layer 1 and
the prompt have both already failed. That is the only honest way to test a
defense-in-depth layer: with every layer above it assumed broken.

The final assertion in most of these is about `tool_client.requests` — what
actually crossed the wire toward the Gateway. Not what the model asked for,
not what the reply said, not what got logged. If a write ever appears in
that list on a read-only run, this feature is unsafe and the test must fail.
"""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ConversationState, InvestigationState
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import (
    _READ_ONLY_GUARD_TOOL_ID,
    AgentOrchestrator,
    _is_confirmed_read_tool,
)
from numi.agent.planner.actions import Conclude, CritiqueVerdict, ProposeToolCall
from numi.common.config import Settings
from numi.common.models.tool import OperationType, ToolCallResponse, ToolCallStatus
from numi.gateway.domain.tool_catalog import build_tool_catalog

# The real production catalog, with its real READ/WRITE classifications —
# the same objects `_continue_investigation` fetches from the Gateway at
# runtime. A hand-rolled fake catalog here would let this test pass while
# the shipped classification said something else entirely.
_REAL_TOOLS = build_tool_catalog(Settings(_env_file=None))
_WRITE_TOOL_IDS = [t.tool_id for t in _REAL_TOOLS if t.operation_type != OperationType.READ]


def test_the_catalog_really_does_classify_a_write_as_a_write():
    """A guard on the guard. Every test below is meaningless if
    `database.kill_session` isn't actually in the catalog as a non-READ tool
    — the assertions would all pass vacuously against a tool that was never
    offered in the first place. Pin it explicitly so this file can never
    quietly degrade into testing nothing."""
    by_id = {t.tool_id: t for t in _REAL_TOOLS}
    assert "database.kill_session" in by_id
    assert by_id["database.kill_session"].operation_type == OperationType.WRITE
    assert by_id["database.get_health"].operation_type == OperationType.READ


class _FakeToolClient:
    """Records everything submitted. `requests` staying empty of writes is
    the actual subject of this file."""

    def __init__(self, *, response: ToolCallResponse | None = None):
        self._response = response or ToolCallResponse(
            status=ToolCallStatus.EXECUTED, message="ok", result={"rows": [], "row_count": 0}
        )
        self.requests: list[object] = []

    async def available_tools(self, channel: str, channel_account_id: str):
        return _REAL_TOOLS

    async def submit(self, request):
        self.requests.append(request)
        return self._response

    async def create_investigation(self, request):
        pass

    async def update_investigation(self, investigation_id, request):
        pass

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        return []

    async def get_cross_server_patterns(
        self, *, playbook_id, environment=None, exclude_server_id=None, limit=5
    ):
        return []

    async def log_decision_event(self, request):
        pass


class _WritePushingLLM:
    """A model that proposes a write regardless of what it was offered.

    Not a strawman: `LLMProvider` is an interface, and while
    `StructuredLLMProvider.decide_next_action` does post-validate a proposed
    tool_id against `available_tool_ids` (llm/base.py), that check lives in
    one provider implementation. A future provider, a subclass, or a bug in
    that validation would all produce exactly this behavior — which is
    precisely why layer 2 must not depend on it. Repeats its last action
    once the script runs out so a loop can always terminate."""

    provider_name = "fake"
    model = "fake-model"

    def __init__(self, actions: list):
        self._actions = list(actions)
        self.calls: list[dict] = []

    async def decide_next_action(self, **kwargs):
        self.calls.append(kwargs)
        if len(self._actions) > 1:
            return self._actions.pop(0)
        return self._actions[0]

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


def _orchestrator(tool_client, llm) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm), tool_client=tool_client, context=None
    )


def _read_only_investigation(**kwargs) -> tuple[ConversationState, InvestigationState]:
    state = ConversationState(
        conversation_id="scheduled-digest:postgres-local:run_1",
        channel="dev",
        channel_thread_id="",
        channel_account_id="dba_l2@example.com",
    )
    state.database_context.update({"instance": "postgres-local", "environment": "development"})
    investigation = InvestigationState(
        investigation_id="inv_scheduled_1",
        problem="Scheduled daily health sweep of postgres-local.",
        read_only=True,
        **kwargs,
    )
    state.investigation = investigation
    return state, investigation


# --------------------------------------------------------------- layer 2 ---


@pytest.mark.asyncio
async def test_a_write_proposal_is_never_submitted_on_a_read_only_run():
    """The single most important assertion in the feature: the model asks to
    kill a session, and nothing reaches the Gateway."""
    state, investigation = _read_only_investigation()
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.kill_session",
                arguments={"session_id": "13400"},
                target={},
                reason="Terminating the head blocker.",
            ),
            Conclude(summary="A blocking chain needs a DBA's attention."),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    reply = await orchestrator._continue_investigation(
        state, investigation, "dev", "dba_l2@example.com"
    )

    assert [r.tool_id for r in tool_client.requests] == []
    assert reply.approval_card is None
    assert state.pending_approval is None


@pytest.mark.asyncio
async def test_every_write_tool_in_the_catalog_is_blocked_not_just_kill_session():
    """Generalizes the assertion above across the entire catalog rather than
    the one write tool that happened to come to mind — the guard is a
    classification check, so nothing about it should be specific to
    `kill_session`, and a newly added write tool must be covered the day it
    is added without anyone remembering to update this file."""
    assert _WRITE_TOOL_IDS, "the catalog exposes no non-READ tools — test is vacuous"
    for tool_id in _WRITE_TOOL_IDS:
        state, investigation = _read_only_investigation()
        tool_client = _FakeToolClient()
        llm = _WritePushingLLM(
            actions=[
                ProposeToolCall(
                    tool_id=tool_id, arguments={}, target={}, reason="Remediating."
                ),
                Conclude(summary="Reported for a DBA to act on."),
            ]
        )
        orchestrator = _orchestrator(tool_client, llm)

        await orchestrator._continue_investigation(
            state, investigation, "dev", "dba_l2@example.com"
        )

        assert tool_client.requests == [], f"{tool_id} was submitted on a read-only run"


@pytest.mark.asyncio
async def test_an_unrecognized_tool_is_refused_rather_than_assumed_safe():
    """The guard fails closed. A tool_id the live catalog says nothing about
    is refused, not waved through on a naming convention — `database.get_*`
    looks read-only and is not a classification. The cost of being wrong
    here is asymmetric: refusing a read loses one line of a report;
    admitting an unclassifiable call loses the entire guarantee, in exactly
    the unattended context where nobody would notice."""
    state, investigation = _read_only_investigation()
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.get_something_that_does_not_exist",
                arguments={},
                target={},
                reason="Looks like a read.",
            ),
            Conclude(summary="done"),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert tool_client.requests == []


def test_the_read_only_predicate_itself_fails_closed():
    """The predicate in isolation, since the whole guard reduces to it."""
    catalog = {t.tool_id: t.operation_type for t in _REAL_TOOLS}
    assert _is_confirmed_read_tool("database.get_health", catalog)
    assert not _is_confirmed_read_tool("database.kill_session", catalog)
    # Unknown tool, and no catalog at all — both refuse.
    assert not _is_confirmed_read_tool("database.get_health", {})
    assert not _is_confirmed_read_tool("database.get_health", None)


@pytest.mark.asyncio
async def test_a_blocked_write_is_reported_as_a_finding_not_silently_dropped():
    """Blocking the action must not lose the insight that prompted it. The
    drop is recorded in the transcript and the evidence — so the model's own
    next turn can see its proposal went nowhere and write it up as a
    recommendation, which is the output this feature is supposed to produce
    — and surfaces on the ScheduledSummary for the digest to print."""
    state, investigation = _read_only_investigation()
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.kill_session",
                arguments={"session_id": "13400"},
                target={},
                reason="Terminating the head blocker.",
            ),
            Conclude(summary="Session 13400 is blocking four others."),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    guard_entries = [
        e for e in investigation.transcript if e.get("tool_id") == _READ_ONLY_GUARD_TOOL_ID
    ]
    assert len(guard_entries) == 1
    assert guard_entries[0]["result"]["dropped_tool_id"] == "database.kill_session"
    assert any("was not submitted" in e for e in investigation.evidence)
    # The model is told *why*, so its next turn can produce a recommendation
    # rather than retrying the same blocked action.
    assert "never takes an action" in guard_entries[0]["result"]["message"]


# --------------------------------------------------------------- layer 1 ---


@pytest.mark.asyncio
async def test_a_write_tool_is_never_even_offered_to_the_model():
    """The cheapest layer: on a read-only run the model's menu contains only
    reads, so in the ordinary case it has nothing to propose."""
    state, investigation = _read_only_investigation()
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    offered = llm.calls[0]["available_tool_ids"]
    assert offered, "no tools were offered at all — the test would pass vacuously"
    assert not set(offered) & set(_WRITE_TOOL_IDS)


# --------------------------------------------------------------- layer 3 ---


@pytest.mark.asyncio
async def test_an_approval_required_response_never_becomes_a_pending_approval():
    """A read tool a deployment's policy puts behind approval must not leave
    an approval card in a channel. Nobody is in the conversation that
    produced it, so a DBA scrolling past would be asked to rubber-stamp an
    action with none of the context a live investigation would have given
    them — and the ephemeral state a scheduled run uses means no later
    `/approve` could resolve against it anyway."""
    state, investigation = _read_only_investigation()
    tool_client = _FakeToolClient(
        response=ToolCallResponse(
            status=ToolCallStatus.APPROVAL_REQUIRED,
            approval_id="apr_123",
            message="Approval required.",
            risk={"risk_level": "MEDIUM", "blast_radius": "INSTANCE"},
        )
    )
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.get_health", arguments={}, target={}, reason="Baseline."
            ),
            Conclude(summary="A check here needs DBA approval."),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    reply = await orchestrator._continue_investigation(
        state, investigation, "dev", "dba_l2@example.com"
    )

    assert state.pending_approval is None
    assert reply.approval_card is None
    assert reply.status != "approval_required"
    assert any("requires DBA approval" in e for e in investigation.evidence)


# ------------------------------------------- the interactive path is intact ---


@pytest.mark.asyncio
async def test_an_ordinary_dba_investigation_still_submits_the_very_same_write():
    """The control. Everything above must be scoped to `read_only` runs and
    nothing else: a DBA asking for a remediation turn by turn still gets the
    normal LLM-proposes / Gateway-approves flow (spec §7, §37), completely
    unchanged. Without this assertion, "no write was submitted" would also
    pass if the guard had accidentally disabled writes everywhere — which
    would be a far larger, much quieter regression than the one the rest of
    this file is guarding against."""
    state, investigation = _read_only_investigation()
    investigation.read_only = False  # an ordinary, DBA-driven investigation
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.kill_session",
                arguments={"session_id": "13400"},
                target={},
                reason="Terminating the head blocker, as the DBA asked.",
            ),
            Conclude(summary="Terminated."),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert [r.tool_id for r in tool_client.requests] == ["database.kill_session"]
    assert not [
        e for e in investigation.transcript if e.get("tool_id") == _READ_ONLY_GUARD_TOOL_ID
    ]


# --------------------------------------------------- end to end, via the ---
# --------------------------------------------------- real entry point -----


@pytest.mark.asyncio
async def test_the_scheduled_entry_point_blocks_a_write_end_to_end():
    """The same guarantee through the actual public entry point the
    scheduler calls — not just through the loop internals — so the
    `read_only=True` wiring in `run_comprehensive_summary` is covered too.
    The playbook's five fixed read steps run first (all reads, all
    submitted), the model then proposes a kill, and only the five reads ever
    reach the Gateway."""
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.kill_session",
                arguments={"session_id": "13400"},
                target={},
                reason="Clearing the blocking chain.",
            ),
            Conclude(summary="Blocking chain found; a DBA should decide."),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    summary = await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    submitted = [r.tool_id for r in tool_client.requests]
    assert "database.kill_session" not in submitted
    assert set(submitted) <= {t.tool_id for t in _REAL_TOOLS if t.operation_type == OperationType.READ}
    # The attempt is visible to the digest rather than hidden.
    assert summary.dropped_proposals == ("database.kill_session",)


@pytest.mark.asyncio
async def test_the_triggered_entry_point_blocks_a_write_end_to_end():
    """Same guarantee through `run_triggered_investigation` — the
    alert-triggered sibling of `run_comprehensive_summary` (see
    `agent.alert_trigger`), sharing the identical `read_only=True` wiring
    via `_continue_investigation`. An unattended path with nobody present
    to click "approve" must never be the one place approval quietly stops
    being required, and that must hold regardless of which of the two
    unattended entry points got there."""
    tool_client = _FakeToolClient()
    llm = _WritePushingLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.kill_session",
                arguments={"session_id": "13400"},
                target={},
                reason="Clearing the blocking chain.",
            ),
            Conclude(summary="Blocking chain found; a DBA should decide."),
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    summary = await orchestrator.run_triggered_investigation(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
        problem="A monitoring alert fired for the postgres-local server — investigate.",
    )

    submitted = [r.tool_id for r in tool_client.requests]
    assert "database.kill_session" not in submitted
    assert set(submitted) <= {t.tool_id for t in _REAL_TOOLS if t.operation_type == OperationType.READ}
    assert summary.dropped_proposals == ("database.kill_session",)
