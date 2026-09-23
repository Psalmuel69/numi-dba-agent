"""A DENIED response whose failure_code means "the call was shaped wrong"
(spec §35/§39) must give the LLM a concrete chance to fix it within the
same investigation, instead of ending the turn on a mistake the model
could plausibly correct — reproduces a live finding: a real model omitted
a required target field, then self-corrected immediately once the exact
Gateway rejection was fed back as an observation on the next turn."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager, ConversationState, InvestigationState, PendingApproval
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import ProposeToolCall
from numi.common.models.tool import ToolCallResponse, ToolCallStatus


class _FakeToolClient:
    def __init__(self, response: ToolCallResponse):
        self.response = response
        self.submit_count = 0

    async def submit(self, request):
        self.submit_count += 1
        return self.response


def _orchestrator(tool_client) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(None),  # unused by _submit_and_relay directly
        tool_client=tool_client,
        context=None,  # unused by _submit_and_relay directly
    )


def _state_and_investigation():
    state = ConversationState(
        conversation_id="conv1", channel="dev", channel_thread_id="", channel_account_id="dba_l2@example.com"
    )
    investigation = InvestigationState(investigation_id="inv1", problem="refresh stats", turn_count=0)
    return state, investigation


def _action() -> ProposeToolCall:
    return ProposeToolCall(
        tool_id="database.update_statistics",
        arguments={},
        target={},
        reason="Refreshing stale statistics.",
    )


@pytest.mark.asyncio
async def test_a_self_correctable_denial_continues_the_investigation():
    response = ToolCallResponse(
        status=ToolCallStatus.DENIED,
        failure_code="INVALID_TARGET",
        message="Missing required target field(s) for this operation: schema_name, object_name.",
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    reply = await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert reply is None  # loop continues, no reply sent to the DBA yet
    assert investigation.transcript[-1]["result"]["failure_code"] == "INVALID_TARGET"
    assert "INVALID_TARGET" in investigation.evidence[-1]


@pytest.mark.asyncio
async def test_a_permissions_denial_still_ends_the_turn_immediately():
    """UNAUTHORIZED (or POLICY_DENIED, TOOL_NOT_AVAILABLE, ...) is a fact no
    retry with different arguments changes — must not burn the turn budget
    retrying something that can only ever fail the same way."""
    response = ToolCallResponse(
        status=ToolCallStatus.DENIED,
        failure_code="UNAUTHORIZED",
        message="Role DBA_L1 cannot call database.update_statistics.",
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    reply = await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l1@example.com")

    assert reply is not None
    assert reply.status == "denied"
    assert investigation.transcript == []


@pytest.mark.asyncio
async def test_a_self_correctable_denial_does_not_loop_past_the_turn_budget():
    """Once the turn cap is reached, even a self-correctable failure ends
    the turn immediately — no infinite retry loop."""
    response = ToolCallResponse(
        status=ToolCallStatus.DENIED, failure_code="INVALID_TARGET", message="Missing target fields."
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    investigation.turn_count = 6  # at the cap

    reply = await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert reply is not None
    assert reply.status == "denied"


@pytest.mark.asyncio
async def test_a_self_correctable_denial_past_the_turn_budget_never_shows_the_raw_detail():
    """Reproduces a live finding: a real DBA's entire reply was a raw
    Pydantic ValidationError dump (field names, "extra_forbidden", a
    pydantic.dev docs URL) — tool_call_handler.py's INVALID_ARGUMENTS
    message is deliberately that raw/technical, meant as self-correction
    feedback for the LLM, not for a human. It only ever reached the DBA
    because the turn budget ran out right on a self-correctable failure,
    skipping the self-correction retry that would normally consume it."""
    raw_pydantic_dump = (
        "[{'type': 'extra_forbidden', 'loc': ('database_name',), "
        "'msg': 'Extra inputs are not permitted', 'input': 'postgres', "
        "'url': 'https://errors.pydantic.dev/2.13/v/extra_forbidden'}]"
    )
    response = ToolCallResponse(
        status=ToolCallStatus.DENIED,
        failure_code="INVALID_ARGUMENTS",
        message=f"Invalid arguments for 'database.get_blocking_sessions': {raw_pydantic_dump}",
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    investigation.turn_count = 6  # at the cap

    reply = await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert reply is not None
    assert reply.status == "denied"
    assert "extra_forbidden" not in reply.text
    assert "pydantic.dev" not in reply.text
    assert "ran out of attempts" in reply.text
    assert "database.update_statistics" in reply.text  # still names which tool, just not the raw detail


@pytest.mark.asyncio
async def test_a_failed_status_is_logged_as_a_failed_step_not_mislabeled_as_executed():
    """FAILED (an adapter-level failure — spec §8's execution layer, not a
    Gateway policy decision) was previously unhandled here and fell through
    to the EXECUTED branch, logging a failure as if it had succeeded. Playbooks
    make this more likely to surface (they proactively call diagnostics a
    given engine/topology may not implement) — must be recorded plainly and
    the investigation must still be able to continue."""
    response = ToolCallResponse(status=ToolCallStatus.FAILED, message="adapter connection timeout")
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    reply = await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert reply is None  # the loop continues — a single failed diagnostic doesn't end the investigation
    assert investigation.transcript[-1]["result"]["error"] == "adapter connection timeout"
    assert "failed" in investigation.evidence[-1].lower()
    assert "adapter connection timeout" in investigation.evidence[-1]


@pytest.mark.asyncio
async def test_a_write_tools_actual_outcome_is_surfaced_not_just_a_generic_completed():
    """Reproduces a live finding: the Gateway's `message` on EXECUTED is a
    hardcoded "Completed." regardless of what actually happened — a real
    kill_session call against a session that was already gone still says
    "Completed.", with the real answer (`terminated: False`) sitting
    unseen in `result.affected`. A DBA reading only the evidence line had
    no way to tell a kill that actually terminated something from one that
    found nothing there. Read tools never populate `affected` (they use
    `rows`/`row_count` instead), so this must be a no-op for them."""
    response = ToolCallResponse(
        status=ToolCallStatus.EXECUTED,
        message="Completed.",
        result={"affected": {"terminated": False, "session_id": "13400"}},
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert investigation.evidence[-1] == (
        "database.update_statistics: Completed. (terminated=False, session_id=13400)"
    )


@pytest.mark.asyncio
async def test_a_read_tools_evidence_line_is_unchanged_by_the_affected_summary():
    response = ToolCallResponse(
        status=ToolCallStatus.EXECUTED,
        message="Completed.",
        result={"rows": [{"a": 1}], "row_count": 1},
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert investigation.evidence[-1] == "database.update_statistics: Completed."


class _TimingOutToolClient:
    """Reproduces the live finding: the Agent's own HTTP call to the Gateway
    can time out (httpcore.ReadTimeout) before any ToolCallResponse ever
    comes back — a genuinely overloaded database made even a trivial
    read-only diagnostic exceed ToolClient's own bound. Distinct from
    ToolCallStatus.FAILED, which means a response DID come back."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.submit_count = 0

    async def submit(self, request):
        self.submit_count += 1
        raise self._exc


@pytest.mark.asyncio
async def test_a_submit_timeout_degrades_to_a_clear_message_not_an_unhandled_500():
    """Previously nothing caught this — it propagated as an unhandled
    exception all the way out of the FastAPI handler (a raw 500), exactly
    like the pre-fix LLM call path used to. Must degrade the same way the
    LLM path does: a clear message, the turn ends, the investigation stays
    resumable."""
    client = _TimingOutToolClient(TimeoutError("timed out"))
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    reply = await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert reply is not None
    assert reply.status == "error"
    assert "database.update_statistics" in reply.text
    assert "did not respond in time" in investigation.evidence[-1]
    assert investigation.transcript[-1]["result"]["error"] == "timed out"


@pytest.mark.asyncio
async def test_a_submit_timeout_does_not_burn_the_rest_of_the_turn_budget():
    """A network-level failure ends the turn immediately (unlike a
    self-correctable DENIED) — retrying more steps against a target that
    just failed to respond at all is unlikely to do anything but compound
    the latency, which is exactly what the production-speed work elsewhere
    in this codebase exists to prevent."""
    client = _TimingOutToolClient(TimeoutError("timed out"))
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(state, investigation, _action(), "dev", "dba_l2@example.com")

    assert client.submit_count == 1  # no retry loop hidden in here


@pytest.mark.asyncio
async def test_an_approved_actions_resubmit_timeout_tells_the_dba_what_to_check():
    """The trickier half of the same gap: by the time the resubmit call
    times out, the Gateway has *already* recorded the approval — silently
    500ing here would leave the DBA not knowing whether their approved
    action actually ran. Must point at the audit trail instead."""

    class _ApprovesThenTimesOut:
        async def approve(self, approval_id, channel, channel_account_id):
            return {"status": "APPROVED"}

        async def submit(self, request):
            raise TimeoutError("timed out")

    context = ContextManager()
    state = context.get_or_create("conv1", "dev", "", "dba_l2@example.com")
    state.pending_approval = PendingApproval(
        approval_id="appr1", tool_id="database.kill_session", summary="Kill it.",
        request={"tool_id": "database.kill_session", "arguments": {}, "target": {}, "reason": "x",
                 "conversation_id": "conv1", "request_id": "req1", "channel": "dev",
                 "channel_account_id": "dba_l2@example.com"},
    )
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(None), tool_client=_ApprovesThenTimesOut(), context=context
    )

    reply = await orchestrator.handle_approval_decision(
        conversation_id="conv1", decision="approve", channel="dev", channel_account_id="dba_l2@example.com"
    )

    assert reply.status == "error"
    assert "appr1" in reply.text
    assert "audit trail" in reply.text.lower()
    assert state.pending_approval is None  # cleared — re-approving isn't meaningful either way
