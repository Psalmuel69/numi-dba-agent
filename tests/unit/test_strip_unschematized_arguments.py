"""Defense-in-depth backstop for the same live finding covered end-to-end
in tests/integration/test_no_args_extra_arguments.py: even with the prompt
telling the model each tool's real argument schema, a real model can still
fold an extra key into `arguments` that tool's own schema forbids (most
often `reason`, confused with `ProposeToolCall`'s own top-level `reason`
field, or a `session_id`/`database_name` it decided was relevant). The
Gateway's argument models are all `extra="forbid"`, so an unstripped extra
key is rejected outright as INVALID_ARGUMENTS — self-correctable, but only
at the cost of a wasted turn and Gateway round-trip every time.

`AgentOrchestrator._strip_unschematized_arguments` (used by
`_submit_and_relay` before a `ToolCallRequest` is ever built) is unit-tested
directly here, independent of the full Gateway pipeline."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ConversationState, InvestigationState
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import ProposeToolCall
from numi.common.models.tool import ToolCallResponse, ToolCallStatus


class _CapturingToolClient:
    """Records the exact ToolCallRequest submitted, so a test can assert on
    what actually left the orchestrator toward the Gateway."""

    def __init__(self, response: ToolCallResponse):
        self.response = response
        self.last_request = None

    async def submit(self, request):
        self.last_request = request
        return self.response


def _orchestrator(tool_client) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(None), tool_client=tool_client, context=None
    )


def _state_and_investigation():
    state = ConversationState(
        conversation_id="conv1", channel="dev", channel_thread_id="", channel_account_id="dba_l2@example.com"
    )
    investigation = InvestigationState(investigation_id="inv1", problem="check blocking", turn_count=0)
    return state, investigation


@pytest.mark.asyncio
async def test_a_no_args_tools_extra_reason_and_session_id_are_stripped_before_submitting():
    """Reproduces the exact live finding: database.get_blocking_sessions
    (NoArgs) proposed with `reason` and `session_id` folded into
    `arguments`. Given the tool's real allowed-keys set (empty), both must
    be dropped before the request is built — the Gateway must never see
    them."""
    response = ToolCallResponse(status=ToolCallStatus.EXECUTED, message="Completed.", result={"rows": []})
    client = _CapturingToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    action = ProposeToolCall(
        tool_id="database.get_blocking_sessions",
        reason="Checking for blocking chains given the reported slowness.",
        arguments={"reason": "Checking for blocking chains given the reported slowness.", "session_id": "42"},
    )

    await orchestrator._submit_and_relay(
        state,
        investigation,
        action,
        "dev",
        "dba_l2@example.com",
        tool_allowed_arguments={"database.get_blocking_sessions": set()},
    )

    assert client.last_request.arguments == {}


@pytest.mark.asyncio
async def test_a_tools_own_required_keys_survive_stripping():
    """The backstop must never drop a key the tool's own schema actually
    declares — only ones outside it."""
    response = ToolCallResponse(status=ToolCallStatus.EXECUTED, message="Completed.", result={"affected": {}})
    client = _CapturingToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    action = ProposeToolCall(
        tool_id="database.update_statistics",
        reason="Refreshing stale statistics.",
        arguments={
            "schema": "Person",
            "table": "Person",
            "reason": "Refreshing stale statistics.",
            "database_name": "AdventureWorks2019",  # not part of UpdateStatisticsArgs
        },
    )

    await orchestrator._submit_and_relay(
        state,
        investigation,
        action,
        "dev",
        "dba_l2@example.com",
        tool_allowed_arguments={"database.update_statistics": {"schema", "table", "reason"}},
    )

    assert client.last_request.arguments == {
        "schema": "Person",
        "table": "Person",
        "reason": "Refreshing stale statistics.",
    }


@pytest.mark.asyncio
async def test_no_schema_map_supplied_skips_stripping_entirely():
    """Backward-compatible default: omitting `tool_allowed_arguments`
    (as every existing caller in test_orchestrator_self_correction.py does)
    must leave `arguments` completely untouched, never guess."""
    response = ToolCallResponse(status=ToolCallStatus.EXECUTED, message="Completed.", result={"rows": []})
    client = _CapturingToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    action = ProposeToolCall(
        tool_id="database.get_blocking_sessions",
        reason="Checking for blocking chains.",
        arguments={"reason": "Checking for blocking chains.", "database_name": "CoreBanking"},
    )

    await orchestrator._submit_and_relay(state, investigation, action, "dev", "dba_l2@example.com")

    assert client.last_request.arguments == {
        "reason": "Checking for blocking chains.",
        "database_name": "CoreBanking",
    }


@pytest.mark.asyncio
async def test_a_tool_id_absent_from_the_map_is_left_untouched():
    """A tool_id the map doesn't mention at all (e.g. it wasn't in the
    available-tools list this turn) is a "we don't know" case, not a "this
    tool needs nothing" case — must not be treated the same as an empty
    set."""
    response = ToolCallResponse(status=ToolCallStatus.EXECUTED, message="Completed.", result={"rows": []})
    client = _CapturingToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    action = ProposeToolCall(
        tool_id="database.get_sessions",
        reason="Checking active sessions.",
        arguments={"reason": "Checking active sessions."},
    )

    await orchestrator._submit_and_relay(
        state,
        investigation,
        action,
        "dev",
        "dba_l2@example.com",
        tool_allowed_arguments={"database.update_statistics": {"schema", "table", "reason"}},
    )

    assert client.last_request.arguments == {"reason": "Checking active sessions."}
