"""Reproduces a recurring live complaint: "on postgres-local" (naming the
server but no database) got "database.get_health was rejected
(INVALID_TARGET): Which database on postgres-local?" over and over, every
single time — even after the model self-corrected within one
investigation, the very next message started from scratch again.
Environment and instance already persist in state.database_context across
messages; database did not. Once a database is actually resolved — the DBA
naming it, or the model self-correcting a rejection within the same
investigation — it's now remembered for the rest of the conversation."""

from __future__ import annotations

import pytest

from numi.agent.planner.actions import ProposeToolCall
from numi.common.models.tool import ToolCallResponse, ToolCallStatus
from tests.unit.test_orchestrator_playbooks import _orchestrator, _state_and_investigation


class _FakeToolClient:
    def __init__(self, response: ToolCallResponse):
        self.response = response
        self.requests: list = []

    async def submit(self, request):
        self.requests.append(request)
        return self.response


def _action(database: str | None = None) -> ProposeToolCall:
    return ProposeToolCall(
        tool_id="database.get_health",
        arguments={},
        target={"database": database} if database else {},
        reason="Checking health.",
    )


@pytest.mark.asyncio
async def test_a_successful_calls_database_is_remembered():
    response = ToolCallResponse(status=ToolCallStatus.EXECUTED, message="ok", result={})
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(
        state, investigation, _action("postgres"), "dev", "dba_l2@example.com"
    )

    assert state.database_context["database"] == "postgres"


@pytest.mark.asyncio
async def test_an_invalid_target_denial_never_remembers_the_database():
    """The exact rejection this fix must never treat as a resolved
    answer — "Which database on X?" means the opposite of resolved."""
    response = ToolCallResponse(
        status=ToolCallStatus.DENIED,
        failure_code="INVALID_TARGET",
        message="Which database on postgres-local? Discovered: AdventureWorks2019, postgres",
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(
        state, investigation, _action("postgres"), "dev", "dba_l2@example.com"
    )

    assert "database" not in state.database_context


@pytest.mark.asyncio
async def test_an_approval_required_response_still_remembers_the_database():
    """Approval-required means target validation already passed — the
    Gateway's pipeline resolves the target before policy/risk/approval."""
    response = ToolCallResponse(
        status=ToolCallStatus.APPROVAL_REQUIRED, approval_id="appr1", message="needs approval"
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(
        state, investigation, _action("postgres"), "dev", "dba_l2@example.com"
    )

    assert state.database_context["database"] == "postgres"


@pytest.mark.asyncio
async def test_a_non_target_denial_still_remembers_the_database():
    """UNAUTHORIZED (or any policy/rate-limit denial) is a fact about the
    role or the call, not the target — the database name was still valid."""
    response = ToolCallResponse(
        status=ToolCallStatus.DENIED, failure_code="UNAUTHORIZED", message="Role cannot call this."
    )
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()

    await orchestrator._submit_and_relay(
        state, investigation, _action("postgres"), "dev", "dba_l2@example.com"
    )

    assert state.database_context["database"] == "postgres"


@pytest.mark.asyncio
async def test_a_call_with_no_database_in_its_target_does_not_clear_a_remembered_one():
    response = ToolCallResponse(status=ToolCallStatus.EXECUTED, message="ok", result={})
    client = _FakeToolClient(response)
    orchestrator = _orchestrator(client)
    state, investigation = _state_and_investigation()
    state.database_context["database"] = "postgres"

    await orchestrator._submit_and_relay(state, investigation, _action(None), "dev", "dba_l2@example.com")

    assert state.database_context["database"] == "postgres"
