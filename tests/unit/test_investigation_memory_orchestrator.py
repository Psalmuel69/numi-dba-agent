"""Orchestrator-side wiring of investigation memory (`_bootstrap_
investigation_memory`, `_problem_statement_for_llm`'s memory section): a
one-time, best-effort Gateway round trip at the start of every investigation
that must never break a turn, never run twice, and must keep recalled
findings out of `investigation.evidence`."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ConversationState, InvestigationState
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude
from numi.common.models.investigation import InvestigationMemoryEntry
from numi.common.models.risk import RiskLevel
from numi.common.models.tool import OperationType, ToolDefinition
from tests.unit.test_orchestrator_playbooks import _FakeLLM


def _tool_def(tool_id: str, operation_type=OperationType.READ) -> ToolDefinition:
    return ToolDefinition(
        tool_id=tool_id,
        version="1",
        description="d",
        operation_type=operation_type,
        risk_level=RiskLevel.LOW,
        reversible=True,
        availability_impact=False,
        data_modification=False,
        requires_approval=False,
        allowed_roles=["DBA_L1"],
        allowed_environments=["development"],
        required_target_scope=["server"],
        argument_schema={},
        result_schema={},
    )


class _FakeToolClient:
    def __init__(
        self,
        *,
        memory_entries: list[InvestigationMemoryEntry] | None = None,
        cross_server_patterns: list[InvestigationMemoryEntry] | None = None,
    ):
        self.create_calls: list = []
        self.update_calls: list = []
        self.memory_calls: list = []
        self.correlate_calls: list = []
        self._memory_entries = memory_entries or []
        self._cross_server_patterns = cross_server_patterns or []

    async def available_tools(self, channel, channel_account_id):
        return [_tool_def("database.get_health")]

    async def create_investigation(self, request):
        self.create_calls.append(request)

    async def update_investigation(self, investigation_id, request):
        self.update_calls.append((investigation_id, request))

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        self.memory_calls.append((server_id, exclude_investigation_id, limit))
        return self._memory_entries

    async def get_cross_server_patterns(
        self, *, playbook_id, environment=None, exclude_server_id=None, limit=5
    ):
        self.correlate_calls.append((playbook_id, environment, exclude_server_id, limit))
        return self._cross_server_patterns


def _orchestrator(tool_client, llm) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm), tool_client=tool_client, context=None
    )


def _state_and_investigation(*, instance: str | None = "postgres-dev-01"):
    state = ConversationState(
        conversation_id="conv1", channel="dev", channel_thread_id="", channel_account_id="dba_l2@example.com"
    )
    state.database_context["environment"] = "development"
    if instance:
        state.database_context["instance"] = instance
    investigation = InvestigationState(investigation_id="inv1", problem="queries are slow")
    return state, investigation


@pytest.mark.asyncio
async def test_bootstrap_creates_the_investigation_and_recalls_memory():
    entry = InvestigationMemoryEntry(
        investigation_id="inv0",
        problem="high CPU last week",
        status="CONCLUDED_VERIFIED",
        findings=["runaway autovacuum"],
        recommendations=["tune autovacuum settings"],
        updated_at="2026-03-01T00:00:00Z",
    )
    tool_client = _FakeToolClient(memory_entries=[entry])
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)
    state, investigation = _state_and_investigation()

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert len(tool_client.create_calls) == 1
    assert tool_client.create_calls[0].server_id == "postgres-dev-01"
    assert tool_client.memory_calls == [("postgres-dev-01", "inv1", 3)]
    assert investigation.memory_context[0]["investigation_id"] == "inv0"
    assert investigation.remote_bootstrap_done is True


@pytest.mark.asyncio
async def test_bootstrap_runs_exactly_once_across_multiple_resumed_turns():
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="first"), Conclude(summary="second")])
    orchestrator = _orchestrator(tool_client, llm)
    state, investigation = _state_and_investigation()

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")
    investigation.status = "INVESTIGATING"  # simulate a resumed, still-open investigation
    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert len(tool_client.create_calls) == 1
    assert len(tool_client.memory_calls) == 1


@pytest.mark.asyncio
async def test_no_server_named_yet_skips_memory_recall_but_still_creates():
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)
    state, investigation = _state_and_investigation(instance=None)

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert len(tool_client.create_calls) == 1
    assert tool_client.create_calls[0].server_id is None
    assert tool_client.memory_calls == []


@pytest.mark.asyncio
async def test_an_unreachable_gateway_never_breaks_the_turn():
    """The real `ToolClient` (not the duck-typed fake above) is what
    actually guarantees create/update/memory calls are best-effort — this
    exercises that real contract end-to-end: a Gateway the transport can't
    reach at all must still let the investigation conclude normally."""
    import httpx

    from numi.agent.tool_client import ToolClient
    from numi.common.service_auth import ServiceTokenIssuer

    class _FailingTools(ToolClient):
        async def available_tools(self, channel, channel_account_id):
            return [_tool_def("database.get_health")]

    def _unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    tool_client = _FailingTools(
        "http://gateway",
        ServiceTokenIssuer("secret", "numi-internal"),
        transport=httpx.MockTransport(_unreachable),
    )
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)
    state, investigation = _state_and_investigation()

    reply = await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert reply.status == "ok"
    assert investigation.memory_context == []


def test_memory_context_is_never_folded_into_evidence():
    """The one invariant that matters most here: memory is prompt
    background, never something `_ungrounded_identifiers` could mistake for
    this investigation's own confirmed evidence."""
    investigation = InvestigationState(investigation_id="inv1", problem="p")
    investigation.memory_context = [
        {
            "investigation_id": "inv0",
            "problem": "old problem",
            "status": "CONCLUDED_VERIFIED",
            "findings": ["OldFinding"],
            "recommendations": [],
            "updated_at": "2026-03-01T00:00:00Z",
        }
    ]
    assert investigation.evidence == []


def test_problem_statement_includes_memory_context_labeled_as_background():
    investigation = InvestigationState(investigation_id="inv1", problem="queries are slow")
    investigation.memory_context = [
        {
            "investigation_id": "inv0",
            "problem": "high CPU last week",
            "status": "CONCLUDED_VERIFIED",
            "findings": ["runaway autovacuum"],
            "recommendations": ["tune autovacuum settings"],
            "updated_at": "2026-03-01T00:00:00Z",
        }
    ]

    problem = AgentOrchestrator._problem_statement_for_llm(investigation)

    assert "background only, not verified findings for THIS investigation" in problem
    assert "runaway autovacuum" in problem


@pytest.mark.asyncio
async def test_a_matched_playbook_fetches_and_folds_in_cross_server_patterns():
    pattern = InvestigationMemoryEntry(
        investigation_id="inv_other",
        server_id="sqlserver-dev-02",
        problem="slow queries last month",
        status="CONCLUDED_VERIFIED",
        findings=["missing index"],
        recommendations=["add index"],
        updated_at="2026-03-01T00:00:00Z",
    )
    tool_client = _FakeToolClient(cross_server_patterns=[pattern])
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)
    state, investigation = _state_and_investigation()
    investigation.playbook_id = "slow_queries"

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert tool_client.correlate_calls == [("slow_queries", "development", "postgres-dev-01", 5)]
    server_ids = [m["server_id"] for m in investigation.memory_context]
    assert "sqlserver-dev-02" in server_ids


@pytest.mark.asyncio
async def test_no_playbook_match_skips_cross_server_correlation_entirely():
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)
    state, investigation = _state_and_investigation()
    assert investigation.playbook_id is None

    await orchestrator._continue_investigation(state, investigation, "dev", "dba_l2@example.com")

    assert tool_client.correlate_calls == []
