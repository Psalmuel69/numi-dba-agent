"""End-to-end acceptance scenario (spec §68).

A verified DBA_L2 asks about CoreBanking production slowness in Teams; the
Agent (Mock LLM) investigates via the real Gateway pipeline and the mock
Execution Service, proposes killing the head blocker, the Gateway requires
approval, the DBA approves, and the action executes and is reported back —
with every hop actually going through HTTP + the real Gateway pipeline
(policy/risk/approval/execution), just wired via in-process ASGI transports
instead of real sockets.
"""

from __future__ import annotations

import httpx

from numi.agent.context_manager import ContextManager, ConversationState, InvestigationState
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, CritiqueVerdict
from numi.agent.tool_client import ToolClient
from numi.common.config import Settings
from numi.common.service_auth import ServiceTokenIssuer
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        control_db_url="sqlite+aiosqlite:///:memory:",
        service_jwt_secret="test-secret",
        service_jwt_issuer="numi-internal",
        llm_provider="mock",
    )


async def _build_orchestrator() -> AgentOrchestrator:
    settings = _settings()
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    execution_transport = httpx.ASGITransport(app=execution_app)
    gateway_app = create_gateway_app(settings, execution_transport=execution_transport)
    # httpx.ASGITransport doesn't emit lifespan events, so create the
    # in-memory sqlite schema explicitly (a real server run — or the
    # TestClient-based tests in test_gateway_api.py — trigger this via
    # the app's lifespan instead).
    await gateway_app.state.gateway.db.create_all()
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    tool_client = ToolClient(
        settings.gateway_base_url, issuer, transport=gateway_transport
    )
    return AgentOrchestrator(
        LLMRegistry.for_testing(MockLLMProvider()), tool_client, ContextManager()
    )


async def test_acceptance_scenario_investigate_approve_execute_verify():
    orchestrator = await _build_orchestrator()

    first = await orchestrator.handle_message(
        channel="teams",
        channel_account_id="aad-mock-l2",
        conversation_id="conv_accept_1",
        channel_thread_id="thread_1",
        message="CoreBanking production is slow. Investigate and tell me what is wrong.",
    )

    assert first.status == "approval_required"
    assert first.approval_card is not None
    assert first.approval_card.tool_id == "database.kill_session"

    second = await orchestrator.handle_approval_decision(
        conversation_id="conv_accept_1",
        decision="approve",
        channel="teams",
        channel_account_id="aad-mock-l2",
    )

    assert second.status == "ok"
    assert "Action approved" in second.text
    assert "MITIGATED" in second.text


async def test_non_dba_task_message_asks_for_clarification():
    orchestrator = await _build_orchestrator()
    reply = await orchestrator.handle_message(
        channel="teams",
        channel_account_id="aad-mock-l2",
        conversation_id="conv_accept_2",
        channel_thread_id="",
        message="What's for lunch?",
    )
    assert reply.status == "clarification"


async def test_greeting_returns_help_text_without_starting_investigation():
    orchestrator = await _build_orchestrator()
    reply = await orchestrator.handle_message(
        channel="teams",
        channel_account_id="aad-mock-l2",
        conversation_id="conv_accept_3",
        channel_thread_id="",
        message="hello",
    )
    assert "Numi" in reply.text


async def test_l1_user_gets_denied_response_not_a_crash():
    orchestrator = await _build_orchestrator()
    reply = await orchestrator.handle_message(
        channel="teams",
        channel_account_id="aad-mock-l1",
        conversation_id="conv_accept_4",
        channel_thread_id="",
        message="CoreBanking production is slow, kill session 9182",
    )
    # DBA_L1 cannot even reach kill_session on this critical prod database.
    assert reply.status == "denied"
    assert "can't do that" in reply.text


class _FakeLLM:
    """Only `decide_next_action` is exercised once comprehensive_summary's 5
    fixed steps are exhausted — stubbed for determinism (matches the
    pattern in tests/unit/test_orchestrator_playbooks.py) so this test
    verifies the real Gateway's target-validation/tool-catalog pipeline,
    not a real model's own heuristics."""

    provider_name = "fake"
    model = "fake-model"

    def __init__(self, actions: list):
        self._actions = list(actions)
        self.calls: list[dict] = []

    async def decide_next_action(self, **kwargs):
        self.calls.append(kwargs)
        return self._actions.pop(0)

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


async def test_comprehensive_summary_runs_exactly_five_tool_calls_through_the_real_gateway():
    """Reproduces (and pins the fix for) a live finding: comprehensive_
    summary's `database.get_storage` step runs with `target={}` (no
    database named) — before get_storage joined the instance-wide tool set
    in tool_catalog.py, the real Gateway's target validation rejected that
    with INVALID_TARGET, triggering a self-correction retry and burning a
    6th tool call within the playbook's 5-step, 6-turn shared budget. Runs
    through the REAL Gateway pipeline (target validation, tool catalog,
    execution) exactly like the acceptance scenario above — only the final
    concluding LLM call is stubbed, since a real model's own heuristics
    aren't what's under test here."""
    orchestrator = await _build_orchestrator()
    state = ConversationState(
        conversation_id="conv_cs_1", channel="teams", channel_thread_id="", channel_account_id="aad-mock-l2"
    )
    state.database_context["environment"] = "development"
    state.database_context["instance"] = "sqlserver-dev-01"
    investigation = InvestigationState(
        investigation_id="inv_cs_1", problem="full health report", playbook_id="comprehensive_summary"
    )
    llm = _FakeLLM(actions=[Conclude(summary="All other checks came back clean.")])

    reply = await orchestrator._run_investigation_loop(
        state,
        investigation,
        [
            "database.get_health", "database.get_blocking_sessions", "database.get_backup_status",
            "database.get_storage", "database.get_error_logs",
        ],
        "teams",
        "aad-mock-l2",
        llm,
        None,
    )

    assert reply.status == "ok"
    assert len(llm.calls) == 1  # exactly one turn left free for the LLM's own conclude call
    tool_ids = [t["tool_id"] for t in investigation.transcript]
    assert tool_ids == [
        "database.get_health",
        "database.get_blocking_sessions",
        "database.get_backup_status",
        "database.get_storage",
        "database.get_error_logs",
    ]
    # Every step, including get_storage, executed cleanly on the first
    # attempt — no self-correction retry inflating this to 6 tool calls.
    for entry in investigation.transcript:
        assert entry["result"].get("failure_code") is None, entry
