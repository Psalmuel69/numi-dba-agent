"""`log_decision_event` fires at exactly the right moments: each of
`_finalize_conclude`'s three rejection paths, and once per drained
cross-provider fallback notice at the end of `handle_message`. See
`gateway.domain.decision_events`'s own docstring for the deliberate scope
cut (why these events and not others)."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, CritiqueVerdict
from tests.unit.test_cross_provider_fallback import _exhausted, _stubbed_registry, _StubProvider
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _orchestrator,
    _state_and_investigation,
)


class _SpyToolClient:
    def __init__(self):
        self.events: list = []

    async def available_tools(self, channel, channel_account_id):
        return []

    async def list_servers(self):
        return []

    async def log_decision_event(self, request):
        self.events.append(request)


class _FakeLLM:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self, actions: list, critiques: list[CritiqueVerdict] | None = None):
        self._actions = list(actions)
        self._critiques = list(critiques) if critiques is not None else [CritiqueVerdict(sound=True)] * 10

    async def decide_next_action(self, **kwargs):
        return self._actions.pop(0)

    async def critique_conclusion(self, **kwargs):
        return self._critiques.pop(0)


@pytest.mark.asyncio
async def test_ungrounded_rejection_logs_a_decision_event():
    state, investigation = _state_and_investigation(playbook_id=None)
    bad = Conclude(summary="Driven by AccountBalanceOutstandings.")
    good = Conclude(summary="Driven by heavy CPU load.")
    tool_client = _SpyToolClient()
    llm = _FakeLLM(actions=[bad, good])
    orchestrator = _orchestrator(tool_client)

    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    event_types = [e.event_type for e in tool_client.events]
    assert "conclusion_rejected_ungrounded_identifiers" in event_types


@pytest.mark.asyncio
async def test_self_critique_rejection_logs_a_decision_event():
    state, investigation = _state_and_investigation(playbook_id=None)
    bad = Conclude(summary="Driven by heavy CPU load.")
    good = Conclude(summary="Driven by heavy CPU load, confirmed.")
    tool_client = _SpyToolClient()
    llm = _FakeLLM(
        actions=[bad, good],
        critiques=[CritiqueVerdict(sound=False, issue="no evidence"), CritiqueVerdict(sound=True)],
    )
    orchestrator = _orchestrator(tool_client)

    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    matching = [e for e in tool_client.events if e.event_type == "conclusion_rejected_self_critique"]
    assert len(matching) == 1
    assert matching[0].payload == {"issue": "no evidence"}
    assert matching[0].investigation_id == investigation.investigation_id


@pytest.mark.asyncio
async def test_a_failed_critique_call_logs_its_own_decision_event():
    class _RaisingCritiqueLLM(_FakeLLM):
        async def critique_conclusion(self, **kwargs):
            raise RuntimeError("provider unavailable")

    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _SpyToolClient()
    llm = _RaisingCritiqueLLM(actions=[Conclude(summary="Driven by heavy CPU load.")])
    orchestrator = _orchestrator(tool_client)

    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    event_types = [e.event_type for e in tool_client.events]
    assert "self_critique_call_failed" in event_types


@pytest.mark.asyncio
async def test_a_cross_provider_fallback_substitution_logs_one_decision_event_per_message():
    """Uses the real `CrossProviderFallbackLLM`/`StructuredLLMProvider`
    plumbing via `_stubbed_registry` (not `LLMRegistry.for_testing`, whose
    `_fixed` shortcut bypasses `resilient_for_conversation` entirely and so
    never threads `state.llm_fallback_notices` into a fresh wrapper) —
    this is specifically about `handle_message`'s drain of that per-message
    sink, which only the real per-call construction populates."""
    stubs = {
        "gemini": _exhausted("gemini"),
        "anthropic": _StubProvider("anthropic", result={"is_dba_task": False, "meta_command": "help"}),
    }
    tool_client = _SpyToolClient()
    orchestrator = AgentOrchestrator(
        llm_registry=_stubbed_registry(stubs, llm_provider="gemini"),
        tool_client=tool_client,
        context=ContextManager(),
    )

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U1",
        conversation_id="conv1",
        channel_thread_id="",
        message="what can you help me with?",
    )

    matching = [e for e in tool_client.events if e.event_type == "llm_cross_provider_fallback_used"]
    assert len(matching) == 1
    assert matching[0].provider == "anthropic"
    assert matching[0].payload == {"failed_providers": ["gemini"]}
