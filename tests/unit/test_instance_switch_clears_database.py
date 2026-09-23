"""A remembered database (see test_database_persistence.py) belongs to
whatever instance was in play when it was resolved — moving to a
different, explicitly-named instance makes it stale and potentially wrong
(a different server may not even have a database by that name), so it
must not silently carry over."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, CritiqueVerdict, IntentExtraction


class _FakeToolClient:
    async def list_servers(self):
        return []

    async def available_tools(self, channel, channel_account_id):
        return []

    async def submit(self, request):
        raise AssertionError("no tool call expected")

    async def create_investigation(self, request):
        pass

    async def update_investigation(self, investigation_id, request):
        pass

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        return []


class _FakeLLM:
    def __init__(self, intent: IntentExtraction):
        self._intent = intent

    async def extract_intent(self, *args, **kwargs):
        return self._intent

    async def decide_next_action(self, **kwargs):
        return Conclude(summary="Nothing to check.")

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


def _orchestrator(llm) -> tuple[AgentOrchestrator, ContextManager]:
    context = ContextManager()
    return (
        AgentOrchestrator(
            llm_registry=LLMRegistry.for_testing(llm), tool_client=_FakeToolClient(), context=context
        ),
        context,
    )


@pytest.mark.asyncio
async def test_switching_to_a_different_named_instance_clears_the_remembered_database():
    intent = IntentExtraction(
        is_dba_task=True,
        instance_hint="sqlserver-dev-01",
        environment_hint="development",
        problem_summary="check health on sqlserver-dev-01",
    )
    orchestrator, context = _orchestrator(_FakeLLM(intent))
    state = context.get_or_create("conv1", "slack", "", "U123")
    state.database_context["instance"] = "postgres-local"
    state.database_context["database"] = "postgres"
    state.database_context["environment"] = "development"

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv1",
        channel_thread_id="",
        message="check health on sqlserver-dev-01",
    )

    assert state.database_context["instance"] == "sqlserver-dev-01"
    assert "database" not in state.database_context


@pytest.mark.asyncio
async def test_the_same_instance_named_again_keeps_the_remembered_database():
    intent = IntentExtraction(
        is_dba_task=True,
        instance_hint="postgres-local",
        environment_hint="development",
        problem_summary="check health on postgres-local",
    )
    orchestrator, context = _orchestrator(_FakeLLM(intent))
    state = context.get_or_create("conv2", "slack", "", "U123")
    state.database_context["instance"] = "postgres-local"
    state.database_context["database"] = "postgres"
    state.database_context["environment"] = "development"

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv2",
        channel_thread_id="",
        message="check health on postgres-local",
    )

    assert state.database_context["database"] == "postgres"


@pytest.mark.asyncio
async def test_naming_a_new_database_explicitly_overrides_the_remembered_one():
    intent = IntentExtraction(
        is_dba_task=True,
        instance_hint="postgres-local",
        environment_hint="development",
        database_hint="AdventureWorks2019",
        problem_summary="check health of AdventureWorks2019 on postgres-local",
    )
    orchestrator, context = _orchestrator(_FakeLLM(intent))
    state = context.get_or_create("conv3", "slack", "", "U123")
    state.database_context["instance"] = "postgres-local"
    state.database_context["database"] = "postgres"
    state.database_context["environment"] = "development"

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv3",
        channel_thread_id="",
        message="check health of AdventureWorks2019 on postgres-local",
    )

    assert state.database_context["database"] == "AdventureWorks2019"
