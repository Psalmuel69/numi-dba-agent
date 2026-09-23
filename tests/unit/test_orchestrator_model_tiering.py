"""`AgentOrchestrator._llm_for`'s task-complexity tiering: proactively
routes `call_type="fast"`/`"strong"` to `Settings.llm_fast_model`/
`llm_strong_model`, but only when the DBA hasn't made an explicit `/model`
choice, and never disabled by a deployment-level provider lock. A single
configured provider is never wrapped by `CrossProviderFallbackLLM` (see
that class's own docstring), so `llm.model` here is always the bare
resolved provider's own attribute — no fallback machinery to see through.
"""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager, ConversationState
from numi.agent.llm.openai_provider import _OPENAI_DEFAULT_MODEL
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, CritiqueVerdict, IntentExtraction
from numi.common.config import Settings


def _orchestrator(**settings_overrides) -> AgentOrchestrator:
    settings = Settings(_env_file=None, openai_api_key="sk-test", **settings_overrides)
    return AgentOrchestrator(
        llm_registry=LLMRegistry(settings), tool_client=None, context=None
    )


def _state(*, provider: str | None = None, model: str | None = None) -> ConversationState:
    state = ConversationState(
        conversation_id="c1", channel="dev", channel_thread_id="", channel_account_id="a"
    )
    state.llm_provider = provider
    state.llm_model = model
    return state


def test_empty_tier_settings_is_byte_identical_to_before():
    orchestrator = _orchestrator()
    llm = orchestrator._llm_for(_state(), call_type="fast")
    assert llm.model == _OPENAI_DEFAULT_MODEL


def test_configured_tiers_apply_per_call_type_with_no_explicit_model():
    orchestrator = _orchestrator(llm_fast_model="gpt-4o-mini", llm_strong_model="o1")
    fast_llm = orchestrator._llm_for(_state(), call_type="fast")
    strong_llm = orchestrator._llm_for(_state(), call_type="strong")
    assert fast_llm.model == "gpt-4o-mini"
    assert strong_llm.model == "o1"


def test_an_explicit_model_choice_bypasses_tiering_entirely():
    """The one invariant that matters most: a DBA's `/model` choice is
    never silently overridden by a tier default, regardless of call_type."""
    orchestrator = _orchestrator(llm_fast_model="gpt-4o-mini", llm_strong_model="o1")
    state = _state(provider="openai", model="gpt-3.5-turbo")
    fast_llm = orchestrator._llm_for(state, call_type="fast")
    strong_llm = orchestrator._llm_for(state, call_type="strong")
    assert fast_llm.model == "gpt-3.5-turbo"
    assert strong_llm.model == "gpt-3.5-turbo"


def test_a_provider_locked_deployment_still_tiers_when_no_explicit_model():
    """`Settings.llm_provider` forcing one vendor only constrains which
    vendor is used — `state.llm_provider`/`llm_model` stay `None` in that
    case, so tiering still applies within the locked vendor."""
    orchestrator = _orchestrator(llm_provider="openai", llm_fast_model="gpt-4o-mini")
    llm = orchestrator._llm_for(_state(), call_type="fast")
    assert llm.model == "gpt-4o-mini"


def test_default_call_type_is_strong():
    orchestrator = _orchestrator(llm_strong_model="o1")
    llm = orchestrator._llm_for(_state())
    assert llm.model == "o1"


# --- call-site spy: which call_type each entry point actually requests -----


class _SpyFakeToolClient:
    async def list_servers(self):
        return []

    async def available_tools(self, channel, channel_account_id):
        return []

    async def create_investigation(self, request):
        pass

    async def update_investigation(self, investigation_id, request):
        pass

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        return []

    async def log_decision_event(self, request):
        pass


class _SpyFakeLLM:
    provider_name = "fake"
    model = "fake-model"

    def __init__(self):
        self._intent = IntentExtraction(
            is_dba_task=True, environment_hint="development", problem_summary="check health"
        )
        self._conclude = Conclude(summary="Nothing wrong found.")

    async def extract_intent(self, *args, **kwargs):
        return self._intent

    async def decide_next_action(self, **kwargs):
        return self._conclude

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


@pytest.mark.asyncio
async def test_extract_intent_requests_the_fast_tier_and_decide_next_action_requests_strong():
    llm = _SpyFakeLLM()
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm),
        tool_client=_SpyFakeToolClient(),
        context=ContextManager(),
    )
    seen_call_types: list[str] = []
    original = orchestrator._llm_for

    def spy(state, call_type="strong"):
        seen_call_types.append(call_type)
        return original(state, call_type)

    orchestrator._llm_for = spy

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U1",
        conversation_id="conv1",
        channel_thread_id="",
        message="queries are slow on postgres-local",
    )

    assert "fast" in seen_call_types  # the extract_intent call
    assert "strong" in seen_call_types  # the investigation loop's decide_next_action
