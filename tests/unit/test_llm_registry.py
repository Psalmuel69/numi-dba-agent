"""LLM registry: which providers are selectable, default resolution, and the
per-conversation build path (spec §34, §36). No network — these never call a
real provider."""

from __future__ import annotations

import pytest

from numi.agent.llm.anthropic_provider import AnthropicLLMProvider
from numi.agent.llm.gemini_provider import GeminiLLMProvider
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.openai_provider import DeepSeekLLMProvider, OpenAILLMProvider
from numi.agent.llm.registry import LLMRegistry
from numi.common.config import Settings


def _registry(**overrides) -> LLMRegistry:
    return LLMRegistry(Settings(_env_file=None, **overrides))


def test_no_keys_falls_back_to_mock_planner():
    reg = _registry()
    assert reg.configured_providers() == []
    assert reg.default() == ("mock", "mock-planner")
    llm = reg.for_conversation(provider=None, model=None)
    assert isinstance(llm, MockLLMProvider)


def test_a_key_makes_that_provider_selectable_and_default():
    reg = _registry(openai_api_key="sk-openai")
    assert reg.configured_providers() == ["openai"]
    assert reg.default()[0] == "openai"
    assert isinstance(reg.build("openai", None), OpenAILLMProvider)


@pytest.mark.parametrize(
    "provider,cls",
    [
        ("anthropic", AnthropicLLMProvider),
        ("openai", OpenAILLMProvider),
        ("gemini", GeminiLLMProvider),
        ("deepseek", DeepSeekLLMProvider),
    ],
)
def test_build_returns_the_right_class_when_key_present(provider, cls):
    reg = _registry(**{f"{provider}_api_key": "test-key"})
    assert isinstance(reg.build(provider, None), cls)


def test_build_without_key_raises():
    reg = _registry(anthropic_api_key="a")
    with pytest.raises(ValueError):
        reg.build("openai", None)


def test_explicit_llm_provider_locks_selection():
    reg = _registry(anthropic_api_key="a", openai_api_key="b", llm_provider="anthropic")
    assert reg.selection_enabled() is False
    assert reg.validate_selection("openai", None) is not None  # can't switch


def test_selection_disabled_flag_blocks_switching():
    reg = _registry(anthropic_api_key="a", allow_user_model_selection=False)
    assert reg.selection_enabled() is False


def test_validate_selection_rejects_unknown_and_unconfigured_providers():
    reg = _registry(anthropic_api_key="a")
    assert "Unknown provider" in reg.validate_selection("bard", None)
    assert "not configured" in reg.validate_selection("openai", None)
    assert reg.validate_selection("anthropic", "claude-sonnet-5") is None


def test_deepseek_uses_openai_wire_protocol_with_its_own_base_url():
    reg = _registry(deepseek_api_key="ds")
    provider = reg.build("deepseek", None)
    assert isinstance(provider, OpenAILLMProvider)
    assert provider.provider_name == "deepseek"
    assert provider._base_url == "https://api.deepseek.com"


def test_for_conversation_prefers_the_conversation_choice_over_default():
    reg = _registry(anthropic_api_key="a", openai_api_key="b")
    llm = reg.for_conversation(provider="openai", model="gpt-4o")
    assert isinstance(llm, OpenAILLMProvider)
    assert llm.model == "gpt-4o"


def test_for_conversation_falls_back_to_default_on_a_stale_choice():
    reg = _registry(anthropic_api_key="a")  # only anthropic configured
    # conversation once picked openai, whose key was since removed
    llm = reg.for_conversation(provider="openai", model="gpt-4o")
    assert isinstance(llm, AnthropicLLMProvider)


async def test_for_testing_pins_one_provider_and_disables_selection():
    reg = LLMRegistry.for_testing(MockLLMProvider())
    assert reg.selection_enabled() is False
    assert isinstance(reg.for_conversation(provider="openai", model="x"), MockLLMProvider)
