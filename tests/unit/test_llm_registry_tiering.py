"""`LLMRegistry.tier_model` — task-complexity tiering (spec: smarter model
routing). Purely the registry-level lookup; `_llm_for`'s "never override an
explicit /model choice" logic is covered separately in
test_orchestrator_model_tiering.py."""

from __future__ import annotations

from numi.agent.llm.registry import LLMRegistry
from numi.common.config import Settings


def _registry(**overrides) -> LLMRegistry:
    return LLMRegistry(Settings(_env_file=None, **overrides))


def test_empty_tier_settings_resolve_to_none():
    """Zero-config deployments see no behavior change at all — `None`
    means "no override," never an empty-string model id reaching a real
    provider constructor."""
    reg = _registry()
    assert reg.tier_model("fast") is None
    assert reg.tier_model("strong") is None


def test_fast_and_strong_resolve_independently():
    reg = _registry(llm_fast_model="gemini-3.1-flash-lite", llm_strong_model="gemini-3.8-pro")
    assert reg.tier_model("fast") == "gemini-3.1-flash-lite"
    assert reg.tier_model("strong") == "gemini-3.8-pro"


def test_only_one_tier_configured_leaves_the_other_none():
    reg = _registry(llm_strong_model="gemini-3.8-pro")
    assert reg.tier_model("fast") is None
    assert reg.tier_model("strong") == "gemini-3.8-pro"
