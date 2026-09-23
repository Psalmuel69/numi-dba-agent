"""Backwards-compatible re-exports.

The LLM provider layer was split into per-vendor modules
(`base`, `mock`, `anthropic_provider`, `openai_provider`, `gemini_provider`)
plus a `registry`. This module keeps the old import paths working.
"""

from __future__ import annotations

from numi.agent.llm.anthropic_provider import AnthropicLLMProvider
from numi.agent.llm.base import LLMProvider, StructuredLLMProvider
from numi.agent.llm.gemini_provider import GeminiLLMProvider
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.openai_provider import DeepSeekLLMProvider, OpenAILLMProvider
from numi.agent.llm.registry import LLMRegistry

__all__ = [
    "LLMProvider",
    "StructuredLLMProvider",
    "MockLLMProvider",
    "AnthropicLLMProvider",
    "OpenAILLMProvider",
    "DeepSeekLLMProvider",
    "GeminiLLMProvider",
    "LLMRegistry",
    "build_llm_registry",
]


def build_llm_registry(settings) -> LLMRegistry:
    return LLMRegistry(settings)
