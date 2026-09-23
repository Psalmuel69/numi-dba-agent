"""LLM provider/model registry (spec §34, §36).

Turns configuration ("which API keys are present") into:
  - the set of providers a DBA may choose from,
  - the live list of models each key can actually use,
  - a concrete `LLMProvider` for a chosen (provider, model),
  - the provider to use for a given conversation (its explicit choice, or
    the configured default).

The Agent never bakes in a single vendor: everything above the registry
(`agent.orchestrator`) only ever sees the vendor-neutral `LLMProvider`
interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from numi.agent.llm.anthropic_provider import AnthropicLLMProvider
from numi.agent.llm.base import LLMProvider
from numi.agent.llm.fallback import CrossProviderFallbackLLM, FallbackEvent
from numi.agent.llm.gemini_provider import GeminiLLMProvider
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.openai_provider import DeepSeekLLMProvider, OpenAILLMProvider
from numi.common.config import Settings

_PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicLLMProvider,
    "openai": OpenAILLMProvider,
    "gemini": GeminiLLMProvider,
    "deepseek": DeepSeekLLMProvider,
}


class LLMRegistry:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._keys = settings._llm_api_keys()
        self._cache: dict[tuple[str, str], LLMProvider] = {}
        # Set by `for_testing` — pins every conversation to one provider and
        # disables selection.
        self._fixed: LLMProvider | None = None

    # -- test seam ---------------------------------------------------------

    @classmethod
    def for_testing(cls, provider: LLMProvider) -> LLMRegistry:
        registry = cls(Settings(_env_file=None, llm_provider="mock"))
        registry._fixed = provider
        return registry

    # -- capabilities ----------------------------------------------------

    def configured_providers(self) -> list[str]:
        return self._settings.configured_llm_providers()

    def selection_enabled(self) -> bool:
        return (
            self._fixed is None
            and self._settings.allow_user_model_selection
            and not self._settings.llm_selection_locked()
        )

    def default(self) -> tuple[str, str]:
        return self._settings.effective_default_llm()

    def tier_model(self, call_type: Literal["fast", "strong"]) -> str | None:
        """The model id a task-complexity tier resolves to, within
        whatever provider the conversation already has — `None` (not an
        empty string) means "no tier configured, use the provider's own
        default," so callers can tell "no override" apart from "override
        to the provider default" without a second sentinel."""
        model = (
            self._settings.llm_fast_model
            if call_type == "fast"
            else self._settings.llm_strong_model
        )
        return model or None

    async def list_models(self, provider: str) -> list[str]:
        if provider not in self._keys or not self._keys[provider].strip():
            return []
        return await self._construct(provider, "").list_models()

    async def describe_available(self) -> str:
        configured = self.configured_providers()
        if not configured:
            return (
                "No LLM providers are configured. Set ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY to enable "
                "one. Currently using the deterministic offline planner."
            )
        lines = []
        for provider in configured:
            models = await self.list_models(provider)
            preview = ", ".join(models[:8]) + ("  …" if len(models) > 8 else "")
            lines.append(f"- {provider}: {preview or '(no models returned)'}")
        current_p, current_m = self.default()
        footer = (
            f"\nCurrent default: {current_p}"
            + (f" / {current_m}" if current_m else " (provider default)")
        )
        if not self.selection_enabled():
            footer += "\n(Per-conversation switching is disabled by configuration.)"
        return "Available LLMs:\n" + "\n".join(lines) + footer

    # -- construction --------------------------------------------------

    def _construct(self, provider: str, model: str) -> LLMProvider:
        if provider == "mock":
            return MockLLMProvider()
        key = self._keys.get(provider, "")
        if not key.strip():
            raise ValueError(f"No API key configured for provider '{provider}'.")
        cls = _PROVIDER_CLASSES.get(provider)
        if cls is None:
            raise ValueError(f"Unknown LLM provider '{provider}'.")
        return cls(key, model)  # type: ignore[call-arg]

    def build(self, provider: str, model: str | None) -> LLMProvider:
        if self._fixed is not None:
            return self._fixed
        cache_key = (provider, model or "")
        if cache_key not in self._cache:
            self._cache[cache_key] = self._construct(provider, model or "")
        return self._cache[cache_key]

    def for_conversation(self, *, provider: str | None, model: str | None) -> LLMProvider:
        """Resolve the provider for a conversation given its (possibly unset)
        explicit choice."""
        if self._fixed is not None:
            return self._fixed
        default_provider, default_model = self.default()
        chosen_provider = provider or default_provider
        chosen_model = model or (default_model if not provider else None)
        try:
            return self.build(chosen_provider, chosen_model)
        except ValueError:
            # A stale/invalid selection falls back to the default rather than
            # erroring mid-conversation.
            return self.build(default_provider, default_model)

    def resilient_for_conversation(
        self,
        *,
        provider: str | None,
        model: str | None,
        notices: list[FallbackEvent] | None = None,
    ) -> LLMProvider:
        """`for_conversation`, wrapped so a total outage of the chosen
        provider escapes to the next configured one instead of dead-ending
        in "temporarily unavailable" (see `agent.llm.fallback`).

        Returns the bare provider — byte-identical to `for_conversation` —
        in the two cases where a fallback chain would be meaningless:

        - the resolved provider is the deterministic offline planner
          (`llm_provider="mock"`, or no keys configured at all, or
          `for_testing`). That path must never reach network code or any
          fallback logic; it is what the entire default test suite runs on.
        - there is nothing else to fall back TO — a single configured
          provider, which is today's actual common case. Not wrapping keeps
          that deployment on exactly the code path (and the single, full
          `_OVERALL_DEADLINE_SECONDS` budget) it had before this existed.

        A locked `LLM_PROVIDER` or an explicit `/model` choice deliberately
        does NOT suppress the chain: an answer from another vendor beats
        telling a DBA the agent is stuck while three usable API keys sit
        idle. What it does guarantee is disclosure — every substitution is
        recorded on `notices` and surfaced in the reply text by
        `AgentOrchestrator.handle_message`.
        """
        if self._fixed is not None:
            # `for_testing` pinned one provider for every conversation —
            # there is nothing to fall back to by construction, and the
            # pinned object is frequently a bare duck-typed test double
            # rather than a real `LLMProvider` subclass. Short-circuited
            # first, exactly as `for_conversation` does, so this layer can
            # never impose a new interface requirement on test doubles.
            return self._fixed
        primary = self.for_conversation(provider=provider, model=model)
        if primary.provider_name == "mock":
            return primary
        rest = [p for p in self.configured_providers() if p != primary.provider_name]
        if not rest:
            return primary
        # `model` is deliberately NOT carried across: a model id is
        # provider-specific ("gemini-3.5-flash" means nothing to Anthropic),
        # so each fallback uses its own default model. Built lazily — a
        # provider whose SDK constructor rejects its key must not break the
        # primary path at construction time.
        fallbacks: list[tuple[str, Callable[[], LLMProvider]]] = [
            (name, (lambda n=name: self.build(n, None))) for name in rest  # type: ignore[misc]
        ]
        return CrossProviderFallbackLLM(primary, fallbacks, notices=notices)

    def validate_selection(self, provider: str, model: str | None) -> str | None:
        """Return an error string if (provider, model) can't be selected, else
        None. Model membership is checked by the caller against `list_models`
        (async)."""
        if not self.selection_enabled():
            return "Model selection is disabled by this deployment's configuration."
        if provider == "mock":
            return None
        if provider not in _PROVIDER_CLASSES:
            return (
                f"Unknown provider '{provider}'. Choose one of: "
                + ", ".join(_PROVIDER_CLASSES)
            )
        if not self._keys.get(provider, "").strip():
            return f"Provider '{provider}' is not configured (no API key)."
        return None
