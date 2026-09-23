"""Anthropic (Claude) provider (spec §34)."""

from __future__ import annotations

from typing import Any

from numi.agent.llm.base import StructuredLLMProvider

_DEFAULT_MODEL = "claude-sonnet-5"

# Fallback list used only if a live `models.list()` call fails (offline /
# permissions). The live call is the source of truth for "what this key can
# actually use".
_KNOWN_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5-1",
    "claude-haiku-4-5-20251001",
]


class AnthropicLLMProvider(StructuredLLMProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str = ""):
        super().__init__(model or _DEFAULT_MODEL)
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def _call_tool(
        self, *, system: str, user: str, schema: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=[{"name": tool_name, "description": "Submit your answer.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        return dict(tool_use.input)

    async def _call_text(self, *, system: str, user: str) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            result = await client.models.list(limit=100)
            return [m.id for m in result.data]
        except Exception:  # noqa: BLE001
            return list(_KNOWN_MODELS)
