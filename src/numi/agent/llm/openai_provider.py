"""OpenAI provider, and DeepSeek (which speaks the same wire protocol) —
spec §34."""

from __future__ import annotations

import json
from typing import Any

from numi.agent.llm.base import StructuredLLMProvider

_OPENAI_DEFAULT_MODEL = "gpt-4o"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

_OPENAI_KNOWN = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "o4-mini"]
_DEEPSEEK_KNOWN = ["deepseek-chat", "deepseek-reasoner"]


class OpenAILLMProvider(StructuredLLMProvider):
    provider_name = "openai"
    _default_model = _OPENAI_DEFAULT_MODEL
    _base_url: str | None = None
    _known_models = _OPENAI_KNOWN

    def __init__(self, api_key: str, model: str = ""):
        super().__init__(model or self._default_model)
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def _call_tool(
        self, *, system: str, user: str, schema: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Submit your answer.",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        call = response.choices[0].message.tool_calls[0]
        return json.loads(call.function.arguments)

    async def _call_text(self, *, system: str, user: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            result = await client.models.list()
            ids = [m.id for m in result.data]
            # Filter to chat-capable families where we can tell.
            chat = [
                i
                for i in ids
                if i.startswith(("gpt-", "o1", "o3", "o4", "chatgpt", "deepseek"))
            ]
            return sorted(chat or ids)
        except Exception:  # noqa: BLE001
            return list(self._known_models)


class DeepSeekLLMProvider(OpenAILLMProvider):
    provider_name = "deepseek"
    _default_model = _DEEPSEEK_DEFAULT_MODEL
    _base_url = "https://api.deepseek.com"
    _known_models = _DEEPSEEK_KNOWN
