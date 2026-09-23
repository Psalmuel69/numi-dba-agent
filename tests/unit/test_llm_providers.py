"""Provider-specific SDK plumbing for OpenAI/DeepSeek, Anthropic, and Gemini
(spec §34) — the request/response translation and per-vendor quirks that sit
underneath `StructuredLLMProvider` (already covered by
test_structured_llm_resilience.py) and the registry/fallback orchestration
(test_llm_registry.py). No network: every SDK client is replaced with a
lightweight fake built from plain attribute-holding objects, matching only
the shape each provider actually reads off a real response — not the
`unittest.mock` style, to keep failures readable as "wrong shape" rather
than "wrong mock call."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from numi.agent.llm.anthropic_provider import AnthropicLLMProvider
from numi.agent.llm.gemini_provider import (
    _DEFAULT_COOLDOWN_SECONDS,
    _MAX_COOLDOWN_SECONDS,
    _MODEL_FALLBACK_CHAIN,
    GeminiLLMProvider,
    _clean_schema,
    _cooldown_seconds,
    _is_model_unavailable_error,
    _meets_min_version,
)
from numi.agent.llm.openai_provider import DeepSeekLLMProvider, OpenAILLMProvider

# --- OpenAI / DeepSeek ------------------------------------------------------


class _FakeToolCall:
    def __init__(self, arguments_json: str):
        self.function = SimpleNamespace(arguments=arguments_json)


class _FakeOpenAIClient:
    """Discriminates tool vs. text calls the same way the real API does:
    a tool call always passes `tools=`/`tool_choice=`, a text call never
    does."""

    def __init__(
        self,
        *,
        tool_args_json: str = "{}",
        text_content: str | None = "hi",
        model_ids: list[str] | None = None,
    ):
        self._tool_args_json = tool_args_json
        self._text_content = text_content
        self._model_ids = model_ids or []
        self.last_kwargs: dict | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if "tools" in kwargs:
            message = SimpleNamespace(tool_calls=[_FakeToolCall(self._tool_args_json)])
        else:
            message = SimpleNamespace(content=self._text_content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def _list(self):
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in self._model_ids])


class _FailingModelsClient(_FakeOpenAIClient):
    async def _list(self):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_openai_call_tool_parses_json_arguments_off_the_tool_call():
    provider = OpenAILLMProvider(api_key="k")
    provider._client = _FakeOpenAIClient(tool_args_json='{"action": "conclude", "summary": "ok"}')
    result = await provider._call_tool(system="s", user="u", schema={"type": "object"}, tool_name="submit")
    assert result == {"action": "conclude", "summary": "ok"}
    assert provider._client.last_kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit"}}


@pytest.mark.asyncio
async def test_openai_call_text_returns_message_content():
    provider = OpenAILLMProvider(api_key="k")
    provider._client = _FakeOpenAIClient(text_content="a plain summary")
    assert await provider._call_text(system="s", user="u") == "a plain summary"


@pytest.mark.asyncio
async def test_openai_call_text_returns_empty_string_when_content_is_none():
    """A tool-call-only response (no message.content) must degrade to "",
    not None — StructuredLLMProvider.summarize_for_human returns this
    directly to the DBA."""
    provider = OpenAILLMProvider(api_key="k")
    provider._client = _FakeOpenAIClient(text_content=None)
    assert await provider._call_text(system="s", user="u") == ""


@pytest.mark.asyncio
async def test_openai_list_models_filters_to_chat_capable_families():
    provider = OpenAILLMProvider(api_key="k")
    provider._client = _FakeOpenAIClient(
        model_ids=["gpt-4o", "gpt-4o-mini", "text-embedding-3-large", "whisper-1", "deepseek-chat"]
    )
    models = await provider.list_models()
    assert models == sorted(["gpt-4o", "gpt-4o-mini", "deepseek-chat"])


@pytest.mark.asyncio
async def test_openai_list_models_falls_back_to_full_id_list_when_nothing_matches_the_chat_filter():
    provider = OpenAILLMProvider(api_key="k")
    provider._client = _FakeOpenAIClient(model_ids=["text-embedding-3-large", "whisper-1"])
    models = await provider.list_models()
    assert models == sorted(["text-embedding-3-large", "whisper-1"])


@pytest.mark.asyncio
async def test_openai_list_models_falls_back_to_known_models_on_api_error():
    provider = OpenAILLMProvider(api_key="k")
    provider._client = _FailingModelsClient()
    assert await provider.list_models() == provider._known_models


def test_openai_get_client_is_cached_across_calls():
    provider = OpenAILLMProvider(api_key="k")
    first = provider._get_client()
    assert provider._get_client() is first
    assert str(first.base_url) == "https://api.openai.com/v1/"


def test_deepseek_provider_points_its_client_at_deepseeks_base_url():
    """DeepSeek reuses OpenAILLMProvider wholesale (same wire protocol) —
    the only thing that must actually differ is the base_url the real SDK
    client is constructed with."""
    provider = DeepSeekLLMProvider(api_key="k")
    client = provider._get_client()
    assert str(client.base_url).rstrip("/") == "https://api.deepseek.com"
    assert provider.provider_name == "deepseek"
    assert provider.model == "deepseek-chat"


# --- Anthropic ---------------------------------------------------------------


class _FakeAnthropicClient:
    def __init__(self, *, content_blocks=None, model_ids: list[str] | None = None):
        self._content_blocks = content_blocks or []
        self._model_ids = model_ids or []
        self.last_kwargs: dict | None = None
        self.messages = SimpleNamespace(create=self._create)
        self.models = SimpleNamespace(list=self._list)

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(content=self._content_blocks)

    async def _list(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in self._model_ids])


class _FailingAnthropicModelsClient(_FakeAnthropicClient):
    async def _list(self, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_anthropic_call_tool_extracts_the_tool_use_blocks_input():
    provider = AnthropicLLMProvider(api_key="k")
    # A real response can carry other block types (e.g. thinking/text)
    # alongside the tool_use block — the code must pick the tool_use one
    # specifically, not just the first block.
    blocks = [
        SimpleNamespace(type="text", text="thinking out loud"),
        SimpleNamespace(type="tool_use", input={"action": "conclude", "summary": "done"}),
    ]
    provider._client = _FakeAnthropicClient(content_blocks=blocks)
    result = await provider._call_tool(system="s", user="u", schema={"type": "object"}, tool_name="submit")
    assert result == {"action": "conclude", "summary": "done"}
    assert provider._client.last_kwargs["tool_choice"] == {"type": "tool", "name": "submit"}


@pytest.mark.asyncio
async def test_anthropic_call_text_joins_every_text_block_and_skips_others():
    provider = AnthropicLLMProvider(api_key="k")
    blocks = [
        SimpleNamespace(type="text", text="Part one. "),
        SimpleNamespace(type="tool_use", input={}),  # must be ignored, not concatenated
        SimpleNamespace(type="text", text="Part two."),
    ]
    provider._client = _FakeAnthropicClient(content_blocks=blocks)
    assert await provider._call_text(system="s", user="u") == "Part one. Part two."


@pytest.mark.asyncio
async def test_anthropic_list_models_returns_live_ids():
    provider = AnthropicLLMProvider(api_key="k")
    provider._client = _FakeAnthropicClient(model_ids=["claude-opus-5", "claude-sonnet-5"])
    assert await provider.list_models() == ["claude-opus-5", "claude-sonnet-5"]


@pytest.mark.asyncio
async def test_anthropic_list_models_falls_back_to_known_models_on_api_error():
    from numi.agent.llm.anthropic_provider import _KNOWN_MODELS

    provider = AnthropicLLMProvider(api_key="k")
    provider._client = _FailingAnthropicModelsClient()
    assert await provider.list_models() == _KNOWN_MODELS


def test_anthropic_get_client_is_cached_across_calls():
    provider = AnthropicLLMProvider(api_key="k")
    first = provider._get_client()
    assert provider._get_client() is first


# --- Gemini --------------------------------------------------------------


def _gemini_provider(model: str = "gemini-3.6-flash") -> GeminiLLMProvider:
    return GeminiLLMProvider(api_key="k", model=model)


class _AsyncIter:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self._items:
            yield item


class _FakeGeminiClient:
    def __init__(self, *, generate_response=None, list_items=None, list_error: Exception | None = None):
        self._generate_response = generate_response
        self._list_items = list_items or []
        self._list_error = list_error
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=self._generate_content, list=self._list)
        )

    async def _generate_content(self, *, model, contents, config):
        return self._generate_response

    async def _list(self):
        if self._list_error is not None:
            raise self._list_error
        return _AsyncIter(self._list_items)


def _gemini_response(*, function_call_args=None, text=None, no_function_call_part=False):
    if function_call_args is not None or no_function_call_part:
        part = SimpleNamespace(
            function_call=SimpleNamespace(args=function_call_args) if function_call_args is not None else None
        )
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
    return SimpleNamespace(text=text)


@pytest.mark.asyncio
async def test_gemini_call_tool_extracts_function_call_args():
    provider = _gemini_provider()
    provider._client = _FakeGeminiClient(
        generate_response=_gemini_response(function_call_args={"action": "conclude"})
    )
    result = await provider._call_tool(
        system="s",
        user="u",
        schema={"type": "object", "properties": {"action": {"type": "string"}}},
        tool_name="submit",
    )
    assert result == {"action": "conclude"}


@pytest.mark.asyncio
async def test_gemini_call_tool_raises_when_no_part_carries_a_function_call():
    provider = _gemini_provider()
    provider._client = _FakeGeminiClient(generate_response=_gemini_response(no_function_call_part=True))
    with pytest.raises(ValueError, match="no function call"):
        await provider._call_tool(system="s", user="u", schema={"type": "object"}, tool_name="submit")


@pytest.mark.asyncio
async def test_gemini_call_text_returns_response_text():
    provider = _gemini_provider()
    provider._client = _FakeGeminiClient(generate_response=_gemini_response(text="a summary"))
    assert await provider._call_text(system="s", user="u") == "a summary"


@pytest.mark.asyncio
async def test_gemini_call_text_returns_empty_string_when_text_is_none():
    provider = _gemini_provider()
    provider._client = _FakeGeminiClient(generate_response=_gemini_response(text=None))
    assert await provider._call_text(system="s", user="u") == ""


@pytest.mark.asyncio
async def test_gemini_list_models_filters_by_min_version_and_capability():
    provider = _gemini_provider()
    items = [
        SimpleNamespace(name="models/gemini-3.6-flash", supported_actions=["generateContent"]),
        # below min version
        SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
        # wrong capability
        SimpleNamespace(name="models/gemini-3.7-flash", supported_actions=["embedContent"]),
        # not gemini at all
        SimpleNamespace(name="models/text-bison", supported_actions=["generateContent"]),
        # no actions listed -> included
        SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=[]),
    ]
    provider._client = _FakeGeminiClient(list_items=items)
    assert await provider.list_models() == ["gemini-3.6-flash", "gemini-3.8-flash"]


@pytest.mark.asyncio
async def test_gemini_list_models_falls_back_to_known_models_on_api_error():
    from numi.agent.llm.gemini_provider import _KNOWN_MODELS

    provider = _gemini_provider()
    provider._client = _FakeGeminiClient(list_error=RuntimeError("boom"))
    assert await provider.list_models() == _KNOWN_MODELS


def test_gemini_get_client_is_cached_across_calls():
    provider = _gemini_provider()
    first = provider._get_client()
    assert provider._get_client() is first


# --- Gemini: per-model quota/capacity fallback ------------------------------
#
# `_with_model_fallback` is tested directly against a fake `call()` (the
# same pattern test_structured_llm_resilience.py uses for the shared retry
# logic) rather than through `_call_tool`, so the model-switching bookkeeping
# is exercised without needing a second layer of fake SDK response objects.


@pytest.mark.asyncio
async def test_with_model_fallback_switches_model_once_on_a_quota_error_then_succeeds():
    provider = _gemini_provider()
    attempted_models: list[str] = []

    async def call():
        attempted_models.append(provider.model)
        if len(attempted_models) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: daily quota")
        return "ok"

    result = await provider._with_model_fallback(call)
    assert result == "ok"
    assert attempted_models == ["gemini-3.6-flash", "gemini-3.7-flash"]
    assert provider.model == "gemini-3.7-flash"
    assert "gemini-3.6-flash" in provider._unavailable_until


@pytest.mark.asyncio
async def test_with_model_fallback_gives_up_after_max_switches_are_exhausted():
    provider = _gemini_provider()
    attempted_models: list[str] = []

    async def call():
        attempted_models.append(provider.model)
        raise RuntimeError("503 UNAVAILABLE: high demand")

    with pytest.raises(RuntimeError, match="high demand"):
        await provider._with_model_fallback(call)
    # _MAX_FALLBACK_SWITCHES=2: original model + 2 switches = 3 attempts.
    assert attempted_models == ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]


@pytest.mark.asyncio
async def test_with_model_fallback_never_retries_an_error_unrelated_to_model_availability():
    provider = _gemini_provider()
    attempts = 0

    async def call():
        nonlocal attempts
        attempts += 1
        raise ValueError("malformed schema")

    with pytest.raises(ValueError):
        await provider._with_model_fallback(call)
    assert attempts == 1
    assert provider.model == "gemini-3.6-flash"  # never switched


def test_next_fallback_model_skips_models_still_on_cooldown():
    provider = _gemini_provider()
    provider._unavailable_until["gemini-3.7-flash"] = __import__("time").monotonic() + 100
    assert provider._next_fallback_model() == "gemini-3.8-flash"


def test_next_fallback_model_returns_none_when_every_other_model_is_on_cooldown():
    provider = _gemini_provider()
    now = __import__("time").monotonic()
    for candidate in _MODEL_FALLBACK_CHAIN:
        if candidate != provider.model:
            provider._unavailable_until[candidate] = now + 100
    assert provider._next_fallback_model() is None


# --- Gemini: small pure-function helpers ------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError(),
        RuntimeError("429 Too Many Requests"),
        RuntimeError("RESOURCE_EXHAUSTED: daily quota"),
        RuntimeError("503 UNAVAILABLE"),
        RuntimeError("the model is currently experiencing high demand"),
    ],
)
def test_is_model_unavailable_error_recognizes_quota_and_capacity_failures(exc):
    assert _is_model_unavailable_error(exc) is True


def test_is_model_unavailable_error_false_for_a_request_specific_problem():
    assert _is_model_unavailable_error(ValueError("invalid schema: unknown field")) is False


def test_meets_min_version_true_for_v3_and_above():
    assert _meets_min_version("gemini-3.6-flash") is True
    assert _meets_min_version("gemini-4.0-flash") is True


def test_meets_min_version_false_below_v3_or_when_unparseable():
    assert _meets_min_version("gemini-2.5-flash") is False
    assert _meets_min_version("text-bison-001") is False


def test_cooldown_seconds_uses_the_apis_own_retry_delay_when_present():
    exc = RuntimeError("... please retry in retryDelay: '24.8s' ...")
    assert _cooldown_seconds(exc) == 24.8


def test_cooldown_seconds_caps_a_very_long_retry_delay():
    exc = RuntimeError("retryDelay: '999s'")
    assert _cooldown_seconds(exc) == _MAX_COOLDOWN_SECONDS


def test_cooldown_seconds_falls_back_to_the_default_when_no_retry_delay_is_present():
    exc = RuntimeError("503 UNAVAILABLE: high demand")
    assert _cooldown_seconds(exc) == _DEFAULT_COOLDOWN_SECONDS


def test_cooldown_seconds_falls_back_to_the_default_when_the_captured_value_cannot_be_parsed(monkeypatch):
    """The real regex only ever captures digits, so this branch is a pure
    defensive fallback — pin it directly by swapping in a regex that can
    capture something float() rejects, rather than leaving it unexercised."""
    import numi.agent.llm.gemini_provider as gemini_provider_module

    monkeypatch.setattr(gemini_provider_module, "_RETRY_DELAY_RE", re.compile(r"retryDelay: (\w+)"))
    exc = RuntimeError("retryDelay: soon")
    assert gemini_provider_module._cooldown_seconds(exc) == _DEFAULT_COOLDOWN_SECONDS


def test_clean_schema_flattens_an_optional_field_to_its_real_type_and_marks_nullable():
    """Pins the live bug this function exists to fix: dropping `anyOf`
    outright (instead of flattening it) silently turned every Optional
    field into an unconstrained `{}`, so a real model stopped respecting
    an enum constraint it was never actually shown."""
    schema = {
        "type": "object",
        "title": "IntentExtraction",
        "$schema": "http://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "environment_hint": {
                "anyOf": [
                    {"type": "string", "enum": ["development", "uat", "production"]},
                    {"type": "null"},
                ],
                "description": "dropped — not in the allow-list",
            },
            "tags": {
                "type": "array",
                "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
    }
    cleaned = _clean_schema(schema)
    assert "title" not in cleaned
    assert "$schema" not in cleaned
    assert "additionalProperties" not in cleaned

    env = cleaned["properties"]["environment_hint"]
    assert "anyOf" not in env
    assert "description" not in env
    assert env["type"] == "string"
    assert env["enum"] == ["development", "uat", "production"]
    assert env["nullable"] is True

    # Recursion into `items` for an array-typed field.
    tag_items = cleaned["properties"]["tags"]["items"]
    assert tag_items["type"] == "string"
    assert tag_items["nullable"] is True


def test_clean_schema_drops_a_genuine_multi_type_union_without_crashing():
    """A real (non-Optional) multi-type anyOf has no single non-null branch
    to flatten to — the function must fall through gracefully (losing type
    info, same as before the Optional-specific fix) rather than raise."""
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert _clean_schema(schema) == {}
