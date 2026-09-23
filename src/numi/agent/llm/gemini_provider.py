"""Google Gemini provider (spec §34).

Uses the unified `google-genai` SDK. Gemini's function-calling schema is a
restricted subset of OpenAPI (no `$defs`, no discriminated unions) — the
flat schema in `agent.llm.base` is written to stay inside that subset, and
the strict discriminated-union validation happens afterwards via
`agent_action_adapter`.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from numi.agent.llm.base import StructuredLLMProvider, logger

# Gemini 2.x and earlier are deliberately excluded from selection — operator
# policy, not a technical limitation. `_meets_min_version` is what actually
# enforces this (against the live `models.list()` result); these two
# constants are only the fallback used if that call fails, so they must
# name real, currently-serving models — not a guessed/rounded id.
_MIN_MAJOR_VERSION = 3
_DEFAULT_MODEL = "gemini-3.6-flash"
_KNOWN_MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash"]

# The free tier's daily quota is per (project, model), and capacity
# ("high demand") shedding is also observed per-model — a project easily
# burns through one model's 20 req/day during real interactive testing, and
# a specific model can stay overloaded across several same-model retries in
# a row (observed live: three straight 503s on gemini-3.7-flash, ~15-20s
# apart, well past what a couple of quick retries can ride out). Ranked
# fallback chain tried, in order, whenever the *current* model reports
# either kind of "this model specifically isn't working right now" — never
# for any other kind of error, which stays with StructuredLLMProvider's own
# same-model retry. Real, currently-serving 3.x+ models, verified live
# 2026-09-11; refresh via list_models() if this goes stale. Mutating
# `self.model` on failure (rather than raising) is deliberate — the
# LLMRegistry caches one provider instance per (provider, model) key, so
# the switch sticks for the rest of this process's requests instead of
# rediscovering the same bad model every time.
_MODEL_FALLBACK_CHAIN = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
]

_VERSION_RE = re.compile(r"gemini-(\d+)")


def _meets_min_version(model_name: str) -> bool:
    match = _VERSION_RE.search(model_name)
    if match is None:
        return False
    return int(match.group(1)) >= _MIN_MAJOR_VERSION


def _is_model_unavailable_error(exc: Exception) -> bool:
    """True for a failure that's specifically about *this model* right now
    — a daily quota (429/RESOURCE_EXHAUSTED), capacity shedding
    (503/UNAVAILABLE/"high demand"), or the request simply never coming
    back (see _REQUEST_TIMEOUT_SECONDS — verified live: a stuck call left
    the whole conversation hanging for minutes with the client-side
    timeout the only thing that ever ended it, since nothing here had
    raised yet for the retry/fallback machinery to react to) — as opposed
    to a request-specific problem (a malformed schema, an auth failure)
    that switching models would not fix."""
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError is this on 3.11+
        return True
    text = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in text
        or "429" in text
        or "quota" in text.lower()
        or "UNAVAILABLE" in text
        or "503" in text
        or "high demand" in text.lower()
    )




_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s")
# Gemini's own 503/429 error payload usually names a retryDelay ("Please
# retry in 24.8s") — use it when present, since it's the API's own estimate
# of when *this* failure specifically clears. When absent (or clearly a
# multi-hour/day quota reset the caller shouldn't just sit and wait for),
# fall back to a short cooldown — long enough to skip a model that's
# genuinely still down, short enough that a model marked unavailable during
# one burst of testing isn't permanently blacklisted for the rest of this
# process's life (verified live: exactly what happened without this — every
# model in the chain ended up marked bad within one long testing session,
# even though several had only hit a brief 503, not the day-long quota).
_DEFAULT_COOLDOWN_SECONDS = 60.0
_MAX_COOLDOWN_SECONDS = 120.0


def _cooldown_seconds(exc: Exception) -> float:
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        try:
            return min(float(match.group(1)), _MAX_COOLDOWN_SECONDS)
        except ValueError:
            pass
    return _DEFAULT_COOLDOWN_SECONDS


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop keys Gemini's schema validator rejects (`description` on the
    root, `$schema`, `title`, `additionalProperties`)."""
    # Pydantic v2 represents `X | None` as anyOf: [<X's schema>, {"type":
    # "null"}] — Gemini's schema subset has no anyOf, and this function used
    # to just drop it via the allow-list below, silently discarding the
    # actual type/enum along with it. That meant EVERY Optional field in
    # this codebase's schemas (IntentExtraction.database_hint/
    # environment_hint/instance_hint, all `str | None`) reached Gemini as an
    # unconstrained `{}` — confirmed live: a real model wrote environment
    # "dev" because nothing ever told it the field was constrained to
    # {development, uat, production} in the first place. Flatten instead:
    # keep the non-null branch's constraints and mark nullable.
    if "anyOf" in schema:
        variants = schema["anyOf"]
        non_null = [v for v in variants if v.get("type") != "null"]
        is_nullable = any(v.get("type") == "null" for v in variants)
        if len(non_null) == 1:
            schema = {**schema, **non_null[0]}
            schema.pop("anyOf", None)
            if is_nullable:
                schema["nullable"] = True
        # else: a real multi-type union — rare in this codebase's schemas;
        # falls through and loses type info same as before this fix.
    allowed = {"type", "properties", "required", "enum", "items", "nullable"}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k not in allowed:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema(pv) if isinstance(pv, dict) else pv for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _clean_schema(v)
        else:
            out[k] = v
    return out


class GeminiLLMProvider(StructuredLLMProvider):
    provider_name = "gemini"
    model: str

    # How long a single generateContent call may run before it's treated as
    # this model not responding (see _is_model_unavailable_error) and
    # retried against a different one. Deliberately tight — a working call
    # in this codebase's live testing has consistently finished in single-
    # digit seconds; this exists to fail fast on a hang, not to patiently
    # wait one out. It also has to leave real room under
    # StructuredLLMProvider._OVERALL_DEADLINE_SECONDS (20s), which bounds
    # the *whole* decision regardless of how many models get tried — a
    # generous per-call timeout just eats that budget on the first model
    # and leaves none for a fallback to even attempt. A class attribute
    # (like _CALL_RETRY_DELAY_SECONDS) so tests can override it.
    _REQUEST_TIMEOUT_SECONDS = 8.0

    # How many *different* models this call will try before giving up, even
    # if more are technically off cooldown — bounds worst-case latency to a
    # small, predictable multiple of _REQUEST_TIMEOUT_SECONDS instead of
    # potentially cascading through the entire fallback chain (verified
    # live: a systemic outage can make several models fail in a row, each
    # consuming its own timeout).
    _MAX_FALLBACK_SWITCHES = 2

    def __init__(self, api_key: str, model: str = ""):
        super().__init__(model or _DEFAULT_MODEL)
        self._api_key = api_key
        self._client = None
        # model -> the monotonic time it becomes eligible again — a cooldown,
        # not a permanent blacklist (see _cooldown_seconds).
        self._unavailable_until: dict[str, float] = {}

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _next_fallback_model(self) -> str | None:
        now = time.monotonic()
        for candidate in _MODEL_FALLBACK_CHAIN:
            if candidate == self.model:
                continue
            if self._unavailable_until.get(candidate, 0.0) <= now:
                return candidate
        return None

    async def _with_model_fallback(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """Run `call()` against `self.model`; on a quota or capacity error
        specific to this model, switch to the next untried model in the
        fallback chain and retry the SAME request, rather than surfacing the
        error or waiting out a same-model retry that may not clear in time —
        both a daily quota and demand-based shedding are a property of
        (project, model), not of the request itself. Bounded by
        _MAX_FALLBACK_SWITCHES — see that attribute's docstring for why an
        unbounded cascade is itself a latency problem, not just a resilience
        feature."""
        switches = 0
        while True:
            try:
                return await asyncio.wait_for(call(), timeout=self._REQUEST_TIMEOUT_SECONDS)
            except Exception as exc:
                if not _is_model_unavailable_error(exc):
                    raise
                cooldown = _cooldown_seconds(exc)
                self._unavailable_until[self.model] = time.monotonic() + cooldown
                if switches >= self._MAX_FALLBACK_SWITCHES:
                    raise
                next_model = self._next_fallback_model()
                if next_model is None:
                    raise
                switches += 1
                logger.warning(
                    "gemini_model_unavailable_switching",
                    from_model=self.model,
                    to_model=next_model,
                    cooldown_seconds=cooldown,
                    switch_number=switches,
                )
                self.model = next_model

    async def _call_tool(
        self, *, system: str, user: str, schema: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        from google.genai import types

        declaration = types.FunctionDeclaration(
            name=tool_name,
            description="Submit your answer.",
            parameters=types.Schema(**_clean_schema(schema)),
        )
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[declaration])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[tool_name],
                )
            ),
        )

        async def call():
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model, contents=user, config=config
            )
            for part in response.candidates[0].content.parts:
                if part.function_call is not None:
                    return dict(part.function_call.args)
            raise ValueError("Gemini returned no function call.")

        return await self._with_model_fallback(call)

    async def _call_text(self, *, system: str, user: str) -> str:
        from google.genai import types

        async def call():
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text or ""

        return await self._with_model_fallback(call)

    async def list_models(self) -> list[str]:
        try:
            client = self._get_client()
            models: list[str] = []
            async for m in await client.aio.models.list():
                actions = getattr(m, "supported_actions", None) or []
                if "generateContent" in actions or not actions:
                    name = (m.name or "").removeprefix("models/")
                    if name.startswith("gemini") and _meets_min_version(name):
                        models.append(name)
            return sorted(set(models)) or list(_KNOWN_MODELS)
        except Exception:  # noqa: BLE001
            return list(_KNOWN_MODELS)
