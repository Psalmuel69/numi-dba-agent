"""Gemini model selection is restricted to 3.x and above — operator policy
enforced in `GeminiLLMProvider.list_models` (spec §34). Also covers the
automatic model-fallback behavior added after live testing repeatedly hit
the free tier's 20 req/day/model quota *and* sustained per-model capacity
shedding (503 "high demand" persisting across several same-model retries)."""

from __future__ import annotations

import asyncio
import time

import pytest

from numi.agent.llm.gemini_provider import (
    _DEFAULT_COOLDOWN_SECONDS,
    _DEFAULT_MODEL,
    _KNOWN_MODELS,
    _MAX_COOLDOWN_SECONDS,
    _MODEL_FALLBACK_CHAIN,
    GeminiLLMProvider,
    _clean_schema,
    _cooldown_seconds,
    _is_model_unavailable_error,
    _meets_min_version,
)
from numi.agent.planner.actions import IntentExtraction


def test_default_and_fallback_models_are_gemini_3_or_later():
    assert _meets_min_version(_DEFAULT_MODEL)
    assert all(_meets_min_version(m) for m in _KNOWN_MODELS)


def test_meets_min_version_accepts_gemini_3_and_above():
    assert _meets_min_version("gemini-3-flash")
    assert _meets_min_version("gemini-3-pro")
    assert _meets_min_version("gemini-4-flash")


def test_meets_min_version_rejects_gemini_2_and_below():
    assert not _meets_min_version("gemini-2.5-pro")
    assert not _meets_min_version("gemini-2.0-flash")
    assert not _meets_min_version("gemini-1.5-flash")


def test_meets_min_version_rejects_unversioned_names():
    assert not _meets_min_version("gemini-pro")
    assert not _meets_min_version("text-embedding-004")


def test_is_model_unavailable_error_recognizes_the_real_gemini_429_shape():
    # Reproduces the live error text verbatim (trimmed).
    exc = RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You "
        "exceeded your current quota... GenerateRequestsPerDayPerProjectPerModel-FreeTier'}}"
    )
    assert _is_model_unavailable_error(exc)


def test_is_model_unavailable_error_recognizes_the_real_gemini_503_shape():
    # Reproduces the live error text verbatim — sustained across several
    # same-model retries in production, which is exactly why 503 triggers
    # a model switch too, not just 429.
    exc = RuntimeError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
        "currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
    )
    assert _is_model_unavailable_error(exc)


def test_is_model_unavailable_error_does_not_misclassify_a_request_specific_failure():
    """A malformed/empty completion is a reasoning problem the SAME model
    may well get right on the very next attempt — not a signal that this
    model itself is unavailable, so it must not trigger a model switch
    (StructuredLLMProvider's own same-model retry handles it instead)."""
    assert not _is_model_unavailable_error(ValueError("Gemini returned no function call."))


def test_is_model_unavailable_error_treats_a_timeout_as_unavailable():
    """Reproduces a live finding: a generateContent call with no timeout at
    all hung for minutes with nothing raised — the retry/fallback machinery
    only reacts to a raised exception, so a bare hang bypassed it entirely
    and the client-side timeout was the only thing that ever ended it."""
    assert _is_model_unavailable_error(TimeoutError())


@pytest.mark.asyncio
async def test_a_hanging_call_times_out_and_switches_to_the_next_model():
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])
    provider._REQUEST_TIMEOUT_SECONDS = 0.05  # instance override, keep the test fast

    async def call():
        if provider.model == _MODEL_FALLBACK_CHAIN[0]:
            await asyncio.sleep(10)  # never actually reached — wait_for cuts it off
        return "ok"

    result = await provider._with_model_fallback(call)
    assert result == "ok"
    assert provider.model == _MODEL_FALLBACK_CHAIN[1]


@pytest.mark.asyncio
async def test_switches_to_the_next_model_on_quota_exhaustion_and_succeeds():
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])
    calls: list[str] = []

    async def call():
        calls.append(provider.model)
        if provider.model == _MODEL_FALLBACK_CHAIN[0]:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, model: " + provider.model)
        return "ok"

    result = await provider._with_model_fallback(call)
    assert result == "ok"
    assert provider.model == _MODEL_FALLBACK_CHAIN[1]
    assert calls == [_MODEL_FALLBACK_CHAIN[0], _MODEL_FALLBACK_CHAIN[1]]


@pytest.mark.asyncio
async def test_switches_to_the_next_model_on_sustained_high_demand_and_succeeds():
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])

    async def call():
        if provider.model == _MODEL_FALLBACK_CHAIN[0]:
            raise RuntimeError("503 UNAVAILABLE: high demand")
        return "ok"

    result = await provider._with_model_fallback(call)
    assert result == "ok"
    assert provider.model == _MODEL_FALLBACK_CHAIN[1]


@pytest.mark.asyncio
async def test_a_request_specific_error_is_never_treated_as_model_unavailability():
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])

    async def call():
        raise ValueError("Gemini returned no function call.")

    with pytest.raises(ValueError, match="no function call"):
        await provider._with_model_fallback(call)
    assert provider.model == _MODEL_FALLBACK_CHAIN[0]  # never switched


@pytest.mark.asyncio
async def test_raises_once_every_fallback_model_is_also_exhausted():
    """With every model actually cooled down, the fallback chain itself has
    nowhere left to go and gives up immediately regardless of the switch
    cap below."""
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])
    now = time.monotonic()
    for model in _MODEL_FALLBACK_CHAIN:
        provider._unavailable_until[model] = now + 300  # all on a long cooldown

    async def call():
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, model: " + provider.model)

    with pytest.raises(RuntimeError, match="429"):
        await provider._with_model_fallback(call)
    # No candidate was ever eligible, so not even one switch happened.
    assert provider.model == _MODEL_FALLBACK_CHAIN[0]


@pytest.mark.asyncio
async def test_fallback_stops_after_max_switches_even_with_models_still_eligible():
    """Latency guarantee, not just resilience: a systemic outage that fails
    every model in turn must not cascade through the whole chain — each
    attempt costs a full _REQUEST_TIMEOUT_SECONDS, and unbounded cascading
    was exactly what let a single decision run for minutes live."""
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])
    attempts: list[str] = []

    async def call():
        attempts.append(provider.model)
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, model: " + provider.model)

    with pytest.raises(RuntimeError, match="429"):
        await provider._with_model_fallback(call)
    # 1 initial attempt + _MAX_FALLBACK_SWITCHES retries on other models —
    # never the full chain, even though every other model was eligible.
    assert len(attempts) == provider._MAX_FALLBACK_SWITCHES + 1
    assert len(set(attempts)) == provider._MAX_FALLBACK_SWITCHES + 1  # each a distinct model


def test_cooldown_seconds_uses_the_apis_own_retry_delay_when_present():
    # Reproduces the live error text verbatim (trimmed).
    exc = RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {... 'details': [{'@type': "
        "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '24.8s'}]}}"
    )
    assert _cooldown_seconds(exc) == 24.8


def test_cooldown_seconds_caps_an_unreasonably_long_retry_delay():
    exc = RuntimeError("429 ... 'retryDelay': '3600s' ...")
    assert _cooldown_seconds(exc) == _MAX_COOLDOWN_SECONDS


def test_cooldown_seconds_falls_back_to_a_default_when_absent():
    assert _cooldown_seconds(RuntimeError("503 UNAVAILABLE: high demand")) == _DEFAULT_COOLDOWN_SECONDS


def test_a_model_becomes_eligible_again_after_its_cooldown_expires():
    """This is the whole point of a cooldown over a permanent blacklist —
    reproduces a live finding: extended testing eventually marked every
    model in the chain "unavailable" with no expiry, permanently stranding
    the provider for the rest of the process even though several of those
    failures were short-lived capacity blips, not the day-long quota."""
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[1])
    # Still cooling down (5 minutes out) -> not offered as a candidate.
    provider._unavailable_until[_MODEL_FALLBACK_CHAIN[0]] = time.monotonic() + 300
    assert provider._next_fallback_model() != _MODEL_FALLBACK_CHAIN[0]

    # Its cooldown has now elapsed -> eligible again.
    provider._unavailable_until[_MODEL_FALLBACK_CHAIN[0]] = time.monotonic() - 1
    assert provider._next_fallback_model() == _MODEL_FALLBACK_CHAIN[0]


@pytest.mark.asyncio
async def test_a_model_switch_sticks_for_the_next_call_on_the_same_instance():
    """The LLMRegistry caches one provider instance per (provider, model)
    key, so this instance-level state is what makes the switch survive
    across separate requests in the same conversation — pinning that here
    since it's the whole reason this isn't tracked per-call instead."""
    provider = GeminiLLMProvider("fake-key", _MODEL_FALLBACK_CHAIN[0])

    async def first_call():
        if provider.model == _MODEL_FALLBACK_CHAIN[0]:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota, model: " + provider.model)
        return "ok"

    await provider._with_model_fallback(first_call)
    assert provider.model == _MODEL_FALLBACK_CHAIN[1]

    # A later, unrelated call on the SAME instance starts from the model
    # it already switched to — never retries the known-bad one.
    seen: list[str] = []

    async def second_call():
        seen.append(provider.model)
        return "ok"

    result = await provider._with_model_fallback(second_call)
    assert result == "ok"
    assert seen == [_MODEL_FALLBACK_CHAIN[1]]


# --- _clean_schema: anyOf (Optional field) flattening -----------------------
#
# Reproduces a live finding: a DBA asked the agent about a database without
# naming an environment in an earlier turn, then wrote "dev" when the agent
# proposed a target — "dev" is not a valid Environment value, and the model
# kept proposing it because nothing ever told it the field was constrained.
# Root cause: Pydantic v2 represents `X | None` as
# `anyOf: [<X's schema>, {"type": "null"}]`, and _clean_schema used to just
# drop `anyOf` (not in its allow-list), silently discarding the type/enum
# along with it — so EVERY Optional field in this codebase's schemas
# (IntentExtraction.database_hint/environment_hint/instance_hint) reached
# Gemini as an unconstrained `{}`.


def test_clean_schema_flattens_an_optional_field_to_nullable():
    raw = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "X"}
    cleaned = _clean_schema(raw)
    assert cleaned == {"type": "string", "nullable": True}


def test_clean_schema_preserves_the_enum_on_an_optional_literal_field():
    raw = {
        "anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}],
        "default": None,
    }
    cleaned = _clean_schema(raw)
    assert cleaned == {"type": "string", "enum": ["a", "b"], "nullable": True}


def test_clean_schema_on_the_real_intent_extraction_schema_keeps_every_field_typed():
    """No Optional field is left as an unconstrained {} after cleaning —
    the actual bug: every one of these used to lose its type entirely."""
    cleaned = _clean_schema(IntentExtraction.model_json_schema())
    props = cleaned["properties"]
    for field in ("database_hint", "environment_hint", "instance_hint"):
        assert props[field].get("type") == "string", f"{field} lost its type"
        assert props[field].get("nullable") is True

    # And specifically: environment_hint is now a real enum, not a free
    # string a model could fill with "dev"/"prod"/anything else.
    assert props["environment_hint"]["enum"] == ["development", "uat", "production"]


def test_environment_hint_rejects_an_abbreviation():
    """Pins the Pydantic-level guarantee behind the schema fix above:
    IntentExtraction itself refuses "dev" — it was never a valid
    Environment value, only ever accepted because nothing constrained it."""
    with pytest.raises(Exception, match="development.*uat.*production|literal_error"):
        IntentExtraction(is_dba_task=True, environment_hint="dev")
    # The real values still work.
    for value in ("development", "uat", "production"):
        IntentExtraction(is_dba_task=True, environment_hint=value)
