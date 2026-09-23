"""Rate limit backends (spec §29) — both the in-memory fixed-window counter
(tests/single-instance dev) and the Redis-backed one required for any
multi-instance deployment, plus the Gateway's choice between them."""

from __future__ import annotations

import pytest

from numi.common.config import Settings
from numi.gateway.api.state import _build_rate_limit_backend
from numi.gateway.domain.rate_limiter import (
    InMemoryRateLimitBackend,
    RedisRateLimitBackend,
)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# InMemoryRateLimitBackend
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_in_memory_backend_allows_up_to_the_limit():
    backend = InMemoryRateLimitBackend()
    for _ in range(5):
        assert await backend.increment_and_check("k", limit=5) is True


@pytest.mark.asyncio
async def test_in_memory_backend_blocks_once_the_limit_is_exceeded():
    backend = InMemoryRateLimitBackend()
    for _ in range(5):
        await backend.increment_and_check("k", limit=5)
    assert await backend.increment_and_check("k", limit=5) is False


@pytest.mark.asyncio
async def test_in_memory_backend_keys_are_independent():
    backend = InMemoryRateLimitBackend()
    for _ in range(5):
        await backend.increment_and_check("k1", limit=5)
    # A different key has its own budget, unaffected by k1 being exhausted.
    assert await backend.increment_and_check("k2", limit=5) is True


@pytest.mark.asyncio
async def test_in_memory_backend_resets_after_the_window_elapses(monkeypatch):
    backend = InMemoryRateLimitBackend()
    now = [1000.0]
    monkeypatch.setattr("numi.common.rate_limit_backend.time.time", lambda: now[0])

    for _ in range(5):
        await backend.increment_and_check("k", limit=5, window_seconds=60)
    assert await backend.increment_and_check("k", limit=5, window_seconds=60) is False

    now[0] += 61  # past the window
    assert await backend.increment_and_check("k", limit=5, window_seconds=60) is True


# --------------------------------------------------------------------------- #
# RedisRateLimitBackend — a fake pipeline stands in for a real Redis server.
# --------------------------------------------------------------------------- #


class _FakePipeline:
    def __init__(self, counters: dict[str, int]):
        self._counters = counters
        self._queued_key: str | None = None

    def incr(self, key: str):
        self._queued_key = key
        return self

    def expire(self, key: str, seconds: int):
        return self

    async def execute(self):
        assert self._queued_key is not None
        self._counters[self._queued_key] = self._counters.get(self._queued_key, 0) + 1
        return [self._counters[self._queued_key], True]


class _FakeRedis:
    def __init__(self):
        self.counters: dict[str, int] = {}

    def pipeline(self):
        return _FakePipeline(self.counters)


@pytest.mark.asyncio
async def test_redis_backend_allows_up_to_the_limit():
    backend = RedisRateLimitBackend(_FakeRedis())
    for _ in range(3):
        assert await backend.increment_and_check("k", limit=3) is True


@pytest.mark.asyncio
async def test_redis_backend_blocks_once_the_limit_is_exceeded():
    fake = _FakeRedis()
    backend = RedisRateLimitBackend(fake)
    for _ in range(3):
        await backend.increment_and_check("k", limit=3)
    assert await backend.increment_and_check("k", limit=3) is False


@pytest.mark.asyncio
async def test_redis_backend_shares_counters_across_separate_instances():
    """The whole point of the Redis backend: two `RedisRateLimitBackend`
    objects (standing in for two Gateway replicas) pointed at the same
    client share one counter, unlike two `InMemoryRateLimitBackend`s."""
    fake = _FakeRedis()
    replica_a = RedisRateLimitBackend(fake)
    replica_b = RedisRateLimitBackend(fake)

    for _ in range(3):
        await replica_a.increment_and_check("k", limit=3)
    # replica_b sees replica_a's usage against the same shared key.
    assert await replica_b.increment_and_check("k", limit=3) is False


# --------------------------------------------------------------------------- #
# GatewayState's choice of backend
# --------------------------------------------------------------------------- #


def test_gateway_defaults_to_in_memory_backend():
    backend = _build_rate_limit_backend(_settings())
    assert isinstance(backend, InMemoryRateLimitBackend)


def test_gateway_uses_redis_backend_when_configured():
    backend = _build_rate_limit_backend(_settings(rate_limit_backend="redis"))
    assert isinstance(backend, RedisRateLimitBackend)
