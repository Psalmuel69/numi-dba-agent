"""Generic fixed-window counting backends — the shared primitive behind
both the Gateway's per-key rate limiting (`gateway.domain.rate_limiter
.RateLimiter`) and the Agent's alert-triggered-investigation cooldown
(`agent.alert_trigger`): "has this key been used more than N times in the
last W seconds?" is the same question either way, just with N=1 for a
cooldown. Two backends satisfy the same interface: an in-memory fixed-window
counter for tests/single-instance local dev, and Redis for anything running
more than one replica — a counter that only counts within one process is
silently wrong the moment a second instance exists, which is exactly the
bug `RedisRateLimitBackend` exists to avoid (see gateway/api/state.py's
`_build_rate_limit_backend`, the first caller this was written for).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class RateLimitBackend(ABC):
    @abstractmethod
    async def increment_and_check(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Returns True if the call is within budget (and records it),
        False if the limit has been exceeded."""


class InMemoryRateLimitBackend(RateLimitBackend):
    def __init__(self) -> None:
        self._buckets: dict[str, tuple[int, float]] = {}  # key -> (count, window_start)

    async def increment_and_check(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.time()
        count, window_start = self._buckets.get(key, (0, now))
        if now - window_start >= window_seconds:
            count, window_start = 0, now
        count += 1
        self._buckets[key] = (count, window_start)
        return count <= limit


class RedisRateLimitBackend(RateLimitBackend):
    """Production backend. Uses a simple INCR + EXPIRE fixed window, which is
    sufficient given the coarse per-minute budgets configured here."""

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    async def increment_and_check(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        count, _ = await pipe.execute()
        return int(count) <= limit
