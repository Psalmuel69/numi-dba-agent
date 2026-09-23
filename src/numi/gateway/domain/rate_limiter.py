"""Rate limiting (spec §29).

Enforced by the Gateway before policy evaluation, at multiple keys
(user/conversation/tool/database/environment), with separate, much stricter
budgets for write and critical operations. `RateLimiter` below is the
Gateway-specific multi-key checker; the two backends it runs on
(`InMemoryRateLimitBackend` for tests/local dev, `RedisRateLimitBackend` for
multi-instance production deployments) are generic enough that
`agent.alert_trigger`'s investigation cooldown reuses them too, so they live
in `common.rate_limit_backend` — re-exported here unchanged so every
existing import of this module keeps working.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from numi.common.models.failures import FailureCode, NumiError
from numi.common.rate_limit_backend import (
    InMemoryRateLimitBackend,
    RateLimitBackend,
    RedisRateLimitBackend,
)

__all__ = [
    "InMemoryRateLimitBackend",
    "RateLimitBackend",
    "RateLimiter",
    "RedisRateLimitBackend",
]


class RateLimiter:
    def __init__(self, config_path: str | Path, backend: RateLimitBackend):
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._limits = raw["rate_limits"]
        self._backend = backend

    def _tier_for(self, operation_type: str, requires_approval: bool) -> str:
        if operation_type == "PRIVILEGED" or requires_approval:
            return "critical"
        if operation_type == "WRITE":
            return "write"
        return "read"

    async def check(
        self,
        *,
        operation_type: str,
        requires_approval: bool,
        user_subject_id: str,
        conversation_id: str,
        tool_id: str,
        database_id: str,
        environment: str,
    ) -> None:
        tier = self._tier_for(operation_type, requires_approval)
        limits = self._limits[tier]
        checks = [
            (f"rl:{tier}:user:{user_subject_id}", limits["per_user_per_minute"]),
            (f"rl:{tier}:conv:{conversation_id}", limits["per_conversation_per_minute"]),
            (f"rl:{tier}:tool:{tool_id}", limits["per_tool_per_minute"]),
            (f"rl:{tier}:db:{database_id}", limits["per_database_per_minute"]),
            (f"rl:{tier}:env:{environment}", limits["per_environment_per_minute"]),
        ]
        for key, limit in checks:
            ok = await self._backend.increment_and_check(key, limit)
            if not ok:
                raise NumiError(
                    FailureCode.RATE_LIMITED,
                    "Rate limit exceeded — please slow down and try again shortly.",
                )
