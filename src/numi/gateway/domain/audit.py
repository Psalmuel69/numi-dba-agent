"""Audit system (spec §24).

Append-only by construction: this module exposes exactly one operation,
`record`, and no update/delete method exists anywhere in its public surface
or in the API routers built on top of it (see gateway.api.routers.audit).
The Agent has no credential or route that could reach this module directly
either — it only ever sees audit data reflected back through read APIs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from numi.gateway.infrastructure.db.models import AuditEventRecord, SecurityEventRecord


def hash_arguments(arguments: dict[str, Any]) -> str:
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(
        self,
        *,
        event_type: str,
        correlation_ids: dict[str, str],
        actor_subject_id: str = "",
        identity_provider: str = "",
        channel: str = "",
        agent_version: str = "",
        tool_id: str = "",
        tool_version: str = "",
        target: dict | None = None,
        arguments: dict | None = None,
        policy_decision: str = "",
        risk: dict | None = None,
        approval_id: str | None = None,
        execution_result: str = "",
        error_code: str | None = None,
        duration_ms: int | None = None,
    ) -> AuditEventRecord:
        record = AuditEventRecord(
            event_type=event_type,
            correlation_ids=correlation_ids,
            actor_subject_id=actor_subject_id,
            identity_provider=identity_provider,
            channel=channel,
            agent_version=agent_version,
            tool_id=tool_id,
            tool_version=tool_version,
            target=target or {},
            arguments_hash=hash_arguments(arguments or {}),
            policy_decision=policy_decision,
            risk=risk or {},
            approval_id=approval_id,
            execution_result=execution_result,
            error_code=error_code,
            duration_ms=duration_ms,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def record_security_event(
        self,
        *,
        event_type: str,
        actor_subject_id: str = "",
        detail: dict | None = None,
        severity: str = "WARNING",
    ) -> SecurityEventRecord:
        record = SecurityEventRecord(
            event_type=event_type,
            actor_subject_id=actor_subject_id,
            detail=detail or {},
            severity=severity,
        )
        self._session.add(record)
        await self._session.flush()
        return record
