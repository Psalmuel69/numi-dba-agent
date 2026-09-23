"""GET /v1/audit/{id} (spec §24, §57).

Read-only. There is deliberately no PUT/PATCH/DELETE route anywhere in this
router or in `numi.gateway.domain.audit` — the audit log is append-only for
the life of the platform.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from numi.gateway.api.deps import get_session, require_agent_service_token
from numi.gateway.infrastructure.db.models import AuditEventRecord

router = APIRouter(prefix="/v1/audit", tags=["audit"], dependencies=[Depends(require_agent_service_token)])


@router.get("/{audit_event_id}")
async def get_audit_event(
    audit_event_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    record = await session.get(AuditEventRecord, audit_event_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit event not found.")
    return {
        "audit_event_id": record.audit_event_id,
        "event_type": record.event_type,
        "correlation_ids": record.correlation_ids,
        "actor_subject_id": record.actor_subject_id,
        "channel": record.channel,
        "tool_id": record.tool_id,
        "tool_version": record.tool_version,
        "target": record.target,
        "arguments_hash": record.arguments_hash,
        "policy_decision": record.policy_decision,
        "risk": record.risk,
        "approval_id": record.approval_id,
        "execution_result": record.execution_result,
        "error_code": record.error_code,
        "duration_ms": record.duration_ms,
        "created_at": record.created_at.isoformat(),
    }
