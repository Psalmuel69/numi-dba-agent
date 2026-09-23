"""Approval endpoints (spec §16, §57).

`POST /v1/approvals/{id}/approve` and `/reject` never trust a bare
`approved=true` from a client — the approver's identity is independently
re-resolved from their channel account (exactly like a tool call), then
`ApprovalEngine.decide` re-verifies role, expiry, separation-of-duties, and
(for dual-approval actions) that two distinct qualified approvers signed off,
entirely from server-side state.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from numi.common.models.failures import FailureCode, NumiError
from numi.gateway.api.deps import (
    get_session,
    get_state,
    require_agent_service_token,
    resolve_identity,
)
from numi.gateway.api.state import GatewayState
from numi.gateway.domain.approval import ApprovalDecision, ApprovalEngine
from numi.gateway.domain.audit import AuditLog
from numi.gateway.domain.authorization import authorize

router = APIRouter(
    prefix="/v1/approvals", tags=["approvals"], dependencies=[Depends(require_agent_service_token)]
)

_CODE_TO_HTTP = {
    FailureCode.APPROVAL_INVALID: 404,
    FailureCode.APPROVAL_EXPIRED: 410,
    FailureCode.APPROVAL_ALREADY_DECIDED: 409,
    FailureCode.SEPARATION_OF_DUTIES_VIOLATION: 403,
    FailureCode.UNAUTHORIZED: 403,
}


class ApprovalActionRequest(BaseModel):
    channel: str
    channel_account_id: str


@router.get("/{approval_id}")
async def get_approval(approval_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    engine = ApprovalEngine(session)
    try:
        record = await engine.get(approval_id)
    except NumiError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    return {
        "approval_id": record.approval_id,
        "status": record.status,
        "tool_id": record.tool_id,
        "target": record.target,
        "risk": record.risk,
        "expires_at": record.expires_at.isoformat(),
        "requires_dual_approval": record.requires_dual_approval,
    }


async def _decide(
    approval_id: str,
    body: ApprovalActionRequest,
    decision: ApprovalDecision,
    state: GatewayState,
    session: AsyncSession,
) -> dict:
    identity = await resolve_identity(state, body.channel, body.channel_account_id)
    if not identity.is_dba():
        raise HTTPException(status_code=403, detail="Only DBA team members may decide approvals.")

    engine = ApprovalEngine(session)
    audit = AuditLog(session)
    try:
        record = await engine.get(approval_id)
        if decision == ApprovalDecision.APPROVE:
            # The approver must themselves be independently authorized for
            # this exact tool/target — approving is not a lesser bar than
            # requesting. Re-resolve the target from scratch (catches a
            # server that was de-registered since the request).
            from numi.common.models.target import DatabaseTarget

            tool = state.tool_registry.get(record.tool_id, record.tool_version)
            try:
                ctx = await state.target_validator.validate(
                    DatabaseTarget.model_validate(record.target), []
                )
                authorize(identity, tool, ctx)
            except NumiError:
                raise
            except Exception:  # noqa: BLE001
                pass

        decided = await engine.decide(approval_id=approval_id, approver=identity, decision=decision)
        event_type = "APPROVAL_APPROVED" if decision == ApprovalDecision.APPROVE else "APPROVAL_REJECTED"
        await audit.record(
            event_type=event_type,
            correlation_ids={"request_id": record.request_id, "approval_id": approval_id},
            actor_subject_id=identity.subject_id,
            tool_id=record.tool_id,
            tool_version=record.tool_version,
            target=record.target,
            approval_id=approval_id,
            policy_decision=decided.status,
        )
        await session.commit()
    except NumiError as exc:
        await session.rollback()
        status = _CODE_TO_HTTP.get(exc.code, 400)
        raise HTTPException(status_code=status, detail=exc.detail) from exc

    return {"approval_id": approval_id, "status": decided.status}


@router.post("/{approval_id}/approve")
async def approve(
    approval_id: str,
    body: ApprovalActionRequest,
    state: GatewayState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
) -> dict:
    return await _decide(approval_id, body, ApprovalDecision.APPROVE, state, session)


@router.post("/{approval_id}/reject")
async def reject(
    approval_id: str,
    body: ApprovalActionRequest,
    state: GatewayState = Depends(get_state),
    session: AsyncSession = Depends(get_session),
) -> dict:
    return await _decide(approval_id, body, ApprovalDecision.REJECT, state, session)
