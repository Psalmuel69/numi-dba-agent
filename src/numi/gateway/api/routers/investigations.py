"""Investigation persistence and recall (spec §25, §57).

Read-write projection of investigation state, maintained by the Agent via
`numi.gateway.domain.investigation_store` — the Gateway itself never
fabricates evidence/findings, it only persists and serves back what the
Agent recorded.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query

from numi.common.models.investigation import (
    InvestigationCreateRequest,
    InvestigationEventCreateRequest,
    InvestigationUpdateRequest,
)
from numi.gateway.api.deps import get_state, require_agent_service_token
from numi.gateway.api.state import GatewayState
from numi.gateway.infrastructure.db.models import InvestigationRecord

router = APIRouter(
    prefix="/v1/investigations", tags=["investigations"], dependencies=[Depends(require_agent_service_token)]
)


def _serialize(record: InvestigationRecord) -> dict:
    return {
        "investigation_id": record.investigation_id,
        "conversation_id": record.conversation_id,
        "server_id": record.server_id,
        "playbook_id": record.playbook_id,
        "environment": record.environment,
        "target": record.target,
        "problem": record.problem,
        "status": record.status,
        "evidence": record.evidence,
        "hypotheses": record.hypotheses,
        "findings": record.findings,
        "recommendations": record.recommendations,
        "actions": record.actions,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


@router.get("/memory/{server_id}")
async def get_investigation_memory(
    server_id: str,
    exclude_investigation_id: str | None = Query(default=None),
    limit: int = Query(default=3, ge=0, le=20),
    state: GatewayState = Depends(get_state),
) -> list[dict]:
    entries = await state.investigation_memory.recall(
        server_id, exclude_investigation_id=exclude_investigation_id, limit=limit
    )
    return [e.model_dump(mode="json") for e in entries]


@router.get("/correlate")
async def correlate_investigations(
    playbook_id: str = Query(...),
    environment: str | None = Query(default=None),
    exclude_server_id: str | None = Query(default=None),
    limit: int = Query(default=5, ge=0, le=20),
    state: GatewayState = Depends(get_state),
) -> list[dict]:
    if not state.settings.cross_server_correlation_enabled:
        return []
    since = dt.datetime.now(dt.UTC) - dt.timedelta(
        days=state.settings.cross_server_correlation_lookback_days
    )
    entries = await state.investigation_memory.correlate(
        playbook_id=playbook_id,
        environment=environment,
        exclude_server_id=exclude_server_id,
        since=since,
        limit=limit,
    )
    return [e.model_dump(mode="json") for e in entries]


@router.get("/{investigation_id}")
async def get_investigation(
    investigation_id: str, state: GatewayState = Depends(get_state)
) -> dict:
    record = await state.investigation_store.get(investigation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return _serialize(record)


@router.post("")
async def create_investigation(
    body: InvestigationCreateRequest, state: GatewayState = Depends(get_state)
) -> dict:
    record = await state.investigation_store.create(**body.model_dump())
    return _serialize(record)


@router.patch("/{investigation_id}")
async def update_investigation(
    investigation_id: str,
    body: InvestigationUpdateRequest,
    state: GatewayState = Depends(get_state),
) -> dict:
    record = await state.investigation_store.update(investigation_id, **body.model_dump())
    if record is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return _serialize(record)


@router.post("/{investigation_id}/events")
async def append_investigation_event(
    investigation_id: str,
    body: InvestigationEventCreateRequest,
    state: GatewayState = Depends(get_state),
) -> dict:
    await state.investigation_store.append_event(investigation_id, body.event_type, body.payload)
    return {"status": "ok"}
