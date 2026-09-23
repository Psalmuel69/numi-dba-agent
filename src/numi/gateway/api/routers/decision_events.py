"""Decision-quality events — durable storage for a handful of degradation
signals, and the rollup query that makes them a reviewable habit rather
than just log lines nobody reads back. See
`gateway.domain.decision_events` for exactly which events and why not all
of them.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query

from numi.common.models.decision_event import DecisionEventCreateRequest
from numi.gateway.api.deps import get_state, require_agent_service_token
from numi.gateway.api.state import GatewayState

router = APIRouter(
    prefix="/v1/decision-events",
    tags=["decision-events"],
    dependencies=[Depends(require_agent_service_token)],
)


@router.post("")
async def create_decision_event(
    body: DecisionEventCreateRequest, state: GatewayState = Depends(get_state)
) -> dict:
    await state.decision_event_store.append(
        event_type=body.event_type,
        conversation_id=body.conversation_id,
        investigation_id=body.investigation_id,
        provider=body.provider,
        model=body.model,
        payload=body.payload,
    )
    return {"status": "ok"}


@router.get("/summary")
async def decision_event_summary(
    since_hours: int = Query(default=24, ge=1, le=24 * 90),
    state: GatewayState = Depends(get_state),
) -> dict:
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=since_hours)
    counts = await state.decision_event_store.counts_by_type(since=since)
    return {"since_hours": since_hours, "counts": counts}
