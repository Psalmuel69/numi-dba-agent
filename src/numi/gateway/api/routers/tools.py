"""GET /v1/tools, GET /v1/tools/{tool_id} (spec §57).

Dynamic tool exposure (spec §36) is a UX convenience only: passing
`channel`/`channel_account_id` filters the listing to what that role could
even attempt, but the actual security boundary is always the Gateway
pipeline in `tool_call_handler`, never this listing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.tool import ToolDefinition
from numi.gateway.api.deps import get_state, require_agent_service_token, resolve_identity
from numi.gateway.api.state import GatewayState

router = APIRouter(prefix="/v1/tools", tags=["tools"], dependencies=[Depends(require_agent_service_token)])


@router.get("", response_model=list[ToolDefinition])
async def list_tools(
    channel: str | None = Query(default=None),
    channel_account_id: str | None = Query(default=None),
    state: GatewayState = Depends(get_state),
) -> list[ToolDefinition]:
    if channel and channel_account_id:
        identity = await resolve_identity(state, channel, channel_account_id)
        role = identity.highest_role()
        if role is None:
            return []
        return state.tool_registry.tools_for_role(role)
    return state.tool_registry.all()


@router.get("/{tool_id}", response_model=ToolDefinition)
async def get_tool(tool_id: str, state: GatewayState = Depends(get_state)) -> ToolDefinition:
    try:
        return state.tool_registry.get(tool_id)
    except NumiError as exc:
        status = 404 if exc.code == FailureCode.TOOL_NOT_FOUND else 409
        raise HTTPException(status_code=status, detail=exc.detail) from exc
