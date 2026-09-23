"""FastAPI dependencies for the Gateway service.

Two things happen on every request before any business logic runs:
  1. the caller must present a valid service token (only the Agent service
     is issued one, scoped to the Gateway's audience) — spec §31.
  2. the human identity is independently re-resolved from the raw channel
     account the caller names, via this service's own `IdentityProvider` —
     never accepted as a pre-verified blob from the caller (spec §3, §62).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from numi.common.models.failures import NumiError
from numi.common.models.identity import VerifiedIdentity
from numi.gateway.api.state import GatewayState


def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    state: GatewayState = request.app.state.gateway
    async with state.db.session() as session:
        yield session


async def require_agent_service_token(
    request: Request, x_service_token: str | None = Header(default=None)
) -> None:
    state: GatewayState = request.app.state.gateway
    if not x_service_token:
        raise HTTPException(status_code=401, detail="Missing service token.")
    try:
        state.service_token_verifier.verify(x_service_token, expected_audience="numi-gateway")
    except NumiError as exc:
        raise HTTPException(status_code=401, detail=exc.detail) from exc


async def resolve_identity(
    state: GatewayState, channel: str, channel_account_id: str
) -> VerifiedIdentity:
    identity = await state.identity_provider.resolve_by_external_account(
        channel, channel_account_id
    )
    if identity is None:
        raise HTTPException(status_code=401, detail="Unable to verify identity for this account.")
    return identity
