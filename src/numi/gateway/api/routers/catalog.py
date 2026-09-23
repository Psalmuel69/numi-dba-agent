"""Catalog + discovery endpoints (spec §11, §39).

  GET  /v1/catalog/servers            — registered servers + catalog summary
  GET  /v1/catalog/servers/{id}       — full discovered catalog for one server
  POST /v1/catalog/refresh            — (DBA_MANAGER) re-run discovery
  POST /v1/catalog/refresh/{id}       — (DBA_MANAGER) re-run for one server
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from numi.common.models.identity import DBARole
from numi.common.observability import get_logger
from numi.gateway.api.deps import get_state, require_agent_service_token, resolve_identity
from numi.gateway.api.state import GatewayState
from numi.gateway.domain.discovery import DiscoveryOrchestrator, clean_discovery_error

logger = get_logger(__name__)

router = APIRouter(
    prefix="/v1/catalog", tags=["catalog"], dependencies=[Depends(require_agent_service_token)]
)


class RefreshRequest(BaseModel):
    channel: str
    channel_account_id: str


def _orchestrator(state: GatewayState) -> DiscoveryOrchestrator:
    return DiscoveryOrchestrator(
        registry=state.server_registry,
        catalog_store=state.catalog_store,
        execution_client=state.execution_client,
        settings=state.settings,
    )


@router.get("/servers")
async def list_servers(state: GatewayState = Depends(get_state)) -> list[dict]:
    out = []
    for server in state.server_registry.all():
        catalog = await state.catalog_store.get(server.id)
        out.append(
            {
                "id": server.id,
                "platform": server.platform.value,
                "environment": server.environment.value,
                "criticality": server.criticality,
                "allowed_roles": [r.value for r in server.allowed_roles],
                "aliases": server.aliases,
                "status": server.status,
                "catalog": None
                if catalog is None
                else {
                    "discovered_at": catalog.discovered_at.isoformat()
                    if catalog.discovered_at
                    else None,
                    "engine_version": catalog.engine_version,
                    "engine_edition": catalog.engine_edition,
                    "database_count": len(catalog.databases),
                    "databases": catalog.database_names(),
                    "warnings": catalog.warnings,
                },
            }
        )
    return out


@router.get("/servers/{server_id}")
async def get_server_catalog(server_id: str, state: GatewayState = Depends(get_state)) -> dict:
    server = state.server_registry.by_id(server_id)
    if server is None:
        raise HTTPException(status_code=404, detail="No such registered server.")
    catalog = await state.catalog_store.get(server_id)
    return {
        "server": {
            "id": server.id,
            "platform": server.platform.value,
            "environment": server.environment.value,
            "host": server.host,
            "criticality": server.criticality,
        },
        "catalog": catalog.model_dump(mode="json") if catalog else None,
    }


async def _require_manager(state: GatewayState, body: RefreshRequest) -> None:
    identity = await resolve_identity(state, body.channel, body.channel_account_id)
    if DBARole.DBA_MANAGER not in identity.dba_roles:
        raise HTTPException(status_code=403, detail="Discovery refresh requires DBA_MANAGER.")


@router.post("/refresh")
async def refresh_all(body: RefreshRequest, state: GatewayState = Depends(get_state)) -> dict:
    await _require_manager(state, body)
    return await _orchestrator(state).refresh_all()


@router.post("/refresh/{server_id}")
async def refresh_one(
    server_id: str, body: RefreshRequest, state: GatewayState = Depends(get_state)
) -> dict:
    await _require_manager(state, body)
    if state.server_registry.by_id(server_id) is None:
        raise HTTPException(status_code=404, detail=f"No registered server '{server_id}'.")
    try:
        catalog = await _orchestrator(state).refresh_server(server_id)
    except Exception as exc:  # noqa: BLE001 — an unreachable server is a normal outcome here
        # Real exception goes to the log only — never to the DBA (same
        # invariant as DiscoveryOrchestrator.refresh_all).
        logger.warning(
            "catalog_refresh_one_failed",
            server_id=server_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return {
            "server_id": server_id,
            "databases": [],
            "warnings": [],
            "error": clean_discovery_error(exc),
        }
    return {
        "server_id": server_id,
        "databases": catalog.database_names(),
        "warnings": catalog.warnings,
    }
