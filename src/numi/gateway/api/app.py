"""Gateway FastAPI app (spec §57).

This process is the DBA Control Gateway described throughout the spec: the
only thing between the Agent's tool requests and the Execution Service.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from numi.common.config import Settings, get_settings
from numi.common.observability import configure_logging, get_logger
from numi.gateway.api.routers import (
    approvals,
    audit,
    catalog,
    decision_events,
    investigations,
    tool_calls,
    tools,
)
from numi.gateway.api.state import GatewayState
from numi.gateway.domain.discovery import DiscoveryOrchestrator

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, *, execution_transport=None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_production()
    configure_logging("gateway", settings.log_level)

    # Built eagerly (not inside `lifespan`) so tests driving this app via
    # `httpx.ASGITransport` — which does not emit ASGI lifespan events —
    # can reach `app.state.gateway` and call `state.db.create_all()`
    # themselves without needing a running server.
    state = GatewayState.build(settings, execution_transport=execution_transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.control_db_url.startswith("sqlite"):
            # Local dev/test convenience only — real deployments apply the
            # Alembic migrations in migrations/versions instead.
            await state.db.create_all()

        # Optional eager estate crawl (NUMI_DISCOVERY_ON_STARTUP=true).
        # Off by default — the catalog also refreshes lazily on first use
        # and via `/discover`. Never blocks startup; failures are logged.
        discovery_task: asyncio.Task | None = None
        if settings.discovery_on_startup:
            discovery_task = asyncio.create_task(_startup_discovery(state))
        yield
        if discovery_task is not None:
            discovery_task.cancel()
        await state.db.dispose()

    app = FastAPI(title="Numi DBA Control Gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = state

    app.include_router(tool_calls.router)
    app.include_router(tools.router)
    app.include_router(approvals.router)
    app.include_router(investigations.router)
    app.include_router(decision_events.router)
    app.include_router(audit.router)
    app.include_router(catalog.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict:
        return {"status": "ready"}

    return app


async def _startup_discovery(state: GatewayState) -> None:
    try:
        orchestrator = DiscoveryOrchestrator(
            registry=state.server_registry,
            catalog_store=state.catalog_store,
            execution_client=state.execution_client,
            settings=state.settings,
        )
        results = await orchestrator.refresh_all()
        logger.info("startup_discovery_complete", results=results)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup_discovery_failed", error=str(exc))


app = create_app()
