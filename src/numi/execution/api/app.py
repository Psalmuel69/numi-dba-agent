"""Execution Service FastAPI app (spec §18, §31).

Exposes two privileged endpoints — `/v1/execute` (run one typed operation)
and `/v1/discover` (crawl a server's metadata into a catalog) — and both
are only ever reachable with a valid, audience-scoped service token minted
by the Gateway. There is no route, header, or flag that lets the Agent or a
channel adapter call this service directly.
"""

from __future__ import annotations

import asyncio
import sys

from fastapi import Depends, FastAPI, Header, HTTPException

from numi.common.config import Settings, get_settings
from numi.common.models.catalog import ServerCatalog
from numi.common.models.execution import DiscoveryRequest, ExecutionRequest, ExecutionResult
from numi.common.observability import configure_logging, get_logger
from numi.common.service_auth import ServiceTokenVerifier
from numi.execution.credentials.provider import build_credential_provider
from numi.execution.discovery.engine import run_discovery
from numi.execution.service import ExecutionService

logger = get_logger(__name__)

# psycopg's async connection mode (used by PostgreSQLQueryExecutor) requires
# a selector-based event loop; Windows' asyncio default (ProactorEventLoop)
# raises `InterfaceError` on connect. Setting the policy here is necessary
# for any in-process caller (tests, a REPL) but NOT sufficient for
# `uvicorn numi.execution.api.app:app` — uvicorn's `asyncio.run()` creates
# its event loop before it imports this module, so the policy must also be
# set earlier than that; see `python -m numi.execution` (`__main__.py`),
# which is the Windows-safe way to start this service directly.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def create_app(settings: Settings | None = None, *, adapter_factory=None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_production()
    configure_logging("execution-service", settings.log_level)

    app = FastAPI(title="Numi Execution Service", version="0.1.0")
    verifier = ServiceTokenVerifier(settings.service_jwt_secret, settings.service_jwt_issuer)
    credential_provider = build_credential_provider(settings)
    service = ExecutionService(settings, credential_provider, adapter_factory=adapter_factory)

    async def require_gateway_service_token(
        x_service_token: str | None = Header(default=None),
    ) -> None:
        if not x_service_token:
            raise HTTPException(status_code=401, detail="Missing service token.")
        try:
            verifier.verify(
                x_service_token, expected_audience=settings.execution_service_token_audience
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=401, detail="Invalid service token.") from exc

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict:
        return {"status": "ready"}

    @app.post(
        "/v1/execute",
        response_model=ExecutionResult,
        dependencies=[Depends(require_gateway_service_token)],
    )
    async def execute(request: ExecutionRequest) -> ExecutionResult:
        logger.info(
            "execution_request_received",
            tool_id=request.tool_id,
            server_id=request.server_id,
            execution_id=request.execution_id,
        )
        result = await service.execute(request)
        logger.info(
            "execution_request_completed",
            execution_id=request.execution_id,
            success=result.success,
            error_code=result.error_code,
        )
        return result

    @app.post(
        "/v1/discover",
        response_model=ServerCatalog,
        dependencies=[Depends(require_gateway_service_token)],
    )
    async def discover(request: DiscoveryRequest) -> ServerCatalog:
        logger.info("discovery_started", server_id=request.server_id, platform=request.platform.value)
        if adapter_factory is not None:
            # Test mode — no real connection; hand back an empty catalog.
            return ServerCatalog(server_id=request.server_id)
        creds = await credential_provider.get_credentials(request.server_id)
        catalog = await run_discovery(
            server_id=request.server_id,
            platform=request.platform,
            credentials=creds,
            max_objects_per_database=request.max_objects_per_database,
        )
        logger.info(
            "discovery_completed",
            server_id=request.server_id,
            databases=len(catalog.databases),
            warnings=len(catalog.warnings),
        )
        return catalog

    return app


app = create_app()
