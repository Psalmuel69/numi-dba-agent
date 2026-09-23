"""Gateway -> Execution Service client (spec §18, §31).

The Gateway is the only caller of the Execution Service. Every call carries
a freshly-minted, short-lived, audience-scoped service token — never a
static shared secret passed as a plain header.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from numi.common.models.catalog import ServerCatalog
from numi.common.models.execution import DiscoveryRequest, ExecutionRequest, ExecutionResult
from numi.common.service_auth import ServiceTokenIssuer


class ExecutionClient(ABC):
    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...

    @abstractmethod
    async def discover(self, request: DiscoveryRequest) -> ServerCatalog: ...


class HttpExecutionClient(ExecutionClient):
    def __init__(
        self,
        base_url: str,
        issuer: ServiceTokenIssuer,
        audience: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url
        self._issuer = issuer
        self._audience = audience
        self._transport = transport

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        token = self._issuer.issue(service_name="gateway", audience=self._audience)
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=request.max_execution_time + 5
        ) as client:
            response = await client.post(
                "/v1/execute",
                json=request.model_dump(mode="json"),
                headers={"X-Service-Token": token},
            )
            response.raise_for_status()
            return ExecutionResult.model_validate(response.json())

    async def discover(self, request: DiscoveryRequest) -> ServerCatalog:
        token = self._issuer.issue(service_name="gateway", audience=self._audience)
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=300
        ) as client:
            response = await client.post(
                "/v1/discover",
                json=request.model_dump(mode="json"),
                headers={"X-Service-Token": token},
            )
            response.raise_for_status()
            return ServerCatalog.model_validate(response.json())


class InProcessExecutionClient(ExecutionClient):
    """Calls an `ExecutionService` instance directly with no HTTP hop.

    Used by integration tests (and could back a single-process/dev-mode
    deployment) — it exercises the exact same `ExecutionService.execute`
    code path as the real HTTP boundary, just without a network round trip.
    """

    def __init__(self, execution_service):
        self._service = execution_service

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return await self._service.execute(request)

    async def discover(self, request: DiscoveryRequest) -> ServerCatalog:
        creds = await self._service._credentials.get_credentials(request.server_id)
        from numi.execution.discovery.engine import run_discovery

        return await run_discovery(
            server_id=request.server_id,
            platform=request.platform,
            credentials=creds,
            max_objects_per_database=request.max_objects_per_database,
        )
