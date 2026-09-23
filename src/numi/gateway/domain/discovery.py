"""Discovery orchestration (Gateway side).

Asks the Execution Service to crawl each registered server and stores the
resulting catalog. Triggered on Gateway startup (best-effort background),
by `POST /v1/catalog/refresh` (DBA_MANAGER), and lazily the first time a
server with no catalog is targeted by a read-only operation.
"""

from __future__ import annotations

import datetime as dt

import httpx

from numi.common.config import Settings
from numi.common.models.catalog import ServerCatalog
from numi.common.models.execution import DiscoveryRequest
from numi.common.observability import get_logger
from numi.gateway.domain.catalog import CatalogStore
from numi.gateway.domain.servers import ServerRegistry
from numi.gateway.infrastructure.execution_client import ExecutionClient

logger = get_logger(__name__)


def clean_discovery_error(exc: Exception) -> str:
    """Map a discovery failure to a short, DBA-facing message.

    The DBA never sees a stack trace or raw exception text (same invariant
    already enforced for `ToolCallResponse` denials, adapter `FAILED`
    messages, and the Agent's own timeout/self-correction messaging — see
    `ExecutionService.execute`'s `except Exception` branch for the
    established style: log the real exception, return a clean fallback).
    An `httpx.HTTPStatusError`'s own `__str__` bakes in the raw request URL
    and an MDN documentation link, which must never reach a channel.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"execution service returned an error (status {exc.response.status_code})"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "could not reach the execution service"
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return "discovery timed out"
    return f"discovery failed ({type(exc).__name__})"


class DiscoveryOrchestrator:
    def __init__(
        self,
        *,
        registry: ServerRegistry,
        catalog_store: CatalogStore,
        execution_client: ExecutionClient,
        settings: Settings,
    ):
        self._registry = registry
        self._store = catalog_store
        self._execution = execution_client
        self._settings = settings

    async def refresh_server(self, server_id: str) -> ServerCatalog:
        server = self._registry.by_id(server_id)
        if server is None:
            raise LookupError(f"No registered server '{server_id}'.")
        catalog = await self._execution.discover(
            DiscoveryRequest(
                server_id=server.id,
                platform=server.platform,
                max_objects_per_database=self._settings.discovery_max_objects_per_database,
            )
        )
        await self._store.put(catalog)
        logger.info(
            "catalog_refreshed",
            server_id=server_id,
            databases=len(catalog.databases),
            warnings=len(catalog.warnings),
        )
        # A login that can read user data is a real, actionable security
        # finding, not a routine discovery detail — it gets its own
        # WARNING-level record so it shows up in log-based alerting without
        # anyone having to read a catalog. The DBA-facing copy of the same
        # finding goes out through `/catalog <server>`; both render
        # `warning_text()` so they can never disagree.
        finding = catalog.least_privilege
        if finding is not None and finding.warning_text():
            logger.warning(
                "least_privilege_violation",
                server_id=server_id,
                login=finding.login,
                granted_object_count=finding.granted_object_count,
                count_is_lower_bound=finding.count_is_lower_bound,
                scope=finding.scope_note,
            )
        return catalog

    async def refresh_all(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for server in self._registry.all():
            if server.status != "active":
                continue
            try:
                cat = await self.refresh_server(server.id)
                results[server.id] = f"ok ({len(cat.databases)} databases)"
            except Exception as exc:  # noqa: BLE001 — one bad server never blocks the rest
                results[server.id] = f"failed: {clean_discovery_error(exc)}"
                # Real exception goes to the log only — never to the DBA
                # (matches ExecutionService.execute's precedent).
                logger.warning(
                    "catalog_refresh_failed",
                    server_id=server.id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
        return results

    async def ensure_fresh(self, server_id: str) -> None:
        """Lazily (re)discover if the catalog is missing or stale."""
        existing = await self._store.get(server_id)
        if existing is not None and existing.discovered_at is not None:
            age = dt.datetime.now(dt.UTC) - existing.discovered_at.replace(
                tzinfo=existing.discovered_at.tzinfo or dt.UTC
            )
            if age < dt.timedelta(minutes=self._settings.discovery_refresh_minutes):
                return
        try:
            await self.refresh_server(server_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lazy_discovery_failed", server_id=server_id, error=str(exc))
