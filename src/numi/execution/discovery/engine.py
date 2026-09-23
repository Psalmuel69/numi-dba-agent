"""Discovery dispatcher — picks the platform discoverer and runs it."""

from __future__ import annotations

from numi.common.models.catalog import ServerCatalog
from numi.common.models.target import Platform
from numi.common.observability import get_logger
from numi.execution.credentials.provider import DatabaseCredentials
from numi.execution.discovery.base import ServerDiscoverer
from numi.execution.discovery.mysql import MySQLDiscoverer
from numi.execution.discovery.postgresql import PostgreSQLDiscoverer
from numi.execution.discovery.sqlserver import SQLServerDiscoverer

logger = get_logger(__name__)

_DISCOVERERS: dict[Platform, type[ServerDiscoverer]] = {
    Platform.SQLSERVER: SQLServerDiscoverer,
    Platform.POSTGRESQL: PostgreSQLDiscoverer,
    Platform.MYSQL: MySQLDiscoverer,
    Platform.MARIADB: MySQLDiscoverer,
}


def _discoverer_for(platform: Platform) -> type[ServerDiscoverer]:
    cls = _DISCOVERERS.get(platform)
    if cls is None:
        raise NotImplementedError(
            f"No discoverer for platform '{platform.value}'. Adding an engine means "
            "implementing ServerDiscoverer and registering it here."
        )
    return cls


async def run_discovery(
    *,
    server_id: str,
    platform: Platform,
    credentials: DatabaseCredentials,
    max_objects_per_database: int = 5000,
) -> ServerCatalog:
    discoverer = _discoverer_for(platform)(
        credentials, max_objects_per_database=max_objects_per_database
    )
    try:
        return await discoverer.discover(server_id)
    except Exception as exc:  # noqa: BLE001 — a connection-level failure reaching a
        # target we don't control (server offline, credentials revoked, network
        # partition) — verified live against a genuinely-unreachable dev server,
        # where it surfaced as an unhandled 500 with a raw driver traceback instead
        # of a clean result. Mirrors ExecutionService.execute()'s same posture
        # (never leak internals in the response, log them server-side); the caller
        # already treats a failed refresh as non-fatal (see
        # gateway.domain.discovery's lazy_discovery_failed), so an empty catalog
        # with a warning is a normal outcome here, not a special case to invent.
        logger.error(
            "discovery_failed",
            server_id=server_id,
            platform=platform.value,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return ServerCatalog(
            server_id=server_id,
            warnings=["Could not connect to the server. See server-side logs for detail."],
        )
