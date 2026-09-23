"""Discovery — reading a server's DBA-relevant *metadata* into the catalog.

Runs in the Execution Service (the only component with database access). A
`ServerDiscoverer` enumerates, for one registered server:

  - engine version / edition and instance-level properties/settings
  - every non-system database: state, size, options
  - each database's schemas and objects (tables / views / indexes / procs)
    with catalog row estimates and sizes — **never row data**
  - available and installed extensions

The diagnostic login only needs read access to catalog / DMV / stats views
(VIEW SERVER STATE + VIEW DEFINITION on SQL Server; the `pg_monitor` role on
PostgreSQL). It should NOT have SELECT on user tables — Numi never reads
table contents.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

from numi.common.models.catalog import LeastPrivilegeFinding, ServerCatalog
from numi.common.observability import get_logger
from numi.execution.adapters.base import QueryExecutor
from numi.execution.credentials.provider import DatabaseCredentials

logger = get_logger(__name__)

#: Row cap for the least-privilege scan. The finding only needs "does this
#: login have SELECT on user data, and roughly how much" — enumerating every
#: table on a large instance would turn a cheap check into a crawl of its
#: own. Hitting the cap sets `count_is_lower_bound`, so the number reported
#: is honest about being a floor rather than silently wrong.
LEAST_PRIVILEGE_SCAN_LIMIT = 200

#: How many object names the finding carries, purely to point the DBA at
#: where to look. Names only, never row data.
_SAMPLE_OBJECT_COUNT = 5


class ServerDiscoverer(ABC):
    """One instance per discovery run, bound to a server's credentials."""

    def __init__(self, credentials: DatabaseCredentials, *, max_objects_per_database: int = 5000):
        self._credentials = credentials
        self._max_objects = max_objects_per_database

    @abstractmethod
    async def discover(self, server_id: str) -> ServerCatalog:
        """Produce the full catalog for `server_id`. Best-effort: a database
        the login can't see is skipped, not fatal."""

    @abstractmethod
    async def _check_least_privilege(self, executor: QueryExecutor) -> LeastPrivilegeFinding:
        """Read-only: does the CONNECTED login have SELECT on user data?

        Each engine implements this against its own privilege introspection,
        and every implementation must be a pure read — this reports, it never
        attempts to revoke anything (see `LeastPrivilegeFinding`'s docstring).

        Where the engine offers an *effective*-permission function
        (`has_table_privilege` on PostgreSQL, `HAS_PERMS_BY_NAME` on SQL
        Server) that is preferred over reading grant tables directly: grant
        tables miss privileges arriving via role membership (SQL Server's
        `db_datareader` is the single most common real case and leaves no row
        in `sys.database_permissions`) and via superuser status (a Postgres
        superuser has SELECT on everything while appearing in
        `information_schema.table_privileges` for nothing). Those are exactly
        the over-privileged logins this check exists to catch, so a
        grant-table-only implementation would miss its own main cases.

        MySQL/MariaDB have no per-object effective-permission function, so
        there the three privilege views are unioned instead — global, schema
        and table level — which is why that implementation looks different.
        """


def build_least_privilege_finding(
    rows: list[dict],
    *,
    login: str,
    scope_note: str,
) -> LeastPrivilegeFinding:
    """Shape a privilege-scan result into a finding. Shared by all three
    engines so the counting/sampling/lower-bound logic exists once.

    `rows` is whatever the engine's scan returned: one row per readable user
    object, with `schema_name` / `object_name` columns.
    """
    names: list[str] = []
    for row in rows[:_SAMPLE_OBJECT_COUNT]:
        schema = row.get("schema_name") or ""
        obj = row.get("object_name") or ""
        names.append(f"{schema}.{obj}" if schema else str(obj))
    return LeastPrivilegeFinding(
        checked=True,
        login=login,
        has_user_table_select=bool(rows),
        granted_object_count=len(rows),
        count_is_lower_bound=len(rows) >= LEAST_PRIVILEGE_SCAN_LIMIT,
        sample_objects=names,
        scope_note=scope_note,
    )


async def run_least_privilege_check(
    discoverer: ServerDiscoverer, executor: QueryExecutor, *, server_id: str
) -> LeastPrivilegeFinding:
    """Call a discoverer's check, converting any failure into an
    un-`checked` finding rather than aborting the crawl.

    A privilege view the login can't read is a normal outcome on a
    locked-down server, and the catalog is still perfectly usable without
    this one field — the same best-effort posture the rest of discovery
    already takes for a database it can't enumerate. Crucially this returns
    `checked=False`, never a clean `has_user_table_select=False`: a check
    that never ran must not be reported as a clean bill of health.
    """
    try:
        return await discoverer._check_least_privilege(executor)
    except Exception as exc:  # noqa: BLE001 — never fatal to a crawl
        logger.warning(
            "least_privilege_check_failed",
            server_id=server_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return LeastPrivilegeFinding(checked=False, scope_note="privilege check could not run")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


async def _fetch(executor: QueryExecutor, sql: str, params: dict | None = None) -> list[dict]:
    return await executor.fetch_all(sql, params or {}, timeout=30)
