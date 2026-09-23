"""MySQL / MariaDB discovery (spec §20/§48).

One connection sees every schema through `information_schema`, so this
enumerates non-system schemas and reads each one's tables / views / indexes
/ routines from the catalog — never row data. "Extensions" for these
engines are server-level plugins/components, reported in
`instance_properties`.
"""

from __future__ import annotations

from numi.common.models.catalog import (
    DiscoveredDatabase,
    DiscoveredObject,
    LeastPrivilegeFinding,
    ServerCatalog,
)
from numi.execution.adapters.base import QueryExecutor
from numi.execution.discovery.base import (
    LEAST_PRIVILEGE_SCAN_LIMIT,
    ServerDiscoverer,
    _fetch,
    _now,
    build_least_privilege_finding,
    run_least_privilege_check,
)

_SYSTEM_SCHEMAS = ("information_schema", "performance_schema", "mysql", "sys")

#: SELECT grants held by the connected login, at all three levels.
#:
#: MySQL/MariaDB have no per-object effective-permission function (no
#: equivalent of PostgreSQL's `has_table_privilege` or SQL Server's
#: `HAS_PERMS_BY_NAME`), so the grant tables themselves are the only source
#: and all three levels must be unioned. Reading only
#: `TABLE_PRIVILEGES` — the obvious single view, and the one the feature
#: request named first — would miss both broader cases entirely:
#: `GRANT SELECT ON *.*` (recorded in USER_PRIVILEGES) and
#: `GRANT SELECT ON appdb.*` (recorded in SCHEMA_PRIVILEGES) leave no row
#: there at all, despite granting strictly MORE access than any per-table
#: grant. A global grant is reported as the single object `*.*`.
#:
#: GRANTEE is stored as `'user'@'host'`, while `CURRENT_USER()` returns
#: `user@host` — hence the REPLACE/CONCAT reshaping rather than a plain
#: equality test. These are the engine's own read-only metadata views;
#: nothing here alters a grant.
_LEAST_PRIVILEGE_SQL = """
    SELECT schema_name, object_name FROM (
        SELECT '*' AS schema_name, '*' AS object_name
          FROM information_schema.USER_PRIVILEGES
         WHERE PRIVILEGE_TYPE = 'SELECT'
           AND GRANTEE = CONCAT("'", REPLACE(CURRENT_USER(), '@', "'@'"), "'")
        UNION ALL
        SELECT TABLE_SCHEMA AS schema_name, '*' AS object_name
          FROM information_schema.SCHEMA_PRIVILEGES
         WHERE PRIVILEGE_TYPE = 'SELECT'
           AND GRANTEE = CONCAT("'", REPLACE(CURRENT_USER(), '@', "'@'"), "'")
           AND TABLE_SCHEMA NOT IN
               ('information_schema', 'performance_schema', 'mysql', 'sys')
        UNION ALL
        SELECT TABLE_SCHEMA AS schema_name, TABLE_NAME AS object_name
          FROM information_schema.TABLE_PRIVILEGES
         WHERE PRIVILEGE_TYPE = 'SELECT'
           AND GRANTEE = CONCAT("'", REPLACE(CURRENT_USER(), '@', "'@'"), "'")
           AND TABLE_SCHEMA NOT IN
               ('information_schema', 'performance_schema', 'mysql', 'sys')
    ) g
    ORDER BY schema_name, object_name
    LIMIT %(limit)s
"""


class MySQLDiscoverer(ServerDiscoverer):
    async def discover(self, server_id: str) -> ServerCatalog:
        from numi.execution.adapters.connections import MySQLQueryExecutor

        ex = MySQLQueryExecutor(self._credentials)
        await ex.connect()
        warnings: list[str] = []
        try:
            ver = (
                await _fetch(
                    ex,
                    "SELECT VERSION() AS version, @@version_comment AS edition",
                )
            )[0]
            plugins = await _fetch(
                ex,
                "SELECT PLUGIN_NAME AS name, PLUGIN_VERSION AS version, "
                "PLUGIN_STATUS AS status, PLUGIN_TYPE AS type "
                "FROM information_schema.PLUGINS ORDER BY PLUGIN_NAME",
            )
            variables = await _fetch(
                ex,
                "SELECT VARIABLE_NAME AS name, VARIABLE_VALUE AS value "
                "FROM performance_schema.global_variables "
                "WHERE VARIABLE_NAME IN ('innodb_buffer_pool_size','innodb_version',"
                "'max_connections','version_compile_os','datadir','character_set_server')",
            )
            instance_properties = {
                "plugins": {p["name"]: p.get("status") for p in plugins},
                "variables": {v["name"]: v["value"] for v in variables},
            }

            schema_rows = await _fetch(
                ex,
                "SELECT SCHEMA_NAME AS name, DEFAULT_CHARACTER_SET_NAME AS charset, "
                "DEFAULT_COLLATION_NAME AS collation "
                "FROM information_schema.SCHEMATA ORDER BY SCHEMA_NAME",
            )
        except Exception:
            await ex.close()
            raise

        databases: list[DiscoveredDatabase] = []
        least_privilege = None
        try:
            # Once per discovery run, on the connection already open.
            least_privilege = await run_least_privilege_check(self, ex, server_id=server_id)
            for row in schema_rows:
                name = row["name"]
                if name in _SYSTEM_SCHEMAS:
                    continue
                db = DiscoveredDatabase(
                    name=name,
                    state="online",
                    options={"charset": row.get("charset"), "collation": row.get("collation")},
                )
                try:
                    await self._fill_database(ex, db, name)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"could not read objects in {name}: {exc}")
                databases.append(db)
        finally:
            await ex.close()

        return ServerCatalog(
            server_id=server_id,
            discovered_at=_now(),
            engine_version=ver.get("version", ""),
            engine_edition=ver.get("edition", "MySQL"),
            instance_properties=instance_properties,
            databases=databases,
            warnings=warnings,
            least_privilege=least_privilege,
        )

    async def _check_least_privilege(self, executor: QueryExecutor) -> LeastPrivilegeFinding:
        rows = await _fetch(executor, _LEAST_PRIVILEGE_SQL, {"limit": LEAST_PRIVILEGE_SCAN_LIMIT})
        return build_least_privilege_finding(
            rows,
            login=self._credentials.username,
            # Unlike PostgreSQL/SQL Server, MySQL's privilege views are
            # genuinely instance-wide, so a clean result here really does
            # mean clean everywhere on this server.
            scope_note="checked instance-wide (MySQL privilege views span every schema)",
        )

    async def _fill_database(self, ex, db: DiscoveredDatabase, schema: str) -> None:
        size_row = await _fetch(
            ex,
            "SELECT SUM(DATA_LENGTH + INDEX_LENGTH) AS size_bytes "
            "FROM information_schema.TABLES WHERE TABLE_SCHEMA = %(s)s",
            {"s": schema},
        )
        db.size_bytes = size_row[0].get("size_bytes") if size_row else None

        tables = await _fetch(
            ex,
            "SELECT TABLE_NAME AS name, TABLE_TYPE AS table_type, TABLE_ROWS AS row_estimate, "
            "(DATA_LENGTH + INDEX_LENGTH) AS size_bytes, ENGINE AS engine "
            "FROM information_schema.TABLES WHERE TABLE_SCHEMA = %(s)s "
            "ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC LIMIT %(limit)s",
            {"s": schema, "limit": self._max_objects},
        )
        indexes = await _fetch(
            ex,
            "SELECT DISTINCT INDEX_NAME AS name, TABLE_NAME AS table_name, "
            "NON_UNIQUE AS non_unique, INDEX_TYPE AS index_type "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = %(s)s "
            "ORDER BY TABLE_NAME, INDEX_NAME LIMIT %(limit)s",
            {"s": schema, "limit": self._max_objects},
        )
        routines = await _fetch(
            ex,
            "SELECT ROUTINE_NAME AS name, ROUTINE_TYPE AS routine_type "
            "FROM information_schema.ROUTINES WHERE ROUTINE_SCHEMA = %(s)s "
            "ORDER BY ROUTINE_NAME LIMIT %(limit)s",
            {"s": schema, "limit": self._max_objects},
        )

        for r in tables:
            kind = "view" if (r.get("table_type") or "").upper() == "VIEW" else "table"
            db.objects.append(
                DiscoveredObject(
                    schema_name=schema,
                    name=r["name"],
                    kind=kind,
                    row_estimate=r.get("row_estimate"),
                    size_bytes=r.get("size_bytes"),
                    properties={"engine": r.get("engine")} if r.get("engine") else {},
                )
            )
        for r in indexes:
            db.objects.append(
                DiscoveredObject(
                    schema_name=schema,
                    name=r["name"],
                    kind="index",
                    properties={
                        "table": r.get("table_name"),
                        "unique": not r.get("non_unique"),
                        "type": r.get("index_type"),
                    },
                )
            )
        for r in routines:
            db.objects.append(
                DiscoveredObject(
                    schema_name=schema,
                    name=r["name"],
                    kind="procedure"
                    if (r.get("routine_type") or "").upper() == "PROCEDURE"
                    else "function",
                )
            )
