"""PostgreSQL adapter (spec §20, §48).

Uses `pg_stat_activity`, `pg_locks`, `pg_stat_statements` (where installed),
`pg_stat_database`/`pg_stat_user_tables`, WAL/replication catalog views, and
`pg_settings` — never ad-hoc LLM-generated SQL. Every statement here is
static, parameterized text; the only variable inputs are validated
identifiers/parameters supplied by already-validated tool arguments.
"""

from __future__ import annotations

from typing import Any

from numi.execution.adapters.base import DatabaseAdapter


def _quote_ident(identifier: str) -> str:
    """Safely quote a Postgres identifier (schema/table/index names cannot be
    bind parameters in DDL). Argument values themselves were already
    constrained by the Pydantic argument schema (length, charset-agnostic)
    before reaching here; this additionally strips embedded quotes."""
    return '"' + identifier.replace('"', '') + '"'


class PostgreSQLAdapter(DatabaseAdapter):
    async def health(self) -> list[dict[str, Any]]:
        # Connection/query counts are cluster-wide (pg_stat_activity is never
        # scoped to one database) — only database_size_bytes is necessarily
        # about the database this connection happens to be on.
        sql = """
            select
                (select count(*) from pg_stat_activity) as active_connections,
                (select setting from pg_settings where name = 'max_connections') as max_connections,
                current_database() as connected_database,
                pg_database_size(current_database()) as database_size_bytes,
                (select extract(epoch from now() - pg_postmaster_start_time())) as uptime_seconds,
                (select count(*) from pg_stat_activity where state = 'active') as active_queries
        """
        return await self._executor.fetch_all(sql)

    async def version(self) -> list[dict[str, Any]]:
        return await self._executor.fetch_all("select version() as version")

    async def sessions(self) -> list[dict[str, Any]]:
        # Cluster-wide by design: pg_stat_activity natively covers every
        # database on the instance. `datname` is surfaced on each row so the
        # caller can tell which database(s) a session belongs to without
        # having to already know — the whole point of running this when the
        # affected database isn't known yet.
        sql = """
            select datname as database_name, pid as session_id, usename as user_name,
                   application_name, client_addr, backend_start, state, wait_event_type,
                   wait_event, query_start, left(query, 200) as query_text
            from pg_stat_activity
            order by backend_start
        """
        return await self._executor.fetch_all(sql)

    async def blocking(self) -> list[dict[str, Any]]:
        sql = """
            select blocked.datname as database_name,
                   blocked.pid as blocked_session_id, blocked.query as blocked_query,
                   blocking.pid as blocking_session_id, blocking.query as blocking_query,
                   blocked.wait_event_type, blocked.wait_event
            from pg_stat_activity blocked
            join pg_locks bl on bl.pid = blocked.pid and not bl.granted
            join pg_locks gl on gl.locktype = bl.locktype
                and gl.database is not distinct from bl.database
                and gl.relation is not distinct from bl.relation
                and gl.granted
            join pg_stat_activity blocking on blocking.pid = gl.pid
            where blocked.pid != blocking.pid
        """
        return await self._executor.fetch_all(sql)

    async def deadlocks(self) -> list[dict[str, Any]]:
        # No `where datname = ...` — deadlock counters for every database on
        # the instance, so a spike shows up regardless of which one it's on.
        sql = "select datname, deadlocks from pg_stat_database"
        return await self._executor.fetch_all(sql)

    async def running_queries(self) -> list[dict[str, Any]]:
        sql = """
            select datname as database_name, pid as session_id,
                   now() - query_start as duration, state, left(query, 500) as query_text
            from pg_stat_activity
            where state != 'idle'
            order by query_start
        """
        return await self._executor.fetch_all(sql)

    async def waits(self) -> list[dict[str, Any]]:
        sql = """
            select datname as database_name, wait_event_type, wait_event,
                   count(*) as waiting_sessions
            from pg_stat_activity
            where wait_event is not null
            group by datname, wait_event_type, wait_event
            order by waiting_sessions desc
        """
        return await self._executor.fetch_all(sql)

    async def query_plan(self, query_id: str) -> list[dict[str, Any]]:
        sql = """
            select queryid, query, calls, total_exec_time, mean_exec_time, rows
            from pg_stat_statements
            where queryid = %(query_id)s
        """
        return await self._pg_stat_statements_query(sql, {"query_id": query_id})

    async def top_queries(self, order_by: str, limit: int) -> list[dict[str, Any]]:
        order_column = {
            "cpu": "total_exec_time",
            "duration": "mean_exec_time",
            "reads": "shared_blks_read",
            "writes": "shared_blks_written",
            "executions": "calls",
        }.get(order_by, "total_exec_time")
        sql = f"""
            select queryid, left(query, 300) as query_text, calls, total_exec_time,
                   mean_exec_time, rows
            from pg_stat_statements
            order by {order_column} desc
            limit %(limit)s
        """
        return await self._pg_stat_statements_query(sql, {"limit": limit})

    async def _pg_stat_statements_query(
        self, sql: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        try:
            return await self._executor.fetch_all(sql, params)
        except Exception as exc:  # noqa: BLE001 — narrowed to the one known-optional dependency
            # pg_stat_statements is an optional extension (the module
            # docstring already says "where installed") — a server that
            # doesn't have it enabled must surface that plainly to the DBA
            # instead of a bare EXECUTION_FAILED. Any other failure (a real
            # connection/permissions/syntax problem) still propagates.
            if "pg_stat_statements" not in str(exc):
                raise
            return [
                {
                    "note": (
                        "The pg_stat_statements extension is not installed/enabled on "
                        "this server, so per-query statistics aren't available. Run "
                        "`CREATE EXTENSION pg_stat_statements;` (after adding it to "
                        "shared_preload_libraries and restarting) to enable this."
                    )
                }
            ]

    async def indexes(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = """
            select indexrelname as index_name, idx_scan, idx_tup_read, idx_tup_fetch
            from pg_stat_user_indexes
            where schemaname = %(schema)s and relname = %(table)s
        """
        return await self._executor.fetch_all(sql, {"schema": schema, "table": table})

    async def statistics(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = """
            select last_vacuum, last_autovacuum, last_analyze, last_autoanalyze,
                   n_live_tup, n_dead_tup
            from pg_stat_user_tables
            where schemaname = %(schema)s and relname = %(table)s
        """
        return await self._executor.fetch_all(sql, {"schema": schema, "table": table})

    async def tables(self) -> list[dict[str, Any]]:
        sql = """
            select schemaname as schema_name, relname as table_name, n_live_tup as row_estimate
            from pg_stat_user_tables
            order by schemaname, relname
        """
        return await self._executor.fetch_all(sql)

    async def storage(self) -> list[dict[str, Any]]:
        # pg_database is a global catalog (no per-connection restriction) —
        # every database on the cluster and its size, genuinely instance-wide,
        # unlike the old `pg_database_size(current_database())` which only
        # ever reported the one database this connection happened to be on.
        # pg_stat_user_tables (the per-table breakdown) is itself
        # connection-scoped in Postgres — you can only see the currently-
        # connected database's own tables through it, unlike
        # pg_stat_activity/pg_locks — so that level of detail can't become
        # instance-wide here without opening a connection per database;
        # `get_tables` (already database-scoped) remains the way to get it.
        sql = """
            select datname as database_name, pg_database_size(datname) as database_size_bytes
            from pg_database
            where datistemplate = false
            order by database_size_bytes desc
        """
        return await self._executor.fetch_all(sql)

    async def transaction_log(self) -> list[dict[str, Any]]:
        sql = """
            select pg_current_wal_lsn() as current_wal_lsn,
                   pg_walfile_name(pg_current_wal_lsn()) as current_wal_file
        """
        return await self._executor.fetch_all(sql)

    async def replication(self) -> list[dict[str, Any]]:
        sql = """
            select application_name, client_addr, state, sync_state,
                   pg_wal_lsn_diff(sent_lsn, replay_lsn) as replay_lag_bytes
            from pg_stat_replication
        """
        return await self._executor.fetch_all(sql)

    async def backups(self) -> list[dict[str, Any]]:
        # pg_stat_archiver reflects WAL archiving health, the closest
        # built-in signal without a third-party backup tool's own catalog.
        sql = """
            select archived_count, failed_count, last_archived_wal, last_archived_time,
                   last_failed_wal, last_failed_time
            from pg_stat_archiver
        """
        return await self._executor.fetch_all(sql)

    async def configuration(self) -> list[dict[str, Any]]:
        sql = "select name, setting, unit, category from pg_settings order by category, name"
        return await self._executor.fetch_all(sql)

    async def error_logs(self, since_minutes: int, limit: int) -> list[dict[str, Any]]:
        # Postgres has no built-in queryable error-log table by default
        # (logs go to files / external log collectors). We surface what the
        # engine itself can tell us — recent database-level error/rollback
        # counters — and document that full-text log tailing is an
        # infrastructure integration (e.g. via pgAudit + a log shipper), not
        # something this adapter fabricates.
        sql = "select datname, xact_rollback, deadlocks, stats_reset from pg_stat_database"
        return await self._executor.fetch_all(sql)

    # --- controlled write operations ------------------------------------------

    async def cancel_query(self, session_id: str, reason: str) -> dict[str, Any]:
        result = await self._executor.execute(
            "select pg_cancel_backend(%(pid)s) as cancelled", {"pid": int(session_id)}
        )
        return {"cancelled": result.get("cancelled", False), "session_id": session_id}

    async def kill_session(self, session_id: str, reason: str) -> dict[str, Any]:
        result = await self._executor.execute(
            "select pg_terminate_backend(%(pid)s) as terminated", {"pid": int(session_id)}
        )
        return {"terminated": result.get("terminated", False), "session_id": session_id}

    async def update_statistics(self, schema: str, table: str) -> dict[str, Any]:
        sql = f"analyze {_quote_ident(schema)}.{_quote_ident(table)}"
        result = await self._executor.execute(sql)
        return {"analyzed": True, "schema": schema, "table": table, **result}

    async def create_index(
        self, schema: str, table: str, columns: list[str], name: str, unique: bool
    ) -> dict[str, Any]:
        unique_kw = "unique " if unique else ""
        cols = ", ".join(_quote_ident(c) for c in columns)
        sql = (
            f"create {unique_kw}index concurrently {_quote_ident(name)} "
            f"on {_quote_ident(schema)}.{_quote_ident(table)} ({cols})"
        )
        result = await self._executor.execute(sql)
        return {"created": True, "index_name": name, **result}

    async def rebuild_index(self, schema: str, table: str, index_name: str) -> dict[str, Any]:
        sql = f"reindex index concurrently {_quote_ident(schema)}.{_quote_ident(index_name)}"
        result = await self._executor.execute(sql)
        return {"rebuilt": True, "index_name": index_name, **result}

    async def modify_configuration(self, parameter: str, value: str) -> dict[str, Any]:
        sql = f"alter system set {_quote_ident(parameter)} = %(value)s"
        result = await self._executor.execute(sql, {"value": value})
        await self._executor.execute("select pg_reload_conf()")
        return {"parameter": parameter, "value": value, "reload_triggered": True, **result}

    async def execute_readonly_sql(self, validated_sql: str) -> list[dict[str, Any]]:
        # `validated_sql` has already been parsed, restricted to a single
        # SELECT, denylist-checked, and row-capped by the Gateway's
        # `sql_validator` — this adapter just runs it as ordinary read SQL.
        return await self._executor.fetch_all(validated_sql)

    async def restart_instance(self) -> dict[str, Any]:
        raise NotImplementedError(
            "restart_instance requires an out-of-band infrastructure action "
            "(e.g. orchestrator/systemd/managed-service API call), not a SQL statement; "
            "wire this to your platform's instance-management API."
        )

    async def failover(self, target_instance: str) -> dict[str, Any]:
        raise NotImplementedError(
            "failover requires the replication manager's control-plane API "
            "(e.g. Patroni/repmgr/cloud-managed failover), not a SQL statement."
        )
