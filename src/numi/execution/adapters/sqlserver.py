"""SQL Server adapter (spec §20, §48).

Uses DMVs (`sys.dm_exec_*`, `sys.dm_os_*`, `sys.dm_tran_*`,
`sys.dm_hadr_*`), Query Store catalog views where available, and
`sys.databases`/`sys.configurations` — never ad-hoc LLM-generated SQL.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from numi.execution.adapters.base import DatabaseAdapter


def _quote_ident(identifier: str) -> str:
    return "[" + identifier.replace("]", "]]") + "]"


class SQLServerAdapter(DatabaseAdapter):
    async def health(self) -> list[dict[str, Any]]:
        sql = """
            SELECT
                (SELECT COUNT(*) FROM sys.dm_exec_sessions WHERE is_user_process = 1) AS active_sessions,
                (SELECT cntr_value FROM sys.dm_os_performance_counters
                    WHERE counter_name = 'Processor Time' ) AS cpu_counter,
                (SELECT SUM(size) * 8 / 1024 FROM sys.master_files) AS total_size_mb,
                (SELECT DATEDIFF(SECOND, sqlserver_start_time, GETUTCDATE())
                    FROM sys.dm_os_sys_info) AS uptime_seconds
        """
        return await self._executor.fetch_all(sql)

    async def version(self) -> list[dict[str, Any]]:
        return await self._executor.fetch_all(
            "SELECT @@VERSION AS version, SERVERPROPERTY('ProductVersion') AS product_version, "
            "SERVERPROPERTY('Edition') AS edition"
        )

    async def sessions(self) -> list[dict[str, Any]]:
        sql = """
            SELECT session_id, login_name, host_name, program_name, status,
                   cpu_time, memory_usage, last_request_start_time
            FROM sys.dm_exec_sessions
            WHERE is_user_process = 1
            ORDER BY last_request_start_time DESC
        """
        return await self._executor.fetch_all(sql)

    async def blocking(self) -> list[dict[str, Any]]:
        sql = """
            SELECT
                r.session_id AS blocked_session_id,
                r.blocking_session_id,
                r.wait_type, r.wait_time, r.wait_resource,
                t.text AS blocked_query_text
            FROM sys.dm_exec_requests r
            CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) t
            WHERE r.blocking_session_id != 0
        """
        return await self._executor.fetch_all(sql)

    async def deadlocks(self) -> list[dict[str, Any]]:
        sql = """
            SELECT xed.value('@timestamp', 'datetime2') AS deadlock_time,
                   xed.query('.') AS deadlock_graph
            FROM (
                SELECT CAST(target_data AS XML) AS target_data
                FROM sys.dm_xe_session_targets st
                JOIN sys.dm_xe_sessions s ON s.address = st.event_session_address
                WHERE s.name = 'system_health'
            ) AS data
            CROSS APPLY target_data.nodes('RingBufferTarget/event[@name="xml_deadlock_report"]')
                AS xed_table(xed)
        """
        return await self._executor.fetch_all(sql)

    async def running_queries(self) -> list[dict[str, Any]]:
        sql = """
            SELECT r.session_id, r.status, r.command, r.wait_type, r.wait_time,
                   r.cpu_time, r.total_elapsed_time, t.text AS query_text
            FROM sys.dm_exec_requests r
            CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) t
            WHERE r.session_id > 50
            ORDER BY r.total_elapsed_time DESC
        """
        return await self._executor.fetch_all(sql)

    async def waits(self) -> list[dict[str, Any]]:
        sql = """
            SELECT wait_type, waiting_tasks_count, wait_time_ms, signal_wait_time_ms
            FROM sys.dm_os_wait_stats
            WHERE wait_time_ms > 0
            ORDER BY wait_time_ms DESC
        """
        return await self._executor.fetch_all(sql)

    async def query_plan(self, query_id: str) -> list[dict[str, Any]]:
        sql = """
            SELECT qsp.plan_id, qsp.query_plan, qsq.query_hash
            FROM sys.query_store_plan qsp
            JOIN sys.query_store_query qsq ON qsq.query_id = qsp.query_id
            WHERE qsq.query_id = %(query_id)s
        """
        return await self._executor.fetch_all(sql, {"query_id": query_id})

    async def top_queries(self, order_by: str, limit: int) -> list[dict[str, Any]]:
        order_column = {
            "cpu": "avg_cpu_time",
            "duration": "avg_duration",
            "reads": "avg_logical_io_reads",
            "writes": "avg_logical_io_writes",
            "executions": "count_executions",
        }.get(order_by, "avg_cpu_time")
        sql = f"""
            SELECT TOP (%(limit)s) qsq.query_id, qsrs.avg_cpu_time, qsrs.avg_duration,
                   qsrs.avg_logical_io_reads, qsrs.avg_logical_io_writes, qsrs.count_executions
            FROM sys.query_store_query qsq
            JOIN sys.query_store_runtime_stats qsrs ON qsrs.plan_id = qsq.query_id
            ORDER BY {order_column} DESC
        """
        return await self._executor.fetch_all(sql, {"limit": limit})

    async def indexes(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = """
            SELECT i.name AS index_name, i.type_desc,
                   us.user_seeks, us.user_scans, us.user_lookups, us.user_updates
            FROM sys.indexes i
            JOIN sys.tables tb ON tb.object_id = i.object_id
            JOIN sys.schemas s ON s.schema_id = tb.schema_id
            LEFT JOIN sys.dm_db_index_usage_stats us
                ON us.object_id = i.object_id AND us.index_id = i.index_id
            WHERE s.name = %(schema)s AND tb.name = %(table)s
        """
        return await self._executor.fetch_all(sql, {"schema": schema, "table": table})

    async def statistics(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = """
            SELECT st.name AS statistics_name, sp.last_updated, sp.rows, sp.rows_sampled,
                   sp.modification_counter
            FROM sys.stats st
            JOIN sys.tables tb ON tb.object_id = st.object_id
            JOIN sys.schemas s ON s.schema_id = tb.schema_id
            CROSS APPLY sys.dm_db_stats_properties(st.object_id, st.stats_id) sp
            WHERE s.name = %(schema)s AND tb.name = %(table)s
        """
        return await self._executor.fetch_all(sql, {"schema": schema, "table": table})

    async def tables(self) -> list[dict[str, Any]]:
        sql = """
            SELECT s.name AS schema_name, t.name AS table_name, p.rows AS row_estimate
            FROM sys.tables t
            JOIN sys.schemas s ON s.schema_id = t.schema_id
            JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
            ORDER BY s.name, t.name
        """
        return await self._executor.fetch_all(sql)

    async def storage(self) -> list[dict[str, Any]]:
        # sys.master_files is ALREADY a cluster-wide catalog view — every
        # data/log file for every database on the instance. The old
        # `WHERE database_id = DB_ID()` was the artificial restriction;
        # removed, and sys.databases joined in so each row names its
        # database (sys.master_files only carries the numeric database_id).
        sql = """
            SELECT d.name AS database_name, mf.name AS file_name, mf.type_desc,
                   mf.size * 8 / 1024 AS size_mb, mf.max_size, mf.growth
            FROM sys.master_files mf
            JOIN sys.databases d ON d.database_id = mf.database_id
        """
        return await self._executor.fetch_all(sql)

    async def transaction_log(self) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sys.dm_db_log_space_usage"
        return await self._executor.fetch_all(sql)

    async def replication(self) -> list[dict[str, Any]]:
        sql = """
            SELECT ag.name AS availability_group, drs.database_id,
                   drs.synchronization_state_desc, drs.is_suspended, drs.log_send_queue_size
            FROM sys.dm_hadr_database_replica_states drs
            JOIN sys.availability_groups ag ON ag.group_id = drs.group_id
        """
        return await self._executor.fetch_all(sql)

    async def backups(self) -> list[dict[str, Any]]:
        sql = """
            SELECT TOP (20) database_name, backup_start_date, backup_finish_date,
                   type, is_copy_only
            FROM msdb.dbo.backupset
            ORDER BY backup_start_date DESC
        """
        return await self._executor.fetch_all(sql)

    async def configuration(self) -> list[dict[str, Any]]:
        # `value` / `value_in_use` are sql_variant — cast so the ODBC layer
        # can decode them.
        sql = (
            "SELECT name, CAST(value AS NVARCHAR(4000)) AS value, "
            "CAST(value_in_use AS NVARCHAR(4000)) AS value_in_use, description "
            "FROM sys.configurations ORDER BY name"
        )
        return await self._executor.fetch_all(sql)

    async def error_logs(self, since_minutes: int, limit: int) -> list[dict[str, Any]]:
        sql = """
            EXEC sys.xp_readerrorlog 0, 1, NULL, NULL,
                 %(since)s, NULL, N'desc'
        """
        # xp_readerrorlog's start_time/end_time parameters are NOT a SQL
        # datetime/datetime2 type despite what the proc signature suggests —
        # it's an extended stored procedure that parses this argument as
        # text internally. Binding a native Python `datetime` (which pyodbc
        # sends as SQL_TYPE_TIMESTAMP) fails with "The format for the date
        # filter is incorrect" (live-verified against sqlserver-dev-01); it
        # wants an unambiguous 'YYYY-MM-DD HH:MM:SS' *string*, one of the
        # formats the engine's own error message documents. Passing the raw
        # `since_minutes` int (the original bug) failed even earlier, with
        # "Invalid Parameter Type" — this was never a valid start_time at
        # all, integer or otherwise.
        since = (datetime.now(UTC) - timedelta(minutes=since_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        # xp_readerrorlog does not support a row-limit parameter directly;
        # the Gateway's Data Policy Layer enforces `limit` on the returned
        # rows regardless of what the engine returns.
        return await self._executor.fetch_all(sql, {"since": since})

    # --- controlled write operations ------------------------------------------

    async def cancel_query(self, session_id: str, reason: str) -> dict[str, Any]:
        result = await self._executor.execute(f"KILL {int(session_id)}")
        return {"cancelled": True, "session_id": session_id, **result}

    async def kill_session(self, session_id: str, reason: str) -> dict[str, Any]:
        result = await self._executor.execute(f"KILL {int(session_id)}")
        return {"terminated": True, "session_id": session_id, **result}

    async def update_statistics(self, schema: str, table: str) -> dict[str, Any]:
        sql = f"UPDATE STATISTICS {_quote_ident(schema)}.{_quote_ident(table)}"
        result = await self._executor.execute(sql)
        return {"updated": True, "schema": schema, "table": table, **result}

    async def create_index(
        self, schema: str, table: str, columns: list[str], name: str, unique: bool
    ) -> dict[str, Any]:
        unique_kw = "UNIQUE " if unique else ""
        cols = ", ".join(_quote_ident(c) for c in columns)
        sql = (
            f"CREATE {unique_kw}NONCLUSTERED INDEX {_quote_ident(name)} "
            f"ON {_quote_ident(schema)}.{_quote_ident(table)} ({cols}) "
            f"WITH (ONLINE = ON)"
        )
        result = await self._executor.execute(sql)
        return {"created": True, "index_name": name, **result}

    async def rebuild_index(self, schema: str, table: str, index_name: str) -> dict[str, Any]:
        sql = (
            f"ALTER INDEX {_quote_ident(index_name)} "
            f"ON {_quote_ident(schema)}.{_quote_ident(table)} REBUILD WITH (ONLINE = ON)"
        )
        result = await self._executor.execute(sql)
        return {"rebuilt": True, "index_name": index_name, **result}

    async def modify_configuration(self, parameter: str, value: str) -> dict[str, Any]:
        result = await self._executor.execute(
            f"EXEC sp_configure {_quote_ident_literal(parameter)}, %(value)s; RECONFIGURE",
            {"value": value},
        )
        return {"parameter": parameter, "value": value, **result}

    async def execute_readonly_sql(self, validated_sql: str) -> list[dict[str, Any]]:
        # `validated_sql` has already been parsed, restricted to a single
        # SELECT, denylist-checked, and row-capped by the Gateway's
        # `sql_validator` — this adapter just runs it as ordinary read SQL.
        return await self._executor.fetch_all(validated_sql)

    async def restart_instance(self) -> dict[str, Any]:
        raise NotImplementedError(
            "restart_instance requires an out-of-band infrastructure action "
            "(e.g. SQL Server Agent job, Windows service control, or cloud-managed "
            "instance API), not a T-SQL statement; wire this to your platform's "
            "instance-management API."
        )

    async def failover(self, target_instance: str) -> dict[str, Any]:
        raise NotImplementedError(
            "failover must target a specific, inventory-resolved Availability Group and "
            "is intentionally not auto-executed from generic adapter code; wire this to "
            "your platform's Always On failover runbook/API."
        )


def _quote_ident_literal(name: str) -> str:
    """sp_configure takes its parameter name as a string literal, not an
    identifier — quote as a T-SQL string literal, escaping embedded quotes."""
    return "'" + name.replace("'", "''") + "'"
