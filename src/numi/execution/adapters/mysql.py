"""MySQL / MariaDB adapter (spec §20, §48).

One adapter for both engines — they share the wire protocol, the InnoDB
storage engine, `information_schema`, and (MySQL 5.7+, MariaDB 10.5+)
`performance_schema`. Where the two diverge (`performance_schema.error_log`
is MySQL-only; replication status DDL differs), the method documents it and
degrades to a clear error rather than guessing.

Every statement here is static text; the only variable inputs are validated
identifiers/parameters from already-validated tool arguments — never
ad-hoc LLM-generated SQL.
"""

from __future__ import annotations

import re
from typing import Any

from numi.execution.adapters.base import DatabaseAdapter

_DEADLOCK_SECTION = re.compile(
    r"LATEST DETECTED DEADLOCK\n-+\n(.*?)(?:\n-{10,}\n|\Z)", re.DOTALL
)


def _quote_ident(identifier: str) -> str:
    """Backtick-quote a MySQL identifier, escaping embedded backticks. The
    argument values were already length/charset constrained by the Pydantic
    argument schema before reaching here."""
    return "`" + identifier.replace("`", "``") + "`"


class MySQLAdapter(DatabaseAdapter):
    async def health(self) -> list[dict[str, Any]]:
        sql = """
            SELECT
              (SELECT VARIABLE_VALUE FROM performance_schema.global_status
                 WHERE VARIABLE_NAME = 'THREADS_CONNECTED') AS active_connections,
              (SELECT VARIABLE_VALUE FROM performance_schema.global_variables
                 WHERE VARIABLE_NAME = 'MAX_CONNECTIONS') AS max_connections,
              (SELECT VARIABLE_VALUE FROM performance_schema.global_status
                 WHERE VARIABLE_NAME = 'THREADS_RUNNING') AS active_queries,
              (SELECT VARIABLE_VALUE FROM performance_schema.global_status
                 WHERE VARIABLE_NAME = 'UPTIME') AS uptime_seconds,
              (SELECT SUM(data_length + index_length) FROM information_schema.tables
                 WHERE table_schema = DATABASE()) AS database_size_bytes
        """
        return await self._executor.fetch_all(sql)

    async def version(self) -> list[dict[str, Any]]:
        return await self._executor.fetch_all(
            "SELECT VERSION() AS version, "
            "@@version_comment AS edition, @@version_compile_os AS os"
        )

    async def sessions(self) -> list[dict[str, Any]]:
        sql = """
            SELECT ID AS session_id, USER AS user_name, HOST AS host, DB AS database_name,
                   COMMAND AS command, TIME AS seconds, STATE AS state,
                   LEFT(INFO, 200) AS query_text
            FROM information_schema.PROCESSLIST
            ORDER BY TIME DESC
        """
        return await self._executor.fetch_all(sql)

    async def blocking(self) -> list[dict[str, Any]]:
        # performance_schema.data_lock_waits: MySQL 8.0+, MariaDB 10.6+.
        sql = """
            SELECT
                rt.trx_mysql_thread_id AS blocked_session_id,
                LEFT(rt.trx_query, 200) AS blocked_query,
                bt.trx_mysql_thread_id AS blocking_session_id,
                LEFT(bt.trx_query, 200) AS blocking_query,
                w.OBJECT_SCHEMA AS object_schema,
                w.OBJECT_NAME AS object_name,
                w.LOCK_TYPE AS lock_type
            FROM performance_schema.data_lock_waits w
            JOIN information_schema.INNODB_TRX rt
                ON rt.trx_id = w.REQUESTING_ENGINE_TRANSACTION_ID
            JOIN information_schema.INNODB_TRX bt
                ON bt.trx_id = w.BLOCKING_ENGINE_TRANSACTION_ID
        """
        return await self._executor.fetch_all(sql)

    async def deadlocks(self) -> list[dict[str, Any]]:
        # InnoDB keeps only the single most recent deadlock, in the text of
        # SHOW ENGINE INNODB STATUS. Persistent deadlock history needs
        # innodb_print_all_deadlocks + the error log (an infra integration).
        rows = await self._executor.fetch_all("SHOW ENGINE INNODB STATUS")
        status = ""
        if rows:
            status = str(rows[0].get("Status") or next(iter(rows[0].values()), ""))
        match = _DEADLOCK_SECTION.search(status)
        return [{"latest_detected_deadlock": match.group(1).strip() if match else None}]

    async def running_queries(self) -> list[dict[str, Any]]:
        sql = """
            SELECT ID AS session_id, TIME AS duration_seconds, STATE AS state,
                   LEFT(INFO, 500) AS query_text
            FROM information_schema.PROCESSLIST
            WHERE COMMAND NOT IN ('Sleep', 'Daemon') AND INFO IS NOT NULL
            ORDER BY TIME DESC
        """
        return await self._executor.fetch_all(sql)

    async def waits(self) -> list[dict[str, Any]]:
        sql = """
            SELECT EVENT_NAME AS wait_event, COUNT_STAR AS waiting_events,
                   SUM_TIMER_WAIT AS total_wait_picoseconds
            FROM performance_schema.events_waits_summary_global_by_event_name
            WHERE COUNT_STAR > 0
            ORDER BY SUM_TIMER_WAIT DESC
        """
        return await self._executor.fetch_all(sql)

    async def query_plan(self, query_id: str) -> list[dict[str, Any]]:
        # MySQL/MariaDB don't cache execution plans. `query_id` is a
        # statement digest; return the digest's aggregate profile.
        sql = """
            SELECT DIGEST AS query_id, LEFT(DIGEST_TEXT, 500) AS query_text,
                   COUNT_STAR AS executions, SUM_TIMER_WAIT AS total_latency_picoseconds,
                   SUM_ROWS_EXAMINED AS rows_examined, SUM_ROWS_SENT AS rows_sent,
                   SUM_CREATED_TMP_DISK_TABLES AS tmp_disk_tables, SUM_NO_INDEX_USED AS no_index_used
            FROM performance_schema.events_statements_summary_by_digest
            WHERE DIGEST = %(query_id)s
        """
        return await self._executor.fetch_all(sql, {"query_id": query_id})

    async def top_queries(self, order_by: str, limit: int) -> list[dict[str, Any]]:
        order_column = {
            "cpu": "SUM_TIMER_WAIT",
            "duration": "AVG_TIMER_WAIT",
            "reads": "SUM_ROWS_EXAMINED",
            "writes": "SUM_ROWS_AFFECTED",
            "executions": "COUNT_STAR",
        }.get(order_by, "SUM_TIMER_WAIT")
        sql = f"""
            SELECT DIGEST AS query_id, LEFT(DIGEST_TEXT, 300) AS query_text,
                   COUNT_STAR AS executions, SUM_TIMER_WAIT AS total_latency_picoseconds,
                   AVG_TIMER_WAIT AS avg_latency_picoseconds, SUM_ROWS_EXAMINED AS rows_examined,
                   SUM_ROWS_SENT AS rows_sent, SUM_ROWS_AFFECTED AS rows_affected
            FROM performance_schema.events_statements_summary_by_digest
            WHERE DIGEST IS NOT NULL
            ORDER BY {order_column} DESC
            LIMIT %(limit)s
        """
        return await self._executor.fetch_all(sql, {"limit": limit})

    async def indexes(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = """
            SELECT s.INDEX_NAME AS index_name, s.NON_UNIQUE AS non_unique,
                   s.SEQ_IN_INDEX AS seq_in_index, s.COLUMN_NAME AS column_name,
                   s.CARDINALITY AS cardinality, s.INDEX_TYPE AS index_type,
                   u.COUNT_READ AS reads, u.COUNT_WRITE AS writes,
                   u.COUNT_FETCH AS fetches
            FROM information_schema.STATISTICS s
            LEFT JOIN performance_schema.table_io_waits_summary_by_index_usage u
                ON u.OBJECT_SCHEMA = s.TABLE_SCHEMA AND u.OBJECT_NAME = s.TABLE_NAME
               AND u.INDEX_NAME = s.INDEX_NAME
            WHERE s.TABLE_SCHEMA = %(schema)s AND s.TABLE_NAME = %(table)s
            ORDER BY s.INDEX_NAME, s.SEQ_IN_INDEX
        """
        return await self._executor.fetch_all(sql, {"schema": schema, "table": table})

    async def statistics(self, schema: str, table: str) -> list[dict[str, Any]]:
        sql = """
            SELECT t.TABLE_ROWS AS row_estimate, t.AVG_ROW_LENGTH AS avg_row_length,
                   t.DATA_LENGTH AS data_length, t.INDEX_LENGTH AS index_length,
                   t.DATA_FREE AS data_free, t.UPDATE_TIME AS last_update_time,
                   t.CREATE_TIME AS create_time, t.ENGINE AS engine
            FROM information_schema.TABLES t
            WHERE t.TABLE_SCHEMA = %(schema)s AND t.TABLE_NAME = %(table)s
        """
        return await self._executor.fetch_all(sql, {"schema": schema, "table": table})

    async def tables(self) -> list[dict[str, Any]]:
        sql = """
            SELECT TABLE_SCHEMA AS schema_name, TABLE_NAME AS table_name,
                   TABLE_ROWS AS row_estimate,
                   (DATA_LENGTH + INDEX_LENGTH) AS size_bytes
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC
        """
        return await self._executor.fetch_all(sql)

    async def storage(self) -> list[dict[str, Any]]:
        # information_schema.TABLES spans every schema on the instance —
        # the old `WHERE TABLE_SCHEMA = DATABASE()` was the artificial
        # restriction. Removed in favor of grouping by schema, so one call
        # reports every schema's size on the instance with the schema name
        # as a column, instead of only the one this connection happened to
        # default to.
        sql = """
            SELECT TABLE_SCHEMA AS schema_name,
                   SUM(DATA_LENGTH) AS data_bytes, SUM(INDEX_LENGTH) AS index_bytes,
                   SUM(DATA_FREE) AS free_bytes,
                   SUM(DATA_LENGTH + INDEX_LENGTH) AS database_size_bytes
            FROM information_schema.TABLES
            GROUP BY TABLE_SCHEMA
        """
        return await self._executor.fetch_all(sql)

    async def transaction_log(self) -> list[dict[str, Any]]:
        # InnoDB redo log counters — the closest built-in signal without
        # parsing SHOW ENGINE INNODB STATUS's LOG section.
        sql = """
            SELECT VARIABLE_NAME AS name, VARIABLE_VALUE AS value
            FROM performance_schema.global_status
            WHERE VARIABLE_NAME IN (
                'INNODB_OS_LOG_WRITTEN', 'INNODB_OS_LOG_PENDING_WRITES',
                'INNODB_LOG_WAITS', 'INNODB_LOG_WRITE_REQUESTS', 'INNODB_LOG_WRITES'
            )
        """
        return await self._executor.fetch_all(sql)

    async def replication(self) -> list[dict[str, Any]]:
        # performance_schema replication tables are populated by MySQL's
        # native replication; MariaDB reports via SHOW REPLICA/SLAVE STATUS
        # (a follow-up if you run MariaDB replicas).
        sql = """
            SELECT c.CHANNEL_NAME AS channel, c.SERVICE_STATE AS io_state,
                   c.SOURCE_UUID AS source_uuid, c.LAST_ERROR_MESSAGE AS io_last_error,
                   a.SERVICE_STATE AS sql_state
            FROM performance_schema.replication_connection_status c
            LEFT JOIN performance_schema.replication_applier_status a
                ON a.CHANNEL_NAME = c.CHANNEL_NAME
        """
        return await self._executor.fetch_all(sql)

    async def backups(self) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "MySQL/MariaDB have no built-in backup catalog — backup history "
            "lives in your backup tool (mysqldump / Percona XtraBackup / "
            "mariabackup / a managed-service snapshot API). Wire get_backup_status "
            "to that tool's catalog."
        )

    async def configuration(self) -> list[dict[str, Any]]:
        sql = """
            SELECT VARIABLE_NAME AS name, VARIABLE_VALUE AS value
            FROM performance_schema.global_variables
            ORDER BY VARIABLE_NAME
        """
        return await self._executor.fetch_all(sql)

    async def error_logs(self, since_minutes: int, limit: int) -> list[dict[str, Any]]:
        # performance_schema.error_log is MySQL 8.0.22+. On MariaDB (no such
        # table) this surfaces as EXECUTION_FAILED; MariaDB's error log is a
        # file, tailed via a log shipper.
        sql = """
            SELECT LOGGED AS logged_at, PRIO AS priority, ERROR_CODE AS error_code,
                   SUBSYSTEM AS subsystem, DATA AS message
            FROM performance_schema.error_log
            WHERE LOGGED > (NOW() - INTERVAL %(since)s MINUTE)
            ORDER BY LOGGED DESC
        """
        return await self._executor.fetch_all(sql, {"since": since_minutes})

    # --- controlled write operations ------------------------------------------

    async def cancel_query(self, session_id: str, reason: str) -> dict[str, Any]:
        result = await self._executor.execute(f"KILL QUERY {int(session_id)}")
        return {"cancelled": True, "session_id": session_id, **result}

    async def kill_session(self, session_id: str, reason: str) -> dict[str, Any]:
        result = await self._executor.execute(f"KILL {int(session_id)}")
        return {"terminated": True, "session_id": session_id, **result}

    async def update_statistics(self, schema: str, table: str) -> dict[str, Any]:
        sql = f"ANALYZE TABLE {_quote_ident(schema)}.{_quote_ident(table)}"
        result = await self._executor.execute(sql)
        return {"analyzed": True, "schema": schema, "table": table, **result}

    async def create_index(
        self, schema: str, table: str, columns: list[str], name: str, unique: bool
    ) -> dict[str, Any]:
        unique_kw = "UNIQUE " if unique else ""
        cols = ", ".join(_quote_ident(c) for c in columns)
        sql = (
            f"CREATE {unique_kw}INDEX {_quote_ident(name)} "
            f"ON {_quote_ident(schema)}.{_quote_ident(table)} ({cols}) "
            f"ALGORITHM=INPLACE, LOCK=NONE"
        )
        result = await self._executor.execute(sql)
        return {"created": True, "index_name": name, **result}

    async def rebuild_index(self, schema: str, table: str, index_name: str) -> dict[str, Any]:
        # InnoDB has no single-index rebuild; a null-rebuild ALTER recreates
        # every index on the table in place.
        sql = f"ALTER TABLE {_quote_ident(schema)}.{_quote_ident(table)} ENGINE=InnoDB"
        result = await self._executor.execute(sql)
        return {
            "rebuilt": True,
            "index_name": index_name,
            "note": "MySQL/MariaDB rebuild all indexes on the table, not just this one.",
            **result,
        }

    async def modify_configuration(self, parameter: str, value: str) -> dict[str, Any]:
        # SET GLOBAL is not persisted across restart (MySQL 8 SET PERSIST is;
        # MariaDB is not) — durable config still lives in my.cnf.
        sql = f"SET GLOBAL {_quote_ident(parameter)} = %(value)s"
        result = await self._executor.execute(sql, {"value": value})
        return {
            "parameter": parameter,
            "value": value,
            "persisted_across_restart": False,
            **result,
        }

    async def restart_instance(self) -> dict[str, Any]:
        raise NotImplementedError(
            "restart_instance requires an out-of-band infrastructure action "
            "(systemd / mysqladmin shutdown + supervisor / managed-service API), "
            "not a SQL statement; wire this to your platform's instance-management API."
        )

    async def failover(self, target_instance: str) -> dict[str, Any]:
        raise NotImplementedError(
            "failover requires the replication topology manager's control-plane "
            "API (Orchestrator / MHA / Group Replication / managed-service failover), "
            "not a SQL statement."
        )
