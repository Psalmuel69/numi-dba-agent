from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from numi.execution.adapters.mysql import MySQLAdapter
from numi.execution.adapters.postgresql import PostgreSQLAdapter
from numi.execution.adapters.sqlserver import SQLServerAdapter
from tests.fakes import FakeQueryExecutor


@pytest.mark.asyncio
async def test_postgres_blocking_uses_pg_locks_and_pg_stat_activity():
    executor = FakeQueryExecutor(canned_rows=[{"blocked_session_id": 100, "blocking_session_id": 200}])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    rows = await adapter.blocking()
    assert rows == [{"blocked_session_id": 100, "blocking_session_id": 200}]
    assert "pg_locks" in executor.executed_sql[0]
    assert "pg_stat_activity" in executor.executed_sql[0]
    assert "current_database()" not in executor.executed_sql[0]
    assert "datname" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_sessions_is_cluster_wide_and_surfaces_database_name():
    """pg_stat_activity natively covers every database on the instance — an
    artificial `where datname = current_database()` filter used to hide that,
    which is exactly what forced the agent to already know the affected
    database before it could even look for it."""
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.sessions()
    sql = executor.executed_sql[0]
    assert "current_database()" not in sql
    assert "datname" in sql


@pytest.mark.asyncio
async def test_postgres_running_queries_is_cluster_wide():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.running_queries()
    sql = executor.executed_sql[0]
    assert "current_database()" not in sql
    assert "datname" in sql


@pytest.mark.asyncio
async def test_postgres_waits_is_cluster_wide():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.waits()
    assert "current_database()" not in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_deadlocks_reports_every_database_not_just_the_current_one():
    executor = FakeQueryExecutor(
        canned_rows=[
            {"datname": "AdventureWorks2019", "deadlocks": 3},
            {"datname": "postgres", "deadlocks": 0},
        ]
    )
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.deadlocks()
    assert "current_database()" not in executor.executed_sql[0]
    assert rows[0]["datname"] == "AdventureWorks2019"


@pytest.mark.asyncio
async def test_postgres_error_logs_is_cluster_wide():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.error_logs(60, 100)
    assert "current_database()" not in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_health_counts_connections_across_the_whole_cluster():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.health()
    sql = executor.executed_sql[0]
    # database_size_bytes legitimately still needs *a* connected database —
    # only the connection/query counts must not be filtered to it.
    assert "where datname = current_database()" not in sql
    assert "pg_database_size(current_database())" in sql


@pytest.mark.asyncio
async def test_postgres_kill_session_calls_pg_terminate_backend():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    await adapter.kill_session("9182", "blocking chain")
    assert "pg_terminate_backend" in executor.executed_sql[0]
    assert executor.executed_params[0] == {"pid": 9182}


@pytest.mark.asyncio
async def test_postgres_top_queries_degrades_when_pg_stat_statements_is_missing():
    """Reproduces a live finding: pg_stat_statements is an optional
    extension (the module docstring already says "where installed"), but
    a server without it enabled used to bare-fail with EXECUTION_FAILED
    and no diagnosable reason. Must degrade to an informative row instead."""
    executor = FakeQueryExecutor(fetch_error=RuntimeError('relation "pg_stat_statements" does not exist'))
    adapter = PostgreSQLAdapter(executor, "TestDatabase")
    rows = await adapter.top_queries("cpu", 10)
    assert len(rows) == 1
    assert "pg_stat_statements" in rows[0]["note"]


@pytest.mark.asyncio
async def test_postgres_query_plan_degrades_when_pg_stat_statements_is_missing():
    executor = FakeQueryExecutor(fetch_error=RuntimeError('relation "pg_stat_statements" does not exist'))
    adapter = PostgreSQLAdapter(executor, "TestDatabase")
    rows = await adapter.query_plan("12345")
    assert len(rows) == 1
    assert "pg_stat_statements" in rows[0]["note"]


@pytest.mark.asyncio
async def test_postgres_top_queries_does_not_swallow_an_unrelated_failure():
    """Only the known-optional pg_stat_statements dependency degrades — a
    real connection/permissions/syntax problem must still propagate."""
    executor = FakeQueryExecutor(fetch_error=RuntimeError("connection reset by peer"))
    adapter = PostgreSQLAdapter(executor, "TestDatabase")
    with pytest.raises(RuntimeError, match="connection reset"):
        await adapter.top_queries("cpu", 10)


@pytest.mark.asyncio
async def test_postgres_create_index_never_receives_raw_sql_from_caller():
    """The Agent supplies structured arguments (schema/table/columns/name),
    never a SQL string — the adapter itself is what constructs SQL."""
    executor = FakeQueryExecutor()
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    await adapter.create_index("dbo", "TransactionPostingHistory", ["TransactionDate"], "IX_TPH_Date", False)
    sql = executor.executed_sql[0]
    assert "create index concurrently" in sql
    assert '"IX_TPH_Date"' in sql
    assert '"TransactionDate"' in sql


@pytest.mark.asyncio
async def test_postgres_storage_is_cluster_wide_and_surfaces_database_name():
    """pg_database is a global catalog (no per-connection restriction) —
    an artificial `pg_database_size(current_database())` used to hide that
    every database's size on the cluster is available in one call. The
    per-table breakdown (pg_stat_user_tables) stays database-scoped — that
    level of detail is still `get_tables`'s job, not this one's."""
    executor = FakeQueryExecutor(
        canned_rows=[
            {"database_name": "analytics_prod", "database_size_bytes": 9000},
            {"database_name": "postgres", "database_size_bytes": 100},
        ]
    )
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.storage()
    sql = executor.executed_sql[0]
    assert "current_database()" not in sql
    assert "pg_database" in sql
    assert "datname" in sql
    assert rows[0]["database_name"] == "analytics_prod"


@pytest.mark.asyncio
async def test_sqlserver_blocking_uses_dm_exec_requests():
    executor = FakeQueryExecutor(canned_rows=[{"blocked_session_id": 9183}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.blocking()
    assert rows == [{"blocked_session_id": 9183}]
    assert "sys.dm_exec_requests" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_kill_session_issues_kill_statement():
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.kill_session("9182", "blocking chain")
    assert executor.executed_sql[0] == "KILL 9182"


@pytest.mark.asyncio
async def test_sqlserver_kill_session_rejects_non_numeric_session_id():
    """session_id is cast to int before being embedded — a session id like
    '9182; DROP TABLE x' cannot become part of the KILL statement."""
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    with pytest.raises(ValueError):
        await adapter.kill_session("9182; DROP TABLE x", "attempted injection")


@pytest.mark.asyncio
async def test_sqlserver_storage_is_instance_wide_and_surfaces_database_name():
    """sys.master_files is ALREADY a cluster-wide catalog view — every
    data/log file for every database on the instance. The old
    `WHERE database_id = DB_ID()` was the artificial restriction; removed,
    with sys.databases joined in for the database name (sys.master_files
    only carries the numeric database_id)."""
    executor = FakeQueryExecutor(
        canned_rows=[{"database_name": "CoreBanking", "file_name": "primary_data", "size_mb": 48213}]
    )
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.storage()
    sql = executor.executed_sql[0]
    assert "DB_ID()" not in sql
    assert "sys.master_files" in sql
    assert "sys.databases" in sql
    assert rows[0]["database_name"] == "CoreBanking"


@pytest.mark.asyncio
async def test_sqlserver_error_logs_passes_a_formatted_datetime_not_raw_minutes():
    """xp_readerrorlog's start_time argument is parsed as TEXT by the
    extended stored procedure, not bound as a SQL datetime type — binding
    the raw `since_minutes` int (the original bug: "Invalid Parameter
    Type") or even a native Python `datetime` (fails live with "The format
    for the date filter is incorrect") both error on a real instance. Only
    an explicit 'YYYY-MM-DD HH:MM:SS' string works (live-verified against
    sqlserver-dev-01). Assert the bound `since` param is such a string,
    roughly `since_minutes` in the past, never a bare int."""
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    before = datetime.now(UTC) - timedelta(minutes=60, seconds=5)

    await adapter.error_logs(since_minutes=60, limit=50)

    assert "xp_readerrorlog" in executor.executed_sql[0]
    since = executor.executed_params[0]["since"]
    assert isinstance(since, str)
    assert not isinstance(since, int)
    parsed = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
    after = datetime.now(UTC) - timedelta(minutes=60) + timedelta(seconds=5)
    assert before.replace(tzinfo=None) <= parsed <= after.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_restart_instance_is_not_a_sql_statement_on_either_adapter():
    """restart/failover require an infrastructure control-plane call, not a
    SQL statement the adapter could construct — this is enforced by raising
    rather than silently no-op'ing."""
    pg = PostgreSQLAdapter(FakeQueryExecutor(), "analytics_prod")
    with pytest.raises(NotImplementedError):
        await pg.restart_instance()

    mssql = SQLServerAdapter(FakeQueryExecutor(), "CoreBanking")
    with pytest.raises(NotImplementedError):
        await mssql.failover("corebanking-prd-02")

    mysql = MySQLAdapter(FakeQueryExecutor(), "app_db")
    with pytest.raises(NotImplementedError):
        await mysql.restart_instance()


@pytest.mark.asyncio
async def test_mysql_blocking_uses_data_lock_waits_and_innodb_trx():
    executor = FakeQueryExecutor(canned_rows=[{"blocked_session_id": 42, "blocking_session_id": 7}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.blocking()
    assert rows == [{"blocked_session_id": 42, "blocking_session_id": 7}]
    assert "performance_schema.data_lock_waits" in executor.executed_sql[0]
    assert "information_schema.INNODB_TRX" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_storage_is_instance_wide_and_surfaces_schema_name():
    """information_schema.TABLES spans every schema on the instance — the
    old `WHERE TABLE_SCHEMA = DATABASE()` was the artificial restriction.
    Removed in favor of GROUP BY TABLE_SCHEMA, so one call reports every
    schema's size with the schema name as a column."""
    executor = FakeQueryExecutor(
        canned_rows=[
            {"schema_name": "app_db", "database_size_bytes": 5000},
            {"schema_name": "analytics", "database_size_bytes": 7000},
        ]
    )
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.storage()
    sql = executor.executed_sql[0]
    assert "TABLE_SCHEMA = DATABASE()" not in sql
    assert "GROUP BY TABLE_SCHEMA" in sql
    assert rows[0]["schema_name"] == "app_db"


@pytest.mark.asyncio
async def test_mysql_kill_session_issues_kill_statement():
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.kill_session("9182", "blocking chain")
    assert executor.executed_sql[0] == "KILL 9182"


@pytest.mark.asyncio
async def test_mysql_kill_session_rejects_non_numeric_session_id():
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    with pytest.raises(ValueError):
        await adapter.kill_session("9182; DROP TABLE x", "attempted injection")


@pytest.mark.asyncio
async def test_mysql_create_index_never_receives_raw_sql_from_caller():
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.create_index("app_db", "orders", ["created_at"], "IX_orders_created", False)
    sql = executor.executed_sql[0]
    assert "CREATE INDEX" in sql
    assert "`IX_orders_created`" in sql
    assert "`created_at`" in sql


@pytest.mark.asyncio
async def test_mysql_deadlocks_extracts_the_latest_detected_section():
    status_text = (
        "=====================================\n"
        "LATEST DETECTED DEADLOCK\n"
        "------------------------\n"
        "*** (1) TRANSACTION:\nsome transaction detail\n"
        "------------\n"
        "WE ROLL BACK TRANSACTION (1)\n"
        "-----------------------------------------\n"
        "END OF INNODB MONITOR OUTPUT\n"
    )
    executor = FakeQueryExecutor(canned_rows=[{"Type": "InnoDB", "Name": "", "Status": status_text}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.deadlocks()
    assert "some transaction detail" in rows[0]["latest_detected_deadlock"]


@pytest.mark.asyncio
async def test_mysql_backups_has_no_builtin_catalog():
    adapter = MySQLAdapter(FakeQueryExecutor(), "app_db")
    with pytest.raises(NotImplementedError):
        await adapter.backups()


@pytest.mark.asyncio
async def test_mysql_failover_is_not_a_sql_statement():
    """Unlike restart_instance (tested above for all three engines), MySQL's
    failover path was never itself exercised — only Postgres/SQL Server were."""
    adapter = MySQLAdapter(FakeQueryExecutor(), "app_db")
    with pytest.raises(NotImplementedError):
        await adapter.failover("app-db-replica-02")


@pytest.mark.asyncio
async def test_postgres_failover_is_not_a_sql_statement():
    adapter = PostgreSQLAdapter(FakeQueryExecutor(), "analytics_prod")
    with pytest.raises(NotImplementedError):
        await adapter.failover("analytics-prod-replica-02")


@pytest.mark.asyncio
async def test_sqlserver_restart_instance_is_not_a_sql_statement():
    adapter = SQLServerAdapter(FakeQueryExecutor(), "CoreBanking")
    with pytest.raises(NotImplementedError):
        await adapter.restart_instance()


# --- health / version / sessions / running_queries / waits / query_plan ---
#
# One test per adapter per method that previously issued a query nothing
# ever asserted against — each checks the DMV/catalog view the docstring
# says it uses, so a future edit that silently swaps in the wrong view (as
# already happened once for the cluster-wide/current-database bugs above)
# is caught the same way those were.


@pytest.mark.asyncio
async def test_postgres_version_queries_the_version_function():
    executor = FakeQueryExecutor(canned_rows=[{"version": "PostgreSQL 16.2"}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.version()
    assert rows == [{"version": "PostgreSQL 16.2"}]
    assert "select version()" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_query_plan_queries_pg_stat_statements_by_queryid():
    executor = FakeQueryExecutor(canned_rows=[{"queryid": 123, "query": "select 1"}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.query_plan("123")
    assert "pg_stat_statements" in executor.executed_sql[0]
    assert executor.executed_params[0] == {"query_id": "123"}


@pytest.mark.asyncio
async def test_postgres_top_queries_orders_by_the_requested_column():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.top_queries("writes", 5)
    sql = executor.executed_sql[0]
    assert "order by shared_blks_written desc" in sql
    assert executor.executed_params[0] == {"limit": 5}


@pytest.mark.asyncio
async def test_postgres_top_queries_falls_back_to_total_exec_time_for_an_unknown_order_by():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.top_queries("not-a-real-column", 5)
    assert "order by total_exec_time desc" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_indexes_scopes_to_the_given_schema_and_table():
    executor = FakeQueryExecutor(canned_rows=[{"index_name": "orders_pkey", "idx_scan": 10}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    await adapter.indexes("public", "orders")
    assert executor.executed_params[0] == {"schema": "public", "table": "orders"}
    assert "pg_stat_user_indexes" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_statistics_reads_vacuum_and_tuple_counters():
    executor = FakeQueryExecutor(canned_rows=[{"n_live_tup": 1000, "n_dead_tup": 5}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.statistics("public", "orders")
    assert rows[0]["n_live_tup"] == 1000
    assert "pg_stat_user_tables" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_tables_lists_every_table_in_the_current_database():
    executor = FakeQueryExecutor(canned_rows=[{"table_name": "orders", "row_estimate": 500}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.tables()
    assert rows[0]["table_name"] == "orders"
    assert "pg_stat_user_tables" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_transaction_log_reads_current_wal_position():
    executor = FakeQueryExecutor(canned_rows=[{"current_wal_lsn": "0/1A2B3C"}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.transaction_log()
    assert rows[0]["current_wal_lsn"] == "0/1A2B3C"
    assert "pg_current_wal_lsn" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_replication_reads_pg_stat_replication():
    executor = FakeQueryExecutor(canned_rows=[{"application_name": "replica-1", "state": "streaming"}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.replication()
    assert rows[0]["state"] == "streaming"
    assert "pg_stat_replication" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_backups_reads_wal_archiver_stats():
    executor = FakeQueryExecutor(canned_rows=[{"archived_count": 42, "failed_count": 0}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.backups()
    assert rows[0]["archived_count"] == 42
    assert "pg_stat_archiver" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_configuration_reads_pg_settings():
    executor = FakeQueryExecutor(canned_rows=[{"name": "shared_buffers", "setting": "16384"}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.configuration()
    assert rows[0]["name"] == "shared_buffers"
    assert "pg_settings" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_error_logs_reads_database_level_error_counters():
    """Postgres has no built-in queryable error-log table — the module
    docstring says so explicitly; error_logs surfaces the closest built-in
    signal (rollback/deadlock counters) rather than fabricating log lines."""
    executor = FakeQueryExecutor(canned_rows=[{"datname": "postgres", "xact_rollback": 3}])
    adapter = PostgreSQLAdapter(executor, "postgres")
    rows = await adapter.error_logs(60, 100)
    assert rows[0]["xact_rollback"] == 3
    assert "pg_stat_database" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_postgres_cancel_query_calls_pg_cancel_backend():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    result = await adapter.cancel_query("9182", "runaway query")
    assert "pg_cancel_backend" in executor.executed_sql[0]
    assert executor.executed_params[0] == {"pid": 9182}
    assert result["session_id"] == "9182"


@pytest.mark.asyncio
async def test_postgres_update_statistics_runs_analyze_on_the_given_table():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    result = await adapter.update_statistics("public", "orders")
    assert executor.executed_sql[0] == 'analyze "public"."orders"'
    assert result["analyzed"] is True


@pytest.mark.asyncio
async def test_postgres_rebuild_index_uses_reindex_concurrently():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    result = await adapter.rebuild_index("public", "orders", "orders_pkey")
    assert executor.executed_sql[0] == 'reindex index concurrently "public"."orders_pkey"'
    assert result["rebuilt"] is True


@pytest.mark.asyncio
async def test_postgres_modify_configuration_sets_and_reloads():
    """`alter system set` alone doesn't take effect until reloaded — this
    must issue both statements, not just the first."""
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    result = await adapter.modify_configuration("work_mem", "64MB")
    assert "alter system set" in executor.executed_sql[0]
    assert executor.executed_params[0] == {"value": "64MB"}
    assert "pg_reload_conf" in executor.executed_sql[1]
    assert result["reload_triggered"] is True


@pytest.mark.asyncio
async def test_postgres_execute_readonly_sql_runs_the_already_validated_text_verbatim():
    """`execute_readonly_sql` trusts the Gateway's validator completely — it
    must not re-wrap, re-quote, or otherwise alter the SQL it's handed."""
    executor = FakeQueryExecutor(canned_rows=[{"n": 1}])
    adapter = PostgreSQLAdapter(executor, "analytics_prod")
    rows = await adapter.execute_readonly_sql("select 1 as n")
    assert executor.executed_sql[0] == "select 1 as n"
    assert rows == [{"n": 1}]


@pytest.mark.asyncio
async def test_mysql_health_queries_performance_schema_status_and_variables():
    executor = FakeQueryExecutor(canned_rows=[{"active_connections": "12"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.health()
    assert rows[0]["active_connections"] == "12"
    assert "performance_schema.global_status" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_version_queries_version_function():
    executor = FakeQueryExecutor(canned_rows=[{"version": "8.0.35"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.version()
    assert rows[0]["version"] == "8.0.35"
    assert "VERSION()" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_sessions_reads_the_processlist():
    executor = FakeQueryExecutor(canned_rows=[{"session_id": 5, "user_name": "app"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.sessions()
    assert rows[0]["session_id"] == 5
    assert "information_schema.PROCESSLIST" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_running_queries_excludes_sleeping_and_daemon_threads():
    executor = FakeQueryExecutor(canned_rows=[{"session_id": 5, "state": "executing"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.running_queries()
    sql = executor.executed_sql[0]
    assert "COMMAND NOT IN ('Sleep', 'Daemon')" in sql
    assert rows[0]["state"] == "executing"


@pytest.mark.asyncio
async def test_mysql_waits_reads_events_waits_summary():
    executor = FakeQueryExecutor(canned_rows=[{"wait_event": "io/file/innodb/innodb_data_file"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.waits()
    assert "events_waits_summary_global_by_event_name" in executor.executed_sql[0]
    assert rows[0]["wait_event"].startswith("io/")


@pytest.mark.asyncio
async def test_mysql_query_plan_looks_up_the_statement_digest():
    executor = FakeQueryExecutor(canned_rows=[{"query_id": "abc123"}])
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.query_plan("abc123")
    assert executor.executed_params[0] == {"query_id": "abc123"}
    assert "events_statements_summary_by_digest" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_top_queries_orders_by_the_requested_column():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.top_queries("reads", 10)
    assert "ORDER BY SUM_ROWS_EXAMINED DESC" in executor.executed_sql[0]
    assert executor.executed_params[0] == {"limit": 10}


@pytest.mark.asyncio
async def test_mysql_top_queries_falls_back_to_total_latency_for_an_unknown_order_by():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.top_queries("bogus", 10)
    assert "ORDER BY SUM_TIMER_WAIT DESC" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_indexes_joins_statistics_and_usage_by_index_name():
    executor = FakeQueryExecutor(canned_rows=[{"index_name": "PRIMARY", "cardinality": 100}])
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.indexes("app_db", "orders")
    assert executor.executed_params[0] == {"schema": "app_db", "table": "orders"}
    assert "information_schema.STATISTICS" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_statistics_reads_table_level_counters():
    executor = FakeQueryExecutor(canned_rows=[{"row_estimate": 100, "engine": "InnoDB"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.statistics("app_db", "orders")
    assert rows[0]["engine"] == "InnoDB"
    assert "information_schema.TABLES" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_tables_lists_base_tables_in_the_current_schema():
    executor = FakeQueryExecutor(canned_rows=[{"table_name": "orders"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.tables()
    sql = executor.executed_sql[0]
    assert "TABLE_TYPE = 'BASE TABLE'" in sql
    assert rows[0]["table_name"] == "orders"


@pytest.mark.asyncio
async def test_mysql_transaction_log_reads_innodb_redo_log_counters():
    executor = FakeQueryExecutor(canned_rows=[{"name": "INNODB_LOG_WRITES", "value": "42"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.transaction_log()
    assert "INNODB_LOG_WRITES" in executor.executed_sql[0]
    assert rows[0]["value"] == "42"


@pytest.mark.asyncio
async def test_mysql_replication_joins_connection_and_applier_status():
    executor = FakeQueryExecutor(canned_rows=[{"channel": "", "io_state": "ON"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.replication()
    assert "replication_connection_status" in executor.executed_sql[0]
    assert "replication_applier_status" in executor.executed_sql[0]
    assert rows[0]["io_state"] == "ON"


@pytest.mark.asyncio
async def test_mysql_configuration_reads_global_variables():
    executor = FakeQueryExecutor(canned_rows=[{"name": "max_connections", "value": "151"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.configuration()
    assert rows[0]["name"] == "max_connections"
    assert "performance_schema.global_variables" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_mysql_error_logs_filters_by_minutes_since():
    executor = FakeQueryExecutor(canned_rows=[{"message": "some error"}])
    adapter = MySQLAdapter(executor, "app_db")
    rows = await adapter.error_logs(60, 50)
    assert executor.executed_params[0] == {"since": 60}
    assert "performance_schema.error_log" in executor.executed_sql[0]
    assert rows[0]["message"] == "some error"


@pytest.mark.asyncio
async def test_mysql_cancel_query_issues_kill_query_not_kill():
    """KILL QUERY stops the running statement but keeps the connection open —
    distinct from KILL (used by kill_session), which drops the connection
    entirely. Using the wrong one would silently disconnect a session the
    caller only meant to interrupt."""
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    await adapter.cancel_query("9182", "runaway query")
    assert executor.executed_sql[0] == "KILL QUERY 9182"


@pytest.mark.asyncio
async def test_mysql_update_statistics_runs_analyze_table():
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    result = await adapter.update_statistics("app_db", "orders")
    assert executor.executed_sql[0] == "ANALYZE TABLE `app_db`.`orders`"
    assert result["analyzed"] is True


@pytest.mark.asyncio
async def test_mysql_rebuild_index_rebuilds_the_whole_table_not_just_one_index():
    """InnoDB has no single-index rebuild — this must say so in the result,
    not silently claim to have rebuilt only the named index."""
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    result = await adapter.rebuild_index("app_db", "orders", "idx_created_at")
    assert executor.executed_sql[0] == "ALTER TABLE `app_db`.`orders` ENGINE=InnoDB"
    assert result["rebuilt"] is True
    assert "rebuild all indexes" in result["note"]


@pytest.mark.asyncio
async def test_mysql_modify_configuration_is_not_persisted_across_restart():
    """SET GLOBAL is a live-only change on MySQL (SET PERSIST would survive a
    restart, but that's a different statement this adapter doesn't issue) —
    the result must say so plainly rather than implying durability."""
    executor = FakeQueryExecutor()
    adapter = MySQLAdapter(executor, "app_db")
    result = await adapter.modify_configuration("max_connections", "500")
    assert executor.executed_sql[0] == "SET GLOBAL `max_connections` = %(value)s"
    assert executor.executed_params[0] == {"value": "500"}
    assert result["persisted_across_restart"] is False


@pytest.mark.asyncio
async def test_sqlserver_health_reads_dmv_counters():
    executor = FakeQueryExecutor(canned_rows=[{"active_sessions": 5}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.health()
    assert rows[0]["active_sessions"] == 5
    assert "sys.dm_exec_sessions" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_version_reads_serverproperty():
    executor = FakeQueryExecutor(canned_rows=[{"edition": "Standard Edition"}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.version()
    assert rows[0]["edition"] == "Standard Edition"
    assert "SERVERPROPERTY" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_sessions_excludes_system_processes():
    executor = FakeQueryExecutor(canned_rows=[{"session_id": 55}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.sessions()
    assert "is_user_process = 1" in executor.executed_sql[0]
    assert rows[0]["session_id"] == 55


@pytest.mark.asyncio
async def test_sqlserver_deadlocks_reads_the_system_health_xevent_session():
    executor = FakeQueryExecutor(canned_rows=[{"deadlock_time": "2026-01-01"}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.deadlocks()
    assert "system_health" in executor.executed_sql[0]
    assert rows[0]["deadlock_time"] == "2026-01-01"


@pytest.mark.asyncio
async def test_sqlserver_running_queries_excludes_system_session_ids():
    """session_id <= 50 is reserved for SQL Server's own internal system
    sessions — excluded so this never reports the engine's own housekeeping
    as a DBA-relevant running query."""
    executor = FakeQueryExecutor(canned_rows=[{"session_id": 55}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.running_queries()
    assert "r.session_id > 50" in executor.executed_sql[0]
    assert rows[0]["session_id"] == 55


@pytest.mark.asyncio
async def test_sqlserver_waits_excludes_zero_wait_time():
    executor = FakeQueryExecutor(canned_rows=[{"wait_type": "PAGEIOLATCH_SH"}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.waits()
    assert "wait_time_ms > 0" in executor.executed_sql[0]
    assert rows[0]["wait_type"] == "PAGEIOLATCH_SH"


@pytest.mark.asyncio
async def test_sqlserver_query_plan_reads_query_store():
    executor = FakeQueryExecutor(canned_rows=[{"plan_id": 1}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.query_plan("42")
    assert executor.executed_params[0] == {"query_id": "42"}
    assert "sys.query_store_plan" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_top_queries_orders_by_the_requested_column():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.top_queries("executions", 10)
    assert "ORDER BY count_executions DESC" in executor.executed_sql[0]
    assert executor.executed_params[0] == {"limit": 10}


@pytest.mark.asyncio
async def test_sqlserver_top_queries_falls_back_to_avg_cpu_time_for_an_unknown_order_by():
    executor = FakeQueryExecutor(canned_rows=[])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.top_queries("bogus", 10)
    assert "ORDER BY avg_cpu_time DESC" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_indexes_joins_usage_stats_by_object_and_index_id():
    executor = FakeQueryExecutor(canned_rows=[{"index_name": "PK_Accounts", "user_seeks": 10}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.indexes("dbo", "Accounts")
    assert executor.executed_params[0] == {"schema": "dbo", "table": "Accounts"}
    assert "sys.dm_db_index_usage_stats" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_statistics_reads_stats_properties():
    executor = FakeQueryExecutor(canned_rows=[{"statistics_name": "stat_1", "rows": 1000}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.statistics("dbo", "Accounts")
    assert rows[0]["rows"] == 1000
    assert "sys.dm_db_stats_properties" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_tables_reads_partition_row_counts():
    executor = FakeQueryExecutor(canned_rows=[{"table_name": "Accounts", "row_estimate": 500}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.tables()
    assert rows[0]["table_name"] == "Accounts"
    assert "sys.partitions" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_transaction_log_reads_log_space_usage():
    executor = FakeQueryExecutor(canned_rows=[{"database_id": 5}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.transaction_log()
    assert "sys.dm_db_log_space_usage" in executor.executed_sql[0]
    assert rows[0]["database_id"] == 5


@pytest.mark.asyncio
async def test_sqlserver_replication_reads_always_on_availability_groups():
    executor = FakeQueryExecutor(canned_rows=[{"availability_group": "AG1"}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.replication()
    assert "sys.dm_hadr_database_replica_states" in executor.executed_sql[0]
    assert rows[0]["availability_group"] == "AG1"


@pytest.mark.asyncio
async def test_sqlserver_backups_reads_msdb_backupset():
    executor = FakeQueryExecutor(canned_rows=[{"database_name": "CoreBanking"}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.backups()
    assert "msdb.dbo.backupset" in executor.executed_sql[0]
    assert rows[0]["database_name"] == "CoreBanking"


@pytest.mark.asyncio
async def test_sqlserver_configuration_casts_sql_variant_columns():
    """`value`/`value_in_use` are sql_variant — must be cast for pyodbc to
    decode them, or a live server 500s on this call."""
    executor = FakeQueryExecutor(canned_rows=[{"name": "max degree of parallelism"}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.configuration()
    sql = executor.executed_sql[0]
    assert "CAST(value AS NVARCHAR(4000))" in sql
    assert "CAST(value_in_use AS NVARCHAR(4000))" in sql
    assert rows[0]["name"] == "max degree of parallelism"


@pytest.mark.asyncio
async def test_sqlserver_cancel_query_and_kill_session_both_issue_kill():
    """Unlike MySQL (KILL QUERY vs KILL), SQL Server has one KILL statement
    for both — cancel_query and kill_session are documented as issuing the
    identical statement, not two different behaviors."""
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    result = await adapter.cancel_query("9182", "runaway query")
    assert executor.executed_sql[0] == "KILL 9182"
    assert result["cancelled"] is True


@pytest.mark.asyncio
async def test_sqlserver_update_statistics_runs_update_statistics():
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    result = await adapter.update_statistics("dbo", "Accounts")
    assert executor.executed_sql[0] == "UPDATE STATISTICS [dbo].[Accounts]"
    assert result["updated"] is True


@pytest.mark.asyncio
async def test_sqlserver_create_index_builds_online_nonclustered_index():
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.create_index("dbo", "Accounts", ["OpenedDate"], "IX_Accounts_Opened", False)
    sql = executor.executed_sql[0]
    assert "CREATE NONCLUSTERED INDEX" in sql
    assert "WITH (ONLINE = ON)" in sql
    assert "[IX_Accounts_Opened]" in sql


@pytest.mark.asyncio
async def test_sqlserver_create_index_honours_the_unique_flag():
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.create_index("dbo", "Accounts", ["AccountNumber"], "UX_Accounts_Number", True)
    assert "CREATE UNIQUE NONCLUSTERED INDEX" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_rebuild_index_runs_online():
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    result = await adapter.rebuild_index("dbo", "Accounts", "PK_Accounts")
    assert executor.executed_sql[0] == (
        "ALTER INDEX [PK_Accounts] ON [dbo].[Accounts] REBUILD WITH (ONLINE = ON)"
    )
    assert result["rebuilt"] is True


@pytest.mark.asyncio
async def test_sqlserver_modify_configuration_quotes_the_parameter_as_a_string_literal():
    """sp_configure takes its parameter name as a T-SQL string literal, not
    an identifier — `[max degree of parallelism]` would be a syntax error;
    it must be `'max degree of parallelism'` with embedded quotes doubled."""
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    result = await adapter.modify_configuration("max degree of parallelism", "4")
    sql = executor.executed_sql[0]
    assert "EXEC sp_configure 'max degree of parallelism', %(value)s; RECONFIGURE" in sql
    assert executor.executed_params[0] == {"value": "4"}
    assert result["parameter"] == "max degree of parallelism"


@pytest.mark.asyncio
async def test_sqlserver_modify_configuration_escapes_an_embedded_quote_in_the_parameter_name():
    executor = FakeQueryExecutor()
    adapter = SQLServerAdapter(executor, "CoreBanking")
    await adapter.modify_configuration("weird'param", "1")
    assert "'weird''param'" in executor.executed_sql[0]


@pytest.mark.asyncio
async def test_sqlserver_execute_readonly_sql_runs_the_already_validated_text_verbatim():
    executor = FakeQueryExecutor(canned_rows=[{"n": 1}])
    adapter = SQLServerAdapter(executor, "CoreBanking")
    rows = await adapter.execute_readonly_sql("SELECT 1 AS n")
    assert executor.executed_sql[0] == "SELECT 1 AS n"
    assert rows == [{"n": 1}]
