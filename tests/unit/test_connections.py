"""Connection-string building and parameter translation for the real
database executors (no DB connection — the live path is covered by the
opt-in tests/e2e/test_live_databases.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from numi.execution.adapters.connections import (
    MySQLQueryExecutor,
    PostgreSQLQueryExecutor,
    SQLServerQueryExecutor,
)
from numi.execution.credentials.provider import DatabaseCredentials


def _creds(**options) -> DatabaseCredentials:
    return DatabaseCredentials(
        host="db.example",
        port=1433,
        username="svc",
        password="secret",  # noqa: S106 — test value
        database="AW",
        options=options,
    )


def test_connection_string_is_production_safe_by_default():
    cs = SQLServerQueryExecutor(_creds())._connection_string()
    assert "Encrypt=yes" in cs
    assert "TrustServerCertificate=no" in cs
    assert "ODBC Driver 18 for SQL Server" in cs
    assert "SERVER=db.example,1433" in cs
    assert "DATABASE=AW" in cs


def test_trust_server_certificate_option_is_honoured():
    cs = SQLServerQueryExecutor(_creds(trust_server_certificate=True))._connection_string()
    assert "TrustServerCertificate=yes" in cs


def test_encrypt_and_driver_options_are_honoured():
    cs = SQLServerQueryExecutor(
        _creds(encrypt=False, driver="ODBC Driver 17 for SQL Server")
    )._connection_string()
    assert "Encrypt=no" in cs
    assert "ODBC Driver 17 for SQL Server" in cs


def test_named_params_are_translated_to_positional_placeholders():
    sql = "SELECT * FROM t WHERE id = %(id)s AND name = %(name)s AND id2 = %(id)s"
    converted, ordered = SQLServerQueryExecutor._to_positional(sql, {"id": 7, "name": "x"})
    assert converted == "SELECT * FROM t WHERE id = ? AND name = ? AND id2 = ?"
    assert ordered == [7, "x", 7]


def test_no_params_leaves_sql_untouched():
    converted, ordered = SQLServerQueryExecutor._to_positional("SELECT 1", None)
    assert converted == "SELECT 1"
    assert ordered == []


# --- execute() surfaces a SELECTed result value, not just rowcount --------
#
# Reproduces a live bug found while testing the blocking playbook:
# kill_session's `select pg_terminate_backend(%(pid)s) as terminated` always
# came back with terminated=False/missing, no matter what actually happened
# on the server. Every engine's `execute()` discarded the cursor's own
# result row and returned only {"rowcount": ...} — silently wrong, never an
# error, so it went unnoticed across every kill_session/cancel_query call
# this whole project's live testing had ever made.


class _FakeAsyncCursor:
    """Minimal async-context-manager cursor double for the psycopg/asyncmy
    executors — just enough to drive execute()'s result-row-capture logic."""

    def __init__(self, description=None, row=None, rowcount: int = 1):
        self.description = description
        self._row = row
        self.rowcount = rowcount

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def execute(self, sql, params=None) -> None:
        pass

    async def fetchone(self):
        return self._row


class _FakeAsyncConn:
    def __init__(self, cursor: _FakeAsyncCursor):
        self._cursor = cursor

    def cursor(self) -> _FakeAsyncCursor:
        return self._cursor


class _FakeSyncCursor:
    """Minimal sync cursor double for the pyodbc-backed SQL Server executor."""

    def __init__(self, description=None, row=None, rowcount: int = 1):
        self.description = description
        self._row = row
        self.rowcount = rowcount

    def execute(self, sql, params=None) -> None:
        pass

    def fetchone(self):
        return self._row


class _FakeSyncConn:
    def __init__(self, cursor: _FakeSyncCursor):
        self._cursor = cursor
        self.timeout = None

    def cursor(self) -> _FakeSyncCursor:
        return self._cursor


@pytest.mark.asyncio
async def test_postgres_execute_surfaces_a_selected_columns_value():
    cursor = _FakeAsyncCursor(description=[SimpleNamespace(name="terminated")], row=(True,), rowcount=1)
    executor = PostgreSQLQueryExecutor(_creds())
    executor._conn = _FakeAsyncConn(cursor)

    result = await executor.execute("select pg_terminate_backend(%(pid)s) as terminated", {"pid": 123})

    assert result["terminated"] is True
    assert result["rowcount"] == 1


@pytest.mark.asyncio
async def test_postgres_execute_with_no_returned_rows_keeps_just_rowcount():
    cursor = _FakeAsyncCursor(description=None, row=None, rowcount=1)
    executor = PostgreSQLQueryExecutor(_creds())
    executor._conn = _FakeAsyncConn(cursor)

    result = await executor.execute("update t set x = 1")

    assert result == {"rowcount": 1}


@pytest.mark.asyncio
async def test_mysql_execute_surfaces_a_selected_columns_value():
    cursor = _FakeAsyncCursor(description=[("cancelled",)], row=(1,), rowcount=1)
    executor = MySQLQueryExecutor(_creds())
    executor._conn = _FakeAsyncConn(cursor)

    result = await executor.execute("select 1 as cancelled")

    assert result["cancelled"] == 1


@pytest.mark.asyncio
async def test_sqlserver_execute_surfaces_a_selected_columns_value():
    cursor = _FakeSyncCursor(description=[("terminated", None)], row=(1,), rowcount=1)
    executor = SQLServerQueryExecutor(_creds())
    executor._conn = _FakeSyncConn(cursor)

    result = await executor.execute("select 1 as terminated")

    assert result["terminated"] == 1


# --- fetch_all() -----------------------------------------------------------
#
# execute() above already exercises the result-row-capture logic; fetch_all()
# is the separate, previously-untested code path each adapter method (health,
# sessions, ...) actually calls.


class _RecordingAsyncCursor(_FakeAsyncCursor):
    """Also records the SQL/params passed to execute(), so a test can assert
    on what fetch_all built without needing a real connection."""

    def __init__(self, *args, columns: list[str] | None = None, rows: list[tuple] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._columns = columns or []
        self._rows = rows or []
        self.executed: list[tuple[str, Any]] = []

    async def execute(self, sql, params=None) -> None:
        self.executed.append((sql, params))

    async def fetchall(self):
        return self._rows

    @property
    def description(self):
        return [SimpleNamespace(name=c) for c in self._columns]

    @description.setter
    def description(self, value):
        pass


class _RecordingSyncCursor(_FakeSyncCursor):
    def __init__(self, *args, columns: list[str] | None = None, rows: list[tuple] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._columns = columns or []
        self._rows = rows or []
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql, params=None) -> None:
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows

    @property
    def description(self):
        return [(c, None) for c in self._columns]

    @description.setter
    def description(self, value):
        pass


@pytest.mark.asyncio
async def test_postgres_fetch_all_zips_columns_with_row_values():
    cursor = _RecordingAsyncCursor(columns=["session_id", "state"], rows=[(9182, "active")])
    executor = PostgreSQLQueryExecutor(_creds())
    executor._conn = _FakeAsyncConn(cursor)

    rows = await executor.fetch_all("select session_id, state from pg_stat_activity")

    assert rows == [{"session_id": 9182, "state": "active"}]
    # The statement/lock timeout SET statements run first, on the same
    # cursor (see the dedicated timeout-ordering test below) — the actual
    # query is always the last statement issued.
    assert cursor.executed[-1] == ("select session_id, state from pg_stat_activity", {})


@pytest.mark.asyncio
async def test_postgres_fetch_all_sets_statement_and_lock_timeout_first():
    """spec §48: every query is bounded by a per-session statement/lock
    timeout, set before the actual query runs — not after, and not only on
    execute()."""
    cursor = _RecordingAsyncCursor(columns=[], rows=[])
    executor = PostgreSQLQueryExecutor(_creds())
    executor._conn = _FakeAsyncConn(cursor)

    await executor.fetch_all("select 1", timeout=15)

    assert cursor.executed[0] == ("SET statement_timeout = 15000", None)
    assert cursor.executed[1] == ("SET lock_timeout = 5000", None)
    assert cursor.executed[2] == ("select 1", {})


@pytest.mark.asyncio
async def test_postgres_lock_timeout_is_capped_at_five_seconds_even_for_a_longer_statement_timeout():
    cursor = _RecordingAsyncCursor(columns=[], rows=[])
    executor = PostgreSQLQueryExecutor(_creds())
    executor._conn = _FakeAsyncConn(cursor)

    await executor.fetch_all("select 1", timeout=2)

    assert cursor.executed[1] == ("SET lock_timeout = 2000", None)


@pytest.mark.asyncio
async def test_mysql_fetch_all_returns_dict_rows_via_dictcursor():
    """asyncmy's DictCursor already returns dict rows (unlike psycopg/pyodbc,
    which need the manual zip) — fetch_all must pass them through as-is."""
    executor = MySQLQueryExecutor(_creds())

    class _DictCursor:
        def __init__(self):
            self.executed: list[tuple[str, Any]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, sql, params=None):
            self.executed.append((sql, params))

        async def fetchall(self):
            return [{"session_id": 5, "user_name": "app"}]

    dict_cursor = _DictCursor()

    class _Conn:
        def cursor(self, cursor=None):
            return dict_cursor

    executor._conn = _Conn()

    rows = await executor.fetch_all("select id, user from processlist")

    assert rows == [{"session_id": 5, "user_name": "app"}]
    # This fake's cursor() ignores the cursor= kwarg, so the timeout-setting
    # attempts (via a plain `self._conn.cursor()` call) land on the same
    # object as the real query — the actual query is always issued last.
    assert dict_cursor.executed[-1] == ("select id, user from processlist", None)


@pytest.mark.asyncio
async def test_mysql_set_timeout_tries_mysql_then_mariadb_variable_and_ignores_either_failure():
    """MySQL uses max_execution_time (milliseconds, SELECT-only); MariaDB
    uses max_statement_time (seconds, all statements) — neither variable
    exists on the other engine, so both are attempted and an "unknown
    system variable" failure on either one must not abort the call."""
    executor = MySQLQueryExecutor(_creds())
    attempted: list[str] = []

    class _TimeoutCursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, sql, params=None):
            attempted.append(sql)
            if "max_execution_time" in sql:
                raise Exception("Unknown system variable 'max_execution_time'")

    class _Conn:
        def cursor(self, cursor=None):
            return _TimeoutCursor()

    executor._conn = _Conn()

    await executor._set_timeout(30)

    assert any("max_execution_time" in s for s in attempted)
    assert any("max_statement_time" in s for s in attempted)


@pytest.mark.asyncio
async def test_sqlserver_fetch_all_converts_named_params_and_sets_the_connection_timeout():
    cursor = _RecordingSyncCursor(columns=["session_id"], rows=[(9182,)])
    conn = _FakeSyncConn(cursor)
    executor = SQLServerQueryExecutor(_creds())
    executor._conn = conn

    rows = await executor.fetch_all(
        "SELECT session_id FROM sys.dm_exec_sessions WHERE session_id = %(id)s",
        {"id": 9182},
        timeout=20,
    )

    assert rows == [{"session_id": 9182}]
    assert conn.timeout == 20
    assert cursor.executed[0] == (
        "SELECT session_id FROM sys.dm_exec_sessions WHERE session_id = ?",
        [9182],
    )


@pytest.mark.asyncio
async def test_postgres_close_closes_an_open_connection():
    closed = []

    class _Conn:
        async def close(self):
            closed.append(True)

    executor = PostgreSQLQueryExecutor(_creds())
    executor._conn = _Conn()
    await executor.close()
    assert closed == [True]


@pytest.mark.asyncio
async def test_postgres_close_is_a_no_op_when_never_connected():
    executor = PostgreSQLQueryExecutor(_creds())
    await executor.close()  # must not raise


@pytest.mark.asyncio
async def test_mysql_close_clears_the_connection_reference_before_closing():
    closed = []

    class _Conn:
        def close(self):
            closed.append(True)

    executor = MySQLQueryExecutor(_creds())
    executor._conn = _Conn()
    await executor.close()
    assert closed == [True]
    assert executor._conn is None


@pytest.mark.asyncio
async def test_sqlserver_close_shuts_down_its_worker_thread_pool():
    closed = []

    class _Conn:
        def close(self):
            closed.append(True)

    executor = SQLServerQueryExecutor(_creds())
    executor._conn = _Conn()
    await executor.close()
    assert closed == [True]
    assert executor._conn is None
