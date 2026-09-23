"""Real database connection layer (spec §48).

Implements `QueryExecutor` (see `execution.adapters.base`) against an actual
live database connection, with the safeguards spec §48 requires:
statement/command timeout, lock timeout (Postgres), transaction handling,
cancellation, connection pooling, and a restricted, diagnostics-oriented
account.

Driver packages (`pyodbc`, `psycopg`, `asyncmy`) are optional extras
(`pip install -e ".[db-drivers]"`) — this module imports them lazily so the
rest of the platform (and all unit tests, which use `FakeQueryExecutor`
instead) works without them installed.
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from numi.execution.credentials.provider import DatabaseCredentials

_NAMED_PARAM_RE = re.compile(r"%\((\w+)\)s")


class PostgreSQLQueryExecutor:
    """Wraps a psycopg (v3) async connection.

    Uses a statement_timeout and lock_timeout set per-session (spec §48) and
    never opens a connection with elevated/superuser privileges — the
    restricted diagnostic role is provisioned outside this codebase and
    supplied via `CredentialProvider`.
    """

    def __init__(self, credentials: DatabaseCredentials):
        self._credentials = credentials
        # Typed as Any (not `psycopg.AsyncConnection | None`) so this module
        # stays importable without the optional `psycopg` dependency — see
        # the module docstring.
        self._conn: Any = None

    async def connect(self) -> None:
        import psycopg  # optional extra; see module docstring

        opts = self._credentials.options or {}
        kwargs: dict[str, Any] = dict(
            host=self._credentials.host,
            port=self._credentials.port,
            user=self._credentials.username,
            password=self._credentials.password.get_secret_value(),
            dbname=self._credentials.database,
            autocommit=True,
            connect_timeout=10,
        )
        # e.g. options: { sslmode: require }  in dev_credentials.yaml
        if opts.get("sslmode"):
            kwargs["sslmode"] = opts["sslmode"]
        self._conn = await psycopg.AsyncConnection.connect(**kwargs)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()

    async def _set_timeouts(self, timeout: int) -> None:
        async with self._conn.cursor() as cur:
            await cur.execute(f"SET statement_timeout = {int(timeout) * 1000}")
            await cur.execute(f"SET lock_timeout = {min(int(timeout), 5) * 1000}")

    async def fetch_all(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> list[dict[str, Any]]:
        await self._set_timeouts(timeout)
        async with self._conn.cursor() as cur:
            await cur.execute(sql, params or {})
            columns = [desc.name for desc in cur.description or []]
            rows = await cur.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]

    async def execute(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> dict[str, Any]:
        await self._set_timeouts(timeout)
        async with self._conn.cursor() as cur:
            await cur.execute(sql, params or {})
            result: dict[str, Any] = {"rowcount": cur.rowcount}
            # A statement like `select pg_terminate_backend(%(pid)s) as
            # terminated` is a SELECT, not a plain DML/DDL command — it
            # returns a row, and adapter callers (kill_session, cancel_query)
            # read that value back out of this dict. Discarding it here (as
            # this used to) silently handed every such caller `False`/absent,
            # regardless of what the server actually did — never surfaced as
            # an error, just a wrong answer.
            if cur.description is not None:
                columns = [desc.name for desc in cur.description]
                row = await cur.fetchone()
                if row is not None:
                    result.update(dict(zip(columns, row, strict=False)))
            return result


class MySQLQueryExecutor:
    """Wraps an asyncmy connection (MySQL + MariaDB).

    asyncmy is natively async (no worker thread needed) and uses the
    PyMySQL `%(name)s` / `%s` paramstyle, which is exactly what both
    adapters already emit. A per-session statement timeout is set where the
    engine supports it (`max_execution_time` on MySQL, `max_statement_time`
    on MariaDB); the diagnostic role is provisioned outside this codebase
    and supplied via `CredentialProvider`.
    """

    def __init__(self, credentials: DatabaseCredentials):
        self._credentials = credentials
        # Typed as Any so this module stays importable without `asyncmy`.
        self._conn: Any = None

    async def connect(self) -> None:
        import asyncmy  # optional extra; see module docstring

        opts = self._credentials.options or {}
        kwargs: dict[str, Any] = dict(
            host=self._credentials.host,
            port=self._credentials.port,
            user=self._credentials.username,
            password=self._credentials.password.get_secret_value(),
            database=self._credentials.database,
            autocommit=True,
            connect_timeout=10,
        )
        if opts.get("ssl"):
            kwargs["ssl"] = opts["ssl"]
        self._conn = await asyncmy.connect(**kwargs)

    async def close(self) -> None:
        conn = self._conn
        if conn is not None:
            self._conn = None
            conn.close()

    async def _set_timeout(self, timeout: int) -> None:
        # MySQL: milliseconds, SELECT-only. MariaDB: seconds, all statements.
        # Neither variable exists on the other engine, so try each and
        # ignore an "unknown system variable" error.
        for stmt in (
            f"SET SESSION max_execution_time = {int(timeout) * 1000}",
            f"SET SESSION max_statement_time = {int(timeout)}",
        ):
            try:
                async with self._conn.cursor() as cur:
                    await cur.execute(stmt)
            except Exception:  # noqa: BLE001 — variable not supported on this engine
                continue

    async def fetch_all(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> list[dict[str, Any]]:
        from asyncmy.cursors import DictCursor

        await self._set_timeout(timeout)
        async with self._conn.cursor(cursor=DictCursor) as cur:
            await cur.execute(sql, params or None)
            return list(await cur.fetchall())

    async def execute(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> dict[str, Any]:
        await self._set_timeout(timeout)
        async with self._conn.cursor() as cur:
            await cur.execute(sql, params or None)
            result: dict[str, Any] = {"rowcount": cur.rowcount}
            if cur.description is not None:
                columns = [desc[0] for desc in cur.description]
                row = await cur.fetchone()
                if row is not None:
                    result.update(dict(zip(columns, row, strict=False)))
            return result


class SQLServerQueryExecutor:
    """Wraps a pyodbc connection. pyodbc is synchronous *and* a connection
    must be used from the one thread that created it, so this executor owns a
    dedicated single worker thread — connect / fetch_all / execute / close
    all run on it — and hands work to it via `loop.run_in_executor`.
    `%(name)s`-style SQL text (used uniformly by both adapters for
    readability) is translated to pyodbc's `?` positional placeholders here.
    """

    def __init__(self, credentials: DatabaseCredentials):
        self._credentials = credentials
        # Typed as Any (not `pyodbc.Connection | None`) so this module stays
        # importable without the optional `pyodbc` dependency — see the
        # module docstring.
        self._conn: Any = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mssql")

    async def _run_on_worker(self, fn):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn)

    def _connection_string(self) -> str:
        opts = self._credentials.options or {}
        # `Encrypt` defaults on and `TrustServerCertificate` defaults off
        # (production-safe). For a dev SQL Server with a self-signed cert,
        # set `options: { trust_server_certificate: true }` on that entry in
        # config/dev_credentials.yaml. `options.driver` overrides the ODBC
        # driver name.
        driver = opts.get("driver", "ODBC Driver 18 for SQL Server")
        encrypt = "yes" if opts.get("encrypt", True) else "no"
        trust = "yes" if opts.get("trust_server_certificate", False) else "no"
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={self._credentials.host},{self._credentials.port};"
            f"DATABASE={self._credentials.database};"
            f"UID={self._credentials.username};"
            f"PWD={self._credentials.password.get_secret_value()};"
            f"Encrypt={encrypt};TrustServerCertificate={trust};"
        )

    async def connect(self) -> None:
        import pyodbc  # optional extra; see module docstring

        conn_str = self._connection_string()

        def _connect() -> Any:
            c = pyodbc.connect(conn_str, timeout=10, autocommit=True)
            c.timeout = 30  # default command timeout, overridden per-call below
            return c

        self._conn = await self._run_on_worker(_connect)

    async def close(self) -> None:
        conn = self._conn
        if conn is not None:
            self._conn = None
            try:
                await self._run_on_worker(conn.close)
            finally:
                self._pool.shutdown(wait=False)

    @staticmethod
    def _to_positional(sql: str, params: dict[str, Any] | None) -> tuple[str, list[Any]]:
        params = params or {}
        ordered: list[Any] = []

        def _replace(match: re.Match) -> str:
            ordered.append(params[match.group(1)])
            return "?"

        converted = _NAMED_PARAM_RE.sub(_replace, sql)
        return converted, ordered

    async def fetch_all(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> list[dict[str, Any]]:
        converted, ordered = self._to_positional(sql, params)

        def _run():
            self._conn.timeout = timeout  # pyodbc query timeout is per-connection
            cursor = self._conn.cursor()
            cursor.execute(converted, ordered) if ordered else cursor.execute(converted)
            columns = [c[0] for c in cursor.description or []]
            rows = cursor.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]

        return await self._run_on_worker(_run)

    async def execute(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> dict[str, Any]:
        converted, ordered = self._to_positional(sql, params)

        def _run():
            self._conn.timeout = timeout  # pyodbc query timeout is per-connection
            cursor = self._conn.cursor()
            cursor.execute(converted, ordered) if ordered else cursor.execute(converted)
            result: dict[str, Any] = {"rowcount": cursor.rowcount}
            if cursor.description is not None:
                columns = [c[0] for c in cursor.description]
                row = cursor.fetchone()
                if row is not None:
                    result.update(dict(zip(columns, row, strict=False)))
            return result

        return await self._run_on_worker(_run)
