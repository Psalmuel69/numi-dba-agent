"""Opt-in end-to-end tests against real database engines (spec §48).

Skipped unless a matching server in config/dev_credentials.yaml is actually
reachable, so `pytest` still passes on a machine with no databases. Bring
the sample databases up with:

    docker compose up -d postgres-sample
    #  (opt-in / heavier images)
    docker compose --profile mssql up -d mssql-sample
    docker compose --profile mysql up -d mysql-sample
    docker compose --profile mariadb up -d mariadb-sample

These exercise the *real* adapter query text against a live connection —
the layer that `FakeQueryExecutor` stands in for elsewhere. Credentials are
looked up by SERVER id (the keys in config/servers.yaml / dev_credentials.yaml),
never by alias.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

import pytest

from numi.execution.adapters.mysql import MySQLAdapter
from numi.execution.adapters.postgresql import PostgreSQLAdapter
from numi.execution.adapters.sqlserver import SQLServerAdapter
from numi.execution.credentials.provider import LocalDevCredentialProvider

_OPT_IN = os.environ.get("RUN_LIVE_DB_TESTS") == "1"
_CREDS_PATH = "config/dev_credentials.yaml"


async def _get_credentials_or_skip(server_id: str):
    try:
        return await LocalDevCredentialProvider(_CREDS_PATH).get_credentials(server_id)
    except Exception as exc:  # noqa: BLE001 — not configured locally, just skip
        pytest.skip(f"'{server_id}' has no entry in {_CREDS_PATH}: {exc}")


async def _pg_executor_or_skip(server_id: str):
    from numi.execution.adapters.connections import PostgreSQLQueryExecutor

    creds = await _get_credentials_or_skip(server_id)
    executor = PostgreSQLQueryExecutor(creds)
    try:
        await asyncio.wait_for(executor.connect(), timeout=3)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"postgres '{server_id}' not reachable: {exc}")
    return executor


async def _mssql_executor_or_skip(server_id: str):
    from numi.execution.adapters.connections import SQLServerQueryExecutor

    creds = await _get_credentials_or_skip(server_id)
    executor = SQLServerQueryExecutor(creds)
    try:
        await asyncio.wait_for(executor.connect(), timeout=5)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"sql server '{server_id}' not reachable: {exc}")
    return executor


async def _mysql_executor_or_skip(server_id: str):
    from numi.execution.adapters.connections import MySQLQueryExecutor

    creds = await _get_credentials_or_skip(server_id)
    executor = MySQLQueryExecutor(creds)
    try:
        await asyncio.wait_for(executor.connect(), timeout=5)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mysql/mariadb '{server_id}' not reachable: {exc}")
    return executor


pytestmark = pytest.mark.skipif(
    not _OPT_IN or not os.path.exists(_CREDS_PATH),
    reason="Live DB tests are opt-in — set RUN_LIVE_DB_TESTS=1 and provide config/dev_credentials.yaml.",
)


async def test_live_postgres_health_and_blocking_queries_run():
    # id from config/servers.yaml -> the `postgres-sample` compose service
    # (host port 5435, database `sample_analytics`).
    executor = await _pg_executor_or_skip("analytics-postgres-prod")
    adapter = PostgreSQLAdapter(executor, "sample_analytics")
    try:
        health = await adapter.health()
        assert isinstance(health, list) and health
        assert "active_connections" in health[0]

        blocking = await adapter.blocking()
        assert isinstance(blocking, list)  # usually empty on an idle db — that's fine

        waits = await adapter.waits()
        assert isinstance(waits, list)
    finally:
        with contextlib.suppress(Exception):
            await executor.close()


async def test_live_sqlserver_health_and_sessions_queries_run():
    # docker compose --profile mssql up -d mssql-sample
    executor = await _mssql_executor_or_skip("sqlserver-dev-01")
    adapter = SQLServerAdapter(executor, "SampleCoreBanking")
    try:
        version = await adapter.version()
        assert version and "version" in version[0]

        sessions = await adapter.sessions()
        assert isinstance(sessions, list)
    finally:
        with contextlib.suppress(Exception):
            await executor.close()


async def test_live_mysql_health_and_sessions_queries_run():
    # docker compose --profile mysql up -d mysql-sample
    executor = await _mysql_executor_or_skip("mysql-dev-01")
    adapter = MySQLAdapter(executor, "sample_app")
    try:
        version = await adapter.version()
        assert version and "version" in version[0]

        health = await adapter.health()
        assert isinstance(health, list) and health

        sessions = await adapter.sessions()
        assert isinstance(sessions, list)
    finally:
        with contextlib.suppress(Exception):
            await executor.close()


async def test_live_mariadb_health_and_sessions_queries_run():
    # docker compose --profile mariadb up -d mariadb-sample
    executor = await _mysql_executor_or_skip("mariadb-dev-01")
    adapter = MySQLAdapter(executor, "sample_app")
    try:
        version = await adapter.version()
        assert version and "version" in version[0]

        sessions = await adapter.sessions()
        assert isinstance(sessions, list)
    finally:
        with contextlib.suppress(Exception):
            await executor.close()
