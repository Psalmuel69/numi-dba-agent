"""General fix for a recurring complaint: "database.get_health was rejected
(INVALID_TARGET): Which database on postgres-local?" even when the DBA never
named one and the diagnostic itself doesn't need one — sessions, blocking
chains, deadlocks, running queries, wait stats, replication status, backup
status, configuration, error logs, health and version are all genuinely
instance/cluster-wide on every engine we support (SQL Server DMVs, MySQL
information_schema/performance_schema, Postgres pg_stat_activity et al. are
never scoped to one database). These tools must never demand a database —
that's what lets the agent *discover* which database is affected (e.g. by
checking what's running server-wide) instead of only ever being told.

get_storage was added to this set in a follow-up (it was deliberately left
database-scoped in the original fix, which was the wrong call): SQL
Server's sys.master_files and MySQL's information_schema.TABLES are
themselves already instance-wide catalogs, and Postgres's pg_database is
too, for the database-size figure — see test_adapters.py for the
per-engine SQL assertions and the honest caveat that Postgres's per-table
breakdown stays database-scoped (pg_stat_user_tables), still available via
get_tables."""

from __future__ import annotations

import datetime as dt

import pytest

from numi.common.config import Settings
from numi.common.models.target import DatabaseTarget, Environment
from numi.gateway.domain.catalog import DiscoveredDatabase, InMemoryCatalogStore, ServerCatalog
from numi.gateway.domain.servers import ServerRegistry
from numi.gateway.domain.target_validation import TargetValidator
from numi.gateway.domain.tool_catalog import build_tool_catalog

_INSTANCE_WIDE_TOOLS = {
    "database.get_health",
    "database.get_version",
    "database.get_sessions",
    "database.get_blocking_sessions",
    "database.get_deadlocks",
    "database.get_running_queries",
    "database.get_wait_statistics",
    "database.get_replication_status",
    "database.get_backup_status",
    "database.get_configuration",
    "database.get_error_logs",
    "database.get_storage",
}

# These stay genuinely database-scoped and must keep requiring one.
_STILL_DATABASE_SCOPED_TOOLS = {
    "database.get_query_plan",
    "database.get_top_queries",
    "database.get_indexes",
    "database.get_statistics",
    "database.get_tables",
    "database.get_transaction_log",
}


def test_instance_wide_tools_do_not_require_a_database():
    catalog = build_tool_catalog(Settings(_env_file=None))
    by_id = {t.tool_id: t for t in catalog}
    for tool_id in _INSTANCE_WIDE_TOOLS:
        assert "database" not in by_id[tool_id].required_target_scope, tool_id
        assert "instance" in by_id[tool_id].required_target_scope, tool_id


def test_genuinely_database_scoped_tools_still_require_one():
    catalog = build_tool_catalog(Settings(_env_file=None))
    by_id = {t.tool_id: t for t in catalog}
    for tool_id in _STILL_DATABASE_SCOPED_TOOLS:
        assert "database" in by_id[tool_id].required_target_scope, tool_id


@pytest.mark.asyncio
async def test_get_sessions_target_validates_with_no_database_named_even_when_catalog_populated():
    """Reproduces the live complaint exactly: postgres-local has a populated
    catalog (several discovered databases), the DBA never named one, and the
    tool being called (get_sessions) is instance-wide — validation must
    proceed instead of asking "Which database on postgres-local?"."""
    registry = ServerRegistry("config/servers.yaml")
    catalog_store = InMemoryCatalogStore()
    await catalog_store.put(
        ServerCatalog(
            server_id="postgres-local",
            discovered_at=dt.datetime.now(dt.UTC),
            databases=[
                DiscoveredDatabase(name="postgres", objects=[]),
                DiscoveredDatabase(name="TestDatabase", objects=[]),
            ],
        )
    )
    validator = TargetValidator(registry, catalog_store)
    target = DatabaseTarget(environment=Environment.DEVELOPMENT, instance="postgres-local")

    ctx = await validator.validate(target, ["environment", "instance"])

    assert ctx.server.id == "postgres-local"
    assert ctx.database == ""
