"""Discovery dispatch + catalog store (no DB — the live crawl is covered by
the opt-in tests/e2e/test_live_databases.py)."""

from __future__ import annotations

import datetime as dt

import httpx
import pytest

from numi.common.models.catalog import DiscoveredDatabase, DiscoveredObject, ServerCatalog
from numi.common.models.execution import DiscoveryRequest, ExecutionRequest, ExecutionResult
from numi.common.models.target import Platform
from numi.execution.credentials.provider import DatabaseCredentials
from numi.execution.discovery.base import (
    LEAST_PRIVILEGE_SCAN_LIMIT,
    build_least_privilege_finding,
    run_least_privilege_check,
)
from numi.execution.discovery.engine import _discoverer_for, run_discovery
from numi.execution.discovery.mysql import MySQLDiscoverer
from numi.execution.discovery.postgresql import PostgreSQLDiscoverer
from numi.execution.discovery.sqlserver import SQLServerDiscoverer, _quote
from numi.gateway.domain.discovery import DiscoveryOrchestrator, clean_discovery_error
from numi.gateway.infrastructure.execution_client import ExecutionClient


def test_dispatch_picks_the_right_discoverer():
    assert _discoverer_for(Platform.SQLSERVER) is SQLServerDiscoverer
    assert _discoverer_for(Platform.POSTGRESQL) is PostgreSQLDiscoverer
    assert _discoverer_for(Platform.MYSQL) is MySQLDiscoverer
    assert _discoverer_for(Platform.MARIADB) is MySQLDiscoverer


def test_dispatch_raises_for_an_unregistered_engine():
    with pytest.raises(NotImplementedError):
        _discoverer_for(Platform.ORACLE)


async def test_run_discovery_returns_a_clean_warning_instead_of_raising_on_connection_failure(
    monkeypatch,
):
    """Verified live: a genuinely offline dev target made discover() raise
    past this dispatcher as an unhandled exception, surfacing at the API
    layer as a raw 500 with a driver traceback instead of the "nothing
    learned this crawl" outcome the rest of discovery already treats as
    non-fatal (gateway.domain.discovery's lazy_discovery_failed expects to
    catch a clean exception, not have one leak driver internals). Every
    platform funnels through this one dispatcher, so the catch belongs
    here — mirrors ExecutionService.execute()'s same posture."""

    async def _raise_connection_refused(self, server_id):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(PostgreSQLDiscoverer, "discover", _raise_connection_refused)

    catalog = await run_discovery(
        server_id="postgres-dev-01",
        platform=Platform.POSTGRESQL,
        credentials=_credentials("postgres"),
    )

    assert catalog.server_id == "postgres-dev-01"
    assert catalog.databases == []
    assert len(catalog.warnings) == 1
    assert "Could not connect" in catalog.warnings[0]


def test_sqlserver_identifier_quoting_is_injection_safe():
    assert _quote("AdventureWorks2019") == "[AdventureWorks2019]"
    assert _quote("weird]name") == "[weird]]name]"


def test_catalog_lookup_is_case_insensitive_and_returns_canonical_name():
    cat = ServerCatalog(
        server_id="s1",
        databases=[
            DiscoveredDatabase(
                name="CoreBanking",
                objects=[DiscoveredObject(schema_name="dbo", name="Accounts", kind="table")],
            )
        ],
    )
    assert cat.database("corebanking").name == "CoreBanking"
    assert cat.database("nope") is None
    assert cat.database_names() == ["CoreBanking"]


class _FailingExecutionClient(ExecutionClient):
    """Always raises the given exception from `discover()` — used to verify
    `DiscoveryOrchestrator` never lets a raw exception string reach the DBA
    (live-reproduced finding: an `httpx.HTTPStatusError`'s own `__str__`
    bakes in the raw request URL and an MDN documentation link)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:  # pragma: no cover
        raise NotImplementedError

    async def discover(self, request: DiscoveryRequest):
        raise self._exc


def _http_500_error() -> httpx.HTTPStatusError:
    """A realistic reproduction of the live finding: httpx's own message
    for a raised `raise_for_status()` includes the raw URL and an MDN link."""
    request = httpx.Request("POST", "http://localhost:8002/v1/discover")
    response = httpx.Response(500, request=request, text="Internal Server Error")
    return httpx.HTTPStatusError(
        "Server error '500 Internal Server Error' for url 'http://localhost:8002/v1/discover'\n"
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
        request=request,
        response=response,
    )


class TestCleanDiscoveryError:
    def test_http_status_error_becomes_a_clean_status_message(self):
        assert clean_discovery_error(_http_500_error()) == (
            "execution service returned an error (status 500)"
        )

    def test_connect_error_becomes_a_clean_unreachable_message(self):
        request = httpx.Request("POST", "http://localhost:8002/v1/discover")
        assert (
            clean_discovery_error(httpx.ConnectError("boom", request=request))
            == "could not reach the execution service"
        )
        assert (
            clean_discovery_error(httpx.ConnectTimeout("boom", request=request))
            == "could not reach the execution service"
        )

    def test_timeout_errors_become_a_clean_timeout_message(self):
        request = httpx.Request("POST", "http://localhost:8002/v1/discover")
        assert clean_discovery_error(httpx.ReadTimeout("boom", request=request)) == "discovery timed out"
        assert clean_discovery_error(httpx.TimeoutException("boom")) == "discovery timed out"

    def test_an_unrecognized_exception_gets_a_generic_clean_fallback(self):
        assert clean_discovery_error(ValueError("some internal detail")) == "discovery failed (ValueError)"


class TestRefreshAllNeverLeaksRawExceptionText:
    async def test_a_failing_server_gets_a_clean_message_not_the_raw_httpx_text(
        self, server_registry, catalog_store, settings
    ):
        exc = _http_500_error()
        orchestrator = DiscoveryOrchestrator(
            registry=server_registry,
            catalog_store=catalog_store,
            execution_client=_FailingExecutionClient(exc),
            settings=settings,
        )

        results = await orchestrator.refresh_all()

        assert results  # config/servers.yaml has at least one active server
        for server_id, message in results.items():
            assert message == "failed: execution service returned an error (status 500)", server_id
            assert "developer.mozilla.org" not in message
            assert "localhost:8002" not in message
            assert "raise_for_status" not in message

    async def test_a_connect_error_also_gets_a_clean_message(self, server_registry, catalog_store, settings):
        request = httpx.Request("POST", "http://localhost:8002/v1/discover")
        orchestrator = DiscoveryOrchestrator(
            registry=server_registry,
            catalog_store=catalog_store,
            execution_client=_FailingExecutionClient(httpx.ConnectError("boom", request=request)),
            settings=settings,
        )

        results = await orchestrator.refresh_all()

        assert results
        for message in results.values():
            assert message == "failed: could not reach the execution service"


class _QueuedFakeExecutor:
    """Like `tests.fakes.FakeQueryExecutor`, but returns a different
    response per call (in the order given) instead of one static
    `canned_rows` for every call — needed here because a single discovery
    step (`_fill_database`, `_objects`, a full `discover()`) issues several
    *different* queries in sequence, each expecting its own shaped rows."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.executed_sql: list[str] = []
        self.executed_params: list[dict | None] = []

    async def fetch_all(self, sql, params=None, *, timeout=30):
        self.executed_sql.append(sql)
        self.executed_params.append(params)
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def execute(self, sql, params=None, *, timeout=30):
        self.executed_sql.append(sql)
        self.executed_params.append(params)
        return {"rowcount": 1}


def _scripted_connection_factory(responses_by_database: dict[str, list]):
    """Builds a fake `*QueryExecutor` class for monkeypatching
    `numi.execution.adapters.connections.<Engine>QueryExecutor`, so a
    discoverer's `discover()` can run end-to-end against scripted responses
    instead of a live database. Each constructed instance gets its own
    response queue, keyed by `credentials.database` — the same way
    PostgreSQL's per-database `_fill_database` picks a different database
    per connection it opens."""

    class _Executor:
        def __init__(self, credentials):
            self.credentials = credentials
            self._responses = list(responses_by_database[credentials.database])
            self.executed_sql: list[str] = []

        async def connect(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def fetch_all(self, sql, params=None, *, timeout=30):
            self.executed_sql.append(sql)
            resp = self._responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

        async def execute(self, sql, params=None, *, timeout=30):
            self.executed_sql.append(sql)
            return {"rowcount": 1}

    return _Executor


def _credentials(database: str = "app_db") -> DatabaseCredentials:
    return DatabaseCredentials(
        host="db.example", port=5432, username="diag", password="secret", database=database
    )


class TestMySQLDiscoverer:
    async def test_check_least_privilege_unions_all_three_grant_levels_and_excludes_system_schemas(
        self,
    ):
        """MySQL has no single effective-permission function (unlike Postgres'
        has_table_privilege / SQL Server's HAS_PERMS_BY_NAME) — a grant can
        arrive at the global, schema, or table level, and only unioning all
        three catches e.g. a stray `GRANT SELECT ON *.*`."""
        executor = _QueuedFakeExecutor([[{"schema_name": "app_db", "object_name": "orders"}]])
        discoverer = MySQLDiscoverer(_credentials())

        finding = await discoverer._check_least_privilege(executor)

        sql = executor.executed_sql[0]
        assert "information_schema.USER_PRIVILEGES" in sql
        assert "information_schema.SCHEMA_PRIVILEGES" in sql
        assert "information_schema.TABLE_PRIVILEGES" in sql
        assert finding.has_user_table_select is True
        assert finding.scope_note == "checked instance-wide (MySQL privilege views span every schema)"

    async def test_fill_database_classifies_tables_views_indexes_and_routines(self):
        executor = _QueuedFakeExecutor(
            [
                [{"size_bytes": 5000}],
                [
                    {"name": "orders", "table_type": "BASE TABLE", "row_estimate": 10, "engine": "InnoDB"},
                    {"name": "v_orders", "table_type": "VIEW", "row_estimate": None, "engine": None},
                ],
                [{"name": "PRIMARY", "table_name": "orders", "non_unique": 0, "index_type": "BTREE"}],
                [
                    {"name": "sp_archive", "routine_type": "PROCEDURE"},
                    {"name": "fn_total", "routine_type": "FUNCTION"},
                ],
            ]
        )
        discoverer = MySQLDiscoverer(_credentials())
        db = DiscoveredDatabase(name="app_db")

        await discoverer._fill_database(executor, db, "app_db")

        assert db.size_bytes == 5000
        kinds = {o.name: o.kind for o in db.objects}
        assert kinds == {
            "orders": "table",
            "v_orders": "view",
            "PRIMARY": "index",
            "sp_archive": "procedure",
            "fn_total": "function",
        }
        index = next(o for o in db.objects if o.name == "PRIMARY")
        assert index.properties["unique"] is True  # non_unique=0 -> unique

    async def test_discover_skips_system_schemas_and_aggregates_instance_properties(self):
        responses = [
            [{"version": "8.0.35", "edition": "MySQL Community"}],
            [{"name": "innodb", "status": "ACTIVE"}],
            [{"name": "max_connections", "value": "151"}],
            [
                {"name": "app_db", "charset": "utf8mb4", "collation": "utf8mb4_general_ci"},
                {"name": "information_schema", "charset": None, "collation": None},
            ],
            [{"schema_name": "app_db", "object_name": "*"}],  # least-privilege check
            [{"size_bytes": 5000}],  # _fill_database("app_db"): size
            [
                {"name": "orders", "table_type": "BASE TABLE", "row_estimate": 10, "engine": "InnoDB"}
            ],  # tables
            [],  # indexes
            [],  # routines
        ]
        factory = _scripted_connection_factory({"app_db": responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.MySQLQueryExecutor", factory)
            catalog = await MySQLDiscoverer(_credentials("app_db")).discover("mysql-01")

        assert catalog.engine_version == "8.0.35"
        assert catalog.instance_properties["plugins"] == {"innodb": "ACTIVE"}
        assert catalog.instance_properties["variables"] == {"max_connections": "151"}
        # information_schema is a system schema — must not become a database.
        assert catalog.database_names() == ["app_db"]
        assert catalog.databases[0].objects[0].name == "orders"
        assert catalog.least_privilege.has_user_table_select is True
        assert catalog.warnings == []

    async def test_discover_records_a_warning_and_keeps_the_database_entry_when_a_schema_read_fails(
        self,
    ):
        """Best-effort crawl: a schema the login can't fully read must not
        abort the whole discovery run — it becomes a warning, and the
        database still appears in the catalog (with whatever partial data
        was gathered before the failure)."""
        responses = [
            [{"version": "8.0.35", "edition": "MySQL Community"}],
            [],  # plugins
            [],  # variables
            [{"name": "app_db", "charset": "utf8mb4", "collation": "utf8mb4_general_ci"}],
            [{"schema_name": "app_db", "object_name": "*"}],  # least-privilege check
            RuntimeError("access denied for information_schema.TABLES"),  # _fill_database: size fails
        ]
        factory = _scripted_connection_factory({"app_db": responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.MySQLQueryExecutor", factory)
            catalog = await MySQLDiscoverer(_credentials("app_db")).discover("mysql-01")

        assert catalog.database_names() == ["app_db"]
        assert len(catalog.warnings) == 1
        assert "could not read objects in app_db" in catalog.warnings[0]
        assert "access denied" in catalog.warnings[0]

    async def test_discover_closes_the_connection_and_reraises_when_initial_metadata_fetch_fails(
        self,
    ):
        """Unlike a single schema failing mid-crawl (degrades to a warning,
        tested above), a failure reading the server's own version/plugins/
        variables/schema list is fatal — there is no partial catalog worth
        returning, so this must close the connection and propagate."""
        responses = [RuntimeError("connection reset by peer")]
        factory = _scripted_connection_factory({"app_db": responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.MySQLQueryExecutor", factory)
            with pytest.raises(RuntimeError, match="connection reset"):
                await MySQLDiscoverer(_credentials("app_db")).discover("mysql-01")


class TestSQLServerDiscoverer:
    async def test_check_least_privilege_uses_has_perms_by_name(self):
        executor = _QueuedFakeExecutor([[{"schema_name": "dbo", "object_name": "Accounts"}]])
        discoverer = SQLServerDiscoverer(_credentials("CoreBanking"))

        finding = await discoverer._check_least_privilege(executor)

        sql = executor.executed_sql[0]
        assert "HAS_PERMS_BY_NAME" in sql
        assert "o.is_ms_shipped = 0" in sql
        assert finding.sample_objects == ["dbo.Accounts"]
        assert "CoreBanking" in finding.scope_note

    async def test_objects_reads_tables_views_indexes_and_routines_scoped_to_the_database(self):
        executor = _QueuedFakeExecutor(
            [
                [{"schema_name": "dbo", "name": "Accounts", "row_estimate": 100, "size_bytes": 8192}],
                [{"schema_name": "dbo", "name": "v_Accounts"}],
                [
                    {
                        "schema_name": "dbo",
                        "name": "PK_Accounts",
                        "index_type": "CLUSTERED",
                        "table_name": "Accounts",
                    }
                ],
                [{"schema_name": "dbo", "name": "sp_Archive", "kind": "procedure"}],
            ]
        )
        discoverer = SQLServerDiscoverer(_credentials("CoreBanking"))

        objects = await discoverer._objects(executor, "CoreBanking")

        kinds = {o.name: o.kind for o in objects}
        assert kinds == {
            "Accounts": "table",
            "v_Accounts": "view",
            "PK_Accounts": "index",
            "sp_Archive": "procedure",
        }
        # Every query is scoped to the specific database via a bracket-quoted
        # three-part name, not the discovery connection's default database.
        for sql in executor.executed_sql:
            assert "[CoreBanking].sys." in sql

    async def test_discover_excludes_tempdb_and_skips_objects_for_an_offline_database(self):
        responses = [
            [
                {
                    "version": "16.0",
                    "edition": "Standard Edition",
                    "level": "RTM",
                    "hadr_enabled": 0,
                    "cpu_count": 4,
                }
            ],
            [{"name": "max server memory (MB)", "value": "4096"}],
            [
                {"name": "CoreBanking", "state_desc": "ONLINE", "size_bytes": 1000},
                {"name": "Archive", "state_desc": "OFFLINE", "size_bytes": 500},
            ],
            # CoreBanking._objects: tables, views, indexes, routines
            [{"schema_name": "dbo", "name": "Accounts", "row_estimate": 10}],
            [],
            [],
            [],
            [{"schema_name": "dbo", "object_name": "Accounts"}],  # least-privilege check
        ]
        factory = _scripted_connection_factory({"CoreBanking": responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.SQLServerQueryExecutor", factory)
            catalog = await SQLServerDiscoverer(_credentials("CoreBanking")).discover("sqlserver-01")

        # tempdb is filtered by the SQL itself (`WHERE d.name <> 'tempdb'`),
        # so it never even appears in db_rows here — this instead verifies
        # the two databases that *did* come back are handled correctly:
        assert catalog.database_names() == ["CoreBanking", "Archive"]
        corebanking = catalog.databases[0]
        archive = catalog.databases[1]
        assert corebanking.state == "online"
        assert len(corebanking.objects) == 1
        # An offline database must not attempt to read its objects at all.
        assert archive.state == "offline"
        assert archive.objects == []

    async def test_discover_records_a_warning_when_an_online_databases_objects_read_fails(self):
        responses = [
            [
                {
                    "version": "16.0",
                    "edition": "Standard Edition",
                    "level": "RTM",
                    "hadr_enabled": 0,
                    "cpu_count": 4,
                }
            ],
            [],
            [{"name": "CoreBanking", "state_desc": "ONLINE", "size_bytes": 1000}],
            RuntimeError("the SELECT permission was denied on the object 'tables'"),  # _objects: tables
            [{"schema_name": "dbo", "object_name": "Accounts"}],  # least-privilege check
        ]
        factory = _scripted_connection_factory({"CoreBanking": responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.SQLServerQueryExecutor", factory)
            catalog = await SQLServerDiscoverer(_credentials("CoreBanking")).discover("sqlserver-01")

        assert catalog.database_names() == ["CoreBanking"]
        assert catalog.databases[0].objects == []
        assert len(catalog.warnings) == 1
        assert "could not read objects in CoreBanking" in catalog.warnings[0]
        assert "SELECT permission was denied" in catalog.warnings[0]


class TestPostgreSQLDiscoverer:
    async def test_check_least_privilege_uses_has_table_privilege(self):
        executor = _QueuedFakeExecutor([[{"schema_name": "public", "object_name": "orders"}]])
        discoverer = PostgreSQLDiscoverer(_credentials("analytics"))

        finding = await discoverer._check_least_privilege(executor)

        sql = executor.executed_sql[0]
        assert "has_table_privilege" in sql
        assert "pg_catalog" in sql
        assert "analytics" in finding.scope_note

    async def test_discover_reports_per_database_size_objects_and_extensions(self):
        boot_responses = [
            [{"server_version": "16.2"}],
            [{"name": "shared_buffers", "setting": "16384", "unit": "8kB"}],
            [{"datname": "app_db", "owner": "postgres", "encoding": "UTF8", "datcollate": "en_US.UTF-8"}],
            [{"schema_name": "public", "object_name": "orders"}],  # least-privilege check
        ]
        per_db_responses = [
            [{"s": 12345}],  # pg_database_size
            [
                {
                    "schema_name": "public",
                    "name": "orders",
                    "kind": "table",
                    "row_estimate": 100,
                    "size_bytes": 500,
                }
            ],
            [
                {
                    "schema_name": "public",
                    "name": "orders_pkey",
                    "table_name": "orders",
                    "idx_scan": 40,
                    "size_bytes": 8192,
                }
            ],
            [{"schema_name": "public", "name": "recalc_totals", "kind": "function"}],
            [{"name": "pg_stat_statements", "default_version": "1.10", "installed_version": "1.10"}],
        ]
        factory = _scripted_connection_factory({"analytics": boot_responses, "app_db": per_db_responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.PostgreSQLQueryExecutor", factory)
            catalog = await PostgreSQLDiscoverer(_credentials("analytics")).discover("pg-01")

        assert catalog.engine_version == "16.2"
        assert catalog.instance_properties["settings"]["shared_buffers"] == "163848kB"
        assert catalog.database_names() == ["app_db"]
        db = catalog.databases[0]
        assert db.size_bytes == 12345
        kinds = {o.name: o.kind for o in db.objects}
        assert kinds == {"orders": "table", "orders_pkey": "index", "recalc_totals": "function"}
        assert db.extensions[0].name == "pg_stat_statements"
        assert catalog.least_privilege.has_user_table_select is True

    async def test_discover_records_a_warning_when_a_databases_own_connection_fails(self):
        """Postgres opens a fresh connection per database (unlike MySQL/SQL
        Server's single shared connection) — that per-database connect() can
        itself fail (permissions, the database being dropped mid-crawl), and
        must degrade to a warning rather than aborting every other database."""
        boot_responses = [
            [{"server_version": "16.2"}],
            [],
            [{"datname": "locked_db", "owner": "postgres", "encoding": "UTF8", "datcollate": "en_US.UTF-8"}],
            [],  # least-privilege check
        ]
        per_db_responses = [RuntimeError("permission denied for database locked_db")]
        factory = _scripted_connection_factory({"analytics": boot_responses, "locked_db": per_db_responses})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numi.execution.adapters.connections.PostgreSQLQueryExecutor", factory)
            catalog = await PostgreSQLDiscoverer(_credentials("analytics")).discover("pg-01")

        assert catalog.database_names() == ["locked_db"]
        assert len(catalog.warnings) == 1
        assert "could not read database locked_db" in catalog.warnings[0]
        assert "permission denied" in catalog.warnings[0]


class TestRunLeastPrivilegeCheck:
    async def test_a_failing_check_returns_unchecked_rather_than_a_false_clean_bill_of_health(self):
        """A privilege view the login can't read is a normal outcome on a
        locked-down server — must come back `checked=False`, never a
        `has_user_table_select=False` that would misreport as "verified
        clean" when the check never actually ran."""

        class _AlwaysFailingDiscoverer(MySQLDiscoverer):
            async def _check_least_privilege(self, executor):
                raise RuntimeError("SELECT command denied to user")

        finding = await run_least_privilege_check(
            _AlwaysFailingDiscoverer(_credentials()), _QueuedFakeExecutor([]), server_id="s1"
        )

        assert finding.checked is False
        assert finding.has_user_table_select is False


class TestBuildLeastPrivilegeFinding:
    def test_hitting_the_scan_cap_marks_the_count_as_a_lower_bound(self):
        rows = [{"schema_name": "public", "object_name": f"t{i}"} for i in range(LEAST_PRIVILEGE_SCAN_LIMIT)]
        finding = build_least_privilege_finding(rows, login="diag", scope_note="test")
        assert finding.count_is_lower_bound is True
        assert finding.granted_object_count == LEAST_PRIVILEGE_SCAN_LIMIT

    def test_below_the_scan_cap_the_count_is_exact(self):
        rows = [{"schema_name": "public", "object_name": "orders"}]
        finding = build_least_privilege_finding(rows, login="diag", scope_note="test")
        assert finding.count_is_lower_bound is False
        assert finding.granted_object_count == 1

    def test_sample_objects_is_capped_independently_of_the_full_count(self):
        rows = [{"schema_name": "public", "object_name": f"t{i}"} for i in range(50)]
        finding = build_least_privilege_finding(rows, login="diag", scope_note="test")
        assert len(finding.sample_objects) == 5
        assert finding.granted_object_count == 50

    def test_no_rows_means_no_user_table_select(self):
        finding = build_least_privilege_finding([], login="diag", scope_note="test")
        assert finding.has_user_table_select is False
        assert finding.granted_object_count == 0


async def test_db_catalog_store_round_trips_through_the_control_db(db):
    from numi.gateway.infrastructure.catalog_store import DbCatalogStore

    store = DbCatalogStore(db.session_factory)
    cat = ServerCatalog(
        server_id="s1",
        discovered_at=dt.datetime.now(dt.UTC),
        engine_version="16.0",
        databases=[DiscoveredDatabase(name="AppDB", size_bytes=123)],
    )
    await store.put(cat)

    # Fresh store instance -> must load from the DB, not memory.
    store2 = DbCatalogStore(db.session_factory)
    loaded = await store2.get("s1")
    assert loaded is not None
    assert loaded.engine_version == "16.0"
    assert loaded.databases[0].name == "AppDB"
    assert loaded.databases[0].size_bytes == 123
