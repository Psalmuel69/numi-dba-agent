"""Least-privilege login check (execution/discovery/*.py).

The architectural premise — "the diagnostic login should NOT have SELECT on
user tables" (`execution/discovery/base.py`'s own module docstring) — was
unverified until this check existed. These drive each engine's real
`_check_least_privilege` through `FakeQueryExecutor` (the pattern from
`tests/unit/test_adapters.py`), never a real connection, for both the
"login can read user data" and "login cannot" cases.
"""

from __future__ import annotations

import pytest

from numi.common.models.catalog import LeastPrivilegeFinding, ServerCatalog
from numi.execution.credentials.provider import DatabaseCredentials
from numi.execution.discovery.base import (
    LEAST_PRIVILEGE_SCAN_LIMIT,
    build_least_privilege_finding,
    run_least_privilege_check,
)
from numi.execution.discovery.mysql import MySQLDiscoverer
from numi.execution.discovery.postgresql import PostgreSQLDiscoverer
from numi.execution.discovery.sqlserver import SQLServerDiscoverer
from tests.fakes import FakeQueryExecutor


def _creds(database: str = "corebanking") -> DatabaseCredentials:
    return DatabaseCredentials(
        host="db.internal",
        port=5432,
        username="numi_diag",
        password="unused-in-these-tests",  # noqa: S106 — a fake, never a real connection
        database=database,
    )


_GRANTED_ROWS = [
    {"schema_name": "public", "object_name": "accounts"},
    {"schema_name": "public", "object_name": "customers"},
]

_DISCOVERERS = [
    pytest.param(PostgreSQLDiscoverer, "postgresql", id="postgresql"),
    pytest.param(SQLServerDiscoverer, "sqlserver", id="sqlserver"),
    pytest.param(MySQLDiscoverer, "mysql", id="mysql"),
]


class TestPerEngineTheCheckDetectsAnOverPrivilegedLogin:
    @pytest.mark.parametrize(("discoverer_cls", "engine"), _DISCOVERERS)
    @pytest.mark.asyncio
    async def test_a_login_with_select_on_user_tables_is_flagged(self, discoverer_cls, engine):
        executor = FakeQueryExecutor(canned_rows=_GRANTED_ROWS)
        discoverer = discoverer_cls(_creds())

        finding = await discoverer._check_least_privilege(executor)

        assert finding.checked is True
        assert finding.has_user_table_select is True
        assert finding.granted_object_count == 2
        assert finding.login == "numi_diag"
        assert finding.sample_objects == ["public.accounts", "public.customers"]
        assert finding.warning_text() is not None
        assert "should be revoked for least-privilege" in finding.warning_text()

    @pytest.mark.parametrize(("discoverer_cls", "engine"), _DISCOVERERS)
    @pytest.mark.asyncio
    async def test_a_properly_scoped_login_produces_a_clean_finding(self, discoverer_cls, engine):
        executor = FakeQueryExecutor(canned_rows=[])
        discoverer = discoverer_cls(_creds())

        finding = await discoverer._check_least_privilege(executor)

        assert finding.checked is True
        assert finding.has_user_table_select is False
        assert finding.granted_object_count == 0
        assert finding.sample_objects == []
        # Nothing to say -> nothing is said. No warning noise on a correctly
        # provisioned server.
        assert finding.warning_text() is None


class TestTheQueriesAreReadOnlyIntrospectionOfTheRightViews:
    """Numi reports, a human DBA acts — the check must never attempt to
    change a grant, and must exclude the system catalogs its login is
    legitimately supposed to read."""

    @pytest.mark.parametrize(("discoverer_cls", "engine"), _DISCOVERERS)
    @pytest.mark.asyncio
    async def test_no_statement_ever_attempts_to_change_a_privilege(
        self, discoverer_cls, engine
    ):
        executor = FakeQueryExecutor(canned_rows=[])
        await discoverer_cls(_creds())._check_least_privilege(executor)

        sql = executor.executed_sql[0].upper()
        for mutation in ("REVOKE", "GRANT ", "ALTER ", "DROP ", "CREATE ", "UPDATE ", "DELETE "):
            assert mutation not in sql, f"{engine}: privilege check must be read-only"
        assert sql.strip().startswith("SELECT")

    @pytest.mark.asyncio
    async def test_postgres_uses_effective_permissions_not_the_grant_table(self):
        """`has_table_privilege` catches a superuser; scanning
        information_schema.table_privileges would not."""
        executor = FakeQueryExecutor(canned_rows=[])
        await PostgreSQLDiscoverer(_creds())._check_least_privilege(executor)
        sql = executor.executed_sql[0]
        assert "has_table_privilege" in sql
        assert "current_user" in sql
        assert "pg_catalog" in sql and "information_schema" in sql  # excluded schemas

    @pytest.mark.asyncio
    async def test_sqlserver_uses_has_perms_by_name_so_db_datareader_is_caught(self):
        """A login whose SELECT comes from db_datareader membership leaves no
        row in sys.database_permissions — the most common real case."""
        executor = FakeQueryExecutor(canned_rows=[])
        await SQLServerDiscoverer(_creds())._check_least_privilege(executor)
        sql = executor.executed_sql[0]
        assert "HAS_PERMS_BY_NAME" in sql
        assert "is_ms_shipped = 0" in sql
        assert "'sys', 'INFORMATION_SCHEMA'" in sql

    @pytest.mark.asyncio
    async def test_mysql_unions_all_three_privilege_levels(self):
        """TABLE_PRIVILEGES alone would miss `GRANT SELECT ON *.*` and
        `GRANT SELECT ON db.*`, which grant strictly more access."""
        executor = FakeQueryExecutor(canned_rows=[])
        await MySQLDiscoverer(_creds())._check_least_privilege(executor)
        sql = executor.executed_sql[0]
        assert "USER_PRIVILEGES" in sql
        assert "SCHEMA_PRIVILEGES" in sql
        assert "TABLE_PRIVILEGES" in sql
        assert "CURRENT_USER()" in sql

    @pytest.mark.parametrize(("discoverer_cls", "engine"), _DISCOVERERS)
    @pytest.mark.asyncio
    async def test_the_scan_is_row_capped(self, discoverer_cls, engine):
        executor = FakeQueryExecutor(canned_rows=[])
        await discoverer_cls(_creds())._check_least_privilege(executor)
        assert executor.executed_params[0] == {"limit": LEAST_PRIVILEGE_SCAN_LIMIT}


class TestFailureIsNeverReportedAsACleanBillOfHealth:
    @pytest.mark.asyncio
    async def test_a_refused_privilege_view_yields_checked_false_not_clean(self):
        """A check that could not run must be distinguishable from one that
        ran and found nothing."""
        executor = FakeQueryExecutor(fetch_error=RuntimeError("permission denied for view"))
        discoverer = PostgreSQLDiscoverer(_creds())

        finding = await run_least_privilege_check(discoverer, executor, server_id="pg-1")

        assert finding.checked is False
        assert finding.has_user_table_select is False
        # ...and therefore says nothing, rather than claiming all is well.
        assert finding.warning_text() is None

    @pytest.mark.asyncio
    async def test_a_failing_check_never_aborts_the_crawl(self):
        executor = FakeQueryExecutor(fetch_error=RuntimeError("boom"))
        # Must not raise.
        finding = await run_least_privilege_check(
            MySQLDiscoverer(_creds()), executor, server_id="mysql-1"
        )
        assert finding.checked is False


class TestFindingShaping:
    def test_hitting_the_scan_cap_reports_an_honest_lower_bound(self):
        rows = [{"schema_name": "app", "object_name": f"t{i}"} for i in range(LEAST_PRIVILEGE_SCAN_LIMIT)]
        finding = build_least_privilege_finding(rows, login="diag", scope_note="n/a")
        assert finding.count_is_lower_bound is True
        assert "at least" in finding.warning_text()

    def test_below_the_cap_the_count_is_exact(self):
        finding = build_least_privilege_finding(_GRANTED_ROWS, login="diag", scope_note="n/a")
        assert finding.count_is_lower_bound is False
        assert "at least" not in finding.warning_text()

    def test_a_single_grant_reads_naturally(self):
        finding = build_least_privilege_finding(_GRANTED_ROWS[:1], login="diag", scope_note="n/a")
        assert "1 user table/view " in finding.warning_text()

    def test_only_object_names_are_carried_never_row_data(self):
        """The sample exists to point a DBA at where to look. It is built
        purely from catalog identifiers."""
        finding = build_least_privilege_finding(
            [{"schema_name": "public", "object_name": "accounts"}],
            login="diag",
            scope_note="n/a",
        )
        assert finding.sample_objects == ["public.accounts"]


class TestTheFindingRoundTripsThroughTheCatalog:
    def test_a_catalog_without_the_field_still_loads(self):
        """Catalogs discovered before this check existed must not break."""
        cat = ServerCatalog.model_validate({"server_id": "s1"})
        assert cat.least_privilege is None

    def test_the_finding_survives_serialization(self):
        cat = ServerCatalog(
            server_id="s1",
            least_privilege=build_least_privilege_finding(
                _GRANTED_ROWS, login="diag", scope_note="scoped"
            ),
        )
        reloaded = ServerCatalog.model_validate(cat.model_dump(mode="json"))
        assert reloaded.least_privilege is not None
        assert reloaded.least_privilege.granted_object_count == 2
        assert reloaded.least_privilege.warning_text() is not None

    def test_warning_text_is_the_single_source_both_surfaces_render(self):
        finding = LeastPrivilegeFinding(
            checked=True, login="diag", has_user_table_select=True, granted_object_count=3
        )
        text = finding.warning_text()
        assert text.startswith("⚠️")
        assert "diag" in text and "3 user table/views" in text
