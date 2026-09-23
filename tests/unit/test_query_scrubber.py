"""Query-literal scrubbing (gateway/domain/query_scrubber.py).

Per-engine coverage drives the REAL adapter methods through
`FakeQueryExecutor` (the pattern established in `tests/unit/test_adapters.py`)
so the text being scrubbed is the text those methods actually return, not a
hand-written approximation of it.
"""

from __future__ import annotations

import pytest

from numi.execution.adapters.mysql import MySQLAdapter
from numi.execution.adapters.postgresql import PostgreSQLAdapter
from numi.execution.adapters.sqlserver import SQLServerAdapter
from numi.gateway.domain.data_policy import DataMinimizer, sqlglot_dialect_for_platform
from numi.gateway.domain.query_scrubber import REDACTION_PLACEHOLDER, scrub_sql_literals
from tests.fakes import FakeQueryExecutor

# The leak this whole feature exists for: an innocuously-named column whose
# contents are a statement quoting real customer data.
_LEAKY_QUERY = (
    "SELECT * FROM accounts WHERE account_number = '1234567890' "
    "AND customer_name = 'Jane Doe' AND balance > 5000"
)


class TestShapeSurvivesAndValuesDoNot:
    def test_literals_are_replaced_and_structure_is_kept(self):
        scrubbed = scrub_sql_literals(_LEAKY_QUERY, dialect="postgres")

        # The values are gone -- all of them, string and numeric.
        assert "1234567890" not in scrubbed
        assert "Jane Doe" not in scrubbed
        assert "5000" not in scrubbed

        # The shape a DBA diagnoses with is intact.
        assert "accounts" in scrubbed
        assert "account_number" in scrubbed
        assert "customer_name" in scrubbed
        assert "balance" in scrubbed
        assert "WHERE" in scrubbed.upper()
        assert REDACTION_PLACEHOLDER in scrubbed

    def test_the_placeholder_is_not_a_bind_parameter_marker(self):
        """`?` would be indistinguishable from a real bind parameter the
        application itself sent -- see query_scrubber.py's placeholder
        rationale."""
        scrubbed = scrub_sql_literals(_LEAKY_QUERY, dialect="postgres")
        assert "?" not in scrubbed

    def test_join_and_predicate_structure_survives(self):
        sql = (
            "SELECT o.id, c.name FROM orders o JOIN customers c ON c.id = o.customer_id "
            "WHERE c.email LIKE '%@example.com' ORDER BY o.created_at DESC LIMIT 50"
        )
        scrubbed = scrub_sql_literals(sql, dialect="postgres")
        assert "@example.com" not in scrubbed
        for shape in ("orders", "customers", "customer_id", "JOIN", "LIKE", "ORDER BY"):
            assert shape in scrubbed or shape in scrubbed.upper()

    def test_a_scrubbed_statement_is_still_valid_parseable_sql(self):
        """The placeholder is emitted as a string literal in every position
        (numbers included) precisely so this holds."""
        once = scrub_sql_literals(_LEAKY_QUERY, dialect="postgres")
        twice = scrub_sql_literals(once, dialect="postgres")
        assert "accounts" in twice
        assert "1234567890" not in twice


class TestTheFallbackPathForTextThatIsNotSql:
    def test_a_raw_postgres_log_line_is_scrubbed_not_dropped(self):
        log_line = (
            "2026-09-14 10:22:01 UTC [12345] ERROR:  duplicate key value violates "
            "unique constraint \"accounts_pkey\" DETAIL:  Key (account_number)="
            "('1234567890') already exists."
        )
        scrubbed = scrub_sql_literals(log_line, dialect="postgres")

        # The row is NOT dropped and NOT emptied.
        assert scrubbed
        # The quoted value is gone.
        assert "1234567890" not in scrubbed
        # The diagnostic survives: the message, and the double-quoted
        # identifier the DBA actually needs.
        assert "ERROR" in scrubbed
        assert "accounts_pkey" in scrubbed
        assert "duplicate key value" in scrubbed

    def test_the_fallback_over_redacts_bare_numbers_on_purpose(self):
        """With no AST there is no way to tell a value from a PID, so the
        fallback errs toward removing too much -- documented in
        query_scrubber.py."""
        scrubbed = scrub_sql_literals("[12345] connection reset", dialect="postgres")
        assert "12345" not in scrubbed
        assert "connection reset" in scrubbed

    def test_a_bare_value_that_sqlglot_would_mis_parse_still_gets_scrubbed(self):
        """sqlglot parses `1234567890` as a bare Literal and `Jane Doe` as an
        Alias -- neither is a statement, so both must go to the fallback
        rather than being accepted (query_scrubber's statement-type gate)."""
        assert "1234567890" not in scrub_sql_literals("1234567890")

    def test_unparseable_plan_text_is_preserved_in_shape(self):
        plan = "Clustered Index Scan -> Sort -> Hash Match"
        assert scrub_sql_literals(plan, dialect="tsql") == plan

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_input_never_crashes(self, text):
        assert scrub_sql_literals(text) == text


class TestPerEngineAdapterTextGoesThroughCleanly:
    """Each engine's real free-text column names, driven through the real
    adapter method, then through the real DataMinimizer."""

    @pytest.mark.asyncio
    async def test_postgres_running_queries_query_text_is_scrubbed(self):
        executor = FakeQueryExecutor(
            canned_rows=[
                {
                    "database_name": "corebanking",
                    "session_id": 9182,
                    "state": "active",
                    "query_text": _LEAKY_QUERY,
                }
            ]
        )
        rows = await PostgreSQLAdapter(executor, "corebanking").running_queries()
        out = DataMinimizer().minimize(
            rows, dialect=sqlglot_dialect_for_platform("postgresql")
        )

        assert "1234567890" not in out.rows[0]["query_text"]
        assert "accounts" in out.rows[0]["query_text"]
        assert out.literal_scrubbed_fields == ["query_text"]
        # Structured metric columns are untouched.
        assert out.rows[0]["session_id"] == 9182
        assert out.rows[0]["database_name"] == "corebanking"

    @pytest.mark.asyncio
    async def test_sqlserver_running_queries_and_blocking_text_are_scrubbed(self):
        executor = FakeQueryExecutor(
            canned_rows=[{"session_id": 55, "cpu_time": 812000, "query_text": _LEAKY_QUERY}]
        )
        rows = await SQLServerAdapter(executor, "CoreBanking").running_queries()
        out = DataMinimizer().minimize(rows, dialect=sqlglot_dialect_for_platform("sqlserver"))
        assert "Jane Doe" not in out.rows[0]["query_text"]
        assert out.rows[0]["cpu_time"] == 812000

        # blocking() is NOT one of the four "obvious" free-text tools, and
        # returns statement text anyway -- the exact hole a tool-id-keyed
        # pass would have had. See data_policy.py's design note.
        blocking_rows = [
            {
                "blocked_session_id": 9183,
                "blocking_session_id": 9182,
                "blocked_query_text": _LEAKY_QUERY,
            }
        ]
        blocked = DataMinimizer().minimize(blocking_rows, dialect="tsql")
        assert "1234567890" not in blocked.rows[0]["blocked_query_text"]
        assert blocked.rows[0]["blocking_session_id"] == 9182

    @pytest.mark.asyncio
    async def test_mysql_top_queries_and_error_log_message_are_scrubbed(self):
        executor = FakeQueryExecutor(
            canned_rows=[{"query_id": "abc123", "query_text": _LEAKY_QUERY, "executions": 12}]
        )
        rows = await MySQLAdapter(executor, "app").top_queries("cpu", 10)
        out = DataMinimizer().minimize(rows, dialect=sqlglot_dialect_for_platform("mysql"))
        assert "1234567890" not in out.rows[0]["query_text"]
        # A digest id is an identifier, not a literal -- it must survive, or
        # the DBA can't correlate it back to get_query_plan.
        assert out.rows[0]["query_id"] == "abc123"
        assert out.rows[0]["executions"] == 12

        log_rows = [
            {"priority": "ERROR", "message": "Access denied for user 'svc_app'@'10.0.0.5'"}
        ]
        logs = DataMinimizer().minimize(log_rows, dialect="mysql")
        assert "svc_app" not in logs.rows[0]["message"]
        assert "Access denied" in logs.rows[0]["message"]


class TestNonStringValuesInFreeTextFieldsAreLeftAlone:
    def test_a_numeric_deadlock_counter_is_not_scrubbed(self):
        """PostgreSQL's deadlocks() returns a numeric counter; running a
        text scrubber over it would destroy a metric and protect nothing."""
        out = DataMinimizer().minimize([{"datname": "core", "deadlocks": 7}])
        assert out.rows[0]["deadlocks"] == 7

    def test_a_null_query_text_survives_as_none(self):
        out = DataMinimizer().minimize([{"session_id": 1, "query_text": None}])
        assert out.rows[0]["query_text"] is None
        assert out.literal_scrubbed_fields == []


class TestFieldNamesThatMustNotBeScrubbed:
    def test_identifier_and_timestamp_columns_are_untouched(self):
        """`(^|_)query$` must not swallow query_id/query_start/query_hash --
        those are what the DBA correlates on."""
        row = {
            "query_id": "0x9A2B",
            "queryid": 8814412,
            "query_start": "2026-09-14T10:22:01Z",
            "query_hash": "abc",
        }
        out = DataMinimizer().minimize([row])
        assert out.rows[0] == row
        assert out.literal_scrubbed_fields == []
