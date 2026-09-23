from __future__ import annotations

import pytest

from numi.common.models.failures import FailureCode, NumiError
from numi.gateway.domain.sql_validator import validate_readonly_sql


def test_valid_select_passes_and_gets_row_capped():
    result = validate_readonly_sql(
        "SELECT id, name FROM dbo.Customers", dialect="tsql", max_result_rows=50
    )
    assert "TOP" in result.normalized_sql.upper() or "LIMIT" in result.normalized_sql.upper()
    assert "customers" in [t.lower() for t in result.referenced_tables]


def test_existing_limit_larger_than_cap_is_reduced():
    result = validate_readonly_sql(
        "SELECT * FROM users LIMIT 10000", dialect="postgres", max_result_rows=100
    )
    assert "LIMIT 100" in result.normalized_sql.upper()


def test_rejects_multiple_statements_stacked_query_injection():
    with pytest.raises(NumiError) as exc:
        validate_readonly_sql(
            "SELECT * FROM users; DROP TABLE users;", dialect="postgres", max_result_rows=100
        )
    assert exc.value.code == FailureCode.SECURITY_BLOCKED


def test_rejects_drop_database_via_readonly_sql_tool():
    with pytest.raises(NumiError) as exc:
        validate_readonly_sql("DROP DATABASE CoreBanking", dialect="tsql", max_result_rows=100)
    assert exc.value.code == FailureCode.SECURITY_BLOCKED


def test_rejects_insert_statement():
    with pytest.raises(NumiError):
        validate_readonly_sql(
            "INSERT INTO users (id) VALUES (1)", dialect="postgres", max_result_rows=100
        )


def test_rejects_update_statement():
    with pytest.raises(NumiError):
        validate_readonly_sql(
            "UPDATE users SET password = 'x'", dialect="postgres", max_result_rows=100
        )


def test_rejects_dangerous_function_xp_cmdshell():
    with pytest.raises(NumiError) as exc:
        validate_readonly_sql(
            "SELECT 1 WHERE 1 = (SELECT xp_cmdshell('dir'))", dialect="tsql", max_result_rows=100
        )
    assert exc.value.code == FailureCode.SECURITY_BLOCKED


def test_rejects_dangerous_function_pg_read_file():
    with pytest.raises(NumiError) as exc:
        validate_readonly_sql(
            "SELECT pg_read_file('/etc/passwd')", dialect="postgres", max_result_rows=100
        )
    assert exc.value.code == FailureCode.SECURITY_BLOCKED


def test_rejects_select_into():
    with pytest.raises(NumiError) as exc:
        validate_readonly_sql(
            "SELECT * INTO new_table FROM users", dialect="tsql", max_result_rows=100
        )
    assert exc.value.code == FailureCode.SECURITY_BLOCKED


def test_rejects_oversized_statement():
    huge = "SELECT " + ",".join(f"col{i}" for i in range(2000)) + " FROM t"
    with pytest.raises(NumiError) as exc:
        validate_readonly_sql(huge, dialect="postgres", max_result_rows=100)
    assert exc.value.code == FailureCode.INVALID_ARGUMENTS
