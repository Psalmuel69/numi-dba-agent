"""Read-only SQL validation (spec §21).

Backs the `database.execute_readonly_sql` tool, which is disabled by
default (spec §8) and, even when explicitly enabled, never executes text the
LLM produced without this module parsing a real AST and rejecting anything
that is not a single, side-effect-free `SELECT`. Regex alone is explicitly
disallowed by the spec — this uses `sqlglot`'s real parser.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from numi.common.models.failures import FailureCode, NumiError

# Functions/constructs that can exfiltrate data, touch the filesystem, or
# otherwise escape the "just reads rows" contract even inside a SELECT.
_DANGEROUS_FUNCTIONS = {
    "xp_cmdshell",
    "sp_configure",
    "opendatasource",
    "openrowset",
    "openquery",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "lo_import",
    "lo_export",
    "dblink",
    "load_file",
    "into_outfile",
    "sleep",
    "pg_sleep",
    "waitfor",
    "benchmark",
}

_MAX_STATEMENT_LENGTH = 4000


@dataclass(frozen=True)
class ValidatedReadOnlySql:
    normalized_sql: str
    referenced_tables: list[str]


def validate_readonly_sql(
    sql: str, *, dialect: str, max_result_rows: int
) -> ValidatedReadOnlySql:
    """Raises NumiError(SECURITY_BLOCKED | INVALID_ARGUMENTS) or returns a
    validated, row-limited, single SELECT statement's AST-derived summary."""
    if len(sql) > _MAX_STATEMENT_LENGTH:
        raise NumiError(FailureCode.INVALID_ARGUMENTS, "SQL statement is too long.")

    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001 — any parser error is a rejection, not a crash
        raise NumiError(FailureCode.INVALID_ARGUMENTS, f"Could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise NumiError(
            FailureCode.SECURITY_BLOCKED,
            "Only a single SQL statement is permitted — reject multi-statement input.",
        )

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise NumiError(
            FailureCode.SECURITY_BLOCKED,
            "Only SELECT statements are permitted through execute_readonly_sql.",
        )

    # No CTEs with side effects, no INTO clause (SELECT ... INTO creates a
    # table in some dialects), no set operations hiding a second statement
    # via UNION with a DML branch (sqlglot would have already rejected DML
    # here, but INTO is checked explicitly since it parses as a SELECT).
    if statement.args.get("into") is not None:
        raise NumiError(FailureCode.SECURITY_BLOCKED, "SELECT ... INTO is not permitted.")

    for func in statement.find_all(exp.Anonymous, exp.Func):
        name = (func.name or "").lower()
        if name in _DANGEROUS_FUNCTIONS:
            raise NumiError(
                FailureCode.SECURITY_BLOCKED, f"Use of function '{name}' is not permitted."
            )

    for node in statement.walk():
        n = node[0] if isinstance(node, tuple) else node
        text = str(getattr(n, "this", "")).lower() if hasattr(n, "this") else ""
        if any(bad in text for bad in _DANGEROUS_FUNCTIONS):
            raise NumiError(FailureCode.SECURITY_BLOCKED, "Statement references a disallowed construct.")

    tables = sorted({t.name for t in statement.find_all(exp.Table)})

    # Enforce a row cap regardless of what the caller asked for.
    existing_limit = statement.args.get("limit")
    if existing_limit is None:
        statement = statement.limit(max_result_rows)
    else:
        try:
            requested = int(existing_limit.expression.this)
            if requested > max_result_rows:
                statement = statement.limit(max_result_rows)
        except (AttributeError, ValueError, TypeError):
            statement = statement.limit(max_result_rows)

    return ValidatedReadOnlySql(normalized_sql=statement.sql(dialect=dialect), referenced_tables=tables)
