"""Query-literal scrubbing — strip VALUES out of free-text SQL, keep SHAPE.

`data_policy.DataMinimizer` masks a whole field when its *name* looks
sensitive (`password`, `ssn`, `account_number`, ...). That is the right
treatment for a column that is nothing but a secret, and it is useless for
the case this module exists for: a field whose name is entirely innocuous
but whose *contents* are a verbatim SQL statement a user actually ran.

`database.get_running_queries` on PostgreSQL returns a `query_text` column
straight out of `pg_stat_activity` (see
`execution/adapters/postgresql.py::running_queries`); the SQL Server and
MySQL adapters return the same shape from `sys.dm_exec_sql_text` and
`information_schema.PROCESSLIST`. Nothing about the *name* "query_text" is
sensitive, so `DataMinimizer` had no reason to touch it — and

    SELECT * FROM accounts WHERE account_number = '1234567890'
                                AND customer_name = 'Jane Doe'

sailed through to the LLM, and from there into a Slack channel, intact.
That is real customer data leaving the database, through the one component
whose entire premise (ARCHITECTURE.md, SECURITY.md control 13) is that
Numi reads diagnostics and never table contents.

The fix has to keep the statement *useful*. A DBA diagnosing a slow or
deadlocked query needs the shape — which tables, which columns, which
joins, which predicates, is there a LIKE '%...' , is the ORDER BY
unindexed. None of that is the literal values. So: parse with `sqlglot`
(the same real parser `sql_validator.py` already uses for the read-only SQL
tool — see that module for the house pattern) and replace every literal
node with a placeholder, leaving table names, column names, keywords and
structure untouched.

**Placeholder choice: `<redacted>`, not `?`.** `?` is the obvious first
instinct and is wrong here: `?` is itself a real bind-parameter marker in
the ODBC/MySQL dialects this system talks to, so a statement scrubbed to
`... WHERE account_number = ?` is indistinguishable from one the
application genuinely sent parameterized. A DBA (and the LLM) could not
tell "Numi removed a value here" from "the app used a bind parameter
here" — and those call for different diagnoses. `<redacted>` can never be
mistaken for something the application wrote, and it matches the existing
`***MASKED***` convention in `data_policy.py`: say plainly that something
was deliberately removed. It is emitted as a *string* literal in every
position (numbers included) so the scrubbed statement still parses as
valid SQL for anyone who wants to re-parse it downstream.

Every literal is replaced, including ones that are obviously harmless in
isolation (a `LIMIT 100`, a `WHERE status = 1`). That is deliberate: the
alternative is a judgment call, per literal, about whether a value is
sensitive — exactly the kind of guess this codebase refuses to make
elsewhere, and the one that fails silently and unrecoverably when it
guesses wrong. Shape is what the DBA needs; values are what must not
leave.

## The fallback path is deliberately less precise

Not everything flowing through these fields is parseable SQL. A
PostgreSQL error-log line, SQL Server's `deadlock_graph` XML, MySQL's
`SHOW ENGINE INNODB STATUS` text blob, a plan's text representation, or a
genuinely truncated fragment (the adapters themselves cut query text at
`left(query, 200)` / `LEFT(INFO, 500)`, so a *valid* statement can arrive
chopped mid-literal) are all normal, expected inputs here. Dropping the
row or raising would blind the DBA to exactly the diagnostics they asked
for, so `scrub_sql_literals` falls back to `_regex_scrub` instead.

That fallback is a blunt instrument and is meant to be: with no AST there
is no way to tell a value from an identifier, so it over-redacts rather
than under-redacts (single-quoted runs and bare numeric runs go, including
timestamps and PIDs in a log line). Two deliberate non-obvious choices:

- Double-quoted text is left alone. In standard SQL a double-quoted token
  is an *identifier*, and in Postgres log text it is almost always the
  object name the DBA actually needs (`violates unique constraint
  "accounts_pkey"`). Redacting it would destroy the diagnostic without
  protecting a value.
- sqlglot is tried FIRST and only a genuine parse failure reaches the
  fallback, so anything that really is SQL gets the precise treatment.

## Why sqlglot's parse result is gated on the statement TYPE

sqlglot is permissive by design, and `ErrorLevel.RAISE` alone is not
enough. Verified against sqlglot 30.17: `"Jane Doe"` parses cleanly as an
`exp.Alias` and round-trips as `Jane AS Doe` — the value survives
unredacted AND the text is mangled. `"1234567890"` parses as a bare
`exp.Literal`. Accepting those would be strictly worse than the fallback.
So a parse counts only when the root node is an actual *statement*
(`_SQL_STATEMENT_TYPES`); a bare expression or fragment is treated as
unparseable and handed to `_regex_scrub`, which redacts both of those
examples correctly.

Covered by `tests/unit/test_query_scrubber.py` and, at the
`DataMinimizer` level, `tests/unit/test_data_policy.py`.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

#: What replaces every literal. See the module docstring for why this is not `?`.
REDACTION_PLACEHOLDER = "<redacted>"

#: Root node types that mean "sqlglot understood this as a real statement".
#: A parse producing anything else (Alias, Column, bare Literal, ...) is a
#: fragment, not SQL, and goes to the regex fallback instead — see the
#: module docstring's `Jane Doe` / `1234567890` findings.
_SQL_STATEMENT_TYPES = (
    exp.Query,  # Select, Union/Intersect/Except, Subquery (sqlglot >= 20)
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Show,
    exp.Command,  # SHOW/EXEC-style statements sqlglot keeps verbatim
)

#: Fallback literal shapes, tried in order within one pass so that a number
#: *inside* a quoted string is consumed by the string branch first and never
#: double-processed. Single-quoted only (see module docstring on why
#: double-quoted identifiers are preserved); `''` is SQL's own escape for an
#: embedded quote, so it must not terminate the run.
_FALLBACK_LITERAL_RE = re.compile(
    r"'(?:[^']|'')*'"  # single-quoted string literal
    r"|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"  # bare integer / decimal run
)


def _redact_literal(node: exp.Expression) -> exp.Expression:
    """`Expression.transform` callback: every literal becomes the placeholder.

    Only `exp.Literal` is replaced — `exp.Null`, `exp.Boolean`, identifiers,
    table and column names are all structure, not values, and are what makes
    the scrubbed statement still worth reading.
    """
    if isinstance(node, exp.Literal):
        return exp.Literal.string(REDACTION_PLACEHOLDER)
    return node


def _regex_scrub(text: str) -> str:
    """Best-effort, deliberately imprecise scrub for text that is not SQL.

    See the module docstring: this runs only when sqlglot could not produce
    a real statement, where there is no structure to distinguish a value
    from an identifier. It over-redacts on purpose.
    """
    return _FALLBACK_LITERAL_RE.sub(REDACTION_PLACEHOLDER, text)


def scrub_sql_literals(text: str, *, dialect: str | None = None) -> str:
    """Return `text` with every literal value replaced, shape preserved.

    Never raises and never returns `None`: this sits on the Gateway's
    mandatory data-minimization path (`tool_call_handler.py` step 10), where
    a crash would turn a successful diagnostic into a FAILED tool call. Any
    input that cannot be parsed as a statement is regex-scrubbed instead.

    `dialect` is a sqlglot dialect name (`postgres` / `tsql` / `mysql`) —
    see `data_policy.sqlglot_dialect_for_platform`. `None` parses with
    sqlglot's dialect-agnostic default, which is correct but slightly less
    precise on engine-specific syntax.
    """
    if not text or not text.strip():
        return text

    try:
        parsed = sqlglot.parse_one(text, read=dialect, error_level=sqlglot.ErrorLevel.RAISE)
    except Exception:  # noqa: BLE001 — ANY parser/tokenizer failure means "not SQL", never a crash
        return _regex_scrub(text)

    if parsed is None or not isinstance(parsed, _SQL_STATEMENT_TYPES):
        return _regex_scrub(text)

    try:
        return parsed.transform(_redact_literal).sql(dialect=dialect)
    except Exception:  # noqa: BLE001 — a generator failure must not lose the row either
        return _regex_scrub(text)
