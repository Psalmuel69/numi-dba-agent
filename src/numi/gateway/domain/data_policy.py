"""Data Minimization / Data Policy Layer (spec §22).

Database results are untrusted and potentially sensitive. Nothing coming
back from the Execution Service reaches the Agent (and therefore the LLM)
without passing through here first: sensitive fields are masked, free-text
SQL fields have their literal values scrubbed, rows and columns are capped,
and the whole result is capped in size.

Field classification is configurable, not a hardcoded finite list —
`DEFAULT_SENSITIVE_FIELD_PATTERNS` and
`DEFAULT_FREE_TEXT_SQL_FIELD_PATTERNS` are sensible defaults the config can
extend.

## Two field classes, two different treatments

There are two genuinely different kinds of dangerous field, and collapsing
them into one pattern list would ruin both:

- **Sensitive by name** (`sensitive_field_patterns`): the field IS a secret
  — `password_hash`, `ssn`, `account_number`. Nothing about its value is
  worth showing a DBA, so the whole value becomes `***MASKED***`.
- **Free-text SQL by name** (`free_text_sql_field_patterns`): the field's
  NAME is innocuous but its CONTENTS are a verbatim statement or log line
  that may quote real data — `query_text`, `blocked_query`,
  `deadlock_graph`, `message`. Masking these outright would delete exactly
  the diagnostic the DBA asked for; leaving them alone leaks customer data.
  So the value is literal-scrubbed and keeps its shape — see
  `query_scrubber.py` for the mechanics and the `<redacted>` rationale.

Sensitive-by-name is checked FIRST and wins: a field matching both lists is
fully masked, never merely scrubbed. That ordering is what keeps this change
strictly additive — every field that was masked before is still masked,
byte for byte (pinned by
`tests/unit/test_data_policy.py::test_existing_field_name_masking_is_completely_unchanged`).

## Why this lives here and not in a second, parallel pass

The alternative considered was a separate minimization pass the Gateway
applies only to the four tool results that obviously return raw text
(`running_queries`, `top_queries`, `deadlocks`, `error_logs`). It was
rejected for two reasons:

1. **It already has a hole on day one.** `database.get_blocking_sessions`
   is not in that list, and all three adapters return `blocked_query` /
   `blocking_query` / `blocked_query_text` from it (see
   `execution/adapters/*.py::blocking`) — full statement text, carrying
   exactly the literals this is meant to catch. `get_sessions` returns
   `query_text` too. A tool-id-keyed list is a list someone has to remember
   to update every time an adapter grows a column; a field-name-keyed rule
   covers them the moment they appear.
2. **`DataMinimizer.apply()` is already the one mandatory seam.** Every
   tool result passes through it, unconditionally, at step 10 of
   `tool_call_handler.py::_handle_inner` — there is no code path around it.
   A second pass would be a second thing to remember to call, drifting out
   of sync with the first.

So this extends the established mechanism rather than running beside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from numi.gateway.domain.query_scrubber import scrub_sql_literals

DEFAULT_SENSITIVE_FIELD_PATTERNS: list[str] = [
    r"password",
    r"pass[_-]?hash",
    r"token",
    r"secret",
    r"api[_-]?key",
    r"card[_-]?number",
    r"account[_-]?number",
    r"\bbvn\b",
    r"\bnin\b",
    r"ssn",
    r"phone",
    r"e[-_]?mail",
    r"address",
    r"auth",
    r"credential",
    r"connection[_-]?string",
]

#: Fields whose VALUE is free text that may embed literals — scrubbed, not
#: masked. Every entry is anchored against a column name a real adapter
#: actually returns; the trailing comment names it, so this list can be
#: audited against `execution/adapters/*.py` rather than taken on faith.
#:
#: The anchoring is load-bearing. `(^|_)query$` deliberately matches
#: `query` / `blocked_query` / `blocking_query` but NOT `query_id`,
#: `queryid`, `query_start`, `query_hash` or `query_plan` — those are
#: identifiers and timestamps the DBA correlates on, and scrubbing them
#: would break `get_query_plan`'s own argument round-trip. A bare
#: `r"query"` would have swallowed all of them.
DEFAULT_FREE_TEXT_SQL_FIELD_PATTERNS: list[str] = [
    r"query_text",  # pg/mysql/sqlserver: running_queries, top_queries, sessions
    r"(^|_)query$",  # sqlserver/mysql blocking(): blocked_query, blocking_query
    r"(^|_)sql(_text)?$",  # generic sql / sql_text columns
    r"(^|_)statement(_text)?$",
    r"(^|_)text$",  # sqlserver: sys.dm_exec_sql_text's raw `text` column
    r"query_plan",  # sqlserver query_plan(): Query Store plan XML
    r"plan_summary",
    r"deadlock_graph",  # sqlserver deadlocks()
    r"latest_detected_deadlock",  # mysql deadlocks(): SHOW ENGINE INNODB STATUS blob
    r"(^|_)message$",  # mysql error_logs(): DATA AS message; also error_message
    r"log_line",
    r"log_text",
]

#: Platform value (`common.models.target.Platform`) -> sqlglot dialect.
#: Mirrors the inline mapping `tool_call_handler.py` already uses for
#: `validate_readonly_sql`, extended to MySQL/MariaDB (which that call site
#: never needed because the read-only SQL tool is SQL Server/Postgres-only
#: in practice). An unknown platform maps to `None` — sqlglot's
#: dialect-agnostic parse, which still handles ordinary SQL correctly.
_SQLGLOT_DIALECTS: dict[str, str] = {
    "sqlserver": "tsql",
    "postgresql": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",
}


def sqlglot_dialect_for_platform(platform: str | None) -> str | None:
    """Best-effort platform -> sqlglot dialect. Never raises; `None` is a
    valid, slightly-less-precise answer, not an error."""
    if not platform:
        return None
    return _SQLGLOT_DIALECTS.get(platform.lower())


@dataclass
class DataPolicyConfig:
    sensitive_field_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_FIELD_PATTERNS)
    )
    # Separate from `sensitive_field_patterns` on purpose — different
    # treatment, not a longer list of the same thing. See the module
    # docstring's "Two field classes, two different treatments".
    free_text_sql_field_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_FREE_TEXT_SQL_FIELD_PATTERNS)
    )
    field_allowlist: set[str] | None = None  # if set, ONLY these fields pass through
    field_denylist: set[str] = field(default_factory=set)
    max_rows: int = 100
    max_columns: int = 50
    max_result_bytes: int = 512_000


@dataclass(frozen=True)
class MinimizedResult:
    """What one minimization pass did, in full.

    `apply()` keeps returning its original 3-tuple so no existing caller or
    test changes shape; `minimize()` is the richer seam that also reports
    `literal_scrubbed_fields`, which the Gateway surfaces in the tool result
    so the DBA (and the LLM reading it) can tell "this statement had its
    values removed on purpose" from "this statement genuinely had no
    literals".
    """

    rows: list[dict[str, Any]]
    masked_fields: list[str]
    truncated: bool
    literal_scrubbed_fields: list[str]


class DataMinimizer:
    def __init__(self, config: DataPolicyConfig | None = None):
        self._config = config or DataPolicyConfig()
        self._sensitive_re = re.compile(
            "|".join(self._config.sensitive_field_patterns), re.IGNORECASE
        )
        # Guarded against an empty pattern list: `"|".join([])` is `""`,
        # which compiles to a regex matching EVERY field name — that would
        # silently scrub the entire result set. An operator who configures
        # no free-text patterns means "scrub nothing".
        patterns = self._config.free_text_sql_field_patterns
        self._free_text_re = (
            re.compile("|".join(patterns), re.IGNORECASE) if patterns else None
        )

    def _is_sensitive(self, field_name: str) -> bool:
        return bool(self._sensitive_re.search(field_name))

    def _is_free_text_sql(self, field_name: str) -> bool:
        if self._free_text_re is None:
            return False
        return bool(self._free_text_re.search(field_name))

    def minimize(
        self,
        rows: list[dict[str, Any]],
        *,
        max_rows: int | None = None,
        dialect: str | None = None,
    ) -> MinimizedResult:
        """Full minimization pass. `dialect` is a sqlglot dialect name for
        the literal scrubber (see `sqlglot_dialect_for_platform`); omitting
        it still scrubs, just dialect-agnostically."""
        cfg = self._config
        row_cap = max_rows or cfg.max_rows
        truncated = len(rows) > row_cap
        rows = rows[:row_cap]

        masked_fields: set[str] = set()
        scrubbed_fields: set[str] = set()
        result: list[dict[str, Any]] = []
        for row in rows:
            clean: dict[str, Any] = {}
            col_count = 0
            for k, v in row.items():
                if cfg.field_allowlist is not None and k not in cfg.field_allowlist:
                    continue
                if k in cfg.field_denylist:
                    continue
                if col_count >= cfg.max_columns:
                    truncated = True
                    break
                if self._is_sensitive(k):
                    # Checked first — a field matching both lists is fully
                    # masked, never merely scrubbed.
                    clean[k] = "***MASKED***"
                    masked_fields.add(k)
                elif isinstance(v, str) and self._is_free_text_sql(k):
                    # `isinstance(v, str)` is not incidental: PostgreSQL's
                    # `deadlocks()` returns a numeric `deadlocks` COUNTER,
                    # and several engines return NULL for an idle session's
                    # query text. Only actual text can carry a literal, and
                    # running a scrubber over an int would destroy a metric
                    # while protecting nothing.
                    scrubbed = scrub_sql_literals(v, dialect=dialect)
                    clean[k] = scrubbed
                    if scrubbed != v:
                        scrubbed_fields.add(k)
                else:
                    clean[k] = v
                col_count += 1
            result.append(clean)

        return MinimizedResult(
            rows=result,
            masked_fields=sorted(masked_fields),
            truncated=truncated,
            literal_scrubbed_fields=sorted(scrubbed_fields),
        )

    def apply(
        self, rows: list[dict[str, Any]], *, max_rows: int | None = None
    ) -> tuple[list[dict[str, Any]], list[str], bool]:
        """Returns (minimized_rows, masked_field_names, truncated).

        Kept at its original 3-tuple shape so every existing caller and test
        is untouched by the literal-scrubbing addition; `minimize()` is the
        richer seam the Gateway itself uses.
        """
        outcome = self.minimize(rows, max_rows=max_rows)
        return outcome.rows, outcome.masked_fields, outcome.truncated
