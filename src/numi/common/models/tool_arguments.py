"""Typed argument models for every registered tool.

These are the ONLY shapes a tool call's `arguments` dict may take. The Agent
must produce structured output conforming to one of these models (spec §35);
the Gateway re-validates independently server-side regardless of what the
Agent claims to have validated (spec §9: "Never allow arbitrary tool
arguments").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Read-only tools --------------------------------------------------------


class NoArgs(_StrictArgs):
    """Used by tools that require no arguments beyond the target."""


class TopQueriesArgs(_StrictArgs):
    order_by: str = Field(default="cpu", pattern="^(cpu|duration|reads|writes|executions)$")
    limit: int = Field(default=10, ge=1, le=100)


class QueryPlanArgs(_StrictArgs):
    query_id: str = Field(min_length=1, max_length=128)


class ErrorLogArgs(_StrictArgs):
    since_minutes: int = Field(default=60, ge=1, le=1440)
    limit: int = Field(default=100, ge=1, le=1000)


class ReadOnlySqlArgs(_StrictArgs):
    """Disabled by default (spec §21). Even when enabled, the statement is
    parsed and validated by a real SQL parser, never executed as raw text."""

    sql: str = Field(min_length=1, max_length=4000)


# --- Controlled write tools --------------------------------------------------


class CancelQueryArgs(_StrictArgs):
    session_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=5, max_length=500)


class KillSessionArgs(_StrictArgs):
    session_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=5, max_length=500)


class UpdateStatisticsArgs(_StrictArgs):
    schema_name: str = Field(min_length=1, max_length=128, alias="schema")
    table: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class CreateIndexArgs(_StrictArgs):
    schema_name: str = Field(min_length=1, max_length=128, alias="schema")
    table: str = Field(min_length=1, max_length=128)
    columns: list[str] = Field(min_length=1, max_length=16)
    name: str = Field(min_length=1, max_length=128)
    unique: bool = False
    reason: str = Field(min_length=5, max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RebuildIndexArgs(_StrictArgs):
    schema_name: str = Field(min_length=1, max_length=128, alias="schema")
    table: str = Field(min_length=1, max_length=128)
    index_name: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ModifyConfigurationArgs(_StrictArgs):
    parameter: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=5, max_length=500)


class RestartInstanceArgs(_StrictArgs):
    reason: str = Field(min_length=5, max_length=500)


class FailoverArgs(_StrictArgs):
    target_instance: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)


# --- Restricted tools (framework only; disabled by default, spec §8/§21) ---


class ExecuteSqlArgs(_StrictArgs):
    sql: str = Field(min_length=1, max_length=4000)
    reason: str = Field(min_length=5, max_length=500)


class RestoreDatabaseArgs(_StrictArgs):
    backup_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)


class CreateDatabaseArgs(_StrictArgs):
    database_name: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)


class DropDatabaseArgs(_StrictArgs):
    database_name: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)


class TruncateTableArgs(_StrictArgs):
    schema_name: str = Field(min_length=1, max_length=128, alias="schema")
    table: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=5, max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class BulkDeleteArgs(_StrictArgs):
    schema_name: str = Field(min_length=1, max_length=128, alias="schema")
    table: str = Field(min_length=1, max_length=128)
    predicate_description: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=5, max_length=500)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
