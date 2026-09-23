"""Canonical database target model (spec §10).

This is the *only* shape a "where should this run" value is allowed to take
anywhere in the system. The LLM never invents a connection string or host —
it fills in fields of this model, and the Gateway resolves/validates every
field against the database inventory (spec §11) before anything downstream
sees it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Environment(str, Enum):
    DEVELOPMENT = "development"
    UAT = "uat"
    PRODUCTION = "production"


class Platform(str, Enum):
    SQLSERVER = "sqlserver"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    # Reserved for future adapters — registering these does not grant them
    # any capability; an adapter must exist and be registered separately.
    ORACLE = "oracle"
    MARIADB = "mariadb"


class DatabaseTarget(BaseModel):
    """A partially- or fully-specified pointer into the database inventory.

    Fields are optional individually because different tools require
    different levels of specificity (spec §10) — but every field that *is*
    supplied must resolve to a real inventory entry; free-text hostnames or
    connection strings are never accepted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: Environment
    platform: Platform | None = None
    cluster: str | None = None
    instance: str | None = None
    database: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    object_name: str | None = Field(default=None, alias="object")

    # Operation-specific scoping identifiers. These are *not* free network
    # addresses — they are validated by the Gateway against live server state
    # returned from prior read-only tool calls, never taken as ground truth
    # from the LLM alone for write operations.
    session_id: str | None = None
    query_id: str | None = None

    @model_validator(mode="after")
    def _require_instance_scope(self) -> DatabaseTarget:
        # At minimum every target must identify an environment; the Gateway's
        # target-validation stage enforces the operation-specific minimum
        # field set (see gateway.domain.target_validation).
        return self

    def scope_key(self) -> str:
        """A stable string identifying the narrowest scope this target names.

        Used for rate limiting and audit correlation, never for authorization
        decisions on its own.
        """
        parts = [
            self.environment.value,
            self.instance or "*",
            self.database or "*",
            self.schema_name or "*",
            self.object_name or "*",
        ]
        return ":".join(parts)


class RequiredTargetFields(str, Enum):
    """Named minimum field-sets a tool can require (spec §10)."""

    INSTANCE_LEVEL = "instance_level"  # environment, instance
    DATABASE_LEVEL = "database_level"  # environment, instance, database
    SESSION_LEVEL = "session_level"  # + session_id
    QUERY_LEVEL = "query_level"  # + query_id
    OBJECT_LEVEL = "object_level"  # + schema, object


REQUIRED_FIELD_SETS: dict[RequiredTargetFields, tuple[str, ...]] = {
    RequiredTargetFields.INSTANCE_LEVEL: ("environment", "instance"),
    RequiredTargetFields.DATABASE_LEVEL: ("environment", "instance", "database"),
    RequiredTargetFields.SESSION_LEVEL: (
        "environment",
        "instance",
        "database",
        "session_id",
    ),
    RequiredTargetFields.QUERY_LEVEL: (
        "environment",
        "instance",
        "database",
        "query_id",
    ),
    RequiredTargetFields.OBJECT_LEVEL: (
        "environment",
        "instance",
        "database",
        "schema_name",
        "object_name",
    ),
}
