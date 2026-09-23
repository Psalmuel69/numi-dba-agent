"""The centralized Tool Registry contents (spec §8, §9).

`build_tool_catalog` is the single place every tool the system knows about is
declared. The Agent can never call anything that is not an entry here, and
disabled/restricted tools are declared with `enabled=False` so that
attempting to invoke them fails with TOOL_NOT_AVAILABLE without ever
reaching the Execution Service (spec test §47).
"""

from __future__ import annotations

from numi.common.config import Settings
from numi.common.models.identity import DBARole
from numi.common.models.results import DiagnosticResult, WriteResult
from numi.common.models.target import Environment
from numi.common.models.tool import OperationType, ToolDefinition
from numi.common.models.tool_arguments import (
    BulkDeleteArgs,
    CancelQueryArgs,
    CreateDatabaseArgs,
    CreateIndexArgs,
    DropDatabaseArgs,
    ErrorLogArgs,
    ExecuteSqlArgs,
    FailoverArgs,
    KillSessionArgs,
    ModifyConfigurationArgs,
    NoArgs,
    QueryPlanArgs,
    ReadOnlySqlArgs,
    RebuildIndexArgs,
    RestartInstanceArgs,
    RestoreDatabaseArgs,
    TopQueriesArgs,
    TruncateTableArgs,
    UpdateStatisticsArgs,
)

_ALL_ENVS = [Environment.DEVELOPMENT, Environment.UAT, Environment.PRODUCTION]
_ALL_ROLES = [DBARole.DBA_L1, DBARole.DBA_L2, DBARole.DBA_L3, DBARole.DBA_MANAGER]
_WRITE_ROLES = [DBARole.DBA_L2, DBARole.DBA_L3, DBARole.DBA_MANAGER]
_ADVANCED_ROLES = [DBARole.DBA_L3, DBARole.DBA_MANAGER]

_READ = DiagnosticResult.model_json_schema()
_WRITE = WriteResult.model_json_schema()

# Maps tool_id -> the Pydantic model its `arguments` must validate against.
# This is the single source of truth the Gateway uses to reject malformed or
# extra arguments (spec §9: "Never allow arbitrary tool arguments") — the
# JSON Schema baked into each ToolDefinition.argument_schema is generated
# from these same models, so the two can never drift.
ARGUMENT_MODELS: dict[str, type] = {
    "database.get_health": NoArgs,
    "database.get_version": NoArgs,
    "database.get_sessions": NoArgs,
    "database.get_blocking_sessions": NoArgs,
    "database.get_deadlocks": NoArgs,
    "database.get_running_queries": NoArgs,
    "database.get_wait_statistics": NoArgs,
    "database.get_query_plan": QueryPlanArgs,
    "database.get_top_queries": TopQueriesArgs,
    "database.get_indexes": NoArgs,
    "database.get_statistics": NoArgs,
    "database.get_tables": NoArgs,
    "database.get_storage": NoArgs,
    "database.get_transaction_log": NoArgs,
    "database.get_replication_status": NoArgs,
    "database.get_backup_status": NoArgs,
    "database.get_configuration": NoArgs,
    "database.get_error_logs": ErrorLogArgs,
    "database.cancel_query": CancelQueryArgs,
    "database.kill_session": KillSessionArgs,
    "database.update_statistics": UpdateStatisticsArgs,
    "database.create_index": CreateIndexArgs,
    "database.rebuild_index": RebuildIndexArgs,
    "database.modify_configuration": ModifyConfigurationArgs,
    "database.restart_instance": RestartInstanceArgs,
    "database.failover": FailoverArgs,
    "database.execute_readonly_sql": ReadOnlySqlArgs,
    "database.execute_sql": ExecuteSqlArgs,
    "database.restore_database": RestoreDatabaseArgs,
    "database.create_database": CreateDatabaseArgs,
    "database.drop_database": DropDatabaseArgs,
    "database.truncate_table": TruncateTableArgs,
    "database.bulk_delete": BulkDeleteArgs,
}


def _read_tool(
    tool_id: str,
    description: str,
    args_model=NoArgs,
    required_scope: list[str] | None = None,
    max_result_rows: int = 100,
) -> ToolDefinition:
    return ToolDefinition(
        tool_id=tool_id,
        version="1.0.0",
        description=description,
        operation_type=OperationType.READ,
        risk_level="LOW",
        reversible=True,
        availability_impact=False,
        data_modification=False,
        requires_approval=False,
        allowed_roles=_ALL_ROLES,
        allowed_environments=_ALL_ENVS,
        required_target_scope=required_scope or ["environment", "instance", "database"],
        argument_schema=args_model.model_json_schema(),
        result_schema=_READ,
        max_execution_time=30,
        max_result_rows=max_result_rows,
        audit_required=True,
        enabled=True,
    )


def _write_tool(
    tool_id: str,
    description: str,
    args_model,
    *,
    risk_level: str,
    reversible: bool,
    availability_impact: bool,
    required_scope: list[str],
    requires_dual_approval: bool = False,
    allowed_roles: list[DBARole] | None = None,
    enabled: bool = True,
) -> ToolDefinition:
    return ToolDefinition(
        tool_id=tool_id,
        version="1.0.0",
        description=description,
        operation_type=OperationType.WRITE,
        risk_level=risk_level,
        reversible=reversible,
        availability_impact=availability_impact,
        data_modification=True,
        requires_approval=True,
        allowed_roles=allowed_roles or _WRITE_ROLES,
        allowed_environments=_ALL_ENVS,
        required_target_scope=required_scope,
        argument_schema=args_model.model_json_schema(),
        result_schema=_WRITE,
        max_execution_time=60,
        max_result_rows=1,
        audit_required=True,
        enabled=enabled,
        requires_dual_approval=requires_dual_approval,
    )


def _restricted_tool(
    tool_id: str,
    description: str,
    args_model,
    *,
    enabled: bool,
    risk_level: str = "CRITICAL",
    reversible: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        tool_id=tool_id,
        version="1.0.0",
        description=description,
        operation_type=OperationType.PRIVILEGED,
        risk_level=risk_level,
        reversible=reversible,
        availability_impact=True,
        data_modification=True,
        requires_approval=True,
        allowed_roles=_ADVANCED_ROLES,
        allowed_environments=_ALL_ENVS,
        required_target_scope=["environment", "instance", "database"],
        argument_schema=args_model.model_json_schema(),
        result_schema=_WRITE,
        max_execution_time=30,
        max_result_rows=100,
        audit_required=True,
        enabled=enabled,
        requires_dual_approval=True,
    )


def build_tool_catalog(settings: Settings) -> list[ToolDefinition]:
    # These diagnostics reflect the whole instance/cluster on every engine we
    # support (SQL Server DMVs, MySQL information_schema/performance_schema,
    # Postgres pg_stat_activity/pg_locks/pg_stat_database — none of them are
    # scoped to one database), so they never need a specific database named.
    # A DBA can ask "what's running / blocking on X" without knowing which
    # database is affected — that's the whole point of running them: to find
    # out. `execution/service.py` already falls back to the credential's own
    # default database when none is given.
    _INSTANCE_WIDE_SCOPE = ["environment", "instance"]

    tools: list[ToolDefinition] = [
        # --- Read-only diagnostics (spec §8) ---
        _read_tool(
            "database.get_health",
            "Overall instance/database health snapshot.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_version",
            "Engine version and edition/build info.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_sessions", "Active session list.", required_scope=_INSTANCE_WIDE_SCOPE
        ),
        _read_tool(
            "database.get_blocking_sessions",
            "Current blocking chains.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_deadlocks", "Recent deadlock graphs.", required_scope=_INSTANCE_WIDE_SCOPE
        ),
        _read_tool(
            "database.get_running_queries",
            "Currently executing queries.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_wait_statistics",
            "Aggregated wait statistics.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_query_plan",
            "Execution plan for a specific query id.",
            args_model=QueryPlanArgs,
            required_scope=["environment", "instance", "database", "query_id"],
        ),
        _read_tool(
            "database.get_top_queries",
            "Top resource-consuming queries.",
            args_model=TopQueriesArgs,
        ),
        _read_tool(
            "database.get_indexes",
            "Index metadata for a schema/table.",
            required_scope=["environment", "instance", "database", "schema_name", "object_name"],
        ),
        _read_tool(
            "database.get_statistics",
            "Statistics metadata/staleness for a schema/table.",
            required_scope=["environment", "instance", "database", "schema_name", "object_name"],
        ),
        _read_tool("database.get_tables", "Table inventory for a database."),
        _read_tool(
            "database.get_storage",
            "Storage/space utilization.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool("database.get_transaction_log", "Transaction log / WAL usage."),
        _read_tool(
            "database.get_replication_status",
            "Replication / Always On / streaming status.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_backup_status",
            "Recent backup history and status.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_configuration",
            "Server/database configuration parameters.",
            required_scope=_INSTANCE_WIDE_SCOPE,
        ),
        _read_tool(
            "database.get_error_logs",
            "Recent error log entries.",
            args_model=ErrorLogArgs,
            required_scope=_INSTANCE_WIDE_SCOPE,
            max_result_rows=200,
        ),
        # --- Controlled write tools (spec §8) ---
        _write_tool(
            "database.cancel_query",
            "Cancel (soft-abort) a single running query without dropping the session.",
            CancelQueryArgs,
            risk_level="LOW",
            reversible=True,
            availability_impact=False,
            required_scope=["environment", "instance", "database", "session_id"],
        ),
        _write_tool(
            "database.kill_session",
            "Terminate a database session.",
            KillSessionArgs,
            risk_level="MEDIUM",
            reversible=False,
            availability_impact=True,
            required_scope=["environment", "instance", "database", "session_id"],
        ),
        _write_tool(
            "database.update_statistics",
            "Refresh optimizer statistics for a table.",
            UpdateStatisticsArgs,
            risk_level="LOW",
            reversible=True,
            availability_impact=False,
            required_scope=["environment", "instance", "database", "schema_name", "object_name"],
        ),
        _write_tool(
            "database.create_index",
            "Create a new index on a table.",
            CreateIndexArgs,
            risk_level="MEDIUM",
            reversible=True,
            availability_impact=True,
            required_scope=["environment", "instance", "database", "schema_name", "object_name"],
        ),
        _write_tool(
            "database.rebuild_index",
            "Rebuild an existing index.",
            RebuildIndexArgs,
            risk_level="MEDIUM",
            reversible=True,
            availability_impact=True,
            required_scope=["environment", "instance", "database", "schema_name", "object_name"],
        ),
        _write_tool(
            "database.modify_configuration",
            "Change a server/database configuration parameter.",
            ModifyConfigurationArgs,
            risk_level="HIGH",
            reversible=True,
            availability_impact=True,
            required_scope=["environment", "instance", "database"],
            allowed_roles=_ADVANCED_ROLES,
        ),
        _write_tool(
            "database.restart_instance",
            "Restart a database instance.",
            RestartInstanceArgs,
            risk_level="CRITICAL",
            reversible=False,
            availability_impact=True,
            required_scope=["environment", "instance"],
            allowed_roles=_ADVANCED_ROLES + [DBARole.DBA_MANAGER],
            requires_dual_approval=True,
        ),
        _write_tool(
            "database.failover",
            "Force a cluster/Always On failover to another instance.",
            FailoverArgs,
            risk_level="CRITICAL",
            reversible=False,
            availability_impact=True,
            required_scope=["environment", "cluster"],
            allowed_roles=[DBARole.DBA_L3, DBARole.DBA_MANAGER],
            requires_dual_approval=True,
        ),
        # --- Restricted tools: framework present, disabled by default (spec §8/§21) ---
        _restricted_tool(
            "database.execute_readonly_sql",
            "Execute a parsed, validated, single read-only statement. Disabled by default.",
            ReadOnlySqlArgs,
            enabled=settings.enable_readonly_sql_tool,
            risk_level="MEDIUM",
            reversible=True,
        ),
        _restricted_tool(
            "database.execute_sql",
            "Execute arbitrary SQL. Disabled by default and NOT recommended to enable.",
            ExecuteSqlArgs,
            enabled=settings.enable_execute_sql_tool,
        ),
        _restricted_tool(
            "database.restore_database",
            "Restore a database from backup. Disabled by default.",
            RestoreDatabaseArgs,
            enabled=settings.enable_restore_database_tool,
        ),
        _restricted_tool(
            "database.create_database",
            "Create a new database. Disabled by default.",
            CreateDatabaseArgs,
            enabled=settings.enable_create_database_tool,
        ),
        _restricted_tool(
            "database.drop_database",
            "Drop a database. Disabled by default.",
            DropDatabaseArgs,
            enabled=settings.enable_drop_database_tool,
        ),
        _restricted_tool(
            "database.truncate_table",
            "Truncate a table. Disabled by default.",
            TruncateTableArgs,
            enabled=settings.enable_truncate_table_tool,
        ),
        _restricted_tool(
            "database.bulk_delete",
            "Bulk-delete rows matching a predicate. Disabled by default.",
            BulkDeleteArgs,
            enabled=settings.enable_bulk_delete_tool,
        ),
    ]
    return tools
