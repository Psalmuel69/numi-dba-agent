# Tool Catalog

Source of truth: `src/numi/gateway/domain/tool_catalog.py`. This document
is a human-readable index of it — if the two ever disagree, the code wins.

Every tool is versioned, schema-validated, and independently re-verified by
the Gateway regardless of what the Agent claims about it (spec §9).

## Read-only diagnostics (enabled by default, all roles)

| Tool | Description | Required target scope |
|---|---|---|
| `database.get_health` | Health/CPU/uptime snapshot | environment, instance, database |
| `database.get_version` | Engine version/edition | environment, instance, database |
| `database.get_sessions` | Active session list | environment, instance, database |
| `database.get_blocking_sessions` | Current blocking chains | environment, instance, database |
| `database.get_deadlocks` | Recent deadlock graphs | environment, instance, database |
| `database.get_running_queries` | Currently executing queries | environment, instance, database |
| `database.get_wait_statistics` | Aggregated wait stats | environment, instance, database |
| `database.get_query_plan` | Plan for a specific query id | + query_id |
| `database.get_top_queries` | Top resource-consuming queries (`order_by`, `limit`) | environment, instance, database |
| `database.get_indexes` | Index metadata for a table | + schema, object |
| `database.get_statistics` | Statistics staleness for a table | + schema, object |
| `database.get_tables` | Table inventory | environment, instance, database |
| `database.get_storage` | Storage/space utilization | environment, instance, database |
| `database.get_transaction_log` | Log/WAL usage | environment, instance, database |
| `database.get_replication_status` | Replication/Always On/streaming status | environment, instance, database |
| `database.get_backup_status` | Recent backup history | environment, instance, database |
| `database.get_configuration` | Server/database configuration parameters | environment, instance, database |
| `database.get_error_logs` | Recent error log entries (`since_minutes`, `limit`) | environment, instance, database |

All: `risk_level=LOW`, `reversible=true`, `availability_impact=false`,
`requires_approval=false`, `max_result_rows=100` (200 for error logs).

## Controlled write tools (enabled by default; policy/approval-gated)

| Tool | Risk | Reversible | Availability impact | Dual approval | Min role (tool-level) |
|---|---|---|---|---|---|
| `database.cancel_query` | LOW | yes | no | no | DBA_L2 |
| `database.kill_session` | MEDIUM | no | yes | no | DBA_L2 |
| `database.update_statistics` | LOW | yes | no | no | DBA_L2 |
| `database.create_index` | MEDIUM | yes | yes | no | DBA_L2 |
| `database.rebuild_index` | MEDIUM | yes | yes | no | DBA_L2 |
| `database.modify_configuration` | HIGH | yes | yes | no | DBA_L3 |
| `database.restart_instance` | CRITICAL | no | yes | **yes** | DBA_L3 |
| `database.failover` | CRITICAL | no | yes | **yes** | DBA_L3 |

"Min role" here is the *tool's* `allowed_roles` floor — the Policy Engine
(see [POLICY_MODEL.md](POLICY_MODEL.md)) can still be more restrictive per
environment (e.g. DBA_L2 requires approval in production even though the
tool itself permits DBA_L2).

`restart_instance` and `modify_configuration` additionally require a change
ticket in production (`config/policy.yaml`'s `change_ticket_required`).

## Restricted tools (framework implemented, disabled by default)

| Tool | Enable via | Notes |
|---|---|---|
| `database.execute_readonly_sql` | `ENABLE_READONLY_SQL_TOOL=true` | Real-parser (`sqlglot`) validated: single SELECT only, no dangerous functions, row-capped. See `gateway/domain/sql_validator.py`. |
| `database.execute_sql` | `ENABLE_EXECUTE_SQL_TOOL=true` | Arbitrary SQL. Argument schema exists; **no adapter implements execution** — enabling the flag alone does not make this tool functional, by design. |
| `database.restore_database` | `ENABLE_RESTORE_DATABASE_TOOL=true` | Framework only |
| `database.create_database` | `ENABLE_CREATE_DATABASE_TOOL=true` | Framework only |
| `database.drop_database` | `ENABLE_DROP_DATABASE_TOOL=true` | Framework only |
| `database.truncate_table` | `ENABLE_TRUNCATE_TABLE_TOOL=true` | Framework only |
| `database.bulk_delete` | `ENABLE_BULK_DELETE_TOOL=true` | Framework only |

All restricted tools: `operation_type=PRIVILEGED`, `requires_dual_approval=true`,
`allowed_roles=[DBA_L3, DBA_MANAGER]`. Calling any of these while disabled
returns `TOOL_NOT_AVAILABLE` before target/policy/execution logic runs at
all — see `tests/security/test_security_suite.py::test_disabled_tool_returns_tool_not_available_without_execution`.

## Adding a new tool

1. Add a Pydantic argument model to `common/models/tool_arguments.py`.
2. Add a `ToolDefinition` entry (and its `ARGUMENT_MODELS` mapping) in
   `gateway/domain/tool_catalog.py`.
3. Implement the corresponding method on `DatabaseAdapter` (and each of
   `SQLServerAdapter`/`PostgreSQLAdapter`/`MySQLAdapter`, or raise
   `NotImplementedError` with a clear message if it's engine-specific).
4. Add dispatch wiring in `execution/service.py`'s `_READ_METHODS`/
   `_WRITE_METHODS` (or the restricted-tool dispatch table).
5. Add policy entries for it in `config/policy.yaml` for every environment
   — remember the Policy Engine fails closed for anything unlisted.
