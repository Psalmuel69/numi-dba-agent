"""Execution Service core dispatcher (spec §18, §50).

This is the ONLY code in the entire platform that is permitted to obtain a
database credential and open a database connection. It receives an already
fully-authorized, already-approved `ExecutionRequest` from the Gateway (never
directly from the Agent — see `execution.api.app` for the service-auth
enforcement that guarantees this) and:

  1. looks up credentials for the target `server_id` via `CredentialProvider`
  2. opens a scoped connection to the real database
  3. dispatches to the one typed adapter method matching `tool_id`
  4. enforces `max_execution_time` via a hard timeout
  5. enforces `max_result_rows` by truncating (never silently dropping the
     fact that it happened — `truncated=True` is always reported)
  6. returns raw (not yet masked) results for the Gateway's Data Policy
     Layer to minimize — this service never applies masking itself, so
     there is exactly one place that decision is made
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from numi.common.config import Settings
from numi.common.models.execution import ExecutionRequest, ExecutionResult
from numi.common.models.failures import FailureCode
from numi.common.models.target import Platform
from numi.common.observability import get_logger
from numi.execution.adapters.base import DatabaseAdapter, QueryExecutor
from numi.execution.adapters.mysql import MySQLAdapter
from numi.execution.adapters.postgresql import PostgreSQLAdapter
from numi.execution.adapters.sqlserver import SQLServerAdapter
from numi.execution.credentials.provider import CredentialProvider

logger = get_logger(__name__)

# A test/integration seam: given an ExecutionRequest, return a ready
# DatabaseAdapter (and an optional handle to close). Production never sets
# this — it always resolves a real credential and opens a real connection.
AdapterFactory = Callable[[ExecutionRequest], Awaitable[tuple[DatabaseAdapter, Any]]]

_READ_METHODS = {
    "database.get_health": ("health", []),
    "database.get_version": ("version", []),
    "database.get_sessions": ("sessions", []),
    "database.get_blocking_sessions": ("blocking", []),
    "database.get_deadlocks": ("deadlocks", []),
    "database.get_running_queries": ("running_queries", []),
    "database.get_wait_statistics": ("waits", []),
    "database.get_query_plan": ("query_plan", ["query_id"]),
    "database.get_top_queries": ("top_queries", ["order_by", "limit"]),
    "database.get_indexes": ("indexes", ["schema_name", "table"]),
    "database.get_statistics": ("statistics", ["schema_name", "table"]),
    "database.get_tables": ("tables", []),
    "database.get_storage": ("storage", []),
    "database.get_transaction_log": ("transaction_log", []),
    "database.get_replication_status": ("replication", []),
    "database.get_backup_status": ("backups", []),
    "database.get_configuration": ("configuration", []),
    "database.get_error_logs": ("error_logs", ["since_minutes", "limit"]),
}

_WRITE_METHODS = {
    "database.cancel_query": ("cancel_query", ["session_id", "reason"]),
    "database.kill_session": ("kill_session", ["session_id", "reason"]),
    "database.update_statistics": ("update_statistics", ["schema_name", "table"]),
    "database.create_index": ("create_index", ["schema_name", "table", "columns", "name", "unique"]),
    "database.rebuild_index": ("rebuild_index", ["schema_name", "table", "index_name"]),
    "database.modify_configuration": ("modify_configuration", ["parameter", "value"]),
    "database.restart_instance": ("restart_instance", []),
    "database.failover": ("failover", ["target_instance"]),
}


def _adapter_class_for(platform: Platform) -> type[DatabaseAdapter]:
    if platform == Platform.SQLSERVER:
        return SQLServerAdapter
    if platform == Platform.POSTGRESQL:
        return PostgreSQLAdapter
    if platform in (Platform.MYSQL, Platform.MARIADB):
        return MySQLAdapter
    raise NotImplementedError(
        f"No adapter registered for platform '{platform.value}'. Adding a new engine "
        "(e.g. Oracle) means implementing DatabaseAdapter and registering it "
        "here — the Agent/Gateway contract does not change."
    )


class ExecutionService:
    def __init__(
        self,
        settings: Settings,
        credential_provider: CredentialProvider,
        *,
        adapter_factory: AdapterFactory | None = None,
    ):
        self._settings = settings
        self._credentials = credential_provider
        # Only ever set by tests (see tests/canned_adapter.py). When None,
        # every execution opens a real database connection.
        self._adapter_factory = adapter_factory

    async def _build_adapter(self, request: ExecutionRequest) -> tuple[DatabaseAdapter, Any]:
        """Returns (adapter, connection_handle_or_None)."""
        if self._adapter_factory is not None:
            return await self._adapter_factory(request)

        creds = await self._credentials.get_credentials(request.server_id)
        # Connect to the specific database this request targets (Postgres
        # binds a connection to one db; SQL Server / MySQL `USE` it). Falls
        # back to the credential's own default db for server-level ops.
        if request.database:
            creds = creds.model_copy(update={"database": request.database})
        adapter_cls = _adapter_class_for(request.platform)

        if request.platform == Platform.SQLSERVER:
            from numi.execution.adapters.connections import SQLServerQueryExecutor

            executor: QueryExecutor = SQLServerQueryExecutor(creds)  # type: ignore[assignment]
        elif request.platform in (Platform.MYSQL, Platform.MARIADB):
            from numi.execution.adapters.connections import MySQLQueryExecutor

            executor = MySQLQueryExecutor(creds)  # type: ignore[assignment]
        else:
            from numi.execution.adapters.connections import PostgreSQLQueryExecutor

            executor = PostgreSQLQueryExecutor(creds)  # type: ignore[assignment]

        await executor.connect()  # type: ignore[attr-defined]
        return adapter_cls(executor, request.database), executor

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        start = time.monotonic()
        handle = None
        try:
            adapter, handle = await self._build_adapter(request)
            result = await asyncio.wait_for(
                self._dispatch(adapter, request), timeout=request.max_execution_time
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            result.duration_ms = duration_ms
            return result
        except TimeoutError:
            return ExecutionResult(
                execution_id=request.execution_id,
                success=False,
                error_code=FailureCode.EXECUTION_TIMEOUT.value,
                error_detail=f"Execution exceeded {request.max_execution_time}s timeout.",
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except NotImplementedError as exc:
            return ExecutionResult(
                execution_id=request.execution_id,
                success=False,
                error_code=FailureCode.TOOL_NOT_AVAILABLE.value,
                error_detail=str(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 — deliberately broad: never leak internals
            # error_detail below promises "server-side logs" — this is that
            # log. Without it, a real driver-level failure (a missing
            # extension, a permissions gap, a syntax error from a platform
            # quirk) is invisible from the client's genuinely-necessary
            # generic message, and undiagnosable without reproducing by hand.
            logger.error(
                "execution_failed",
                tool_id=request.tool_id,
                server_id=request.server_id,
                execution_id=request.execution_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return ExecutionResult(
                execution_id=request.execution_id,
                success=False,
                error_code=FailureCode.EXECUTION_FAILED.value,
                error_detail="The database operation failed. See server-side logs for detail.",
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        finally:
            if handle is not None:
                await handle.close()

    async def _dispatch(self, adapter: DatabaseAdapter, request: ExecutionRequest) -> ExecutionResult:
        tool_id = request.tool_id
        args = request.arguments

        if tool_id in _READ_METHODS:
            method_name, arg_names = _READ_METHODS[tool_id]
            method = getattr(adapter, method_name)
            call_args = [args[name] for name in arg_names]
            rows = await method(*call_args)
            truncated = len(rows) > request.max_result_rows
            rows = rows[: request.max_result_rows]
            columns = sorted({k for row in rows for k in row.keys()})
            return ExecutionResult(
                execution_id=request.execution_id,
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
            )

        if tool_id in _WRITE_METHODS:
            method_name, arg_names = _WRITE_METHODS[tool_id]
            method = getattr(adapter, method_name)
            call_args = [args[name] for name in arg_names]
            affected = await method(*call_args)
            return ExecutionResult(
                execution_id=request.execution_id,
                success=True,
                affected=affected,
            )

        # Restricted tools — only reachable at all if the Gateway's tool
        # registry marked them enabled; the adapter method itself still
        # raises NotImplementedError unless a real engine-specific
        # implementation has been supplied.
        restricted_dispatch = {
            "database.execute_readonly_sql": lambda: adapter.execute_readonly_sql(args["sql"]),
            "database.execute_sql": lambda: adapter.execute_sql(args["sql"]),
            "database.restore_database": lambda: adapter.restore_database(args["backup_id"]),
            "database.create_database": lambda: adapter.create_database(args["database_name"]),
            "database.drop_database": lambda: adapter.drop_database(args["database_name"]),
            "database.truncate_table": lambda: adapter.truncate_table(
                args["schema_name"], args["table"]
            ),
            "database.bulk_delete": lambda: adapter.bulk_delete(
                args["schema_name"], args["table"], args["predicate_description"]
            ),
        }
        if tool_id in restricted_dispatch:
            outcome = await restricted_dispatch[tool_id]()
            if isinstance(outcome, list):
                return ExecutionResult(
                    execution_id=request.execution_id,
                    success=True,
                    rows=outcome,
                    row_count=len(outcome),
                )
            return ExecutionResult(execution_id=request.execution_id, success=True, affected=outcome)

        return ExecutionResult(
            execution_id=request.execution_id,
            success=False,
            error_code=FailureCode.TOOL_NOT_FOUND.value,
            error_detail=f"Execution Service has no dispatch entry for '{tool_id}'.",
        )
