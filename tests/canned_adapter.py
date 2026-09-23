"""Deterministic in-memory `DatabaseAdapter` for the test suite.

This is a TEST DOUBLE — it lives in `tests/`, never in `src/`, so the
shipped Execution Service has no fake-data code path. It is injected into
`ExecutionService` via the `adapter_factory` seam (see `tests/stack.py`) so
the gateway pipeline tests (policy / risk / approval / audit) can assert on
deterministic execution results without needing a live SQL Server or
PostgreSQL.

Real adapter *query logic* (which DMV / pg_stat query maps to which tool) is
covered separately in `tests/unit/test_adapters.py` via `FakeQueryExecutor`,
and true end-to-end behaviour against real engines is covered by the
opt-in `tests/e2e/test_live_databases.py`.

The canned data reproduces the spec's worked scenario (§54/§68): a
43-session blocking chain headed by session ``9182``.
"""

from __future__ import annotations

from typing import Any

from numi.common.models.execution import ExecutionRequest
from numi.execution.adapters.base import DatabaseAdapter


class CannedDatabaseAdapter(DatabaseAdapter):
    def __init__(self, database: str):
        super().__init__(executor=None, database=database)  # type: ignore[arg-type]

    async def health(self) -> list[dict[str, Any]]:
        return [
            {
                "active_sessions": 42,
                "cpu_percent": 71,
                "database_size_mb": 48213,
                "uptime_seconds": 864_211,
                "status": "ONLINE",
            }
        ]

    async def version(self) -> list[dict[str, Any]]:
        return [{"version": "Canned Engine 1.0 (test double)"}]

    async def sessions(self) -> list[dict[str, Any]]:
        return [
            {"session_id": "9182", "login": "svc_app", "status": "running", "cpu_time": 812000},
            {"session_id": "9183", "login": "svc_app", "status": "sleeping", "cpu_time": 210},
        ]

    async def blocking(self) -> list[dict[str, Any]]:
        return [
            {
                "blocked_session_id": str(9183 + i),
                "blocking_session_id": "9182",
                "wait_type": "LCK_M_X",
                "wait_time_ms": 14000 + i * 500,
            }
            for i in range(43)
        ]

    async def deadlocks(self) -> list[dict[str, Any]]:
        return []

    async def running_queries(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": "9182",
                "duration_ms": 840_000,
                "query_text": "UPDATE TransactionPostingHistory SET ...",
            }
        ]

    async def waits(self) -> list[dict[str, Any]]:
        return [
            {"wait_type": "LCK_M_X", "waiting_tasks_count": 43, "wait_time_ms": 602_000},
            {"wait_type": "PAGEIOLATCH_SH", "waiting_tasks_count": 3, "wait_time_ms": 4_200},
        ]

    async def query_plan(self, query_id: str) -> list[dict[str, Any]]:
        return [{"query_id": query_id, "plan_summary": "Clustered Index Scan -> Sort -> Hash Match"}]

    async def top_queries(self, order_by: str, limit: int) -> list[dict[str, Any]]:
        return [
            {"query_id": f"q{i}", "metric": order_by, "value": 1000 - i * 10}
            for i in range(min(limit, 5))
        ]

    async def indexes(self, schema: str, table: str) -> list[dict[str, Any]]:
        return [{"schema": schema, "table": table, "index_name": f"IX_{table}_1", "user_seeks": 1200}]

    async def statistics(self, schema: str, table: str) -> list[dict[str, Any]]:
        return [{"schema": schema, "table": table, "last_updated": "2026-08-20T02:00:00Z"}]

    async def tables(self) -> list[dict[str, Any]]:
        return [{"schema": "dbo", "table": "TransactionPostingHistory", "row_estimate": 48_213_000}]

    async def storage(self) -> list[dict[str, Any]]:
        return [{"file_name": "primary_data", "size_mb": 48213, "growth": "10%"}]

    async def transaction_log(self) -> list[dict[str, Any]]:
        return [{"log_size_mb": 8192, "log_used_percent": 34}]

    async def replication(self) -> list[dict[str, Any]]:
        return [{"replica": "secondary-01", "synchronization_state": "SYNCHRONIZED"}]

    async def backups(self) -> list[dict[str, Any]]:
        return [{"backup_type": "FULL", "finished_at": "2026-08-21T02:00:00Z", "status": "SUCCESS"}]

    async def configuration(self) -> list[dict[str, Any]]:
        return [{"name": "max_connections", "value": "500"}]

    async def error_logs(self, since_minutes: int, limit: int) -> list[dict[str, Any]]:
        return []

    async def cancel_query(self, session_id: str, reason: str) -> dict[str, Any]:
        return {"cancelled": True, "session_id": session_id}

    async def kill_session(self, session_id: str, reason: str) -> dict[str, Any]:
        return {"terminated": True, "session_id": session_id}

    async def update_statistics(self, schema: str, table: str) -> dict[str, Any]:
        return {"updated": True, "schema": schema, "table": table}

    async def create_index(
        self, schema: str, table: str, columns: list[str], name: str, unique: bool
    ) -> dict[str, Any]:
        return {"created": True, "index_name": name}

    async def rebuild_index(self, schema: str, table: str, index_name: str) -> dict[str, Any]:
        return {"rebuilt": True, "index_name": index_name}

    async def modify_configuration(self, parameter: str, value: str) -> dict[str, Any]:
        return {"parameter": parameter, "value": value}

    async def restart_instance(self) -> dict[str, Any]:
        return {"restarted": True}

    async def failover(self, target_instance: str) -> dict[str, Any]:
        return {"failed_over_to": target_instance}


async def canned_adapter_factory(request: ExecutionRequest) -> tuple[DatabaseAdapter, None]:
    """Drop-in for `ExecutionService(adapter_factory=...)` in tests."""
    return CannedDatabaseAdapter(request.database), None
