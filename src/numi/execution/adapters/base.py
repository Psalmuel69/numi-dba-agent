"""DatabaseAdapter interface (spec §20).

One adapter instance is bound to one already-open, already-scoped connection
(via a `QueryExecutor`) for exactly one execution — it never receives a
credential itself, never brokers its own connection pooling policy (that
lives in `execution.adapters.connections`), and never accepts a raw SQL
string from a caller for any of the typed methods below. Only the two
explicitly-restricted `execute_sql` / `execute_readonly_sql` methods take
SQL text at all, and those are unreachable unless their tool is enabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class QueryExecutor(Protocol):
    """Thin boundary between an adapter's SQL text and an actual database
    connection. Production implementations wrap a real pyodbc/asyncpg
    connection with timeouts, cancellation, and pooling (see
    `execution.adapters.connections`); tests inject a fake executor with
    canned rows so adapter *logic* (which DMV/pg_stat query maps to which
    tool, how results are shaped) is verified without a live database
    (spec §32/§67 — unit tests must not require real databases)."""

    async def fetch_all(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> list[dict[str, Any]]: ...

    async def execute(
        self, sql: str, params: dict[str, Any] | None = None, *, timeout: int = 30
    ) -> dict[str, Any]: ...


class DatabaseAdapter(ABC):
    def __init__(self, executor: QueryExecutor, database: str):
        self._executor = executor
        self._database = database

    # --- read-only diagnostics ------------------------------------------------

    @abstractmethod
    async def health(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def version(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def sessions(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def blocking(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def deadlocks(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def running_queries(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def waits(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def query_plan(self, query_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def top_queries(self, order_by: str, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def indexes(self, schema: str, table: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def statistics(self, schema: str, table: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def tables(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def storage(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def transaction_log(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def replication(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def backups(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def configuration(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def error_logs(self, since_minutes: int, limit: int) -> list[dict[str, Any]]: ...

    # --- controlled write operations ------------------------------------------

    @abstractmethod
    async def cancel_query(self, session_id: str, reason: str) -> dict[str, Any]: ...

    @abstractmethod
    async def kill_session(self, session_id: str, reason: str) -> dict[str, Any]: ...

    @abstractmethod
    async def update_statistics(self, schema: str, table: str) -> dict[str, Any]: ...

    @abstractmethod
    async def create_index(
        self, schema: str, table: str, columns: list[str], name: str, unique: bool
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def rebuild_index(self, schema: str, table: str, index_name: str) -> dict[str, Any]: ...

    @abstractmethod
    async def modify_configuration(self, parameter: str, value: str) -> dict[str, Any]: ...

    @abstractmethod
    async def restart_instance(self) -> dict[str, Any]: ...

    @abstractmethod
    async def failover(self, target_instance: str) -> dict[str, Any]: ...

    # --- restricted operations (framework only; gated by tool.enabled) --------

    async def execute_readonly_sql(self, validated_sql: str) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "execute_readonly_sql is not implemented for this adapter/is disabled."
        )

    async def execute_sql(self, sql: str) -> dict[str, Any]:
        raise NotImplementedError("execute_sql is disabled by default and not implemented.")

    async def restore_database(self, backup_id: str) -> dict[str, Any]:
        raise NotImplementedError("restore_database is disabled by default and not implemented.")

    async def create_database(self, database_name: str) -> dict[str, Any]:
        raise NotImplementedError("create_database is disabled by default and not implemented.")

    async def drop_database(self, database_name: str) -> dict[str, Any]:
        raise NotImplementedError("drop_database is disabled by default and not implemented.")

    async def truncate_table(self, schema: str, table: str) -> dict[str, Any]:
        raise NotImplementedError("truncate_table is disabled by default and not implemented.")

    async def bulk_delete(
        self, schema: str, table: str, predicate_description: str
    ) -> dict[str, Any]:
        raise NotImplementedError("bulk_delete is disabled by default and not implemented.")
