"""Gateway <-> Execution Service internal contract.

This is a *different, narrower* contract than the Agent-facing `ToolCallRequest`
(spec §18): by the time a request reaches here, the target has been resolved
to one registered server + one discovered database, policy/risk/approval
have already been cleared, and the only thing left to do is run one
specific, typed operation and return raw (not-yet-masked) results for the
Gateway's Data Policy Layer to minimize. The Execution Service never sees a
raw user message, an LLM completion, or anything about approvals/policy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from numi.common.models.target import Platform


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    tool_id: str
    tool_version: str
    platform: Platform
    # Registered server id — the Execution Service looks up credentials by
    # this, never a connection string.
    server_id: str
    # The specific database to connect to / operate on ("" for server-level
    # operations like restart_instance).
    database: str
    schema_name: str | None = None
    object_name: str | None = None
    session_id: str | None = None
    query_id: str | None = None
    arguments: dict[str, Any]
    max_execution_time: int
    max_result_rows: int


class DiscoveryRequest(BaseModel):
    """Gateway -> Execution Service: crawl one registered server's metadata."""

    model_config = ConfigDict(extra="forbid")

    server_id: str
    platform: Platform
    max_objects_per_database: int = 5000


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    success: bool
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    truncated: bool = False
    affected: dict[str, Any] = {}
    error_code: str | None = None
    error_detail: str | None = None
    verification_status: str = "NOT_APPLICABLE"
    verification_detail: str = ""
    duration_ms: int = 0
