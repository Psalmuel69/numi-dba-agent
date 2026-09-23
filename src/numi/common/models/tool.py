"""Tool contract model (spec §9).

Every tool the Agent can even *name* is described by a `ToolDefinition`
registered in the Gateway's Tool Registry. The LLM selects a `tool_id` and
fills in an `arguments` dict; it never invents a new tool, a new argument, or
executes anything the registry doesn't already know about (spec §35).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from numi.common.models.identity import DBARole
from numi.common.models.risk import RiskLevel
from numi.common.models.target import Environment


class OperationType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    PRIVILEGED = "PRIVILEGED"


class ToolDefinition(BaseModel):
    """Static, versioned metadata about one tool (spec §9).

    `argument_schema` / `result_schema` are JSON Schema documents (produced
    from a Pydantic model via `.model_json_schema()`) used to validate the
    Agent's tool call and the Execution Service's result before either one
    crosses a trust boundary.
    """

    model_config = ConfigDict(frozen=True)

    tool_id: str
    version: str
    description: str
    operation_type: OperationType
    risk_level: RiskLevel
    reversible: bool
    availability_impact: bool
    data_modification: bool
    requires_approval: bool
    allowed_roles: list[DBARole]
    allowed_environments: list[Environment]
    required_target_scope: list[str]
    argument_schema: dict[str, Any]
    result_schema: dict[str, Any]
    max_execution_time: int = 30
    max_result_rows: int = 100
    audit_required: bool = True
    enabled: bool = True
    requires_change_ticket: bool = False
    requires_dual_approval: bool = False


class ToolCallRequest(BaseModel):
    """What the Agent sends to the Gateway. Untrusted until fully validated.

    Note there is no `identity` or `role` field here: the Agent identifies
    the *channel account* that originated the request, and the Gateway
    independently re-resolves a `VerifiedIdentity` from it via its own
    `IdentityProvider` (spec §3, §5, §37, §62) — it never accepts an
    identity or role object asserted by the Agent at face value.
    """

    tool_id: str
    tool_version: str | None = None
    arguments: dict[str, Any]
    target: dict[str, Any]
    reason: str
    conversation_id: str
    investigation_id: str | None = None
    request_id: str
    channel: str
    channel_account_id: str
    # An approval_id must be supplied for any tool call that a prior Gateway
    # response marked as APPROVAL_REQUIRED; it is independently re-verified,
    # never trusted at face value.
    approval_id: str | None = None
    change_id: str | None = None


class ToolCallStatus(str, Enum):
    EXECUTED = "EXECUTED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    DENIED = "DENIED"
    FAILED = "FAILED"


class ToolCallResponse(BaseModel):
    status: ToolCallStatus
    execution_id: str | None = None
    approval_id: str | None = None
    result: dict[str, Any] | None = None
    failure_code: str | None = None
    message: str
    risk: dict[str, Any] | None = None
    policy_decision: str | None = None
