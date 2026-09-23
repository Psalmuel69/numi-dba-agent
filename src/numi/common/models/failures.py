"""Explicit failure model (spec §28).

Every privileged boundary in the system returns one of these codes instead of
leaking stack traces, SQL errors, or internal exceptions to a chat channel.
`FailureCode` is the *only* vocabulary the Agent and channel adapters are
allowed to render to a human; anything else must be mapped into one of these
first.
"""

from __future__ import annotations

from enum import Enum


class FailureCode(str, Enum):
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_NOT_AVAILABLE = "TOOL_NOT_AVAILABLE"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INVALID_TARGET = "INVALID_TARGET"
    POLICY_DENIED = "POLICY_DENIED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    APPROVAL_INVALID = "APPROVAL_INVALID"
    APPROVAL_MISMATCH = "APPROVAL_MISMATCH"
    APPROVAL_ALREADY_DECIDED = "APPROVAL_ALREADY_DECIDED"
    SEPARATION_OF_DUTIES_VIOLATION = "SEPARATION_OF_DUTIES_VIOLATION"
    DUAL_APPROVAL_REQUIRED = "DUAL_APPROVAL_REQUIRED"
    RATE_LIMITED = "RATE_LIMITED"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    RESULT_TRUNCATED = "RESULT_TRUNCATED"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    STATE_CHANGED_REAPPROVAL_REQUIRED = "STATE_CHANGED_REAPPROVAL_REQUIRED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    CHANGE_TICKET_REQUIRED = "CHANGE_TICKET_REQUIRED"


class NumiError(Exception):
    """Base class for all errors that carry a safe, user-facing FailureCode.

    `detail` is safe to show to the requesting user. `internal_detail` is for
    logs/audit only and must never be forwarded to a channel adapter.
    """

    def __init__(
        self,
        code: FailureCode,
        detail: str,
        *,
        internal_detail: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.internal_detail = internal_detail or detail

    def to_dict(self) -> dict:
        return {"code": self.code.value, "detail": self.detail}
