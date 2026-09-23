"""Authorization gate (spec §3, §5, §37).

This is independent of, and evaluated before, the Policy Engine's
fine-grained ALLOW/DENY/REQUIRES_APPROVAL decision. It answers three coarse
questions using ONLY the independently-verified identity — never anything
claimed in chat text or by the LLM:

  1. Is this person a member of the DBA team at all?
  2. Does their verified role even appear in this tool's allowed_roles?
  3. Does their verified role appear in the target server's allowed_roles
     (or a per-database override) — spec §11?
  4. Is this tool permitted in this target's environment at all?

A conversation switching from one server/database to another, or the
passage of time invalidating a cached role, must re-run this check —
nothing here is cached across a conversation turn (spec §37).
"""

from __future__ import annotations

from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.identity import VerifiedIdentity
from numi.common.models.tool import ToolDefinition
from numi.gateway.domain.target_validation import TargetContext


def authorize(
    identity: VerifiedIdentity,
    tool: ToolDefinition,
    ctx: TargetContext,
) -> None:
    """Raises NumiError(UNAUTHORIZED) on any failure; returns None on success."""
    if not identity.is_dba():
        raise NumiError(
            FailureCode.UNAUTHORIZED,
            "This account is not a member of the DBA team.",
        )

    if not identity.mfa_satisfied:
        raise NumiError(
            FailureCode.AUTHENTICATION_FAILED,
            "Multi-factor authentication has not been satisfied for this session.",
        )

    user_roles = set(identity.dba_roles)

    if not (user_roles & set(tool.allowed_roles)):
        raise NumiError(
            FailureCode.UNAUTHORIZED,
            f"Your role does not permit use of '{tool.tool_id}'.",
        )

    if ctx.allowed_roles and not (user_roles & set(ctx.allowed_roles)):
        where = f"{ctx.server.id}/{ctx.database}" if ctx.database else ctx.server.id
        raise NumiError(
            FailureCode.UNAUTHORIZED,
            f"Your role does not permit access to '{where}'.",
        )

    if ctx.environment not in tool.allowed_environments:
        raise NumiError(
            FailureCode.UNAUTHORIZED,
            f"'{tool.tool_id}' is not permitted in environment '{ctx.environment.value}'.",
        )
