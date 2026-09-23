"""The Gateway pipeline (spec's central diagram, §1/§74).

This is the ONLY path from an Agent tool call to an actual database
operation. Every stage below is mandatory and none can be skipped by
anything the Agent, the LLM, or a chat message claims:

  Tool Registry -> Argument Validation -> Target Validation -> Authorization
  -> Rate Limiting -> Policy Engine -> Risk Engine -> Approval Engine
  -> Execution -> Data Minimization -> Audit

If a required approval doesn't exist yet, execution stops and an
`APPROVAL_REQUIRED` response is returned instead of ever reaching the
Execution Service.
"""

from __future__ import annotations

import time

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from numi.common.ids import new_id
from numi.common.models.execution import ExecutionRequest
from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.identity import VerifiedIdentity
from numi.common.models.target import DatabaseTarget
from numi.common.models.tool import ToolCallRequest, ToolCallResponse, ToolCallStatus
from numi.gateway.domain.approval import ApprovalContext, ApprovalEngine
from numi.gateway.domain.audit import AuditLog
from numi.gateway.domain.authorization import authorize
from numi.gateway.domain.data_policy import DataMinimizer, sqlglot_dialect_for_platform
from numi.gateway.domain.policy_engine import PolicyDecision, PolicyEngine
from numi.gateway.domain.rate_limiter import RateLimiter
from numi.gateway.domain.risk_engine import RiskEngine
from numi.gateway.domain.servers import ServerRegistry
from numi.gateway.domain.sql_validator import validate_readonly_sql
from numi.gateway.domain.target_validation import TargetValidator
from numi.gateway.domain.tool_catalog import ARGUMENT_MODELS
from numi.gateway.domain.tool_registry import ToolRegistry
from numi.gateway.infrastructure.execution_client import ExecutionClient


def _enrich_target(target: DatabaseTarget, args_dict: dict) -> DatabaseTarget:
    """Fills object/session/query-level target fields from validated
    arguments when the caller only supplied them there — a single source of
    truth (the validated arguments model) rather than requiring the Agent to
    duplicate values across `target` and `arguments`."""
    updates = {}
    if args_dict.get("schema_name") and not target.schema_name:
        updates["schema"] = args_dict["schema_name"]
    if args_dict.get("table") and not target.object_name:
        updates["object"] = args_dict["table"]
    if args_dict.get("session_id") and not target.session_id:
        updates["session_id"] = args_dict["session_id"]
    if args_dict.get("query_id") and not target.query_id:
        updates["query_id"] = args_dict["query_id"]
    if not updates:
        return target
    return target.model_copy(update=updates)


class ToolCallHandler:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        registry: ServerRegistry,
        target_validator: TargetValidator,
        policy_engine: PolicyEngine,
        risk_engine: RiskEngine,
        rate_limiter: RateLimiter,
        data_minimizer: DataMinimizer,
        execution_client: ExecutionClient,
        session: AsyncSession,
        discovery=None,
        agent_version: str = "unknown",
        channel: str = "unknown",
        identity_provider_name: str = "unknown",
    ):
        self._tools = tool_registry
        self._registry = registry
        self._discovery = discovery
        self._targets = target_validator
        self._policy = policy_engine
        self._risk = risk_engine
        self._rate_limiter = rate_limiter
        self._minimizer = data_minimizer
        self._execution = execution_client
        self._session = session
        self._approvals = ApprovalEngine(session)
        self._audit = AuditLog(session)
        self._agent_version = agent_version
        self._channel = channel
        self._identity_provider_name = identity_provider_name

    async def handle(
        self, identity: VerifiedIdentity, request: ToolCallRequest
    ) -> ToolCallResponse:
        start = time.monotonic()
        correlation = {
            "conversation_id": request.conversation_id,
            "request_id": request.request_id,
            "investigation_id": request.investigation_id or "",
        }
        try:
            response = await self._handle_inner(identity, request, correlation)
            return response
        except NumiError as exc:
            # `_handle_inner` raises `NumiError` from two structurally different
            # places, and this is the one place that tells them apart: a
            # pre-execution refusal (target/auth/policy/risk/approval — the
            # Gateway never even attempted the operation) vs. the single
            # raise site after `self._execution.execute(...)` returns
            # `success=False` (step 9) — a real execution attempt that was
            # made and didn't succeed. Only `EXECUTION_FAILED`,
            # `EXECUTION_TIMEOUT`, and `DATABASE_UNAVAILABLE` can be raised
            # from that post-execution site (see the `FailureCode(result.error_code)`
            # construction there); every other code is necessarily a
            # pre-execution refusal. Centralizing the split here (rather than
            # threading a status distinction through every raise site in
            # `_handle_inner`) keeps `_handle_inner` reading as a linear
            # pipeline while still letting the Agent's orchestrator apply the
            # already-correct, already-tested distinction downstream: DENIED
            # is a policy fact that ends the turn, FAILED is evidence the
            # investigation records and continues past (see
            # `_submit_and_relay` in agent/orchestrator.py). Live-verified:
            # a real `database.get_error_logs` adapter failure against
            # sqlserver-dev-01 was previously misreported as DENIED and
            # aborted the whole investigation.
            is_execution_outcome = exc.code in (
                FailureCode.EXECUTION_FAILED,
                FailureCode.EXECUTION_TIMEOUT,
                FailureCode.DATABASE_UNAVAILABLE,
            )
            status = ToolCallStatus.FAILED if is_execution_outcome else ToolCallStatus.DENIED
            await self._audit.record(
                event_type="TOOL_CALL_FAILED" if is_execution_outcome else "TOOL_CALL_DENIED",
                correlation_ids=correlation,
                actor_subject_id=identity.subject_id,
                identity_provider=self._identity_provider_name,
                channel=self._channel,
                agent_version=self._agent_version,
                tool_id=request.tool_id,
                target=request.target,
                arguments=request.arguments,
                error_code=exc.code.value,
                duration_ms=int((time.monotonic() - start) * 1000),
            )
            if exc.code in (
                FailureCode.UNAUTHORIZED,
                FailureCode.AUTHENTICATION_FAILED,
                FailureCode.APPROVAL_MISMATCH,
                FailureCode.APPROVAL_INVALID,
                FailureCode.SEPARATION_OF_DUTIES_VIOLATION,
            ):
                await self._audit.record_security_event(
                    event_type=exc.code.value,
                    actor_subject_id=identity.subject_id,
                    detail={"tool_id": request.tool_id, "detail": exc.detail},
                )
            return ToolCallResponse(
                status=status,
                failure_code=exc.code.value,
                message=exc.detail,
            )

    async def _handle_inner(
        self, identity: VerifiedIdentity, request: ToolCallRequest, correlation: dict
    ) -> ToolCallResponse:
        # 1. Tool Registry
        tool = self._tools.get(request.tool_id, request.tool_version)

        # 2. Argument validation — strict, schema-bound, no arbitrary fields
        args_model = ARGUMENT_MODELS.get(tool.tool_id)
        if args_model is None:
            raise NumiError(FailureCode.TOOL_NOT_FOUND, f"No argument schema for '{tool.tool_id}'.")
        try:
            validated_args = args_model.model_validate(request.arguments)
        except ValidationError as exc:
            raise NumiError(
                FailureCode.INVALID_ARGUMENTS, f"Invalid arguments for '{tool.tool_id}': {exc.errors()[:3]}"
            ) from exc
        args_dict = validated_args.model_dump(mode="json")

        # 3. Target parsing + enrichment + resolution against inventory
        try:
            raw_target = DatabaseTarget.model_validate(request.target)
        except ValidationError as exc:
            raise NumiError(FailureCode.INVALID_TARGET, f"Invalid target: {exc.errors()[:3]}") from exc
        target = _enrich_target(raw_target, args_dict)

        # Lazily (re)discover the target server's catalog so validation can
        # check the database/object against real, current metadata. Only for
        # read-only operations, and never fatal.
        if self._discovery is not None and tool.operation_type.value == "READ":
            try:
                server = self._registry.resolve(target)
                await self._discovery.ensure_fresh(server.id)
            except Exception:  # noqa: BLE001 — validate() will surface the real error
                pass

        ctx = await self._targets.validate(target, tool.required_target_scope)

        # 3.5. Real-parser SQL validation for the (disabled-by-default) read-only
        # SQL tool — never reached for typed tools, and args_dict["sql"] is
        # replaced with the normalized, row-capped statement before it goes
        # anywhere near the Execution Service.
        if tool.tool_id == "database.execute_readonly_sql":
            dialect = "tsql" if ctx.platform.value == "sqlserver" else "postgres"
            validated_sql = validate_readonly_sql(
                args_dict["sql"], dialect=dialect, max_result_rows=tool.max_result_rows
            )
            args_dict = {**args_dict, "sql": validated_sql.normalized_sql}

        # 4. Authorization (independent of policy; hard identity/role gate)
        authorize(identity, tool, ctx)

        # 5. Rate limiting
        await self._rate_limiter.check(
            operation_type=tool.operation_type.value,
            requires_approval=tool.requires_approval,
            user_subject_id=identity.subject_id,
            conversation_id=request.conversation_id,
            tool_id=tool.tool_id,
            database_id=ctx.server.id,
            environment=ctx.environment.value,
        )

        # 6. Policy Engine
        role = identity.highest_role()
        evaluation = self._policy.evaluate(
            environment=ctx.environment,
            tool=tool,
            role=role,
            ctx=ctx,
            change_id=request.change_id,
        )
        if evaluation.decision == PolicyDecision.DENY:
            raise NumiError(
                FailureCode.POLICY_DENIED, f"Policy denies '{tool.tool_id}' for role {role.value}."
            )
        if evaluation.requires_change_ticket and not request.change_id:
            raise NumiError(
                FailureCode.CHANGE_TICKET_REQUIRED,
                f"'{tool.tool_id}' in {ctx.environment.value} requires an approved change ticket.",
            )

        # 7. Risk Engine
        risk = self._risk.assess(tool=tool, environment=ctx.environment, ctx=ctx)

        # 8. Approval Engine
        needs_approval = evaluation.decision == PolicyDecision.REQUIRES_APPROVAL
        approval_ctx = ApprovalContext(
            request_id=request.request_id,
            actor=identity,
            tool=tool,
            target=target,
            normalized_arguments=args_dict,
            environment=ctx.environment.value,
            database_id=f"{ctx.server.id}/{ctx.database}" if ctx.database else ctx.server.id,
            risk=risk,
        )
        if needs_approval:
            if not request.approval_id:
                record = await self._approvals.create(
                    approval_ctx, requires_dual_approval=evaluation.requires_dual_approval
                )
                await self._audit.record(
                    event_type="APPROVAL_REQUESTED",
                    correlation_ids=correlation,
                    actor_subject_id=identity.subject_id,
                    identity_provider=self._identity_provider_name,
                    channel=self._channel,
                    agent_version=self._agent_version,
                    tool_id=tool.tool_id,
                    tool_version=tool.version,
                    target=target.model_dump(mode="json"),
                    arguments=args_dict,
                    policy_decision=evaluation.decision.value,
                    risk=risk.model_dump(mode="json"),
                    approval_id=record.approval_id,
                )
                return ToolCallResponse(
                    status=ToolCallStatus.APPROVAL_REQUIRED,
                    approval_id=record.approval_id,
                    message="This action requires DBA approval before it will run.",
                    risk=risk.model_dump(mode="json"),
                    policy_decision=evaluation.decision.value,
                )
            # Approval must have been granted for THIS EXACT action already.
            await self._approvals.verify_for_execution(
                approval_id=request.approval_id,
                expected_action_hash=approval_ctx.action_hash(),
            )

        # 9. Execution (only reachable once ALLOW, or REQUIRES_APPROVAL + verified)
        execution_id = new_id("exec")
        exec_request = ExecutionRequest(
            execution_id=execution_id,
            tool_id=tool.tool_id,
            tool_version=tool.version,
            platform=ctx.platform,
            server_id=ctx.server.id,
            database=ctx.database,
            schema_name=target.schema_name,
            object_name=target.object_name,
            session_id=target.session_id,
            query_id=target.query_id,
            arguments={
                **args_dict,
                **{
                    k: v
                    for k, v in {
                        "schema_name": target.schema_name,
                        "table": target.object_name,
                        "session_id": target.session_id,
                        "query_id": target.query_id,
                    }.items()
                    if v is not None
                },
            },
            max_execution_time=tool.max_execution_time,
            max_result_rows=tool.max_result_rows,
        )
        result = await self._execution.execute(exec_request)

        if not result.success:
            raise NumiError(
                FailureCode(result.error_code) if result.error_code else FailureCode.EXECUTION_FAILED,
                result.error_detail or "Execution failed.",
            )

        if needs_approval:
            # Consume the approval so it cannot authorize a second execution
            # (spec §43 — duplicate execution / replay of an approved action).
            await self._approvals.mark_executed(request.approval_id)

        # 10. Data Minimization (applied here, once, before anything reaches the Agent)
        # `minimize` rather than `apply` so the literal-scrubbing outcome is
        # reported too (see data_policy.py's MinimizedResult). The platform's
        # sqlglot dialect is passed through so `query_text`-style fields are
        # parsed with the engine's own syntax rather than the generic
        # fallback — same mapping idea as the `validate_readonly_sql` call in
        # step 3.5 above, now shared via `sqlglot_dialect_for_platform`.
        minimized = self._minimizer.minimize(
            result.rows,
            max_rows=tool.max_result_rows,
            dialect=sqlglot_dialect_for_platform(ctx.platform.value),
        )
        masked_rows = minimized.rows
        masked_fields = minimized.masked_fields
        truncated = minimized.truncated

        # 11. Audit (success)
        await self._audit.record(
            event_type="TOOL_CALL_EXECUTED",
            correlation_ids=correlation,
            actor_subject_id=identity.subject_id,
            identity_provider=self._identity_provider_name,
            channel=self._channel,
            agent_version=self._agent_version,
            tool_id=tool.tool_id,
            tool_version=tool.version,
            target=target.model_dump(mode="json"),
            arguments=args_dict,
            policy_decision=evaluation.decision.value,
            risk=risk.model_dump(mode="json"),
            approval_id=request.approval_id,
            execution_result="SUCCESS",
        )

        return ToolCallResponse(
            status=ToolCallStatus.EXECUTED,
            execution_id=execution_id,
            result={
                "columns": result.columns,
                "rows": masked_rows,
                "row_count": len(masked_rows),
                "truncated": truncated or result.truncated,
                "masked_fields": masked_fields,
                # Distinct from `masked_fields`: these fields are still
                # present and still readable — only their literal VALUES
                # were replaced. Reported separately so a DBA can tell a
                # scrubbed statement from one that genuinely had no
                # literals (see data_policy.py's MinimizedResult).
                "literal_scrubbed_fields": minimized.literal_scrubbed_fields,
                "affected": result.affected,
            },
            risk=risk.model_dump(mode="json"),
            policy_decision=evaluation.decision.value,
            message="Completed.",
        )
