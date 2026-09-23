from __future__ import annotations

import pytest

from numi.common.config import Settings
from numi.common.ids import new_id
from numi.common.models.catalog import ServerCatalog
from numi.common.models.execution import DiscoveryRequest, ExecutionRequest, ExecutionResult
from numi.common.models.tool import ToolCallRequest, ToolCallStatus
from numi.execution.service import ExecutionService
from numi.gateway.domain.data_policy import DataMinimizer
from numi.gateway.domain.risk_engine import RiskEngine
from numi.gateway.domain.tool_call_handler import ToolCallHandler
from numi.gateway.infrastructure.execution_client import ExecutionClient, InProcessExecutionClient
from tests.canned_adapter import canned_adapter_factory


def _settings() -> Settings:
    return Settings(_env_file=None)


class _FailingExecutionClient(ExecutionClient):
    """A stand-in for the Execution Service reporting a genuine execution
    attempt that failed — e.g. the real, live `database.get_error_logs`
    adapter bug against sqlserver-dev-01 this reproduces: the Execution
    Service itself returns `success=False, error_code="EXECUTION_FAILED"`,
    never raising. Used to drive that exact shape through the real
    `ToolCallHandler.handle` without needing a broken adapter query."""

    def __init__(self, error_code: str = "EXECUTION_FAILED", error_detail: str = "adapter execution failed"):
        self._error_code = error_code
        self._error_detail = error_detail

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            execution_id=request.execution_id,
            success=False,
            error_code=self._error_code,
            error_detail=self._error_detail,
        )

    async def discover(self, request: DiscoveryRequest) -> ServerCatalog:  # pragma: no cover - unused
        raise NotImplementedError


async def _make_handler(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, execution_client=None
):
    if execution_client is None:
        settings = _settings()
        execution_service = ExecutionService(
            settings, credential_provider=None, adapter_factory=canned_adapter_factory  # type: ignore[arg-type]
        )
        execution_client = InProcessExecutionClient(execution_service)
    session_cm = db.session()
    session = await session_cm.__aenter__()
    handler = ToolCallHandler(
        tool_registry=tool_registry,
        registry=server_registry,
        target_validator=target_validator,
        policy_engine=policy_engine,
        risk_engine=RiskEngine(),
        rate_limiter=rate_limiter,
        data_minimizer=DataMinimizer(),
        execution_client=execution_client,
        session=session,
        agent_version="test",
        channel="dev",
        identity_provider_name="mock",
    )
    return handler, session, session_cm


@pytest.mark.asyncio
async def test_read_only_health_check_executes_end_to_end(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, identity_provider
):
    handler, session, cm = await _make_handler(
        db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter
    )
    try:
        identity = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
        request = ToolCallRequest(
            tool_id="database.get_health",
            arguments={},
            target={"environment": "production", "database": "CoreBanking"},
            reason="investigating slowness",
            conversation_id="conv_1",
            request_id=new_id("req"),
            channel="slack",
            channel_account_id="U_MOCK_PLACEHOLDER",
        )
        response = await handler.handle(identity, request)
        assert response.status == ToolCallStatus.EXECUTED
        assert response.result["rows"][0]["active_sessions"] == 42
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_dba_l1_denied_from_killing_session_in_production(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, identity_provider
):
    handler, session, cm = await _make_handler(
        db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter
    )
    try:
        identity = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L1")
        request = ToolCallRequest(
            tool_id="database.kill_session",
            arguments={"session_id": "9182", "reason": "blocking chain"},
            target={"environment": "production", "database": "CoreBanking"},
            reason="mitigate blocking",
            conversation_id="conv_2",
            request_id=new_id("req"),
            channel="slack",
            channel_account_id="U_MOCK_PLACEHOLDER",
        )
        response = await handler.handle(identity, request)
        assert response.status == ToolCallStatus.DENIED
        assert response.failure_code == "UNAUTHORIZED"
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_full_approval_workflow_kill_session(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, identity_provider
):
    """Mirrors the acceptance scenario in spec §54/§68: investigate, propose
    kill_session, require approval, approve, execute, verify."""
    handler, session, cm = await _make_handler(
        db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter
    )
    try:
        identity = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
        request = ToolCallRequest(
            tool_id="database.kill_session",
            arguments={"session_id": "9182", "reason": "blocking 43 sessions"},
            target={"environment": "production", "database": "CoreBanking"},
            reason="mitigate blocking",
            conversation_id="conv_3",
            request_id=new_id("req"),
            channel="slack",
            channel_account_id="U_MOCK_PLACEHOLDER",
        )
        first = await handler.handle(identity, request)
        assert first.status == ToolCallStatus.APPROVAL_REQUIRED
        approval_id = first.approval_id
        assert approval_id

        approver = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
        from numi.gateway.domain.approval import ApprovalDecision, ApprovalEngine

        engine = ApprovalEngine(session)
        await engine.decide(
            approval_id=approval_id, approver=approver, decision=ApprovalDecision.APPROVE
        )

        second_request = request.model_copy(
            update={"approval_id": approval_id, "request_id": new_id("req")}
        )
        second = await handler.handle(identity, second_request)
        assert second.status == ToolCallStatus.EXECUTED
        assert second.result["affected"]["terminated"] is True
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_an_actual_execution_attempt_that_fails_is_reported_as_failed_not_denied(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, identity_provider
):
    """Live-verified bug (Slack test against sqlserver-dev-01, comprehensive_summary
    playbook): `database.get_error_logs` hit a genuine Execution Service failure
    (`success=False, error_code=EXECUTION_FAILED`) — a real *attempt* that didn't
    succeed, not a Gateway refusal. `handle_tool_call`'s single outer
    `except NumiError` previously converted this to `ToolCallStatus.DENIED`
    unconditionally (correct only for pre-execution refusals), which aborted the
    whole investigation instead of reaching the orchestrator's already-correct
    `ToolCallStatus.FAILED` handling (record the failure as evidence and keep
    going — see `_submit_and_relay` in orchestrator.py)."""
    handler, session, cm = await _make_handler(
        db,
        tool_registry,
        server_registry,
        target_validator,
        policy_engine,
        rate_limiter,
        execution_client=_FailingExecutionClient(),
    )
    try:
        identity = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
        request = ToolCallRequest(
            tool_id="database.get_error_logs",
            arguments={"since_minutes": 60, "limit": 50},
            target={"environment": "production", "database": "CoreBanking"},
            reason="comprehensive_summary playbook step",
            conversation_id="conv_5",
            request_id=new_id("req"),
            channel="slack",
            channel_account_id="U_MOCK_PLACEHOLDER",
        )
        response = await handler.handle(identity, request)
        assert response.status == ToolCallStatus.FAILED
        assert response.failure_code == "EXECUTION_FAILED"
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_a_genuine_pre_execution_refusal_still_returns_denied(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, identity_provider
):
    """Regression guard for the FAILED/DENIED split above: a target that was
    never even accepted (INVALID_TARGET) never reaches execution at all, so it
    must stay DENIED, not become FAILED."""
    handler, session, cm = await _make_handler(
        db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter
    )
    try:
        identity = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
        request = ToolCallRequest(
            tool_id="database.get_health",
            arguments={},
            target={
                "environment": "production",
                "instance": "no-such-server-registered",
                "database": "CoreBanking",
            },
            reason="investigating slowness",
            conversation_id="conv_6",
            request_id=new_id("req"),
            channel="slack",
            channel_account_id="U_MOCK_PLACEHOLDER",
        )
        response = await handler.handle(identity, request)
        assert response.status == ToolCallStatus.DENIED
        assert response.failure_code == "INVALID_TARGET"
    finally:
        await cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_execution_denied_if_agent_alters_action_after_approval(
    db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter, identity_provider
):
    """The exact scenario from spec §44 run through the full Gateway pipeline."""
    handler, session, cm = await _make_handler(
        db, tool_registry, server_registry, target_validator, policy_engine, rate_limiter
    )
    try:
        identity = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
        request = ToolCallRequest(
            tool_id="database.kill_session",
            arguments={"session_id": "9182", "reason": "blocking chain"},
            target={"environment": "production", "database": "CoreBanking"},
            reason="mitigate blocking",
            conversation_id="conv_4",
            request_id=new_id("req"),
            channel="slack",
            channel_account_id="U_MOCK_PLACEHOLDER",
        )
        first = await handler.handle(identity, request)
        approval_id = first.approval_id

        from numi.gateway.domain.approval import ApprovalDecision, ApprovalEngine

        engine = ApprovalEngine(session)
        await engine.decide(
            approval_id=approval_id, approver=identity, decision=ApprovalDecision.APPROVE
        )

        tampered = request.model_copy(
            update={
                "approval_id": approval_id,
                "request_id": new_id("req"),
                "arguments": {"session_id": "9183", "reason": "blocking chain"},
            }
        )
        response = await handler.handle(identity, tampered)
        assert response.status == ToolCallStatus.DENIED
        assert response.failure_code == "APPROVAL_MISMATCH"
    finally:
        await cm.__aexit__(None, None, None)
