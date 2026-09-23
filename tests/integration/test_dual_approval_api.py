"""Dual (two-person) approval, end to end through the Gateway HTTP API
(spec §17).

Only two `/v1/tool-calls` submissions are made per test (the initial
proposal and the post-approval execution) — critical-tier operations are
deliberately rate-limited to a couple of attempts per minute per user/
conversation (config/rate_limits.yaml), and a third submission would
(correctly) be rejected as RATE_LIMITED rather than exercising what this
test is about. The "not yet fully approved" denial path is already covered
at the engine level in tests/unit/test_approval.py.
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from numi.common.config import Settings
from numi.common.service_auth import ServiceTokenIssuer
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        control_db_url="sqlite+aiosqlite:///:memory:",
        service_jwt_secret="test-secret",
        service_jwt_issuer="numi-internal",
    )


def _build():
    settings = _settings()
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    execution_transport = httpx.ASGITransport(app=execution_app)
    gateway_app = create_gateway_app(settings, execution_transport=execution_transport)
    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    token = issuer.issue(service_name="agent", audience="numi-gateway")
    return gateway_app, token


def _submit_restart(
    client: TestClient,
    token: str,
    requester_account: str,
    request_id: str,
    approval_id: str | None = None,
):
    body = {
        "tool_id": "database.restart_instance",
        "arguments": {"reason": "applying patch"},
        "target": {"environment": "development", "instance": "sqlserver-dev-01"},
        "reason": "applying patch",
        "conversation_id": "conv_dual",
        "request_id": request_id,
        "channel": "slack",
        "channel_account_id": requester_account,
    }
    if approval_id:
        body["approval_id"] = approval_id
    return client.post("/v1/tool-calls", headers={"X-Service-Token": token}, json=body)


def _approve(client: TestClient, token: str, approval_id: str, account: str):
    return client.post(
        f"/v1/approvals/{approval_id}/approve",
        headers={"X-Service-Token": token},
        json={"channel": "slack", "channel_account_id": account},
    )


def test_restart_instance_requires_two_distinct_approvers_over_http():
    gateway_app, token = _build()
    with TestClient(gateway_app) as client:
        first = _submit_restart(client, token, "U_MOCK_L3", "req_dual_1")
        assert first.status_code == 200
        body = first.json()
        assert body["status"] == "APPROVAL_REQUIRED"
        approval_id = body["approval_id"]

        # The requester (U_MOCK_L3) cannot approve their own critical action.
        self_approve = _approve(client, token, approval_id, "U_MOCK_L3")
        assert self_approve.status_code == 403

        # First distinct approver.
        first_approve = _approve(client, token, approval_id, "U_MOCK_MGR")
        assert first_approve.status_code == 200
        assert first_approve.json()["status"] == "AWAITING_SECOND_APPROVAL"

        # The same approver trying again must not count as the second one.
        dup_approve = _approve(client, token, approval_id, "U_MOCK_MGR")
        assert dup_approve.status_code == 403

        # Second, distinct, independently-authorized approver completes it.
        second_approve = _approve(client, token, approval_id, "U_MOCK_L3B")
        assert second_approve.status_code == 200
        assert second_approve.json()["status"] == "APPROVED"

        final = _submit_restart(client, token, "U_MOCK_L3", "req_dual_2", approval_id)
        assert final.json()["status"] == "EXECUTED"


def test_dual_approval_not_yet_complete_denies_execution_over_http():
    gateway_app, token = _build()
    with TestClient(gateway_app) as client:
        first = _submit_restart(client, token, "U_MOCK_L3", "req_dual_only_1")
        approval_id = first.json()["approval_id"]

        _approve(client, token, approval_id, "U_MOCK_MGR")  # only one approver so far

        retry = _submit_restart(client, token, "U_MOCK_L3", "req_dual_only_2", approval_id)
        assert retry.json()["status"] == "DENIED"
        assert retry.json()["failure_code"] == "APPROVAL_REQUIRED"
