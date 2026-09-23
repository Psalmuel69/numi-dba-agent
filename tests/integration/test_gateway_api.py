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


def _build_clients():
    settings = _settings()
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    execution_transport = httpx.ASGITransport(app=execution_app)
    gateway_app = create_gateway_app(settings, execution_transport=execution_transport)
    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    agent_token = issuer.issue(service_name="agent", audience="numi-gateway")
    return gateway_app, agent_token


def test_tool_call_requires_service_token():
    gateway_app, _token = _build_clients()
    with TestClient(gateway_app) as client:
        response = client.post(
            "/v1/tool-calls",
            json={
                "tool_id": "database.get_health",
                "arguments": {},
                "target": {"environment": "production", "database": "CoreBanking"},
                "reason": "test",
                "conversation_id": "conv_x",
                "request_id": "req_x",
                "channel": "slack",
                "channel_account_id": "U_MOCK_L2",
            },
        )
        assert response.status_code == 401


def test_full_http_round_trip_health_check_via_gateway_to_execution_service():
    gateway_app, token = _build_clients()
    with TestClient(gateway_app) as client:
        response = client.post(
            "/v1/tool-calls",
            headers={"X-Service-Token": token},
            json={
                "tool_id": "database.get_health",
                "arguments": {},
                "target": {"environment": "production", "database": "CoreBanking"},
                "reason": "test",
                "conversation_id": "conv_x",
                "request_id": "req_x",
                "channel": "slack",
                "channel_account_id": "U_MOCK_L2",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "EXECUTED"
        assert body["result"]["rows"][0]["active_sessions"] == 42


def test_non_dba_channel_account_is_denied_over_http():
    gateway_app, token = _build_clients()
    with TestClient(gateway_app) as client:
        response = client.post(
            "/v1/tool-calls",
            headers={"X-Service-Token": token},
            json={
                "tool_id": "database.get_health",
                "arguments": {},
                "target": {"environment": "production", "database": "CoreBanking"},
                "reason": "test",
                "conversation_id": "conv_x",
                "request_id": "req_x",
                "channel": "slack",
                "channel_account_id": "U_MOCK_NONDBA",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "DENIED"
        assert body["failure_code"] == "UNAUTHORIZED"


def test_unverified_channel_account_is_rejected_before_reaching_agent_logic():
    gateway_app, token = _build_clients()
    with TestClient(gateway_app) as client:
        response = client.post(
            "/v1/tool-calls",
            headers={"X-Service-Token": token},
            json={
                "tool_id": "database.get_health",
                "arguments": {},
                "target": {"environment": "production", "database": "CoreBanking"},
                "reason": "test",
                "conversation_id": "conv_x",
                "request_id": "req_x",
                "channel": "slack",
                "channel_account_id": "U_TOTALLY_UNKNOWN",
            },
        )
        assert response.status_code == 401


def test_list_tools_filtered_by_role_over_http():
    gateway_app, token = _build_clients()
    with TestClient(gateway_app) as client:
        response = client.get(
            "/v1/tools",
            headers={"X-Service-Token": token},
            params={"channel": "slack", "channel_account_id": "U_MOCK_L1"},
        )
        assert response.status_code == 200
        tool_ids = {t["tool_id"] for t in response.json()}
        assert "database.get_health" in tool_ids
        assert "database.kill_session" not in tool_ids  # L1 not in allowed_roles
