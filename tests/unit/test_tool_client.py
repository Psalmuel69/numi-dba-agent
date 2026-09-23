"""ToolClient (spec §6, §18, §36) — the Agent's only route to a tool call.

No live HTTP: `httpx.MockTransport` stands in for the Gateway, matching the
client's own `transport=` constructor param (there specifically so this
class is testable without a real server)."""

from __future__ import annotations

import httpx
import pytest

from numi.agent.tool_client import ToolClient
from numi.common.models.tool import ToolCallRequest, ToolCallStatus
from numi.common.service_auth import ServiceTokenIssuer


def _issuer() -> ServiceTokenIssuer:
    return ServiceTokenIssuer("shared-secret", "numi-internal")


def _client(handler) -> ToolClient:
    return ToolClient("http://gateway", _issuer(), transport=httpx.MockTransport(handler))


def _tool_definition_payload(**overrides) -> dict:
    payload = {
        "tool_id": "database.get_health",
        "version": "1",
        "description": "Checks server health.",
        "operation_type": "READ",
        "risk_level": "LOW",
        "reversible": True,
        "availability_impact": False,
        "data_modification": False,
        "requires_approval": False,
        "allowed_roles": ["DBA_L1"],
        "allowed_environments": ["development"],
        "required_target_scope": ["server"],
        "argument_schema": {},
        "result_schema": {},
    }
    payload.update(overrides)
    return payload


def _tool_call_request(**overrides) -> ToolCallRequest:
    fields = {
        "tool_id": "database.get_health",
        "arguments": {},
        "target": {"environment": "development"},
        "reason": "investigating slowness",
        "conversation_id": "conv_1",
        "request_id": "req_1",
        "channel": "slack",
        "channel_account_id": "U_MOCK_L2",
    }
    fields.update(overrides)
    return ToolCallRequest(**fields)


@pytest.mark.asyncio
async def test_available_tools_sends_a_service_token_and_parses_the_list():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["token"] = request.headers.get("X-Service-Token")
        return httpx.Response(200, json=[_tool_definition_payload()])

    client = _client(handler)
    tools = await client.available_tools("slack", "U_MOCK_L2")

    assert seen["path"] == "/v1/tools"
    assert seen["params"] == {"channel": "slack", "channel_account_id": "U_MOCK_L2"}
    assert seen["token"]
    assert len(tools) == 1
    assert tools[0].tool_id == "database.get_health"


@pytest.mark.asyncio
async def test_available_tools_raises_on_a_server_error():
    client = _client(lambda request: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        await client.available_tools("slack", "U_MOCK_L2")


@pytest.mark.asyncio
async def test_list_servers_returns_the_raw_json_list():
    client = _client(lambda request: httpx.Response(200, json=[{"id": "postgres-dev-01"}]))
    servers = await client.list_servers()
    assert servers == [{"id": "postgres-dev-01"}]


@pytest.mark.asyncio
async def test_get_server_catalog_returns_the_catalog():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/catalog/servers/postgres-dev-01"
        return httpx.Response(200, json={"server_id": "postgres-dev-01"})

    client = _client(handler)
    catalog = await client.get_server_catalog("postgres-dev-01")
    assert catalog == {"server_id": "postgres-dev-01"}


@pytest.mark.asyncio
async def test_get_server_catalog_returns_empty_dict_for_an_undiscovered_server():
    """404 here means "never discovered yet", not a failure — the caller
    treats an empty dict as "no catalog", distinct from a real error."""
    client = _client(lambda request: httpx.Response(404))
    assert await client.get_server_catalog("unknown-server") == {}


@pytest.mark.asyncio
async def test_get_server_catalog_still_raises_on_a_real_server_error():
    client = _client(lambda request: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_server_catalog("postgres-dev-01")


@pytest.mark.asyncio
async def test_refresh_catalog_without_a_server_id_hits_the_bare_refresh_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/catalog/refresh"
        return httpx.Response(200, json={"status": "OK"})

    client = _client(handler)
    result = await client.refresh_catalog("slack", "U_MOCK_L2")
    assert result == {"status": "OK"}


@pytest.mark.asyncio
async def test_refresh_catalog_with_a_server_id_hits_the_scoped_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/catalog/refresh/postgres-dev-01"
        return httpx.Response(200, json={"status": "OK"})

    client = _client(handler)
    await client.refresh_catalog("slack", "U_MOCK_L2", server_id="postgres-dev-01")


@pytest.mark.asyncio
async def test_refresh_catalog_degrades_a_server_error_to_a_clean_error_dict():
    """Unlike submit()/available_tools(), a failed refresh is never fatal to
    the caller — verified live: a genuinely unreachable target during
    discovery must not block the tool call that triggered the refresh."""
    client = _client(lambda request: httpx.Response(500, json={"detail": "unreachable"}))
    result = await client.refresh_catalog("slack", "U_MOCK_L2", server_id="postgres-dev-01")
    assert result == {"status": "ERROR", "detail": "unreachable"}


@pytest.mark.asyncio
async def test_submit_parses_a_successful_tool_call_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/tool-calls"
        return httpx.Response(
            200,
            json={"status": "EXECUTED", "message": "ok", "result": {"rows": []}},
        )

    client = _client(handler)
    response = await client.submit(_tool_call_request())
    assert response.status == ToolCallStatus.EXECUTED
    assert response.result == {"rows": []}


@pytest.mark.asyncio
async def test_submit_raises_on_a_server_error_instead_of_returning_a_fake_response():
    """Distinct on purpose from refresh_catalog/approve/reject's clean-dict
    degradation: a tool call's outcome is never safe to paper over with a
    synthetic status, so this is the one call site orchestrator.py's own
    tool_call_submit_failed handling exists to catch."""
    client = _client(lambda request: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        await client.submit(_tool_call_request())


@pytest.mark.asyncio
async def test_approve_posts_the_decision_and_returns_the_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/approvals/appr_1/approve"
        return httpx.Response(200, json={"status": "APPROVED"})

    client = _client(handler)
    result = await client.approve("appr_1", "slack", "U_MOCK_L3")
    assert result == {"status": "APPROVED"}


@pytest.mark.asyncio
async def test_approve_degrades_a_server_error_to_a_clean_error_dict():
    client = _client(lambda request: httpx.Response(409, json={"detail": "already decided"}))
    result = await client.approve("appr_1", "slack", "U_MOCK_L3")
    assert result == {"status": "ERROR", "detail": "already decided"}


@pytest.mark.asyncio
async def test_reject_posts_the_decision_and_returns_the_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/approvals/appr_1/reject"
        return httpx.Response(200, json={"status": "REJECTED"})

    client = _client(handler)
    result = await client.reject("appr_1", "slack", "U_MOCK_L3")
    assert result == {"status": "REJECTED"}


@pytest.mark.asyncio
async def test_reject_degrades_a_server_error_to_a_clean_error_dict():
    client = _client(lambda request: httpx.Response(409, json={"detail": "already decided"}))
    result = await client.reject("appr_1", "slack", "U_MOCK_L3")
    assert result == {"status": "ERROR", "detail": "already decided"}
