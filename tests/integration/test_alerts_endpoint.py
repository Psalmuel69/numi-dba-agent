"""Alert-triggered investigation across the service boundary — the
`channels./webhooks/alerts` -> `agent./v1/alerts/trigger` hop, and (in the
full-stack test) all the way through to a real Gateway/Execution round-trip
and a captured Slack post.

Mirrors `test_digest_notify_endpoint.py`'s reasoning: authentication and the
audience-string handshake between two independently-tested services are
exactly the kind of bug neither side's own unit tests can see.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from numi.agent.api.app import create_app as create_agent_app
from numi.channels.api.app import create_app as create_channels_app
from numi.channels.slack.sender import SlackMessageSender
from numi.common.service_auth import ServiceTokenIssuer
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory
from tests.stack import test_settings as build_test_settings

ALERT_SECRET = "test-alert-webhook-secret"


def _sign(secret: str, timestamp: str, body: bytes) -> str:
    base = f"{timestamp}.".encode() + body
    return "sha256=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def _post_alert(client, secret, body: dict, *, timestamp: str | None = None):
    raw = json.dumps(body).encode()
    ts = timestamp or str(int(time.time()))
    return client.post(
        "/webhooks/alerts",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "X-Numi-Alert-Timestamp": ts,
            "X-Numi-Alert-Signature": _sign(secret, ts, raw),
        },
    )


# ------------------------------------------------------------- Agent endpoint ---
# Auth-only: the runner behind this endpoint (server resolution, the
# investigation, publishing) is covered by test_alert_trigger.py and
# test_scheduled_digest_never_writes.py. This file only needs to prove the
# route rejects what it should before ever reaching that logic.


@pytest.fixture
def agent_settings():
    return build_test_settings(alert_webhook_slack_channel="C_ALERTS")


def test_alerts_trigger_without_a_service_token_is_rejected(agent_settings):
    app = create_agent_app(agent_settings)
    with TestClient(app) as client:
        response = client.post("/v1/alerts/trigger", json={"server": "postgres-local"})
    assert response.status_code == 401


def test_alerts_trigger_with_a_token_for_a_different_audience_is_rejected(agent_settings):
    app = create_agent_app(agent_settings)
    issuer = ServiceTokenIssuer(agent_settings.service_jwt_secret, agent_settings.service_jwt_issuer)
    wrong_audience = issuer.issue(service_name="channels", audience="numi-gateway")
    with TestClient(app) as client:
        response = client.post(
            "/v1/alerts/trigger",
            json={"server": "postgres-local"},
            headers={"X-Service-Token": wrong_audience},
        )
    assert response.status_code == 401


# --------------------------------------------------------------- Channels endpoint ---


@pytest.fixture
def channels_settings():
    return build_test_settings(alert_webhook_secret=ALERT_SECRET)


@pytest.fixture
def agent_call_log():
    """Every request the channels app's outbound client would have sent to
    the Agent, via an `httpx.MockTransport` standing in for the real
    network hop — so these tests exercise the webhook route's own auth,
    parsing, and dedup logic without a live Agent process."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    return calls, httpx.MockTransport(handler)


@pytest.fixture
def channels_app(channels_settings, agent_call_log):
    _, transport = agent_call_log
    return create_channels_app(channels_settings, agent_transport=transport)


@pytest.fixture
def posted(monkeypatch):
    captured: list[tuple[str, str, list]] = []

    async def _capture(self, channel: str, text: str, blocks: list) -> None:
        captured.append((channel, text, blocks))

    monkeypatch.setattr(SlackMessageSender, "post_message", _capture)
    return captured


def test_alerts_webhook_without_a_signature_is_rejected(channels_app, agent_call_log):
    calls, _ = agent_call_log
    with TestClient(channels_app) as client:
        response = client.post("/webhooks/alerts", json={"server": "postgres-local"})
    assert response.status_code == 401
    assert calls == []


def test_alerts_webhook_with_a_wrong_secret_is_rejected(channels_app, agent_call_log):
    calls, _ = agent_call_log
    with TestClient(channels_app) as client:
        response = _post_alert(client, "wrong-secret", {"server": "postgres-local"})
    assert response.status_code == 401
    assert calls == []


def test_alerts_webhook_with_an_unconfigured_secret_always_rejects(agent_call_log):
    """The one-switch guarantee: an empty `alert_webhook_secret` must refuse
    every request, never fall back to treating them as trusted."""
    _, transport = agent_call_log
    app = create_channels_app(build_test_settings(alert_webhook_secret=""), agent_transport=transport)
    with TestClient(app) as client:
        # Even a "correctly" self-signed request (signed with the empty
        # string) must not be accepted.
        response = _post_alert(client, "", {"server": "postgres-local"})
    assert response.status_code == 401


def test_alerts_webhook_missing_server_field_is_rejected(channels_app, agent_call_log):
    calls, _ = agent_call_log
    with TestClient(channels_app) as client:
        response = _post_alert(client, ALERT_SECRET, {"metric": "cpu"})
    assert response.status_code == 400
    assert calls == []


def test_a_valid_signed_alert_is_forwarded_to_the_agent(channels_app, agent_call_log):
    calls, _ = agent_call_log
    with TestClient(channels_app) as client:
        response = _post_alert(
            client, ALERT_SECRET, {"server": "postgres-local", "metric": "connections_used_pct"}
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(calls) == 1
    forwarded = json.loads(calls[0].content)
    assert forwarded["server"] == "postgres-local"
    assert forwarded["metric"] == "connections_used_pct"


def test_alerts_webhook_retry_with_the_same_alert_id_is_deduplicated(channels_app, agent_call_log):
    calls, _ = agent_call_log
    body = {"server": "postgres-local", "alert_id": "alert-fingerprint-123"}
    with TestClient(channels_app) as client:
        first = _post_alert(client, ALERT_SECRET, body)
        second = _post_alert(client, ALERT_SECRET, body)
    assert first.status_code == 200
    assert second.status_code == 200
    # Only the first delivery actually reached the Agent.
    assert len(calls) == 1


def test_alerts_webhook_without_an_alert_id_is_never_deduplicated(channels_app, agent_call_log):
    """Optional field: a sender that omits it simply gets no dedup, not a
    rejected request."""
    calls, _ = agent_call_log
    body = {"server": "postgres-local"}
    with TestClient(channels_app) as client:
        _post_alert(client, ALERT_SECRET, body)
        _post_alert(client, ALERT_SECRET, body)
    assert len(calls) == 2


# --------------------------------------------------------- full stack, wired ---


@pytest.mark.asyncio
async def test_a_signed_alert_reaches_a_real_investigation_and_posts_to_slack():
    """The whole path: a signed external webhook -> Channels -> Agent ->
    Gateway -> Execution -> back to Channels' `/v1/notify` -> a captured
    Slack post. Uses the real, deterministic mock LLM planner
    (`llm_provider="mock"`, `tests.stack`'s default) — no live model call,
    but every other hop is the real thing.

    `tests.stack.Stack` isn't used here: it wires Agent -> Gateway but not
    Agent -> Channels (nothing in the ordinary DBA-chat path needs that leg
    — only the digest/alert paths do), so this test wires all four apps
    itself, the same way test_digest_notify_endpoint.py wires that one
    extra leg on its own. Channels -> Agent needs Agent already built, and
    Agent -> Channels needs Channels already built; broken by constructing
    the Agent -> Channels transport with a placeholder `app` and fixing it
    up once the Channels app actually exists — `ASGITransport` doesn't
    resolve `.app` until a request is actually made through it, by which
    point the fix-up has already happened."""
    settings = build_test_settings(
        alert_webhook_secret=ALERT_SECRET,
        alert_webhook_slack_channel="C_ALERTS",
        alert_webhook_identity_channel="dev",
        alert_webhook_identity_account="dba_l2@example.com",
    )
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    gateway_app = create_gateway_app(
        settings, execution_transport=httpx.ASGITransport(app=execution_app)
    )
    await gateway_app.state.gateway.db.create_all()

    channels_transport = httpx.ASGITransport(app=None)  # fixed up below
    agent_app = create_agent_app(
        settings,
        gateway_transport=httpx.ASGITransport(app=gateway_app),
        channels_transport=channels_transport,
    )
    channels_app = create_channels_app(settings, agent_transport=httpx.ASGITransport(app=agent_app))
    channels_transport.app = channels_app

    posted: list[tuple[str, str, list]] = []

    async def _capture(self, channel: str, text: str, blocks: list) -> None:
        posted.append((channel, text, blocks))

    import numi.channels.slack.sender as sender_module

    original = sender_module.SlackMessageSender.post_message
    sender_module.SlackMessageSender.post_message = _capture
    try:
        body = {
            "server": "postgres-local",
            "metric": "connections_used_pct",
            "current_value": "92",
            "threshold": "85",
            "severity": "warning",
            "source": "test-monitor",
        }
        raw = json.dumps(body).encode()
        ts = str(int(time.time()))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=channels_app), base_url="http://channels"
        ) as client:
            response = await client.post(
                "/webhooks/alerts",
                content=raw,
                headers={
                    "Content-Type": "application/json",
                    "X-Numi-Alert-Timestamp": ts,
                    "X-Numi-Alert-Signature": _sign(ALERT_SECRET, ts, raw),
                },
            )
    finally:
        sender_module.SlackMessageSender.post_message = original

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert len(posted) == 1
    channel, text, _ = posted[0]
    assert channel == "C_ALERTS"
    assert "postgres-local" in text
    assert "connections_used_pct" in text
