"""The Channels service's `/v1/notify` endpoint — the digest's one delivery
route (Agent -> Channels -> Slack).

Two things are worth testing across the service boundary rather than in
isolation. First, authentication: this is the only inbound path on the
Channels service that isn't a webhook authenticated by its own transport's
scheme, and an unauthenticated "post arbitrary text into a DBA channel"
endpoint would be a real hole — so a missing, forged, or wrong-audience
token must all be refused. Second, the audience string itself: the Agent
issues `numi-channels` and the Channels service verifies `numi-channels`,
and a mismatch between those two literals is exactly the kind of bug no
unit test on either side can see, but which would silently break the digest
every morning in production.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from numi.agent.scheduled_report import ChannelsDigestPublisher
from numi.channels.api.app import create_app as create_channels_app
from numi.channels.slack.sender import SlackMessageSender
from numi.common.service_auth import ServiceTokenIssuer

# Aliased on import: pytest would otherwise collect the helper's own
# `test_`-prefixed name as a test case in this module.
from tests.stack import test_settings as build_test_settings


@pytest.fixture
def settings():
    return build_test_settings()


@pytest.fixture
def channels_app(settings):
    return create_channels_app(settings)


@pytest.fixture
def issuer(settings):
    return ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)


@pytest.fixture
def posted(monkeypatch):
    """Records what would have gone to Slack. Patched on the class so the
    sender the app already constructed is covered."""
    captured: list[tuple[str, str, list]] = []

    async def _capture(self, channel: str, text: str, blocks: list) -> None:
        captured.append((channel, text, blocks))

    monkeypatch.setattr(SlackMessageSender, "post_message", _capture)
    return captured


def test_notify_without_a_service_token_is_rejected(channels_app, posted):
    with TestClient(channels_app) as client:
        response = client.post("/v1/notify", json={"channel_id": "C_DBA", "text": "hi"})

    assert response.status_code == 401
    assert posted == []


def test_notify_with_a_forged_service_token_is_rejected(channels_app, settings, posted):
    forged = ServiceTokenIssuer("wrong-secret", settings.service_jwt_issuer).issue(
        service_name="agent", audience="numi-channels"
    )

    with TestClient(channels_app) as client:
        response = client.post(
            "/v1/notify",
            json={"channel_id": "C_DBA", "text": "hi"},
            headers={"X-Service-Token": forged},
        )

    assert response.status_code == 401
    assert posted == []


def test_notify_with_a_token_for_a_different_audience_is_rejected(channels_app, issuer, posted):
    """A correctly-signed token minted for another hop (Agent -> Gateway)
    must not be replayable here — audience scoping is the point of
    `common.service_auth`."""
    wrong_audience = issuer.issue(service_name="agent", audience="numi-gateway")

    with TestClient(channels_app) as client:
        response = client.post(
            "/v1/notify",
            json={"channel_id": "C_DBA", "text": "hi"},
            headers={"X-Service-Token": wrong_audience},
        )

    assert response.status_code == 401
    assert posted == []


def test_a_valid_notify_posts_the_text_to_the_requested_channel(channels_app, issuer, posted):
    token = issuer.issue(service_name="agent", audience="numi-channels")

    with TestClient(channels_app) as client:
        response = client.post(
            "/v1/notify",
            json={"channel_id": "C_DBA", "text": "Numi daily health digest — ..."},
            headers={"X-Service-Token": token},
        )

    assert response.status_code == 200
    assert len(posted) == 1
    channel, text, blocks = posted[0]
    assert channel == "C_DBA"
    assert text.startswith("Numi daily health digest")
    # Text only: a digest can never carry an approval card, so there is no
    # path here for an unattended run to put a clickable action in front of
    # a DBA (see `NotifyRequest`'s own docstring).
    assert all(block.get("type") != "actions" for block in blocks)


@pytest.mark.asyncio
async def test_the_agents_publisher_reaches_the_channels_endpoint(channels_app, issuer, posted):
    """The two halves wired together over ASGI: the Agent's publisher mints
    a token and the Channels service accepts it. Catches an audience-literal
    mismatch that neither side's own tests could see."""
    publisher = ChannelsDigestPublisher(
        "http://channels", issuer, transport=httpx.ASGITransport(app=channels_app)
    )

    await publisher.publish(channel_id="C_DBA", text="digest body")

    assert [(c, t) for c, t, _ in posted] == [("C_DBA", "digest body")]
