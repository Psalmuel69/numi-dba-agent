"""Channels service tests — the full stack wired end to end.

channels -> agent -> gateway -> execution, all via in-process ASGI
transports, verifying the whole chain described in spec §4/§66 works,
including identity verification, signature verification, and the dev mock
channel.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from numi.agent.api.app import create_app as create_agent_app
from numi.channels.api.app import _slack_conversation_id
from numi.channels.api.app import create_app as create_channels_app
from numi.channels.slack.sender import SlackMessageSender
from numi.common.config import Settings
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        control_db_url="sqlite+aiosqlite:///:memory:",
        service_jwt_secret="test-secret",
        service_jwt_issuer="numi-internal",
        llm_provider="mock",
        slack_signing_secret="test-slack-signing-secret",
        teams_app_password="dev-teams-shared-token",
        **overrides,
    )


def _build_full_stack(settings: Settings):
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    execution_transport = httpx.ASGITransport(app=execution_app)
    gateway_app = create_gateway_app(settings, execution_transport=execution_transport)
    # TestClient wraps the *channels* app's lifespan only; the gateway app's
    # in-memory sqlite schema is created explicitly here instead of relying
    # on its own (unentered) lifespan.
    asyncio.run(gateway_app.state.gateway.db.create_all())
    gateway_transport = httpx.ASGITransport(app=gateway_app)
    agent_app = create_agent_app(settings, gateway_transport=gateway_transport)
    agent_transport = httpx.ASGITransport(app=agent_app)
    channels_app = create_channels_app(settings, agent_transport=agent_transport)
    return gateway_app, channels_app


def _sign_slack(secret: str, body: bytes, timestamp: str) -> str:
    base = f"v0:{timestamp}:".encode() + body
    return "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def test_slack_bad_signature_is_rejected():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=b'{"type": "url_verification", "challenge": "abc"}',
            headers={
                "X-Slack-Request-Timestamp": str(int(time.time())),
                "X-Slack-Signature": "v0=deadbeef",
            },
        )
        assert response.status_code == 401


def test_slack_url_verification_challenge():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    body = b'{"type": "url_verification", "challenge": "abc123"}'
    ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, body, ts)
    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=body,
            headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200
        assert response.json() == {"challenge": "abc123"}


def test_slack_message_from_verified_dba_is_processed():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    import json

    body = json.dumps(
        {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U_MOCK_L2",
                "text": "hello",
                "channel": "C123",
                "ts": "111.222",
            },
        }
    ).encode()
    ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, body, ts)
    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=body,
            headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_slack_message_edit_event_with_no_top_level_user_does_not_crash_the_webhook():
    """Reproduces a live finding: an unhandled `KeyError: 'user'` 500'd the
    whole webhook. `event.get("bot_id")` alone doesn't cover every
    `message`-typed event this service must ignore -- a `message_changed`
    (edit) carries its author nested under `event["message"]["user"]`, not
    at the top level `event["user"]` the handler used unconditionally.
    Slack retries a delivery it never got a 200 for, so one crash like this
    can also cascade into repeated retries, not just a single dropped
    event. Any `message`-typed event missing a top-level "user" (edits,
    deletions, and other subtypes alike) must be skipped exactly like a
    bot-authored message already is, not crash."""
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    import json

    body = json.dumps(
        {
            "type": "event_callback",
            "event_id": "Ev_EDIT_TEST_1",
            "event": {
                "type": "message",
                "subtype": "message_changed",
                "channel": "C123",
                "ts": "111.300",
                "message": {"user": "U_MOCK_L2", "text": "edited text"},
                "previous_message": {"user": "U_MOCK_L2", "text": "original text"},
                # No top-level "user" -- this is the exact shape that crashed.
            },
        }
    ).encode()
    ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, body, ts)
    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=body,
            headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_a_retried_slack_event_is_deduplicated_not_reprocessed(capsys):
    """Reproduces a live finding: Slack retries a webhook delivery it
    hasn't gotten a fast ack for (its own ~3s timeout), reusing the same
    event_id — and this handler awaits the full Agent round-trip before
    ever returning, which a real multi-turn investigation can easily
    exceed. A real DBA's single Slack message produced two different,
    garbled replies as a result. The fix must make a retried delivery
    (same event_id) a no-op instead of a second, independent run against
    the same shared conversation."""
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    import json

    body = json.dumps(
        {
            "type": "event_callback",
            "event_id": "Ev_DEDUP_TEST_1",
            "event": {
                "type": "message",
                "user": "U_MOCK_L2",
                "text": "hello",
                "channel": "C123",
                "ts": "111.222",
            },
        }
    ).encode()
    ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, body, ts)
    headers = {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig}
    with TestClient(channels_app) as client:
        first = client.post("/webhooks/slack", content=body, headers=headers)
        assert first.status_code == 200
        # Slack's own retry: identical payload, identical event_id.
        second = client.post("/webhooks/slack", content=body, headers=headers)
        assert second.status_code == 200

    assert "slack_event_retry_deduplicated" in capsys.readouterr().out


def test_teams_webhook_requires_valid_dev_token():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/teams",
            json={"from": {"aadObjectId": "aad-mock-l2"}, "text": "hi", "conversation": {"id": "c1"}},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401


def test_teams_webhook_with_valid_dev_token_and_verified_dba():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/teams",
            json={"from": {"aadObjectId": "aad-mock-l2"}, "text": "hi", "conversation": {"id": "c1"}},
            headers={"Authorization": "Bearer dev-teams-shared-token"},
        )
        assert response.status_code == 200


def test_dev_chat_full_round_trip_investigation_and_approval():
    """Spec §66's exact worked example, run through the whole stack."""
    settings = _settings()
    gateway_app, channels_app = _build_full_stack(settings)
    with TestClient(channels_app) as client:
        response = client.post(
            "/dev/chat",
            json={
                "user": "dba_l2@example.com",
                "message": "Check blocking on CoreBanking production.",
                "conversation_id": "dev_conv_1",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "approval_required"
        approval_id = body["approval_card"]["approval_id"]

        approve_response = client.post(
            "/dev/chat/events",
            json={
                "user": "dba_l2@example.com",
                "conversation_id": "dev_conv_1",
                "approval_id": approval_id,
                "decision": "approve",
            },
        )
        assert approve_response.status_code == 200
        assert "Action approved" in approve_response.json()["text"]


def test_dev_chat_unknown_user_is_rejected():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    with TestClient(channels_app) as client:
        response = client.post(
            "/dev/chat", json={"user": "nobody@example.com", "message": "hi"}
        )
        assert response.status_code == 401


def test_dev_chat_non_dba_user_is_denied():
    settings = _settings()
    _gw, channels_app = _build_full_stack(settings)
    with TestClient(channels_app) as client:
        response = client.post(
            "/dev/chat", json={"user": "notadba@example.com", "message": "hi"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "denied"


def test_slack_conversation_id_for_a_threaded_reply_is_unchanged():
    """A threaded reply's conversation is still scoped by thread_ts alone
    — unchanged, and deliberately shared by whoever else replies in the
    same thread, since a Slack thread is inherently one collaborative
    investigation."""
    assert (
        _slack_conversation_id("C123", "U_A", "100.001")
        == _slack_conversation_id("C123", "U_B", "100.001")
        == "slack:C123:100.001"
    )


def test_slack_conversation_id_for_plain_messages_from_the_same_dba_is_stable():
    """Live-reproduced regression: an ordinary, non-threaded message used
    to fall back to its own `ts` — unique to that one message — so two
    consecutive plain messages from the same DBA in the same channel
    (exactly how a DBA naturally follows up when the bot "responds to
    plain messages too", with no @-mention and no thread reply) got two
    different conversation_ids, and therefore two entirely separate,
    empty `ConversationState`s at the Agent. Scoped to `(channel, user)`
    now, so consecutive plain messages share one conversation."""
    first = _slack_conversation_id("C123", "U_MOCK_L2", "")
    second = _slack_conversation_id("C123", "U_MOCK_L2", "")
    assert first == second


def test_slack_conversation_id_for_plain_messages_from_different_dbas_never_crosses():
    """The fix must not go too far the other way: two different DBAs
    typing plain (non-threaded) messages in the same shared channel are
    still independent conversations, never merged into one."""
    assert _slack_conversation_id("C123", "U_MOCK_L2", "") != _slack_conversation_id(
        "C123", "U_OTHER", ""
    )


def test_three_plain_non_threaded_slack_messages_from_the_same_dba_share_one_conversation():
    """The exact live reproduction, end to end through the real
    `/webhooks/slack` handler (signature verification, identity
    resolution, the whole path) but with a stub `/v1/chat` standing in for
    the Agent — this isolates the channels-layer bug precisely, without
    depending on any LLM behavior. All three turns of the real, live
    sequence, in order, none of them a threaded reply and none but the
    first naming any entity at all — exactly as this channel invites ("no
    @-mention needed... responds to plain messages too"):

    1. "check backup health on postgres-local" (names the server; the
       investigation that follows concludes normally).
    2. "and what about replication?" — zero named entities; got the wrong
       "Which environment should I investigate?" live.
    3. "development" — a bare answer to exactly that clarification,
       matching `orchestrator._ENVIRONMENT_ANSWER_RE`'s deterministic
       resume path (see tests/unit/test_environment_clarification.py);
       got the generic "tell me more" chitchat fallback live instead of
       resuming.

    All three must reach the Agent as the SAME conversation_id. Before the
    fix, every one of these (having no `thread_ts` of its own) fell back
    to its own unique `ts` — three different, brand-new, empty
    `ConversationState`s in a row. The Agent's own orchestrator logic is
    NOT the bug (verified separately: tests/unit/test_environment_
    clarification.py's test_a_later_fresh_investigation_never_reasks_for_
    an_already_known_environment exercises the same zero-entity follow-up
    shape, and test_a_bare_environment_answer_continues_without_calling_
    extract_intent exercises the same bare-answer resume, both passing —
    only reachable at all when conversation_id is actually held constant,
    which is exactly what a real, non-threaded Slack exchange failed to
    do)."""
    settings = _settings()
    seen_conversation_ids: list[str] = []

    stub_agent = FastAPI()

    @stub_agent.post("/v1/chat")
    async def _chat(body: dict) -> dict:
        seen_conversation_ids.append(body["conversation_id"])
        return {"text": "ok"}

    channels_app = create_channels_app(
        settings, agent_transport=httpx.ASGITransport(app=stub_agent)
    )

    def _post_plain_message(text: str, ts: str) -> None:
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U_MOCK_L2",
                    "text": text,
                    "channel": "C123",
                    "ts": ts,
                    # Deliberately no "thread_ts" — an ordinary channel
                    # message, never posted as a threaded reply, exactly
                    # like the live reproduction.
                },
            }
        ).encode()
        request_ts = str(int(time.time()))
        sig = _sign_slack(settings.slack_signing_secret, body, request_ts)
        response = client.post(
            "/webhooks/slack",
            content=body,
            headers={"X-Slack-Request-Timestamp": request_ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200

    with TestClient(channels_app) as client:
        _post_plain_message("check backup health on postgres-local", "100.001")
        _post_plain_message("and what about replication?", "100.002")
        _post_plain_message("development", "100.003")

    assert len(seen_conversation_ids) == 3
    assert seen_conversation_ids[0] == seen_conversation_ids[1] == seen_conversation_ids[2]


def test_slack_interactive_button_click_resolves_to_the_same_conversation_as_the_originating_message():
    """The sibling bug to the one `_slack_conversation_id` fixed above, found
    live: clicking Approve/Reject on an approval card never resolved the
    pending approval. `/webhooks/slack/interactive` computed its own
    `conversation_id` from the approval CARD's `container.message_ts` --
    unique to that one card message and never used as a conversation_id
    anywhere else -- instead of going through `_slack_conversation_id` like
    the regular message path (this file's tests above) now does. Since the
    card's own message_ts can never match the conversation_id the original
    request (and its `state.pending_approval`) actually lives under,
    `_call_agent_event`'s approve/reject lookup was guaranteed to find no
    matching `ConversationState`, no matter who clicked or what card.

    Isolates the channels-layer contract with a stub `/v1/chat` and
    `/v1/chat/events` standing in for the Agent -- the same pattern as
    `test_three_plain_non_threaded_slack_messages_from_the_same_dba_share_one_conversation`
    above -- so it doesn't depend on any LLM/Gateway behavior. See
    `test_slack_approval_card_button_click_actually_resolves_the_pending_approval_end_to_end`
    below for the full, real-stack round trip proving a button click
    actually resolves a real pending approval, not just that the ids happen
    to match in isolation."""
    settings = _settings()
    seen_chat_conversation_ids: list[str] = []
    seen_event_conversation_ids: list[str] = []

    stub_agent = FastAPI()

    @stub_agent.post("/v1/chat")
    async def _chat(body: dict) -> dict:
        seen_chat_conversation_ids.append(body["conversation_id"])
        return {
            "text": "This requires approval.",
            "status": "approval_required",
            "approval_card": {
                "approval_id": "appr-123",
                "tool_id": "kill_session",
                "target_summary": "corebanking-prod",
                "reason": "blocking session",
                "risk_level": "high",
                "blast_radius": "single session",
            },
        }

    @stub_agent.post("/v1/chat/events")
    async def _events(body: dict) -> dict:
        seen_event_conversation_ids.append(body["conversation_id"])
        return {"text": "Action approved.", "status": "ok"}

    channels_app = create_channels_app(
        settings, agent_transport=httpx.ASGITransport(app=stub_agent)
    )

    plain_message_body = json.dumps(
        {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U_MOCK_L2",
                "text": "check blocking on corebanking production",
                "channel": "C123",
                "ts": "300.001",
                # No thread_ts -- an ordinary channel message, exactly like
                # the approval card the Agent's reply to it will carry.
            },
        }
    ).encode()
    request_ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, plain_message_body, request_ts)

    # Slack's real block_actions shape: the card's OWN message_ts
    # (container.message_ts) is deliberately different from anything the
    # originating plain message used -- that mismatch is exactly the bug.
    interactive_payload = json.dumps(
        {
            "type": "block_actions",
            "user": {"id": "U_MOCK_L2"},
            "channel": {"id": "C123"},
            "container": {"type": "message", "message_ts": "300.999", "channel_id": "C123"},
            "actions": [{"action_id": "numi_approve", "value": "appr-123"}],
        }
    )

    with TestClient(channels_app) as client:
        chat_response = client.post(
            "/webhooks/slack",
            content=plain_message_body,
            headers={"X-Slack-Request-Timestamp": request_ts, "X-Slack-Signature": sig},
        )
        assert chat_response.status_code == 200

        interactive_response = client.post(
            "/webhooks/slack/interactive", data={"payload": interactive_payload}
        )
        assert interactive_response.status_code == 200

    assert len(seen_chat_conversation_ids) == 1
    assert len(seen_event_conversation_ids) == 1
    assert seen_chat_conversation_ids[0] == seen_event_conversation_ids[0]


def test_slack_approval_card_button_click_actually_resolves_the_pending_approval_end_to_end(monkeypatch):
    """The full, real-stack proof that the fix isn't just id-equality on
    paper: a real plain (non-threaded) Slack message drives a real
    investigation through the real Agent -> Gateway -> Execution stack to
    APPROVAL_REQUIRED (spec §66's exact worked example, from
    `test_dev_chat_full_round_trip_investigation_and_approval` above,
    replayed through Slack instead of `/dev/chat`); the approval card
    actually posted back to Slack is captured; and clicking its Approve
    button (`/webhooks/slack/interactive`, the real form-encoded shape
    Slack sends) must resolve the SAME `state.pending_approval` the
    investigation created -- not silently miss it and fall back to the
    generic "There is no pending approval on this conversation" error
    `AgentOrchestrator.handle_approval_decision` returns whenever
    conversation_id doesn't match."""
    settings = _settings()
    _gateway_app, channels_app = _build_full_stack(settings)

    posted: list[tuple[str, str, list[dict]]] = []

    async def _capture_post_message(self, channel, text, blocks):
        posted.append((channel, text, blocks))

    monkeypatch.setattr(SlackMessageSender, "post_message", _capture_post_message)

    plain_message_body = json.dumps(
        {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U_MOCK_L2",
                "text": "Check blocking on CoreBanking production.",
                "channel": "C_APPROVAL",
                "ts": "400.001",
                # No thread_ts -- an ordinary channel message.
            },
        }
    ).encode()
    request_ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, plain_message_body, request_ts)

    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=plain_message_body,
            headers={"X-Slack-Request-Timestamp": request_ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200
        assert len(posted) == 1
        _channel, _text, blocks = posted[0]
        actions_block = next(b for b in blocks if b["type"] == "actions")
        approval_id = actions_block["elements"][0]["value"]
        assert approval_id

        # The real Slack interactive payload shape: form-encoded
        # `payload={...}`, the card's own (different) message_ts in
        # `container`, no thread_ts (the card was posted as a plain
        # channel message, not a threaded reply).
        interactive_payload = json.dumps(
            {
                "type": "block_actions",
                "user": {"id": "U_MOCK_L2"},
                "channel": {"id": "C_APPROVAL"},
                "container": {
                    "type": "message",
                    "message_ts": "400.777",
                    "channel_id": "C_APPROVAL",
                },
                "actions": [{"action_id": "numi_approve", "value": approval_id}],
            }
        )
        interactive_response = client.post(
            "/webhooks/slack/interactive", data={"payload": interactive_payload}
        )
        assert interactive_response.status_code == 200

    assert len(posted) == 2
    _channel, approve_text, _blocks = posted[1]
    assert "Action approved" in approve_text
    assert "There is no pending approval" not in approve_text


def test_slack_approval_card_buttons_collapse_after_a_decision_is_clicked(monkeypatch):
    """A resolved approval card must stop offering both buttons, not just
    stop honoring a second click server-side -- found live: the card sat
    fully clickable forever after a real Approve/Reject, since nothing ever
    rewrote the original message. `slack_interactive` must call
    `SlackMessageSender.update_message` (`chat.update`) against the SAME
    message (`payload["message"]["ts"]`, Slack's own echo of the card being
    acted on) with the `actions` block (matched by its `numi_approval_*`
    block_id) replaced by a static resolved line -- proof the buttons are
    actually gone, not merely that the approval itself resolved (already
    covered above)."""
    settings = _settings()
    _gateway_app, channels_app = _build_full_stack(settings)

    posted: list[tuple[str, str, list[dict]]] = []
    updated: list[tuple[str, str, str, list[dict]]] = []

    async def _capture_post_message(self, channel, text, blocks):
        posted.append((channel, text, blocks))

    async def _capture_update_message(self, channel, ts, text, blocks):
        updated.append((channel, ts, text, blocks))

    monkeypatch.setattr(SlackMessageSender, "post_message", _capture_post_message)
    monkeypatch.setattr(SlackMessageSender, "update_message", _capture_update_message)

    plain_message_body = json.dumps(
        {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U_MOCK_L2",
                "text": "Check blocking on CoreBanking production.",
                "channel": "C_COLLAPSE",
                "ts": "500.001",
            },
        }
    ).encode()
    request_ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, plain_message_body, request_ts)

    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=plain_message_body,
            headers={"X-Slack-Request-Timestamp": request_ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200
        assert len(posted) == 1
        _channel, _text, card_blocks = posted[0]
        actions_block = next(b for b in card_blocks if b["type"] == "actions")
        approval_id = actions_block["elements"][0]["value"]
        assert approval_id

        # Slack's real block_actions payload echoes the full original
        # message (ts + blocks) it was clicked on -- exactly what
        # `resolve_approval_blocks` needs to rewrite it in place.
        interactive_payload = json.dumps(
            {
                "type": "block_actions",
                "user": {"id": "U_MOCK_L2"},
                "channel": {"id": "C_COLLAPSE"},
                "container": {
                    "type": "message",
                    "message_ts": "500.777",
                    "channel_id": "C_COLLAPSE",
                },
                "message": {"ts": "500.777", "blocks": card_blocks, "text": _text},
                "actions": [{"action_id": "numi_reject", "value": approval_id}],
            }
        )
        interactive_response = client.post(
            "/webhooks/slack/interactive", data={"payload": interactive_payload}
        )
        assert interactive_response.status_code == 200

    assert len(updated) == 1
    channel, ts, _text, rebuilt_blocks = updated[0]
    assert channel == "C_COLLAPSE"
    assert ts == "500.777"
    # The actions block (both buttons) is gone entirely.
    assert not any(b["type"] == "actions" for b in rebuilt_blocks)
    # Replaced by a static, resolved line naming who decided and what.
    resolved_block = next(b for b in rebuilt_blocks if b["type"] == "context")
    resolved_text = resolved_block["elements"][0]["text"]
    assert "Rejected" in resolved_text
    assert "Dev DBA L2" in resolved_text
    # Every other block from the original card (the summary text, the
    # divider, the approval detail section) survives untouched.
    assert len(rebuilt_blocks) == len(card_blocks)


def test_slack_approval_card_buttons_stay_live_after_a_failed_decision(monkeypatch):
    """A card must NOT collapse when the click didn't actually resolve the
    approval -- found live: a separation-of-duties rejection ("the
    requester cannot approve their own critical action") still left the
    SAME card actionable by a different, eligible DBA, but the card
    collapsed to a false '✅ Approved by <requester>' anyway, since the
    first version of this fix collapsed on *which button was clicked*
    rather than on whether the decision actually closed the approval.
    `AgentReply.approval_still_pending=True` (set by
    `AgentOrchestrator.handle_approval_decision` for exactly this case, and
    for AWAITING_SECOND_APPROVAL) must suppress the collapse entirely."""
    settings = _settings()

    stub_agent = FastAPI()

    @stub_agent.post("/v1/chat")
    async def _chat(body: dict) -> dict:
        return {
            "text": "This requires approval.",
            "status": "approval_required",
            "approval_card": {
                "approval_id": "appr-sod-1",
                "tool_id": "restart_instance",
                "target_summary": "postgres-local",
                "reason": "DBA requested restart",
                "risk_level": "CRITICAL",
                "blast_radius": "SINGLE_DATABASE",
            },
        }

    @stub_agent.post("/v1/chat/events")
    async def _events(body: dict) -> dict:
        return {
            "text": "Approval failed: The requester cannot approve their own critical action.",
            "status": "error",
            "approval_still_pending": True,
        }

    channels_app = create_channels_app(settings, agent_transport=httpx.ASGITransport(app=stub_agent))

    posted: list[tuple[str, str, list[dict]]] = []
    updated: list[tuple[str, str, str, list[dict]]] = []

    async def _capture_post_message(self, channel, text, blocks):
        posted.append((channel, text, blocks))

    async def _capture_update_message(self, channel, ts, text, blocks):
        updated.append((channel, ts, text, blocks))

    monkeypatch.setattr(SlackMessageSender, "post_message", _capture_post_message)
    monkeypatch.setattr(SlackMessageSender, "update_message", _capture_update_message)

    plain_message_body = json.dumps(
        {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U_MOCK_L2",
                "text": "restart the postgres-local instance",
                "channel": "C_SOD",
                "ts": "600.001",
            },
        }
    ).encode()
    request_ts = str(int(time.time()))
    sig = _sign_slack(settings.slack_signing_secret, plain_message_body, request_ts)

    with TestClient(channels_app) as client:
        response = client.post(
            "/webhooks/slack",
            content=plain_message_body,
            headers={"X-Slack-Request-Timestamp": request_ts, "X-Slack-Signature": sig},
        )
        assert response.status_code == 200
        _channel, _text, card_blocks = posted[0]

        # The same requester clicking Approve on their own card.
        interactive_payload = json.dumps(
            {
                "type": "block_actions",
                "user": {"id": "U_MOCK_L2"},
                "channel": {"id": "C_SOD"},
                "container": {"type": "message", "message_ts": "600.777", "channel_id": "C_SOD"},
                "message": {"ts": "600.777", "blocks": card_blocks, "text": _text},
                "actions": [{"action_id": "numi_approve", "value": "appr-sod-1"}],
            }
        )
        interactive_response = client.post(
            "/webhooks/slack/interactive", data={"payload": interactive_payload}
        )
        assert interactive_response.status_code == 200

    # The failure was still posted as a normal message...
    assert len(posted) == 2
    assert "cannot approve their own" in posted[1][1]
    # ...but the card's own buttons were never touched.
    assert updated == []
