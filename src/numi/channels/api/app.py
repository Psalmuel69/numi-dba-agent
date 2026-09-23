"""Channels service FastAPI app (spec §4, §57, §66).

Hosts the Slack and Microsoft Teams webhook adapters plus a `/dev/chat`
mock channel for local development/testing without real Slack/Teams
credentials. Every path here does the same five things before anything
reaches the Agent:

  1. verify webhook/signature (or bot token)
  2. identify the raw channel account
  3. resolve enterprise identity (UX-level check via IdentityProvider)
  4. verify DBA membership (UX-level check — Gateway re-verifies for real)
  5. forward to the Agent's `/v1/chat`, then render the reply back

Channel adapters never execute a database operation and never talk to the
Gateway or Execution Service directly (spec §4).
"""

from __future__ import annotations

import json
import time

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from numi.agent.reply import AgentReply
from numi.channels.alerts.signature import AlertSignatureError, verify_alert_signature
from numi.channels.slack.blocks import render_reply_blocks, resolve_approval_blocks
from numi.channels.slack.sender import SlackMessageSender
from numi.channels.slack.signature import SlackSignatureError, verify_slack_signature
from numi.channels.teams.auth import BotFrameworkJWTVerifier, DevTeamsAuthVerifier, TeamsAuthError
from numi.channels.teams.cards import render_reply_card
from numi.channels.teams.sender import TeamsMessageSender
from numi.common.config import Settings, get_settings
from numi.common.identity import build_identity_provider
from numi.common.observability import configure_logging, get_logger
from numi.common.service_auth import ServiceTokenIssuer, ServiceTokenVerifier

logger = get_logger(__name__)


class NotifyRequest(BaseModel):
    """An unsolicited message the Agent asks this service to deliver — today
    only the scheduled daily health digest (`agent.scheduled_report`).

    Note what is deliberately absent: no identity, no approval_id, no
    conversation. This endpoint delivers text to a channel and does nothing
    else. It cannot start an investigation, cannot render an approval card
    (`render_reply_blocks` only emits one for a reply carrying an
    `approval_card`, which an `AgentReply` built here never has), and so
    cannot be turned into a way to get an action in front of a DBA for a
    click. That matters more than it looks: this is the one inbound path on
    this service whose content originates from an LLM-backed process rather
    than from a human, so its blast radius is kept to "posts text"."""

    channel_id: str
    text: str


class DevChatRequest(BaseModel):
    user: str  # the mock directory's `channel_accounts.dev` value, e.g. "dba_l2@example.com"
    message: str
    conversation_id: str = "dev-conversation"


def _slack_conversation_id(channel: str, slack_user_id: str, thread_ts: str) -> str:
    """The Agent-side `conversation_id` this Slack event belongs to.

    A threaded reply (`thread_ts` non-empty) always scopes to that thread,
    unchanged — deliberately shared by whoever replies in it, since a
    Slack thread is inherently one collaborative investigation.

    An ordinary, non-threaded message (`thread_ts` empty) used to fall
    back to `event["ts"]` — that message's own timestamp, unique to it
    alone and never matched by any later message. That silently gave
    *every* plain follow-up typed into the channel its own brand-new,
    empty `ConversationState` — verified live: a real DBA's entirely
    ordinary "and what about replication?", typed straight into a channel
    that (by design — see the webhook's own no-bot-id/message-type check)
    responds to plain messages with no @-mention required, arrived with no
    `thread_ts` of its own (it was never posted as a threaded reply, just
    the next line in the channel) and got a fresh conversation with none
    of `state.database_context`'s already-established environment/
    instance in it — and a follow-up bare "development" answer to the
    resulting misfire got classified as chitchat instead of resuming,
    because there was no `state.investigation` left to resume either.
    Every symptom `agent.orchestrator`'s own environment-persistence logic
    (see ARCHITECTURE.md's "A known environment must survive across
    investigations") was built to prevent still happened, because the
    conversation itself was silently a different one every single time —
    the orchestrator's own state-persistence logic was never actually
    reached with the same `ConversationState` twice. Confirmed by direct
    trace of `agent.orchestrator.handle_message`/`context_manager.
    ContextManager.get_or_create`: neither has any bug reachable when
    `conversation_id` is actually held constant across turns — see
    `tests/unit/test_environment_clarification.py`'s
    `test_a_later_fresh_investigation_never_reasks_for_an_already_known_environment`,
    which exercises the exact same zero-named-entity follow-up sentence
    shape and passes.

    Fixed by scoping a non-threaded message to `(channel, user)` instead
    of `(channel, message)`: every plain message the same DBA sends in
    this channel, outside of an explicit thread, is now one continuing
    conversation — exactly what `state.database_context` and the
    fresh-vs-resume `investigation.is_concluded` gate in
    `orchestrator.handle_message` already exist to manage over time —
    while two different DBAs typing in the same channel still get
    independent conversations, never crossed."""
    return f"slack:{channel}:{thread_ts or slack_user_id}"


def create_app(settings: Settings | None = None, *, agent_transport=None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_production()
    configure_logging("channels", settings.log_level)

    identity_provider = build_identity_provider(settings)
    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    # Inbound, for the one endpoint another of our services calls (`/v1/notify`
    # — see `NotifyRequest`). Every other path into this service is a webhook
    # authenticated by its own transport's scheme (Slack signature, Bot
    # Framework JWT); this one is a same-trust-domain hop and uses the same
    # signed, audience-scoped service token as Agent -> Gateway and
    # Gateway -> Execution, never a bare header (spec §31).
    verifier = ServiceTokenVerifier(settings.service_jwt_secret, settings.service_jwt_issuer)
    slack_sender = SlackMessageSender(settings.slack_bot_token)

    # Slack's Events API retries a delivery it hasn't gotten a fast ack for
    # (its own ~3s timeout) — reusing the SAME event_id on each retry, up to
    # a few times. This whole handler awaits the full Agent round-trip
    # (potentially many seconds under LLM load) before ever returning, so a
    # slow investigation reliably triggers at least one retry — verified
    # live: a real DBA's single message produced two different, garbled
    # replies, and a later reply was processed once as a real answer and
    # again, independently, as if it were a brand-new message entirely. An
    # in-memory, TTL'd set of recently-seen event ids makes a retried
    # delivery a no-op instead of a second, independent run of the whole
    # handler against a shared, mutable conversation. Per-process only
    # (matches this dev deployment's other in-memory state, e.g. rate
    # limiting) — a multi-instance deployment would share this via Redis
    # instead, same as that.
    _seen_slack_event_ids: dict[str, float] = {}
    _SEEN_EVENT_TTL_SECONDS = 300.0  # comfortably longer than Slack's own retry window

    # Same retry-deduplication shape as the Slack event_id cache just above,
    # for the same reason: a monitoring system that doesn't get a fast ack
    # commonly retries, and the field it uses for that ("alert_id" here —
    # every vendor names it differently, e.g. Alertmanager's `fingerprint`)
    # is optional, so a sender that omits it simply gets no dedup rather
    # than a rejected request. Kept as its own cache rather than reusing
    # Slack's: the two features are independent (see
    # `Settings.alert_webhook_secret`'s comment), and sharing one dict would
    # couple their eviction and, worse, their key spaces if a Slack
    # event_id and an alert_id ever collided.
    _seen_alert_ids: dict[str, float] = {}

    def _alert_already_seen(alert_id: str | None) -> bool:
        if not alert_id:
            return False
        now = time.monotonic()
        for expired_id in [
            aid for aid, seen_at in _seen_alert_ids.items() if now - seen_at > _SEEN_EVENT_TTL_SECONDS
        ]:
            del _seen_alert_ids[expired_id]
        if alert_id in _seen_alert_ids:
            return True
        _seen_alert_ids[alert_id] = now
        return False

    def _slack_event_already_seen(event_id: str | None) -> bool:
        if not event_id:
            return False
        now = time.monotonic()
        for expired_id in [
            eid for eid, seen_at in _seen_slack_event_ids.items() if now - seen_at > _SEEN_EVENT_TTL_SECONDS
        ]:
            del _seen_slack_event_ids[expired_id]
        if event_id in _seen_slack_event_ids:
            return True
        _seen_slack_event_ids[event_id] = now
        return False

    async def _no_connector_token() -> str | None:
        return None

    teams_sender = TeamsMessageSender(_no_connector_token)
    teams_verifier = (
        BotFrameworkJWTVerifier(settings.teams_app_id)
        if settings.is_production()
        else DevTeamsAuthVerifier(settings.teams_app_password)
    )

    app = FastAPI(title="Numi Channel Adapters", version="0.1.0")

    async def _call_agent_chat(
        *, channel: str, channel_account_id: str, conversation_id: str, channel_thread_id: str, message: str
    ) -> dict:
        token = issuer.issue(service_name="channels", audience="numi-agent")
        async with httpx.AsyncClient(
            base_url=settings.agent_base_url,
            transport=agent_transport,
            # A real LLM-backed investigation is several reasoning turns,
            # each with its own model call plus a Gateway round-trip — easily
            # a couple of minutes, unlike the near-instant deterministic mock
            # planner. Matches the Gateway's own 300s execution/discovery
            # timeouts rather than risking a client-side cutoff mid-thought.
            timeout=300,
        ) as client:
            response = await client.post(
                "/v1/chat",
                headers={"X-Service-Token": token},
                json={
                    "channel": channel,
                    "channel_account_id": channel_account_id,
                    "conversation_id": conversation_id,
                    "channel_thread_id": channel_thread_id,
                    "message": message,
                },
            )
            response.raise_for_status()
            return response.json()

    async def _call_agent_event(
        *, channel: str, channel_account_id: str, conversation_id: str, approval_id: str, decision: str
    ) -> dict:
        token = issuer.issue(service_name="channels", audience="numi-agent")
        async with httpx.AsyncClient(
            base_url=settings.agent_base_url,
            transport=agent_transport,
            timeout=300,  # see _call_agent_chat — same real-LLM latency reasoning
        ) as client:
            response = await client.post(
                "/v1/chat/events",
                headers={"X-Service-Token": token},
                json={
                    "channel": channel,
                    "channel_account_id": channel_account_id,
                    "conversation_id": conversation_id,
                    "approval_id": approval_id,
                    "decision": decision,
                },
            )
            response.raise_for_status()
            return response.json()

    async def _call_agent_alert_trigger(
        *,
        server: str,
        metric: str,
        current_value: str,
        threshold: str,
        severity: str,
        source: str,
        message: str,
    ) -> dict:
        token = issuer.issue(service_name="channels", audience="numi-agent")
        async with httpx.AsyncClient(
            base_url=settings.agent_base_url,
            transport=agent_transport,
            timeout=300,  # see _call_agent_chat — same real-LLM latency reasoning
        ) as client:
            response = await client.post(
                "/v1/alerts/trigger",
                headers={"X-Service-Token": token},
                json={
                    "server": server,
                    "metric": metric,
                    "current_value": current_value,
                    "threshold": threshold,
                    "severity": severity,
                    "source": source,
                    "message": message,
                },
            )
            response.raise_for_status()
            return response.json()

    async def require_agent_service_token(x_service_token: str | None = Header(default=None)) -> None:
        if not x_service_token:
            raise HTTPException(status_code=401, detail="Missing service token.")
        try:
            verifier.verify(x_service_token, expected_audience="numi-channels")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=401, detail="Invalid service token.") from exc

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # ------------------------------------------------ Outbound (Agent-initiated)

    @app.post("/v1/notify", dependencies=[Depends(require_agent_service_token)])
    async def notify(body: NotifyRequest) -> dict:
        """Deliver an unsolicited message to a channel on the Agent's behalf
        — the scheduled daily digest's one delivery route.

        Delivery lives here rather than in the Agent for the same reason
        every other outbound Slack call does: this is the only service that
        holds a channel credential, and the only one that knows how a
        message should be rendered for each channel (see
        `channels.slack.blocks`, and ARCHITECTURE.md's "Adding a channel").
        A Teams destination for the same digest is a change to this
        function, not to the Agent."""
        logger.info("notify_received", channel=body.channel_id, length=len(body.text))
        reply = AgentReply(text=body.text)
        await slack_sender.post_message(body.channel_id, reply.text, render_reply_blocks(reply))
        return {"ok": True}

    # ------------------------------------------------------------------ Slack

    @app.post("/webhooks/slack")
    async def slack_webhook(
        request: Request,
        x_slack_request_timestamp: str | None = Header(default=None),
        x_slack_signature: str | None = Header(default=None),
    ) -> dict:
        raw_body = await request.body()
        try:
            verify_slack_signature(
                signing_secret=settings.slack_signing_secret,
                request_body=raw_body,
                timestamp_header=x_slack_request_timestamp,
                signature_header=x_slack_signature,
            )
        except SlackSignatureError as exc:
            logger.warning("slack_signature_rejected", detail=str(exc))
            raise HTTPException(status_code=401, detail="Invalid Slack signature.") from exc

        payload = json.loads(raw_body or b"{}")

        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge", "")}

        if payload.get("type") == "event_callback":
            if _slack_event_already_seen(payload.get("event_id")):
                logger.info(
                    "slack_event_retry_deduplicated",
                    event_id=payload.get("event_id"),
                    retry_num=request.headers.get("X-Slack-Retry-Num"),
                )
                return {"ok": True}

            event = payload.get("event", {})
            # `event.get("bot_id")` alone doesn't cover every message this
            # service must ignore rather than act on -- found live (a real
            # 500, unhandled `KeyError`, crashing the whole webhook):
            # message_changed (edits), message_deleted, and other
            # `message`-typed subtypes carry no top-level "user" at all
            # (an edit's author lives nested under event["message"]["user"]
            # instead). None of these are a DBA sending Numi a fresh
            # instruction, so skip anything without a top-level "user" the
            # same way a bot message is already skipped, rather than
            # assuming the key exists.
            slack_user_id = event.get("user")
            if event.get("type") != "message" or event.get("bot_id") or not slack_user_id:
                return {"ok": True}
            identity = await identity_provider.resolve_by_external_account("slack", slack_user_id)
            if identity is None or not identity.is_dba():
                await slack_sender.post_message(
                    event["channel"], "Sorry, this account isn't recognized as a DBA team member.", []
                )
                return {"ok": True}

            conversation_id = _slack_conversation_id(
                event.get("channel", ""), slack_user_id, event.get("thread_ts", "")
            )
            reply = await _call_agent_chat(
                channel="slack",
                channel_account_id=slack_user_id,
                conversation_id=conversation_id,
                channel_thread_id=event.get("thread_ts", event.get("ts", "")),
                message=event.get("text", ""),
            )

            agent_reply = AgentReply.model_validate(reply)
            await slack_sender.post_message(
                event["channel"], agent_reply.text, render_reply_blocks(agent_reply)
            )
            return {"ok": True}

        return {"ok": True}

    @app.post("/webhooks/slack/interactive")
    async def slack_interactive(request: Request) -> dict:
        form = await request.form()
        raw_payload = form.get("payload", "{}")
        if not isinstance(raw_payload, str):
            raise HTTPException(status_code=400, detail="Malformed interactive payload.")
        payload = json.loads(raw_payload)

        user_id = payload.get("user", {}).get("id", "")
        actions = payload.get("actions", [])
        if not actions:
            return {"ok": True}
        action = actions[0]
        approval_id = action.get("value")
        decision = "approve" if action.get("action_id") == "numi_approve" else "reject"

        identity = await identity_provider.resolve_by_external_account("slack", user_id)
        if identity is None or not identity.is_dba():
            return {"text": "This account isn't recognized as a DBA team member."}

        channel_id = payload.get("channel", {}).get("id", "")
        # NOT the card's own container.message_ts -- that's unique to this
        # one card message and matched by nothing else the conversation_id
        # is ever computed from (see _slack_conversation_id's docstring and
        # ARCHITECTURE.md's "A stable conversation_id is a channels-layer
        # responsibility, not the Agent's"). An approval card lives in the
        # SAME conversation as the investigation that produced it, so this
        # must be computed exactly like the regular message path: from
        # (channel, clicking user, thread), via the same helper. A threaded
        # card carries its thread_ts on `container` or `message` (Slack
        # populates either depending on payload shape/version); a
        # non-threaded card carries neither, exactly like a non-threaded
        # regular message, and falls back to (channel, user) the same way.
        thread_ts = payload.get("container", {}).get("thread_ts") or payload.get(
            "message", {}
        ).get("thread_ts", "")
        conversation_id = _slack_conversation_id(channel_id, user_id, thread_ts)
        reply = await _call_agent_event(
            channel="slack",
            channel_account_id=user_id,
            conversation_id=conversation_id,
            approval_id=approval_id,
            decision=decision,
        )

        agent_reply = AgentReply.model_validate(reply)
        await slack_sender.post_message(channel_id, agent_reply.text, render_reply_blocks(agent_reply))

        # Collapse the card's own Approve/Reject buttons in place so neither
        # is still clickable after this decision — but ONLY when the
        # approval is actually, finally closed (agent_reply.approval_still_
        # pending is False). Found live: a separation-of-duties rejection
        # ("the requester cannot approve their own critical action") still
        # leaves the SAME card actionable by a different, eligible DBA --
        # collapsing it to a false "✅ Approved by <requester>" on that
        # failure would both lie about the outcome and hide a still-live
        # approval from the one person who could actually act on it. Same
        # reasoning for AWAITING_SECOND_APPROVAL (dual-approval's first leg).
        # Best-effort otherwise: the approval decision itself is already
        # fully resolved above regardless of whether this cosmetic update
        # succeeds, so a failure here (e.g. the bot token lacks chat:write,
        # or Slack is briefly unavailable) is logged, not raised, and never
        # blocks the real decision reaching the DBA in the message just
        # posted.
        message = payload.get("message", {})
        message_ts = message.get("ts", "")
        original_blocks = message.get("blocks", [])
        if message_ts and original_blocks and approval_id and not agent_reply.approval_still_pending:
            try:
                await slack_sender.update_message(
                    channel_id,
                    message_ts,
                    message.get("text", ""),
                    resolve_approval_blocks(
                        original_blocks, approval_id, decision, identity.display_name
                    ),
                )
            except Exception:
                logger.warning("slack_approval_card_collapse_failed", approval_id=approval_id, exc_info=True)
        return {"ok": True}

    # ------------------------------------------------------------------ Teams

    @app.post("/webhooks/teams")
    async def teams_webhook(request: Request, authorization: str | None = Header(default=None)) -> dict:
        try:
            await teams_verifier.verify(authorization)
        except TeamsAuthError as exc:
            logger.warning("teams_auth_rejected", detail=str(exc))
            raise HTTPException(status_code=401, detail="Invalid Teams bot token.") from exc

        activity = await request.json()
        aad_object_id = activity.get("from", {}).get("aadObjectId") or activity.get("from", {}).get("id", "")

        identity = await identity_provider.resolve_by_external_account("teams", aad_object_id)
        if identity is None or not identity.is_dba():
            return {"type": "message", "text": "Sorry, this account isn't recognized as a DBA team member."}

        conversation_id = f"teams:{activity.get('conversation', {}).get('id', '')}"

        value = activity.get("value") or {}
        if value.get("numi_action") in ("approve", "reject"):
            reply = await _call_agent_event(
                channel="teams",
                channel_account_id=aad_object_id,
                conversation_id=conversation_id,
                approval_id=value.get("approval_id", ""),
                decision=value["numi_action"],
            )
        else:
            reply = await _call_agent_chat(
                channel="teams",
                channel_account_id=aad_object_id,
                conversation_id=conversation_id,
                channel_thread_id=activity.get("conversation", {}).get("id", ""),
                message=activity.get("text", ""),
            )

        agent_reply = AgentReply.model_validate(reply)
        card = render_reply_card(agent_reply)
        await teams_sender.send_reply_to_activity(
            activity.get("serviceUrl", ""),
            activity.get("conversation", {}).get("id", ""),
            activity.get("id", ""),
            card,
        )
        return {"type": "message", "text": agent_reply.text}

    # ------------------------------------------------------------------ Alerts

    @app.post("/webhooks/alerts")
    async def alerts_webhook(
        request: Request,
        x_numi_alert_timestamp: str | None = Header(default=None),
        x_numi_alert_signature: str | None = Header(default=None),
    ) -> dict:
        """An external monitoring system reporting a threshold breach — see
        `agent.alert_trigger` for what happens after this hands off to the
        Agent. Signature-verified the same way `/webhooks/slack` is
        (`verify_slack_signature` there, `verify_alert_signature` here):
        rejected outright, before the body is even parsed as JSON, if it
        doesn't verify.

        `alert_webhook_secret` unset is a hard failure here, not a silent
        allow — `verify_alert_signature` itself refuses an empty secret
        rather than this route deciding whether to call it, so there is
        exactly one place either signature route can end up trusting an
        unconfigured webhook: nowhere."""
        raw_body = await request.body()
        try:
            verify_alert_signature(
                secret=settings.alert_webhook_secret,
                request_body=raw_body,
                timestamp_header=x_numi_alert_timestamp,
                signature_header=x_numi_alert_signature,
            )
        except AlertSignatureError as exc:
            logger.warning("alert_signature_rejected", detail=str(exc))
            raise HTTPException(status_code=401, detail="Invalid alert signature.") from exc

        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Malformed alert payload.") from exc

        if _alert_already_seen(payload.get("alert_id")):
            logger.info("alert_retry_deduplicated", alert_id=payload.get("alert_id"))
            return {"ok": True}

        server = payload.get("server", "")
        if not server:
            raise HTTPException(status_code=400, detail="'server' is required.")

        result = await _call_agent_alert_trigger(
            server=server,
            metric=payload.get("metric", ""),
            current_value=str(payload.get("current_value", "")),
            threshold=str(payload.get("threshold", "")),
            severity=payload.get("severity", ""),
            source=payload.get("source", ""),
            message=payload.get("message", ""),
        )
        if not result.get("ok"):
            logger.warning("alert_trigger_failed", server=server, error=result.get("error"))
            # Still a 200: the request itself was valid and accepted — what
            # failed (an unresolvable server, the feature being
            # unconfigured) is reported in the body for observability, not
            # as a delivery failure the sender's own retry/backoff logic
            # should react to (retrying "unknown server" changes nothing).
            return {"ok": False, "error": result.get("error", "")}
        return {"ok": True}

    # ------------------------------------------------------------------ Dev/mock channel (spec §66)

    @app.post("/dev/chat")
    async def dev_chat(body: DevChatRequest) -> dict:
        """Mock channel for local development (spec §66) — still goes
        through real identity resolution + the real Agent/Gateway pipeline;
        only the transport (no real Slack/Teams servers) is mocked."""
        identity = await identity_provider.resolve_by_external_account("dev", body.user)
        if identity is None:
            raise HTTPException(status_code=401, detail="Unknown development user.")
        if not identity.is_dba():
            return {"text": "This account isn't recognized as a DBA team member.", "status": "denied"}

        reply = await _call_agent_chat(
            channel="dev",
            channel_account_id=body.user,
            conversation_id=body.conversation_id,
            channel_thread_id="",
            message=body.message,
        )
        return reply

    @app.post("/dev/chat/events")
    async def dev_chat_event(body: dict) -> dict:
        reply = await _call_agent_event(
            channel="dev",
            channel_account_id=body["user"],
            conversation_id=body.get("conversation_id", "dev-conversation"),
            approval_id=body["approval_id"],
            decision=body["decision"],
        )
        return reply

    return app


app = create_app()
