"""Agent FastAPI app (spec §57).

`POST /v1/chat` is only ever called by a channel adapter that has *already*
verified the sender's enterprise identity and DBA-team membership — but the
Agent (and everything downstream) treats that as convenience, not as a
security boundary: the Gateway independently re-resolves identity from the
same `channel`/`channel_account_id` pair on every tool call regardless
(spec §62).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from numi.agent.alert_trigger import AlertPayload, AlertTriggerRunner
from numi.agent.context_manager import ContextManager
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.reply import AgentReply
from numi.agent.scheduled_report import (
    ChannelsDigestPublisher,
    DailyDigestRunner,
    schedule_daily_digest,
)
from numi.agent.tool_client import ToolClient
from numi.common.config import Settings, get_settings
from numi.common.observability import configure_logging, get_logger
from numi.common.service_auth import ServiceTokenIssuer, ServiceTokenVerifier

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    channel: str
    channel_account_id: str
    conversation_id: str
    channel_thread_id: str = ""
    message: str


class ApprovalEventRequest(BaseModel):
    channel: str
    channel_account_id: str
    conversation_id: str
    approval_id: str
    decision: str  # "approve" | "reject"


class AlertTriggerRequest(BaseModel):
    """What `channels.api.app`'s `/webhooks/alerts` forwards after verifying
    the external monitoring system's signature — see `agent.alert_trigger`."""

    server: str
    metric: str = ""
    current_value: str = ""
    threshold: str = ""
    severity: str = ""
    source: str = ""
    message: str = ""


class AlertTriggerResponse(BaseModel):
    ok: bool
    error: str = ""


def create_app(
    settings: Settings | None = None, *, gateway_transport=None, channels_transport=None
) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_production()
    configure_logging("agent", settings.log_level)

    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    verifier = ServiceTokenVerifier(settings.service_jwt_secret, settings.service_jwt_issuer)
    tool_client = ToolClient(settings.gateway_base_url, issuer, transport=gateway_transport)
    llm_registry = LLMRegistry(settings)
    context = ContextManager()
    orchestrator = AgentOrchestrator(llm_registry, tool_client, context)
    logger.info(
        "llm_configured",
        default_provider=settings.effective_default_llm()[0],
        configured=settings.configured_llm_providers(),
        selection_enabled=llm_registry.selection_enabled(),
    )

    # The scheduled daily digest (agent service only — it owns the
    # orchestrator and, unlike `execution`, holds no database credential;
    # unlike `channels`, it is not a webhook front door whose lifecycle is
    # driven by inbound traffic). Constructed unconditionally but *started*
    # only if configured: `schedule_daily_digest` returns None and registers
    # nothing when `daily_report_slack_channel` is unset, which is the
    # default — see `agent.scheduled_report` for why "no channel" is the one
    # and only off switch.
    digest_runner = DailyDigestRunner(
        orchestrator=orchestrator,
        settings=settings,
        publisher=ChannelsDigestPublisher(
            settings.channels_base_url, issuer, transport=channels_transport
        ),
    )

    # The alert-triggered investigation (see `agent.alert_trigger`) — same
    # "constructed unconditionally, only ever does anything if configured"
    # shape as the digest above, and for the same reason: `handle_alert`
    # itself refuses to run when `alert_webhook_slack_channel` is unset, so
    # there is nothing here to gate a second time.
    alert_trigger_runner = AlertTriggerRunner(
        orchestrator=orchestrator,
        settings=settings,
        publisher=ChannelsDigestPublisher(
            settings.channels_base_url, issuer, transport=channels_transport
        ),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Started here rather than at import time so a test importing this
        # module (or `create_app` being called to inspect routes) never
        # spins up a background job — and so the scheduler's event loop is
        # the one actually serving requests.
        scheduler = schedule_daily_digest(digest_runner, settings)
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown(wait=False)

    app = FastAPI(title="Numi AI DBA Agent", version="0.1.0", lifespan=lifespan)

    async def require_channel_service_token(x_service_token: str | None = Header(default=None)) -> None:
        if not x_service_token:
            raise HTTPException(status_code=401, detail="Missing service token.")
        try:
            verifier.verify(x_service_token, expected_audience="numi-agent")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=401, detail="Invalid service token.") from exc

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat", response_model=AgentReply, dependencies=[Depends(require_channel_service_token)])
    async def chat(body: ChatRequest) -> AgentReply:
        logger.info("chat_received", channel=body.channel, conversation_id=body.conversation_id)
        return await orchestrator.handle_message(
            channel=body.channel,
            channel_account_id=body.channel_account_id,
            conversation_id=body.conversation_id,
            channel_thread_id=body.channel_thread_id,
            message=body.message,
        )

    @app.post(
        "/v1/chat/events", response_model=AgentReply, dependencies=[Depends(require_channel_service_token)]
    )
    async def chat_events(body: ApprovalEventRequest) -> AgentReply:
        logger.info(
            "approval_event_received", conversation_id=body.conversation_id, decision=body.decision
        )
        return await orchestrator.handle_approval_decision(
            conversation_id=body.conversation_id,
            decision=body.decision,
            channel=body.channel,
            channel_account_id=body.channel_account_id,
        )

    @app.post(
        "/v1/alerts/trigger",
        response_model=AlertTriggerResponse,
        dependencies=[Depends(require_channel_service_token)],
    )
    async def alerts_trigger(body: AlertTriggerRequest) -> AlertTriggerResponse:
        logger.info("alert_trigger_received", server=body.server, metric=body.metric)
        outcome = await alert_trigger_runner.handle_alert(
            AlertPayload(
                server=body.server,
                metric=body.metric,
                current_value=body.current_value,
                threshold=body.threshold,
                severity=body.severity,
                source=body.source,
                message=body.message,
            )
        )
        return AlertTriggerResponse(ok=outcome.ok, error=outcome.error)

    return app


app = create_app()
