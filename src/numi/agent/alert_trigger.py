"""Alert-triggered investigation (spec §7, §42 — the event-driven sibling of
`agent.scheduled_report`'s clock-driven digest).

An external monitoring system POSTs a threshold-breach alert to Channels'
`/webhooks/alerts`; Channels verifies it (`channels.alerts.signature`) and
forwards it to the Agent's `POST /v1/alerts/trigger`, which is this
module's one entry point (`AlertTriggerRunner.handle_alert`). From there
the shape is identical to the daily digest, and deliberately so:

**Nothing new touches a database.** Every call this path makes goes through
the same `ToolClient` -> Gateway -> Execution Service pipeline as a live
DBA's message, authorized as a real configured DBA identity
(`Settings.alert_webhook_identity_account`) that the Gateway independently
re-resolves per call (spec §62). There is no webhook bypass, no elevated
service role, and no second route to a database.

**Structurally read-only.** `orchestrator.run_triggered_investigation` sets
`read_only=True` on the investigation, enforced at the same three points
`ARCHITECTURE.md`'s "Scheduled daily digest: proactive, and structurally
read-only" section documents for the digest. An alert can trigger analysis
and a recommendation; it can never trigger a write, with or without
approval — an unattended path with nobody present to click "approve" must
never be the one place approval quietly stops being required.

**Delivery stays in Channels.** Reuses `scheduled_report.ChannelsDigestPublisher`
as-is: it already does exactly one thing, "ask Channels to deliver this
text to this channel_id," which is precisely what a finished alert
investigation needs too. No second delivery path, no new channel credential
anywhere near the Agent.

**A cooldown, not just alert_id dedup.** `channels.api.app`'s `alert_id`
dedup catches a literal retried delivery of the *same* firing; it does
nothing for a flapping metric that genuinely re-breaches its threshold
every few minutes, which would otherwise run a full LLM-driven
investigation — and post a fresh message — on every single occurrence.
`AlertTriggerRunner` also enforces a per-`(server, metric)` cooldown
(`Settings.alert_webhook_cooldown_seconds`) using the same
`RateLimitBackend` the Gateway's rate limiter runs on (`limit=1` over the
cooldown window is exactly what a cooldown is), so it is Redis-backed and
correctly shared across replicas the moment `RATE_LIMIT_BACKEND=redis` is
set — an in-memory-only cooldown would silently reset per replica, letting
a load-balanced deployment re-investigate on every request regardless.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from numi.agent.orchestrator import AgentOrchestrator, ScheduledSummary
from numi.agent.scheduled_report import DigestPublisher
from numi.common.config import Settings
from numi.common.observability import get_logger
from numi.common.rate_limit_backend import (
    InMemoryRateLimitBackend,
    RateLimitBackend,
    RedisRateLimitBackend,
)
from numi.common.server_reference import normalize_server_reference

logger = get_logger(__name__)


def _build_cooldown_backend(settings: Settings) -> RateLimitBackend:
    """Mirrors `gateway.api.state._build_rate_limit_backend` exactly — same
    setting, same reasoning: reuse `RATE_LIMIT_BACKEND` rather than a second
    toggle an operator would have to remember exists and keep in sync."""
    if settings.rate_limit_backend == "redis":
        import redis.asyncio as redis

        return RedisRateLimitBackend(redis.from_url(settings.redis_url))
    return InMemoryRateLimitBackend()


@dataclasses.dataclass(frozen=True)
class AlertPayload:
    """The alert a monitoring system reported, already validated by the
    transport layer (`agent.api.app.AlertTriggerRequest`) — this module
    never sees the raw HTTP body."""

    server: str
    metric: str = ""
    current_value: str = ""
    threshold: str = ""
    severity: str = ""
    source: str = ""
    message: str = ""


@dataclasses.dataclass(frozen=True)
class AlertTriggerOutcome:
    """What handling one alert produced — enough for the API layer to
    choose an HTTP status without re-deriving any of this module's logic
    (see `agent.api.app`'s `/v1/alerts/trigger`)."""

    ok: bool
    text: str = ""
    error: str = ""
    # True only when a per-(server, metric) cooldown suppressed a real
    # investigation — distinct from `error`, which means something went
    # wrong. A cooldown hit is the feature working as intended.
    suppressed: bool = False


def resolve_server(server_ref: str, servers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match an alert's `server` field against the registered id/aliases a
    monitoring system was configured with — the same normalization
    (`common.server_reference`) every other server-hint resolution in this
    codebase uses, for the same "SQL Server Dev 1" vs "sqlserver-dev-01"
    reason.

    Exact-match only (id/alias equality after normalizing), deliberately
    stricter than `AgentOrchestrator._environment_for_instance`'s
    substring matching: that convenience only ever *skips* an optional
    clarifying question when it's ambiguous, so a false-positive substring
    match costs nothing (the Gateway still independently validates the
    real target). Here, a wrong match is the difference between
    investigating the server that actually breached its threshold and a
    different one entirely, with no DBA present to notice or correct it
    before it's already posted — so an ambiguous or absent match returns
    None and is reported as "unknown server," never guessed."""
    needle = normalize_server_reference(server_ref)
    matches = []
    for s in servers:
        names = {str(s.get("id", "")), *(str(a) for a in (s.get("aliases") or []))}
        if any(normalize_server_reference(n) == needle for n in names if n):
            matches.append(s)
    return matches[0] if len(matches) == 1 else None


def build_problem_statement(alert: AlertPayload) -> str:
    """The investigation's opening problem statement — layer zero of the
    read-only guarantee, exactly as `orchestrator._SCHEDULED_SUMMARY_PROBLEM`
    is for the digest: the only layer that shapes what the model *wants* to
    do rather than blocking what it tried to do, so it says outright that
    nothing proposed here is ever executed and nobody is present to ask a
    clarifying question of."""
    details = []
    if alert.metric:
        detail = f"metric `{alert.metric}`"
        if alert.current_value:
            detail += f" is currently {alert.current_value}"
        if alert.threshold:
            detail += f" (threshold: {alert.threshold})"
        details.append(detail)
    if alert.severity:
        details.append(f"severity: {alert.severity}")
    if alert.source:
        details.append(f"reported by: {alert.source}")
    if alert.message:
        details.append(f"alert message: {alert.message}")
    detail_text = "; ".join(details) if details else "no further detail was provided by the alert"

    return (
        f"A monitoring alert fired for the {{server_id}} server — {detail_text}. "
        "This is an unattended, proactive investigation with no DBA in the "
        "conversation to answer a question or approve anything. Investigate this "
        "specific alert and report findings and a recommendation as text only: "
        "only read-only diagnostics are available to you here, and nothing you "
        "propose will be executed, approved, or acted on automatically. If this "
        "needs remediation, describe what you would recommend and why, for a DBA "
        "to decide on — never as an action you are taking. Do not ask a "
        "clarifying question; nobody is there to answer it."
    )


def _format_alert_result(alert: AlertPayload, summary: ScheduledSummary) -> str:
    header = f"Numi alert-triggered investigation — {summary.server_id} ({summary.environment})"
    trigger_line = f"Triggered by: {alert.metric or 'monitoring alert'}"
    if alert.source:
        trigger_line += f" via {alert.source}"

    if not summary.ok:
        return (
            f"{header}\n{trigger_line}\n\n"
            f"Could not complete the investigation: {summary.error or 'unknown error'}."
        )

    lines = [header, trigger_line, "", summary.text]
    if summary.dropped_proposals:
        proposed = ", ".join(sorted({t for t in summary.dropped_proposals if t}))
        lines.append(
            f"\n(Numi would have proposed {proposed} here. An alert-triggered "
            "investigation never executes an action — treat this as a "
            "recommendation to act on, not something that was done.)"
        )
    return "\n".join(lines)


class AlertTriggerRunner:
    """One alert's worth of work: resolve the server, run the investigation,
    post the result — the event-driven counterpart to
    `scheduled_report.DailyDigestRunner`, and built the same way on purpose:
    the entire behavior is reachable and assertable by calling
    `handle_alert()` directly, no HTTP or webhook signature involved, so
    tests exercise the actual investigation/read-only logic instead of
    re-mocking the transport layer for every case."""

    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        settings: Settings,
        publisher: DigestPublisher,
        cooldown_backend: RateLimitBackend | None = None,
    ):
        self._orchestrator = orchestrator
        self._settings = settings
        self._publisher = publisher
        self._cooldown_backend = cooldown_backend or _build_cooldown_backend(settings)

    async def handle_alert(self, alert: AlertPayload) -> AlertTriggerOutcome:
        channel_id = self._settings.alert_webhook_slack_channel
        if not channel_id:
            # Belt and braces: the webhook route in `channels.api.app`
            # already refuses every request when `alert_webhook_secret` is
            # unset, so this is unreachable via a real webhook call. It is
            # here for a direct/manual call, and so this feature's "on"
            # state is never split across two config values that could
            # drift out of sync (see `Settings.alert_webhook_secret`'s own
            # comment).
            logger.info("alert_trigger_skipped_no_channel")
            return AlertTriggerOutcome(ok=False, error="Alert-triggered investigation is not configured.")

        try:
            servers = await self._orchestrator.list_registered_servers()
        except Exception as exc:  # noqa: BLE001 — see module docstring: report, don't raise.
            logger.warning(
                "alert_trigger_server_registry_unavailable",
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            return AlertTriggerOutcome(ok=False, error="The server registry was unreachable.")

        server = resolve_server(alert.server, servers)
        if server is None:
            logger.warning("alert_trigger_unknown_server", server_ref=alert.server)
            return AlertTriggerOutcome(
                ok=False,
                error=f"'{alert.server}' does not match exactly one registered server id or alias.",
            )

        server_id = str(server["id"])
        environment = str(server.get("environment", ""))

        cooldown_seconds = self._settings.alert_webhook_cooldown_seconds
        if cooldown_seconds > 0:
            cooldown_key = f"alert-cooldown:{server_id}:{alert.metric or '_'}"
            within_budget = await self._cooldown_backend.increment_and_check(
                cooldown_key, limit=1, window_seconds=cooldown_seconds
            )
            if not within_budget:
                logger.info(
                    "alert_trigger_cooldown_active",
                    server_id=server_id,
                    metric=alert.metric,
                    cooldown_seconds=cooldown_seconds,
                )
                return AlertTriggerOutcome(ok=True, suppressed=True)

        problem = build_problem_statement(alert).format(server_id=server_id)

        logger.info("alert_trigger_starting", server_id=server_id, metric=alert.metric)
        summary = await self._orchestrator.run_triggered_investigation(
            server_id=server_id,
            environment=environment,
            channel=self._settings.alert_webhook_identity_channel,
            channel_account_id=self._settings.alert_webhook_identity_account,
            problem=problem,
        )
        text = _format_alert_result(alert, summary)

        try:
            await self._publisher.publish(channel_id=channel_id, text=text)
        except Exception as exc:  # noqa: BLE001 — see module docstring: report, don't raise.
            logger.warning(
                "alert_trigger_publish_failed",
                channel=channel_id,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            return AlertTriggerOutcome(
                ok=False, text=text, error="Investigated, but could not deliver the result."
            )

        return AlertTriggerOutcome(ok=True, text=text)
