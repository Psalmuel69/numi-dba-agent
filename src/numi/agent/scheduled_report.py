"""The scheduled, multi-server morning digest (spec §7, §42).

This is the deferred half of `playbooks.library.comprehensive_summary`.
That playbook was always written as "the building block a future scheduled,
multi-server morning report would call once per server" — deliberately one
server per invocation, deliberately not scheduled — and its own docstring,
README.md's Playbooks paragraph, and ARCHITECTURE.md all said so. This
module is that scheduled orchestration: once a day, run
`comprehensive_summary` against every registered server and post ONE
combined digest to a configured channel.

**The constraint this module exists inside.** Proactive output is text and a
recommendation, never an action. A scheduled run investigates and reports;
it never approves anything, never submits a write, and never does anything a
DBA didn't ask for turn by turn. That is not enforced here — this module
only *requests* it, via `orchestrator.run_comprehensive_summary`, which
marks the investigation `read_only` and enforces it at three points inside
the investigation loop (see `InvestigationState.read_only` and
ARCHITECTURE.md's "Scheduled daily digest: proactive, and structurally
read-only"). Enforcement deliberately lives next to the code that submits
tool calls rather than here, because here is the wrong altitude for it: a
guarantee that only holds if the caller remembers to ask for it is not a
guarantee.

**Nothing new touches a database.** Every call this path makes goes through
the same `ToolClient` -> Gateway -> Execution Service pipeline as a live
DBA's message, authorized as a real configured DBA identity
(`Settings.daily_report_identity_account`) that the Gateway independently
re-resolves per call (spec §62). There is no scheduled-job bypass, no
elevated service role, and no second route to a database — a scheduled run
can see exactly what that DBA account could have seen by typing "daily
summary" into Slack, and nothing more.

**Opt-in, with one switch.** An unset `daily_report_slack_channel` means no
scheduler object is constructed, no job is registered, and no background
task exists — not a scheduler that wakes daily to find it has nowhere to
post. See `schedule_daily_digest`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from numi.agent.orchestrator import AgentOrchestrator, ScheduledSummary
from numi.common.config import Settings
from numi.common.observability import get_logger
from numi.common.service_auth import ServiceTokenIssuer

logger = get_logger(__name__)

# The scheduler's job id. Stable and explicit so `schedule_daily_digest`'s
# result is directly assertable ("is the job registered or not?") without a
# test ever having to wait for a real clock tick — see
# tests/unit/test_daily_digest_scheduling.py.
DAILY_DIGEST_JOB_ID = "numi_daily_digest"

# A sweep that somehow outran a whole day must never overlap the next one
# (max_instances), a digest missed because the Agent was restarting is worth
# posting up to an hour late but not at an arbitrary later hour
# (misfire_grace_time), and several missed occurrences must collapse into
# one post rather than a burst (coalesce). These are the three APScheduler
# knobs that actually matter for a once-daily report; everything else is
# left at its default on purpose.
_MAX_CONCURRENT_RUNS = 1
_MISFIRE_GRACE_SECONDS = 3600


class DigestPublisher(Protocol):
    """Where a finished digest goes. An interface, not a concrete Slack
    client, for one structural reason: the Agent process holds no channel
    credential and must not start holding one. `ChannelsDigestPublisher`
    below posts to the Channels service, which already owns every outbound
    Slack/Teams token and every piece of rendering (see
    `channels.slack.sender`, and ARCHITECTURE.md's "Adding a channel") — so
    a Teams or Discord destination later is a Channels-side change, not a
    change here. Tests substitute a recording double."""

    async def publish(self, *, channel_id: str, text: str) -> None: ...


class ChannelsDigestPublisher:
    """Posts a digest by asking the Channels service to deliver it
    (`POST /v1/notify`), with the same signed, audience-scoped service token
    every other hop between our own services uses
    (`common.service_auth`) — never a bare "trust me" header.

    This is a new arrow in the topology (Agent -> Channels), and it points
    the way it does deliberately. The alternative — giving the Agent process
    `SLACK_BOT_TOKEN` and letting it call `chat.postMessage` itself — would
    put a channel credential in the one service that runs
    attacker-influenceable model output, and would fork outbound Slack
    rendering into a second place that could drift from
    `channels.slack.blocks`. Keeping delivery in Channels costs one HTTP hop
    and keeps every channel credential in exactly one service."""

    def __init__(
        self,
        base_url: str,
        issuer: ServiceTokenIssuer,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url
        self._issuer = issuer
        self._transport = transport

    async def publish(self, *, channel_id: str, text: str) -> None:
        token = self._issuer.issue(service_name="agent", audience="numi-channels")
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=30
        ) as client:
            response = await client.post(
                "/v1/notify",
                headers={"X-Service-Token": token},
                json={"channel_id": channel_id, "text": text},
            )
            response.raise_for_status()


def select_servers(servers: Sequence[dict[str, Any]], configured_subset: str) -> list[dict[str, Any]]:
    """Which registered servers this morning's digest sweeps, from the same
    `/v1/catalog/servers` payload `orchestrator._list_servers_cached` already
    reads (id / environment / aliases / status / catalog) — no new Gateway
    endpoint and no second source of truth about what exists.

    Two filters, both deliberate. A server whose registry `status` isn't
    `active` is skipped silently: `gateway.domain.servers` already excludes
    those from target resolution, so sweeping one would only ever produce a
    denial reported as a failure, and a decommissioned server appearing
    every morning as "could not be checked" is noise that trains a DBA to
    ignore the section that matters. `configured_subset` (empty = every
    active server) then narrows by id or alias, matched case-insensitively,
    so an operator can name servers the same way they would in chat. Registry
    order is preserved — the digest reads in a stable order every day rather
    than shuffling with dictionary iteration."""
    active = [s for s in servers if (s.get("status") or "active") == "active"]
    wanted = {part.strip().lower() for part in configured_subset.split(",") if part.strip()}
    if not wanted:
        return active
    selected = []
    for server in active:
        names = {str(server.get("id", "")).lower()} | {
            str(alias).lower() for alias in (server.get("aliases") or [])
        }
        if names & wanted:
            selected.append(server)
    return selected


def _server_block(summary: ScheduledSummary) -> str:
    lines = [f"*{summary.server_id}* ({summary.environment})", summary.text]
    if summary.dropped_proposals:
        # Surfaced, never swallowed: the model wanting to act and being
        # structurally stopped is information a DBA should see. Worded so
        # it can't be misread as "Numi did something" — see
        # `orchestrator._submit_and_relay`'s read-only gate, which is what
        # actually stopped it.
        proposed = ", ".join(sorted({t for t in summary.dropped_proposals if t}))
        lines.append(
            f"(Numi would have proposed {proposed} here. A scheduled sweep never "
            "executes an action — treat this as a recommendation to act on, not "
            "something that was done.)"
        )
    return "\n".join(lines)


def build_digest(
    summaries: Sequence[ScheduledSummary], *, generated_at: dt.datetime
) -> str:
    """Assemble one combined, multi-server digest.

    Applies `comprehensive_summary`'s own "report ONLY deviations, then say
    plainly that everything else came back clean" discipline a second time,
    one level up: that guidance keeps a single server's report scannable,
    and without the same discipline across servers a ten-server estate would
    produce ten paragraphs of "X is fine" every morning — which is the same
    wall of text, just bigger. So a server that flagged nothing gets its
    name in one closing line, not a block of its own.

    "Flagged nothing" is decided structurally (`ScheduledSummary.is_clean`
    — the typed `Conclude` action named no root cause and recommended
    nothing), never by pattern-matching the model's prose for reassuring
    words. There is no honest way to parse "everything looks healthy" out of
    free text, and guessing wrong in the quiet direction is exactly the
    failure this whole section is meant to prevent.

    Failures are the part that must not be quiet. A server whose
    investigation never completed — unreachable, discovery never finished,
    every call denied, the LLM provider down — gets its own explicitly
    labelled section and is counted separately in the header. A DBA reading
    "6 servers checked, all healthy" when it was really "6 attempted, 2
    never responded" is worse off than if no digest had been sent at all,
    so the header's arithmetic always states both numbers, even when the
    failure count is zero."""
    stamp = generated_at.strftime("%Y-%m-%d %H:%M")
    header = f"Numi daily health digest — {stamp} UTC"
    if not summaries:
        return (
            f"{header}\n\nNo servers were swept: no active registered server matched "
            "this digest's configuration. Check `config/servers.yaml` and "
            "`NUMI_DAILY_REPORT_SERVERS`."
        )

    checked = [s for s in summaries if s.ok]
    failed = [s for s in summaries if not s.ok]
    flagged = [s for s in checked if not s.is_clean]
    clean = [s for s in checked if s.is_clean]

    sections = [
        header,
        f"{len(summaries)} servers swept: {len(checked)} checked, "
        f"{len(failed)} could not be checked.",
    ]

    if flagged:
        sections.append("*NEEDS ATTENTION*")
        sections.extend(_server_block(s) for s in flagged)

    if failed:
        # Second, not first, and never folded in with the findings above: a
        # failure is a gap in coverage rather than a finding about a
        # database, and conflating the two is how "couldn't check it" starts
        # reading like "checked it, it's fine".
        sections.append("*COULD NOT BE CHECKED*")
        sections.extend(
            f"*{s.server_id}* ({s.environment}) — {s.error or 'the investigation did not complete'}"
            for s in failed
        )

    if clean:
        names = ", ".join(s.server_id for s in clean)
        sections.append(f"All other checks came back clean: {names}.")
    elif not flagged and not failed:  # pragma: no cover - unreachable: checked == clean + flagged
        sections.append("All other checks came back clean.")

    return "\n\n".join(sections)


class DailyDigestRunner:
    """One morning's work: sweep the selected servers, assemble the digest,
    post it once.

    Separated from `schedule_daily_digest` below so the entire behavior of
    the feature is reachable and assertable by calling `run_once()`
    directly, with no clock involved — a test must never sleep for a day,
    and a scheduler that can only be exercised by waiting for it is a
    scheduler nobody tests."""

    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        settings: Settings,
        publisher: DigestPublisher,
    ):
        self._orchestrator = orchestrator
        self._settings = settings
        self._publisher = publisher

    async def _summarize(self, server: dict[str, Any]) -> ScheduledSummary:
        """One server, never raising. A sweep that aborts halfway because
        one server threw would silently drop every server after it in
        registry order — the digest would look complete and simply be
        missing servers, which is the exact failure mode `build_digest`'s
        "could not be checked" section exists to prevent. So anything the
        per-server run raises becomes a reported failure for that server and
        the sweep continues.

        The raw exception is logged in full and never reaches the digest
        text, matching the no-raw-error invariant the rest of this pipeline
        holds (ARCHITECTURE.md): an `httpx` error stringifies with the
        request URL and a documentation link, which is neither useful nor
        appropriate in a channel a whole DBA team reads."""
        server_id = str(server.get("id", ""))
        environment = str(server.get("environment", ""))
        try:
            return await self._orchestrator.run_comprehensive_summary(
                server_id=server_id,
                environment=environment,
                channel=self._settings.daily_report_identity_channel,
                channel_account_id=self._settings.daily_report_identity_account,
            )
        except Exception as exc:  # noqa: BLE001 — deliberately broad; see the docstring.
            logger.warning(
                "scheduled_summary_raised",
                server_id=server_id,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            return ScheduledSummary(
                server_id=server_id,
                environment=environment,
                investigation_id="",
                status="error",
                text="",
                error="the health sweep failed to run against this server",
            )

    async def run_once(self) -> str | None:
        """Build and post exactly one digest. Returns the digest text (for
        tests and for a manual invocation), or None when the feature is
        disabled.

        This is the function the scheduler calls, so it must never raise:
        an exception escaping into APScheduler's job runner kills nothing
        useful and just logs a traceback where nobody looks. Every failure
        is therefore either reported *inside* the digest (a per-server
        failure) or degrades to a digest that says what went wrong (the
        server registry being unreachable) — the one outcome deliberately
        not chosen anywhere here is silence, because a digest that simply
        doesn't arrive is indistinguishable from a quiet morning."""
        channel_id = self._settings.daily_report_slack_channel
        if not channel_id:
            # Belt and braces: `schedule_daily_digest` never registers a job
            # without a channel, so this is unreachable via the scheduler.
            # It is here for a direct/manual call.
            logger.info("daily_digest_skipped_no_channel")
            return None

        generated_at = dt.datetime.now(dt.UTC)
        try:
            servers = await self._orchestrator.list_registered_servers()
            selected = select_servers(servers, self._settings.daily_report_servers)
        except Exception as exc:  # noqa: BLE001 — see the docstring: never silence.
            logger.warning(
                "daily_digest_server_registry_unavailable",
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            digest = (
                f"Numi daily health digest — {generated_at.strftime('%Y-%m-%d %H:%M')} UTC"
                "\n\nNo servers could be swept: the server registry was unreachable, so "
                "nothing was checked this morning. This is a gap in coverage, not a "
                "clean bill of health."
            )
        else:
            logger.info(
                "daily_digest_starting", server_count=len(selected), channel=channel_id
            )
            summaries = [await self._summarize(server) for server in selected]
            digest = build_digest(summaries, generated_at=generated_at)
            logger.info(
                "daily_digest_built",
                server_count=len(summaries),
                failed=sum(1 for s in summaries if not s.ok),
            )

        try:
            await self._publisher.publish(channel_id=channel_id, text=digest)
        except Exception as exc:  # noqa: BLE001 — see the docstring.
            logger.warning(
                "daily_digest_publish_failed",
                channel=channel_id,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
        return digest


def schedule_daily_digest(
    runner: DailyDigestRunner, settings: Settings
) -> AsyncIOScheduler | None:
    """Register the once-daily job, or don't — returns the running scheduler,
    or None when the digest is disabled.

    Opt-in is the whole contract here: with no `daily_report_slack_channel`
    configured (the default), this constructs nothing, starts nothing, and
    registers nothing, so a deployment that never asked for a proactive
    digest carries no background machinery, no extra thread, and no daily
    wake-up at all. Returning None rather than an idle scheduler is what
    makes that directly assertable — see
    tests/unit/test_daily_digest_scheduling.py, which checks both
    directions without ever advancing a clock.

    APScheduler (rather than a hand-rolled `asyncio.sleep` loop) for one
    reason worth the dependency: the correctness of "every day at 06:00" is
    almost entirely in the edge cases — a sleep-until-next-hour loop has to
    get missed occurrences, overlapping runs, and drift right by hand, and
    each of those is a bug that only shows up in production at 6am. Those
    are `coalesce` / `max_instances` / `misfire_grace_time` here, and the
    trigger is pinned explicitly to UTC so the digest lands at the same
    real moment regardless of the host's timezone."""
    if not settings.daily_report_slack_channel:
        logger.info("daily_digest_disabled")
        return None

    scheduler = AsyncIOScheduler(timezone=dt.UTC)
    scheduler.add_job(
        runner.run_once,
        CronTrigger(hour=settings.daily_report_hour_utc, minute=0, timezone=dt.UTC),
        id=DAILY_DIGEST_JOB_ID,
        name="Numi daily multi-server health digest",
        coalesce=True,
        max_instances=_MAX_CONCURRENT_RUNS,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "daily_digest_scheduled",
        hour_utc=settings.daily_report_hour_utc,
        channel=settings.daily_report_slack_channel,
        servers=settings.daily_report_servers or "(all active)",
    )
    return scheduler
