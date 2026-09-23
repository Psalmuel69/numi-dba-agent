"""Multi-server digest assembly (`agent.scheduled_report`).

Tested by calling `build_digest` / `select_servers` / `DailyDigestRunner
.run_once` directly with fabricated per-server results — no clock, no
scheduler tick, no LLM, no Slack. "One digest run" is a pure enough unit to
test on its own, and a test that waited for a real 06:00 would be untestable
by construction (see test_daily_digest_scheduling.py for the scheduling half,
which likewise never advances a clock).

The property that matters most here is that a server whose investigation
*failed* is visibly reported rather than dropped. A digest saying "6 servers
checked, all healthy" when it was really "6 attempted, 2 never responded" is
worse than no digest at all: it actively tells a DBA to stop looking.
"""

from __future__ import annotations

import datetime as dt

import pytest

from numi.agent.orchestrator import ScheduledSummary
from numi.agent.scheduled_report import DailyDigestRunner, build_digest, select_servers
from numi.common.config import Settings

_AT = dt.datetime(2026, 3, 4, 6, 0, tzinfo=dt.UTC)


def _clean(server_id: str, environment: str = "production") -> ScheduledSummary:
    return ScheduledSummary(
        server_id=server_id,
        environment=environment,
        investigation_id=f"inv_{server_id}",
        status="ok",
        text="All checks came back clean.",
    )


def _flagged(server_id: str, environment: str = "production") -> ScheduledSummary:
    return ScheduledSummary(
        server_id=server_id,
        environment=environment,
        investigation_id=f"inv_{server_id}",
        status="ok",
        text="Backups: no successful full backup in 9 days. All other checks came back clean.",
        findings=("The full-backup chain is broken.",),
        recommendations=("Run a full backup and investigate the failing job.",),
    )


def _failed(server_id: str, environment: str = "uat") -> ScheduledSummary:
    return ScheduledSummary(
        server_id=server_id,
        environment=environment,
        investigation_id=f"inv_{server_id}",
        status="ok",
        text="Everything looks healthy.",  # the model's prose — deliberately reassuring
        error="no diagnostic call succeeded — the server may be unreachable",
    )


# ------------------------------------------------------------ build_digest ---


def test_a_failed_server_is_reported_not_silently_dropped():
    """The headline property. Three servers: one flagged, one clean, one
    that never responded. The failure must appear under its own heading,
    with its reason, and must not be counted as checked."""
    digest = build_digest(
        [_flagged("core-banking-prod"), _clean("analytics-prod"), _failed("mysql-uat-01")],
        generated_at=_AT,
    )

    assert "3 servers swept: 2 checked, 1 could not be checked." in digest
    assert "COULD NOT BE CHECKED" in digest
    assert "mysql-uat-01" in digest
    assert "may be unreachable" in digest
    # And its own reassuring prose must NOT be what gets printed for it —
    # that text is exactly the lie this section exists to prevent.
    assert "Everything looks healthy." not in digest


def test_a_failed_server_never_appears_as_clean():
    """A subtler version of the same mistake: a failed server must not be
    swept into the closing "all other checks came back clean" line, which is
    where a DBA's eye goes to stop worrying."""
    digest = build_digest([_clean("analytics-prod"), _failed("mysql-uat-01")], generated_at=_AT)

    clean_line = next(line for line in digest.splitlines() if "came back clean" in line)
    assert "analytics-prod" in clean_line
    assert "mysql-uat-01" not in clean_line


def test_only_deviations_get_their_own_block():
    """`comprehensive_summary`'s own "report ONLY deviations, then say
    plainly that everything else came back clean" discipline, applied a
    second time at the multi-server level — without it, a ten-server estate
    produces ten paragraphs of "X is fine" every morning, which is the same
    wall of text the playbook's guidance already rejects, just bigger."""
    digest = build_digest(
        [_flagged("core-banking-prod"), _clean("analytics-prod"), _clean("postgres-local")],
        generated_at=_AT,
    )

    assert "NEEDS ATTENTION" in digest
    assert "no successful full backup in 9 days" in digest
    # The clean servers are named once, in one line — not given a block each.
    assert "All other checks came back clean: analytics-prod, postgres-local." in digest
    assert digest.count("analytics-prod") == 1
    assert digest.count("postgres-local") == 1


def test_an_all_clean_morning_is_short_and_still_states_the_arithmetic():
    digest = build_digest([_clean("a"), _clean("b")], generated_at=_AT)

    assert "2 servers swept: 2 checked, 0 could not be checked." in digest
    assert "All other checks came back clean: a, b." in digest
    assert "NEEDS ATTENTION" not in digest
    assert "COULD NOT BE CHECKED" not in digest


def test_an_all_failed_morning_never_reads_as_a_clean_bill_of_health():
    """The worst case, stated explicitly: nothing was checked at all."""
    digest = build_digest([_failed("a"), _failed("b")], generated_at=_AT)

    assert "2 servers swept: 0 checked, 2 could not be checked." in digest
    assert "COULD NOT BE CHECKED" in digest
    assert "came back clean" not in digest


def test_the_digest_is_stamped_and_says_when_nothing_matched():
    assert "2026-03-04 06:00 UTC" in build_digest([_clean("a")], generated_at=_AT)

    empty = build_digest([], generated_at=_AT)
    assert "No servers were swept" in empty
    assert "came back clean" not in empty


def test_a_blocked_write_proposal_is_surfaced_in_the_servers_block():
    """"Numi wanted to act and was structurally stopped" is information a
    DBA should see, worded so it can never be misread as "Numi did
    something"."""
    summary = ScheduledSummary(
        server_id="core-banking-prod",
        environment="production",
        investigation_id="inv_1",
        status="ok",
        text="Session 13400 is blocking four others.",
        findings=("A long-running transaction is holding locks.",),
        dropped_proposals=("database.kill_session",),
    )

    digest = build_digest([summary], generated_at=_AT)

    assert "database.kill_session" in digest
    assert "never executes an action" in digest
    assert "would have proposed" in digest


# ---------------------------------------------------------- select_servers ---


def test_inactive_servers_are_skipped_and_registry_order_is_kept():
    servers = [
        {"id": "a", "status": "active", "aliases": []},
        {"id": "retired", "status": "decommissioned", "aliases": []},
        {"id": "b", "aliases": []},  # status absent — treated as active
    ]

    assert [s["id"] for s in select_servers(servers, "")] == ["a", "b"]


def test_a_configured_subset_matches_on_id_or_alias_case_insensitively():
    servers = [
        {"id": "core-banking-prod", "status": "active", "aliases": ["corebank"]},
        {"id": "analytics-prod", "status": "active", "aliases": []},
        {"id": "postgres-local", "status": "active", "aliases": []},
    ]

    selected = select_servers(servers, " CoreBank , postgres-local ")

    assert [s["id"] for s in selected] == ["core-banking-prod", "postgres-local"]


# ------------------------------------------------------- DailyDigestRunner ---


class _RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(self, *, channel_id: str, text: str) -> None:
        self.published.append((channel_id, text))


class _FakeOrchestrator:
    """Stands in for `AgentOrchestrator` at exactly the two methods the
    runner uses."""

    def __init__(self, *, servers, summaries=None, raises=None, registry_error=None):
        self._servers = servers
        self._summaries = summaries or {}
        self._raises = raises or {}
        self._registry_error = registry_error
        self.swept: list[str] = []

    async def list_registered_servers(self):
        if self._registry_error is not None:
            raise self._registry_error
        return self._servers

    async def run_comprehensive_summary(self, *, server_id, environment, channel, channel_account_id):
        self.swept.append(server_id)
        if server_id in self._raises:
            raise self._raises[server_id]
        return self._summaries[server_id]


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


@pytest.mark.asyncio
async def test_run_once_sweeps_every_server_and_publishes_one_combined_message():
    """ONE message, not one per server — the whole point of a digest."""
    servers = [
        {"id": "core-banking-prod", "environment": "production", "status": "active"},
        {"id": "analytics-prod", "environment": "production", "status": "active"},
    ]
    orchestrator = _FakeOrchestrator(
        servers=servers,
        summaries={
            "core-banking-prod": _flagged("core-banking-prod"),
            "analytics-prod": _clean("analytics-prod"),
        },
    )
    publisher = _RecordingPublisher()
    runner = DailyDigestRunner(
        orchestrator=orchestrator,
        settings=_settings(daily_report_slack_channel="C_DBA"),
        publisher=publisher,
    )

    digest = await runner.run_once()

    assert orchestrator.swept == ["core-banking-prod", "analytics-prod"]
    assert len(publisher.published) == 1
    channel_id, text = publisher.published[0]
    assert channel_id == "C_DBA"
    assert text == digest
    assert "core-banking-prod" in text and "analytics-prod" in text


@pytest.mark.asyncio
async def test_one_server_raising_never_aborts_the_sweep():
    """A sweep that stopped at the first exception would silently drop every
    server after it in registry order — the digest would look complete and
    simply be missing servers, which is precisely the invisible-gap failure
    the "could not be checked" section exists to prevent."""
    servers = [
        {"id": "first", "environment": "production", "status": "active"},
        {"id": "boom", "environment": "production", "status": "active"},
        {"id": "last", "environment": "production", "status": "active"},
    ]
    orchestrator = _FakeOrchestrator(
        servers=servers,
        summaries={"first": _clean("first"), "last": _clean("last")},
        raises={"boom": RuntimeError("connection reset by peer")},
    )
    publisher = _RecordingPublisher()
    runner = DailyDigestRunner(
        orchestrator=orchestrator,
        settings=_settings(daily_report_slack_channel="C_DBA"),
        publisher=publisher,
    )

    digest = await runner.run_once()

    assert orchestrator.swept == ["first", "boom", "last"]
    assert "3 servers swept: 2 checked, 1 could not be checked." in digest
    assert "boom" in digest
    # The raw exception text never reaches the channel — same no-raw-error
    # invariant the rest of this pipeline holds.
    assert "connection reset by peer" not in digest


@pytest.mark.asyncio
async def test_an_unreachable_server_registry_still_posts_a_visible_gap():
    """Silence is the one outcome never chosen: a digest that simply doesn't
    arrive is indistinguishable from a quiet morning."""
    orchestrator = _FakeOrchestrator(servers=[], registry_error=RuntimeError("gateway down"))
    publisher = _RecordingPublisher()
    runner = DailyDigestRunner(
        orchestrator=orchestrator,
        settings=_settings(daily_report_slack_channel="C_DBA"),
        publisher=publisher,
    )

    digest = await runner.run_once()

    assert len(publisher.published) == 1
    assert "not a clean bill of health" in digest
    assert "gateway down" not in digest


@pytest.mark.asyncio
async def test_a_publish_failure_never_escapes_into_the_scheduler():
    """`run_once` is a scheduler job: an exception escaping it kills nothing
    useful and just logs a traceback where nobody looks."""

    class _BrokenPublisher:
        async def publish(self, *, channel_id: str, text: str) -> None:
            raise RuntimeError("slack is down")

    orchestrator = _FakeOrchestrator(
        servers=[{"id": "a", "environment": "production", "status": "active"}],
        summaries={"a": _clean("a")},
    )
    runner = DailyDigestRunner(
        orchestrator=orchestrator,
        settings=_settings(daily_report_slack_channel="C_DBA"),
        publisher=_BrokenPublisher(),
    )

    digest = await runner.run_once()  # must not raise

    assert digest is not None


@pytest.mark.asyncio
async def test_run_once_is_a_no_op_with_no_channel_configured():
    """Belt and braces for the disabled case — the scheduler never registers
    a job without a channel (see test_daily_digest_scheduling.py), but a
    direct/manual call must be inert too, not post somewhere by default."""
    orchestrator = _FakeOrchestrator(servers=[{"id": "a", "environment": "production"}])
    publisher = _RecordingPublisher()
    runner = DailyDigestRunner(
        orchestrator=orchestrator, settings=_settings(), publisher=publisher
    )

    assert await runner.run_once() is None
    assert publisher.published == []
    assert orchestrator.swept == []
