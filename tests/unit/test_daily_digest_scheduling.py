"""Opt-in wiring for the scheduled daily digest
(`agent.scheduled_report.schedule_daily_digest`).

Deliberately tests only *registration*, never firing: whether a cron job
actually triggers at 06:00 UTC is APScheduler's responsibility and testing it
would mean either waiting a day or monkey-patching a clock to assert someone
else's library works. What is ours — and what would actually break a
deployment — is the opt-in contract: with nothing configured there must be no
scheduler object, no job, and no background task at all; with a channel
configured there must be exactly one job, at the configured hour, in UTC.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from numi.agent.scheduled_report import DAILY_DIGEST_JOB_ID, schedule_daily_digest
from numi.common.config import Settings


class _UnusedRunner:
    """The scheduler must never call this during registration — only at a
    real tick, which no test here provokes."""

    async def run_once(self) -> str | None:  # pragma: no cover - never invoked
        raise AssertionError("the digest job must not run at registration time")


def test_no_channel_configured_means_nothing_is_scheduled_at_all():
    """The default, and the entire off switch. Not "a scheduler that wakes
    up daily and finds it has nowhere to post" — no scheduler object exists,
    so a deployment that never asked for a proactive digest carries no
    background machinery and no extra thread."""
    scheduler = schedule_daily_digest(_UnusedRunner(), Settings(_env_file=None))

    assert scheduler is None


def test_no_channel_configured_is_not_an_error():
    """Disabled is a normal state, not a misconfiguration — it must not
    raise, warn-and-die, or require any other setting to be present. This is
    what lets `create_app` construct the runner unconditionally and simply
    not start it."""
    settings = Settings(_env_file=None, daily_report_hour_utc=9, daily_report_servers="a,b")

    assert schedule_daily_digest(_UnusedRunner(), settings) is None


@pytest.mark.asyncio
async def test_a_configured_channel_registers_exactly_one_daily_job():
    """Async because `AsyncIOScheduler.start()` binds to the running event
    loop — which is exactly how it is started in production, inside the
    Agent app's FastAPI lifespan (see `agent.api.app.create_app`), rather
    than at import time."""
    settings = Settings(
        _env_file=None, daily_report_slack_channel="C_DBA_ALERTS", daily_report_hour_utc=6
    )
    scheduler = schedule_daily_digest(_UnusedRunner(), settings)

    assert scheduler is not None
    try:
        jobs = scheduler.get_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.id == DAILY_DIGEST_JOB_ID
        # The hour actually configured, and pinned to UTC — so the digest
        # lands at the same real moment regardless of the host's timezone.
        fields = {f.name: str(f) for f in job.trigger.fields}
        assert fields["hour"] == "6"
        assert fields["minute"] == "0"
        assert str(job.trigger.timezone) == "UTC"
        # Never two overlapping sweeps, and missed occurrences collapse into
        # one post rather than a burst.
        assert job.max_instances == 1
        assert job.coalesce is True
    finally:
        scheduler.shutdown(wait=False)


@pytest.mark.asyncio
async def test_the_configured_hour_is_honored():
    settings = Settings(
        _env_file=None, daily_report_slack_channel="C_DBA_ALERTS", daily_report_hour_utc=23
    )
    scheduler = schedule_daily_digest(_UnusedRunner(), settings)

    assert scheduler is not None
    try:
        fields = {f.name: str(f) for f in scheduler.get_jobs()[0].trigger.fields}
        assert fields["hour"] == "23"
    finally:
        scheduler.shutdown(wait=False)


def test_an_out_of_range_hour_is_rejected_at_configuration_time():
    """Bounded on the Settings field rather than clamped at schedule time, so
    a typo fails the process at startup — next to every other configuration
    mistake — instead of silently running at an hour nobody intended."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, daily_report_hour_utc=25)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, daily_report_hour_utc=-1)
