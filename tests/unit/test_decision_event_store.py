"""`DbDecisionEventStore` — durable storage for a handful of decision-
quality signals (see `gateway.domain.decision_events`'s own docstring for
which ones and why not all of them). Mirrors `test_catalog_store.py`'s
fixture style."""

from __future__ import annotations

import datetime as dt

import pytest
import pytest_asyncio

from numi.gateway.domain.decision_events import DbDecisionEventStore
from numi.gateway.infrastructure.db.session import Database


@pytest_asyncio.fixture
async def database():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    yield db
    await db.dispose()


@pytest.mark.asyncio
async def test_append_then_counts_by_type_reflects_it(database):
    store = DbDecisionEventStore(database.session_factory)
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)

    await store.append(event_type="conclusion_rejected_ungrounded_identifiers")

    counts = await store.counts_by_type(since=since)
    assert counts == {"conclusion_rejected_ungrounded_identifiers": 1}


@pytest.mark.asyncio
async def test_counts_by_type_groups_multiple_event_types(database):
    store = DbDecisionEventStore(database.session_factory)
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)

    await store.append(event_type="conclusion_rejected_self_critique")
    await store.append(event_type="conclusion_rejected_self_critique")
    await store.append(event_type="llm_cross_provider_fallback_used")

    counts = await store.counts_by_type(since=since)
    assert counts == {"conclusion_rejected_self_critique": 2, "llm_cross_provider_fallback_used": 1}


@pytest.mark.asyncio
async def test_counts_by_type_excludes_events_before_the_since_cutoff(database):
    store = DbDecisionEventStore(database.session_factory)
    await store.append(event_type="conclusion_rejected_pending_verification")

    future_since = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    counts = await store.counts_by_type(since=future_since)

    assert counts == {}


@pytest.mark.asyncio
async def test_append_persists_optional_fields(database):
    from sqlalchemy import select

    from numi.gateway.infrastructure.db.models import LlmDecisionEventRecord

    store = DbDecisionEventStore(database.session_factory)
    await store.append(
        event_type="conclusion_rejected_self_critique",
        conversation_id="conv_1",
        investigation_id="inv_1",
        provider="gemini",
        model="gemini-3.8-flash",
        payload={"issue": "no evidence for this"},
    )

    async with database.session_factory() as session:
        rows = (await session.execute(select(LlmDecisionEventRecord))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.conversation_id == "conv_1"
    assert row.investigation_id == "inv_1"
    assert row.provider == "gemini"
    assert row.model == "gemini-3.8-flash"
    assert row.payload == {"issue": "no evidence for this"}


@pytest.mark.asyncio
async def test_a_fresh_store_instance_reads_what_another_instance_wrote(database):
    writer = DbDecisionEventStore(database.session_factory)
    await writer.append(event_type="conclusion_rejected_ungrounded_identifiers")

    reader = DbDecisionEventStore(database.session_factory)
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    counts = await reader.counts_by_type(since=since)

    assert counts == {"conclusion_rejected_ungrounded_identifiers": 1}
