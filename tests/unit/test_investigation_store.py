"""`DbInvestigationStore` — the write path for what was previously a
write-orphaned table (see `gateway.domain.investigation_store`'s own
docstring). Mirrors `test_catalog_store.py`'s fixture style."""

from __future__ import annotations

import pytest
import pytest_asyncio

from numi.gateway.domain.investigation_store import DbInvestigationStore
from numi.gateway.infrastructure.db.session import Database


@pytest_asyncio.fixture
async def database():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    yield db
    await db.dispose()


def _fields(investigation_id: str, **overrides) -> dict:
    defaults = dict(
        investigation_id=investigation_id,
        conversation_id="conv_1",
        user_subject_id="U_MOCK_L2",
        server_id="postgres-dev-01",
        target={"instance": "postgres-dev-01", "environment": "development"},
        problem="high CPU",
        status="INVESTIGATING",
    )
    defaults.update(overrides)
    return defaults


@pytest.mark.asyncio
async def test_get_on_an_unknown_investigation_returns_none(database):
    store = DbInvestigationStore(database.session_factory)
    assert await store.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_create_then_get_round_trips_through_the_database(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1"))

    fetched = await store.get("inv_1")

    assert fetched is not None
    assert fetched.server_id == "postgres-dev-01"
    assert fetched.problem == "high CPU"
    assert fetched.status == "INVESTIGATING"


@pytest.mark.asyncio
async def test_create_is_idempotent_on_a_duplicate_id(database):
    """A retried create (the Agent never saw the response to its first
    attempt) must return the existing row, not 500 or duplicate it."""
    store = DbInvestigationStore(database.session_factory)
    first = await store.create(**_fields("inv_1", problem="first"))
    second = await store.create(**_fields("inv_1", problem="second"))

    assert second.investigation_id == first.investigation_id
    assert second.problem == "first"


@pytest.mark.asyncio
async def test_update_changes_only_the_fields_given(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1"))

    updated = await store.update("inv_1", status="CONCLUDED_VERIFIED", findings=["root cause found"])

    assert updated.status == "CONCLUDED_VERIFIED"
    assert updated.findings == ["root cause found"]
    assert updated.problem == "high CPU"  # untouched


@pytest.mark.asyncio
async def test_update_on_an_unknown_investigation_returns_none(database):
    store = DbInvestigationStore(database.session_factory)
    assert await store.update("does-not-exist", status="CONCLUDED_VERIFIED") is None


@pytest.mark.asyncio
async def test_a_fresh_store_instance_reads_what_another_instance_wrote(database):
    writer = DbInvestigationStore(database.session_factory)
    await writer.create(**_fields("inv_1"))

    reader = DbInvestigationStore(database.session_factory)
    fetched = await reader.get("inv_1")

    assert fetched is not None
    assert fetched.investigation_id == "inv_1"


@pytest.mark.asyncio
async def test_recent_for_server_orders_newest_first_and_respects_limit(database):
    store = DbInvestigationStore(database.session_factory)
    for i in range(5):
        await store.create(**_fields(f"inv_{i}", conversation_id=f"conv_{i}"))

    recent = await store.recent_for_server("postgres-dev-01", limit=2)

    assert len(recent) == 2
    assert recent[0].investigation_id == "inv_4"
    assert recent[1].investigation_id == "inv_3"


@pytest.mark.asyncio
async def test_recent_for_server_only_returns_that_server(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1", server_id="postgres-dev-01"))
    await store.create(**_fields("inv_2", server_id="sqlserver-dev-01"))

    recent = await store.recent_for_server("postgres-dev-01", limit=10)

    assert [r.investigation_id for r in recent] == ["inv_1"]


@pytest.mark.asyncio
async def test_recent_for_server_excludes_the_given_id(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1"))
    await store.create(**_fields("inv_2"))

    recent = await store.recent_for_server(
        "postgres-dev-01", limit=10, exclude_investigation_id="inv_2"
    )

    assert [r.investigation_id for r in recent] == ["inv_1"]


@pytest.mark.asyncio
async def test_append_event_persists_a_row(database):
    from sqlalchemy import select

    from numi.gateway.infrastructure.db.models import InvestigationEventRecord

    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1"))

    await store.append_event("inv_1", "OBSERVATION", {"note": "high wait events"})

    async with database.session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(InvestigationEventRecord).where(
                        InvestigationEventRecord.investigation_id == "inv_1"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].event_type == "OBSERVATION"
    assert rows[0].payload == {"note": "high wait events"}


@pytest.mark.asyncio
async def test_find_similar_matches_by_playbook_id(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1", playbook_id="slow_queries", server_id="server-a"))
    await store.create(**_fields("inv_2", playbook_id="high_cpu", server_id="server-b"))

    matches = await store.find_similar(playbook_id="slow_queries")

    assert [m.investigation_id for m in matches] == ["inv_1"]


@pytest.mark.asyncio
async def test_find_similar_excludes_the_asking_server(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1", playbook_id="slow_queries", server_id="server-a"))
    await store.create(**_fields("inv_2", playbook_id="slow_queries", server_id="server-b"))

    matches = await store.find_similar(playbook_id="slow_queries", exclude_server_id="server-a")

    assert [m.investigation_id for m in matches] == ["inv_2"]


@pytest.mark.asyncio
async def test_find_similar_filters_by_environment_when_given(database):
    store = DbInvestigationStore(database.session_factory)
    await store.create(
        **_fields("inv_1", playbook_id="slow_queries", server_id="server-a", environment="production")
    )
    await store.create(
        **_fields("inv_2", playbook_id="slow_queries", server_id="server-b", environment="development")
    )

    matches = await store.find_similar(playbook_id="slow_queries", environment="production")

    assert [m.investigation_id for m in matches] == ["inv_1"]


@pytest.mark.asyncio
async def test_find_similar_respects_a_since_cutoff(database):
    import datetime as dt

    store = DbInvestigationStore(database.session_factory)
    await store.create(**_fields("inv_1", playbook_id="slow_queries", server_id="server-a"))

    future_since = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
    matches = await store.find_similar(playbook_id="slow_queries", since=future_since)

    assert matches == []
