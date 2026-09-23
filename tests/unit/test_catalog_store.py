"""`DbCatalogStore` — the control-DB-backed catalog store `TargetValidator`
and discovery both read/write through. Untested until now despite sitting
directly behind the `server_catalogs` migration bug (0002): this store's
`_ensure_loaded` is exactly what raised `UndefinedTableError` on a database
migrated before that table existed, and nothing caught it because nothing
here was ever exercised against a real session/table round-trip.
"""

from __future__ import annotations

import datetime as dt

import pytest
import pytest_asyncio
from sqlalchemy import select

from numi.common.models.catalog import ServerCatalog
from numi.gateway.infrastructure.catalog_store import DbCatalogStore
from numi.gateway.infrastructure.db.models import ServerCatalogRecord
from numi.gateway.infrastructure.db.session import Database


@pytest_asyncio.fixture
async def database():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    yield db
    await db.dispose()


def _catalog(server_id: str, **overrides) -> ServerCatalog:
    defaults = dict(
        server_id=server_id,
        discovered_at=dt.datetime(2026, 3, 4, 6, 0, tzinfo=dt.UTC),
        engine_version="16.2",
        engine_edition="Standard",
    )
    defaults.update(overrides)
    return ServerCatalog(**defaults)


@pytest.mark.asyncio
async def test_get_on_an_unknown_server_returns_none(database):
    store = DbCatalogStore(database.session_factory)
    assert await store.get("does-not-exist") is None


@pytest.mark.asyncio
async def test_put_then_get_round_trips_through_the_database(database):
    """The actual property migration 0002 was fixing: a catalog written by
    one call is readable back, via the real table, not just the in-memory
    cache."""
    store = DbCatalogStore(database.session_factory)
    catalog = _catalog("postgres-local")

    await store.put(catalog)
    fetched = await store.get("postgres-local")

    assert fetched is not None
    assert fetched.server_id == "postgres-local"
    assert fetched.engine_version == "16.2"
    assert fetched.discovered_at == catalog.discovered_at


@pytest.mark.asyncio
async def test_a_fresh_store_instance_reads_what_another_instance_wrote(database):
    """Simulates two Gateway replicas sharing the control DB: the whole
    reason this store exists (over the in-memory-only CatalogStore) is that
    a catalog written by one process is visible to another via the shared
    table, not just process-local state."""
    writer = DbCatalogStore(database.session_factory)
    await writer.put(_catalog("sqlserver-dev-01", engine_version="2022"))

    reader = DbCatalogStore(database.session_factory)
    fetched = await reader.get("sqlserver-dev-01")

    assert fetched is not None
    assert fetched.engine_version == "2022"


@pytest.mark.asyncio
async def test_put_for_an_existing_server_updates_in_place_not_a_duplicate(database):
    store = DbCatalogStore(database.session_factory)
    await store.put(_catalog("postgres-local", engine_version="15.0"))
    await store.put(_catalog("postgres-local", engine_version="16.2"))

    fetched = await store.get("postgres-local")
    assert fetched.engine_version == "16.2"

    async with database.session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(ServerCatalogRecord).where(ServerCatalogRecord.server_id == "postgres-local")
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_all_returns_every_stored_catalog(database):
    store = DbCatalogStore(database.session_factory)
    await store.put(_catalog("postgres-local"))
    await store.put(_catalog("sqlserver-dev-01"))

    all_catalogs = await store.all()

    assert {c.server_id for c in all_catalogs} == {"postgres-local", "sqlserver-dev-01"}


@pytest.mark.asyncio
async def test_a_corrupt_row_is_skipped_not_fatal_to_startup(database):
    """`_ensure_loaded` must not take the whole Gateway down because one
    row's JSON no longer matches the current `ServerCatalog` schema (e.g.
    after a model change) — a missing catalog just means the next lookup
    falls through to a fresh discovery, which is recoverable; a startup
    crash across the whole estate is not."""
    async with database.session_factory() as session:
        session.add(
            ServerCatalogRecord(
                server_id="corrupt-entry",
                catalog={"this": "does not match ServerCatalog's required fields at all", "server_id": None},
            )
        )
        session.add(
            ServerCatalogRecord(
                server_id="postgres-local",
                catalog=_catalog("postgres-local").model_dump(mode="json"),
            )
        )
        await session.commit()

    store = DbCatalogStore(database.session_factory)

    assert await store.get("corrupt-entry") is None
    fetched = await store.get("postgres-local")
    assert fetched is not None and fetched.server_id == "postgres-local"


@pytest.mark.asyncio
async def test_ensure_loaded_only_queries_the_database_once(database):
    """The in-process cache exists specifically so target validation isn't a
    DB round-trip on every tool call (see the module docstring) — assert
    that's actually true rather than just documented."""
    async with database.session_factory() as session:
        session.add(
            ServerCatalogRecord(
                server_id="postgres-local",
                catalog=_catalog("postgres-local").model_dump(mode="json"),
            )
        )
        await session.commit()

    calls = {"count": 0}
    original_sf = database.session_factory

    def counting_session_factory():
        calls["count"] += 1
        return original_sf()

    store = DbCatalogStore(counting_session_factory)

    await store.get("postgres-local")
    calls_after_first = calls["count"]
    assert calls_after_first >= 1

    await store.get("postgres-local")
    await store.get("some-other-id")
    await store.all()

    assert calls["count"] == calls_after_first  # no additional query after the first load
