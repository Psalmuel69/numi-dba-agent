"""Control-DB-backed CatalogStore.

Persists the discovered catalog so it survives a Gateway restart and is
shared across replicas, with a small in-process cache in front so target
validation doesn't hit the DB on every tool call.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from numi.common.models.catalog import ServerCatalog
from numi.gateway.domain.catalog import CatalogStore
from numi.gateway.infrastructure.db.models import ServerCatalogRecord


class DbCatalogStore(CatalogStore):
    def __init__(self, session_factory: async_sessionmaker):
        self._sf = session_factory
        self._cache: dict[str, ServerCatalog] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._sf() as session:
            rows = (await session.execute(select(ServerCatalogRecord))).scalars().all()
        for row in rows:
            try:
                self._cache[row.server_id] = ServerCatalog.model_validate(row.catalog)
            except Exception:  # noqa: BLE001 — a corrupt row shouldn't break startup
                continue
        self._loaded = True

    async def get(self, server_id: str) -> ServerCatalog | None:
        await self._ensure_loaded()
        return self._cache.get(server_id)

    async def put(self, catalog: ServerCatalog) -> None:
        await self._ensure_loaded()
        self._cache[catalog.server_id] = catalog
        payload = catalog.model_dump(mode="json")
        async with self._sf() as session:
            existing = await session.get(ServerCatalogRecord, catalog.server_id)
            if existing is None:
                session.add(
                    ServerCatalogRecord(
                        server_id=catalog.server_id,
                        discovered_at=catalog.discovered_at,
                        engine_version=catalog.engine_version,
                        engine_edition=catalog.engine_edition,
                        catalog=payload,
                    )
                )
            else:
                existing.discovered_at = catalog.discovered_at
                existing.engine_version = catalog.engine_version
                existing.engine_edition = catalog.engine_edition
                existing.catalog = payload
            await session.commit()

    async def all(self) -> list[ServerCatalog]:
        await self._ensure_loaded()
        return list(self._cache.values())
