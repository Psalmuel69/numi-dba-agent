"""Catalog storage for the Gateway.

The catalog *models* live in `numi.common.models.catalog` (shared with the
Execution Service's discovery crawler). This module is just the store the
Gateway reads for target validation and `/catalog`, and writes after a
discovery run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numi.common.models.catalog import (
    DiscoveredDatabase,
    DiscoveredExtension,
    DiscoveredObject,
    ServerCatalog,
)

__all__ = [
    "DiscoveredDatabase",
    "DiscoveredExtension",
    "DiscoveredObject",
    "ServerCatalog",
    "CatalogStore",
    "InMemoryCatalogStore",
]


class CatalogStore(ABC):
    @abstractmethod
    async def get(self, server_id: str) -> ServerCatalog | None: ...

    @abstractmethod
    async def put(self, catalog: ServerCatalog) -> None: ...

    @abstractmethod
    async def all(self) -> list[ServerCatalog]: ...


class InMemoryCatalogStore(CatalogStore):
    """Process-local. A DB-backed store
    (`numi.gateway.infrastructure.catalog_store`) is used when the control
    DB is available so the catalog survives a restart and is shared across
    replicas."""

    def __init__(self) -> None:
        self._by_server: dict[str, ServerCatalog] = {}

    async def get(self, server_id: str) -> ServerCatalog | None:
        return self._by_server.get(server_id)

    async def put(self, catalog: ServerCatalog) -> None:
        self._by_server[catalog.server_id] = catalog

    async def all(self) -> list[ServerCatalog]:
        return list(self._by_server.values())
