"""Investigation persistence (Gateway side).

Read-write counterpart to what was previously a write-orphaned table:
`InvestigationRecord`/`InvestigationEventRecord` have existed in the schema
since migration 0001, but nothing ever wrote to them — only a read-only
`GET /v1/investigations/{id}` existed, whose own docstring anticipated this
exact module. This is that module.

Deliberately NOT a `DbCatalogStore`-style cache-in-front store: a catalog is
one row per registered server (small, bounded, worth loading whole into
memory), while investigations are unbounded and append-heavy. This is a
thin write-through store instead — every operation hits the DB directly.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from numi.gateway.infrastructure.db.models import InvestigationEventRecord, InvestigationRecord


class InvestigationStore(ABC):
    @abstractmethod
    async def create(self, **fields: Any) -> InvestigationRecord: ...

    @abstractmethod
    async def update(self, investigation_id: str, **fields: Any) -> InvestigationRecord | None: ...

    @abstractmethod
    async def get(self, investigation_id: str) -> InvestigationRecord | None: ...

    @abstractmethod
    async def recent_for_server(
        self,
        server_id: str,
        *,
        limit: int = 3,
        exclude_investigation_id: str | None = None,
    ) -> list[InvestigationRecord]: ...

    @abstractmethod
    async def append_event(self, investigation_id: str, event_type: str, payload: dict) -> None: ...

    @abstractmethod
    async def find_similar(
        self,
        *,
        playbook_id: str,
        environment: str | None = None,
        exclude_server_id: str | None = None,
        since: dt.datetime | None = None,
        limit: int = 10,
    ) -> list[InvestigationRecord]: ...


class DbInvestigationStore(InvestigationStore):
    def __init__(self, session_factory: async_sessionmaker):
        self._sf = session_factory

    async def create(self, **fields: Any) -> InvestigationRecord:
        async with self._sf() as session:
            existing = await session.get(InvestigationRecord, fields.get("investigation_id"))
            if existing is not None:
                # Idempotent on a duplicate id — a retried create (e.g. the
                # Agent never saw the response to its first attempt) must
                # not 500 or duplicate the row.
                return existing
            record = InvestigationRecord(**fields)
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def update(self, investigation_id: str, **fields: Any) -> InvestigationRecord | None:
        async with self._sf() as session:
            record = await session.get(InvestigationRecord, investigation_id)
            if record is None:
                return None
            for key, value in fields.items():
                if value is not None:
                    setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
            return record

    async def get(self, investigation_id: str) -> InvestigationRecord | None:
        async with self._sf() as session:
            return await session.get(InvestigationRecord, investigation_id)

    async def recent_for_server(
        self,
        server_id: str,
        *,
        limit: int = 3,
        exclude_investigation_id: str | None = None,
    ) -> list[InvestigationRecord]:
        # Fetch one extra row when excluding, so filtering it out still
        # leaves `limit` results rather than silently under-returning.
        fetch_limit = limit + 1 if exclude_investigation_id else limit
        async with self._sf() as session:
            stmt = (
                select(InvestigationRecord)
                .where(InvestigationRecord.server_id == server_id)
                .order_by(InvestigationRecord.created_at.desc())
                .limit(fetch_limit)
            )
            rows = list((await session.execute(stmt)).scalars().all())
        if exclude_investigation_id:
            rows = [r for r in rows if r.investigation_id != exclude_investigation_id][:limit]
        return rows

    async def append_event(self, investigation_id: str, event_type: str, payload: dict) -> None:
        async with self._sf() as session:
            session.add(
                InvestigationEventRecord(
                    investigation_id=investigation_id, event_type=event_type, payload=payload
                )
            )
            await session.commit()

    async def find_similar(
        self,
        *,
        playbook_id: str,
        environment: str | None = None,
        exclude_server_id: str | None = None,
        since: dt.datetime | None = None,
        limit: int = 10,
    ) -> list[InvestigationRecord]:
        """Structural correlation only — same `playbook_id` (the
        deterministic scenario match `agent.playbooks.library.match_playbook`
        already makes, reused rather than a second vocabulary), optionally
        the same `environment`, never the asking server itself. No fuzzy
        text similarity; see `InvestigationMemory.correlate`'s own docstring
        for why that's a deliberate v1 scope cut, not an oversight."""
        async with self._sf() as session:
            stmt = (
                select(InvestigationRecord)
                .where(InvestigationRecord.playbook_id == playbook_id)
                .order_by(InvestigationRecord.created_at.desc())
                .limit(limit)
            )
            if environment is not None:
                stmt = stmt.where(InvestigationRecord.environment == environment)
            if exclude_server_id is not None:
                stmt = stmt.where(InvestigationRecord.server_id != exclude_server_id)
            if since is not None:
                stmt = stmt.where(InvestigationRecord.created_at >= since)
            return list((await session.execute(stmt)).scalars().all())
