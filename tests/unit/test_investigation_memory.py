"""`InvestigationMemory.recall` — the read side of investigation memory.
Same "look up, degrade to nothing rather than fail hard" shape as
`DiscoveryOrchestrator.ensure_fresh`, tested the same way its own test
suite does (a fake/failing store, never a real DB round trip here — the
real round trip is covered by test_investigation_store.py)."""

from __future__ import annotations

import datetime as dt

import pytest

from numi.gateway.domain.investigation_memory import InvestigationMemory
from numi.gateway.domain.investigation_store import InvestigationStore


class _Row:
    def __init__(self, investigation_id, status, findings=None, recommendations=None, server_id=None):
        self.investigation_id = investigation_id
        self.server_id = server_id
        self.problem = "high CPU"
        self.status = status
        self.findings = findings or ["some finding"]
        self.recommendations = recommendations or ["some recommendation"]
        self.updated_at = dt.datetime(2026, 3, 4, 6, 0, tzinfo=dt.UTC)


class _FakeStore(InvestigationStore):
    def __init__(self, rows=None, *, raises=False):
        self._rows = rows or []
        self._raises = raises

    async def create(self, **fields):
        raise NotImplementedError

    async def update(self, investigation_id, **fields):
        raise NotImplementedError

    async def get(self, investigation_id):
        raise NotImplementedError

    async def recent_for_server(self, server_id, *, limit=3, exclude_investigation_id=None):
        if self._raises:
            raise RuntimeError("db unreachable")
        return self._rows[:limit]

    async def append_event(self, investigation_id, event_type, payload):
        raise NotImplementedError

    async def find_similar(
        self, *, playbook_id, environment=None, exclude_server_id=None, since=None, limit=10
    ):
        if self._raises:
            raise RuntimeError("db unreachable")
        return self._rows[:limit]


@pytest.mark.asyncio
async def test_recall_returns_entries_for_concluded_investigations():
    store = _FakeStore([_Row("inv_1", "CONCLUDED_VERIFIED")])
    memory = InvestigationMemory(store)

    entries = await memory.recall("postgres-dev-01")

    assert len(entries) == 1
    assert entries[0].investigation_id == "inv_1"
    assert entries[0].findings == ["some finding"]


@pytest.mark.asyncio
async def test_recall_excludes_investigations_still_in_progress():
    store = _FakeStore(
        [_Row("inv_1", "INVESTIGATING"), _Row("inv_2", "AWAITING_CLARIFICATION")]
    )
    memory = InvestigationMemory(store)

    entries = await memory.recall("postgres-dev-01")

    assert entries == []


@pytest.mark.asyncio
async def test_recall_swallows_a_store_failure_and_returns_empty():
    store = _FakeStore(raises=True)
    memory = InvestigationMemory(store)

    entries = await memory.recall("postgres-dev-01")

    assert entries == []


@pytest.mark.asyncio
async def test_recall_with_limit_zero_never_queries_the_store():
    store = _FakeStore([_Row("inv_1", "CONCLUDED_VERIFIED")])
    memory = InvestigationMemory(store)

    entries = await memory.recall("postgres-dev-01", limit=0)

    assert entries == []


@pytest.mark.asyncio
async def test_correlate_returns_matches_tagged_with_their_own_server():
    store = _FakeStore(
        [_Row("inv_1", "CONCLUDED_VERIFIED", server_id="sqlserver-dev-02")]
    )
    memory = InvestigationMemory(store)

    entries = await memory.correlate(playbook_id="slow_queries")

    assert len(entries) == 1
    assert entries[0].server_id == "sqlserver-dev-02"


@pytest.mark.asyncio
async def test_correlate_with_no_playbook_id_never_queries_the_store():
    store = _FakeStore([_Row("inv_1", "CONCLUDED_VERIFIED")])
    memory = InvestigationMemory(store)

    entries = await memory.correlate(playbook_id=None)

    assert entries == []


@pytest.mark.asyncio
async def test_correlate_excludes_investigations_still_in_progress():
    store = _FakeStore([_Row("inv_1", "INVESTIGATING")])
    memory = InvestigationMemory(store)

    entries = await memory.correlate(playbook_id="slow_queries")

    assert entries == []


@pytest.mark.asyncio
async def test_correlate_swallows_a_store_failure_and_returns_empty():
    store = _FakeStore(raises=True)
    memory = InvestigationMemory(store)

    entries = await memory.correlate(playbook_id="slow_queries")

    assert entries == []
