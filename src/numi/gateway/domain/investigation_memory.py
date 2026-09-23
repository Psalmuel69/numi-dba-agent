"""Recall recent findings for a server before a new investigation starts —
the same "look up, degrade to nothing rather than fail hard" shape as
`DiscoveryOrchestrator.ensure_fresh`.
"""

from __future__ import annotations

import datetime as dt

from numi.common.models.investigation import InvestigationMemoryEntry
from numi.common.observability import get_logger
from numi.gateway.domain.investigation_store import InvestigationStore

logger = get_logger(__name__)

# Only a concluded investigation's findings/recommendations are meaningful
# background for a new one — one still in progress (or awaiting
# clarification/verification) has nothing settled to recall yet.
_CONCLUDED_PREFIX = "CONCLUDED"


class InvestigationMemory:
    def __init__(self, store: InvestigationStore):
        self._store = store

    async def recall(
        self,
        server_id: str,
        *,
        exclude_investigation_id: str | None = None,
        limit: int = 3,
    ) -> list[InvestigationMemoryEntry]:
        if limit <= 0:
            return []
        try:
            # Over-fetch: the store's `limit` doesn't know about the
            # concluded-only filter applied below, so asking for exactly
            # `limit` rows could under-return if recent ones are still
            # in-progress. A small multiplier keeps this a single query
            # without adding a status filter to the store's generic
            # interface (which `find_similar`/Phase 5 doesn't want).
            records = await self._store.recent_for_server(
                server_id,
                limit=limit * 4,
                exclude_investigation_id=exclude_investigation_id,
            )
        except Exception as exc:  # noqa: BLE001 — a lookup failure must never block
            # a new investigation from starting; recall is an enhancement,
            # not a dependency.
            logger.warning(
                "investigation_memory_lookup_failed", server_id=server_id, error=str(exc)
            )
            return []
        return [
            InvestigationMemoryEntry(
                investigation_id=r.investigation_id,
                server_id=r.server_id,
                problem=r.problem,
                status=r.status,
                findings=r.findings,
                recommendations=r.recommendations,
                updated_at=r.updated_at,
            )
            for r in records
            if r.status.startswith(_CONCLUDED_PREFIX)
        ][:limit]

    async def correlate(
        self,
        *,
        playbook_id: str | None,
        environment: str | None = None,
        exclude_server_id: str | None = None,
        since: dt.datetime | None = None,
        limit: int = 5,
    ) -> list[InvestigationMemoryEntry]:
        """Cross-server pattern correlation — "this same symptom happened
        on N other servers recently." Structural only: matches by shared
        `playbook_id` (the deterministic scenario match
        `agent.playbooks.library.match_playbook` already makes) and
        optionally `environment`, never fuzzy text similarity — there is no
        embeddings/vector-search infrastructure anywhere in this codebase,
        and building one is a much larger, separate investment than this
        feature implies. `None` means no playbook matched (a fully freeform
        investigation), which has no meaningful scenario to correlate on —
        returns `[]` immediately rather than querying "every investigation
        with no playbook", which isn't a real pattern."""
        if not playbook_id:
            return []
        try:
            records = await self._store.find_similar(
                playbook_id=playbook_id,
                environment=environment,
                exclude_server_id=exclude_server_id,
                since=since,
                limit=limit * 4,
            )
        except Exception as exc:  # noqa: BLE001 — same posture as recall(): an
            # enhancement, never a dependency.
            logger.warning(
                "investigation_correlation_lookup_failed", playbook_id=playbook_id, error=str(exc)
            )
            return []
        return [
            InvestigationMemoryEntry(
                investigation_id=r.investigation_id,
                server_id=r.server_id,
                problem=r.problem,
                status=r.status,
                findings=r.findings,
                recommendations=r.recommendations,
                updated_at=r.updated_at,
            )
            for r in records
            if r.status.startswith(_CONCLUDED_PREFIX)
        ][:limit]
