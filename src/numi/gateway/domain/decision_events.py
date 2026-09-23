"""Decision-quality events (Gateway side) — the durable, queryable
counterpart to a handful of structured log lines.

**Scope, deliberately not exhaustive.** In scope: `conclusion_rejected_
ungrounded_identifiers`, `conclusion_rejected_pending_verification`,
`conclusion_rejected_self_critique`/`self_critique_call_failed`
(orchestrator.py), and cross-provider fallback substitutions (already
collected per-conversation on `ConversationState.llm_fallback_notices` as
`FallbackEvent`s — this just also persists them). Out of scope for now:
`llm_call_retrying`, `llm_call_deadline_exceeded`,
`decide_next_action_validation_failed` (`agent/llm/base.py`), and
`gemini_model_unavailable_switching` (`agent/llm/gemini_provider.py`) —
these fire from the registry's cached, cross-conversation provider
instances, which have no per-conversation sink and no `ToolClient`
reference to persist through; capturing them durably would mean threading
a new `events` kwarg through every provider's `decide_next_action`/
`extract_intent` signature, cascading through the ABC, all four vendors,
`CrossProviderFallbackLLM`, and every test double — a real future
extension, not something quietly skipped here.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from numi.gateway.infrastructure.db.models import LlmDecisionEventRecord


class DecisionEventStore(ABC):
    @abstractmethod
    async def append(
        self,
        *,
        event_type: str,
        conversation_id: str | None = None,
        investigation_id: str | None = None,
        provider: str = "",
        model: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def counts_by_type(self, *, since: dt.datetime) -> dict[str, int]: ...


class DbDecisionEventStore(DecisionEventStore):
    """Write-mostly, no in-process cache — the whole point is a durable,
    queryable record, and the volume here (a handful of events per
    investigation, at most) never justifies one."""

    def __init__(self, session_factory: async_sessionmaker):
        self._sf = session_factory

    async def append(
        self,
        *,
        event_type: str,
        conversation_id: str | None = None,
        investigation_id: str | None = None,
        provider: str = "",
        model: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        async with self._sf() as session:
            session.add(
                LlmDecisionEventRecord(
                    event_type=event_type,
                    conversation_id=conversation_id,
                    investigation_id=investigation_id,
                    provider=provider,
                    model=model,
                    payload=payload or {},
                )
            )
            await session.commit()

    async def counts_by_type(self, *, since: dt.datetime) -> dict[str, int]:
        """The actual review-loop query: "what's degrading, and how often,
        since I last checked" — one `GROUP BY`, not a dashboard."""
        async with self._sf() as session:
            rows = (
                await session.execute(
                    select(LlmDecisionEventRecord.event_type, func.count())
                    .where(LlmDecisionEventRecord.created_at >= since)
                    .group_by(LlmDecisionEventRecord.event_type)
                )
            ).all()
        return {event_type: count for event_type, count in rows}
