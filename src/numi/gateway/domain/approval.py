"""Approval Engine (spec §15, §16, §17, §44).

Approvals are cryptographically bound to the *exact* action they were issued
for: actor, tool, tool version, target, normalized arguments, environment,
database, and risk classification are hashed together into `action_hash`.
Any tool-call submitted against an approval_id is re-hashed the same way at
execution time and compared — if a single byte of the underlying request
differs (spec's example: "kill session 9182" approved, then execution
attempted against "9183"), the mismatch is detected here, independent of
anything the Agent or the approving client claims.

Approvals always expire. There is no code path that creates a permanent
approval, and re-approving an expired or already-decided approval is
rejected rather than silently reused.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession

from numi.common.ids import new_id
from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.identity import VerifiedIdentity
from numi.common.models.risk import RiskAssessment
from numi.common.models.target import DatabaseTarget
from numi.common.models.tool import ToolDefinition
from numi.gateway.infrastructure.db.models import ApprovalEventRecord, ApprovalRecord

_DEFAULT_TTL_SECONDS = 600  # 10 minutes, per the spec's worked example (§16)


def _naive_utc(value: dt.datetime) -> dt.datetime:
    """Normalizes to a naive UTC datetime for comparison.

    SQLite (used for local dev/tests) does not round-trip timezone info on
    DateTime columns, so a value read back from the DB may be naive even
    though it was written as timezone-aware UTC. Postgres (production)
    preserves the offset. Comparing on a consistently-naive-UTC basis avoids
    "can't compare offset-naive and offset-aware datetimes" while still
    being correct, since every datetime this module writes is UTC.
    """
    if value.tzinfo is not None:
        return value.astimezone(dt.UTC).replace(tzinfo=None)
    return value


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    AWAITING_SECOND_APPROVAL = "AWAITING_SECOND_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    # Set once execution has actually completed against this approval — an
    # approved-but-already-used approval can never authorize a *second*
    # execution, even if it hasn't expired yet (spec §43's "duplicate
    # execution" / replay test).
    EXECUTED = "EXECUTED"


class ApprovalDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


def compute_action_hash(
    *,
    actor_subject_id: str,
    tool_id: str,
    tool_version: str,
    target: dict,
    normalized_arguments: dict,
    environment: str,
    database_id: str,
    risk_level: str,
) -> str:
    canonical = json.dumps(
        {
            "actor": actor_subject_id,
            "tool_id": tool_id,
            "tool_version": tool_version,
            "target": target,
            "arguments": normalized_arguments,
            "environment": environment,
            "database_id": database_id,
            "risk_level": risk_level,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovalContext:
    """Everything needed to both create an approval and later recompute its
    action hash — kept together so the two computations can never drift."""

    request_id: str
    actor: VerifiedIdentity
    tool: ToolDefinition
    target: DatabaseTarget
    normalized_arguments: dict
    environment: str
    database_id: str
    risk: RiskAssessment

    def action_hash(self) -> str:
        return compute_action_hash(
            actor_subject_id=self.actor.subject_id,
            tool_id=self.tool.tool_id,
            tool_version=self.tool.version,
            target=self.target.model_dump(mode="json"),
            normalized_arguments=self.normalized_arguments,
            environment=self.environment,
            database_id=self.database_id,
            risk_level=self.risk.risk_level.value,
        )


class ApprovalEngine:
    def __init__(self, session: AsyncSession, default_ttl_seconds: int = _DEFAULT_TTL_SECONDS):
        self._session = session
        self._ttl = default_ttl_seconds

    async def create(
        self,
        ctx: ApprovalContext,
        *,
        requires_dual_approval: bool,
        now: dt.datetime | None = None,
    ) -> ApprovalRecord:
        now = now or dt.datetime.now(dt.UTC)
        record = ApprovalRecord(
            approval_id=new_id("appr"),
            request_id=ctx.request_id,
            actor_subject_id=ctx.actor.subject_id,
            tool_id=ctx.tool.tool_id,
            tool_version=ctx.tool.version,
            target=ctx.target.model_dump(mode="json"),
            normalized_arguments=ctx.normalized_arguments,
            risk=ctx.risk.model_dump(mode="json"),
            policy={"database_id": ctx.database_id, "environment": ctx.environment},
            action_hash=ctx.action_hash(),
            requires_dual_approval=requires_dual_approval,
            status=ApprovalStatus.PENDING.value,
            created_at=now,
            expires_at=now + dt.timedelta(seconds=self._ttl),
        )
        self._session.add(record)
        self._session.add(
            ApprovalEventRecord(
                approval_id=record.approval_id,
                event_type="CREATED",
                actor_subject_id=ctx.actor.subject_id,
                detail={"tool_id": ctx.tool.tool_id},
                created_at=now,
            )
        )
        await self._session.flush()
        return record

    async def _get(self, approval_id: str) -> ApprovalRecord:
        record = await self._session.get(ApprovalRecord, approval_id)
        if record is None:
            raise NumiError(FailureCode.APPROVAL_INVALID, "Approval not found.")
        return record

    async def _expire_if_needed(self, record: ApprovalRecord, now: dt.datetime) -> None:
        # Expiry applies whether or not a decision has been made: an
        # APPROVED-but-unused approval must still lapse if execution is not
        # attempted within its validity window (spec §15 — "never create
        # permanent approvals"). Only REJECTED/EXPIRED are terminal no-ops.
        if record.status in (
            ApprovalStatus.PENDING.value,
            ApprovalStatus.AWAITING_SECOND_APPROVAL.value,
            ApprovalStatus.APPROVED.value,
        ) and _naive_utc(record.expires_at) < _naive_utc(now):
            record.status = ApprovalStatus.EXPIRED.value
            self._session.add(
                ApprovalEventRecord(
                    approval_id=record.approval_id,
                    event_type="EXPIRED",
                    created_at=now,
                )
            )
            await self._session.flush()

    async def decide(
        self,
        *,
        approval_id: str,
        approver: VerifiedIdentity,
        decision: ApprovalDecision,
        now: dt.datetime | None = None,
    ) -> ApprovalRecord:
        now = now or dt.datetime.now(dt.UTC)
        record = await self._get(approval_id)
        await self._expire_if_needed(record, now)

        if record.status == ApprovalStatus.EXPIRED.value:
            raise NumiError(FailureCode.APPROVAL_EXPIRED, "This approval request has expired.")
        if record.status in (ApprovalStatus.APPROVED.value, ApprovalStatus.REJECTED.value):
            raise NumiError(
                FailureCode.APPROVAL_ALREADY_DECIDED,
                f"This approval was already {record.status.lower()}.",
            )

        if decision == ApprovalDecision.REJECT:
            record.status = ApprovalStatus.REJECTED.value
            record.decided_at = now
            self._session.add(
                ApprovalEventRecord(
                    approval_id=approval_id,
                    event_type="REJECTED",
                    actor_subject_id=approver.subject_id,
                    created_at=now,
                )
            )
            await self._session.flush()
            return record

        # APPROVE
        if record.requires_dual_approval:
            if approver.subject_id == record.actor_subject_id:
                raise NumiError(
                    FailureCode.SEPARATION_OF_DUTIES_VIOLATION,
                    "The requester cannot approve their own critical action.",
                )
            if record.approver_1_subject_id is None:
                record.approver_1_subject_id = approver.subject_id
                record.status = ApprovalStatus.AWAITING_SECOND_APPROVAL.value
                self._session.add(
                    ApprovalEventRecord(
                        approval_id=approval_id,
                        event_type="APPROVER_1_APPROVED",
                        actor_subject_id=approver.subject_id,
                        created_at=now,
                    )
                )
                await self._session.flush()
                return record

            if record.approver_1_subject_id == approver.subject_id:
                raise NumiError(
                    FailureCode.SEPARATION_OF_DUTIES_VIOLATION,
                    "A second, different approver is required for this critical action.",
                )

            record.approver_2_subject_id = approver.subject_id
            record.status = ApprovalStatus.APPROVED.value
            record.decided_at = now
            self._session.add(
                ApprovalEventRecord(
                    approval_id=approval_id,
                    event_type="APPROVER_2_APPROVED",
                    actor_subject_id=approver.subject_id,
                    created_at=now,
                )
            )
            await self._session.flush()
            return record

        record.status = ApprovalStatus.APPROVED.value
        record.approver_1_subject_id = approver.subject_id
        record.decided_at = now
        self._session.add(
            ApprovalEventRecord(
                approval_id=approval_id,
                event_type="APPROVED",
                actor_subject_id=approver.subject_id,
                created_at=now,
            )
        )
        await self._session.flush()
        return record

    async def verify_for_execution(
        self,
        *,
        approval_id: str,
        expected_action_hash: str,
        now: dt.datetime | None = None,
    ) -> ApprovalRecord:
        """Independently re-verifies an approval immediately before execution.

        Never trusts a client-supplied `approved=true` — re-derives
        everything from the persisted record and the freshly recomputed hash
        of the *current* tool-call request.
        """
        now = now or dt.datetime.now(dt.UTC)
        record = await self._get(approval_id)
        await self._expire_if_needed(record, now)

        if record.status == ApprovalStatus.EXPIRED.value:
            raise NumiError(FailureCode.APPROVAL_EXPIRED, "This approval request has expired.")
        if record.status in (
            ApprovalStatus.PENDING.value,
            ApprovalStatus.AWAITING_SECOND_APPROVAL.value,
        ):
            raise NumiError(
                FailureCode.APPROVAL_REQUIRED,
                "This action has not yet been fully approved.",
            )
        if record.status == ApprovalStatus.REJECTED.value:
            raise NumiError(FailureCode.APPROVAL_INVALID, "This action was rejected.")
        if record.status == ApprovalStatus.EXECUTED.value:
            raise NumiError(
                FailureCode.APPROVAL_INVALID,
                "This approval has already been used to execute an action and cannot be reused.",
            )

        if record.action_hash != expected_action_hash:
            raise NumiError(
                FailureCode.APPROVAL_MISMATCH,
                "The action being executed does not match what was approved.",
            )

        return record

    async def mark_executed(self, approval_id: str, now: dt.datetime | None = None) -> None:
        """Consumes the approval so it cannot authorize a second execution
        (spec §43 — duplicate execution / replay)."""
        now = now or dt.datetime.now(dt.UTC)
        record = await self._get(approval_id)
        record.status = ApprovalStatus.EXECUTED.value
        self._session.add(
            ApprovalEventRecord(approval_id=approval_id, event_type="EXECUTED", created_at=now)
        )
        await self._session.flush()

    async def get(self, approval_id: str) -> ApprovalRecord:
        return await self._get(approval_id)
