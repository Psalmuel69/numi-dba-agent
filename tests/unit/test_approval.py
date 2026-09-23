from __future__ import annotations

import datetime as dt

import pytest

from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.risk import BlastRadius, RiskAssessment, RiskLevel
from numi.common.models.target import DatabaseTarget, Environment
from numi.gateway.domain.approval import (
    ApprovalContext,
    ApprovalDecision,
    ApprovalEngine,
    compute_action_hash,
)


def _risk() -> RiskAssessment:
    return RiskAssessment(
        risk_level=RiskLevel.MEDIUM,
        score=50,
        blast_radius=BlastRadius.SINGLE_SESSION,
        reversible=False,
        availability_impact=True,
        reason_codes=[],
    )


def _ctx(tool_registry, identity, session_id: str) -> ApprovalContext:
    tool = tool_registry.get("database.kill_session")
    target = DatabaseTarget(
        environment=Environment.PRODUCTION,
        instance="corebanking-prd-01",
        database="CoreBanking",
        session_id=session_id,
    )
    return ApprovalContext(
        request_id="req_test",
        actor=identity,
        tool=tool,
        target=target,
        normalized_arguments={"session_id": session_id, "reason": "blocking chain"},
        environment="production",
        database_id="corebanking-prd-01",
        risk=_risk(),
    )


@pytest.mark.asyncio
async def test_approval_mismatch_when_action_changes_after_approval(
    db, tool_registry, identity_provider
):
    """Spec §44: agent requests kill session 9182, gets approved, then
    attempts to execute against session 9183 instead. Must be denied with
    APPROVAL_MISMATCH."""
    approver = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L3")

    async with db.session() as session:
        engine = ApprovalEngine(session)
        ctx_9182 = _ctx(tool_registry, approver, "9182")
        record = await engine.create(ctx_9182, requires_dual_approval=False)
        await engine.decide(
            approval_id=record.approval_id, approver=approver, decision=ApprovalDecision.APPROVE
        )

        # Attacker/agent now tries to execute against a DIFFERENT session id,
        # reusing the same approval_id.
        ctx_9183 = _ctx(tool_registry, approver, "9183")
        with pytest.raises(NumiError) as exc:
            await engine.verify_for_execution(
                approval_id=record.approval_id,
                expected_action_hash=ctx_9183.action_hash(),
            )
        assert exc.value.code == FailureCode.APPROVAL_MISMATCH


@pytest.mark.asyncio
async def test_approval_expired_denies_execution(db, tool_registry, identity_provider):
    """Spec §44: approval expires before execution is attempted."""
    approver = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L3")

    async with db.session() as session:
        engine = ApprovalEngine(session, default_ttl_seconds=1)
        ctx = _ctx(tool_registry, approver, "9182")
        record = await engine.create(ctx, requires_dual_approval=False)
        await engine.decide(
            approval_id=record.approval_id, approver=approver, decision=ApprovalDecision.APPROVE
        )

        later = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=120)
        with pytest.raises(NumiError) as exc:
            await engine.verify_for_execution(
                approval_id=record.approval_id,
                expected_action_hash=ctx.action_hash(),
                now=later,
            )
        assert exc.value.code == FailureCode.APPROVAL_EXPIRED


@pytest.mark.asyncio
async def test_matching_action_hash_passes_verification(db, tool_registry, identity_provider):
    approver = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L3")

    async with db.session() as session:
        engine = ApprovalEngine(session)
        ctx = _ctx(tool_registry, approver, "9182")
        record = await engine.create(ctx, requires_dual_approval=False)
        await engine.decide(
            approval_id=record.approval_id, approver=approver, decision=ApprovalDecision.APPROVE
        )
        verified = await engine.verify_for_execution(
            approval_id=record.approval_id, expected_action_hash=ctx.action_hash()
        )
        assert verified.status == "APPROVED"


@pytest.mark.asyncio
async def test_client_side_approved_true_is_never_trusted(db, tool_registry, identity_provider):
    """There is no code path that accepts a bare boolean — execution always
    re-derives approval state from the persisted record via approval_id."""
    approver = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L3")
    async with db.session() as session:
        engine = ApprovalEngine(session)
        ctx = _ctx(tool_registry, approver, "9182")
        record = await engine.create(ctx, requires_dual_approval=False)
        # Never approved.
        with pytest.raises(NumiError) as exc:
            await engine.verify_for_execution(
                approval_id=record.approval_id, expected_action_hash=ctx.action_hash()
            )
        assert exc.value.code == FailureCode.APPROVAL_REQUIRED


@pytest.mark.asyncio
async def test_requester_cannot_self_approve_dual_approval_action(
    db, tool_registry, identity_provider
):
    requester = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L3")
    async with db.session() as session:
        engine = ApprovalEngine(session)
        ctx = _ctx(tool_registry, requester, "9182")
        record = await engine.create(ctx, requires_dual_approval=True)
        with pytest.raises(NumiError) as exc:
            await engine.decide(
                approval_id=record.approval_id,
                approver=requester,
                decision=ApprovalDecision.APPROVE,
            )
        assert exc.value.code == FailureCode.SEPARATION_OF_DUTIES_VIOLATION


@pytest.mark.asyncio
async def test_dual_approval_requires_two_distinct_approvers(
    db, tool_registry, identity_provider
):
    requester = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L2")
    approver1 = await identity_provider.resolve_by_external_account("slack", "U_MOCK_L3")
    approver2 = await identity_provider.resolve_by_external_account("slack", "U_MOCK_MGR")

    async with db.session() as session:
        engine = ApprovalEngine(session)
        ctx = _ctx(tool_registry, requester, "9182")
        record = await engine.create(ctx, requires_dual_approval=True)

        await engine.decide(
            approval_id=record.approval_id, approver=approver1, decision=ApprovalDecision.APPROVE
        )
        # Not fully approved yet — execution must still be denied.
        with pytest.raises(NumiError) as exc:
            await engine.verify_for_execution(
                approval_id=record.approval_id, expected_action_hash=ctx.action_hash()
            )
        assert exc.value.code == FailureCode.APPROVAL_REQUIRED

        # Same approver trying again must not count as the second approver.
        with pytest.raises(NumiError) as exc:
            await engine.decide(
                approval_id=record.approval_id,
                approver=approver1,
                decision=ApprovalDecision.APPROVE,
            )
        assert exc.value.code == FailureCode.SEPARATION_OF_DUTIES_VIOLATION

        await engine.decide(
            approval_id=record.approval_id, approver=approver2, decision=ApprovalDecision.APPROVE
        )
        verified = await engine.verify_for_execution(
            approval_id=record.approval_id, expected_action_hash=ctx.action_hash()
        )
        assert verified.status == "APPROVED"


def test_action_hash_changes_if_any_material_field_changes():
    base = dict(
        actor_subject_id="u1",
        tool_id="database.kill_session",
        tool_version="1.0.0",
        target={"session_id": "9182"},
        normalized_arguments={"session_id": "9182"},
        environment="production",
        database_id="corebanking-prd-01",
        risk_level="MEDIUM",
    )
    h1 = compute_action_hash(**base)
    changed = dict(base, target={"session_id": "9183"})
    h2 = compute_action_hash(**changed)
    assert h1 != h2
