from __future__ import annotations

import datetime as dt

from numi.common.models.identity import DBARole
from numi.common.models.target import Environment
from numi.gateway.domain.policy_engine import PolicyDecision


def test_fails_closed_for_unlisted_tool(policy_engine, tool_registry, make_ctx):
    tool = tool_registry.get("database.get_health")
    ctx = make_ctx("corebanking-sqlserver-prod")
    # Mutate to an id guaranteed absent from policy.yaml by asking about a
    # nonexistent tool id via the internal table directly.
    evaluation = policy_engine.evaluate(
        environment=Environment.PRODUCTION,
        tool=tool.model_copy(update={"tool_id": "database.totally_unknown_tool"}),
        role=DBARole.DBA_L3,
        ctx=ctx,
    )
    assert evaluation.decision == PolicyDecision.DENY
    assert "no_policy_entry_fail_closed" in evaluation.reasons


def test_dba_l1_cannot_kill_session_in_production(policy_engine, tool_registry, make_ctx):
    tool = tool_registry.get("database.kill_session")
    ctx = make_ctx("corebanking-sqlserver-prod")
    evaluation = policy_engine.evaluate(
        environment=Environment.PRODUCTION, tool=tool, role=DBARole.DBA_L1, ctx=ctx
    )
    assert evaluation.decision == PolicyDecision.DENY


def test_dba_l2_requires_approval_to_kill_session_in_production(
    policy_engine, tool_registry, make_ctx
):
    tool = tool_registry.get("database.kill_session")
    ctx = make_ctx("corebanking-sqlserver-prod")
    evaluation = policy_engine.evaluate(
        environment=Environment.PRODUCTION, tool=tool, role=DBARole.DBA_L2, ctx=ctx
    )
    assert evaluation.decision == PolicyDecision.REQUIRES_APPROVAL


def test_kill_session_requires_approval_even_in_development_for_every_role(
    policy_engine, tool_registry, make_ctx
):
    """Found live: kill_session was ALLOW for every role in development, so
    a real session got silently terminated with zero human confirmation
    step at all -- not even the requester's own approval click. Development
    stays low-friction for read tools and non-disruptive writes
    (update_statistics, cancel_query), but kill_session actually severs a
    real connection, which is disruptive enough to always warrant a pause.
    This is deliberately a single-approval gate, not dual -- kill_session's
    risk_level is MEDIUM (tool_catalog.py), so the requester can still
    approve their own request in one extra click (see
    test_approval.py/ApprovalEngine.approve's requires_dual_approval
    branch) -- this only adds a confirm-before-you-kill-it step, not a
    second-person requirement."""
    tool = tool_registry.get("database.kill_session")
    ctx = make_ctx("postgres-local")
    for role in DBARole:
        evaluation = policy_engine.evaluate(
            environment=Environment.DEVELOPMENT, tool=tool, role=role, ctx=ctx
        )
        assert evaluation.decision == PolicyDecision.REQUIRES_APPROVAL


def test_read_only_tool_allowed_for_all_roles_in_production(policy_engine, tool_registry, make_ctx):
    tool = tool_registry.get("database.get_blocking_sessions")
    ctx = make_ctx("corebanking-sqlserver-prod")
    for role in DBARole:
        evaluation = policy_engine.evaluate(
            environment=Environment.PRODUCTION, tool=tool, role=role, ctx=ctx
        )
        assert evaluation.decision == PolicyDecision.ALLOW


def test_restart_instance_requires_dual_approval(policy_engine, tool_registry, make_ctx):
    tool = tool_registry.get("database.restart_instance")
    ctx = make_ctx("corebanking-sqlserver-prod")
    evaluation = policy_engine.evaluate(
        environment=Environment.DEVELOPMENT,
        tool=tool,
        role=DBARole.DBA_L2,
        ctx=ctx,
    )
    assert evaluation.requires_dual_approval is True


def test_change_ticket_required_flagged_when_missing_in_production(
    policy_engine, tool_registry, make_ctx
):
    tool = tool_registry.get("database.create_index")
    ctx = make_ctx("corebanking-sqlserver-prod")
    evaluation = policy_engine.evaluate(
        environment=Environment.PRODUCTION,
        tool=tool,
        role=DBARole.DBA_L3,
        ctx=ctx,
        change_id=None,
    )
    assert evaluation.requires_change_ticket is True
    assert "change_ticket_required_but_missing" in evaluation.reasons


def test_availability_impacting_write_escalates_outside_maintenance_window(
    policy_engine, tool_registry, make_ctx
):
    tool = tool_registry.get("database.kill_session")
    # sqlserver-dev-01's maintenance window is effectively all-day, so use the
    # uat entry instead, whose window is 22:00-23:59 Africa/Lagos, to force a
    # definitely-outside-window instant.
    uat_ctx = make_ctx("sqlserver-uat-01")
    noon_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
    evaluation = policy_engine.evaluate(
        environment=Environment.UAT,
        tool=tool,
        role=DBARole.DBA_L3,
        ctx=uat_ctx,
        now_utc=noon_utc,
    )
    assert evaluation.in_maintenance_window is False
    # base table already says ALLOW for L3/uat/kill_session; must be escalated
    assert evaluation.decision == PolicyDecision.REQUIRES_APPROVAL
    assert "escalated_outside_maintenance_window" in evaluation.reasons
