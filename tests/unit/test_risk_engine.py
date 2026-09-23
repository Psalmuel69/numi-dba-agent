from __future__ import annotations

from numi.common.models.risk import BlastRadius, ReasonCode, RiskLevel
from numi.common.models.target import Environment
from numi.gateway.domain.risk_engine import RiskEngine


def test_read_tool_on_critical_prod_db_is_low_risk(tool_registry, make_ctx):
    engine = RiskEngine()
    tool = tool_registry.get("database.get_health")
    ctx = make_ctx("corebanking-sqlserver-prod")
    risk = engine.assess(tool=tool, environment=Environment.PRODUCTION, ctx=ctx)
    assert risk.risk_level == RiskLevel.LOW
    assert risk.blast_radius == BlastRadius.SINGLE_OBJECT
    assert ReasonCode.READ_OPERATION in risk.reason_codes


def test_kill_session_on_critical_prod_db_is_at_least_medium(tool_registry, make_ctx):
    engine = RiskEngine()
    tool = tool_registry.get("database.kill_session")
    ctx = make_ctx("corebanking-sqlserver-prod")
    risk = engine.assess(tool=tool, environment=Environment.PRODUCTION, ctx=ctx)
    assert risk.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert risk.blast_radius == BlastRadius.SINGLE_SESSION
    assert risk.reversible is False
    assert ReasonCode.CRITICAL_DATABASE in risk.reason_codes


def test_failover_is_always_critical(tool_registry, make_ctx):
    engine = RiskEngine()
    tool = tool_registry.get("database.failover")
    ctx = make_ctx("sqlserver-dev-01")  # even a low-criticality dev server
    risk = engine.assess(tool=tool, environment=Environment.DEVELOPMENT, ctx=ctx)
    assert risk.risk_level == RiskLevel.CRITICAL
    assert risk.blast_radius == BlastRadius.CLUSTER


def test_risk_never_reported_below_tool_floor(tool_registry, make_ctx):
    engine = RiskEngine()
    tool = tool_registry.get("database.restart_instance")
    ctx = make_ctx("sqlserver-dev-01")
    risk = engine.assess(tool=tool, environment=Environment.DEVELOPMENT, ctx=ctx)
    assert risk.risk_level == RiskLevel.CRITICAL  # tool floor is CRITICAL
