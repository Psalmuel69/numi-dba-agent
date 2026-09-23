"""Risk Engine (spec §13, §14).

Computes the authoritative `RiskAssessment` for a proposed tool call. The
LLM may narrate risk in prose to the user, but this is the only place a
`RiskAssessment` object is actually produced, and the Policy/Approval
engines act on this output, never on anything the Agent says about risk.
"""

from __future__ import annotations

from numi.common.models.risk import BlastRadius, ReasonCode, RiskAssessment, RiskLevel
from numi.common.models.target import Environment
from numi.common.models.tool import ToolDefinition
from numi.gateway.domain.target_validation import TargetContext

_BASE_SCORE = {
    RiskLevel.LOW: 10,
    RiskLevel.MEDIUM: 35,
    RiskLevel.HIGH: 60,
    RiskLevel.CRITICAL: 85,
}

_TOOL_BLAST_RADIUS: dict[str, BlastRadius] = {
    "database.cancel_query": BlastRadius.SINGLE_QUERY,
    "database.kill_session": BlastRadius.SINGLE_SESSION,
    "database.update_statistics": BlastRadius.SINGLE_OBJECT,
    "database.create_index": BlastRadius.SINGLE_OBJECT,
    "database.rebuild_index": BlastRadius.SINGLE_OBJECT,
    "database.modify_configuration": BlastRadius.SINGLE_DATABASE,
    "database.restart_instance": BlastRadius.SINGLE_DATABASE,
    "database.failover": BlastRadius.CLUSTER,
    "database.execute_sql": BlastRadius.SINGLE_OBJECT,
    "database.execute_readonly_sql": BlastRadius.SINGLE_QUERY,
    "database.restore_database": BlastRadius.SINGLE_DATABASE,
    "database.create_database": BlastRadius.SINGLE_DATABASE,
    "database.drop_database": BlastRadius.SINGLE_DATABASE,
    "database.truncate_table": BlastRadius.SINGLE_OBJECT,
    "database.bulk_delete": BlastRadius.SINGLE_OBJECT,
}

_CRITICALITY_POINTS = {"critical": 20, "high": 12, "standard": 4, "low": 0}


class RiskEngine:
    def assess(
        self,
        *,
        tool: ToolDefinition,
        environment: Environment,
        ctx: TargetContext,
        affected_object_count: int = 1,
        affected_session_count: int = 1,
        current_load_critical: bool = False,
    ) -> RiskAssessment:
        reasons: list[ReasonCode] = []
        score = _BASE_SCORE[RiskLevel(tool.risk_level)]

        # Environment/criticality materially raise risk only for operations
        # that can actually change state or availability. A pure read against
        # a critical production database carries essentially the same risk
        # as the same read anywhere else — it is the blast radius of a
        # *write* that production/criticality should amplify.
        impactful = tool.data_modification or tool.availability_impact

        if environment == Environment.PRODUCTION:
            reasons.append(ReasonCode.PRODUCTION)
            if impactful:
                score += 15
        else:
            reasons.append(ReasonCode.NON_PRODUCTION)

        if impactful:
            score += _CRITICALITY_POINTS.get(ctx.criticality, 0)
        if ctx.criticality == "critical":
            reasons.append(ReasonCode.CRITICAL_DATABASE)
        elif ctx.criticality == "high":
            reasons.append(ReasonCode.HIGH_CRITICALITY_DATABASE)

        if tool.data_modification:
            score += 10
            reasons.append(ReasonCode.WRITE_OPERATION)
        else:
            reasons.append(ReasonCode.READ_OPERATION)

        if not tool.reversible:
            score += 15
            reasons.append(ReasonCode.IRREVERSIBLE)

        if tool.availability_impact:
            score += 10
            reasons.append(ReasonCode.AVAILABILITY_IMPACT)

        blast_radius = _TOOL_BLAST_RADIUS.get(tool.tool_id, BlastRadius.SINGLE_OBJECT)
        if affected_object_count > 1 and blast_radius in (
            BlastRadius.SINGLE_OBJECT,
            BlastRadius.SINGLE_DATABASE,
        ):
            blast_radius = BlastRadius.MULTIPLE_OBJECTS
            reasons.append(ReasonCode.MANY_AFFECTED_OBJECTS)
        if affected_session_count > 10:
            reasons.append(ReasonCode.MANY_AFFECTED_SESSIONS)
            score += 5

        if current_load_critical:
            score += 10
            reasons.append(ReasonCode.HIGH_CURRENT_LOAD)

        if tool.operation_type.value == "PRIVILEGED":
            reasons.append(ReasonCode.PRIVILEGED_OPERATION)

        score = max(0, min(100, score))

        if score >= 85:
            level = RiskLevel.CRITICAL
        elif score >= 60:
            level = RiskLevel.HIGH
        elif score >= 30:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        # Never report a risk level lower than the tool's own declared floor.
        floor = RiskLevel(tool.risk_level)
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        if order.index(level) < order.index(floor):
            level = floor

        if blast_radius in (BlastRadius.CLUSTER, BlastRadius.ENTERPRISE, BlastRadius.MULTIPLE_DATABASES):
            reasons.append(ReasonCode.LARGE_BLAST_RADIUS)

        return RiskAssessment(
            risk_level=level,
            score=score,
            blast_radius=blast_radius,
            reversible=tool.reversible,
            availability_impact=tool.availability_impact,
            reason_codes=reasons,
        )
