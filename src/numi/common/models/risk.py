"""Risk and blast-radius vocabulary (spec §13, §14).

Computed exclusively by the Gateway's Risk Engine. The LLM may *echo* a risk
assessment back to the user in prose, but it never produces the
authoritative `RiskAssessment` — that always comes from
`gateway.domain.risk_engine`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BlastRadius(str, Enum):
    SINGLE_QUERY = "SINGLE_QUERY"
    SINGLE_SESSION = "SINGLE_SESSION"
    SINGLE_OBJECT = "SINGLE_OBJECT"
    SINGLE_DATABASE = "SINGLE_DATABASE"
    MULTIPLE_OBJECTS = "MULTIPLE_OBJECTS"
    MULTIPLE_DATABASES = "MULTIPLE_DATABASES"
    CLUSTER = "CLUSTER"
    ENTERPRISE = "ENTERPRISE"


class ReasonCode(str, Enum):
    PRODUCTION = "PRODUCTION"
    NON_PRODUCTION = "NON_PRODUCTION"
    CRITICAL_DATABASE = "CRITICAL_DATABASE"
    HIGH_CRITICALITY_DATABASE = "HIGH_CRITICALITY_DATABASE"
    WRITE_OPERATION = "WRITE_OPERATION"
    READ_OPERATION = "READ_OPERATION"
    IRREVERSIBLE = "IRREVERSIBLE"
    AVAILABILITY_IMPACT = "AVAILABILITY_IMPACT"
    OUTSIDE_MAINTENANCE_WINDOW = "OUTSIDE_MAINTENANCE_WINDOW"
    INSIDE_MAINTENANCE_WINDOW = "INSIDE_MAINTENANCE_WINDOW"
    HIGH_CURRENT_LOAD = "HIGH_CURRENT_LOAD"
    LARGE_BLAST_RADIUS = "LARGE_BLAST_RADIUS"
    MANY_AFFECTED_SESSIONS = "MANY_AFFECTED_SESSIONS"
    MANY_AFFECTED_OBJECTS = "MANY_AFFECTED_OBJECTS"
    PRIVILEGED_OPERATION = "PRIVILEGED_OPERATION"


class RiskAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk_level: RiskLevel
    score: int  # 0-100
    blast_radius: BlastRadius
    reversible: bool
    availability_impact: bool
    reason_codes: list[ReasonCode]
