"""Policy Engine (spec §12, §40, §41, §63).

Independent of, and unmodifiable by, the LLM. Loaded once from
`config/policy.yaml` at process start; nothing at runtime — not a chat
message, not an agent claim, not tool arguments — can change a policy
decision except the explicit inputs this module is given by the Gateway
orchestrator (environment, tool, role, maintenance-window state, change
ticket presence, current load).

Fails closed: any (environment, tool, role) combination absent from
configuration evaluates to DENY (`default_decision`), never ALLOW.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from numi.common.models.identity import DBARole
from numi.common.models.target import Environment
from numi.common.models.tool import ToolDefinition
from numi.gateway.domain.target_validation import TargetContext


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


@dataclass(frozen=True)
class PolicyEvaluation:
    decision: PolicyDecision
    requires_dual_approval: bool
    requires_change_ticket: bool
    in_maintenance_window: bool
    reasons: list[str]


def is_within_maintenance_window(
    window: dict | None, now_utc: dt.datetime | None = None
) -> bool:
    window = window or {}
    if not window.get("start") or not window.get("end"):
        return True  # no window configured -> not restricted by one
    now_utc = now_utc or dt.datetime.now(dt.UTC)
    tz: dt.tzinfo
    try:
        tz = ZoneInfo(window.get("timezone", "UTC"))
    except Exception:
        tz = dt.UTC
    local_now = now_utc.astimezone(tz)
    start_h, start_m = (int(x) for x in window["start"].split(":"))
    end_h, end_m = (int(x) for x in window["end"].split(":"))
    start = local_now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = local_now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if start <= end:
        return start <= local_now <= end
    # window wraps midnight
    return local_now >= start or local_now <= end


class PolicyEngine:
    def __init__(self, config_path: str | Path):
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self._default_decision = PolicyDecision(raw.get("default_decision", "DENY"))
        self._table: dict[str, dict[str, dict[str, PolicyDecision]]] = {
            env: {
                tool_id: {role: PolicyDecision(decision) for role, decision in role_map.items()}
                for tool_id, role_map in tool_map.items()
            }
            for env, tool_map in raw.get("environments", {}).items()
        }
        self._change_ticket_required: dict[str, set[str]] = {
            env: set(tool_ids) for env, tool_ids in raw.get("change_ticket_required", {}).items()
        }
        self._dual_approval_required: set[str] = set(raw.get("dual_approval_required", []))

    def evaluate(
        self,
        *,
        environment: Environment,
        tool: ToolDefinition,
        role: DBARole,
        ctx: TargetContext,
        change_id: str | None = None,
        current_load_critical: bool = False,
        now_utc: dt.datetime | None = None,
    ) -> PolicyEvaluation:
        reasons: list[str] = []

        env_table = self._table.get(environment.value, {})
        tool_table = env_table.get(tool.tool_id)
        if tool_table is None:
            decision = self._default_decision
            reasons.append("no_policy_entry_fail_closed")
        else:
            decision = tool_table.get(role.value, self._default_decision)
            if role.value not in tool_table:
                reasons.append("no_role_entry_fail_closed")

        in_window = is_within_maintenance_window(ctx.maintenance_window, now_utc)
        if tool.availability_impact:
            reasons.append(
                "inside_maintenance_window" if in_window else "outside_maintenance_window"
            )
            # An availability-impacting change proposed outside the declared
            # maintenance window is never auto-approved, even if the base
            # table said ALLOW.
            if decision == PolicyDecision.ALLOW and not in_window:
                decision = PolicyDecision.REQUIRES_APPROVAL
                reasons.append("escalated_outside_maintenance_window")

        if current_load_critical and tool.data_modification and decision == PolicyDecision.ALLOW:
            decision = PolicyDecision.REQUIRES_APPROVAL
            reasons.append("escalated_current_load_critical")

        requires_change_ticket = tool.tool_id in self._change_ticket_required.get(
            environment.value, set()
        )
        if requires_change_ticket and not change_id and decision != PolicyDecision.DENY:
            reasons.append("change_ticket_required_but_missing")

        requires_dual_approval = (
            tool.requires_dual_approval or tool.tool_id in self._dual_approval_required
        )

        return PolicyEvaluation(
            decision=decision,
            requires_dual_approval=requires_dual_approval,
            requires_change_ticket=requires_change_ticket,
            in_maintenance_window=in_window,
            reasons=reasons,
        )
