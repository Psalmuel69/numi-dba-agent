"""Tool Registry (spec §8, §36).

Holds the versioned catalog of every tool the platform knows about and
answers two questions:
  1. "does this tool_id/version exist and is it enabled?" (used by every
     tool call, always — the actual security boundary)
  2. "which tools could this role even attempt?" (used only to shape what
     the Agent is offered — a UX/convenience filter, NOT a security control;
     spec §36 is explicit that Gateway enforcement is what matters)
"""

from __future__ import annotations

from numi.common.config import Settings
from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.identity import DBARole
from numi.common.models.tool import ToolDefinition
from numi.gateway.domain.tool_catalog import build_tool_catalog


class ToolRegistry:
    def __init__(self, settings: Settings):
        self._tools: dict[str, ToolDefinition] = {
            t.tool_id: t for t in build_tool_catalog(settings)
        }

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get(self, tool_id: str, tool_version: str | None = None) -> ToolDefinition:
        tool = self._tools.get(tool_id)
        if tool is None:
            raise NumiError(FailureCode.TOOL_NOT_FOUND, f"Unknown tool '{tool_id}'.")
        if tool_version is not None and tool_version != tool.version:
            raise NumiError(
                FailureCode.TOOL_NOT_FOUND,
                f"Tool '{tool_id}' version '{tool_version}' not found (latest is '{tool.version}').",
            )
        if not tool.enabled:
            raise NumiError(
                FailureCode.TOOL_NOT_AVAILABLE,
                f"Tool '{tool_id}' is disabled by platform configuration.",
            )
        return tool

    def tools_for_role(self, role: DBARole) -> list[ToolDefinition]:
        """Convenience filter only — see module docstring."""
        return [t for t in self._tools.values() if t.enabled and role in t.allowed_roles]
