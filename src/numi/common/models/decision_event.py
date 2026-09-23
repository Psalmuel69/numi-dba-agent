"""Decision-quality event wire contract — the Agent posts these best-effort
to the Gateway (see `agent.tool_client.ToolClient.log_decision_event`)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionEventCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    conversation_id: str | None = None
    investigation_id: str | None = None
    provider: str = ""
    model: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
