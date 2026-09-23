"""What the Agent hands back to a channel adapter — never a raw LLM string.

`approval_card` is populated only when the Gateway itself returned
APPROVAL_REQUIRED; the Agent cannot manufacture one on its own, since it has
no authority to decide that an approval is needed in the first place.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ApprovalCard(BaseModel):
    approval_id: str
    tool_id: str
    target_summary: str
    reason: str
    risk_level: str
    blast_radius: str
    expires_in_seconds: int = 600


class AgentReply(BaseModel):
    text: str
    status: str = "ok"  # ok | approval_required | denied | error | clarification
    approval_card: ApprovalCard | None = None
    investigation_id: str | None = None
    result_data: dict[str, Any] | None = None
    # True only for a reply to an approve/reject decision where the
    # approval is STILL open afterward -- a separation-of-duties (or other)
    # rejection that leaves the request awaiting a *different* approver, or
    # one leg of a dual-approval requirement still needing a second one.
    # False (the default) covers both "not an approval decision at all" and
    # "this approval is now genuinely closed" (rejected, or fully approved
    # and resubmitted -- see `AgentOrchestrator.handle_approval_decision`,
    # the only place this is ever set True). A channel adapter uses this,
    # not `status`, to decide whether an approval card's buttons may be
    # collapsed: `status == "error"` alone can't tell "this specific
    # identity can't approve their own request, try someone else" apart
    # from "this approval is dead, don't bother" -- collapsing the card on
    # the former would hide it from the DBA who actually can still act on
    # it.
    approval_still_pending: bool = False
