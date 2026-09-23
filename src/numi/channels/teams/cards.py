"""Microsoft Teams Adaptive Card rendering for Agent replies (spec §16)."""

from __future__ import annotations

from typing import Any

from numi.agent.reply import AgentReply


def render_reply_card(reply: AgentReply) -> dict[str, Any]:
    body: list[dict[str, Any]] = [{"type": "TextBlock", "text": reply.text, "wrap": True}]
    actions: list[dict[str, Any]] = []

    if reply.approval_card is not None:
        card = reply.approval_card
        body.append({"type": "TextBlock", "text": "AI DBA ACTION REQUIRES APPROVAL", "weight": "bolder"})
        body.append(
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Operation", "value": card.tool_id},
                    {"title": "Target", "value": card.target_summary},
                    {"title": "Reason", "value": card.reason},
                    {"title": "Risk", "value": card.risk_level},
                    {"title": "Blast radius", "value": card.blast_radius},
                    {"title": "Expires in", "value": f"{card.expires_in_seconds // 60} minutes"},
                ],
            }
        )
        actions = [
            {
                "type": "Action.Submit",
                "title": "Approve",
                "data": {"numi_action": "approve", "approval_id": card.approval_id},
            },
            {
                "type": "Action.Submit",
                "title": "Reject",
                "data": {"numi_action": "reject", "approval_id": card.approval_id},
            },
        ]

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
        "actions": actions,
    }
