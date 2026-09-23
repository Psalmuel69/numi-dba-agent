"""Slack Block Kit rendering for Agent replies (spec §16).

Purely presentational — the `action_id`/`value` encoding here carries an
`approval_id` for the button click handler to forward, but clicking a
button is never itself trusted as an approval; see
`channels.api.app`'s interactive-action handler, which re-verifies
everything through the Agent -> Gateway path exactly like any other request.
"""

from __future__ import annotations

from typing import Any

from numi.agent.reply import AgentReply


def render_reply_blocks(reply: AgentReply) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": reply.text}}
    ]
    if reply.approval_card is not None:
        card = reply.approval_card
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*AI DBA ACTION REQUIRES APPROVAL*\n"
                        f"*Operation:* `{card.tool_id}`\n"
                        f"*Target:* {card.target_summary}\n"
                        f"*Reason:* {card.reason}\n"
                        f"*Risk:* {card.risk_level}\n"
                        f"*Blast radius:* {card.blast_radius}\n"
                        f"*Expires in:* {card.expires_in_seconds // 60} minutes"
                    ),
                },
            }
        )
        blocks.append(
            {
                "type": "actions",
                "block_id": f"numi_approval_{card.approval_id}",
                "elements": [
                    {
                        "type": "button",
                        "style": "primary",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "action_id": "numi_approve",
                        "value": card.approval_id,
                    },
                    {
                        "type": "button",
                        "style": "danger",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "action_id": "numi_reject",
                        "value": card.approval_id,
                    },
                ],
            }
        )
    return blocks


def resolve_approval_blocks(
    original_blocks: list[dict[str, Any]], approval_id: str, decision: str, decided_by: str
) -> list[dict[str, Any]]:
    """Rebuild a posted approval card's blocks with its `actions` block (the
    Approve/Reject buttons) replaced by a static resolved line — used to
    collapse both buttons the instant either is clicked, via `chat.update`,
    rather than leaving a resolved card sitting there fully clickable
    forever. Slack buttons have no "disabled but still visible" state to
    toggle; swapping the interactive block for a plain one (same pattern
    real Slack apps — GitHub, PagerDuty, ...— use for this) is what actually
    removes them. Matches on `block_id` (`numi_approval_{approval_id}`, set
    when the card was first rendered above) rather than block position or
    type, so it only ever touches the one block that was this specific
    card's own buttons — never a coincidentally-shaped block belonging to
    a different message. `original_blocks` is Slack's own echo of the
    message being acted on (`payload["message"]["blocks"]`), not anything
    reconstructed from the Agent's reply, since the reply for an
    approve/reject *decision* carries no approval_card of its own (it's
    already resolved by the time this handler gets it back)."""
    icon = "✅" if decision == "approve" else "❌"
    label = "Approved" if decision == "approve" else "Rejected"
    resolved_block_id = f"numi_approval_{approval_id}"
    rebuilt: list[dict[str, Any]] = []
    for block in original_blocks:
        if block.get("block_id") == resolved_block_id:
            rebuilt.append(
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"{icon} *{label}* by {decided_by}"}
                    ],
                }
            )
        else:
            rebuilt.append(block)
    return rebuilt
