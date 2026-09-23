"""Outgoing Teams message delivery via the Bot Framework Connector API."""

from __future__ import annotations

from typing import Any

import httpx

from numi.common.observability import get_logger

logger = get_logger(__name__)


class TeamsMessageSender:
    def __init__(self, get_connector_token):
        # `get_connector_token` is an async callable returning a bearer
        # token for the Bot Framework Connector Service (obtained via an
        # Azure AD client-credentials flow in production); dev deployments
        # pass a stub that returns None.
        self._get_connector_token = get_connector_token

    async def send_reply_to_activity(
        self, service_url: str, conversation_id: str, activity_id: str, card: dict[str, Any]
    ) -> None:
        token = await self._get_connector_token()
        if not token:
            logger.info("teams_message_dev_stub", conversation_id=conversation_id, card=card)
            return
        async with httpx.AsyncClient(base_url=service_url) as client:
            response = await client.post(
                f"/v3/conversations/{conversation_id}/activities/{activity_id}",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "type": "message",
                    "attachments": [
                        {"contentType": "application/vnd.microsoft.card.adaptive", "content": card}
                    ],
                },
            )
            response.raise_for_status()
