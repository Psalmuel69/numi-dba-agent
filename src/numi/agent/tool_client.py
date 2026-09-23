"""ToolClient (spec §6, §18, §36).

The Agent's *only* route to a tool call is through this client, which talks
exclusively to the Gateway over HTTP with a signed service token. There is
no other network path out of the Agent process toward a database, the
Execution Service, or the Gateway's internals.
"""

from __future__ import annotations

from typing import Any

import httpx

from numi.common.models.decision_event import DecisionEventCreateRequest
from numi.common.models.investigation import (
    InvestigationCreateRequest,
    InvestigationEventCreateRequest,
    InvestigationMemoryEntry,
    InvestigationUpdateRequest,
)
from numi.common.models.tool import ToolCallRequest, ToolCallResponse, ToolDefinition
from numi.common.observability import get_logger
from numi.common.service_auth import ServiceTokenIssuer

logger = get_logger(__name__)

# A bound on the Agent's own wait for one tool-call round trip (Gateway,
# and whatever it takes to the Execution Service and the real database) —
# was 60s. Verified live: a genuinely overloaded database (a session
# holding a query-memory grant for minutes) made even a trivial read-only
# diagnostic exceed that, and since nothing on the Agent side caught the
# resulting httpcore.ReadTimeout, it surfaced as an unhandled 500 instead
# of a clear message — the caller (`orchestrator._submit_and_relay`) is
# what actually degrades that into an AgentReply now, but it can only do
# that once this bound is short enough not to compound into minutes across
# a playbook's several steps. Mirrors `GeminiLLMProvider
# ._REQUEST_TIMEOUT_SECONDS` — same reasoning, other side of the pipeline.
_SUBMIT_TIMEOUT_SECONDS = 15.0


class ToolClient:
    def __init__(
        self,
        base_url: str,
        issuer: ServiceTokenIssuer,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._base_url = base_url
        self._issuer = issuer
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        token = self._issuer.issue(service_name="agent", audience="numi-gateway")
        return {"X-Service-Token": token}

    async def available_tools(
        self, channel: str, channel_account_id: str
    ) -> list[ToolDefinition]:
        async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
            response = await client.get(
                "/v1/tools",
                params={"channel": channel, "channel_account_id": channel_account_id},
                headers=self._headers(),
            )
            response.raise_for_status()
            return [ToolDefinition.model_validate(t) for t in response.json()]

    async def list_servers(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
            response = await client.get("/v1/catalog/servers", headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def get_server_catalog(self, server_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
            response = await client.get(
                f"/v1/catalog/servers/{server_id}", headers=self._headers()
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json()

    async def refresh_catalog(
        self, channel: str, channel_account_id: str, server_id: str | None = None
    ) -> dict[str, Any]:
        path = "/v1/catalog/refresh" + (f"/{server_id}" if server_id else "")
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=300
        ) as client:
            response = await client.post(
                path,
                json={"channel": channel, "channel_account_id": channel_account_id},
                headers=self._headers(),
            )
            if response.status_code >= 400:
                return {"status": "ERROR", "detail": response.json().get("detail", "error")}
            return response.json()

    async def submit(self, request: ToolCallRequest) -> ToolCallResponse:
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=_SUBMIT_TIMEOUT_SECONDS
        ) as client:
            response = await client.post(
                "/v1/tool-calls", json=request.model_dump(mode="json"), headers=self._headers()
            )
            response.raise_for_status()
            return ToolCallResponse.model_validate(response.json())

    async def approve(self, approval_id: str, channel: str, channel_account_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
            response = await client.post(
                f"/v1/approvals/{approval_id}/approve",
                json={"channel": channel, "channel_account_id": channel_account_id},
                headers=self._headers(),
            )
            if response.status_code >= 400:
                return {"status": "ERROR", "detail": response.json().get("detail", "error")}
            return response.json()

    async def reject(self, approval_id: str, channel: str, channel_account_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
            response = await client.post(
                f"/v1/approvals/{approval_id}/reject",
                json={"channel": channel, "channel_account_id": channel_account_id},
                headers=self._headers(),
            )
            if response.status_code >= 400:
                return {"status": "ERROR", "detail": response.json().get("detail", "error")}
            return response.json()

    # -- Investigation persistence/recall -----------------------------------
    # All four below are best-effort by construction: memory is an
    # enhancement to an investigation, never a dependency of it, so a
    # Gateway hiccup here must never surface to the caller as an exception
    # — verified-live precedent for this posture is `alert_trigger.py`'s
    # treatment of its cooldown backend the same way.

    async def create_investigation(self, request: InvestigationCreateRequest) -> None:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
                response = await client.post(
                    "/v1/investigations",
                    json=request.model_dump(mode="json"),
                    headers=self._headers(),
                )
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — never block an investigation on this
            logger.warning(
                "create_investigation_failed",
                investigation_id=request.investigation_id,
                error=str(exc),
            )

    async def update_investigation(
        self, investigation_id: str, request: InvestigationUpdateRequest
    ) -> None:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
                response = await client.patch(
                    f"/v1/investigations/{investigation_id}",
                    json=request.model_dump(mode="json"),
                    headers=self._headers(),
                )
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "update_investigation_failed", investigation_id=investigation_id, error=str(exc)
            )

    async def append_investigation_event(
        self, investigation_id: str, request: InvestigationEventCreateRequest
    ) -> None:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
                response = await client.post(
                    f"/v1/investigations/{investigation_id}/events",
                    json=request.model_dump(mode="json"),
                    headers=self._headers(),
                )
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "append_investigation_event_failed",
                investigation_id=investigation_id,
                error=str(exc),
            )

    async def get_investigation_memory(
        self, server_id: str, *, exclude_investigation_id: str | None = None, limit: int = 3
    ) -> list[InvestigationMemoryEntry]:
        try:
            async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
                response = await client.get(
                    f"/v1/investigations/memory/{server_id}",
                    params={
                        k: v
                        for k, v in {
                            "exclude_investigation_id": exclude_investigation_id,
                            "limit": limit,
                        }.items()
                        if v is not None
                    },
                    headers=self._headers(),
                )
                response.raise_for_status()
                return [InvestigationMemoryEntry.model_validate(e) for e in response.json()]
        except Exception as exc:  # noqa: BLE001 — a lookup failure means "no memory,"
            # not "the investigation can't start."
            logger.warning("get_investigation_memory_failed", server_id=server_id, error=str(exc))
            return []

    async def get_cross_server_patterns(
        self,
        *,
        playbook_id: str,
        environment: str | None = None,
        exclude_server_id: str | None = None,
        limit: int = 5,
    ) -> list[InvestigationMemoryEntry]:
        """Best-effort, same posture as `get_investigation_memory` — a
        lookup failure means "no cross-server pattern," never "the
        investigation can't proceed."""
        try:
            async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
                response = await client.get(
                    "/v1/investigations/correlate",
                    params={
                        k: v
                        for k, v in {
                            "playbook_id": playbook_id,
                            "environment": environment,
                            "exclude_server_id": exclude_server_id,
                            "limit": limit,
                        }.items()
                        if v is not None
                    },
                    headers=self._headers(),
                )
                response.raise_for_status()
                return [InvestigationMemoryEntry.model_validate(e) for e in response.json()]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_cross_server_patterns_failed", playbook_id=playbook_id, error=str(exc)
            )
            return []

    async def log_decision_event(self, request: DecisionEventCreateRequest) -> None:
        """Best-effort, same posture as the four methods above — pure
        telemetry, never worth blocking or failing the caller over. On
        total failure this logs locally so the signal isn't silently lost
        even from local logs (the one place this differs from the others:
        there is no other record of this event at all if both the Gateway
        call and this log line were to vanish)."""
        try:
            async with httpx.AsyncClient(base_url=self._base_url, transport=self._transport) as client:
                response = await client.post(
                    "/v1/decision-events",
                    json=request.model_dump(mode="json"),
                    headers=self._headers(),
                )
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "log_decision_event_failed", event_type=request.event_type, error=str(exc)
            )
