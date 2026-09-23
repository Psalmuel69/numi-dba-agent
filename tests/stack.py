"""Shared helper to build the full in-process stack (execution -> gateway ->
agent -> channels) for integration/security tests, wired via
`httpx.ASGITransport` instead of real sockets."""

from __future__ import annotations

import httpx

from numi.agent.api.app import create_app as create_agent_app
from numi.channels.api.app import create_app as create_channels_app
from numi.common.config import Settings
from numi.common.service_auth import ServiceTokenIssuer
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory


def test_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        control_db_url="sqlite+aiosqlite:///:memory:",
        service_jwt_secret="test-secret",
        service_jwt_issuer="numi-internal",
        llm_provider="mock",
        slack_signing_secret="test-slack-signing-secret",
        teams_app_password="dev-teams-shared-token",
    )
    base.update(overrides)
    return Settings(**base)


class Stack:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
        execution_transport = httpx.ASGITransport(app=self.execution_app)
        self.gateway_app = create_gateway_app(settings, execution_transport=execution_transport)
        gateway_transport = httpx.ASGITransport(app=self.gateway_app)
        self.agent_app = create_agent_app(settings, gateway_transport=gateway_transport)
        agent_transport = httpx.ASGITransport(app=self.agent_app)
        self.channels_app = create_channels_app(settings, agent_transport=agent_transport)

        self._issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)

    async def init_db(self) -> None:
        await self.gateway_app.state.gateway.db.create_all()

    def agent_service_token(self) -> str:
        return self._issuer.issue(service_name="agent", audience="numi-gateway")

    def channels_service_token(self) -> str:
        return self._issuer.issue(service_name="channels", audience="numi-agent")

    def forged_token(self, secret: str = "wrong-secret") -> str:
        forged_issuer = ServiceTokenIssuer(secret, self.settings.service_jwt_issuer)
        return forged_issuer.issue(service_name="agent", audience="numi-gateway")


async def build_stack(**settings_overrides) -> Stack:
    stack = Stack(test_settings(**settings_overrides))
    await stack.init_db()
    return stack
