"""The least-privilege finding is actually VISIBLE to a DBA.

The unit tests in `tests/unit/test_least_privilege_check.py` prove each
engine's discovery module produces the finding. That is only half the
feature: a finding nobody ever sees is the same as no finding. This drives
the real `/catalog <server>` chat command through the real Agent
orchestrator against the real Gateway (in-process ASGI, the pattern in
`tests/integration/test_agent_orchestrator.py`) and asserts the warning
reaches the reply text a DBA actually reads.

Discovery itself is not exercised here — the stack's Execution app is built
with the canned adapter factory, so `/v1/discover` deliberately returns an
empty catalog (see `tests/integration/test_catalog_api.py`'s own note).
The catalog carrying the finding is written straight into the Gateway's
catalog store instead, which is exactly what a real discovery run would
have persisted.
"""

from __future__ import annotations

import datetime as dt

import httpx

from numi.agent.context_manager import ContextManager
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.tool_client import ToolClient
from numi.common.config import Settings
from numi.common.models.catalog import (
    DiscoveredDatabase,
    DiscoveredObject,
    LeastPrivilegeFinding,
    ServerCatalog,
)
from numi.common.service_auth import ServiceTokenIssuer
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory

_SERVER_ID = "sqlserver-dev-01"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        control_db_url="sqlite+aiosqlite:///:memory:",
        service_jwt_secret="test-secret",
        service_jwt_issuer="numi-internal",
        llm_provider="mock",
    )


def _catalog(least_privilege: LeastPrivilegeFinding | None) -> ServerCatalog:
    return ServerCatalog(
        server_id=_SERVER_ID,
        discovered_at=dt.datetime.now(dt.UTC),
        engine_version="16.0.1000",
        engine_edition="Developer Edition",
        databases=[
            DiscoveredDatabase(
                name="CoreBanking",
                state="online",
                size_bytes=48_213_000_000,
                objects=[DiscoveredObject(schema_name="dbo", name="Accounts", kind="table")],
            )
        ],
        least_privilege=least_privilege,
    )


async def _build(least_privilege: LeastPrivilegeFinding | None) -> AgentOrchestrator:
    settings = _settings()
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    gateway_app = create_gateway_app(
        settings, execution_transport=httpx.ASGITransport(app=execution_app)
    )
    # ASGITransport emits no lifespan events, so build the schema explicitly.
    await gateway_app.state.gateway.db.create_all()
    await gateway_app.state.gateway.catalog_store.put(_catalog(least_privilege))

    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    tool_client = ToolClient(
        settings.gateway_base_url,
        issuer,
        transport=httpx.ASGITransport(app=gateway_app),
    )
    return AgentOrchestrator(
        LLMRegistry.for_testing(MockLLMProvider()), tool_client, ContextManager()
    )


async def _catalog_reply(orchestrator: AgentOrchestrator) -> str:
    reply = await orchestrator.handle_message(
        conversation_id="conv-least-privilege",
        channel_thread_id="thread-least-privilege",
        channel="slack",
        channel_account_id="U_MOCK_L2",
        message=f"/catalog {_SERVER_ID}",
    )
    return reply.text


async def test_an_over_privileged_login_is_surfaced_through_the_catalog_command():
    orchestrator = await _build(
        LeastPrivilegeFinding(
            checked=True,
            login="numi_diag",
            has_user_table_select=True,
            granted_object_count=12,
            sample_objects=["dbo.Accounts", "dbo.Customers"],
            scope_note="checked against the 'CoreBanking' database",
        )
    )

    text = await _catalog_reply(orchestrator)

    # The DBA sees the warning plainly, with the number and where to look.
    assert "⚠️" in text
    assert "12 user table/views" in text
    assert "numi_diag" in text
    assert "should be revoked for least-privilege" in text
    assert "dbo.Accounts" in text
    # ...and the honest scope caveat travels with it.
    assert "checked against the 'CoreBanking' database" in text
    # The ordinary catalog content is still there — the warning is added, not
    # substituted for what /catalog already showed.
    assert "CoreBanking" in text
    assert "Developer Edition" in text


async def test_a_properly_scoped_login_adds_no_warning_noise():
    orchestrator = await _build(
        LeastPrivilegeFinding(
            checked=True,
            login="numi_diag",
            has_user_table_select=False,
            granted_object_count=0,
            scope_note="checked against the 'CoreBanking' database",
        )
    )

    text = await _catalog_reply(orchestrator)

    assert "⚠️" not in text
    assert "least-privilege" not in text
    assert "CoreBanking" in text  # the catalog itself still renders


async def test_a_check_that_never_ran_makes_no_claim_either_way():
    """`checked=False` must not render as a clean bill of health."""
    orchestrator = await _build(
        LeastPrivilegeFinding(checked=False, scope_note="privilege check could not run")
    )

    text = await _catalog_reply(orchestrator)

    assert "⚠️" not in text
    assert "least-privilege" not in text


async def test_a_catalog_predating_this_feature_still_renders():
    """A catalog discovered before the finding existed round-trips as
    `least_privilege=None` and must not break /catalog."""
    orchestrator = await _build(None)

    text = await _catalog_reply(orchestrator)

    assert "CoreBanking" in text
    assert "⚠️" not in text
