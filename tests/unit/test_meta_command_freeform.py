"""Reproduces a live finding: "List all my servers" (sent as an ordinary
Slack message, not the literal `/servers` command) was treated as a fresh
investigation request and hit the environment-clarification gate for no
reason — the DBA should never have to know or use the exact slash syntax.
`IntentExtraction.meta_command` recognizes these as a request for one of
the agent's own utility actions, and handle_message routes it to the same
handler the exact slash command already uses."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager, PendingApproval
from numi.agent.llm.mock import MockLLMProvider
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import IntentExtraction


class _FakeToolClient:
    def __init__(self, servers=None):
        self._servers = servers or []

    async def list_servers(self):
        return self._servers

    async def available_tools(self, channel, channel_account_id):
        return []

    async def get_server_catalog(self, server_id):
        return {"server": {"id": server_id}, "catalog": None}

    async def refresh_catalog(self, channel, channel_account_id, server_id=None):
        return {"status": "OK", "server_id": server_id}


class _FakeLLM:
    def __init__(self, intent: IntentExtraction):
        self._intent = intent

    async def extract_intent(self, *args, **kwargs):
        return self._intent


def _orchestrator(llm, servers=None) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm),
        tool_client=_FakeToolClient(servers),
        context=ContextManager(),
    )


@pytest.mark.asyncio
async def test_a_freeform_servers_request_lists_servers_not_a_fresh_investigation():
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="servers"))
    servers = [
        {
            "id": "postgres-local",
            "environment": "development",
            "platform": "postgresql",
            "criticality": "standard",
        }
    ]
    orchestrator = _orchestrator(llm, servers=servers)

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv1",
        channel_thread_id="",
        message="List all my servers",
    )

    assert reply.status != "clarification"
    assert "postgres-local" in reply.text


@pytest.mark.asyncio
async def test_a_freeform_playbooks_request_lists_playbooks():
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="playbooks"))
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv2",
        channel_thread_id="",
        message="what playbooks do you have",
    )

    assert "Slow Query Investigation" in reply.text


@pytest.mark.asyncio
async def test_a_freeform_discover_request_names_the_instance_hint_as_the_target():
    llm = _FakeLLM(
        IntentExtraction(is_dba_task=False, meta_command="discover", instance_hint="postgres-local")
    )
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv3", channel_thread_id="",
        message="please run discovery on postgres-local",
    )

    assert "Discovery complete" in reply.text
    assert "postgres-local" in reply.text


@pytest.mark.asyncio
async def test_a_freeform_discover_all_request_formats_clean_lines_not_a_dict_repr():
    """Live-reproduced finding: `/discover` with no target refreshes every
    registered server, and the reply used to be a raw Python dict repr
    (`{'sqlserver-dev-01': 'ok (12 databases)', ...}`) dumped straight into
    the message text. It must instead be one clean, readable line per
    server — matching how `_handle_servers_command` already formats its own
    per-server listing."""

    class _RefreshAllToolClient(_FakeToolClient):
        async def refresh_catalog(self, channel, channel_account_id, server_id=None):
            assert server_id is None
            return {
                "sqlserver-dev-01": "ok (12 databases)",
                "sqlserver-uat-01": "failed: execution service returned an error (status 500)",
            }

    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="discover", instance_hint=None))
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm),
        tool_client=_RefreshAllToolClient(),
        context=ContextManager(),
    )

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv3b", channel_thread_id="",
        message="run discovery on everything",
    )

    # Never a raw dict repr (Python's own str() of a dict).
    assert "{'sqlserver-dev-01'" not in reply.text
    assert "{\"sqlserver-dev-01\"" not in reply.text
    assert "- sqlserver-dev-01: ok (12 databases)" in reply.text
    assert "- sqlserver-uat-01: failed: execution service returned an error (status 500)" in reply.text


@pytest.mark.asyncio
async def test_a_freeform_discover_request_for_one_server_reports_its_clean_failure():
    """The single-server `/discover <id>` path (gateway's
    `POST /v1/catalog/refresh/{id}`) reports a failure via an `"error"` key
    on an otherwise-200 response, not the `{"status": "ERROR"}` shape — this
    must also surface as clean text, not silently look like a success."""

    class _FailingOneToolClient(_FakeToolClient):
        async def refresh_catalog(self, channel, channel_account_id, server_id=None):
            return {
                "server_id": server_id,
                "databases": [],
                "warnings": [],
                "error": "could not reach the execution service",
            }

    llm = _FakeLLM(
        IntentExtraction(is_dba_task=False, meta_command="discover", instance_hint="sqlserver-uat-01")
    )
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm),
        tool_client=_FailingOneToolClient(),
        context=ContextManager(),
    )

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv3c", channel_thread_id="",
        message="run discovery on sqlserver-uat-01",
    )

    assert "sqlserver-uat-01" in reply.text
    assert "could not reach the execution service" in reply.text
    assert reply.status == "error"


@pytest.mark.asyncio
async def test_a_freeform_catalog_request_with_no_server_named_asks_which_one():
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="catalog", instance_hint=None))
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv4", channel_thread_id="",
        message="show me the catalog",
    )

    assert "which server" in reply.text.lower()


@pytest.mark.asyncio
async def test_a_freeform_status_request_with_no_investigation_says_so():
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="status"))
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv5", channel_thread_id="",
        message="what's the status",
    )

    assert "no active investigation" in reply.text.lower()


@pytest.mark.asyncio
async def test_a_freeform_approve_with_a_pending_approval_actually_approves_it():
    from numi.common.models.tool import ToolCallResponse, ToolCallStatus

    class _ApprovingToolClient(_FakeToolClient):
        async def approve(self, approval_id, channel, channel_account_id):
            return {"status": "APPROVED"}

        async def submit(self, request):
            return ToolCallResponse(status=ToolCallStatus.EXECUTED, message="ok", result={"ok": True})

    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="approve"))
    context = ContextManager()
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm), tool_client=_ApprovingToolClient(), context=context
    )
    state = context.get_or_create("conv6b", "slack", "", "U123")
    state.pending_approval = PendingApproval(
        approval_id="appr1",
        tool_id="database.kill_session",
        summary="Kill the blocker.",
        request={
            "tool_id": "database.kill_session",
            "arguments": {},
            "target": {},
            "reason": "x",
            "conversation_id": "conv6b",
            "request_id": "req1",
            "channel": "slack",
            "channel_account_id": "U123",
        },
    )

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv6b",
        channel_thread_id="",
        message="go ahead",
    )

    assert "no pending approval" not in reply.text.lower()
    assert state.pending_approval is None  # consumed


@pytest.mark.asyncio
async def test_a_freeform_approve_with_no_pending_approval_falls_through_to_a_real_instruction():
    """The fix for a real live finding: "go ahead and terminate session
    19860" matched the same "go ahead" phrasing as reacting to a shown
    approval card, but nothing was pending — it must never just dead-end
    with "there is no pending approval"; a message naming a specific
    action is far more likely a fresh instruction than a non-sequitur."""
    llm = _FakeLLM(
        IntentExtraction(
            is_dba_task=False,
            meta_command="approve",
            problem_summary="go ahead and terminate session 19860 on postgres-local",
        )
    )
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv6",
        channel_thread_id="",
        message="go ahead and terminate session 19860 on postgres-local",
    )

    assert "no pending approval" not in reply.text.lower()
    # No environment was named either, so the fresh investigation this
    # falls through to correctly asks for it next — proof it's actually
    # running as a real DBA task now, not dead-ending.
    assert reply.status == "clarification"


@pytest.mark.asyncio
async def test_a_freeform_help_request_gets_the_help_text():
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="help"))
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv7", channel_thread_id="",
        message="what can you do",
    )

    assert "Numi" in reply.text


@pytest.mark.asyncio
async def test_what_environments_and_databases_do_you_have_access_to_lists_servers():
    """Live-reproduced finding: this exact phrasing got the generic canned
    fallback instead of an answer. It most naturally maps to the existing
    "servers" meta_command, whose reply already names each server's
    environment — and (see the next test) now also its databases."""
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="servers"))
    servers = [
        {
            "id": "postgres-local",
            "environment": "development",
            "platform": "postgresql",
            "criticality": "standard",
        }
    ]
    orchestrator = _orchestrator(llm, servers=servers)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv8", channel_thread_id="",
        message="What environments and databases do you have access to?",
    )

    assert reply.status != "clarification"
    assert "development" in reply.text
    assert "postgres-local" in reply.text


@pytest.mark.asyncio
async def test_servers_reply_names_actual_databases_not_just_a_count():
    """The one real content gap identified for the "environments and
    databases" question: /servers previously only ever showed a bare
    database COUNT per server, never names — even though the names are
    already present on the very same list_servers() response (the same
    field AgentOrchestrator._known_database_names already reads). No new
    discovery call or tool is involved."""
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="servers"))
    servers = [
        {
            "id": "postgres-local",
            "environment": "development",
            "platform": "postgresql",
            "criticality": "standard",
            "catalog": {
                "database_count": 2,
                "discovered_at": "2026-01-01T00:00:00",
                "databases": ["CoreBanking", "Reporting"],
            },
        }
    ]
    orchestrator = _orchestrator(llm, servers=servers)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv9", channel_thread_id="",
        message="what servers do you have",
    )

    assert "CoreBanking" in reply.text
    assert "Reporting" in reply.text


@pytest.mark.asyncio
async def test_a_freeform_models_request_lists_available_models():
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="models"))
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv10", channel_thread_id="",
        message="what models can I use",
    )

    assert reply.status != "clarification"
    # For a test registry with no configured providers this is the honest
    # "none configured" message — the point of this test is that it routed
    # to the models handler at all, not the specific copy.
    assert "LLM" in reply.text or "model" in reply.text.lower()


@pytest.mark.asyncio
async def test_who_can_approve_requests_is_recognized_and_answered_honestly():
    """Live-reproduced finding: this got the generic canned fallback. It's
    an RBAC/policy question, not answered verbatim by any of the original 8
    meta-commands — this asserts it's now recognized as a legitimate
    informational question and answered with real, honest, general
    content, never the old boilerplate and never a made-up specific."""
    llm = _FakeLLM(IntentExtraction(is_dba_task=False, meta_command="approvers"))
    orchestrator = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv11", channel_thread_id="",
        message="who can approve requests from you?",
    )

    assert reply.status != "clarification"
    assert "Try:" not in reply.text  # not the generic _HELP_TEXT fallback
    assert "role" in reply.text.lower()
    assert "policy.yaml" not in reply.text  # never dumps the raw config


class TestMockPlannerMetaCommandDetection:
    """The deterministic offline planner recognizes the same free-text
    phrasings, deliberately more conservatively than the real-provider
    prompt (see mock.py's own module comment) to keep this test-only
    planner's false-positive rate low."""

    @pytest.mark.asyncio
    async def test_list_servers_phrasing(self):
        result = await MockLLMProvider().extract_intent("list my servers", known_database_names=[])
        assert result.meta_command == "servers"

    @pytest.mark.asyncio
    async def test_playbooks_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "show me your playbooks", known_database_names=[]
        )
        assert result.meta_command == "playbooks"

    @pytest.mark.asyncio
    async def test_discover_phrasing(self):
        result = await MockLLMProvider().extract_intent("run discovery please", known_database_names=[])
        assert result.meta_command == "discover"

    @pytest.mark.asyncio
    async def test_an_ordinary_investigation_request_is_not_misdetected_as_meta(self):
        result = await MockLLMProvider().extract_intent(
            "Why is CoreBanking so slow right now?", known_database_names=["CoreBanking"]
        )
        assert result.meta_command is None
        assert result.is_dba_task is True

    @pytest.mark.asyncio
    async def test_environments_and_databases_access_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "What environments and databases do you have access to?", known_database_names=[]
        )
        assert result.meta_command == "servers"

    @pytest.mark.asyncio
    async def test_what_servers_do_you_know_about_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "what servers do you know about", known_database_names=[]
        )
        assert result.meta_command == "servers"

    @pytest.mark.asyncio
    async def test_what_can_you_see_phrasing(self):
        result = await MockLLMProvider().extract_intent("what can you see", known_database_names=[])
        assert result.meta_command == "servers"

    @pytest.mark.asyncio
    async def test_whats_registered_phrasing(self):
        result = await MockLLMProvider().extract_intent("what's registered", known_database_names=[])
        assert result.meta_command == "servers"

    @pytest.mark.asyncio
    async def test_what_do_you_have_access_to_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "what do you have access to", known_database_names=[]
        )
        assert result.meta_command == "servers"

    @pytest.mark.asyncio
    async def test_models_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "what models can I use", known_database_names=[]
        )
        assert result.meta_command == "models"

    @pytest.mark.asyncio
    async def test_switch_model_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "switch to a different model", known_database_names=[]
        )
        assert result.meta_command == "models"

    @pytest.mark.asyncio
    async def test_who_can_approve_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "who can approve requests from you?", known_database_names=[]
        )
        assert result.meta_command == "approvers"

    @pytest.mark.asyncio
    async def test_who_needs_to_sign_off_phrasing(self):
        result = await MockLLMProvider().extract_intent(
            "who needs to sign off on this action?", known_database_names=[]
        )
        assert result.meta_command == "approvers"
