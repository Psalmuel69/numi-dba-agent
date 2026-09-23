"""Coverage for orchestrator.py branches not exercised elsewhere:

- The literal exact-slash-command path (`_handle_command_if_any`), which is
  a distinct code path from the free-text `meta_command` routing already
  covered by `test_meta_command_freeform.py` — reached only when the DBA
  types the exact syntax (e.g. `/approve appr1`), never via an LLM-classified
  intent.
- `/model <provider> [<model>]` switching (registry validation, membership
  check, and the two "show current selection" branches).
- `/catalog <id>` rendering real catalog data, including the least-privilege
  warning line.
- The environment-clarification re-ask when a reply doesn't answer the
  question, and `_classify_potential_topic_shift`'s early-exit branches.
"""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager, InvestigationState, PendingApproval
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import IntentExtraction
from numi.common.config import Settings
from numi.common.models.tool import ToolCallResponse, ToolCallStatus


class _FakeToolClient:
    def __init__(self, servers=None, catalog=None):
        self._servers = servers or []
        self._catalog = catalog or {}
        self.approve_calls: list[tuple] = []
        self.reject_calls: list[tuple] = []

    async def list_servers(self):
        return self._servers

    async def available_tools(self, channel, channel_account_id):
        return []

    async def get_server_catalog(self, server_id):
        return self._catalog.get(server_id, {"server": {"id": server_id}, "catalog": None})

    async def refresh_catalog(self, channel, channel_account_id, server_id=None):
        return {"status": "OK", "server_id": server_id}

    async def approve(self, approval_id, channel, channel_account_id):
        self.approve_calls.append((approval_id, channel, channel_account_id))
        return {"status": "APPROVED"}

    async def reject(self, approval_id, channel, channel_account_id):
        self.reject_calls.append((approval_id, channel, channel_account_id))

    async def submit(self, request):
        return ToolCallResponse(status=ToolCallStatus.EXECUTED, message="ok", result={"ok": True})


class _FakeLLM:
    def __init__(self, intent: IntentExtraction | None = None):
        self._intent = intent

    async def extract_intent(self, *args, **kwargs):
        return self._intent


def _orchestrator(llm=None, servers=None, catalog=None, registry=None) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=registry or LLMRegistry.for_testing(llm),
        tool_client=_FakeToolClient(servers, catalog),
        context=ContextManager(),
    )


def _pending_approval(**overrides) -> PendingApproval:
    defaults = dict(
        approval_id="appr1",
        tool_id="database.kill_session",
        summary="Kill the blocker.",
        request={
            "tool_id": "database.kill_session",
            "arguments": {},
            "target": {},
            "reason": "x",
            "conversation_id": "conv1",
            "request_id": "req1",
            "channel": "slack",
            "channel_account_id": "U123",
        },
    )
    defaults.update(overrides)
    return PendingApproval(**defaults)


# --------------------------------------------------------------------------- #
# Literal `/approve <id>` and `/reject <id>` — the exact-syntax path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_slash_approve_with_a_mismatched_id_does_not_approve_anything():
    orchestrator = _orchestrator()
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    state.pending_approval = _pending_approval(approval_id="appr1")

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/approve wrong-id",
    )

    assert reply.status == "error"
    assert "doesn't match" in reply.text
    assert state.pending_approval is not None  # untouched


@pytest.mark.asyncio
async def test_slash_reject_with_a_mismatched_id_does_not_reject_anything():
    orchestrator = _orchestrator()
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    state.pending_approval = _pending_approval(approval_id="appr1")

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/reject wrong-id",
    )

    assert reply.status == "error"
    assert "doesn't match" in reply.text
    assert state.pending_approval is not None


@pytest.mark.asyncio
async def test_slash_approve_with_the_matching_id_actually_approves():
    orchestrator = _orchestrator()
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    state.pending_approval = _pending_approval(approval_id="appr1")

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/approve appr1",
    )

    assert state.pending_approval is None
    assert "completed successfully" in reply.text


@pytest.mark.asyncio
async def test_slash_reject_with_the_matching_id_actually_rejects():
    orchestrator = _orchestrator()
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    state.pending_approval = _pending_approval(approval_id="appr1")

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/reject appr1",
    )

    assert state.pending_approval is None
    assert "rejected" in reply.text.lower()


# --------------------------------------------------------------------------- #
# Other literal slash commands (`_handle_command_if_any`)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_slash_servers_lists_registered_servers():
    servers = [
        {
            "id": "postgres-local",
            "environment": "development",
            "platform": "postgresql",
            "criticality": "standard",
        }
    ]
    orchestrator = _orchestrator(servers=servers)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/servers",
    )

    assert "postgres-local" in reply.text


@pytest.mark.asyncio
async def test_slash_catalog_with_no_id_asks_which_server():
    orchestrator = _orchestrator()

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/catalog",
    )

    assert "which server" in reply.text.lower()


@pytest.mark.asyncio
async def test_slash_catalog_renders_full_catalog_data_including_least_privilege_warning():
    catalog = {
        "sqlserver-dev-01": {
            "server": {"id": "sqlserver-dev-01"},
            "catalog": {
                "engine_edition": "Standard",
                "engine_version": "16.2",
                "databases": [
                    {
                        "name": "CoreBanking",
                        "state": "ONLINE",
                        "size_bytes": 5_000_000,
                        "objects": [{"kind": "table"}, {"kind": "table"}, {"kind": "view"}],
                        "extensions": [],
                    }
                ],
                "warnings": [],
                "least_privilege": {
                    "checked": True,
                    "login": "numi_diag",
                    "has_user_table_select": True,
                    "granted_object_count": 3,
                    "count_is_lower_bound": False,
                    "sample_objects": ["dbo.Accounts"],
                    "scope_note": "current database only",
                },
            },
        }
    }
    orchestrator = _orchestrator(catalog=catalog)

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/catalog sqlserver-dev-01",
    )

    assert "CoreBanking" in reply.text
    assert "Standard 16.2" in reply.text
    assert "numi_diag" in reply.text
    assert "current database only" in reply.text


@pytest.mark.asyncio
async def test_slash_catalog_for_an_undiscovered_server_says_to_run_discover():
    orchestrator = _orchestrator(
        catalog={"sqlserver-dev-01": {"server": {"id": "sqlserver-dev-01"}, "catalog": None}}
    )

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/catalog sqlserver-dev-01",
    )

    assert "/discover sqlserver-dev-01" in reply.text


@pytest.mark.asyncio
async def test_slash_discover_with_a_server_id_refreshes_it():
    orchestrator = _orchestrator()

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/discover sqlserver-dev-01",
    )

    assert "Discovery complete" in reply.text
    assert "sqlserver-dev-01" in reply.text


@pytest.mark.asyncio
async def test_slash_help_returns_help_text():
    orchestrator = _orchestrator()
    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/help",
    )
    assert reply.text


@pytest.mark.asyncio
async def test_slash_approvers_returns_approval_model_text():
    orchestrator = _orchestrator()
    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/approvers",
    )
    assert reply.text


# --------------------------------------------------------------------------- #
# `/model` — current selection and switching
# --------------------------------------------------------------------------- #


def _real_registry(**settings_overrides) -> LLMRegistry:
    return LLMRegistry(Settings(_env_file=None, llm_provider="mock", **settings_overrides))


@pytest.mark.asyncio
async def test_slash_model_with_no_args_shows_the_deployment_default_when_unset():
    orchestrator = _orchestrator(registry=_real_registry())

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/model",
    )

    assert "deployment default" in reply.text
    assert "mock" in reply.text


@pytest.mark.asyncio
async def test_slash_model_with_no_args_shows_the_conversations_own_choice_once_set():
    orchestrator = _orchestrator(registry=_real_registry())
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    state.llm_provider = "mock"
    state.llm_model = None

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/model",
    )

    assert "this conversation" in reply.text


@pytest.mark.asyncio
async def test_slash_model_switch_to_mock_succeeds_and_is_scoped_to_the_conversation():
    orchestrator = _orchestrator(registry=_real_registry())

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/model mock",
    )

    assert "set to mock" in reply.text.lower()
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    assert state.llm_provider == "mock"


@pytest.mark.asyncio
async def test_slash_model_switch_is_rejected_when_selection_is_disabled():
    orchestrator = _orchestrator(registry=_real_registry(allow_user_model_selection=False))

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/model mock",
    )

    assert reply.status == "error"
    assert "disabled" in reply.text.lower()


@pytest.mark.asyncio
async def test_slash_model_switch_to_an_unknown_provider_is_rejected():
    orchestrator = _orchestrator(registry=_real_registry())

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/model carrier-pigeon",
    )

    assert reply.status == "error"
    assert "Unknown provider" in reply.text


@pytest.mark.asyncio
async def test_slash_model_switch_to_an_unconfigured_provider_is_rejected():
    orchestrator = _orchestrator(registry=_real_registry())

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/model anthropic",
    )

    assert reply.status == "error"
    assert "not configured" in reply.text


@pytest.mark.asyncio
async def test_slash_models_lists_available_providers():
    orchestrator = _orchestrator(registry=_real_registry())

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="/models",
    )

    assert "Available LLMs" in reply.text or "No LLM providers" in reply.text


# --------------------------------------------------------------------------- #
# Environment re-ask when the reply doesn't answer the question
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_non_matching_reply_while_the_environment_is_unknown_re_asks_instead_of_guessing():
    orchestrator = _orchestrator(llm=_FakeLLM())
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")
    state.investigation = InvestigationState(investigation_id="inv1", problem="slow queries")

    reply = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv1",
        channel_thread_id="", message="I'm not sure, can you check anyway?",
    )

    assert reply.status == "clarification"
    assert "which environment" in reply.text.lower()
    assert "environment" not in state.database_context


# --------------------------------------------------------------------------- #
# `_classify_potential_topic_shift` — early-exit branches
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_topic_shift_classification_short_circuits_below_the_word_floor_without_calling_the_llm():
    llm = _FakeLLM(IntentExtraction(is_dba_task=True, instance_hint="postgres-local"))
    orchestrator = _orchestrator(llm=llm)
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")

    result = await orchestrator._classify_potential_topic_shift(state, "development")

    assert result is None


@pytest.mark.asyncio
async def test_topic_shift_classification_returns_none_for_chitchat():
    llm = _FakeLLM(
        IntentExtraction(is_dba_task=False, is_greeting_or_chitchat=True, instance_hint="postgres-local")
    )
    orchestrator = _orchestrator(llm=llm)
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")

    result = await orchestrator._classify_potential_topic_shift(
        state, "hey there how's it going today friend"
    )

    assert result is None


@pytest.mark.asyncio
async def test_topic_shift_classification_returns_none_when_no_target_is_named():
    llm = _FakeLLM(IntentExtraction(is_dba_task=True))
    orchestrator = _orchestrator(llm=llm)
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")

    result = await orchestrator._classify_potential_topic_shift(
        state, "please just go ahead and fix it now thanks"
    )

    assert result is None


@pytest.mark.asyncio
async def test_topic_shift_classification_returns_the_intent_when_it_names_its_own_target():
    intent = IntentExtraction(is_dba_task=True, instance_hint="postgres-local")
    llm = _FakeLLM(intent)
    orchestrator = _orchestrator(llm=llm)
    state = orchestrator._context.get_or_create("conv1", "slack", "", "U123")

    result = await orchestrator._classify_potential_topic_shift(
        state, "restart the postgres-local instance right now"
    )

    assert result is intent
