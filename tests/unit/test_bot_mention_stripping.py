"""Reproduces a live finding: sending the exact, already-advertised
"@Numi DBA Agent /models" command in Slack (the ordinary way of
addressing the bot in a shared channel) got the WRONG reply — "Which
environment should I investigate...?" — instead of listing models.

Root cause, confirmed by reading the real code: Slack delivers an
@-mention as a literal `<@U0BOTID>` token in the message text
(`channels/api/app.py`'s slack_webhook does `message=event.get("text", "")`,
forwarded completely unmodified — nothing anywhere in the channels layer
strips it). So the text actually reaching `AgentOrchestrator.handle_message`
was "<@U0BOTID> /models", not "/models" — which `_handle_command_if_any`'s
exact `stripped in ("/models", "/model")` check does not match, so it fell
through past every literal-command check into the normal DBA-task path,
which (with an investigation already active and no environment set yet)
produced the environment-clarification gate instead.

The fix strips a leading/trailing bot-mention token once, centrally, at
the very start of `handle_message` — before ANY of the exact-match,
free-form meta_command, or investigation-resume logic sees the message."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager, InvestigationState
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator, _strip_bot_mention_noise
from numi.agent.planner.actions import IntentExtraction


class _FakeToolClient:
    async def list_servers(self):
        return []

    async def available_tools(self, channel, channel_account_id):
        return []


class _FakeLLM:
    """extract_intent raises if ever called — proves the literal-command
    path (or the fast environment-answer path) handled the message without
    ever needing a real classification call."""

    async def extract_intent(self, *args, **kwargs):
        raise AssertionError("extract_intent must not be called for a literal slash command")

    async def decide_next_action(self, **kwargs):
        raise AssertionError("decide_next_action must not be called here")


def _orchestrator() -> tuple[AgentOrchestrator, ContextManager]:
    context = ContextManager()
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(_FakeLLM()),
        tool_client=_FakeToolClient(),
        context=context,
    )
    return orchestrator, context


class TestStripBotMentionNoise:
    def test_strips_a_leading_mention(self):
        assert _strip_bot_mention_noise("<@U0BOTID> /models") == "/models"

    def test_strips_a_trailing_mention(self):
        assert _strip_bot_mention_noise("development <@U0BOTID>") == "development"

    def test_strips_multiple_leading_mentions(self):
        assert _strip_bot_mention_noise("<@U1> <@U2> /help") == "/help"

    def test_never_strips_a_mid_message_mention(self):
        # A mention naming a DIFFERENT person in the middle of a real
        # sentence carries real meaning and must survive.
        text = "check with <@U999> about approving this"
        assert _strip_bot_mention_noise(text) == text

    def test_a_message_with_no_mention_is_unchanged_besides_stripping(self):
        assert _strip_bot_mention_noise("  /status  ") == "/status"


@pytest.mark.asyncio
async def test_literal_slash_models_with_a_leading_mention_lists_models_not_environment_clarification():
    """The exact live reproduction: an active investigation with no
    environment set yet, then the DBA sends the exact advertised command
    with the bot mentioned (as Slack actually delivers it)."""
    orchestrator, context = _orchestrator()
    state = context.get_or_create("conv1", "slack", "thread1", "U123")
    state.investigation = InvestigationState(investigation_id="inv1", problem="check overall health")

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv1",
        channel_thread_id="thread1",
        message="<@U0BOTID> /models",
    )

    assert "Which environment" not in reply.text
    assert reply.status != "clarification"
    assert "LLM" in reply.text or "model" in reply.text.lower()


@pytest.mark.asyncio
async def test_literal_help_with_a_leading_mention_still_matches_the_exact_command():
    orchestrator, _context = _orchestrator()

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv2",
        channel_thread_id="",
        message="<@U0BOTID> /help",
    )

    assert "Numi" in reply.text


@pytest.mark.asyncio
async def test_literal_status_with_a_leading_mention_still_matches_the_exact_command():
    orchestrator, _context = _orchestrator()

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv3",
        channel_thread_id="",
        message="<@U0BOTID> /status",
    )

    assert "no active investigation" in reply.text.lower()


@pytest.mark.asyncio
async def test_a_freeform_meta_command_still_recognized_via_the_llm_with_a_leading_mention():
    """The stripped message is what actually reaches extract_intent too —
    not just the exact-match literal-command path."""

    class _FakeMetaLLM:
        def __init__(self):
            self.seen_message = None

        async def extract_intent(self, message, *args, **kwargs):
            self.seen_message = message
            return IntentExtraction(is_dba_task=False, meta_command="help")

    llm = _FakeMetaLLM()
    context = ContextManager()
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm), tool_client=_FakeToolClient(), context=context
    )

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv4",
        channel_thread_id="",
        message="<@U0BOTID> what can you do",
    )

    assert llm.seen_message == "what can you do"
    assert "Numi" in reply.text
