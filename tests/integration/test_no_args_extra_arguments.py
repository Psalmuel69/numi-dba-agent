"""Reproduces a live finding from Slack testing: the LLM planner repeatedly
proposed `database.get_blocking_sessions` / `database.get_sessions` (both
`NoArgs` in `gateway.domain.tool_catalog.ARGUMENT_MODELS` — they take zero
arguments beyond `target`) with an extra `reason` or `database_name` folded
into `arguments`. The real Gateway's `NoArgs` model is `extra="forbid"`, so
this is rejected outright as INVALID_ARGUMENTS. The orchestrator's
self-correction path (`_SELF_CORRECTABLE_DENIAL_CODES`) already recovers
within the same turn, so this never broke an investigation — but it wasted
an extra LLM call and Gateway round-trip every single time.

Root cause: `_continue_investigation` built `tool_requirements` (the map of
each tool_id's real argument schema, injected into the planner's prompt) by
*omitting* any tool whose required-arguments list is empty — so a NoArgs
tool never appeared in that map at all. The model had no explicit signal
that a given tool_id needs nothing in `arguments`; it only ever saw entries
for tools that DO need something, and reasonably (but wrongly) generalized
from the `arguments` schema's superset of possible keys (which includes
`reason`, `session_id`, `database_name` for the tools that actually need
them).

This test drives the real orchestrator against the real Gateway + Execution
Service pipeline (in-process ASGI, exactly like
tests/integration/test_agent_orchestrator.py) with a scripted planner that
reproduces the exact bad call observed live, and asserts the fix's two
layers both hold:

1. The prompt fix: `tool_requirements` now includes every available tool,
   including NoArgs ones mapped to `[]` — verified separately in
   tests/unit/test_tool_requirements_prompt.py against the real orchestrator
   construction (not just base.py's prompt-building in isolation).
2. The defense-in-depth fix: even if a planner still adds an unschematized
   key, the orchestrator strips it before ever building the
   `ToolCallRequest` it sends to the Gateway — so the bad call the model
   proposed here is never actually rejected, and the investigation
   proceeds on the very first attempt.
"""

from __future__ import annotations

import httpx

from numi.agent.context_manager import ContextManager
from numi.agent.llm.base import LLMProvider
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import AgentAction, Conclude, IntentExtraction, ProposeToolCall
from numi.agent.tool_client import ToolClient
from numi.common.config import Settings
from numi.common.service_auth import ServiceTokenIssuer
from numi.execution.api.app import create_app as create_execution_app
from numi.gateway.api.app import create_app as create_gateway_app
from tests.canned_adapter import canned_adapter_factory


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        control_db_url="sqlite+aiosqlite:///:memory:",
        service_jwt_secret="test-secret",
        service_jwt_issuer="numi-internal",
        llm_provider="mock",
    )


class _ScriptedProvider(LLMProvider):
    """A stand-in for a real API-backed provider that reproduces the exact
    bad completion observed live: a NoArgs tool called with a `reason` and
    `database_name` stuffed into `arguments`, even though `ProposeToolCall`
    already carries its own top-level `reason`."""

    provider_name = "scripted"

    def __init__(self, actions: list[AgentAction]):
        super().__init__("scripted-model")
        self._actions = list(actions)
        self.decide_calls: list[dict] = []

    async def extract_intent(self, message, known_database_names, known_server_hints=None):
        return IntentExtraction(
            is_dba_task=True,
            environment_hint="production",
            database_hint="CoreBanking",
            problem_summary=message,
        )

    async def decide_next_action(
        self, *, problem_statement, available_tool_ids, transcript, turn_count, tool_requirements=None
    ):
        self.decide_calls.append(
            {"turn_count": turn_count, "transcript": list(transcript), "tool_requirements": tool_requirements}
        )
        return self._actions.pop(0)

    async def summarize_for_human(self, *, problem_statement, transcript):
        return "summary"


async def _build_orchestrator(provider: LLMProvider) -> AgentOrchestrator:
    settings = _settings()
    execution_app = create_execution_app(settings, adapter_factory=canned_adapter_factory)
    execution_transport = httpx.ASGITransport(app=execution_app)
    gateway_app = create_gateway_app(settings, execution_transport=execution_transport)
    await gateway_app.state.gateway.db.create_all()
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
    tool_client = ToolClient(settings.gateway_base_url, issuer, transport=gateway_transport)
    return AgentOrchestrator(LLMRegistry.for_testing(provider), tool_client, ContextManager())


async def test_a_no_args_tool_called_with_an_extra_reason_is_never_denied_by_the_gateway():
    """The exact live bug: get_blocking_sessions (NoArgs) proposed with a
    `reason` and `database_name` folded into `arguments`. Before the fix,
    this reached the Gateway as-is and came back DENIED/INVALID_ARGUMENTS,
    burning one self-correction turn. After the fix, the orchestrator
    strips the unschematized keys before ever submitting, so the very first
    attempt succeeds — no self-correction round-trip at all."""
    provider = _ScriptedProvider(
        [
            ProposeToolCall(
                tool_id="database.get_blocking_sessions",
                reason="Checking for blocking chains given the reported slowness.",
                arguments={
                    "reason": "Checking for blocking chains given the reported slowness.",
                    "database_name": "CoreBanking",
                },
            ),
            Conclude(summary="no blocking chains were found", confidence="unable_to_confirm"),
        ]
    )
    orchestrator = await _build_orchestrator(provider)

    reply = await orchestrator.handle_message(
        channel="teams",
        channel_account_id="aad-mock-l2",
        conversation_id="conv_no_args_1",
        channel_thread_id="thread_1",
        # Deliberately free of any playbook trigger word (see
        # agent.playbooks.library.PLAYBOOKS) — a matched playbook would run
        # its own deterministic steps first and never consult the scripted
        # planner below on the very first turn.
        message="Please look into CoreBanking on production, users are complaining.",
    )

    assert reply.status == "ok"
    # Only two decide_next_action calls: the bad-but-stripped tool call,
    # then the conclude — never a third call to self-correct a denial.
    assert len(provider.decide_calls) == 2
    first_call_transcript = provider.decide_calls[1]["transcript"]
    assert len(first_call_transcript) == 1
    assert first_call_transcript[0]["tool_id"] == "database.get_blocking_sessions"
    # Never denied — no failure_code anywhere in the transcript entry.
    assert "failure_code" not in first_call_transcript[0]["result"]


async def test_tool_requirements_passed_to_the_planner_names_every_no_args_tool_explicitly():
    """The prompt-level half of the fix: the real orchestrator's own
    tool_requirements map (not a hand-built dict in a unit test) must
    include NoArgs tools mapped to an empty list, not omit them — so the
    model is explicitly told "this tool takes nothing" instead of having to
    infer it from silence."""
    provider = _ScriptedProvider([Conclude(summary="nothing to check", confidence="unable_to_confirm")])
    orchestrator = await _build_orchestrator(provider)

    await orchestrator.handle_message(
        channel="teams",
        channel_account_id="aad-mock-l2",
        conversation_id="conv_no_args_2",
        channel_thread_id="thread_1",
        message="Please look into CoreBanking on production, users are complaining.",
    )

    tool_requirements = provider.decide_calls[0]["tool_requirements"]
    assert tool_requirements is not None
    assert tool_requirements.get("database.get_blocking_sessions") == []
    assert tool_requirements.get("database.get_sessions") == []
