"""StructuredLLMProvider (the shared plumbing behind Anthropic/OpenAI/Gemini/
DeepSeek) must never let a provider outage or a malformed completion crash
the chat request — both degrade to a clear response instead (spec §34).

Reproduces two real failures found running live against Gemini 3.5 Flash:
a transient 503 propagating as an unhandled exception, and a read-only
`propose_tool_call` completion missing `reason` (a Pydantic-required field
the flat cross-provider schema can't mark conditionally-required)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from numi.agent.llm.base import StructuredLLMProvider
from numi.agent.planner.actions import (
    AskClarification,
    CritiqueVerdict,
    IntentExtraction,
    ProposeToolCall,
)


class _FakeStructuredProvider(StructuredLLMProvider):
    provider_name = "fake"
    _CALL_RETRY_DELAY_SECONDS = 0  # keep the retry-on-failure tests instant

    def __init__(self, *, tool_result: dict[str, Any] | None = None, tool_error: Exception | None = None):
        super().__init__("fake-model")
        self._tool_result = tool_result
        self._tool_error = tool_error
        self.call_count = 0

    async def _call_tool(self, *, system: str, user: str, schema: dict, tool_name: str) -> dict:
        self.call_count += 1
        if self._tool_error is not None:
            raise self._tool_error
        return self._tool_result or {}

    async def _call_text(self, *, system: str, user: str) -> str:
        return ""


@pytest.mark.asyncio
async def test_decide_next_action_degrades_on_a_transient_provider_outage():
    provider = _FakeStructuredProvider(tool_error=RuntimeError("503 UNAVAILABLE: high demand"))
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, AskClarification)
    assert "fake" in action.question and "try again" in action.question.lower()
    # First attempt + _CALL_RETRIES retries, all failing the same way.
    assert provider.call_count == provider._CALL_RETRIES + 1


@pytest.mark.asyncio
async def test_decide_next_action_recovers_after_a_transient_failure():
    """A call that fails once and then succeeds must not be treated as a
    permanent outage — this is the whole point of retrying."""

    class _FlakyThenOk(_FakeStructuredProvider):
        async def _call_tool(self, *, system, user, schema, tool_name):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("503 UNAVAILABLE: high demand")
            return {
                "action": "propose_tool_call",
                "tool_id": "database.get_health",
                "reason": "Baseline check.",
            }

    provider = _FlakyThenOk()
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, ProposeToolCall)
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_decide_next_action_retries_a_malformed_completion_not_just_call_failures():
    """Reproduces the live finding: a completion missing a required field
    (not a network/outage failure) is *also* worth one more try — the same
    prompt was observed to succeed on a later attempt in production."""

    class _MalformedThenOk(_FakeStructuredProvider):
        async def _call_tool(self, *, system, user, schema, tool_name):
            self.call_count += 1
            if self.call_count == 1:
                return {"action": "propose_tool_call", "tool_id": "database.get_health"}  # no reason
            return {
                "action": "propose_tool_call",
                "tool_id": "database.get_health",
                "reason": "Baseline check.",
            }

    provider = _MalformedThenOk()
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, ProposeToolCall)
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_a_fresh_call_failure_after_an_earlier_validation_failure_is_not_mislabeled():
    """Reproduces the live finding: attempt 1 returns a malformed completion
    (validation fails), attempt 2 hits a fresh outage before any response
    comes back at all. The final message must be the outage one — a stale
    `last_raw` from attempt 1 must not make this look like a validation
    problem on attempt 2, which never even got a response to validate."""

    class _MalformedThenOutage(_FakeStructuredProvider):
        async def _call_tool(self, *, system, user, schema, tool_name):
            self.call_count += 1
            if self.call_count == 1:
                return {"action": "propose_tool_call", "tool_id": "database.get_health"}  # no reason
            raise RuntimeError("503 UNAVAILABLE: high demand")

    provider = _MalformedThenOutage()
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, AskClarification)
    assert "temporarily unavailable" in action.question.lower()
    assert provider.call_count == provider._CALL_RETRIES + 1


@pytest.mark.asyncio
async def test_decide_next_action_degrades_on_a_malformed_completion():
    provider = _FakeStructuredProvider(
        tool_result={"action": "propose_tool_call", "tool_id": "database.get_health"}
    )
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, AskClarification)


@pytest.mark.asyncio
async def test_decide_next_action_accepts_a_read_tool_call_with_reason():
    """The bug this pins: `reason` is Pydantic-required on ProposeToolCall
    for every tool call, read or write — the schema itself can't enforce
    that conditionally, so the system prompt must, and the model must
    actually follow it."""
    provider = _FakeStructuredProvider(
        tool_result={
            "action": "propose_tool_call",
            "tool_id": "database.get_health",
            "reason": "Establishing a baseline before investigating further.",
        }
    )
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, ProposeToolCall)
    assert action.tool_id == "database.get_health"


@pytest.mark.asyncio
async def test_decide_next_action_recovers_a_reason_left_only_inside_arguments():
    """Reproduces a live finding: after the target/arguments schema fix, a
    real model correctly filled `arguments.reason` (and session_id) but
    still left the *top-level* `reason` out — "reason" appearing at two
    nesting levels seems to read as one field, said once. Copying it up
    (never inventing a value) is what actually closes this, on top of the
    prompt already saying they're separate fields."""
    provider = _FakeStructuredProvider(
        tool_result={
            "action": "propose_tool_call",
            "tool_id": "database.cancel_query",
            "arguments": {"session_id": "72", "reason": "Cartesian join burning CPU."},
            "target": {"session_id": "72"},
        }
    )
    action = await provider.decide_next_action(
        problem_statement="check activity", available_tool_ids=["database.cancel_query"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, ProposeToolCall)
    assert action.reason == "Cartesian join burning CPU."
    assert action.arguments["reason"] == "Cartesian join burning CPU."


@pytest.mark.asyncio
async def test_decide_next_action_never_invents_a_top_level_reason():
    """The recovery is copy-only — if arguments has no reason either, the
    call must still fail validation and degrade normally, not fabricate
    one."""
    provider = _FakeStructuredProvider(
        tool_result={
            "action": "propose_tool_call",
            "tool_id": "database.cancel_query",
            "arguments": {"session_id": "72"},
        }
    )
    action = await provider.decide_next_action(
        problem_statement="check activity", available_tool_ids=["database.cancel_query"],
        transcript=[], turn_count=0,
    )
    assert isinstance(action, AskClarification)


@pytest.mark.asyncio
async def test_extract_intent_degrades_on_a_transient_provider_outage():
    provider = _FakeStructuredProvider(tool_error=RuntimeError("connection reset"))
    intent = await provider.extract_intent("CoreBanking is slow", known_database_names=["CoreBanking"])
    assert isinstance(intent, IntentExtraction)
    assert intent.is_dba_task is True


@pytest.mark.asyncio
async def test_a_single_decision_never_exceeds_the_overall_deadline():
    """The actual production guarantee this session's latency work landed
    on: no matter how many retries or fallback models a provider tries
    internally, one decide_next_action call is never worse than
    _OVERALL_DEADLINE_SECONDS late. Verified live this was the real gap —
    several individually-reasonable timeouts had no shared ceiling, and
    their product let one decision run for minutes."""

    class _AlwaysSlowProvider(_FakeStructuredProvider):
        _OVERALL_DEADLINE_SECONDS = 0.05  # instance override, keep the test fast

        async def _call_tool(self, *, system, user, schema, tool_name):
            self.call_count += 1
            await asyncio.sleep(10)  # never actually reached
            return {}

    provider = _AlwaysSlowProvider()
    start = time.monotonic()
    action = await provider.decide_next_action(
        problem_statement="check health", available_tool_ids=["database.get_health"],
        transcript=[], turn_count=0,
    )
    elapsed = time.monotonic() - start
    assert isinstance(action, AskClarification)
    assert elapsed < 2.0  # nowhere near the real 10s sleep or a 20s default deadline


@pytest.mark.asyncio
async def test_critique_conclusion_returns_a_sound_verdict():
    provider = _FakeStructuredProvider(tool_result={"sound": True})
    verdict = await provider.critique_conclusion(
        problem_statement="check health",
        transcript=[],
        proposed_summary="High CPU driven by a runaway query.",
        proposed_root_cause="Missing index on Orders.CustomerId",
        proposed_confidence="likely",
        proposed_recommendation="Add the index.",
    )
    assert isinstance(verdict, CritiqueVerdict)
    assert verdict.sound is True


@pytest.mark.asyncio
async def test_critique_conclusion_returns_an_unsound_verdict_with_its_issue():
    provider = _FakeStructuredProvider(
        tool_result={"sound": False, "issue": "No tool result ever mentioned CustomerId."}
    )
    verdict = await provider.critique_conclusion(
        problem_statement="check health",
        transcript=[],
        proposed_summary="High CPU.",
        proposed_root_cause="Missing index on Orders.CustomerId",
        proposed_confidence="likely",
        proposed_recommendation=None,
    )
    assert verdict.sound is False
    assert verdict.issue == "No tool result ever mentioned CustomerId."


@pytest.mark.asyncio
async def test_critique_conclusion_retries_a_malformed_completion_once():
    class _OnceMalformedProvider(_FakeStructuredProvider):
        _CALL_RETRY_DELAY_SECONDS = 0

        def __init__(self):
            super().__init__()
            self._responses = [{"issue": "missing the required sound field"}, {"sound": True}]

        async def _call_tool(self, *, system, user, schema, tool_name):
            self.call_count += 1
            return self._responses.pop(0)

    provider = _OnceMalformedProvider()
    verdict = await provider.critique_conclusion(
        problem_statement="p",
        transcript=[],
        proposed_summary="s",
        proposed_root_cause=None,
        proposed_confidence="unable_to_confirm",
        proposed_recommendation=None,
    )
    assert verdict.sound is True
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_critique_conclusion_propagates_a_persistent_failure():
    """Deliberately does NOT degrade like decide_next_action does —
    `orchestrator._self_critique_conclude` is the one place that turns any
    exception here into "fail open, accept the conclusion" (see its own
    docstring); this layer has nothing useful to do differently."""
    provider = _FakeStructuredProvider(tool_error=RuntimeError("persistent outage"))
    with pytest.raises(RuntimeError, match="persistent outage"):
        await provider.critique_conclusion(
            problem_statement="p",
            transcript=[],
            proposed_summary="s",
            proposed_root_cause=None,
            proposed_confidence="unable_to_confirm",
            proposed_recommendation=None,
        )
