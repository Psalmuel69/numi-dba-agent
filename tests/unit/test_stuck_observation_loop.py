"""A real model can get stuck restating the same finding as one
record_observation after another instead of ever emitting action=conclude
— reproduced live on the replication playbook: all 3 steps ran correctly
(get_replication_status legitimately returned empty — no replica
configured, a perfectly answerable "no replication set up" conclusion),
but the model spent its remaining turns re-recording that same finding as
evidence instead of concluding, hit the full 6-turn cap, and fell back to
the generic "no confirmed root cause" message even though the real answer
was clear after the very first observation.

`_MAX_CONSECUTIVE_RECORD_OBSERVATIONS` (any other action resets the count)
stops asking once the pattern is clearly stuck, instead of waiting out the
rest of the turn budget — see orchestrator.py for the live-verified
rationale."""

from __future__ import annotations

import pytest

from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, ProposeToolCall, RecordObservation
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeLLM,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)


@pytest.mark.asyncio
async def test_repeated_record_observations_stop_before_the_full_turn_cap():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    # Six near-duplicate observations offered — the fix must never consume
    # more than _MAX_CONSECUTIVE_RECORD_OBSERVATIONS + 1 of them.
    llm = _FakeLLM(actions=[RecordObservation(text="Still nothing conclusive.")] * 6)
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 2  # stopped asking well short of all 6 offered
    assert investigation.turn_count < 6
    # Only record_observation ever ran — no write, no verification —
    # collapses to CONCLUDED_NO_ACTION (see InvestigationStage).
    assert investigation.status == "CONCLUDED_NO_ACTION"
    assert "without reaching a confirmed root cause" in reply.text


@pytest.mark.asyncio
async def test_a_new_tool_call_or_conclude_resets_the_consecutive_count():
    """The counter tracks a *consecutive* streak, not a cumulative total —
    a model that alternates observations with real progress (a new tool
    call) must never be cut off just because it used record_observation
    more than once across the whole investigation."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            RecordObservation(text="Checking baseline first."),
            ProposeToolCall(tool_id="database.get_health", reason="Baseline.", arguments={}, target={}),
            RecordObservation(text="Health looks fine."),
            ProposeToolCall(tool_id="database.get_health", reason="Recheck.", arguments={}, target={}),
            Conclude(summary="No issue found after checking twice."),
        ]
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 5  # every offered action was actually consumed
    assert reply.status == "ok"
    assert "No issue found after checking twice" in reply.text


@pytest.mark.asyncio
async def test_the_replication_playbook_no_longer_burns_the_full_turn_cap():
    """The exact live scenario this fix closes: all 3 replication-playbook
    steps run correctly, then the model keeps 'observing' instead of
    concluding — must stop well short of turn 6, not exhaust it."""
    state, investigation = _state_and_investigation(playbook_id="replication")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[RecordObservation(text="No replication configured, re-confirming.")] * 6
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    # 3 deterministic playbook steps (no LLM call) + only 2 of the 6
    # offered observations before the early exit fires.
    assert len(llm.calls) == 2
    assert investigation.turn_count == 5
    assert "without reaching a confirmed root cause" in reply.text


def test_the_nudge_only_appears_after_an_observation_has_been_recorded():
    from numi.agent.context_manager import InvestigationState

    investigation = InvestigationState(investigation_id="inv1", problem="check CoreBanking")
    assert "MUST use action=conclude" not in AgentOrchestrator._problem_statement_for_llm(investigation)

    investigation.consecutive_record_observations = 1
    assert "MUST use action=conclude" in AgentOrchestrator._problem_statement_for_llm(investigation)
