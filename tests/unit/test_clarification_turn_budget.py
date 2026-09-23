"""A clarification (the model asking "which server?", "which database?",
...) is the DBA narrowing down a target, not the model looping on
diagnostics — the thing `_MAX_INVESTIGATION_TURNS` exists to bound.
Reproduced live: a multi-step targeting dialogue (environment -> server ->
database) burned most of the shared 6-turn budget on clarifications alone,
then hit the cap right as real diagnostics started succeeding — the DBA got
"I've run several diagnostic steps without reaching a confirmed root
cause" instead of an actual answer, even though two real tool calls had
just completed.

`_MAX_CLARIFICATION_TURNS` bounds a run of *consecutive* AskClarification
turns independently instead, so an unresolved back-and-forth still can't
run forever across many separate requests (turn_count alone never catches
that, since it's never incremented for a clarification)."""

from __future__ import annotations

import pytest

from numi.agent.planner.actions import AskClarification, Conclude
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeLLM,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)


@pytest.mark.asyncio
async def test_a_clarification_does_not_consume_the_diagnostic_turn_budget():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[AskClarification(question="Which server?")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "clarification"
    assert investigation.turn_count == 0
    assert investigation.clarification_count == 1
    # The loop is now genuinely blocked on the DBA's answer — reflected in
    # `status` (not just `clarification_count`) so `/status` can say so.
    assert investigation.status == "AWAITING_CLARIFICATION"
    assert investigation.effective_status == "AWAITING_CLARIFICATION"


@pytest.mark.asyncio
async def test_a_real_action_resets_the_clarification_streak():
    state, investigation = _state_and_investigation(playbook_id=None)
    investigation.clarification_count = 3  # one short of the cap
    investigation.status = "AWAITING_CLARIFICATION"  # left over from an earlier turn
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Resolved after all.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "ok"
    assert investigation.clarification_count == 0
    # A real action (here, a Conclude) means the investigation was no
    # longer actually blocked on a clarification the moment this call
    # resumed it — status must not still say AWAITING_CLARIFICATION.
    assert investigation.status == "CONCLUDED_NO_ACTION"


@pytest.mark.asyncio
async def test_repeated_clarifications_eventually_give_up_instead_of_looping_forever():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    # More than _MAX_CLARIFICATION_TURNS (4) offered — the fix must stop
    # well before consuming all of them.
    llm = _FakeLLM(actions=[AskClarification(question="Which one?")] * 6)
    orchestrator = _orchestrator(tool_client)

    # Simulates 5 separate requests, each resuming the same investigation —
    # a clarification never increments turn_count, so the outer while loop
    # would otherwise never stop calling the LLM across repeated calls.
    replies = []
    for _ in range(5):
        replies.append(
            await orchestrator._run_investigation_loop(
                state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
            )
        )

    assert len(llm.calls) == 5  # stopped asking, not exhausted the offered 6
    # Only AskClarification was ever offered — no tool call ran, so this
    # collapses to CONCLUDED_NO_ACTION (see InvestigationStage).
    assert investigation.status == "CONCLUDED_NO_ACTION"
    assert "restate what you'd like me to check" in replies[-1].text
    assert replies[-1].status != "clarification"


@pytest.mark.asyncio
async def test_a_real_action_resets_the_streak_across_separate_resumed_calls():
    """The realistic shape: each clarification is its own separate request
    (a new Slack message), not one long-lived loop — the streak has to
    persist and reset correctly across independent
    `_run_investigation_loop` calls sharing the same investigation."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    orchestrator = _orchestrator(tool_client)

    llm = _FakeLLM(actions=[AskClarification(question="Which server?")])
    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )
    llm._actions.append(AskClarification(question="Which database?"))
    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )
    assert investigation.clarification_count == 2
    # Still blocked on the DBA across these separate, resumed calls — not
    # just a within-one-call detail.
    assert investigation.status == "AWAITING_CLARIFICATION"

    llm._actions.append(Conclude(summary="Resolved after all."))
    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert investigation.clarification_count == 0
    assert investigation.status == "CONCLUDED_NO_ACTION"
