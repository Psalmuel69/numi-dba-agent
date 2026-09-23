"""Confirms the turn-budget math when a playbook's fixed steps are followed
by freeform "not enough evidence yet, propose more" extensions (see
test_playbook_freeform_extension.py for the mechanism itself): the two
never get separate budgets. Reading `_run_investigation_loop` directly (not
assuming): `investigation.turn_count` is incremented once for EVERY
deterministic playbook step (`_next_playbook_action` branch) exactly like
it is for every non-clarification `decide_next_action` turn (the freeform
branch) — both draw from the exact same `_MAX_INVESTIGATION_TURNS` counter,
checked by the same `while investigation.turn_count < _MAX_INVESTIGATION_TURNS`
condition. A playbook can never buy an investigation extra turns; a long
playbook simply leaves fewer freeform turns available afterward.

This also confirms the corollary the task cares about: a model that keeps
saying "not enough, need more" forever cannot use a playbook to dodge the
existing turn-cap / final-chance-to-conclude safety net (see
test_final_chance_to_conclude.py) — it still fires exactly once, exactly
where it always did, regardless of how the budget was spent getting there."""

from __future__ import annotations

import pytest

from numi.agent.planner.actions import AskClarification, Conclude, ProposeToolCall
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeLLM,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)

# blocking has 4 fixed steps (see playbooks/library.py) — asserted directly
# so this test fails loudly, not silently, if that shape ever changes.
_BLOCKING_STEP_COUNT = 4


@pytest.mark.asyncio
async def test_playbook_steps_and_freeform_extensions_share_one_turn_budget():
    state, investigation = _state_and_investigation(playbook_id="blocking")
    tool_client = _FakeToolClient()
    # Exactly enough freeform "inconclusive" extensions to fill the budget
    # left after the playbook's 4 fixed steps (6 - 4 = 2), then a
    # cooperative final-chance conclude (the one bounded extra call outside
    # the main loop — see _MAX_INVESTIGATION_TURNS's own docstring).
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.get_health", reason="Still unclear.", arguments={}, target={}
            ),
            ProposeToolCall(
                tool_id="database.get_top_queries", reason="Still unclear.", arguments={}, target={}
            ),
            Conclude(summary="Resolved on the last available turn."),
        ]
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    # 4 fixed playbook steps + 2 freeform diagnostic calls = 6 real tool
    # calls, all through the same pipeline.
    assert len(tool_client.requests) == _BLOCKING_STEP_COUNT + 2
    # turn_count reflects every one of those (playbook steps included) —
    # never counted separately, and it never exceeds the cap.
    assert investigation.turn_count == 6
    # 2 freeform decide_next_action calls (inside the main loop) + exactly
    # 1 bounded final-chance call once the budget ran out = 3, not one per
    # playbook step (those stay zero-LLM-call).
    assert len(llm.calls) == 3
    # The final-chance call offers no further tool calls — same safety net
    # a fully freeform investigation gets, unchanged by a playbook having
    # been involved earlier in the same investigation.
    assert llm.calls[-1]["available_tool_ids"] == []
    assert reply.status == "ok"
    assert "Resolved on the last available turn." in reply.text
    assert "without reaching a confirmed root cause" not in reply.text


@pytest.mark.asyncio
async def test_an_investigation_cannot_bypass_the_turn_cap_via_endless_playbook_extensions():
    """The model never converges — every freeform turn after the playbook,
    and even the one bounded final chance, keep asking for more instead of
    concluding. Must still terminate at the existing safe fallback, never
    loop forever or grant extra turns just because a playbook was active."""
    state, investigation = _state_and_investigation(playbook_id="blocking")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.get_health", reason="Need more.", arguments={}, target={}
            ),
            ProposeToolCall(
                tool_id="database.get_top_queries", reason="Need more still.", arguments={}, target={}
            ),
            # Ignores "you MUST respond with action=conclude now" on the one
            # bounded final-chance call — mirrors
            # test_an_uncooperative_final_chance_still_falls_through_to_the_
            # safe_fallback in test_final_chance_to_conclude.py, combined
            # here with a playbook actually having run first.
            AskClarification(question="Which server did you mean?"),
        ]
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert investigation.turn_count == 6  # never exceeds _MAX_INVESTIGATION_TURNS
    assert len(tool_client.requests) == _BLOCKING_STEP_COUNT + 2
    # Exactly 3 LLM calls total (2 in-budget + 1 final chance) — never more,
    # proving there is no way to keep extending turns forever by having the
    # model perpetually claim "not enough evidence yet".
    assert len(llm.calls) == 3
    # Every step and freeform call here is a read (get_health/get_top_
    # queries/the blocking playbook's own steps) — no write ever ran, so
    # this collapses to CONCLUDED_NO_ACTION (see InvestigationStage).
    assert investigation.status == "CONCLUDED_NO_ACTION"
    assert "without reaching a confirmed root cause" in reply.text
