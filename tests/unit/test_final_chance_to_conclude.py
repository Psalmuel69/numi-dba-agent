"""Reproduces a live finding: a real investigation ("terminate the blocking
query on postgres-local") used its blocking playbook's 4 deterministic
steps, then its last 2 turns on legitimate remediation attempts
(kill_session, then cancel_query as a fallback) that both turned up
nothing to act on — genuinely useful information — but hit
_MAX_INVESTIGATION_TURNS with zero turns left to report that, and the DBA
got the generic "no confirmed root cause" fallback instead of an actual
answer. `_run_investigation_loop` now spends exactly one bounded, final
call asking the model to conclude with whatever it has, before ever
falling back to that generic message."""

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


@pytest.mark.asyncio
async def test_a_cooperative_final_chance_produces_a_real_conclusion_not_the_fallback():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(tool_id="database.get_health", reason="Baseline.", arguments={}, target={})
        ]
        * 6  # fills the entire turn budget without ever concluding
        + [Conclude(summary="Nothing found blocking; two remediation attempts both no-ops.")]
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 7  # the 6 in-budget turns + exactly one last-chance call
    assert reply.status == "ok"
    assert "Nothing found blocking; two remediation attempts both no-ops." in reply.text
    assert "without reaching a confirmed root cause" not in reply.text


@pytest.mark.asyncio
async def test_the_final_chance_offers_no_further_tool_calls():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(tool_id="database.get_health", reason="Baseline.", arguments={}, target={})
        ]
        * 6
        + [Conclude(summary="Done.")]
    )
    orchestrator = _orchestrator(tool_client)

    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert llm.calls[-1]["available_tool_ids"] == []


@pytest.mark.asyncio
async def test_an_uncooperative_final_chance_still_falls_through_to_the_safe_fallback():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(tool_id="database.get_health", reason="Baseline.", arguments={}, target={})
        ]
        * 6
        + [AskClarification(question="Which server did you mean?")]  # ignores "you MUST conclude"
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 7  # still exactly one bounded extra call, never a loop
    assert "without reaching a confirmed root cause" in reply.text
    # No write/verification was ever involved (only get_health reads) — the
    # bounded set of CONCLUDED_* stages (see InvestigationStage on
    # InvestigationState) collapses to CONCLUDED_NO_ACTION here.
    assert investigation.status == "CONCLUDED_NO_ACTION"
    assert investigation.is_concluded is True
