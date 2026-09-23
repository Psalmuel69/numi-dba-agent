"""Verifies a capability `_run_investigation_loop` appears to support but
that, before this test, had never actually been pinned down: once a matched
playbook's fixed steps are exhausted, `_next_playbook_action` returns None
and the loop falls through to a normal `decide_next_action` call carrying
the FULL, unrestricted `available_ids` — not one narrowed to `Conclude`
only. `_problem_statement_for_llm` only ever nudges ("if the evidence
gathered is enough to conclude, conclude now") — it never forces a
conclusion, so the model can legitimately propose one or more additional,
freeform diagnostic tool calls outside the playbook's fixed step list
before ever concluding.

These tests drive that exact path with a scripted LLM: the blocking
playbook's 4 fixed steps run first (zero LLM calls), then the first
`decide_next_action` call proposes `database.get_health` — a tool NOT in
the blocking playbook's step list — instead of concluding, and only the
*second* `decide_next_action` call concludes. This proves the mechanism
actually works end to end: the extra call is submitted through the real
`_submit_and_relay` pipeline (arguments stripped per `tool_allowed_arguments`
exactly like any other freeform call), its result is folded into evidence
and the transcript, `turn_count` accounts for it out of the SAME shared
`_MAX_INVESTIGATION_TURNS` budget as the playbook's own steps (not a
separate allowance), and the final conclusion is built from all of it.

See also test_turn_budget_playbook_plus_freeform.py for confirmation that
this extension mechanism still cannot outrun the turn cap — a playbook
that never converges still terminates via the existing final-chance/
safe-fallback machinery exactly as a fully freeform investigation would."""

from __future__ import annotations

import pytest

from numi.agent.planner.actions import Conclude, ProposeToolCall
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeLLM,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)

# blocking's 4 fixed steps, in order (see playbooks/library.py) — asserted
# against directly so this test breaks loudly if the playbook's shape ever
# changes, rather than silently testing something else.
_BLOCKING_FIXED_STEPS = [
    "database.get_blocking_sessions",
    "database.get_running_queries",
    "database.get_wait_statistics",
    "database.get_sessions",
]


@pytest.mark.asyncio
async def test_a_freeform_tool_call_proposed_after_the_playbook_exhausts_is_actually_submitted():
    state, investigation = _state_and_investigation(playbook_id="blocking")
    tool_client = _FakeToolClient()
    # database.get_health is NOT one of blocking's 4 fixed steps — a genuine
    # extension beyond the playbook, not a step it already contains.
    extra_call = ProposeToolCall(
        tool_id="database.get_health",
        reason="Blocking evidence alone doesn't explain it — checking overall health.",
        arguments={"reason": "should be stripped, not a real get_health argument"},
        target={},
    )
    conclude = Conclude(
        summary="Session 42 is the head blocker, holding a lock for 12 minutes.",
        likely_root_cause="A long-running uncommitted transaction in session 42.",
        recommendation="Kill session 42 after confirming with the app team.",
        confidence="likely",
    )
    llm = _FakeLLM(actions=[extra_call, conclude])
    orchestrator = _orchestrator(tool_client)

    # get_health takes no arguments at all — declaring that here (an empty
    # required list AND an empty allowed-arguments set) exercises the same
    # tool_requirements/tool_allowed_arguments plumbing a real call uses,
    # confirming it's correctly threaded through this specific code path
    # (post-playbook-exhaustion), not just the original freeform path.
    tool_requirements = {t: [] for t in _ALL_READ_TOOL_IDS}
    tool_allowed_arguments = {t: set() for t in _ALL_READ_TOOL_IDS}

    reply = await orchestrator._run_investigation_loop(
        state,
        investigation,
        _ALL_READ_TOOL_IDS,
        "dev",
        "dba_l2@example.com",
        llm,
        tool_requirements,
        tool_allowed_arguments,
    )

    # All 4 fixed steps ran first, then exactly one extra, freeform call —
    # submitted through the real Gateway-facing pipeline (the fake tool
    # client stands in for the Gateway itself; everything upstream of it,
    # including argument stripping, is the real code).
    submitted_ids = [r.tool_id for r in tool_client.requests]
    assert submitted_ids == [*_BLOCKING_FIXED_STEPS, "database.get_health"]

    # tool_allowed_arguments was honored for the extra call exactly as for
    # any other freeform ProposeToolCall — the bogus `reason` key never
    # reached the "Gateway".
    assert tool_client.requests[-1].arguments == {}

    # Exactly one LLM call to propose the extension, one more to conclude —
    # not one per playbook step (those stay zero-LLM-call) and not skipped
    # just because a playbook was active.
    assert len(llm.calls) == 2
    # The first decide_next_action call was offered the FULL tool menu, not
    # one narrowed to conclude-only — this is the crux of the mechanism.
    assert llm.calls[0]["available_tool_ids"] == _ALL_READ_TOOL_IDS
    assert "database.get_health" in llm.calls[0]["available_tool_ids"]

    # turn_count accounts for the extra call out of the SAME shared budget:
    # 4 fixed playbook steps + 1 freeform tool call + 1 conclude call = 6.
    assert investigation.turn_count == 6

    # The extra call's own result is folded into evidence/transcript before
    # the conclude call runs, and the final report reflects it.
    assert any("database.get_health" in e for e in investigation.evidence)
    assert investigation.transcript[-1]["tool_id"] == "database.get_health"
    assert reply.status == "ok"
    assert "Session 42 is the head blocker" in reply.text
    assert "Kill session 42" in reply.text


@pytest.mark.asyncio
async def test_the_freeform_extensions_problem_statement_names_both_directions():
    """The prompt handed to the first post-playbook decide_next_action call
    must not only say what to do when evidence IS enough (conclude) — it
    must also spell out the converse: propose more read-only diagnostics,
    not limited to the playbook's own steps, when it is NOT enough yet."""
    state, investigation = _state_and_investigation(playbook_id="blocking")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(
                tool_id="database.get_health", reason="Need more.", arguments={}, target={}
            ),
            Conclude(summary="done"),
        ]
    )
    orchestrator = _orchestrator(tool_client)

    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    problem = llm.calls[0]["problem_statement"]
    assert "conclude now" in problem.lower()
    assert "not limited to" in problem.lower() or "not limited to this playbook" in problem.lower()
    assert "additional" in problem.lower() and "diagnostic" in problem.lower()
