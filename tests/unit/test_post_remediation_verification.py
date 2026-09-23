"""Structural post-remediation verification (README.md's lifecycle diagram:
"... → DATABASE → VERIFICATION → AUDIT → AI DBA → USER" — a shared external
playbook spec's own mandate is even more explicit: "Never mark an incident
as resolved merely because an action was submitted. Resolution requires
independent verification.").

Reproduces the exact live-observed gap: whether a remediation actually gets
re-checked after execution was previously left entirely to the model's own
discretion within its turn budget. A real model has voluntarily said things
like "Subsequent session and blocking checks confirmed that session X has
been successfully terminated" — but nothing ever forced that, so a DBA
could just as easily get a clean "Completed" summary when the underlying
condition never actually cleared.

`InvestigationState.pending_verification`/`last_verification`,
`orchestrator._VERIFICATION_TOOLS_BY_WRITE_TOOL`,
`orchestrator._verification_still_shows_condition`, and the check wired
into `_finalize_conclude`/`_format_report` are the fix — mirroring the
existing `_ungrounded_identifiers` grounding-check self-correction pattern
(see test_conclusion_grounding.py): reject a Conclude that would report a
write as done without an independent re-check, and give the model one more
bounded try, never an infinite loop."""

from __future__ import annotations

import pytest

from numi.agent.orchestrator import _verification_still_shows_condition
from numi.agent.planner.actions import Conclude, ProposeToolCall
from numi.common.models.tool import OperationType, ToolCallResponse, ToolCallStatus
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeLLM,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)


def _kill_session_call(session_id: str = "500") -> ProposeToolCall:
    return ProposeToolCall(
        tool_id="database.kill_session",
        arguments={"session_id": session_id, "reason": "Killing the head blocker."},
        target={},
        reason="Terminating session 500, the head blocker.",
    )


def _blocking_check_call() -> ProposeToolCall:
    return ProposeToolCall(
        tool_id="database.get_blocking_sessions",
        arguments={},
        target={},
        reason="Confirming the blocking chain is actually clear now.",
    )


_KILL_EXECUTED = ToolCallResponse(
    status=ToolCallStatus.EXECUTED,
    message="Completed.",
    result={"affected": {"terminated": True, "session_id": "500"}},
)


# --- Direct unit tests of the helper --------------------------------------


def test_verification_still_shows_condition_true_when_session_still_blocking():
    result = {"rows": [{"blocked_session_id": "12", "blocking_session_id": "500"}]}
    assert _verification_still_shows_condition("database.get_blocking_sessions", "500", result) is True


def test_verification_still_shows_condition_false_when_session_gone():
    result = {"rows": []}
    assert _verification_still_shows_condition("database.get_blocking_sessions", "500", result) is False


def test_verification_still_shows_condition_checks_get_sessions_own_field():
    result = {"rows": [{"session_id": "500", "state": "idle"}]}
    assert _verification_still_shows_condition("database.get_sessions", "500", result) is True
    assert _verification_still_shows_condition("database.get_sessions", "999", result) is False


def test_verification_still_shows_condition_never_guesses_without_a_session_id():
    result = {"rows": [{"session_id": "500"}]}
    assert _verification_still_shows_condition("database.get_sessions", None, result) is False


# --- The exact live-observed pattern: write, then immediate Conclude -----


@pytest.mark.asyncio
async def test_kill_session_then_immediate_conclude_without_recheck_is_rejected():
    """Reproduces the exact previously-observed live pattern: the model
    proposes database.kill_session, gets EXECUTED (terminated=True), then
    immediately proposes Conclude claiming success WITHOUT ever having
    re-checked blocking/sessions afterward. Must be rejected/nudged, not
    accepted at face value — the loop gets a bounded retry, exactly like an
    ungrounded conclusion."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient(responses=[_KILL_EXECUTED])
    bad_conclude = Conclude(summary="Done — session 500 has been terminated. Blocking is resolved.")
    good_recheck = _blocking_check_call()
    good_conclude = Conclude(summary="Session 500 terminated and confirmed gone from blocking.")
    llm = _FakeLLM(actions=[_kill_session_call(), bad_conclude, good_recheck, good_conclude])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 4  # kill, rejected conclude, the recheck, accepted conclude
    assert investigation.transcript[1]["tool_id"] == "internal.verification_check"
    assert "kill_session" in investigation.transcript[1]["result"]["message"]
    assert reply.status == "ok"
    # The eventual recheck (default fake-client response, empty rows) shows
    # the session is gone — RESOLVED, so the final stage is VERIFIED even
    # though the first Conclude attempt was rejected along the way.
    assert investigation.status == "CONCLUDED_VERIFIED"


@pytest.mark.asyncio
async def test_a_proper_recheck_afterward_is_accepted_with_a_verified_framing():
    """The other half of the same scenario: the model DOES properly
    re-check afterward (get_blocking_sessions comes back with no rows for
    session 500) before concluding — accepted normally, and the reply
    states the outcome was independently verified, not just a bare
    'Completed'."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient(
        responses=[
            _KILL_EXECUTED,
            ToolCallResponse(
                status=ToolCallStatus.EXECUTED, message="ok", result={"rows": [], "row_count": 0}
            ),
        ]
    )
    conclude = Conclude(summary="Killed the head blocker; blocking is now clear.", confidence="confirmed")
    llm = _FakeLLM(actions=[_kill_session_call(), _blocking_check_call(), conclude])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 3  # no rejection needed this time
    assert reply.status == "ok"
    assert investigation.last_verification == "RESOLVED"
    assert "independently re-checked afterward and confirmed resolved" in reply.text
    assert "internal.verification_check" not in [t.get("tool_id") for t in investigation.transcript]
    # investigation.status is derived from these same last_verification/
    # pending_verification signals (see AgentOrchestrator._conclusion_stage)
    # — never a second, independent judgment that could disagree.
    assert investigation.status == "CONCLUDED_VERIFIED"
    assert investigation.is_concluded is True


@pytest.mark.asyncio
async def test_a_recheck_showing_the_session_is_still_there_is_reported_unresolved():
    """The remediation executed, and a recheck DID happen — but it shows
    the write didn't actually take effect. The reply must say so plainly,
    not report a bare 'Completed' for a kill that didn't actually clear
    anything."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient(
        responses=[
            _KILL_EXECUTED,
            ToolCallResponse(
                status=ToolCallStatus.EXECUTED,
                message="ok",
                result={
                    "rows": [{"blocked_session_id": "12", "blocking_session_id": "500"}],
                    "row_count": 1,
                },
            ),
        ]
    )
    conclude = Conclude(summary="Killed session 500.")
    llm = _FakeLLM(actions=[_kill_session_call(), _blocking_check_call(), conclude])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "ok"
    assert investigation.last_verification == "UNRESOLVED"
    assert "did NOT actually resolve" in reply.text
    assert investigation.status == "CONCLUDED_UNRESOLVED"


@pytest.mark.asyncio
async def test_repeated_conclude_without_ever_rechecking_falls_through_to_unverified_framing():
    """Bounded by the same turn budget as the grounding check — a model
    that never goes back to actually check still can't loop forever. The
    one bounded last-chance call (after the turn cap, no further tool
    calls offered) accepts the conclusion rather than throwing away
    everything the investigation did, but must plainly say the write was
    never independently verified — never silently keep calling it
    'Completed'."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient(responses=[_KILL_EXECUTED])
    bad_conclude = Conclude(summary="Done — session 500 terminated.")
    # 1 kill_session call + 6 Conclude attempts (5 rejected mid-loop + 1
    # accepted on the final bounded last-chance call) = 7 actions, matching
    # _MAX_INVESTIGATION_TURNS(6) + the one last-chance call.
    llm = _FakeLLM(actions=[_kill_session_call(), *([bad_conclude] * 6)])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(tool_client.requests) == 1  # only the original kill_session — no recheck ever happened
    assert reply.status == "ok"  # accepted, not the generic no-root-cause fallback
    assert "NOT independently verified" in reply.text
    assert investigation.pending_verification is not None  # still pending — never silently cleared
    assert investigation.status == "CONCLUDED_UNVERIFIED"
    # Found live: the identical bad_conclude was rejected 5 times mid-loop
    # (same reason, every time, since nothing in this scenario ever changes
    # between attempts), and each rejection used to append the same
    # sentence to investigation.evidence verbatim — a DBA reading the final
    # summary saw "(a draft conclusion after database.kill_session was
    # rejected...)" repeated 5 times in a row. The evidence list (DBA-
    # facing) must collapse consecutive identical rejections to one entry;
    # only the transcript (what the model itself sees to try to
    # course-correct) repeats it every turn.
    rejection_note = (
        "(a draft conclusion after database.kill_session was rejected — "
        "not yet independently verified)"
    )
    assert investigation.evidence.count(rejection_note) == 1
    assert (
        len([t for t in investigation.transcript if t.get("tool_id") == "internal.verification_check"]) == 5
    )


# --- Scoping: only the mapped writes trigger this, and only real writes --


@pytest.mark.asyncio
async def test_a_write_tool_with_no_known_correlated_check_is_unaffected():
    """update_statistics has no cheap, obvious correlated re-check (see
    _VERIFICATION_TOOLS_BY_WRITE_TOOL's own docstring) — must behave
    exactly as before this feature, no pending_verification, no rejection."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient(
        responses=[
            ToolCallResponse(status=ToolCallStatus.EXECUTED, message="Completed.", result={"affected": {}})
        ]
    )
    update_stats = ProposeToolCall(
        tool_id="database.update_statistics",
        arguments={"schema": "dbo", "table": "Orders", "reason": "Stale stats."},
        target={},
        reason="Refreshing stale statistics on dbo.Orders.",
    )
    conclude = Conclude(summary="Refreshed statistics on dbo.Orders.")
    llm = _FakeLLM(actions=[update_stats, conclude])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 2  # accepted on the first conclude attempt, no rejection
    assert reply.status == "ok"
    assert investigation.pending_verification is None
    assert "Verification:" not in reply.text
    # A write ran, but one with no correlated re-check at all — never
    # CONCLUDED_UNVERIFIED (that's reserved for a write that DOES have one
    # and simply never got to it), just the same plain outcome as any
    # investigation with no pending_verification/last_verification signal.
    assert investigation.status == "CONCLUDED_NO_ACTION"


@pytest.mark.asyncio
async def test_tool_operation_types_is_a_defense_in_depth_confirmation():
    """If the tool catalog itself no longer classifies kill_session as a
    WRITE (an edge case, but the whole point of checking `operation_type`
    rather than trusting the static mapping alone), this must not set
    pending_verification — mirrors _strip_unschematized_arguments's own
    belt-and-suspenders reasoning."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient(responses=[_KILL_EXECUTED])
    conclude = Conclude(summary="Killed session 500.")
    llm = _FakeLLM(actions=[_kill_session_call(), conclude])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state,
        investigation,
        _ALL_READ_TOOL_IDS,
        "dev",
        "dba_l2@example.com",
        llm,
        None,
        None,
        {"database.kill_session": OperationType.READ},  # misclassified on purpose
    )

    assert len(llm.calls) == 2  # accepted immediately — no verification check applied
    assert reply.status == "ok"
    assert investigation.pending_verification is None
