"""The bounded investigation-stage model (see `InvestigationStage` on
`context_manager.InvestigationState`): a small, deliberately incomplete
subset of a much richer 17-state external playbook-spec lifecycle
(NEW -> TRIAGED -> ... -> AWAITING_APPROVAL -> ... -> RESOLVED/CLOSED),
scoped to exactly the stages this codebase's own control flow can actually
and accurately observe itself transitioning through:

- INVESTIGATING / AWAITING_CLARIFICATION / CONCLUDED_* are real values
  `investigation.status` is assigned in `orchestrator.py` — see
  test_clarification_turn_budget.py, test_post_remediation_verification.py,
  and the other files that reproduce each transition through the real
  investigation loop.
- AWAITING_VERIFICATION is deliberately never one of those assigned
  values — it's computed on the fly by `InvestigationState.effective_status`
  from `pending_verification` (the existing, already-authoritative record
  of whether a post-write re-check is outstanding), so there is exactly one
  place that decides whether a re-check is pending, never two that could
  drift apart.

This file covers the two pieces those other files don't: the plain
dataclass properties themselves (`is_concluded`/`effective_status`), and
`/status`'s own plain-language phrasing for each stage — including
AWAITING_VERIFICATION, which (being derived rather than stored) needs a
scenario where a write's pending_verification is set without yet reaching
a Conclude, something the other verification tests never stop to observe
mid-flight."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import InvestigationState
from tests.unit.test_orchestrator_playbooks import _FakeToolClient, _orchestrator, _state_and_investigation
from tests.unit.test_post_remediation_verification import _KILL_EXECUTED, _kill_session_call

# --- InvestigationState.is_concluded / effective_status, in isolation ----


def test_is_concluded_true_for_every_concluded_stage_and_false_for_the_others():
    for stage in (
        "CONCLUDED_VERIFIED",
        "CONCLUDED_UNRESOLVED",
        "CONCLUDED_UNVERIFIED",
        "CONCLUDED_NO_ACTION",
    ):
        inv = InvestigationState(investigation_id="x", problem="p", status=stage)
        assert inv.is_concluded is True, stage

    for stage in ("INVESTIGATING", "AWAITING_CLARIFICATION"):
        inv = InvestigationState(investigation_id="x", problem="p", status=stage)
        assert inv.is_concluded is False, stage


def test_effective_status_mirrors_pending_verification_only_while_plain_investigating():
    inv = InvestigationState(investigation_id="x", problem="p")
    assert inv.effective_status == "INVESTIGATING"

    inv.pending_verification = {
        "tool_id": "database.kill_session",
        "verification_tools": ("database.get_blocking_sessions",),
        "session_id": "500",
        "reason": "Killing the head blocker.",
    }
    assert inv.effective_status == "AWAITING_VERIFICATION"

    # A clarification actively blocking the loop right now takes precedence
    # over a re-check that can simply wait.
    inv.status = "AWAITING_CLARIFICATION"
    assert inv.effective_status == "AWAITING_CLARIFICATION"

    # A concluded investigation is final — pending_verification left set
    # (e.g. the CONCLUDED_UNVERIFIED last-chance case) never resurrects it.
    inv.status = "CONCLUDED_UNVERIFIED"
    assert inv.effective_status == "CONCLUDED_UNVERIFIED"

    # Clearing pending_verification and returning to plain INVESTIGATING
    # drops back to reporting that directly — never a stale
    # AWAITING_VERIFICATION once the signal it mirrors is gone.
    inv.status = "INVESTIGATING"
    inv.pending_verification = None
    assert inv.effective_status == "INVESTIGATING"


# --- /status phrasing for each stage --------------------------------------


def test_status_reports_no_active_investigation_when_none_exists():
    state, _ = _state_and_investigation(playbook_id=None)
    orchestrator = _orchestrator(_FakeToolClient())

    reply = orchestrator._status_reply(state)

    assert reply.text == "No active investigation on this conversation."


def test_status_reports_plain_investigating():
    state, investigation = _state_and_investigation(playbook_id=None)
    state.investigation = investigation
    orchestrator = _orchestrator(_FakeToolClient())

    reply = orchestrator._status_reply(state)

    assert "Investigating." in reply.text


def test_status_reports_awaiting_clarification():
    state, investigation = _state_and_investigation(playbook_id=None)
    investigation.status = "AWAITING_CLARIFICATION"
    state.investigation = investigation
    orchestrator = _orchestrator(_FakeToolClient())

    reply = orchestrator._status_reply(state)

    assert "Awaiting your answer to a clarifying question." in reply.text


@pytest.mark.asyncio
async def test_status_reports_awaiting_verification_of_the_specific_tool_while_pending():
    """`pending_verification` is set by a real EXECUTED kill_session going
    through `_submit_and_relay` (the actual mechanism — see
    test_post_remediation_verification.py), not fabricated by hand, so this
    is the genuine mid-investigation moment `/status` needs to describe:
    the write ran, but the loop hasn't gotten to (or hasn't yet returned
    from) the correlated re-check."""
    state, investigation = _state_and_investigation(playbook_id=None)
    state.investigation = investigation
    tool_client = _FakeToolClient(responses=[_KILL_EXECUTED])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._submit_and_relay(
        state, investigation, _kill_session_call(), "dev", "dba_l2@example.com", None, None
    )

    assert reply is None  # executed — the loop would simply continue
    assert investigation.pending_verification is not None
    assert investigation.status == "INVESTIGATING"  # never itself set to AWAITING_VERIFICATION
    assert investigation.effective_status == "AWAITING_VERIFICATION"

    status_reply = orchestrator._status_reply(state)
    assert "Awaiting independent verification of database.kill_session." in status_reply.text


@pytest.mark.parametrize(
    ("stage", "expected_phrase"),
    [
        ("CONCLUDED_VERIFIED", "independently re-checked and confirmed resolved"),
        ("CONCLUDED_UNRESOLVED", "did NOT resolve the condition"),
        ("CONCLUDED_UNVERIFIED", "never independently verified"),
        ("CONCLUDED_NO_ACTION", "Concluded."),
    ],
)
def test_status_reports_the_right_phrasing_for_each_concluded_outcome(stage, expected_phrase):
    state, investigation = _state_and_investigation(playbook_id=None)
    investigation.status = stage
    state.investigation = investigation
    orchestrator = _orchestrator(_FakeToolClient())

    reply = orchestrator._status_reply(state)

    assert expected_phrase in reply.text


def test_status_still_shows_the_playbook_step_note_alongside_the_stage():
    """The stage phrasing is additive — the existing playbook/step note
    (see test_orchestrator_playbooks.py::
    test_status_command_reports_the_active_playbook_and_step) must still be
    there too, unchanged."""
    state, investigation = _state_and_investigation(playbook_id="high_cpu")
    investigation.playbook_step = 2
    state.investigation = investigation
    orchestrator = _orchestrator(_FakeToolClient())

    reply = orchestrator._status_reply(state)

    assert "Investigating." in reply.text
    assert "High CPU Investigation" in reply.text
    assert "2/4" in reply.text
