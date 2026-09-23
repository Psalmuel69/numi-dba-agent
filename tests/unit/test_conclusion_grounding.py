"""A Conclude action's free-text fields are never validated against
anything by Pydantic — `summary`/`likely_root_cause`/`recommendation` can
say whatever the model wants. Reproduces a live finding: a real conclusion
named three CamelCase-looking table names (`AccountBalanceOutstandings`,
`AccountBalances`, `TransactionPostingHistory_2`) that do not exist in the
database at all, instead of the real table names its own tool call had
actually returned — confirmed by querying the live database directly.
`_ungrounded_identifiers` (and its wiring into the Conclude branch of
`_run_investigation_loop`) is the fix: reject a conclusion naming something
that never appeared anywhere in this investigation, and give the model one
more bounded chance to revise it."""

from __future__ import annotations

import pytest

from numi.agent.orchestrator import _ungrounded_identifiers
from numi.agent.planner.actions import Conclude
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeLLM,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)


def _investigation_with(*, transcript=None, problem="check CoreBanking", evidence=None):
    from numi.agent.context_manager import InvestigationState

    return InvestigationState(
        investigation_id="inv1",
        problem=problem,
        transcript=transcript or [],
        evidence=evidence or [],
    )


def test_flags_a_camelcase_name_never_seen_anywhere():
    investigation = _investigation_with(
        transcript=[
            {"tool_id": "database.get_top_queries", "reason": "x", "result": {"rows": [{"table": "Branch"}]}}
        ]
    )
    conclusion = Conclude(
        summary="Driven by AccountBalanceOutstandings.",
        likely_root_cause="Heavy scans on AccountBalanceOutstandings and TransactionPostingHistory.",
    )
    assert set(_ungrounded_identifiers(conclusion, investigation)) == {
        "AccountBalanceOutstandings",
        "TransactionPostingHistory",
    }


def test_allows_a_name_present_in_the_transcript():
    investigation = _investigation_with(
        transcript=[
            {
                "tool_id": "database.get_top_queries",
                "reason": "x",
                "result": {"rows": [{"table": "OrderHistory"}]},
            }
        ]
    )
    conclusion = Conclude(summary="Driven by OrderHistory scans.")
    assert _ungrounded_identifiers(conclusion, investigation) == []


def test_allows_a_name_from_the_dbas_own_problem_statement():
    """The DBA's own terminology (e.g. a real SQL Server feature name) must
    never be flagged just because it isn't a tool call's own output."""
    investigation = _investigation_with(problem="check AlwaysOn replication status on CoreBanking")
    conclusion = Conclude(summary="AlwaysOn is healthy on CoreBanking.")
    assert _ungrounded_identifiers(conclusion, investigation) == []


def test_dedupes_repeated_names():
    investigation = _investigation_with()
    conclusion = Conclude(
        summary="FakeTableName appears twice.", likely_root_cause="FakeTableName is the cause."
    )
    assert _ungrounded_identifiers(conclusion, investigation) == ["FakeTableName"]


def test_ignores_a_conclusion_with_no_camelcase_names():
    investigation = _investigation_with()
    conclusion = Conclude(summary="No blocking or high CPU found.", recommendation="Check again later.")
    assert _ungrounded_identifiers(conclusion, investigation) == []


@pytest.mark.asyncio
async def test_an_ungrounded_conclusion_is_rejected_and_the_loop_retries():
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    bad = Conclude(summary="Driven by AccountBalanceOutstandings.")
    good = Conclude(summary="Driven by heavy CPU load, no fabricated names here.")
    llm = _FakeLLM(actions=[bad, good])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 2  # rejected once, accepted on the second try
    assert reply.status == "ok"
    assert "Summary: Driven by AccountBalanceOutstandings" not in reply.text
    assert "Summary: Driven by heavy CPU load" in reply.text
    assert investigation.transcript[-1]["tool_id"] == "internal.grounding_check"
    assert "AccountBalanceOutstandings" in investigation.transcript[-1]["result"]["rejected"]


@pytest.mark.asyncio
async def test_repeated_ungrounded_conclusions_fall_through_to_the_safe_fallback():
    """The rejection is bounded by the same turn cap as everything else —
    a model that keeps insisting on an unverified claim never gets it
    shown to the DBA; it just falls through to the generic "no confirmed
    root cause" message instead. Includes the one bounded last-chance
    call the turn cap now gets (see _finalize_conclude) — it must reject
    an ungrounded conclusion there too, not accept it just because it's
    the final attempt."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    bad = Conclude(summary="Driven by AccountBalanceOutstandings.")
    llm = _FakeLLM(actions=[bad] * 7)  # _MAX_INVESTIGATION_TURNS + the last-chance call
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert "Summary:" not in reply.text  # the fallback path, never the rejected conclusion's own text
    assert "without reaching a confirmed root cause" in reply.text
