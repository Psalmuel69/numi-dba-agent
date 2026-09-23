"""`AgentOrchestrator.run_comprehensive_summary` — the per-server building
block the scheduled daily digest calls once per registered server (see
agent.scheduled_report).

The central claim under test is *reuse*: this entry point must run the real
`comprehensive_summary` playbook through the real investigation loop, not a
parallel reimplementation of it. So the expected tool sequence here is read
out of `playbooks.library` itself rather than written down a second time — if
someone changes the playbook's steps, this test follows them automatically,
and if someone reimplements the sweep somewhere else, it stops matching.
"""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ContextManager
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, CritiqueVerdict
from numi.agent.playbooks.library import get_playbook
from numi.common.config import Settings
from numi.common.models.tool import ToolCallResponse, ToolCallStatus
from numi.gateway.domain.tool_catalog import build_tool_catalog

# The real, production tool catalog — including each tool's real
# `operation_type`. Deliberately not a hand-written list of fake
# ToolDefinitions: the read-only guarantee this feature rests on is defined
# in terms of the catalog's own READ/WRITE classification, so a test that
# invented its own classification would be testing a different system than
# the one that ships.
_REAL_TOOLS = build_tool_catalog(Settings(_env_file=None))

# What the playbook itself says it checks. Read, never restated.
_EXPECTED_STEPS = [step.tool_id for step in get_playbook("comprehensive_summary").steps]


class _FakeToolClient:
    def __init__(self, *, responses: list[ToolCallResponse] | None = None):
        self._responses = responses
        self.requests: list[object] = []

    async def available_tools(self, channel: str, channel_account_id: str):
        return _REAL_TOOLS

    async def submit(self, request):
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return ToolCallResponse(
            status=ToolCallStatus.EXECUTED, message="ok", result={"rows": [], "row_count": 0}
        )

    async def create_investigation(self, request):
        pass

    async def update_investigation(self, investigation_id, request):
        pass

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        return []

    async def get_cross_server_patterns(
        self, *, playbook_id, environment=None, exclude_server_id=None, limit=5
    ):
        return []

    async def log_decision_event(self, request):
        pass


class _FakeLLM:
    def __init__(self, actions: list):
        self._actions = list(actions)
        self.calls: list[dict] = []

    async def decide_next_action(self, **kwargs):
        self.calls.append(kwargs)
        return self._actions.pop(0)

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


def _orchestrator(tool_client, llm, context=None) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm), tool_client=tool_client, context=context
    )


@pytest.mark.asyncio
async def test_it_runs_the_real_comprehensive_summary_playbook_steps():
    """Reuse, stated as an assertion: the exact evidence-gathering sequence
    `comprehensive_summary` declares — no more, no fewer, in its order — must
    be what actually gets submitted, with no LLM call in between (that is the
    whole point of a playbook; see agent.playbooks.library's module
    docstring), and exactly one LLM call at the end to interpret it all."""
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Nothing deviating from normal.")])
    orchestrator = _orchestrator(tool_client, llm)

    summary = await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    assert [r.tool_id for r in tool_client.requests] == _EXPECTED_STEPS
    assert len(llm.calls) == 1
    assert summary.ok
    assert summary.server_id == "postgres-local"
    assert summary.environment == "development"
    assert "Nothing deviating from normal." in summary.text


@pytest.mark.asyncio
async def test_the_final_llm_call_carries_the_playbooks_own_conclusion_guidance():
    """The scheduled path must inherit the playbook's "report ONLY
    deviations, then say plainly that everything else came back clean"
    framing — the same guidance an interactive run gets. If this stopped
    arriving, the digest would silently start producing walls of "X is
    fine"."""
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)

    await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    problem = llm.calls[0]["problem_statement"]
    assert "Comprehensive Health Summary" in problem
    assert "all other checks came back clean" in problem
    # ...and the scheduled framing on top of it: text only, nobody watching.
    assert "unattended" in problem
    assert "never as an action you are taking" in problem


@pytest.mark.asyncio
async def test_every_call_targets_the_requested_server_and_environment():
    """The environment is supplied by the caller from the server registry's
    own record rather than asked for (nobody is there to answer), and must
    reach every single tool call's target — an unattended sweep that
    silently targeted the wrong environment would be far worse than one that
    failed."""
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm)

    await orchestrator.run_comprehensive_summary(
        server_id="core-banking-prod",
        environment="production",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    assert tool_client.requests
    for request in tool_client.requests:
        assert request.target["instance"] == "core-banking-prod"
        assert request.target["environment"] == "production"


@pytest.mark.asyncio
async def test_a_server_where_no_diagnostic_succeeded_is_reported_as_not_ok():
    """The failure case the digest exists to make visible. Every step comes
    back FAILED (an unreachable server, a discovery that never completed) —
    the model will still happily write a fluent "nothing concerning found"
    paragraph, and `ScheduledSummary.ok` must not believe it. Decided
    structurally from whether any call actually executed, never by reading
    the prose."""
    failures = [
        ToolCallResponse(status=ToolCallStatus.FAILED, message="database unavailable")
        for _ in _EXPECTED_STEPS
    ]
    tool_client = _FakeToolClient(responses=failures)
    llm = _FakeLLM(actions=[Conclude(summary="Everything looks healthy.")])
    orchestrator = _orchestrator(tool_client, llm)

    summary = await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    assert not summary.ok
    assert "no diagnostic call succeeded" in summary.error
    # The reassuring text still exists, but `ok` is what the digest branches
    # on — this is exactly the "6 checked, all healthy" lie being prevented.
    assert "Everything looks healthy." in summary.text


@pytest.mark.asyncio
async def test_findings_and_recommendations_make_a_server_non_clean():
    """`is_clean` drives whether a server gets its own block in the digest or
    collapses into the closing "all other checks came back clean" line. It
    reads the typed Conclude action's own root-cause/recommendation fields —
    never a keyword scan of the summary text."""
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            Conclude(
                summary="Backup chain is broken.",
                likely_root_cause="No successful full backup in 9 days.",
                confidence="confirmed",
                recommendation="Run a full backup and investigate the failing job.",
            )
        ]
    )
    orchestrator = _orchestrator(tool_client, llm)

    summary = await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    assert summary.ok
    assert not summary.is_clean
    assert summary.findings == ("No successful full backup in 9 days.",)
    assert summary.recommendations == ("Run a full backup and investigate the failing job.",)


@pytest.mark.asyncio
async def test_a_clean_run_is_marked_clean():
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="All checks came back clean.")])
    orchestrator = _orchestrator(tool_client, llm)

    summary = await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    assert summary.ok and summary.is_clean


@pytest.mark.asyncio
async def test_a_scheduled_run_never_registers_a_conversation():
    """A scheduled sweep must be invisible to the conversation layer. If it
    registered state with the ContextManager it could collide with — or
    clobber — a real DBA's live database_context, in-progress investigation,
    or pending approval on whatever conversation_id it happened to reuse, and
    would leave a concluded investigation sitting in a conversation nobody
    started."""
    context = ContextManager()
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client, llm, context=context)

    await orchestrator.run_comprehensive_summary(
        server_id="postgres-local",
        environment="development",
        channel="dev",
        channel_account_id="dba_l2@example.com",
    )

    assert context._conversations == {}
