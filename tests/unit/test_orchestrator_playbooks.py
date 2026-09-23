"""Playbook wiring in the investigation loop (spec §7): when a matched
playbook is active, its steps must run as deterministic tool calls with NO
LLM call in between — that's the entire point (see
agent.playbooks.library's module docstring for the rationale) — and the LLM
must be asked only once, at the end, to interpret the gathered evidence and
conclude. An unmatched problem must behave exactly as before this feature
existed: every turn goes through the LLM."""

from __future__ import annotations

import pytest

from numi.agent.context_manager import ConversationState, InvestigationState
from numi.agent.llm.registry import LLMRegistry
from numi.agent.orchestrator import AgentOrchestrator
from numi.agent.planner.actions import Conclude, CritiqueVerdict, ProposeToolCall
from numi.common.models.tool import ToolCallResponse, ToolCallStatus


class _FakeToolClient:
    def __init__(
        self,
        response: ToolCallResponse | None = None,
        *,
        responses: list[ToolCallResponse] | None = None,
    ):
        self._responses = responses
        self._default = response or ToolCallResponse(status=ToolCallStatus.EXECUTED, message="ok", result={})
        self.requests: list[object] = []

    async def submit(self, request):
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return self._default

    async def log_decision_event(self, request):
        pass


class _FakeLLM:
    """Only `decide_next_action` is exercised by `_run_investigation_loop`
    — the other LLMProvider methods are never reached by it."""

    provider_name = "fake"
    model = "fake-model"

    def __init__(self, actions: list):
        self._actions = list(actions)
        self.calls: list[dict] = []

    async def decide_next_action(self, **kwargs):
        self.calls.append(kwargs)
        return self._actions.pop(0)

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


def _orchestrator(tool_client) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(None), tool_client=tool_client, context=None
    )


def _state_and_investigation(
    *, playbook_id: str | None = None
) -> tuple[ConversationState, InvestigationState]:
    state = ConversationState(
        conversation_id="conv1", channel="dev", channel_thread_id="", channel_account_id="dba_l2@example.com"
    )
    state.database_context["environment"] = "development"
    investigation = InvestigationState(
        investigation_id="inv1", problem="queries are slow on CoreBanking", playbook_id=playbook_id
    )
    return state, investigation


_ALL_READ_TOOL_IDS = [
    "database.get_health", "database.get_version", "database.get_sessions",
    "database.get_blocking_sessions", "database.get_deadlocks", "database.get_running_queries",
    "database.get_wait_statistics", "database.get_query_plan", "database.get_top_queries",
    "database.get_indexes", "database.get_statistics", "database.get_tables",
    "database.get_storage", "database.get_transaction_log", "database.get_replication_status",
    "database.get_backup_status", "database.get_configuration", "database.get_error_logs",
]

_COMPREHENSIVE_SUMMARY_TOOL_IDS = [
    "database.get_health",
    "database.get_blocking_sessions",
    "database.get_backup_status",
    "database.get_storage",
    "database.get_error_logs",
]


@pytest.mark.asyncio
async def test_a_matched_playbooks_steps_run_with_zero_intermediate_llm_calls():
    state, investigation = _state_and_investigation(playbook_id="slow_queries")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Root cause identified.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    # slow_queries has 5 steps — all 5 must run as tool calls before the LLM
    # is ever asked anything, and the LLM must be asked exactly once (to
    # conclude), not once per step.
    assert tool_client.requests[:5][0].tool_id == "database.get_health"
    assert [r.tool_id for r in tool_client.requests] == [
        "database.get_health",
        "database.get_running_queries",
        "database.get_top_queries",
        "database.get_wait_statistics",
        "database.get_blocking_sessions",
    ]
    assert len(llm.calls) == 1
    assert reply.status == "ok"
    assert "slow queries playbook" in reply.text.lower() or "Slow Query Investigation" in reply.text


@pytest.mark.asyncio
async def test_the_final_llm_call_carries_the_playbooks_conclusion_guidance():
    state, investigation = _state_and_investigation(playbook_id="high_cpu")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client)

    await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 1
    problem = llm.calls[0]["problem_statement"]
    assert "High CPU Investigation" in problem
    assert "driving CPU" in problem  # from the playbook's conclusion_guidance


@pytest.mark.asyncio
async def test_an_unmatched_investigation_asks_the_llm_every_turn_unchanged():
    """No playbook_id set — must behave exactly as before this feature:
    every step is an LLM decision, none is skipped."""
    state, investigation = _state_and_investigation(playbook_id=None)
    tool_client = _FakeToolClient()
    llm = _FakeLLM(
        actions=[
            ProposeToolCall(tool_id="database.get_health", reason="Baseline.", arguments={}, target={}),
            Conclude(summary="done"),
        ]
    )
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert len(llm.calls) == 2
    assert len(tool_client.requests) == 1
    assert reply.status == "ok"
    assert "playbook" not in reply.text.lower()


@pytest.mark.asyncio
async def test_a_playbook_step_whose_tool_is_unavailable_is_skipped_without_an_llm_call():
    state, investigation = _state_and_investigation(playbook_id="high_cpu")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client)

    # high_cpu's steps are get_health, get_top_queries, get_running_queries,
    # get_wait_statistics — drop get_top_queries from what's available.
    available = [t for t in _ALL_READ_TOOL_IDS if t != "database.get_top_queries"]

    await orchestrator._run_investigation_loop(
        state, investigation, available, "dev", "dba_l2@example.com", llm, None
    )

    called_tools = [r.tool_id for r in tool_client.requests]
    assert "database.get_top_queries" not in called_tools
    assert called_tools == [
        "database.get_health",
        "database.get_running_queries",
        "database.get_wait_statistics",
    ]
    assert len(llm.calls) == 1  # the skip itself never touched the LLM


@pytest.mark.asyncio
async def test_a_failed_playbook_step_is_recorded_and_the_investigation_continues():
    state, investigation = _state_and_investigation(playbook_id="backups")
    responses = [
        ToolCallResponse(status=ToolCallStatus.FAILED, message="adapter timeout"),
        ToolCallResponse(status=ToolCallStatus.EXECUTED, message="ok", result={}),
    ]
    tool_client = _FakeToolClient(responses=responses)
    llm = _FakeLLM(actions=[Conclude(summary="done")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    # backups is get_backup_status, get_storage — both must have been tried
    # despite the first one failing.
    assert [r.tool_id for r in tool_client.requests] == ["database.get_backup_status", "database.get_storage"]
    assert any("failed" in e for e in investigation.evidence)
    assert reply.status == "ok"  # the investigation still reached a conclusion


@pytest.mark.asyncio
async def test_status_command_reports_the_active_playbook_and_step():
    state, investigation = _state_and_investigation(playbook_id="high_cpu")
    state.investigation = investigation
    investigation.playbook_step = 2
    orchestrator = _orchestrator(_FakeToolClient())

    reply = await orchestrator._handle_command_if_any(state, "/status", "dev", "dba_l2@example.com")

    assert reply is not None
    assert "High CPU Investigation" in reply.text
    assert "2/4" in reply.text


@pytest.mark.asyncio
async def test_playbooks_command_lists_every_playbook():
    orchestrator = _orchestrator(_FakeToolClient())
    reply = await orchestrator._handle_command_if_any(
        ConversationState(conversation_id="c", channel="dev", channel_thread_id="", channel_account_id="a"),
        "/playbooks",
        "dev",
        "a",
    )
    assert reply is not None
    assert "Slow Query Investigation" in reply.text
    assert "Deadlock Investigation" in reply.text
    assert "Configuration Tuning Review" in reply.text


@pytest.mark.asyncio
async def test_transaction_log_playbook_runs_its_four_steps_with_zero_intermediate_llm_calls():
    """The split-off transaction_log playbook (see agent.playbooks.library —
    previously folded into the combined 'storage' playbook with no
    replication/backup checks at all) must run its full step sequence,
    including the replication-lag and backup-status checks that are the
    whole point of the split, before the LLM is asked anything."""
    state, investigation = _state_and_investigation(playbook_id="transaction_log")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Identified the reuse blocker.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert [r.tool_id for r in tool_client.requests] == [
        "database.get_transaction_log",
        "database.get_replication_status",
        "database.get_backup_status",
        "database.get_health",
    ]
    assert len(llm.calls) == 1
    problem = llm.calls[0]["problem_statement"]
    assert "Transaction Log Investigation" in problem
    assert "replication lag" in problem  # from the playbook's own conclusion_guidance
    assert reply.status == "ok"


@pytest.mark.asyncio
async def test_storage_playbook_runs_its_two_steps_post_split():
    """The `storage` playbook, post-split, is capacity-only (get_storage +
    get_health) — it must no longer run the old combined 3-step sequence
    that included get_transaction_log."""
    state, investigation = _state_and_investigation(playbook_id="storage")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Capacity looks fine.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert [r.tool_id for r in tool_client.requests] == [
        "database.get_storage",
        "database.get_health",
    ]
    assert len(llm.calls) == 1
    assert reply.status == "ok"


@pytest.mark.asyncio
async def test_configuration_review_playbook_runs_its_two_steps_with_zero_intermediate_llm_calls():
    state, investigation = _state_and_investigation(playbook_id="configuration_review")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Reviewed configuration.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert [r.tool_id for r in tool_client.requests] == [
        "database.get_configuration",
        "database.get_health",
    ]
    assert len(llm.calls) == 1
    problem = llm.calls[0]["problem_statement"]
    assert "Configuration Tuning Review" in problem
    # from the playbook's own conclusion_guidance — the scoping limitation
    # must reach the LLM call, not just live in the source file.
    assert "hardware" in problem or "instance-class" in problem
    assert reply.status == "ok"


# --- deepened playbooks (deadlocks, errors) that gained new steps against ---
# --- the same external playbook specification — see agent.playbooks.library ---


@pytest.mark.asyncio
async def test_deadlocks_playbook_runs_its_four_steps_with_zero_intermediate_llm_calls():
    """The deadlocks playbook gained a `get_sessions` step (checking for
    long-running transactions contributing to the lock cycle) — it must
    still run entirely before the LLM is ever asked anything, and the LLM
    must be asked exactly once, to conclude."""
    state, investigation = _state_and_investigation(playbook_id="deadlocks")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Identified the lock cycle.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert [r.tool_id for r in tool_client.requests] == [
        "database.get_deadlocks",
        "database.get_blocking_sessions",
        "database.get_running_queries",
        "database.get_sessions",
    ]
    assert len(llm.calls) == 1
    problem = llm.calls[0]["problem_statement"]
    assert "Deadlock Investigation" in problem
    assert "lock-acquisition cycle" in problem  # from the playbook's own conclusion_guidance
    assert reply.status == "ok"


@pytest.mark.asyncio
async def test_errors_playbook_runs_its_five_steps_with_zero_intermediate_llm_calls():
    """The errors playbook gained `get_blocking_sessions` and `get_deadlocks`
    steps (correlating error-log findings against other signals) — five
    steps total, still one turn short of the shared 6-turn budget, so the
    model must still get its own unrestricted turn at the end."""
    state, investigation = _state_and_investigation(playbook_id="errors")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Classified the errors found.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert [r.tool_id for r in tool_client.requests] == [
        "database.get_error_logs",
        "database.get_health",
        "database.get_blocking_sessions",
        "database.get_deadlocks",
        "database.get_running_queries",
    ]
    assert len(llm.calls) == 1
    problem = llm.calls[0]["problem_statement"]
    assert "Error Log Investigation" in problem
    assert "Correlate against the other signals gathered" in problem
    assert reply.status == "ok"


# --- comprehensive_summary (new playbook — a broad, single-server sweep, ---
# --- not tied to a specific symptom — see agent.playbooks.library) ---


@pytest.mark.asyncio
async def test_comprehensive_summary_playbook_runs_all_five_steps_with_zero_intermediate_llm_calls():
    """comprehensive_summary is the broadest playbook in the library — every
    one of its 5 steps must run as a deterministic tool call before the LLM
    is ever asked anything, and the LLM must be asked exactly once (to
    conclude), matching the pattern every other playbook follows. Its step
    count is exactly 5 — one short of the shared _MAX_INVESTIGATION_TURNS
    budget (6), not one per instance-wide read tool (see the playbook's own
    description in library.py) — so, like every other playbook in this
    library, it leaves the model its own unrestricted final turn via the
    normal in-loop path rather than the turn-budget-exhausted ("final
    chance") fallback."""
    state, investigation = _state_and_investigation(playbook_id="comprehensive_summary")
    tool_client = _FakeToolClient()
    llm = _FakeLLM(actions=[Conclude(summary="Comprehensive summary complete.")])
    orchestrator = _orchestrator(tool_client)

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert [r.tool_id for r in tool_client.requests] == _COMPREHENSIVE_SUMMARY_TOOL_IDS
    assert len(llm.calls) == 1
    problem = llm.calls[0]["problem_statement"]
    assert "Comprehensive Health Summary" in problem
    # from the playbook's own conclusion_guidance
    assert "all other checks came back clean" in problem
    assert reply.status == "ok"
