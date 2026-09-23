"""Self-critique pass before accepting a Conclude (`_self_critique_conclude`,
`_finalize_conclude`'s third check) — a second LLM opinion slotted into the
exact reject-and-retry scaffolding `_ungrounded_identifiers`/
`pending_verification` already use. See `CritiqueVerdict`'s own docstring
for what this catches that those two structural checks don't."""

from __future__ import annotations

import pytest

from numi.agent.llm.mock import MockLLMProvider
from numi.agent.planner.actions import Conclude, CritiqueVerdict
from numi.common.config import get_settings
from tests.unit.test_orchestrator_playbooks import (
    _ALL_READ_TOOL_IDS,
    _FakeToolClient,
    _orchestrator,
    _state_and_investigation,
)


class _ScriptedCritiqueLLM:
    """`decide_next_action` and `critique_conclusion` each drawn from their
    own separate scripted list — models these as independent calls, since
    `_finalize_conclude` calls the latter only after a Conclude action is
    already proposed."""

    provider_name = "fake"
    model = "fake-model"

    def __init__(self, actions: list, critiques: list[CritiqueVerdict] | None = None):
        self._actions = list(actions)
        self._critiques = list(critiques) if critiques is not None else None
        self._raise_critique: Exception | None = None
        self.critique_calls: list[dict] = []

    def raise_on_critique(self, exc: Exception) -> None:
        self._raise_critique = exc

    async def decide_next_action(self, **kwargs):
        return self._actions.pop(0)

    async def critique_conclusion(self, **kwargs):
        self.critique_calls.append(kwargs)
        if self._raise_critique is not None:
            raise self._raise_critique
        return self._critiques.pop(0)


@pytest.mark.asyncio
async def test_the_base_llmprovider_default_never_rejects_anything():
    """The deterministic offline mock planner this entire default test
    suite runs on inherits `LLMProvider.critique_conclusion`'s concrete,
    non-abstract default unconditionally — never `@abstractmethod`, so it
    keeps instantiating with zero changes."""
    mock = MockLLMProvider()
    verdict = await mock.critique_conclusion(
        problem_statement="p",
        transcript=[],
        proposed_summary="anything at all",
        proposed_root_cause=None,
        proposed_confidence="unable_to_confirm",
        proposed_recommendation=None,
    )
    assert verdict == CritiqueVerdict(sound=True)


@pytest.mark.asyncio
async def test_a_sound_conclusion_is_accepted_on_the_first_critique():
    state, investigation = _state_and_investigation(playbook_id=None)
    good = Conclude(summary="Driven by heavy CPU load, confirmed via get_top_queries.")
    llm = _ScriptedCritiqueLLM(actions=[good], critiques=[CritiqueVerdict(sound=True)])
    orchestrator = _orchestrator(_FakeToolClient())

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "ok"
    assert len(llm.critique_calls) == 1


@pytest.mark.asyncio
async def test_an_unsound_conclusion_is_rejected_and_the_loop_retries():
    state, investigation = _state_and_investigation(playbook_id=None)
    bad = Conclude(summary="Must be a memory leak.")
    good = Conclude(summary="Driven by heavy CPU load.")
    llm = _ScriptedCritiqueLLM(
        actions=[bad, good],
        critiques=[
            CritiqueVerdict(sound=False, issue="No tool result mentions memory at all."),
            CritiqueVerdict(sound=True),
        ],
    )
    orchestrator = _orchestrator(_FakeToolClient())

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "ok"
    assert "Summary: Must be a memory leak" not in reply.text
    assert "Summary: Driven by heavy CPU load" in reply.text
    assert investigation.transcript[-1]["tool_id"] == "internal.self_critique_check"
    assert "No tool result mentions memory" in investigation.transcript[-1]["result"]["issue"]
    assert len(llm.critique_calls) == 2


@pytest.mark.asyncio
async def test_a_critique_call_failure_fails_open_and_accepts_the_conclusion():
    """The one invariant that matters most: critique failing (a timeout, a
    provider outage) must never be worse than not having critiqued at
    all — the conclusion still reaches the DBA."""
    state, investigation = _state_and_investigation(playbook_id=None)
    good = Conclude(summary="Driven by heavy CPU load.")
    llm = _ScriptedCritiqueLLM(actions=[good])
    llm.raise_on_critique(RuntimeError("provider unavailable"))
    orchestrator = _orchestrator(_FakeToolClient())

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "ok"
    assert "Summary: Driven by heavy CPU load" in reply.text


@pytest.mark.asyncio
async def test_the_final_chance_call_skips_critique_entirely():
    """No tool budget left to act on new guidance during the one bounded
    last-chance call — rejecting there would only throw away everything the
    investigation found in favor of the generic no-root-cause fallback,
    exactly like the existing pending_verification check's own reasoning."""
    state, investigation = _state_and_investigation(playbook_id=None)
    conclude = Conclude(summary="Whatever it takes to reach the last-chance call.")
    llm = _ScriptedCritiqueLLM(actions=[conclude] * 7)

    async def _never_called(**kwargs):
        raise AssertionError("critique_conclusion must not be called on the final-chance call")

    llm.critique_conclusion = _never_called
    orchestrator = _orchestrator(_FakeToolClient())

    reply = await orchestrator._run_investigation_loop(
        state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
    )

    assert reply.status == "ok"
    assert "Whatever it takes" in reply.text


@pytest.mark.asyncio
async def test_self_critique_disabled_by_settings_skips_the_llm_call_entirely(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("SELF_CRITIQUE_ENABLED", "false")
    try:
        state, investigation = _state_and_investigation(playbook_id=None)
        good = Conclude(summary="Driven by heavy CPU load.")
        llm = _ScriptedCritiqueLLM(actions=[good])

        async def _never_called(**kwargs):
            raise AssertionError("critique_conclusion must not be called when disabled")

        llm.critique_conclusion = _never_called
        orchestrator = _orchestrator(_FakeToolClient())

        reply = await orchestrator._run_investigation_loop(
            state, investigation, _ALL_READ_TOOL_IDS, "dev", "dba_l2@example.com", llm, None
        )

        assert reply.status == "ok"
    finally:
        get_settings.cache_clear()
