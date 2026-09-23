"""decide_next_action must actually tell the model each tool's real
required-argument keys, not just its bare tool_id — otherwise the model
has nothing but the name to guess from (spec §35).

Reproduces a live finding: asked the (real Gemini) agent to update
statistics on AdventureWorks2019's Person.Person table. It correctly named
database.update_statistics and correctly scoped `target`, but sent an
empty `arguments` — because nothing told it the tool separately requires
`schema`/`table`/`reason` there too (UpdateStatisticsArgs, validated
independently of `target`, before the two are ever reconciled)."""

from __future__ import annotations

import pytest

from numi.agent.llm.base import StructuredLLMProvider
from numi.agent.llm.mock import MockLLMProvider


class _CapturingProvider(StructuredLLMProvider):
    provider_name = "fake"

    def __init__(self):
        super().__init__("fake-model")
        self.last_user_message: str | None = None

    async def _call_tool(self, *, system: str, user: str, schema: dict, tool_name: str) -> dict:
        self.last_user_message = user
        return {
            "action": "propose_tool_call",
            "tool_id": "database.update_statistics",
            "reason": "Refreshing stale statistics.",
            "target": {"schema": "Person", "object": "Person"},
            "arguments": {"schema": "Person", "table": "Person", "reason": "Refreshing stale statistics."},
        }

    async def _call_text(self, *, system: str, user: str) -> str:
        return ""


@pytest.mark.asyncio
async def test_decide_next_action_tells_the_model_each_tools_required_arguments():
    provider = _CapturingProvider()
    await provider.decide_next_action(
        problem_statement="Refresh stale statistics on Person.Person",
        available_tool_ids=["database.get_health", "database.update_statistics"],
        transcript=[],
        turn_count=0,
        tool_requirements={
            "database.update_statistics": ["schema", "table", "reason"],
            "database.get_health": [],
        },
    )
    assert provider.last_user_message is not None
    assert "database.update_statistics" in provider.last_user_message
    assert "schema" in provider.last_user_message
    assert "table" in provider.last_user_message


@pytest.mark.asyncio
async def test_decide_next_action_omits_the_requirements_line_when_none_given():
    """No tool_requirements (e.g. every available tool needs no arguments)
    must not inject an empty/confusing line into the prompt."""
    provider = _CapturingProvider()
    await provider.decide_next_action(
        problem_statement="Check health",
        available_tool_ids=["database.get_health"],
        transcript=[],
        turn_count=0,
    )
    assert provider.last_user_message is not None
    assert "Required `arguments` keys" not in provider.last_user_message


@pytest.mark.asyncio
async def test_mock_provider_accepts_tool_requirements_without_needing_it():
    """MockLLMProvider's deterministic scenarios never need schema/table
    arguments — it must still accept the parameter for signature
    compatibility with the real providers."""
    action = await MockLLMProvider().decide_next_action(
        problem_statement="check health",
        available_tool_ids=["database.get_health"],
        transcript=[],
        turn_count=0,
        tool_requirements={"database.update_statistics": ["schema", "table", "reason"]},
    )
    assert action is not None
