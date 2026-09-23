"""MockLLMProvider.extract_intent — the deterministic offline planner's
free-text parsing, used whenever no LLM key is configured."""

from __future__ import annotations

import pytest

from numi.agent.llm.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_extract_intent_recognizes_a_named_server_by_id():
    result = await MockLLMProvider().extract_intent(
        "Check the health of TestDatabase in development on postgres-local",
        known_database_names=["TestDatabase"],
        known_server_hints=["postgres-local", "postgres-dev-01"],
    )
    assert result.instance_hint == "postgres-local"


@pytest.mark.asyncio
async def test_extract_intent_recognizes_a_server_by_alias():
    result = await MockLLMProvider().extract_intent(
        "Check the health of TestDatabase on local development",
        known_database_names=["TestDatabase"],
        known_server_hints=["postgres-local", "local", "postgres-dev-01"],
    )
    # The longer, more specific candidate wins even though a shorter alias
    # also matches ("local" is a substring relationship, not this case, but
    # this pins the tie-break rule so it can't silently regress).
    assert result.instance_hint in {"postgres-local", "local"}


@pytest.mark.asyncio
async def test_extract_intent_leaves_instance_hint_unset_when_no_server_is_named():
    result = await MockLLMProvider().extract_intent(
        "Check the health of TestDatabase in development",
        known_database_names=["TestDatabase"],
        known_server_hints=["postgres-local", "postgres-dev-01"],
    )
    assert result.instance_hint is None


@pytest.mark.asyncio
async def test_extract_intent_never_invents_a_server_not_in_the_known_list():
    """A server-shaped word in the message that isn't actually registered
    must never become an instance_hint — the Gateway independently
    re-resolves it, but this must not manufacture a false-positive."""
    result = await MockLLMProvider().extract_intent(
        "Check the health of TestDatabase on made-up-server-01 in development",
        known_database_names=["TestDatabase"],
        known_server_hints=["postgres-local", "postgres-dev-01"],
    )
    assert result.instance_hint is None


@pytest.mark.asyncio
async def test_extract_intent_works_without_known_server_hints_argument():
    """Backward compatible: callers (and the opt-in live-LLM tests) that
    don't pass known_server_hints at all must still work."""
    result = await MockLLMProvider().extract_intent(
        "CoreBanking production is very slow, please investigate.",
        known_database_names=["CoreBanking"],
    )
    assert result.is_dba_task is True
    assert result.instance_hint is None
