"""Reproduces a live finding: "sql server dev 1" (a real DBA's own phrasing)
didn't resolve to the registered server "sqlserver-dev-01" — the existing
lowercase-substring matching doesn't bridge spacing/punctuation differences
or zero-padding ("1" vs "01"). `normalize_server_reference` closes that
specific gap; it's a pure function shared between ServerRegistry (Gateway)
and _environment_for_instance (Agent) so the two never drift."""

from __future__ import annotations

import pytest

from numi.common.models.target import DatabaseTarget, Environment
from numi.common.server_reference import normalize_server_reference
from numi.gateway.domain.servers import ServerRegistry


def test_normalizes_spacing_punctuation_and_zero_padding_identically():
    assert normalize_server_reference("SQL Server Dev 1") == normalize_server_reference(
        "sqlserver-dev-01"
    )
    assert normalize_server_reference("sqlserver_dev_1") == normalize_server_reference(
        "sqlserver-dev-01"
    )


def test_distinct_names_still_normalize_differently():
    assert normalize_server_reference("postgres-local") != normalize_server_reference(
        "sqlserver-dev-01"
    )


async def test_resolves_the_exact_live_finding(server_registry: ServerRegistry):
    target = DatabaseTarget(environment=Environment.DEVELOPMENT, instance="sql server dev 1")
    candidates = server_registry.find_candidates(target)
    assert len(candidates) == 1
    assert candidates[0].id == "sqlserver-dev-01"


async def test_still_resolves_the_exact_registered_id_unchanged(server_registry: ServerRegistry):
    """Normalization is additive — the existing exact/substring path must
    keep working exactly as before."""
    target = DatabaseTarget(environment=Environment.DEVELOPMENT, instance="sqlserver-dev-01")
    candidates = server_registry.find_candidates(target)
    assert len(candidates) == 1
    assert candidates[0].id == "sqlserver-dev-01"


@pytest.mark.parametrize(
    "hint",
    ["SQL Server Dev 1", "sql-server-dev-1", "sql_server_dev_01", "SQLServerDev01"],
)
async def test_several_real_world_spellings_all_resolve(server_registry: ServerRegistry, hint: str):
    target = DatabaseTarget(environment=Environment.DEVELOPMENT, instance=hint)
    candidates = server_registry.find_candidates(target)
    assert len(candidates) == 1
    assert candidates[0].id == "sqlserver-dev-01"
