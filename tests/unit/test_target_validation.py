from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.target import DatabaseTarget, Environment
from numi.gateway.domain.catalog import (
    DiscoveredDatabase,
    DiscoveredObject,
    InMemoryCatalogStore,
    ServerCatalog,
)
from numi.gateway.domain.servers import AmbiguousServerError, ServerRegistry
from numi.gateway.domain.target_validation import TargetValidator


async def test_resolves_unambiguous_server_by_alias(target_validator):
    target = DatabaseTarget(environment=Environment.PRODUCTION, instance="core-banking")
    ctx = await target_validator.validate(target, ["environment", "instance"])
    assert ctx.server.id == "corebanking-sqlserver-prod"
    assert ctx.criticality == "critical"


async def test_unregistered_server_is_invalid_target(target_validator):
    target = DatabaseTarget(environment=Environment.PRODUCTION, instance="totally-unknown-box")
    with pytest.raises(NumiError) as exc:
        await target_validator.validate(target, ["environment", "instance"])
    assert exc.value.code == FailureCode.INVALID_TARGET


async def test_cannot_cross_environments_implicitly(target_validator):
    # A dev server's id/alias must never resolve when environment=production.
    target = DatabaseTarget(environment=Environment.PRODUCTION, instance="sqlserver-dev-01")
    with pytest.raises(NumiError):
        await target_validator.validate(target, ["environment", "instance"])


async def test_per_database_override_is_applied(target_validator):
    # corebanking-sqlserver-prod is `critical`; `master` has a `low` override.
    target = DatabaseTarget(
        environment=Environment.PRODUCTION, instance="core-banking", database="master"
    )
    ctx = await target_validator.validate(target, ["environment", "instance"])
    assert ctx.criticality == "low"


async def test_database_validated_against_the_discovered_catalog():
    registry = ServerRegistry("config/servers.yaml")
    catalog = InMemoryCatalogStore()
    await catalog.put(
        ServerCatalog(
            server_id="corebanking-sqlserver-prod",
            discovered_at=dt.datetime.now(dt.UTC),
            databases=[
                DiscoveredDatabase(
                    name="CoreBanking",
                    objects=[DiscoveredObject(schema_name="dbo", name="Accounts", kind="table")],
                )
            ],
        )
    )
    validator = TargetValidator(registry, catalog)

    ok = await validator.validate(
        DatabaseTarget(
            environment=Environment.PRODUCTION, instance="core-banking", database="corebanking"
        ),
        ["environment", "instance", "database"],
    )
    assert ok.database == "CoreBanking"  # canonical casing from the catalog

    with pytest.raises(NumiError) as exc:
        await validator.validate(
            DatabaseTarget(
                environment=Environment.PRODUCTION, instance="core-banking", database="NoSuchDb"
            ),
            ["environment", "instance", "database"],
        )
    assert exc.value.code == FailureCode.INVALID_TARGET
    assert "not found" in exc.value.detail


async def test_unknown_object_is_rejected_when_catalog_is_populated():
    registry = ServerRegistry("config/servers.yaml")
    catalog = InMemoryCatalogStore()
    await catalog.put(
        ServerCatalog(
            server_id="corebanking-sqlserver-prod",
            databases=[
                DiscoveredDatabase(
                    name="CoreBanking",
                    objects=[DiscoveredObject(schema_name="dbo", name="Accounts", kind="table")],
                )
            ],
        )
    )
    validator = TargetValidator(registry, catalog)
    with pytest.raises(NumiError):
        await validator.validate(
            DatabaseTarget(
                environment=Environment.PRODUCTION,
                instance="core-banking",
                database="CoreBanking",
                object="GhostTable",
            ),
            ["environment", "instance", "database", "object"],
        )


def test_llm_cannot_supply_arbitrary_connection_target():
    with pytest.raises(ValidationError):
        DatabaseTarget(
            environment=Environment.PRODUCTION,
            instance="core-banking",
            connection_string="Server=evil;Trusted_Connection=True;",
        )


# --- Fuzzy server-reference resolution (real DBAs rarely use a server's ---
# --- exact registered id/alias — an abbreviation, nickname, or an IP     ---
# --- address (or a fragment of one) is at least as common in practice.  ---


async def test_resolves_server_by_its_exact_registered_host_ip(target_validator):
    # corebanking-sqlserver-prod's host in config/servers.yaml is 192.168.0.100.
    target = DatabaseTarget(environment=Environment.PRODUCTION, instance="192.168.0.100")
    ctx = await target_validator.validate(target, ["environment", "instance"])
    assert ctx.server.id == "corebanking-sqlserver-prod"


async def test_resolves_server_by_a_fragment_of_its_host_ip(target_validator):
    # A DBA saying just "0.100" (the distinctive tail of 192.168.0.100) must
    # resolve exactly like the full IP would — unambiguous within production,
    # since analytics-postgres-prod's host (localhost) doesn't contain it.
    target = DatabaseTarget(environment=Environment.PRODUCTION, instance="0.100")
    ctx = await target_validator.validate(target, ["environment", "instance"])
    assert ctx.server.id == "corebanking-sqlserver-prod"


async def test_a_host_fragment_shared_by_several_servers_stays_ambiguous(server_registry):
    """Never a silent guess: every development-tier server in
    config/servers.yaml shares the host "localhost" — asking for it by
    host alone must surface every match, not pick one, exactly like an
    ambiguous id/alias already does."""
    target = DatabaseTarget(environment=Environment.DEVELOPMENT, instance="localhost")
    candidates = server_registry.find_candidates(target)
    assert len(candidates) > 1
    with pytest.raises(AmbiguousServerError):
        server_registry.resolve(target)
