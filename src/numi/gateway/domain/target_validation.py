"""Target validation (spec §10, §11 — server-level).

Confirms a proposed target (a) resolves to exactly one registered server,
(b) names a database/schema/object that discovery actually found on it
(when the catalog is populated), and (c) carries every field the tool
requires. The LLM never gets to supply a target that skips this — every
tool call goes through `TargetValidator.validate` before policy/risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from numi.common.models.failures import FailureCode, NumiError
from numi.common.models.identity import DBARole
from numi.common.models.target import DatabaseTarget, Environment, Platform
from numi.gateway.domain.catalog import CatalogStore, DiscoveredDatabase
from numi.gateway.domain.servers import AmbiguousServerError, ServerEntry, ServerRegistry

_FIELD_ALIASES = {"schema": "schema_name", "object": "object_name"}


@dataclass(frozen=True)
class TargetContext:
    """Everything downstream (authz / policy / risk / execution) needs about
    a resolved target. Replaces the old per-database `InventoryEntry`."""

    target: DatabaseTarget
    server: ServerEntry
    database: str
    criticality: str
    classification: str
    allowed_roles: list[DBARole]
    discovered_database: DiscoveredDatabase | None

    # --- adapters for the engines that used to read InventoryEntry ---------

    @property
    def id(self) -> str:  # server id
        return self.server.id

    @property
    def environment(self) -> Environment:
        return self.server.environment

    @property
    def platform(self) -> Platform:
        return self.server.platform

    @property
    def maintenance_window(self) -> dict:
        return self.server.maintenance_window


class TargetValidator:
    def __init__(self, registry: ServerRegistry, catalog: CatalogStore):
        self._registry = registry
        self._catalog = catalog

    def _check_required_fields(
        self, target: DatabaseTarget, server: ServerEntry, database: str, required_scope: list[str]
    ) -> None:
        effective = {
            "environment": target.environment.value,
            "instance": target.instance or server.id,
            "cluster": target.cluster or server.id,
            "database": database,
            "schema_name": target.schema_name,
            "object_name": target.object_name,
            "session_id": target.session_id,
            "query_id": target.query_id,
        }
        missing = [
            name for name in required_scope
            if not effective.get(_FIELD_ALIASES.get(name, name))
        ]
        if missing:
            raise NumiError(
                FailureCode.INVALID_TARGET,
                f"Missing required target field(s) for this operation: {', '.join(missing)}.",
            )

    async def validate(
        self, target: DatabaseTarget, required_scope: list[str]
    ) -> TargetContext:
        # 1. server
        try:
            server = self._registry.resolve(target)
        except AmbiguousServerError as exc:
            names = ", ".join(f"{c.id} ({c.environment.value})" for c in exc.candidates)
            raise NumiError(
                FailureCode.INVALID_TARGET,
                f"Multiple servers match this target — please say which one: {names}.",
            ) from exc
        except LookupError as exc:
            raise NumiError(
                FailureCode.INVALID_TARGET,
                "No registered server matches the given target.",
            ) from exc

        # 2. database — validated against the discovered catalog if we have one
        catalog = await self._catalog.get(server.id)
        database = (target.database or "").strip()
        discovered_db: DiscoveredDatabase | None = None

        if catalog and catalog.databases:
            if not database:
                # A tool that needs a database, on a server we've discovered,
                # but none named: help the caller pick.
                if "database" in required_scope:
                    names = ", ".join(catalog.database_names()[:20])
                    raise NumiError(
                        FailureCode.INVALID_TARGET,
                        f"Which database on {server.id}? Discovered: {names}",
                    )
            else:
                discovered_db = catalog.database(database)
                if discovered_db is None:
                    names = ", ".join(catalog.database_names()[:20])
                    raise NumiError(
                        FailureCode.INVALID_TARGET,
                        f"Database '{database}' was not found on {server.id}. "
                        f"Discovered databases: {names}",
                    )
                database = discovered_db.name  # canonical casing

        # 3. required fields
        self._check_required_fields(target, server, database, required_scope)

        # 4. schema / object (only when the catalog knows the database)
        if discovered_db is not None and target.object_name:
            object_names = {o.name.lower() for o in discovered_db.objects}
            if object_names and target.object_name.lower() not in object_names:
                raise NumiError(
                    FailureCode.INVALID_TARGET,
                    f"Object '{target.object_name}' was not found in "
                    f"{server.id}/{database}.",
                )

        eff = server.effective_for(database)
        return TargetContext(
            target=target,
            server=server,
            database=database,
            criticality=eff.criticality,
            classification=eff.classification,
            allowed_roles=eff.allowed_roles,
            discovered_database=discovered_db,
        )
