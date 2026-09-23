"""Server registry (spec §10, §11, §39 — reframed around servers).

You register a *server* (one host/port/engine/environment) with one
least-privilege diagnostic credential. Individual databases are **not**
registered — they are discovered (see `numi.gateway.domain.catalog` and
`numi.execution.discovery`). Authorization, criticality, and the
maintenance window live at the server level, with optional per-database
overrides for the handful that need them.

The security invariant is unchanged: the Agent can only ever target a
server in this registry, and a database/object that discovery actually
found on it. It can never name an arbitrary host.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from numi.common.models.identity import DBARole
from numi.common.models.target import DatabaseTarget, Environment, Platform
from numi.common.server_reference import normalize_server_reference


class DatabaseOverride(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    criticality: str | None = None
    classification: str | None = None
    allowed_roles: list[DBARole] | None = None


class ServerEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    platform: Platform
    environment: Environment
    host: str
    port: int
    # Human label(s) used to resolve fuzzy references ("the core-banking
    # server", "prod postgres"). `id` always matches too.
    aliases: list[str] = Field(default_factory=list)
    organization: str = ""
    region: str = ""
    criticality: str = "standard"
    classification: str = "internal"
    owner: str = ""
    support_team: str = ""
    maintenance_window: dict = Field(default_factory=dict)
    allowed_roles: list[DBARole] = Field(default_factory=list)
    # Per-database exceptions to the server-level criticality/roles.
    database_overrides: dict[str, DatabaseOverride] = Field(default_factory=dict)
    status: str = "active"

    def effective_for(self, database: str | None) -> EffectivePolicy:
        override = self.database_overrides.get(database or "", DatabaseOverride())
        return EffectivePolicy(
            criticality=override.criticality or self.criticality,
            classification=override.classification or self.classification,
            allowed_roles=override.allowed_roles
            if override.allowed_roles is not None
            else list(self.allowed_roles),
        )


class EffectivePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    criticality: str
    classification: str
    allowed_roles: list[DBARole]


class AmbiguousServerError(Exception):
    def __init__(self, candidates: list[ServerEntry]):
        self.candidates = candidates
        super().__init__(f"{len(candidates)} servers match the given target.")


class ServerRegistry:
    def __init__(self, config_path: str | Path):
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        self._entries: list[ServerEntry] = [
            ServerEntry.model_validate(e) for e in raw.get("servers", [])
        ]

    def all(self) -> list[ServerEntry]:
        return list(self._entries)

    def by_id(self, server_id: str) -> ServerEntry | None:
        return next((e for e in self._entries if e.id == server_id), None)

    def find_candidates(self, target: DatabaseTarget) -> list[ServerEntry]:
        """Every registered server consistent with the (possibly partial)
        target. Never widens beyond `environment`."""
        candidates = [
            e for e in self._entries
            if e.environment == target.environment and e.status == "active"
        ]
        if target.platform is not None:
            candidates = [e for e in candidates if e.platform == target.platform]

        # `instance` and `cluster` on DatabaseTarget both name the server here.
        for hint in (target.instance, target.cluster):
            if not hint:
                continue
            needle = hint.strip().lower()
            # Also matches the registered host (IP or hostname) — a real
            # DBA will as often say "10.1.3.7" or a fragment of it ("3.7")
            # as a registered id/alias, and neither was ever checked here
            # before. And also matches on a normalized form (see
            # normalize_server_reference) so "sql server dev 1" resolves
            # the same "sqlserver-dev-01" would — spacing, punctuation, and
            # zero-padding differences a human plainly doesn't mean
            # anything by. Still exact-first, substring-fallback, and
            # still ambiguous-or-no-match -> LookupError/AmbiguousServerError
            # below rather than a guess — every widening here only ever
            # adds *candidates* for that same "ask, don't guess" fallback
            # to consider, never weakens it (a false merge just becomes an
            # ambiguous match instead of a clean single one, never a wrong
            # confident one). An operator should still add generous aliases
            # for anything none of this can derive on its own (a nickname
            # with no textual relation to the id/host, e.g. "MI01" for a
            # managed instance not named anything like that).
            needle_normalized = normalize_server_reference(needle)
            exact = [
                e for e in candidates
                if e.id.lower() == needle
                or needle in {a.lower() for a in e.aliases}
                or e.host.lower() == needle
                or normalize_server_reference(e.id) == needle_normalized
                or any(normalize_server_reference(a) == needle_normalized for a in e.aliases)
            ]
            if exact:
                candidates = exact
            else:
                candidates = [
                    e for e in candidates
                    if needle in e.id.lower()
                    or any(needle in a.lower() for a in e.aliases)
                    or needle in e.host.lower()
                    or needle_normalized in normalize_server_reference(e.id)
                    or any(
                        needle_normalized in normalize_server_reference(a) for a in e.aliases
                    )
                ]

        # A database-name hint narrows *only* when it uniquely points at one
        # server via its aliases (discovery does the real db->server mapping).
        if target.database and len(candidates) > 1:
            needle = target.database.strip().lower()
            narrowed = [
                e for e in candidates
                if any(needle in a.lower() for a in e.aliases) or needle in e.id.lower()
            ]
            if narrowed:
                candidates = narrowed

        return candidates

    def resolve(self, target: DatabaseTarget) -> ServerEntry:
        candidates = self.find_candidates(target)
        if not candidates:
            raise LookupError("No registered server matches the given target.")
        if len(candidates) > 1:
            raise AmbiguousServerError(candidates)
        return candidates[0]
