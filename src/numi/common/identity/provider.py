"""IdentityProvider abstraction (spec §3).

`IdentityProvider` is the *only* legitimate source of a `VerifiedIdentity`.
Channel adapters call `resolve_by_external_account` with the raw account id
from Slack/Teams; the Agent, Gateway, and Policy Engine never see — and
never trust — a raw channel username again after that point.

This module intentionally contains no HTTP calls to any real IdP. The
production implementation (OIDC/AAD) lives in each service's infrastructure
layer and satisfies this same interface; only the interface and the
development/test mock live here, in the shared package, since every service
needs to be able to construct one.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from numi.common.models.identity import DBARole, VerifiedIdentity


class IdentityProvider(ABC):
    """Resolves a channel account into a verified enterprise identity.

    Implementations MUST NOT accept a display name, a free-text username, or
    any claim asserted by the LLM as input — only a channel-verified account
    identifier (e.g. a Slack user id from a signature-verified webhook, or an
    AAD object id from a verified Teams bot token).
    """

    @abstractmethod
    async def resolve_by_external_account(
        self, channel: str, external_account_id: str
    ) -> VerifiedIdentity | None:
        """Return the verified identity for this channel account, or None."""

    @abstractmethod
    async def refresh(self, subject_id: str) -> VerifiedIdentity | None:
        """Re-resolve an identity by subject id (used to re-check group
        membership / MFA state at authorization time, rather than trusting a
        cached identity indefinitely)."""


class GroupRoleMapping:
    """Enterprise group -> `DBARole` mapping, loaded from configuration.

    Extracted so the mock and the real OIDC provider derive roles through
    *the same* code from *the same* config section: which enterprise group
    grants which DBA role is a security-relevant decision that must live
    entirely in `config/identity.yaml` (see SECURITY.md, "Identity"), never
    in a provider's own logic, and certainly never in two divergent copies
    of it.

    Two independent gates, both required: membership of a `dba_team` group
    makes someone a DBA at all, and a role's own group list grants that
    specific role. Someone in `Enterprise-DBA-L3` but *not* in any
    `dba_team` group gets no roles — so removing a leaver from one group
    revokes everything.
    """

    def __init__(self, identity_cfg: dict[str, Any]) -> None:
        self._dba_team_groups: set[str] = set(identity_cfg["groups"]["dba_team"])
        self._role_group_map: dict[DBARole, set[str]] = {
            DBARole(role_name): set(role_cfg["groups"])
            for role_name, role_cfg in identity_cfg["roles"].items()
        }

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> GroupRoleMapping:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return cls(raw["identity"])

    def derive_roles(self, groups: list[str]) -> list[DBARole]:
        group_set = set(groups)
        if not (group_set & self._dba_team_groups):
            return []
        return [
            role
            for role, required_groups in self._role_group_map.items()
            if group_set & required_groups
        ]


class _DirectoryEntry:
    __slots__ = ("subject_id", "email", "display_name", "channel_accounts", "groups", "mfa")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.subject_id: str = raw["subject_id"]
        self.email: str = raw["email"]
        self.display_name: str = raw["display_name"]
        self.channel_accounts: dict[str, str] = raw.get("channel_accounts", {})
        self.groups: list[str] = raw.get("groups", [])
        self.mfa: bool = raw.get("mfa_satisfied", False)


class MockIdentityProvider(IdentityProvider):
    """Config-driven identity provider for local development and tests.

    Loads `config/identity.yaml`, builds group -> role mappings, and resolves
    channel accounts against a fictitious directory declared in that same
    file. Never used in production — production wires a real OIDC provider
    satisfying the same `IdentityProvider` interface instead.
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._roles = GroupRoleMapping(raw["identity"])
        directory_entries = list(raw.get("mock_directory", []))

        # Optional local-only overlay (gitignored — see config/
        # local_identity.example.yaml) for real personal entries when
        # testing against a real channel (e.g. your own Slack workspace)
        # with your own real account. Kept out of config/identity.yaml
        # itself, which is committed to version control and documents
        # itself as fictitious-users-only; this file is a sibling of
        # whatever `config_path` is, so it works the same way for every
        # service that constructs a MockIdentityProvider without any
        # extra configuration.
        local_path = config_path.parent / "local_identity.yaml"
        if local_path.exists():
            local_raw = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
            directory_entries.extend(local_raw.get("mock_directory", []))

        self._directory: list[_DirectoryEntry] = [
            _DirectoryEntry(entry) for entry in directory_entries
        ]

    def _to_identity(self, entry: _DirectoryEntry) -> VerifiedIdentity:
        return VerifiedIdentity(
            subject_id=entry.subject_id,
            email=entry.email,
            display_name=entry.display_name,
            enterprise_groups=list(entry.groups),
            dba_roles=self._roles.derive_roles(entry.groups),
            mfa_satisfied=entry.mfa,
            authenticated_at=dt.datetime.now(dt.UTC).isoformat(),
        )

    async def resolve_by_external_account(
        self, channel: str, external_account_id: str
    ) -> VerifiedIdentity | None:
        for entry in self._directory:
            if entry.channel_accounts.get(channel) == external_account_id:
                return self._to_identity(entry)
        return None

    async def refresh(self, subject_id: str) -> VerifiedIdentity | None:
        for entry in self._directory:
            if entry.subject_id == subject_id:
                return self._to_identity(entry)
        return None
