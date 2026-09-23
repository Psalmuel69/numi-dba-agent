"""Verified enterprise identity model (spec §3).

`VerifiedIdentity` is produced *only* by an `IdentityProvider` implementation
(never parsed out of a chat message, never asserted by the LLM). Everything
downstream — policy, risk, approval — reasons about a DBA purely in terms of
this object.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class DBARole(str, Enum):
    DBA_L1 = "DBA_L1"
    DBA_L2 = "DBA_L2"
    DBA_L3 = "DBA_L3"
    DBA_MANAGER = "DBA_MANAGER"


# Coarse privilege ordering used only for display / sanity-checking; actual
# authorization decisions always go through the Policy Engine's explicit
# role -> decision tables, never through this ordering.
ROLE_RANK: dict[DBARole, int] = {
    DBARole.DBA_L1: 1,
    DBARole.DBA_L2: 2,
    DBARole.DBA_L3: 3,
    DBARole.DBA_MANAGER: 4,
}


class VerifiedIdentity(BaseModel):
    """The output of independent identity verification.

    This is never constructed from a Slack/Teams display name, a username
    typed into a message, or an LLM claim. It is the return value of
    `IdentityProvider.resolve(...)`, which authenticates against the
    enterprise identity provider (OIDC in production; a config-driven mock
    in development/tests).
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str  # stable IdP subject / object id — never a display name
    email: str
    display_name: str
    enterprise_groups: list[str]
    dba_roles: list[DBARole]
    mfa_satisfied: bool
    authenticated_at: str  # ISO-8601 UTC timestamp of the auth event

    def is_dba(self) -> bool:
        return len(self.dba_roles) > 0

    def has_role(self, role: DBARole) -> bool:
        return role in self.dba_roles

    def highest_role(self) -> DBARole | None:
        if not self.dba_roles:
            return None
        return max(self.dba_roles, key=lambda r: ROLE_RANK[r])
