"""`config/identity.yaml` is committed to version control and documents
itself as fictitious-users-only. Testing against a real channel (e.g. a
real Slack workspace) needs a real personal entry somewhere, though — a
gitignored sibling `local_identity.yaml` lets that happen without ever
putting real personal data in the committed file (see
config/local_identity.example.yaml)."""

from __future__ import annotations

import pytest

from numi.common.identity.provider import MockIdentityProvider

_BASE_IDENTITY_YAML = """
identity:
  groups:
    dba_team:
      - "Enterprise-DBA"
  roles:
    DBA_L2:
      groups:
        - "Enterprise-DBA-L2"
    DBA_MANAGER:
      groups:
        - "Enterprise-DBA-Managers"

mock_directory:
  - subject_id: "mock-001"
    email: "mock@example.test"
    display_name: "Mock User"
    channel_accounts:
      slack: "U_MOCK"
    groups: ["Enterprise-DBA", "Enterprise-DBA-L2"]
    mfa_satisfied: true
"""

_LOCAL_IDENTITY_YAML = """
mock_directory:
  - subject_id: "real-001"
    email: "real.person@example.com"
    display_name: "Real Person"
    channel_accounts:
      slack: "U0REAL123"
    groups: ["Enterprise-DBA", "Enterprise-DBA-Managers"]
    mfa_satisfied: true
"""


@pytest.mark.asyncio
async def test_a_local_identity_overlay_is_merged_in_when_present(tmp_path):
    (tmp_path / "identity.yaml").write_text(_BASE_IDENTITY_YAML, encoding="utf-8")
    (tmp_path / "local_identity.yaml").write_text(_LOCAL_IDENTITY_YAML, encoding="utf-8")

    provider = MockIdentityProvider(tmp_path / "identity.yaml")

    mock_identity = await provider.resolve_by_external_account("slack", "U_MOCK")
    assert mock_identity is not None
    real_identity = await provider.resolve_by_external_account("slack", "U0REAL123")
    assert real_identity is not None
    assert real_identity.email == "real.person@example.com"
    assert "DBA_MANAGER" in [r.value for r in real_identity.dba_roles]


@pytest.mark.asyncio
async def test_no_overlay_file_is_not_an_error(tmp_path):
    (tmp_path / "identity.yaml").write_text(_BASE_IDENTITY_YAML, encoding="utf-8")
    # No local_identity.yaml written alongside it.

    provider = MockIdentityProvider(tmp_path / "identity.yaml")

    assert await provider.resolve_by_external_account("slack", "U_MOCK") is not None
    assert await provider.resolve_by_external_account("slack", "U0REAL123") is None


@pytest.mark.asyncio
async def test_the_real_committed_identity_yaml_still_loads_cleanly():
    """config/identity.yaml itself must never require a local overlay to
    load — the overlay is additive only."""
    provider = MockIdentityProvider("config/identity.yaml")
    identity = await provider.resolve_by_external_account("slack", "U_MOCK_L2")
    assert identity is not None
    assert identity.email == "dba.l2@example.test"
