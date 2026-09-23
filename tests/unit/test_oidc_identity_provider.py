"""Real OIDC/SCIM identity provider (spec §3).

No OIDC tenant exists in this environment, so every test here injects a
fake HTTP client into `OIDCIdentityProvider(http_client=...)` and asserts on
this codebase's behavior — the discovery/token/SCIM request shapes it
emits, how it parses a user resource, and (most importantly) that it fails
*closed* on everything else. Zero network calls, same injectable-client
pattern as `tests/unit/test_secrets_providers.py` and the LLM provider
tests.

The contract being pinned, from `MockIdentityProvider` and
`IdentityProvider`'s own docstrings: an identity that cannot be proven
resolves to `None`, never an exception — every call site
(`gateway/api/deps.py::resolve_identity`, `channels/api/app.py`) is written
against that and denies the request on `None`.
"""

from __future__ import annotations

import pytest

from numi.common.identity.factory import build_identity_provider
from numi.common.identity.oidc_provider import OIDCIdentityProvider, _scim_string_literal
from numi.common.identity.provider import MockIdentityProvider
from numi.common.models.failures import NumiError

_IDENTITY_YAML = """
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

oidc:
  directory_endpoint: "https://idp.example.test/scim/v2/Users"
  channel_attributes:
    slack: "urn:numi:slackUserId"
    teams: "externalId"
  groups_attribute: "groups"
  mfa_attribute: "mfaEnrolled"
  scopes: "scim:read"
"""

_ISSUER = "https://idp.example.test"
_DISCOVERY = {
    "issuer": _ISSUER,
    "token_endpoint": f"{_ISSUER}/oauth2/v1/token",
    "jwks_uri": f"{_ISSUER}/oauth2/v1/keys",
    "userinfo_endpoint": f"{_ISSUER}/oauth2/v1/userinfo",
}

_SCIM_USER = {
    "id": "idp-subject-0001",
    "userName": "dba.l2@example.test",
    "active": True,
    "displayName": "Real DBA L2",
    "emails": [{"value": "dba.l2@example.test", "primary": True}],
    "groups": [
        {"value": "g-1", "display": "Enterprise-DBA"},
        {"value": "g-2", "display": "Enterprise-DBA-L2"},
    ],
    "mfaEnrolled": True,
}


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHTTPClient:
    """Minimal stand-in for `httpx.AsyncClient` — the two coroutines
    `OIDCIdentityProvider` actually uses.

    `directory_responses` is a list consumed one per SCIM call, so a test
    can prove `refresh` issued its OWN request rather than replaying an
    earlier answer.
    """

    def __init__(self, directory_responses=None, *, token_status=200, discovery_status=200):
        self.directory_responses = list(directory_responses or [])
        self.token_status = token_status
        self.discovery_status = discovery_status
        self.get_calls: list[dict] = []
        self.post_calls: list[dict] = []

    async def get(self, url, headers=None, params=None):
        self.get_calls.append({"url": url, "headers": headers or {}, "params": params or {}})
        if url.endswith("/.well-known/openid-configuration"):
            return _FakeResponse(self.discovery_status, _DISCOVERY)
        if not self.directory_responses:
            raise AssertionError(f"unexpected extra directory call to {url}")
        return self.directory_responses.pop(0)

    async def post(self, url, data=None, headers=None):
        self.post_calls.append({"url": url, "data": data or {}})
        if self.token_status != 200:
            return _FakeResponse(self.token_status, {"error": "invalid_client"})
        return _FakeResponse(200, {"access_token": "svc-token-abc", "expires_in": 3600})

    @property
    def directory_get_calls(self) -> list[dict]:
        return [c for c in self.get_calls if "/scim/" in c["url"]]


@pytest.fixture
def identity_config(tmp_path):
    path = tmp_path / "identity.yaml"
    path.write_text(_IDENTITY_YAML, encoding="utf-8")
    return path


def _provider(identity_config, http_client) -> OIDCIdentityProvider:
    return OIDCIdentityProvider(
        issuer=_ISSUER,
        client_id="numi-gateway",
        client_secret="shhh",
        identity_config_path=identity_config,
        http_client=http_client,
    )


def _found(user=None) -> _FakeResponse:
    return _FakeResponse(200, {"totalResults": 1, "Resources": [user or dict(_SCIM_USER)]})


def _not_found() -> _FakeResponse:
    return _FakeResponse(200, {"totalResults": 0, "Resources": []})


# --------------------------------------------------------------------------- #
# resolve_by_external_account — the happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_known_slack_account_resolves_to_a_verified_identity(identity_config):
    http = _FakeHTTPClient([_found()])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")

    assert identity is not None
    assert identity.subject_id == "idp-subject-0001"
    assert identity.email == "dba.l2@example.test"
    assert identity.display_name == "Real DBA L2"
    assert identity.mfa_satisfied is True
    # Roles are derived from directory groups via config/identity.yaml's
    # mapping — the very same GroupRoleMapping the mock uses.
    assert [r.value for r in identity.dba_roles] == ["DBA_L2"]
    assert identity.is_dba()


@pytest.mark.asyncio
async def test_the_scim_query_filters_on_the_configured_channel_attribute(identity_config):
    http = _FakeHTTPClient([_found()])
    provider = _provider(identity_config, http)

    await provider.resolve_by_external_account("slack", "U012ABC")

    call = http.directory_get_calls[0]
    assert call["url"] == "https://idp.example.test/scim/v2/Users"
    assert call["params"]["filter"] == 'urn:numi:slackUserId eq "U012ABC"'
    assert call["headers"]["Authorization"] == "Bearer svc-token-abc"


@pytest.mark.asyncio
async def test_a_teams_account_uses_its_own_configured_attribute(identity_config):
    http = _FakeHTTPClient([_found()])
    provider = _provider(identity_config, http)

    await provider.resolve_by_external_account("teams", "aad-object-123")

    assert http.directory_get_calls[0]["params"]["filter"] == 'externalId eq "aad-object-123"'


@pytest.mark.asyncio
async def test_the_client_credentials_token_is_requested_from_the_discovered_endpoint(
    identity_config,
):
    http = _FakeHTTPClient([_found()])
    provider = _provider(identity_config, http)

    await provider.resolve_by_external_account("slack", "U012ABC")

    assert http.post_calls[0]["url"] == _DISCOVERY["token_endpoint"]
    assert http.post_calls[0]["data"]["grant_type"] == "client_credentials"
    assert http.post_calls[0]["data"]["client_id"] == "numi-gateway"
    assert http.post_calls[0]["data"]["scope"] == "scim:read"


@pytest.mark.asyncio
async def test_a_flat_list_of_group_names_is_accepted_too(identity_config):
    """Plenty of IdPs project groups as plain strings rather than SCIM's
    complex multi-valued form."""
    user = dict(_SCIM_USER, groups=["Enterprise-DBA", "Enterprise-DBA-Managers"])
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert [r.value for r in identity.dba_roles] == ["DBA_MANAGER"]


# --------------------------------------------------------------------------- #
# resolve_by_external_account — fail closed
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_unknown_external_account_resolves_to_none(identity_config):
    http = _FakeHTTPClient([_not_found()])
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U_NOBODY") is None


@pytest.mark.asyncio
async def test_an_idp_error_returns_none_rather_than_raising(identity_config):
    """The contract every call site is written against: a broken IdP denies
    the request, it does not blow up the webhook handler."""
    http = _FakeHTTPClient([_FakeResponse(500, None)])
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_a_transport_failure_returns_none_rather_than_raising(identity_config):
    class _ExplodingClient(_FakeHTTPClient):
        async def get(self, url, headers=None, params=None):
            if url.endswith("/.well-known/openid-configuration"):
                return _FakeResponse(200, _DISCOVERY)
            raise ConnectionError("connection refused")

    provider = _provider(identity_config, _ExplodingClient())
    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_a_failed_client_credentials_grant_returns_none(identity_config):
    http = _FakeHTTPClient([_found()], token_status=401)
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_an_inactive_directory_account_never_resolves(identity_config):
    """A disabled leaver is the single most important case for this lookup
    to get right."""
    http = _FakeHTTPClient([_found(dict(_SCIM_USER, active=False))])
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_an_ambiguous_match_resolves_to_none(identity_config):
    """Two directory entries claiming the same Slack id means the directory
    is inconsistent — picking either would be choosing an identity
    arbitrarily."""
    http = _FakeHTTPClient(
        [
            _FakeResponse(
                200,
                {
                    "totalResults": 2,
                    "Resources": [dict(_SCIM_USER), dict(_SCIM_USER, id="idp-subject-0002")],
                },
            )
        ]
    )
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_a_channel_with_no_configured_attribute_resolves_to_none(identity_config):
    """No standard says where a Discord id lives in a directory schema, so
    an unmapped channel fails closed instead of guessing an attribute."""
    http = _FakeHTTPClient([])
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("discord", "12345") is None
    assert http.directory_get_calls == []


@pytest.mark.asyncio
async def test_a_blank_account_id_never_reaches_the_directory(identity_config):
    http = _FakeHTTPClient([])
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "") is None
    assert await provider.resolve_by_external_account("slack", "   ") is None
    assert http.directory_get_calls == []


@pytest.mark.asyncio
async def test_a_user_resource_without_an_id_resolves_to_none(identity_config):
    """Without the IdP's stable subject id there is nothing for `refresh` to
    re-resolve later."""
    user = {k: v for k, v in _SCIM_USER.items() if k != "id"}
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_an_unconfigured_directory_endpoint_resolves_to_none(tmp_path):
    path = tmp_path / "identity.yaml"
    path.write_text(_IDENTITY_YAML.replace('"https://idp.example.test/scim/v2/Users"', '""'))
    http = _FakeHTTPClient([])

    provider = _provider(path, http)
    assert await provider.resolve_by_external_account("slack", "U012ABC") is None
    assert http.get_calls == []


@pytest.mark.asyncio
async def test_missing_mfa_attribute_reads_as_not_satisfied(identity_config):
    """An IdP that doesn't publish the attribute must not accidentally read
    as "MFA done"."""
    user = {k: v for k, v in _SCIM_USER.items() if k != "mfaEnrolled"}
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.mfa_satisfied is False


# --------------------------------------------------------------------------- #
# SCIM filter injection
# --------------------------------------------------------------------------- #


def test_a_scim_filter_literal_escapes_quotes_and_backslashes():
    """An unescaped quote would let a crafted account id close the literal
    and append its own filter terms — an authentication bypass, since the
    provider takes the single match."""
    assert _scim_string_literal('U0" or userName pr or "x') == (
        '"U0\\" or userName pr or \\"x"'
    )
    assert _scim_string_literal("back\\slash") == '"back\\\\slash"'


@pytest.mark.asyncio
async def test_a_crafted_account_id_cannot_break_out_of_the_filter(identity_config):
    http = _FakeHTTPClient([_not_found()])
    provider = _provider(identity_config, http)

    await provider.resolve_by_external_account("slack", 'U0" or userName pr or "')

    sent = http.directory_get_calls[0]["params"]["filter"]
    # Exactly one unescaped quote pair remains — the one this code wrote.
    assert sent == 'urn:numi:slackUserId eq "U0\\" or userName pr or \\""'


# --------------------------------------------------------------------------- #
# refresh — always a real, separate call
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_refresh_reads_the_subject_straight_from_the_directory(identity_config):
    http = _FakeHTTPClient([_FakeResponse(200, dict(_SCIM_USER))])
    provider = _provider(identity_config, http)

    identity = await provider.refresh("idp-subject-0001")

    assert identity is not None
    assert identity.subject_id == "idp-subject-0001"
    assert http.directory_get_calls[0]["url"] == (
        "https://idp.example.test/scim/v2/Users/idp-subject-0001"
    )


@pytest.mark.asyncio
async def test_refresh_makes_its_own_call_and_never_replays_a_cached_identity(identity_config):
    """The whole purpose of `refresh`: group membership is re-derived at
    authorization time, so a revoked role takes effect on the very next
    tool call.

    The fake serves a *different* second response — an account demoted out
    of every DBA group. If `refresh` returned anything cached from the
    initial resolve, it would still report DBA_L2 here.
    """
    demoted = dict(_SCIM_USER, groups=[{"value": "g-9", "display": "Everyone"}])
    http = _FakeHTTPClient([_found(), _FakeResponse(200, demoted)])
    provider = _provider(identity_config, http)

    first = await provider.resolve_by_external_account("slack", "U012ABC")
    assert first is not None
    assert [r.value for r in first.dba_roles] == ["DBA_L2"]

    refreshed = await provider.refresh("idp-subject-0001")

    # A second, independent directory request was actually issued...
    assert len(http.directory_get_calls) == 2
    assert http.directory_get_calls[1]["url"].endswith("/Users/idp-subject-0001")
    # ...and the revocation took effect immediately.
    assert refreshed is not None
    assert refreshed.dba_roles == []
    assert refreshed.is_dba() is False


@pytest.mark.asyncio
async def test_refresh_of_an_unknown_subject_returns_none(identity_config):
    http = _FakeHTTPClient([_FakeResponse(404, None)])
    provider = _provider(identity_config, http)

    assert await provider.refresh("idp-subject-nobody") is None


@pytest.mark.asyncio
async def test_refresh_returns_none_on_an_idp_error_rather_than_raising(identity_config):
    http = _FakeHTTPClient([_FakeResponse(503, None)])
    provider = _provider(identity_config, http)

    assert await provider.refresh("idp-subject-0001") is None


@pytest.mark.asyncio
async def test_the_service_token_is_reused_across_lookups(identity_config):
    """Numi's *own* client-credentials token is cached (it authenticates
    the service, not the user); the user's groups are not, as the test
    above pins."""
    http = _FakeHTTPClient([_found(), _FakeResponse(200, dict(_SCIM_USER))])
    provider = _provider(identity_config, http)

    await provider.resolve_by_external_account("slack", "U012ABC")
    await provider.refresh("idp-subject-0001")

    assert len(http.post_calls) == 1  # one token grant, two directory calls
    assert len(http.directory_get_calls) == 2


# --------------------------------------------------------------------------- #
# build_identity_provider routing
# --------------------------------------------------------------------------- #


def _settings(**overrides):
    from numi.common.config import Settings

    return Settings(_env_file=None, **overrides)


def test_build_identity_provider_defaults_to_the_mock_unchanged():
    """Adding the OIDC option must not change behavior for anyone still on
    the default."""
    settings = _settings()
    assert settings.identity_provider == "mock"
    provider = build_identity_provider(settings)
    assert isinstance(provider, MockIdentityProvider)


def test_build_identity_provider_routes_to_the_real_oidc_class():
    provider = build_identity_provider(
        _settings(
            identity_provider="oidc",
            oidc_issuer=_ISSUER,
            oidc_client_id="numi-gateway",
            oidc_client_secret="shhh",
        )
    )
    assert isinstance(provider, OIDCIdentityProvider)


def test_build_identity_provider_refuses_oidc_without_an_issuer():
    with pytest.raises(NumiError) as exc_info:
        build_identity_provider(_settings(identity_provider="oidc"))
    assert "OIDC_ISSUER" in exc_info.value.detail


def test_build_identity_provider_rejects_an_unknown_name():
    """A typo must never silently fall back to the fictitious mock
    directory."""
    with pytest.raises(NumiError):
        build_identity_provider(_settings(identity_provider="okta-ish"))


# --------------------------------------------------------------------------- #
# Discovery document — caching and malformed responses
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_discovery_document_is_fetched_once_and_cached(identity_config):
    """Static endpoint metadata for the life of the process — unlike
    anything about a user (see the module docstring)."""
    http = _FakeHTTPClient([_found(), _found()])
    provider = _provider(identity_config, http)

    await provider.resolve_by_external_account("slack", "U012ABC")
    await provider.resolve_by_external_account("slack", "U012ABC")

    discovery_calls = [c for c in http.get_calls if c["url"].endswith("openid-configuration")]
    assert len(discovery_calls) == 1


@pytest.mark.asyncio
async def test_a_non_200_discovery_response_returns_none_rather_than_raising(identity_config):
    http = _FakeHTTPClient([], discovery_status=503)
    provider = _provider(identity_config, http)

    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_a_non_dict_discovery_document_returns_none_rather_than_raising(identity_config):
    class _MalformedDiscoveryClient(_FakeHTTPClient):
        async def get(self, url, headers=None, params=None):
            self.get_calls.append({"url": url, "headers": headers or {}, "params": params or {}})
            if url.endswith("/.well-known/openid-configuration"):
                return _FakeResponse(200, ["not", "an", "object"])
            raise AssertionError("should never reach the directory")

    provider = _provider(identity_config, _MalformedDiscoveryClient())
    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_a_discovery_document_missing_a_token_endpoint_returns_none(identity_config):
    class _NoTokenEndpointClient(_FakeHTTPClient):
        async def get(self, url, headers=None, params=None):
            self.get_calls.append({"url": url, "headers": headers or {}, "params": params or {}})
            if url.endswith("/.well-known/openid-configuration"):
                return _FakeResponse(200, {"issuer": _ISSUER})  # no token_endpoint
            raise AssertionError("should never reach the directory")

    provider = _provider(identity_config, _NoTokenEndpointClient())
    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


@pytest.mark.asyncio
async def test_a_token_response_carrying_no_access_token_returns_none(identity_config):
    class _EmptyTokenClient(_FakeHTTPClient):
        async def post(self, url, data=None, headers=None):
            self.post_calls.append({"url": url, "data": data or {}})
            return _FakeResponse(200, {"token_type": "Bearer"})  # no access_token

    provider = _provider(identity_config, _EmptyTokenClient())
    assert await provider.resolve_by_external_account("slack", "U012ABC") is None


# --------------------------------------------------------------------------- #
# SCIM user resource parsing — fallback branches
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_non_list_groups_attribute_is_treated_as_no_groups(identity_config):
    user = dict(_SCIM_USER, groups="not-a-list")
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.enterprise_groups == []


@pytest.mark.asyncio
async def test_a_plain_string_emails_field_is_accepted(identity_config):
    user = dict(_SCIM_USER, emails="single@example.test")
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.email == "single@example.test"


@pytest.mark.asyncio
async def test_emails_with_no_primary_flag_falls_back_to_the_first_value(identity_config):
    user = dict(_SCIM_USER, emails=[{"value": "secondary@example.test"}])
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.email == "secondary@example.test"


@pytest.mark.asyncio
async def test_no_emails_at_all_falls_back_to_username(identity_config):
    user = {k: v for k, v in _SCIM_USER.items() if k != "emails"}
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.email == _SCIM_USER["userName"]


@pytest.mark.asyncio
async def test_display_name_falls_back_to_name_formatted_when_displayname_is_absent(identity_config):
    user = {k: v for k, v in _SCIM_USER.items() if k != "displayName"}
    user["name"] = {"formatted": "Formatted Name"}
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.display_name == "Formatted Name"


@pytest.mark.asyncio
async def test_display_name_falls_back_to_username_when_nothing_else_is_present(identity_config):
    user = {k: v for k, v in _SCIM_USER.items() if k not in ("displayName", "name")}
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.display_name == _SCIM_USER["userName"]


@pytest.mark.asyncio
async def test_a_string_mfa_attribute_of_yes_reads_as_satisfied(identity_config):
    user = dict(_SCIM_USER, mfaEnrolled="yes")
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.mfa_satisfied is True


@pytest.mark.asyncio
async def test_a_string_mfa_attribute_of_false_text_reads_as_not_satisfied(identity_config):
    user = dict(_SCIM_USER, mfaEnrolled="false")
    http = _FakeHTTPClient([_found(user)])
    provider = _provider(identity_config, http)

    identity = await provider.resolve_by_external_account("slack", "U012ABC")
    assert identity is not None
    assert identity.mfa_satisfied is False


# --------------------------------------------------------------------------- #
# refresh — fail-closed branches not already covered above
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_refresh_with_an_unconfigured_directory_endpoint_returns_none(tmp_path):
    path = tmp_path / "identity.yaml"
    path.write_text(_IDENTITY_YAML.replace('"https://idp.example.test/scim/v2/Users"', '""'))
    http = _FakeHTTPClient([])

    provider = _provider(path, http)
    assert await provider.refresh("idp-subject-0001") is None
    assert http.get_calls == []


@pytest.mark.asyncio
async def test_refresh_with_a_blank_subject_id_never_reaches_the_directory(identity_config):
    http = _FakeHTTPClient([])
    provider = _provider(identity_config, http)

    assert await provider.refresh("") is None
    assert await provider.refresh("   ") is None
    assert http.directory_get_calls == []


@pytest.mark.asyncio
async def test_refresh_of_a_malformed_non_object_payload_returns_none(identity_config):
    """A SCIM server that returns a JSON array instead of a user resource
    must fail closed rather than raise deep inside attribute access."""
    http = _FakeHTTPClient([_FakeResponse(200, ["not", "a", "user", "resource"])])
    provider = _provider(identity_config, http)

    assert await provider.refresh("idp-subject-0001") is None
