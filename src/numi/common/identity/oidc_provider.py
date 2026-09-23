"""Real OIDC-backed `IdentityProvider` (spec §3).

Targets the *standard*, not one vendor: OpenID Connect Discovery for
endpoint metadata, OAuth 2.0 client credentials for this service's own
directory access, and SCIM 2.0 (RFC 7643/7644) for the directory query
itself. Okta, Entra ID (Azure AD), Ping and generic OIDC/SCIM deployments
are all reachable through this one class with configuration only.

## The hard part: a webhook has no token

A browser login hands you an ID token, and the whole OIDC validation story
applies. A Slack or Teams webhook does not: after signature verification
all you hold is a *channel-native account id* — a Slack user id (`U012ABC`)
or an AAD object id. There is no token to validate, no `sub` claim, no
`userinfo` call available (userinfo needs the end user's own access token,
which this service will never possess).

So `resolve_by_external_account` is necessarily a **directory lookup**, not
a token validation: "which enterprise account carries this Slack user id as
an attribute?" This class answers it with a SCIM `GET /Users?filter=<attr>
eq "<id>"` against the IdP's directory, where `<attr>` is the per-channel
attribute configured in `config/identity.yaml`'s `oidc.channel_attributes`.

That configuration deliberately mirrors `MockIdentityProvider`'s
`channel_accounts` map, one layer up: the mock stores
`channel_accounts: {slack: U_MOCK_L2}` *per fictitious user*, while this
provider stores, once, *which directory attribute holds that value* and
asks the real IdP. Same shape, same meaning, real source of truth — and
the mapping is still configuration, never code.

**There is no default channel→attribute mapping, on purpose.** No standard
says where a Slack user id lives in a directory schema; it is invariably a
custom/extension attribute an operator populates at directory-sync time
(SCIM provisioning from Slack/Teams, an HR feed, or a scripted backfill).
An unconfigured channel therefore resolves to `None` — fail closed — rather
than guessing an attribute name and silently matching the wrong person.

## What this class will not accept

`resolve_by_external_account` takes a channel-verified account id and
nothing else. There is no code path here that accepts a display name, an
email typed into a chat message, or any value an LLM produced — matching
`IdentityProvider`'s own contract. The account id is additionally treated
as hostile input when it is interpolated into a SCIM filter (see
`_scim_string_literal`), because it arrives from outside this system even
though the channel adapter authenticated the envelope that carried it.

## Fail closed means `None`, not an exception

`MockIdentityProvider`'s contract is that an unknown account returns
`None`, and every call site (`gateway/api/deps.py::resolve_identity`,
`channels/api/app.py`) is written against that. A network failure, an
expired client secret, a 500 from the IdP, or a malformed user resource all
therefore return `None` here too: an identity that cannot be *proven* is an
identity that does not exist, and the caller's existing "not a recognized
DBA" path denies the request. Nothing is ever raised past these two
methods.

## What is cached, and what is deliberately not

Cached: the discovery document (static metadata) and this service's own
client-credentials access token (until shortly before it expires).

**Never cached: the user's groups and MFA state.** `refresh(subject_id)`
exists precisely so authorization re-derives them at decision time rather
than trusting an identity resolved minutes or hours earlier — a DBA removed
from `Enterprise-DBA-L3` must lose L3 on the very next tool call. Every
`refresh` performs a real `GET /Users/<id>` against the directory; there is
no memoized `VerifiedIdentity` anywhere in this class to return instead.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import yaml

from numi.common.identity.provider import GroupRoleMapping, IdentityProvider
from numi.common.models.identity import VerifiedIdentity
from numi.common.observability import get_logger

logger = get_logger(__name__)

# Refresh the client-credentials token this many seconds before it actually
# expires, so a request never races its own token's expiry in flight.
_TOKEN_EXPIRY_SKEW_SECONDS = 30.0
# Used when the token endpoint omits `expires_in`.
_DEFAULT_TOKEN_LIFETIME_SECONDS = 300.0

_DEFAULT_TIMEOUT_SECONDS = 10.0


def _scim_string_literal(value: str) -> str:
    """Quote a value for use inside a SCIM filter expression.

    SCIM filters are a query language, and the account id interpolated into
    one arrives from outside this system. An unescaped `"` would let a
    crafted account id close the literal and append its own filter terms —
    the SCIM equivalent of SQL injection, and here it would be an
    *authentication* bypass, not just a bad query (`... or userName pr`
    matches every user, and this provider takes the first result). Escaping
    backslash first, then the quote, is the whole of RFC 7644 §3.4.2.2's
    string-literal grammar.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class OIDCIdentityProvider(IdentityProvider):
    """`IdentityProvider` backed by a real OIDC IdP + SCIM directory.

    Configuration comes from two places, mirroring how the rest of this
    codebase splits secrets from policy:

    - **Environment** (`common/config.py`): `OIDC_ISSUER`,
      `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET` — the service's own
      credentials for talking to the IdP.
    - **`config/identity.yaml`**: the `identity` section (group → role
      mapping, shared verbatim with the mock) and an `oidc` section
      describing this deployment's directory:

      ```yaml
      oidc:
        directory_endpoint: "https://idp.example.com/scim/v2/Users"
        channel_attributes:
          slack: "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:slackUserId"
          teams: "externalId"
        groups_attribute: "groups"
        mfa_attribute: "mfaEnrolled"
        scopes: "scim:read"
      ```

    `http_client` is the injectable seam: anything exposing `httpx`-shaped
    `async get(url, headers=..., params=...)` and `async post(url,
    data=..., headers=...)` coroutines returning an object with
    `status_code` and `.json()`. Left as `None` (production), an
    `httpx.AsyncClient` is built lazily on first use. Tests pass a fake and
    no network call is ever made.

    **On `mfa_satisfied`.** For a chat channel there is no authentication
    *event* to describe — the DBA did not just log in, they typed a message
    into Slack. What the directory can assert is enrollment/enforcement
    state, so that is what this field carries here, read from the
    configured `mfa_attribute`. When the IdP asserts nothing, it is
    `False` — fail closed, matching every other unknown in this module.
    """

    def __init__(
        self,
        *,
        issuer: str,
        client_id: str = "",
        client_secret: str = "",
        identity_config_path: str | Path = "config/identity.yaml",
        http_client: Any = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        # `Any` rather than `httpx.AsyncClient | None`: the point of the
        # seam is that a test double satisfying the same two coroutines is
        # equally valid here.
        self._http: Any = http_client
        self._owns_http = http_client is None

        raw = yaml.safe_load(Path(identity_config_path).read_text(encoding="utf-8")) or {}
        self._roles = GroupRoleMapping(raw["identity"])

        oidc_cfg: dict[str, Any] = raw.get("oidc") or {}
        self._directory_endpoint: str = (oidc_cfg.get("directory_endpoint") or "").rstrip("/")
        self._channel_attributes: dict[str, str] = oidc_cfg.get("channel_attributes") or {}
        self._groups_attribute: str = oidc_cfg.get("groups_attribute") or "groups"
        self._mfa_attribute: str = oidc_cfg.get("mfa_attribute") or "mfaEnrolled"
        self._scopes: str = oidc_cfg.get("scopes") or ""

        # Caches — metadata and this service's own token only; never a
        # resolved identity (see the module docstring).
        self._discovery: dict[str, Any] | None = None
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

    # --- plumbing ---------------------------------------------------------- #

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def aclose(self) -> None:
        """Release the lazily-built HTTP client (no-op for an injected one,
        which belongs to whoever passed it in)."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _discovery_document(self) -> dict[str, Any]:
        """Fetch and cache `/.well-known/openid-configuration`.

        Endpoint metadata is static for the life of a deployment, so this is
        cached for the process — unlike anything about a *user*.
        """
        if self._discovery is not None:
            return self._discovery
        url = f"{self._issuer}/.well-known/openid-configuration"
        response = await self._client().get(url)
        if response.status_code != 200:
            raise RuntimeError(f"OIDC discovery returned HTTP {response.status_code}")
        document = response.json()
        if not isinstance(document, dict):
            raise RuntimeError("OIDC discovery document was not a JSON object")
        self._discovery = document
        return document

    async def _bearer_token(self) -> str:
        """This *service's* access token for directory calls (OAuth 2.0
        client credentials) — never a user's token.

        Cached until shortly before expiry: it authenticates Numi to the
        IdP, and re-minting it per lookup would add a full round trip to
        every single tool call without changing any authorization outcome.
        The user's own group/MFA state is what must never be cached, and it
        is read fresh on every lookup regardless of this token's age.
        """
        now = time.monotonic()
        if self._access_token and now < self._access_token_expires_at:
            return self._access_token

        discovery = await self._discovery_document()
        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise RuntimeError("OIDC discovery document has no token_endpoint")

        form = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scopes:
            form["scope"] = self._scopes

        response = await self._client().post(
            token_endpoint,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            # Never log the body: a token endpoint's error response can
            # echo the client secret back in some deployments.
            raise RuntimeError(f"OIDC token endpoint returned HTTP {response.status_code}")
        payload = response.json()
        token = (payload or {}).get("access_token") if isinstance(payload, dict) else None
        if not token:
            raise RuntimeError("OIDC token response carried no access_token")

        lifetime = float(payload.get("expires_in") or _DEFAULT_TOKEN_LIFETIME_SECONDS)
        self._access_token = str(token)
        self._access_token_expires_at = now + max(lifetime - _TOKEN_EXPIRY_SKEW_SECONDS, 0.0)
        return self._access_token

    async def _directory_get(self, url: str, params: dict[str, str] | None = None) -> Any:
        token = await self._bearer_token()
        response = await self._client().get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/scim+json"},
            params=params or {},
        )
        if response.status_code == 404:
            return None  # a genuinely unknown subject, not an error
        if response.status_code != 200:
            raise RuntimeError(f"SCIM directory returned HTTP {response.status_code}")
        return response.json()

    # --- SCIM resource parsing --------------------------------------------- #

    def _groups_of(self, user: dict[str, Any]) -> list[str]:
        """Group names out of a SCIM user resource.

        SCIM models `groups` as complex multi-valued attributes
        (`{"value": <id>, "display": <name>}`), but plenty of IdPs project a
        plain list of strings into a custom claim instead — and the group
        *names* are what `config/identity.yaml` maps to roles. Accept both,
        and take `display` in preference to the opaque `value`.
        """
        raw = user.get(self._groups_attribute) or []
        if not isinstance(raw, list):
            return []
        groups: list[str] = []
        for item in raw:
            if isinstance(item, str):
                groups.append(item)
            elif isinstance(item, dict):
                name = item.get("display") or item.get("value")
                if isinstance(name, str):
                    groups.append(name)
        return groups

    @staticmethod
    def _email_of(user: dict[str, Any]) -> str:
        raw = user.get("emails")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            primary = next(
                (e for e in raw if isinstance(e, dict) and e.get("primary") and e.get("value")),
                None,
            )
            if primary:
                return str(primary["value"])
            first = next((e for e in raw if isinstance(e, dict) and e.get("value")), None)
            if first:
                return str(first["value"])
        user_name = user.get("userName")
        return str(user_name) if user_name else ""

    @staticmethod
    def _display_name_of(user: dict[str, Any]) -> str:
        display = user.get("displayName")
        if isinstance(display, str) and display:
            return display
        name = user.get("name")
        if isinstance(name, dict):
            formatted = name.get("formatted")
            if isinstance(formatted, str) and formatted:
                return formatted
        user_name = user.get("userName")
        return str(user_name) if user_name else ""

    def _mfa_of(self, user: dict[str, Any]) -> bool:
        """Directory-asserted MFA state, defaulting to False.

        Anything other than an explicit affirmative is treated as "not
        satisfied": an IdP that doesn't publish this attribute must not
        accidentally read as "MFA done".
        """
        value = user.get(self._mfa_attribute)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return False

    def _to_identity(self, user: dict[str, Any]) -> VerifiedIdentity | None:
        """Build a `VerifiedIdentity` from a SCIM user resource, or `None`
        if the resource can't support one.

        A user resource with no `id` cannot be re-resolved later (that id is
        exactly what `refresh` is keyed on), and an *inactive* account must
        never resolve at all — a disabled leaver is the single most
        important case for this whole lookup to get right.
        """
        if not isinstance(user, dict):
            return None
        subject_id = user.get("id")
        if not subject_id:
            return None
        if user.get("active") is False:
            logger.warning("oidc_identity_inactive_account", subject_id=str(subject_id))
            return None

        groups = self._groups_of(user)
        return VerifiedIdentity(
            subject_id=str(subject_id),
            email=self._email_of(user),
            display_name=self._display_name_of(user),
            enterprise_groups=groups,
            dba_roles=self._roles.derive_roles(groups),
            mfa_satisfied=self._mfa_of(user),
            authenticated_at=dt.datetime.now(dt.UTC).isoformat(),
        )

    # --- IdentityProvider -------------------------------------------------- #

    async def resolve_by_external_account(
        self, channel: str, external_account_id: str
    ) -> VerifiedIdentity | None:
        """Find the enterprise account carrying this channel account id.

        Fails closed (`None`) on: an unconfigured directory endpoint, a
        channel with no configured attribute, a blank account id, no match,
        more than one match, or any transport/auth failure.

        The ambiguous-match case matters: two directory entries claiming the
        same Slack id means the directory itself is inconsistent, and
        picking either one would be choosing an identity arbitrarily. That
        is exactly the kind of guess an identity provider must never make.
        """
        if not self._directory_endpoint:
            logger.warning("oidc_directory_endpoint_not_configured", channel=channel)
            return None

        account_id = (external_account_id or "").strip()
        if not account_id:
            return None

        attribute = self._channel_attributes.get(channel)
        if not attribute:
            logger.warning("oidc_channel_attribute_not_configured", channel=channel)
            return None

        try:
            payload = await self._directory_get(
                self._directory_endpoint,
                params={"filter": f"{attribute} eq {_scim_string_literal(account_id)}"},
            )
        except Exception as exc:  # noqa: BLE001 — an unprovable identity is no identity
            logger.warning(
                "oidc_identity_resolution_failed", channel=channel, error=repr(exc)
            )
            return None

        resources = (payload or {}).get("Resources") if isinstance(payload, dict) else None
        if not isinstance(resources, list) or not resources:
            return None
        if len(resources) > 1:
            logger.warning(
                "oidc_identity_ambiguous_external_account",
                channel=channel,
                match_count=len(resources),
            )
            return None
        return self._to_identity(resources[0])

    async def refresh(self, subject_id: str) -> VerifiedIdentity | None:
        """Re-read this subject straight from the directory.

        This is a real `GET /Users/<id>` every time — never a cached
        `VerifiedIdentity`, and never the result of the earlier
        `resolve_by_external_account` call. That is the entire purpose of
        the method: group membership and MFA state are re-derived at
        authorization time, so a revoked role takes effect on the next tool
        call rather than whenever a cache happens to lapse.
        """
        if not self._directory_endpoint:
            logger.warning("oidc_directory_endpoint_not_configured")
            return None

        subject = (subject_id or "").strip()
        if not subject:
            return None

        try:
            payload = await self._directory_get(f"{self._directory_endpoint}/{subject}")
        except Exception as exc:  # noqa: BLE001 — see resolve_by_external_account
            logger.warning("oidc_identity_refresh_failed", error=repr(exc))
            return None

        if payload is None:
            return None
        return self._to_identity(payload)
