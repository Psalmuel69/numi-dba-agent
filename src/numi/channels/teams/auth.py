"""Microsoft Teams bot request authentication (spec §4, §43).

Every inbound Teams activity carries a bearer JWT issued by the Bot
Framework/Azure AD, signed by a key published at Microsoft's JWKS endpoint.
`BotFrameworkJWTVerifier` is the production implementation; `DevTeamsAuthVerifier`
is a clearly-labeled local-development stand-in so the full webhook ->
identity -> Agent flow can be exercised without real Azure AD credentials
(spec §66).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TeamsAuthError(Exception):
    pass


class TeamsAuthVerifier(ABC):
    @abstractmethod
    async def verify(self, authorization_header: str | None) -> str:
        """Returns the verified caller's AAD object id ('aud'/'appid' claim
        context) on success; raises TeamsAuthError otherwise."""


class BotFrameworkJWTVerifier(TeamsAuthVerifier):
    """Validates the Bot Framework's JWT against Microsoft's published JWKS,
    checking signature, issuer, audience (== this bot's app id), and
    expiry — never trusting an unsigned or self-asserted identity claim."""

    _OPENID_METADATA_URL = "https://login.botframework.com/v1/.well-known/openidconfiguration"

    def __init__(self, app_id: str):
        self._app_id = app_id
        self._jwks_client = None

    def _get_jwks_client(self):
        if self._jwks_client is None:
            import jwt

            # In production this resolves `jwks_uri` from the OpenID
            # metadata document above; PyJWKClient handles fetch + cache.
            self._jwks_client = jwt.PyJWKClient(
                "https://login.botframework.com/v1/.well-known/keys"
            )
        return self._jwks_client

    async def verify(self, authorization_header: str | None) -> str:
        import jwt

        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise TeamsAuthError("Missing bearer token.")
        token = authorization_header.removeprefix("Bearer ").strip()

        try:
            signing_key = self._get_jwks_client().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._app_id,
                issuer="https://api.botframework.com",
            )
        except Exception as exc:  # noqa: BLE001
            raise TeamsAuthError(f"Teams bot token validation failed: {exc}") from exc

        return claims.get("appid", "")


class DevTeamsAuthVerifier(TeamsAuthVerifier):
    """LOCAL DEVELOPMENT ONLY. Accepts a static shared bearer value from
    configuration instead of validating a real Azure AD-issued JWT. Never
    selected in production (see `channels.api.app.build_teams_verifier`)."""

    def __init__(self, dev_shared_token: str):
        self._dev_shared_token = dev_shared_token

    async def verify(self, authorization_header: str | None) -> str:
        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise TeamsAuthError("Missing bearer token.")
        token = authorization_header.removeprefix("Bearer ").strip()
        if not self._dev_shared_token or token != self._dev_shared_token:
            raise TeamsAuthError("Invalid development bot token.")
        return "dev-bot"
