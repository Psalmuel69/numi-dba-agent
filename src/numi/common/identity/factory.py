"""Identity provider selection (`IDENTITY_PROVIDER`).

The analogue of `execution.credentials.provider.build_credential_provider`,
and it exists for the same reason: *which* implementation a deployment runs
is a configuration decision, made in exactly one place, so no service ever
grows its own `if settings.identity_provider == ...` ladder. Every call
site — the Gateway's `GatewayState`, the Channels app — constructs its
provider through this function and is otherwise written purely against the
`IdentityProvider` interface.

Lives in its own module rather than in `provider.py` so that importing the
interface (or the mock) never drags in the OIDC implementation and its
transitive HTTP machinery.
"""

from __future__ import annotations

from numi.common.config import Settings
from numi.common.identity.provider import IdentityProvider, MockIdentityProvider
from numi.common.models.failures import FailureCode, NumiError


def build_identity_provider(settings: Settings) -> IdentityProvider:
    """Construct the configured `IdentityProvider`.

    `mock` (the default) is unchanged from before this factory existed:
    `MockIdentityProvider(settings.identity_config_path)`, the config-driven
    dev/test directory. `Settings.validate_for_production` separately
    refuses to start a production process while it is selected.

    `oidc` builds the real `OIDCIdentityProvider`. It is rejected here, at
    startup, if `OIDC_ISSUER` is unset — a provider that could never
    resolve anyone would otherwise fail closed on every request instead,
    which is safe but diagnosable only from request logs. Failing at
    construction turns a misconfiguration into one clear startup error.

    An unrecognized name is an error, never a silent fallback to the mock:
    a typo in `IDENTITY_PROVIDER` must not quietly hand a production
    deployment a fictitious directory.
    """
    name = (settings.identity_provider or "").strip().lower()

    if name == "mock":
        return MockIdentityProvider(settings.identity_config_path)

    if name == "oidc":
        # Imported here, not at module scope, so `mock` deployments never
        # load the OIDC implementation at all.
        from numi.common.identity.oidc_provider import OIDCIdentityProvider

        if not settings.oidc_issuer:
            raise NumiError(
                FailureCode.DEPENDENCY_UNAVAILABLE,
                "IDENTITY_PROVIDER=oidc but OIDC_ISSUER is not set. Refusing to start "
                "with an identity provider that could never resolve anyone.",
            )
        return OIDCIdentityProvider(
            issuer=settings.oidc_issuer,
            client_id=settings.oidc_client_id,
            client_secret=settings.oidc_client_secret,
            identity_config_path=settings.identity_config_path,
        )

    raise NumiError(
        FailureCode.DEPENDENCY_UNAVAILABLE,
        f"Unknown identity provider '{settings.identity_provider}'. "
        "Expected one of: mock | oidc.",
    )
