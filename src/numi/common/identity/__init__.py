from numi.common.identity.factory import build_identity_provider
from numi.common.identity.provider import (
    GroupRoleMapping,
    IdentityProvider,
    MockIdentityProvider,
)

# `OIDCIdentityProvider` is deliberately NOT re-exported here: it is
# imported lazily by `build_identity_provider` so a `mock` deployment never
# loads it. Import it from `numi.common.identity.oidc_provider` directly.
__all__ = [
    "GroupRoleMapping",
    "IdentityProvider",
    "MockIdentityProvider",
    "build_identity_provider",
]
