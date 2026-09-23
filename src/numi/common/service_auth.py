"""Service-to-service authentication (spec §31).

Every hop between our own services (Agent -> Gateway, Gateway -> Execution
Service) presents a signed, short-lived, audience-scoped token — never a
bare header like `X-Internal-Request: true`. Tokens are HMAC-signed and
time-limited using `itsdangerous`; production deployments can swap this for
mTLS or OIDC client-credentials without changing the call sites, since both
ends only ever see `issue()` / `verify()`.
"""

from __future__ import annotations

from dataclasses import dataclass

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from numi.common.models.failures import FailureCode, NumiError


@dataclass(frozen=True)
class ServiceIdentity:
    service_name: str
    audience: str


class ServiceTokenIssuer:
    def __init__(self, secret: str, issuer: str):
        self._secret = secret
        self._issuer = issuer

    def issue(self, *, service_name: str, audience: str) -> str:
        serializer = URLSafeTimedSerializer(self._secret, salt=audience)
        return serializer.dumps({"iss": self._issuer, "sub": service_name, "aud": audience})


class ServiceTokenVerifier:
    def __init__(self, secret: str, issuer: str, max_age_seconds: int = 60):
        self._secret = secret
        self._issuer = issuer
        self._max_age = max_age_seconds

    def verify(self, token: str, *, expected_audience: str) -> ServiceIdentity:
        serializer = URLSafeTimedSerializer(self._secret, salt=expected_audience)
        try:
            payload = serializer.loads(token, max_age=self._max_age)
        except SignatureExpired as exc:
            raise NumiError(
                FailureCode.AUTHENTICATION_FAILED, "Service token has expired."
            ) from exc
        except BadSignature as exc:
            raise NumiError(
                FailureCode.AUTHENTICATION_FAILED, "Service token signature is invalid."
            ) from exc

        if payload.get("iss") != self._issuer or payload.get("aud") != expected_audience:
            raise NumiError(
                FailureCode.AUTHENTICATION_FAILED, "Service token issuer/audience mismatch."
            )
        return ServiceIdentity(service_name=payload["sub"], audience=payload["aud"])
