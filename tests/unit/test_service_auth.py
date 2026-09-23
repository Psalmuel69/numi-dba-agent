"""Service-to-service token issue/verify (spec §31). No live HTTP here —
this is the primitive every internal call (Agent -> Gateway, Gateway ->
Execution Service) trusts to prove which service is calling."""

from __future__ import annotations

import pytest

from numi.common.models.failures import FailureCode, NumiError
from numi.common.service_auth import ServiceTokenIssuer, ServiceTokenVerifier


def test_issue_then_verify_round_trips_the_service_identity():
    issuer = ServiceTokenIssuer("shared-secret", "numi-internal")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal")

    token = issuer.issue(service_name="agent", audience="numi-gateway")
    identity = verifier.verify(token, expected_audience="numi-gateway")

    assert identity.service_name == "agent"
    assert identity.audience == "numi-gateway"


def test_expired_token_is_rejected(monkeypatch):
    issuer = ServiceTokenIssuer("shared-secret", "numi-internal")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal", max_age_seconds=60)

    now = [1_000_000.0]
    monkeypatch.setattr("itsdangerous.timed.time.time", lambda: now[0])
    token = issuer.issue(service_name="agent", audience="numi-gateway")

    now[0] += 61
    with pytest.raises(NumiError) as exc_info:
        verifier.verify(token, expected_audience="numi-gateway")
    assert exc_info.value.code == FailureCode.AUTHENTICATION_FAILED
    assert "expired" in str(exc_info.value)


def test_a_token_still_within_its_max_age_is_accepted(monkeypatch):
    issuer = ServiceTokenIssuer("shared-secret", "numi-internal")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal", max_age_seconds=60)

    now = [1_000_000.0]
    monkeypatch.setattr("itsdangerous.timed.time.time", lambda: now[0])
    token = issuer.issue(service_name="agent", audience="numi-gateway")

    now[0] += 59
    identity = verifier.verify(token, expected_audience="numi-gateway")
    assert identity.service_name == "agent"


def test_a_tampered_token_is_rejected():
    issuer = ServiceTokenIssuer("shared-secret", "numi-internal")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal")

    token = issuer.issue(service_name="agent", audience="numi-gateway")
    # Flip a character well inside the string, not the last one: an
    # unpadded base64url string's trailing character can carry unused
    # padding bits, so mutating it doesn't always change the decoded bytes
    # (verified live: passed locally, failed in CI on a token whose length
    # happened to land in that insensitive case).
    middle = len(token) // 2
    replacement = "A" if token[middle] != "A" else "B"
    tampered = token[:middle] + replacement + token[middle + 1 :]
    assert tampered != token

    with pytest.raises(NumiError) as exc_info:
        verifier.verify(tampered, expected_audience="numi-gateway")
    assert exc_info.value.code == FailureCode.AUTHENTICATION_FAILED
    assert "signature is invalid" in str(exc_info.value)


def test_a_token_signed_with_a_different_secret_is_rejected():
    issuer = ServiceTokenIssuer("attacker-secret", "numi-internal")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal")

    token = issuer.issue(service_name="agent", audience="numi-gateway")

    with pytest.raises(NumiError) as exc_info:
        verifier.verify(token, expected_audience="numi-gateway")
    assert exc_info.value.code == FailureCode.AUTHENTICATION_FAILED


def test_a_token_issued_for_a_different_audience_is_rejected():
    """The audience is folded into the signing salt (see the module
    docstring), so a token minted for one audience fails signature
    verification outright against another — it never reaches the payload's
    own `aud` field. Same outcome either way: the caller only sees
    AUTHENTICATION_FAILED, never which check tripped."""
    issuer = ServiceTokenIssuer("shared-secret", "numi-internal")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal")

    token = issuer.issue(service_name="agent", audience="numi-gateway")

    with pytest.raises(NumiError) as exc_info:
        verifier.verify(token, expected_audience="numi-execution")
    assert exc_info.value.code == FailureCode.AUTHENTICATION_FAILED


def test_a_token_from_a_different_issuer_is_rejected_despite_a_valid_signature():
    """The one case a shared-secret + shared-audience attacker could still
    pull off: minting a structurally valid, correctly-signed token that
    just claims a different `iss`. The signature alone can't catch this —
    it only proves *a* holder of the secret signed it, not which one — so
    `verify()` must check `iss` itself against a token that otherwise
    verifies cleanly."""
    forged_issuer = ServiceTokenIssuer("shared-secret", "some-other-system")
    verifier = ServiceTokenVerifier("shared-secret", "numi-internal")

    token = forged_issuer.issue(service_name="agent", audience="numi-gateway")

    with pytest.raises(NumiError) as exc_info:
        verifier.verify(token, expected_audience="numi-gateway")
    assert exc_info.value.code == FailureCode.AUTHENTICATION_FAILED
    assert "issuer/audience mismatch" in str(exc_info.value)
