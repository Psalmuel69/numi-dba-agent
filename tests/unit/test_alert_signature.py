"""Alert webhook signature verification (channels.alerts.signature)."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from numi.channels.alerts.signature import AlertSignatureError, verify_alert_signature

SECRET = "a-real-webhook-secret"
BODY = b'{"server": "winpg", "metric": "replication_lag_seconds"}'


def _sign(secret: str, timestamp: str, body: bytes) -> str:
    base = f"{timestamp}.".encode() + body
    return "sha256=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()


def test_valid_signature_and_fresh_timestamp_is_accepted():
    ts = "1700000000"
    verify_alert_signature(
        secret=SECRET,
        request_body=BODY,
        timestamp_header=ts,
        signature_header=_sign(SECRET, ts, BODY),
        now=1700000000.0,
    )  # should not raise


def test_empty_secret_is_always_rejected_never_a_silent_pass():
    ts = "1700000000"
    with pytest.raises(AlertSignatureError, match="not configured"):
        verify_alert_signature(
            secret="",
            request_body=BODY,
            timestamp_header=ts,
            signature_header=_sign("anything", ts, BODY),
            now=1700000000.0,
        )


def test_missing_headers_are_rejected():
    with pytest.raises(AlertSignatureError, match="Missing"):
        verify_alert_signature(
            secret=SECRET, request_body=BODY, timestamp_header=None, signature_header=None
        )


def test_wrong_secret_is_rejected():
    ts = "1700000000"
    with pytest.raises(AlertSignatureError, match="mismatch"):
        verify_alert_signature(
            secret=SECRET,
            request_body=BODY,
            timestamp_header=ts,
            signature_header=_sign("a-different-secret", ts, BODY),
            now=1700000000.0,
        )


def test_tampered_body_is_rejected():
    ts = "1700000000"
    signature = _sign(SECRET, ts, BODY)
    with pytest.raises(AlertSignatureError, match="mismatch"):
        verify_alert_signature(
            secret=SECRET,
            request_body=BODY + b"tampered",
            timestamp_header=ts,
            signature_header=signature,
            now=1700000000.0,
        )


def test_stale_timestamp_is_rejected_as_a_replay():
    ts = "1700000000"
    signature = _sign(SECRET, ts, BODY)
    with pytest.raises(AlertSignatureError, match="too old"):
        verify_alert_signature(
            secret=SECRET,
            request_body=BODY,
            timestamp_header=ts,
            signature_header=signature,
            now=1700000000.0 + 600,  # 10 minutes later, past the 5-minute default skew
        )


def test_non_numeric_timestamp_is_rejected():
    with pytest.raises(AlertSignatureError, match="Invalid timestamp"):
        verify_alert_signature(
            secret=SECRET,
            request_body=BODY,
            timestamp_header="not-a-number",
            signature_header="sha256=irrelevant",
        )
