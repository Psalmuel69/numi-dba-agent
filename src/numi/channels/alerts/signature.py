"""Alert webhook request signature verification.

An external monitoring system (Prometheus Alertmanager, Datadog, a cloud
provider's own alarms, ...) is not one of our own services and carries none
of the internal service-to-service trust `common.service_auth` provides, so
its webhook needs its own signature scheme — the same shape as
`channels.slack.signature`'s (an HMAC over a timestamp-bound body, with a
freshness check that defeats replaying a captured request), generalized
past Slack's specific header names and `v0:` prefix since no external
sender here shares Slack's exact convention.

This is the first gate on every inbound alert request — a request that
fails this check is never parsed for a server name, a metric, or anything
else; it is rejected outright, since nothing about it is trustworthy yet.
"""

from __future__ import annotations

import hashlib
import hmac
import time


class AlertSignatureError(Exception):
    pass


def verify_alert_signature(
    *,
    secret: str,
    request_body: bytes,
    timestamp_header: str | None,
    signature_header: str | None,
    max_clock_skew_seconds: int = 60 * 5,
    now: float | None = None,
) -> None:
    """Raises `AlertSignatureError` if the request cannot be verified.

    `secret` empty is always a failure, never a silent pass-through — an
    unconfigured webhook must reject every request, not accept them as
    trusted-by-default. Checks both the HMAC signature *and* timestamp
    freshness, the latter being what defeats a replay of a previously-valid,
    captured request.

    `signature_header` is expected as `sha256=<hex>`, and `timestamp_header`
    as a Unix timestamp (seconds) — send both `X-Numi-Alert-Timestamp` and
    `X-Numi-Alert-Signature: sha256=<hmac>` computed over
    `f"{timestamp}.{body}"` with the shared secret.
    """
    if not secret:
        raise AlertSignatureError("Alert webhook is not configured (no secret set).")
    if not timestamp_header or not signature_header:
        raise AlertSignatureError("Missing alert signature headers.")

    try:
        ts = int(timestamp_header)
    except ValueError as exc:
        raise AlertSignatureError("Invalid timestamp header.") from exc

    now = now if now is not None else time.time()
    if abs(now - ts) > max_clock_skew_seconds:
        raise AlertSignatureError("Request timestamp is too old or too far in the future.")

    base = f"{timestamp_header}.".encode() + request_body
    computed = "sha256=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed, signature_header):
        raise AlertSignatureError("Signature mismatch.")
