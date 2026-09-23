"""Slack request signature verification (spec §4, §43).

Implements Slack's documented signing-secret HMAC scheme. This is the first
gate on every inbound Slack request — a request that fails this check is
never parsed for a user id, a message, or anything else; it is rejected
outright, since nothing about it is trustworthy yet.
"""

from __future__ import annotations

import hashlib
import hmac
import time


class SlackSignatureError(Exception):
    pass


def verify_slack_signature(
    *,
    signing_secret: str,
    request_body: bytes,
    timestamp_header: str | None,
    signature_header: str | None,
    max_clock_skew_seconds: int = 60 * 5,
    now: float | None = None,
) -> None:
    """Raises `SlackSignatureError` if the request cannot be verified.

    Checks both the HMAC signature *and* timestamp freshness — the latter is
    what defeats a replay of a previously-valid, captured request (spec §43's
    replay-attack test).
    """
    if not timestamp_header or not signature_header:
        raise SlackSignatureError("Missing Slack signature headers.")

    try:
        ts = int(timestamp_header)
    except ValueError as exc:
        raise SlackSignatureError("Invalid timestamp header.") from exc

    now = now if now is not None else time.time()
    if abs(now - ts) > max_clock_skew_seconds:
        raise SlackSignatureError("Request timestamp is too old or too far in the future.")

    base = f"v0:{timestamp_header}:".encode() + request_body
    computed = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed, signature_header):
        raise SlackSignatureError("Signature mismatch.")
