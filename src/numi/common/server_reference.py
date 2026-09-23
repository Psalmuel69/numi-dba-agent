"""Normalizing a free-text server reference (spec: the DBA should never be
bound to a fixed message structure).

Real DBAs don't reliably use a server's exact registered id — "sql server
dev 1" and "sqlserver-dev-01" are the same server to a human, differing
only in spacing, punctuation, and how the trailing number is padded.
`ServerRegistry.find_candidates` (the Gateway's own, independently
re-resolving matcher) and `AgentOrchestrator._environment_for_instance`
(the Agent-side convenience that avoids a redundant "which environment"
question) both need the exact same normalization, so it lives here, in
`common/`, rather than being duplicated or drifting between the two.

This only ever *widens* what counts as a candidate match, on top of the
existing exact/substring comparison against the raw string — it never
narrows or replaces it, and it never resolves an ambiguous or absent match
on its own: a hint that (after normalizing) matches more than one
registered server still surfaces every match for the caller to ask about,
exactly like an ambiguous id/alias/host already does, and stays a
`LookupError` when nothing matches at all. The normalization can only ever
cause the same "ask, don't guess" fallback to fire slightly more or less
often — it can never cause a *wrong* confident match, since a false
merge just becomes an ambiguous one instead of a clean single match.
"""

from __future__ import annotations

import re

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIGIT_RUN_RE = re.compile(r"\d+")


def normalize_server_reference(text: str) -> str:
    """Lowercase, strip every separator (space/hyphen/underscore/dot/...),
    and collapse each run of digits to its plain integer value (no leading
    zeros) — so "SQL Server Dev 1", "sqlserver-dev-01", and
    "sqlserver_dev_1" all normalize to the identical "sqlserverdev1".

    Deliberately NOT applied to an IP address/host comparison — collapsing
    "192.168.0.100"'s separators would merge its octets into a single
    number and make fragment matching (e.g. "0.100") nonsensical; IP/host
    matching stays on the raw, dot-preserved string.
    """
    text = _NON_ALNUM_RE.sub("", text.lower())
    return _DIGIT_RUN_RE.sub(lambda m: str(int(m.group())), text)
