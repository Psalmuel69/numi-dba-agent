"""ULID-based identifier generation.

ULIDs are used (rather than plain UUIDv4) because they are lexicographically
sortable by creation time, which makes audit-log and investigation ordering
possible without a separate `created_at` sort in the common case, while still
being globally unique.
"""

from __future__ import annotations

from ulid import ULID


def new_id(prefix: str) -> str:
    """Return a prefixed, sortable, globally unique id, e.g. `req_01J...`."""
    return f"{prefix}_{ULID()!s}"
