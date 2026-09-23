"""Discovered catalog models — what Numi has learned about each registered
server.

Shared between the Execution Service (discovery crawler, which produces
these) and the Gateway (which caches and serves them). Everything here is
DBA *metadata*: database names/states/sizes, schema and object names, index
stats, available extensions, server/instance properties. **Never table or
view row data.**

Object names and comments coming from a database are untrusted strings —
treated as data, never instructions (spec §23).
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class DiscoveredObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_name: str
    name: str
    kind: str  # table | view | index | procedure | function | sequence
    row_estimate: int | None = None
    size_bytes: int | None = None
    properties: dict = Field(default_factory=dict)


class DiscoveredExtension(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    installed_version: str | None = None
    default_version: str | None = None
    available: bool = True


class DiscoveredDatabase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    state: str = "unknown"
    size_bytes: int | None = None
    options: dict = Field(default_factory=dict)
    objects: list[DiscoveredObject] = Field(default_factory=list)
    extensions: list[DiscoveredExtension] = Field(default_factory=list)


class LeastPrivilegeFinding(BaseModel):
    """Whether the diagnostic login can actually read user data.

    `discovery/base.py`'s own module docstring has always stated the
    architectural premise — "The diagnostic login ... should NOT have SELECT
    on user tables; Numi never reads table contents" — but nothing ever
    checked it. That made it a statement of intent, not a control: a login
    provisioned with `db_datareader`, or a Postgres superuser, or a MySQL
    account with a stray `GRANT SELECT ON *.*`, would work perfectly and
    silently carry far more authority than the design calls for.

    This is the *observed* answer, produced once per discovery run by each
    engine's `ServerDiscoverer._check_least_privilege`. It is READ-ONLY
    introspection of the engine's own privilege views — Numi reports, a
    human DBA revokes. Nothing in this codebase attempts a REVOKE.

    A positive finding (`has_user_table_select=True`) is conclusive. A
    negative one is scoped: see `scope_note` and
    `ServerDiscoverer._check_least_privilege`'s own docstring — on
    PostgreSQL and SQL Server, privilege visibility is per-database, so the
    check covers the database the discovery connection is bound to, not
    necessarily every database on the instance. MySQL's privilege views are
    genuinely instance-wide, so there the negative is instance-wide too.
    """

    model_config = ConfigDict(extra="ignore")

    #: False when the check could not run at all (engine refused the
    #: privilege view, connection died mid-crawl). Distinct from
    #: "ran and found nothing" — never report a clean bill of health for a
    #: check that never happened.
    checked: bool = False
    login: str = ""
    has_user_table_select: bool = False
    granted_object_count: int = 0
    #: True when the scan hit its own row cap, so the count is a floor.
    count_is_lower_bound: bool = False
    #: A few "schema.object" names, purely so the DBA knows where to look.
    #: Names only — never any row data from those objects.
    sample_objects: list[str] = Field(default_factory=list)
    scope_note: str = ""

    def warning_text(self) -> str | None:
        """The DBA-facing one-liner, or `None` when there is nothing to say.

        Rendered by `agent/orchestrator.py::_handle_catalog_command` and
        logged by `gateway/domain/discovery.py::refresh_server`. Built here
        so both surfaces say the same thing rather than deriving their own
        wording (the same single-source-of-truth reasoning
        `_verification_note` follows for verification outcomes).
        """
        if not self.checked or not self.has_user_table_select:
            return None
        approx = "at least " if self.count_is_lower_bound else ""
        plural = "" if self.granted_object_count == 1 else "s"
        detail = f" (e.g. {', '.join(self.sample_objects[:3])})" if self.sample_objects else ""
        return (
            f"⚠️ This server's diagnostic login ({self.login or 'unknown'}) has "
            f"SELECT on {approx}{self.granted_object_count} user table/view{plural}{detail} "
            f"— should be revoked for least-privilege; Numi never needs to read table data."
        )


class ServerCatalog(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server_id: str
    discovered_at: dt.datetime | None = None
    engine_version: str = ""
    engine_edition: str = ""
    instance_properties: dict = Field(default_factory=dict)
    databases: list[DiscoveredDatabase] = Field(default_factory=list)
    # Non-fatal problems hit during the crawl (a db the login couldn't read).
    warnings: list[str] = Field(default_factory=list)
    # Deliberately NOT folded into `warnings` above: that list is free text
    # about what this one crawl couldn't read, whereas this is a structured,
    # durable security finding about the login itself, which the renderer
    # needs the count/sample/login from. Optional so a catalog produced
    # before this check existed (or by an engine that can't run it)
    # round-trips unchanged through the catalog store.
    least_privilege: LeastPrivilegeFinding | None = None

    def database(self, name: str) -> DiscoveredDatabase | None:
        lowered = name.strip().lower()
        return next((d for d in self.databases if d.name.lower() == lowered), None)

    def database_names(self) -> list[str]:
        return [d.name for d in self.databases]
