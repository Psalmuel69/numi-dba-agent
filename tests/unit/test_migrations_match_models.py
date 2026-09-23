"""Guards the exact bug fixed by migration 0002 (spec §58): a model added to
`gateway/infrastructure/db/models.py` without a matching Alembic migration
silently never gets its table on a database that was already migrated
before the model existed (`alembic upgrade head` is then a no-op, since the
revision that would have created it is already stamped applied). This ran
alembic against a throwaway database from scratch and diffs the resulting
tables against `Base.metadata`, so that drift fails here — in CI, on every
push — instead of surfacing later as a live 500 on someone else's database.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect

from numi.gateway.infrastructure.db.models import Base

REPO_ROOT = Path(__file__).resolve().parents[2]

# The exact tables migration 0001 built via `Base.metadata.create_all()` at
# the time it was authored. 0001 is a live snapshot of *whatever* Base.metadata
# is when it runs — so on a database migrated fresh today, it would also
# create any table added to models.py *after* 0001 was written, silently
# masking a missing migration for it (this is exactly why the fresh-DB test
# above couldn't have caught the bug 0002 fixes). Don't add a new table name
# here: give it its own explicit `op.create_table(...)` in a new revision
# instead, and it'll be picked up automatically by the check below.
TABLES_COVERED_BY_0001 = {
    "agent_sessions", "approval_events", "approvals", "audit_events",
    "change_requests", "conversations", "identity_groups", "incidents",
    "investigation_events", "investigations", "knowledge_documents",
    "role_bindings", "roles", "security_events", "tool_executions",
    "tool_requests", "tool_versions", "tools", "users",
}


def test_alembic_head_creates_a_table_for_every_current_model(tmp_path):
    db_path = tmp_path / "migration_check.db"
    env = dict(os.environ, CONTROL_DB_URL=f"sqlite+aiosqlite:///{db_path}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    migrated_tables = set(inspect(create_engine(f"sqlite:///{db_path}")).get_table_names())
    migrated_tables.discard("alembic_version")
    model_tables = set(Base.metadata.tables.keys())

    missing = model_tables - migrated_tables
    assert not missing, (
        f"{sorted(missing)} are defined in models.py but no migration creates them — "
        "a database migrated before this model existed will never get the table. "
        "Add an explicit `op.create_table(...)` for it in a new migration revision."
    )


def test_every_table_added_after_0001_has_an_explicit_migration():
    """0001's `create_all()` is live-metadata-driven, so the fresh-DB test
    above would never fail even if a new model's table is only ever created
    by accident (i.e. only because 0001 happens to run against today's
    models.py) rather than by a real, version-controlled migration for it.
    This is a static check instead: any table not already covered by 0001
    must be named in an `op.create_table(...)` call somewhere under
    migrations/versions/ — proving it has its own migration, not a free
    ride from 0001's dynamic snapshot."""
    model_tables = set(Base.metadata.tables.keys())
    new_tables = model_tables - TABLES_COVERED_BY_0001

    versions_dir = REPO_ROOT / "migrations" / "versions"
    migration_source = "\n".join(
        f.read_text(encoding="utf-8")
        for f in versions_dir.glob("*.py")
        if f.name != "__init__.py"
    )
    create_table_calls = set(re.findall(r'create_table\(\s*"([^"]+)"', migration_source))

    unmigrated = new_tables - create_table_calls
    assert not unmigrated, (
        f"{sorted(unmigrated)} were added to models.py after 0001 but no migration "
        "file has an explicit `op.create_table(...)` for them — add one in a new "
        "revision (see migrations/versions/0002_server_catalogs.py for the pattern)."
    )


def test_alembic_head_adds_investigations_server_id_column(tmp_path):
    """The two checks above only ever look at table presence, not columns —
    neither would have caught a migration that creates a table but forgets
    a column added to its model afterward (exactly migration 0003's shape:
    `investigations` already existed via 0001, `server_id` was added to the
    model later and needs its own explicit `op.add_column`)."""
    db_path = tmp_path / "migration_check_server_id.db"
    env = dict(os.environ, CONTROL_DB_URL=f"sqlite+aiosqlite:///{db_path}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    columns = {c["name"] for c in inspect(create_engine(f"sqlite:///{db_path}")).get_columns("investigations")}
    assert "server_id" in columns


def test_alembic_head_adds_investigations_correlation_columns(tmp_path):
    """Same gap as the test above, for migration 0005's two columns."""
    db_path = tmp_path / "migration_check_correlation_columns.db"
    env = dict(os.environ, CONTROL_DB_URL=f"sqlite+aiosqlite:///{db_path}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    columns = {c["name"] for c in inspect(create_engine(f"sqlite:///{db_path}")).get_columns("investigations")}
    assert {"playbook_id", "environment"} <= columns
