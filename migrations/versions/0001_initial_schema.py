"""initial control-plane schema

Revision ID: 0001
Revises:
Create Date: 2026-01-01

This revision creates every control-plane table directly from the
SQLAlchemy ORM metadata (`numi.gateway.infrastructure.db.models.Base`)
rather than hand-written `op.create_table` calls, since the models module
is itself the single source of truth for the schema (spec §58). Subsequent
schema changes should add incremental revisions using
`op.create_table` / `op.add_column` as normal.
"""

from __future__ import annotations

from alembic import op

from numi.gateway.infrastructure.db.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
