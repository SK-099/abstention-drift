"""documents: web / YouTube / text sources and source guides

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as b:
        b.add_column(sa.Column("source_type", sa.String(length=20), nullable=False, server_default="file"))
        b.add_column(sa.Column("source_url", sa.String(length=2000), nullable=True))
        b.add_column(sa.Column("guide", sa.JSON(), nullable=True))
        b.add_column(sa.Column("guide_status", sa.String(length=20), nullable=False, server_default="none"))


def downgrade() -> None:
    with op.batch_alter_table("documents") as b:
        b.drop_column("guide_status")
        b.drop_column("guide")
        b.drop_column("source_url")
        b.drop_column("source_type")
