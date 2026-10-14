"""notebook sharing: members (viewer / editor) and link access

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15 14:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("collections") as b:
        b.add_column(sa.Column("link_access", sa.String(length=10), nullable=False, server_default="none"))
    op.create_table(
        "collection_members",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("principal", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=10), nullable=False),
        sa.Column("added_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["added_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "principal"),
    )
    op.create_index(op.f("ix_collection_members_collection_id"), "collection_members", ["collection_id"])
    op.create_index(op.f("ix_collection_members_principal"), "collection_members", ["principal"])


def downgrade() -> None:
    op.drop_index(op.f("ix_collection_members_principal"), table_name="collection_members")
    op.drop_index(op.f("ix_collection_members_collection_id"), table_name="collection_members")
    op.drop_table("collection_members")
    with op.batch_alter_table("collections") as b:
        b.drop_column("link_access")
