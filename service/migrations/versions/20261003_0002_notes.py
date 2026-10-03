"""notes (notebook notes and saved answers)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("source_message_id", sa.String(length=36), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_notes_collection_id"), "notes", ["collection_id"], unique=False)
    op.create_index(op.f("ix_notes_user_id"), "notes", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_notes_user_id"), table_name="notes")
    op.drop_index(op.f("ix_notes_collection_id"), table_name="notes")
    op.drop_table("notes")
