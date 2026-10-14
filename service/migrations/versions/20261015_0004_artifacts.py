"""artifacts (Studio outputs: briefing doc, study guide, FAQ, timeline, mind map, flashcards, quiz, audio)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-15 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("document_ids", sa.JSON(), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column("instructions", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_artifacts_collection_id"), "artifacts", ["collection_id"], unique=False)
    op.create_index(op.f("ix_artifacts_user_id"), "artifacts", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_artifacts_user_id"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_collection_id"), table_name="artifacts")
    op.drop_table("artifacts")
