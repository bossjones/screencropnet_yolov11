"""initial classification_jobs table

Revision ID: 0001_init
Revises:
Create Date: 2026-06-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_init"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "classification_jobs",
        sa.Column("job_id", sa.String(), primary_key=True),
        sa.Column("batch_id", sa.String(), nullable=False),
        sa.Column("original_path", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "processing",
                "done",
                "failed",
                name="jobstatus",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("is_twitter", sa.Boolean(), nullable=True),
        sa.Column("pred_class", sa.String(), nullable=True),
        sa.Column("pred_prob", sa.Float(), nullable=True),
        sa.Column("time_for_pred", sa.Float(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_classification_jobs_batch_id", "classification_jobs", ["batch_id"])
    op.create_index("ix_classification_jobs_status", "classification_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_classification_jobs_status", table_name="classification_jobs")
    op.drop_index("ix_classification_jobs_batch_id", table_name="classification_jobs")
    op.drop_table("classification_jobs")
