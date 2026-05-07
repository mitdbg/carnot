"""Add upload_jobs table for async file upload tracking

Revision ID: b2c3d4e5f6a7
Revises: 9a2b4d6e8f10
Create Date: 2026-05-07 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: str | Sequence[str] | None = '9a2b4d6e8f10'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'upload_jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('total_files', sa.Integer(), nullable=True),
        sa.Column('processed_files', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_upload_jobs_user_id', 'upload_jobs', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_upload_jobs_user_id', table_name='upload_jobs')
    op.drop_table('upload_jobs')
