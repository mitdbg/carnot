"""Add cost_budget column to messages table

Revision ID: 4b2c91d53f7e
Revises: 3a0279ec6f19
Create Date: 2026-02-21 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '4b2c91d53f7e'
down_revision: str | Sequence[str] | None = '3a0279ec6f19'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add cost_budget column to messages table to track the maximum dollar
    # amount the user was willing to spend when submitting this query
    op.add_column('messages', sa.Column('cost_budget', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('messages', 'cost_budget')
