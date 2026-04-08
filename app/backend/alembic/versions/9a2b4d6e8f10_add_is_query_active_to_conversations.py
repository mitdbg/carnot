"""Add is_query_active column to conversations table

Revision ID: 9a2b4d6e8f10
Revises: 8f7e3c1a4b2d
Create Date: 2026-03-20 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9a2b4d6e8f10'
down_revision: str | Sequence[str] | None = '8f7e3c1a4b2d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add is_query_active boolean column to conversations."""
    op.add_column(
        'conversations',
        sa.Column('is_query_active', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )


def downgrade() -> None:
    """Remove is_query_active column from conversations."""
    op.drop_column('conversations', 'is_query_active')
