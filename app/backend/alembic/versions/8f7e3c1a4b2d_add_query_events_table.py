"""Add query_events table for real-time SSE event persistence

Revision ID: 8f7e3c1a4b2d
Revises: 6d4a9e2f3b1c
Create Date: 2026-03-15 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '8f7e3c1a4b2d'
down_revision: str | Sequence[str] | None = '6d4a9e2f3b1c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the query_events table for per-event cost persistence."""
    op.create_table(
        'query_events',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('conversation_id', sa.Integer(),
                  sa.ForeignKey('conversations.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('payload', JSONB(), nullable=False),
        sa.Column('step_cost_usd', sa.Float(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('ix_query_events_id', 'query_events', ['id'])
    op.create_index('ix_query_events_session_id', 'query_events', ['session_id'])
    op.create_index('ix_query_events_conv_id_created',
                    'query_events', ['conversation_id', 'created_at'])


def downgrade() -> None:
    """Drop the query_events table."""
    op.drop_index('ix_query_events_conv_id_created', table_name='query_events')
    op.drop_index('ix_query_events_session_id', table_name='query_events')
    op.drop_index('ix_query_events_id', table_name='query_events')
    op.drop_table('query_events')
