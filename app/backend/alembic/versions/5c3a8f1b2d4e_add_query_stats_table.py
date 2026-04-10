"""Add query_stats table

Revision ID: 5c3a8f1b2d4e
Revises: 4b2c91d53f7e
Create Date: 2026-03-03 10:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '5c3a8f1b2d4e'
down_revision: str | Sequence[str] | None = '4b2c91d53f7e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the query_stats table for per-step cost/latency tracking."""
    op.create_table(
        'query_stats',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('conversation_id', sa.Integer(),
                  sa.ForeignKey('conversations.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('query_iteration', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('step_type', sa.String(), nullable=False),
        sa.Column('message_id', sa.Integer(),
                  sa.ForeignKey('messages.id', ondelete='SET NULL'),
                  nullable=True),

        # Per-step metrics
        sa.Column('cost_usd', sa.Float(), nullable=True),
        sa.Column('wall_clock_secs', sa.Float(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),

        # Full stats JSON blob
        sa.Column('stats_json', JSONB(), nullable=True),

        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('ix_query_stats_id', 'query_stats', ['id'])
    op.create_index('ix_query_stats_session_id', 'query_stats', ['session_id'])
    op.create_index('ix_query_stats_conversation_id', 'query_stats', ['conversation_id'])
    op.create_index('ix_query_stats_message_id', 'query_stats', ['message_id'])


def downgrade() -> None:
    """Drop the query_stats table."""
    op.drop_index('ix_query_stats_message_id', table_name='query_stats')
    op.drop_index('ix_query_stats_conversation_id', table_name='query_stats')
    op.drop_index('ix_query_stats_session_id', table_name='query_stats')
    op.drop_index('ix_query_stats_id', table_name='query_stats')
    op.drop_table('query_stats')
