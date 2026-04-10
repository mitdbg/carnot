"""Add workspaces and notebooks tables

Revision ID: 6d4a9e2f3b1c
Revises: 5c3a8f1b2d4e
Create Date: 2025-07-11 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = '6d4a9e2f3b1c'
down_revision: str | Sequence[str] | None = '5c3a8f1b2d4e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create workspaces table, notebooks table, backfill, and wire FKs."""

    # 1. Create workspaces table
    op.create_table(
        'workspaces',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False,
                  server_default='Untitled Workspace'),
        sa.Column('dataset_ids', sa.String(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('ix_workspaces_id', 'workspaces', ['id'])
    op.create_index('ix_workspaces_user_id', 'workspaces', ['user_id'])
    op.create_index('ix_workspaces_session_id', 'workspaces', ['session_id'], unique=True)

    # 2. Add workspace_id column to conversations (nullable initially for backfill)
    op.add_column('conversations',
                  sa.Column('workspace_id', sa.Integer(), nullable=True))

    # 3. Backfill: create one workspace per existing conversation,
    #    copying user_id, session_id, title, and dataset_ids.
    op.execute("""
        INSERT INTO workspaces (user_id, session_id, title, dataset_ids,
                                created_at, updated_at)
        SELECT user_id, session_id,
               COALESCE(title, 'Untitled Workspace'),
               dataset_ids,
               created_at, updated_at
        FROM conversations
    """)

    # 4. Link each conversation to its workspace via session_id match
    op.execute("""
        UPDATE conversations c
        SET workspace_id = w.id
        FROM workspaces w
        WHERE c.session_id = w.session_id
    """)

    # 5. Make workspace_id NOT NULL and add the FK constraint
    op.alter_column('conversations', 'workspace_id', nullable=False)
    op.create_foreign_key(
        'fk_conversations_workspace_id', 'conversations',
        'workspaces', ['workspace_id'], ['id'],
        ondelete='CASCADE',
    )

    # 6. Create notebooks table
    op.create_table(
        'notebooks',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('workspace_id', sa.Integer(),
                  sa.ForeignKey('workspaces.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('conversation_id', sa.Integer(),
                  sa.ForeignKey('conversations.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('notebook_uuid', sa.String(), unique=True, nullable=False),
        sa.Column('label', sa.String(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('plan_json', JSONB(), nullable=True),
        sa.Column('cells_json', JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index('ix_notebooks_id', 'notebooks', ['id'])

    # 7. Add notebook_id FK to query_stats
    op.add_column('query_stats',
                  sa.Column('notebook_id', sa.Integer(),
                            sa.ForeignKey('notebooks.id', ondelete='SET NULL'),
                            nullable=True))

    # 8. Drop dataset_ids from conversations (now lives on workspaces)
    op.drop_column('conversations', 'dataset_ids')


def downgrade() -> None:
    """Reverse: restore dataset_ids on conversations, drop notebooks + workspaces."""

    # Restore dataset_ids column on conversations
    op.add_column('conversations',
                  sa.Column('dataset_ids', sa.String(), nullable=True))

    # Backfill dataset_ids from workspace back to conversation
    op.execute("""
        UPDATE conversations c
        SET dataset_ids = w.dataset_ids
        FROM workspaces w
        WHERE c.workspace_id = w.id
    """)

    # Drop notebook_id from query_stats
    op.drop_column('query_stats', 'notebook_id')

    # Drop notebooks table
    op.drop_index('ix_notebooks_id', table_name='notebooks')
    op.drop_table('notebooks')

    # Remove workspace_id FK and column from conversations
    op.drop_constraint('fk_conversations_workspace_id', 'conversations',
                       type_='foreignkey')
    op.drop_column('conversations', 'workspace_id')

    # Drop workspaces table
    op.drop_index('ix_workspaces_session_id', table_name='workspaces')
    op.drop_index('ix_workspaces_user_id', table_name='workspaces')
    op.drop_index('ix_workspaces_id', table_name='workspaces')
    op.drop_table('workspaces')
