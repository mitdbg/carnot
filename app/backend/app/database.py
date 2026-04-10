import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

# Re-export catalog ORM models from the carnot library so that existing
# imports like ``from app.database import Dataset`` continue to work.
# The canonical definitions now live in ``carnot.storage.models``.
from carnot.storage.models import DatasetModel as Dataset  # noqa: F401
from carnot.storage.models import IndexEntryModel as IndexEntry  # noqa: F401


# read secrets
def read_secret(secret_name: str) -> str:
    with open(f"/run/secrets/{secret_name}") as secret_file:
        return secret_file.read().strip()

DB_USER = read_secret("db_user")
DB_PASSWORD = read_secret("db_password")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", 5432)

# Database URL
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for app-specific models.
# NOTE: The catalog models (Dataset, IndexEntry) use a separate Base from
# ``carnot.storage.models``.  Alembic's ``env.py`` should import both
# Bases' metadata for ``--autogenerate`` to see all tables.
Base = declarative_base()

# Merge catalog tables (datasets, indices) into the app Base metadata so
# that models defined here (e.g. DatasetFile) can reference them via
# foreign keys at runtime — same merge that Alembic env.py performs.
from carnot.storage.models import Base as _CatalogBase  # noqa: E402

for _table in _CatalogBase.metadata.tables.values():
    if _table.key not in Base.metadata.tables:
        _table.tometadata(Base.metadata)

# ── App-specific database models ────────────────────────────────────────

class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id = Column(String, primary_key=True, unique=True, index=True, nullable=False)
    openai_api_key = Column(String, nullable=True)
    anthropic_api_key = Column(String, nullable=True)
    gemini_api_key = Column(String, nullable=True)
    together_api_key = Column(String, nullable=True)
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    file_path = Column(String, unique=True, nullable=False)
    shared = Column(Boolean, default=False)
    upload_date = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))  # noqa: UP017

class DatasetFile(Base):
    __tablename__ = "dataset_files"

    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False, primary_key=True)
    
    # Add explicit indexes for faster COUNT queries and lookups
    __table_args__ = (
        Index('ix_dataset_files_dataset_id', 'dataset_id'),
        Index('ix_dataset_files_file_id', 'file_id'),
    )


class Workspace(Base):
    """Top-level entity representing a user's research context.

    A workspace contains one or more conversations (strictly one today)
    and zero or more notebooks.  The sidebar lists workspaces.

    Representation invariant:
        - ``session_id`` is unique across all workspaces.
        - ``title`` is never NULL (defaults to ``'Untitled Workspace'``).

    Abstraction function:
        Represents a named research workspace for a user, scoped to a
        set of datasets (``dataset_ids``), containing child conversations
        and notebooks.
    """
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=False, server_default="Untitled Workspace")
    dataset_ids = Column(String, nullable=True)  # Comma-separated dataset IDs
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))  # noqa: UP017
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))  # noqa: UP017


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=True)  # Auto-generated from first query
    is_query_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))  # noqa: UP017
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))  # noqa: UP017

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # 'user', 'agent', 'status', 'error', 'result'
    content = Column(Text, nullable=False)
    type = Column(String, nullable=True)  # Message type: 'natural-language-plan', 'logical-plan', etc.
    csv_file = Column(String, nullable=True)  # For result messages
    row_count = Column(Integer, nullable=True)  # For result messages
    cost_budget = Column(Float, nullable=True)  # Maximum dollar amount user was willing to spend for this query
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))  # noqa: UP017

class QueryStats(Base):
    """Tracks cost, latency, and token usage for a single step (plan or execute).

    Each API call (``POST /plan`` or ``POST /execute``) inserts **one row**.
    Within a conversation the ``query_iteration`` groups plan step(s) and their
    corresponding execution step together, so:

    - ``SELECT * WHERE conversation_id = ? AND query_iteration = ?``
      gives all steps of a single plan→execute cycle.
    - ``SELECT * WHERE conversation_id = ?``
      gives every step across the whole conversation (sum for total cost).

    Representation invariant:
        - ``conversation_id`` references a valid conversation.
        - ``query_iteration`` >= 1 and is monotonically increasing within
          a ``(conversation_id)`` group.
        - ``step_type`` is one of ``'plan'`` or ``'execute'``.
        - ``cost_usd`` is non-null (a row is only created after a step
          finishes).

    Abstraction function:
        Represents the cost/latency/token breakdown for a single step
        (plan **or** execute) within a conversation.
        ``query_iteration`` groups related plan and execute steps, and
        ``step_type`` distinguishes them.
    """

    __tablename__ = "query_stats"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String, nullable=False, index=True)
    query = Column(Text, nullable=True)
    query_iteration = Column(Integer, nullable=False, default=1)
    step_type = Column(String, nullable=False)  # 'plan' or 'execute'
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    notebook_id = Column(Integer, ForeignKey("notebooks.id", ondelete="SET NULL"), nullable=True)

    # Per-step metrics
    cost_usd = Column(Float, nullable=True)
    wall_clock_secs = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)

    # Full stats JSON blob for detailed drilldown
    stats_json = Column(JSONB, nullable=True)

    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))  # noqa: UP017
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))  # noqa: UP017


class Notebook(Base):
    """Lightweight metadata row for a notebook so it survives page refresh.

    The heavy in-memory ``NotebookState`` (plan, datasets_store, cell
    execution) lives in ``query.py``'s ``active_notebooks`` dict and is
    evicted after a timeout.  This DB row stores enough metadata to
    restore the notebook tab and, via ``plan_json``, to re-create the
    in-memory state on demand.

    Representation invariant:
        - ``notebook_uuid`` is unique across all notebooks.
        - ``workspace_id`` references a valid workspace.
        - ``conversation_id`` is NULL or references the conversation
          that spawned this notebook.

    Abstraction function:
        Represents a persisted notebook whose visual state (cell list)
        is captured in ``cells_json`` and whose logical plan is in
        ``plan_json``.
    """
    __tablename__ = "notebooks"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    notebook_uuid = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    plan_json = Column(JSONB, nullable=True)
    cells_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))  # noqa: UP017
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))  # noqa: UP017


class QueryEvent(Base):
    """Append-only log of SSE events emitted during planning and execution.

    Each row captures a single SSE event (``step_detail``, ``result``,
    ``execution_stats``, ``done``, ``error``) together with its per-step
    cost.  Summing ``step_cost_usd`` over a conversation gives the total
    cost for a workspace, enabling the frontend to restore the cost pill
    after a page navigation without replaying the SSE stream.

    Representation invariant:
        - ``conversation_id`` references a valid conversation.
        - ``event_type`` is one of ``"step_detail"``, ``"planning_stats"``,
          ``"execution_stats"``, ``"result"``, ``"error"``, ``"done"``.
        - ``source`` is ``"planning"`` or ``"execution"`` when
          ``event_type == "step_detail"``; may be ``None`` otherwise.
        - ``payload`` is a non-empty JSONB dict matching the SSE event
          structure.

    Abstraction function:
        Represents a single SSE event persisted to the database so that
        workspace cost can be reconstructed without an active stream.
    """
    __tablename__ = "query_events"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    source = Column(String, nullable=True)
    payload = Column(JSONB, nullable=False)
    step_cost_usd = Column(Float, nullable=True)
    created_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),  # noqa: UP017
    )

    __table_args__ = (
        Index("ix_query_events_conv_id_created", "conversation_id", "created_at"),
    )


# dependency to get database session
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# database initialization is handled by Alembic migrations
async def init_db():
    pass
