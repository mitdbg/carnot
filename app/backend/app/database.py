import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
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
DB_NAME = read_secret("db_name")
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


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=True)  # Auto-generated from first query
    dataset_ids = Column(String, nullable=True)  # Comma-separated dataset IDs
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
