import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func


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

# Base class for models
Base = declarative_base()

# Database models
class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id = Column(String, primary_key=True, unique=True, index=True, nullable=False)
    openai_api_key = Column(String, nullable=True)
    anthropic_api_key = Column(String, nullable=True)
    gemini_api_key = Column(String, nullable=True)
    together_api_key = Column(String, nullable=True)
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    name = Column(String, unique=True, nullable=False, index=True)
    shared = Column(Boolean, default=False)
    annotation = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    file_path = Column(String, unique=True, nullable=False)
    shared = Column(Boolean, default=False)
    upload_date = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

class DatasetFile(Base):
    __tablename__ = "dataset_files"

    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False, primary_key=True)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=True)  # Auto-generated from first query
    dataset_ids = Column(String, nullable=True)  # Comma-separated dataset IDs
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'status', 'error', 'result'
    content = Column(Text, nullable=False)
    csv_file = Column(String, nullable=True)  # For result messages
    row_count = Column(Integer, nullable=True)  # For result messages
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc))

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

