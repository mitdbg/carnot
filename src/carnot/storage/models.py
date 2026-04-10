"""SQLAlchemy ORM models for the Carnot catalog tables.

These models live inside the ``carnot`` library so that the storage
catalogs (:class:`DatasetCatalog`, :class:`IndexCatalog`) can talk to
PostgreSQL **without** importing from the web-application layer
(``app.database``).

The web application's own ``app.database`` module should import or
mirror these definitions rather than defining them independently.

Only the tables needed by the catalogs are defined here.  Application-
specific tables (``Conversation``, ``Message``, ``UserSettings``, etc.)
remain in ``app.database``.

Usage::

    from carnot.storage.models import DatasetModel, IndexEntryModel

Representation invariant:
    - ``Base`` is a single shared declarative base; all models in this
      module inherit from it.
    - Column names and types match the Alembic-managed schema.

Abstraction function:
    Each model class represents a row in the corresponding PostgreSQL
    table.  An instance maps to one row; the class maps to the table
    as a whole.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import declarative_base

Base = declarative_base()
"""Shared declarative base for all Carnot catalog models."""


class DatasetModel(Base):
    """ORM model for the ``datasets`` table.

    Mirrors the schema managed by Alembic migrations in the web app.
    """

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    name = Column(String, unique=True, nullable=False, index=True)
    shared = Column(Boolean, default=False)
    annotation = Column(Text, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class IndexEntryModel(Base):
    """ORM model for the ``indices`` table.

    Stores metadata for persisted indices (flat, hierarchical, chroma,
    faiss).  The actual index data lives in the storage backend; the
    ``storage_uri`` column points to it.
    """

    __tablename__ = "indices"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    index_type = Column(String, nullable=False)
    config_json = Column(Text, nullable=True)
    storage_uri = Column(String, nullable=True, default="")
    item_count = Column(Integer, nullable=True)
    is_stale = Column(Boolean, default=False)
    created_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("ix_indices_dataset_name", "dataset_id", "name", unique=True),
    )
