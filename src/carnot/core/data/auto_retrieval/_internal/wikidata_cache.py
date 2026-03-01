"""SQLite cache for Wikidata resolve, parent, and label lookups.

Tables
------
resolve_cache
    (concept_key, normalized_text) -> (qid, label, description, ts)
    qid="" means "searched, nothing found" (cached miss).

parents_cache
    (concept_key, qid, property_id, max_depth, language) -> (parents_json, ts)
    parents_json is a JSON array of [qid, label] pairs.

labels
    (qid) -> (label, ts)
    Populated **only** by explicit ``put_labels_batch`` calls.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class WikidataCache:
    """Thin wrapper around a single SQLite file for Wikidata results.

    Use ``batch_writes()`` as a context manager when doing many inserts
    (e.g. during canonicalization) to defer commits until the end, avoiding
    one fsync per row.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._batch_depth = 0       # supports nested batch_writes calls
        self._create_tables()

    @contextmanager
    def batch_writes(self) -> Iterator[None]:
        """Defer SQLite commits until the outermost context exits."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._conn.commit()

    def _maybe_commit(self) -> None:
        """Commit immediately unless inside a ``batch_writes()`` block."""
        if self._batch_depth == 0:
            self._conn.commit()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS resolve_cache (
                concept_key     TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                qid             TEXT NOT NULL DEFAULT '',
                label           TEXT NOT NULL DEFAULT '',
                description     TEXT NOT NULL DEFAULT '',
                ts              REAL NOT NULL,
                PRIMARY KEY (concept_key, normalized_text)
            );

            CREATE TABLE IF NOT EXISTS parents_cache (
                concept_key TEXT    NOT NULL,
                qid         TEXT    NOT NULL,
                property_id TEXT    NOT NULL,
                max_depth   INTEGER NOT NULL,
                language    TEXT    NOT NULL DEFAULT 'en',
                parents_json TEXT   NOT NULL,
                ts          REAL    NOT NULL,
                PRIMARY KEY (concept_key, qid, property_id, max_depth, language)
            );

            CREATE TABLE IF NOT EXISTS labels (
                qid   TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                ts    REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS inferred_profile_sets (
                profile_set_key TEXT NOT NULL,
                concept_key     TEXT NOT NULL,
                profile_json    TEXT NOT NULL,
                ts              REAL NOT NULL,
                PRIMARY KEY (profile_set_key, concept_key)
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Resolve cache
    # ------------------------------------------------------------------
    def get_resolved(
        self, concept_key: str, normalized_text: str
    ) -> Optional[Tuple[str, str, str]]:
        """Return (qid, label, description) or None if not cached.

        qid=="" means a cached miss (we searched, found nothing).
        """
        row = self._conn.execute(
            "SELECT qid, label, description FROM resolve_cache "
            "WHERE concept_key = ? AND normalized_text = ?",
            (concept_key, normalized_text),
        ).fetchone()
        return row  # type: ignore[return-value]

    def put_resolved(
        self,
        concept_key: str,
        normalized_text: str,
        qid: str,
        label: str,
        description: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO resolve_cache "
            "(concept_key, normalized_text, qid, label, description, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (concept_key, normalized_text, qid, label, description, time.time()),
        )
        self._maybe_commit()

    # ------------------------------------------------------------------
    # Parents cache
    # ------------------------------------------------------------------
    def get_parents(
        self,
        concept_key: str,
        qid: str,
        property_id: str,
        max_depth: int,
        language: str = "en",
    ) -> Optional[List[Tuple[str, str]]]:
        """Return list of (qid, label) pairs or None if not cached."""
        row = self._conn.execute(
            "SELECT parents_json FROM parents_cache "
            "WHERE concept_key = ? AND qid = ? AND property_id = ? "
            "AND max_depth = ? AND language = ?",
            (concept_key, qid, property_id, max_depth, language),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])  # type: ignore[return-value]

    def put_parents(
        self,
        concept_key: str,
        qid: str,
        property_id: str,
        max_depth: int,
        parents: List[Tuple[str, str]],
        language: str = "en",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO parents_cache "
            "(concept_key, qid, property_id, max_depth, language, parents_json, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (concept_key, qid, property_id, max_depth, language,
             json.dumps(parents), time.time()),
        )
        self._maybe_commit()

    # ------------------------------------------------------------------
    # Labels — explicit only
    # ------------------------------------------------------------------
    def get_label(self, qid: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT label FROM labels WHERE qid = ?", (qid,)
        ).fetchone()
        return row[0] if row else None

    def get_labels_batch(self, qids: List[str]) -> Dict[str, str]:
        """Return {qid: label} for all *qids* that are in the labels table."""
        if not qids:
            return {}
        out: Dict[str, str] = {}
        # SQLite parameter limit is ~999; chunk if needed.
        for start in range(0, len(qids), 900):
            chunk = qids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for row in self._conn.execute(
                f"SELECT qid, label FROM labels WHERE qid IN ({placeholders})",
                chunk,
            ):
                out[row[0]] = row[1]
        return out

    def put_labels_batch(self, entries: List[Tuple[str, str]]) -> None:
        """Write [(qid, label), ...] to the labels table."""
        if not entries:
            return
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO labels (qid, label, ts) VALUES (?, ?, ?)",
            [(qid, label, now) for qid, label in entries if qid and label],
        )
        self._maybe_commit()

    def collect_all_cached_labels(self) -> Dict[str, str]:
        """Gather every (qid, label) pair from resolve_cache and parents_cache.

        This does NOT touch the labels table — it reads from the raw
        caches so the caller can populate labels explicitly.
        """
        labels: Dict[str, str] = {}
        # From resolve_cache
        for row in self._conn.execute(
            "SELECT qid, label FROM resolve_cache WHERE qid != '' AND label != ''"
        ):
            labels[row[0]] = row[1]
        # From parents_cache (parse JSON arrays)
        for row in self._conn.execute("SELECT parents_json FROM parents_cache"):
            for parent_qid, parent_label in json.loads(row[0]):
                if parent_qid and parent_label and parent_qid not in labels:
                    labels[parent_qid] = parent_label
        return labels

    # ------------------------------------------------------------------
    # Inferred concept profiles
    # ------------------------------------------------------------------
    def get_inferred_profile_set(self, profile_set_key: str) -> Dict[str, str]:
        rows = self._conn.execute(
            "SELECT concept_key, profile_json FROM inferred_profile_sets "
            "WHERE profile_set_key = ?",
            (profile_set_key,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def put_inferred_profile_set(
        self,
        profile_set_key: str,
        profiles_by_key: Dict[str, str],
    ) -> None:
        if not profile_set_key or not profiles_by_key:
            return
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO inferred_profile_sets "
            "(profile_set_key, concept_key, profile_json, ts) VALUES (?, ?, ?, ?)",
            [(profile_set_key, k, v, now) for k, v in profiles_by_key.items()],
        )
        self._maybe_commit()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        counts = {}
        for table in ("resolve_cache", "parents_cache", "labels"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0] if row else 0
        return counts

    def close(self) -> None:
        self._conn.close()
