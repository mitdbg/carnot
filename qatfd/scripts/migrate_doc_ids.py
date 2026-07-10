"""In-place `doc_id` migration: promote a collection's retrieval unit without rebuilding.

Before 2026-07-09 the QAMPARI and FreshStack collections used the embedded CHUNK as the `doc_id`
(the SearchAgent's retrieval unit — what it reads, prunes, and returns). CORPUS_MODEL.md promotes
the unit to the natural document: the Wikipedia article (its numeric `page_id`) for QAMPARI, the
source file (`file_id`) for FreshStack. Both new ids are derivable from the row id itself (the
chunk id encodes them), so the migration is a batched, metadata-only rewrite —
`Collection.update(ids, metadatas=...)` touches neither the embeddings nor the HNSW index; no
re-embedding, no re-ingestion:

  qampari:    row id "{page_id}__{n}"                -> doc_id = page_id  (cross-checked against
              the row's `page_id` metadata when present; mismatch aborts)
  freshstack: row id "{repo}/{path}_{start}_{end}"   -> doc_id = file_id ("{repo}/{path}"),
                                                        element_id = start_byte (monotonic within
                                                        the file, so grep/doc assembly order holds)

The transform is idempotent (already-migrated rows are detected and skipped) and reversible (the
pre-migration doc_id equals the still-stored chunk_id). Progress is checkpointed (a row cursor) to
a sidecar file every batch, so a killed run resumes where it left off.

Row enumeration: in embedded mode (--chroma-path) rows are enumerated straight from the store's
sqlite (`SELECT id, embedding_id FROM embeddings WHERE id > ? ORDER BY id LIMIT ?` — read-only,
keyset pagination, O(batch) per query; the checkpoint is the last processed sqlite rowid). Chroma's
`get(offset=...)` walks the table from the start on every call (~13s per batch at offset 12M on the
25.9M-row QAMPARI store — hours of pure pagination), so it is used only in server mode (--host),
where the sqlite file isn't reachable (the checkpoint is then the row offset).

Do NOT run against a collection a live chroma server is serving from the same directory in
embedded mode — stop the server first, or point --host/--port at it instead.

Usage:
  # FreshStack (one topic at a time; ~50k rows, minutes):
  python3 scripts/migrate_doc_ids.py --benchmark freshstack \
      --chroma-path benchmarks/freshstack/langchain/chromadb \
      --collection-name freshstack-langchain-qwen-0.6b
  # QAMPARI (25.9M rows): use --direct-sql (~75 min; the API row-update path measured ~20 HOURS
  # on this store). It rewrites the doc_id rows of `embedding_metadata` in one batched SQL
  # UPDATE — verified byte-identical to the API path on a synthetic store (embeddings, documents,
  # the fulltext index, and all other metadata keys are untouched; only doc_id's string_value —
  # and its (key, string_value) index — changes, and sqlite maintains the index itself):
  python3 scripts/migrate_doc_ids.py --benchmark qampari --direct-sql \
      --chroma-path benchmarks/qampari/chromadb --collection-name qwen-qampari-0.6b
"""

import argparse
import os
import sqlite3
import time
from collections.abc import Iterator

import chromadb

from qatfd.keys import freshstack_byte_range, freshstack_file_id

# chroma validates batches against the client max (5461); stay under it for updates.
UPDATE_BATCH = 5000


def _qampari_new_meta(chunk_id: str, meta: dict) -> dict | None:
    """New metadata for a QAMPARI row, or None when already migrated. doc_id becomes the article
    page_id — the id's prefix ("{page_id}__{n}"; page_id is numeric, so it never contains "__")."""
    page_id = chunk_id.rsplit("__", 1)[0]
    if not page_id or page_id == chunk_id:
        raise ValueError(f"qampari row {chunk_id!r}: cannot derive page_id from the id")
    meta_page_id = str(meta.get("page_id", ""))
    if meta_page_id and meta_page_id != page_id:
        raise ValueError(
            f"qampari row {chunk_id!r}: id-derived page_id {page_id!r} != metadata page_id {meta_page_id!r}"
        )
    if meta.get("doc_id") == page_id:
        return None
    return {**meta, "doc_id": page_id}


def _freshstack_new_meta(chunk_id: str, meta: dict) -> dict | None:
    """New metadata for a FreshStack row, or None when already migrated. doc_id becomes the source
    file; element_id becomes the chunk's start_byte (its within-file order)."""
    file_id = freshstack_file_id(chunk_id)
    span = freshstack_byte_range(chunk_id)
    element_id = span[0] if span else 0
    if meta.get("doc_id") == file_id and meta.get("element_id") == element_id:
        return None
    return {**meta, "doc_id": file_id, "element_id": element_id}


TRANSFORMS = {"qampari": _qampari_new_meta, "freshstack": _freshstack_new_meta}


def _checkpoint_path(args) -> str:
    base = args.chroma_path if args.chroma_path else "."
    return os.path.join(base, f".migrate_{args.collection_name}.cursor")


def _load_cursor(path: str) -> int:
    if os.path.exists(path):
        with open(path) as f:
            return int(f.read().strip() or 0)
    return 0


def _iter_batches_sqlite(
    chroma_path: str, collection, batch_size: int, cursor: int
) -> Iterator[tuple[int, list[str], list[dict]]]:
    """(new_cursor, ids, metadatas) batches enumerated by sqlite rowid keyset pagination.
    Metadata still comes through the chroma API (indexed get-by-ids), so all writes AND reads of
    metadata stay inside chroma; sqlite is only read for the id list."""
    db = sqlite3.connect(f"file:{os.path.join(chroma_path, 'chroma.sqlite3')}?mode=ro", uri=True)
    while True:
        rows = db.execute(
            "SELECT id, embedding_id FROM embeddings WHERE id > ? ORDER BY id LIMIT ?",
            (cursor, batch_size),
        ).fetchall()
        if not rows:
            return
        ids = [str(r[1]) for r in rows]
        got = collection.get(ids=ids, include=["metadatas"])
        meta_by_id = {str(i): (m or {}) for i, m in zip(got.get("ids") or [], got.get("metadatas") or [])}
        cursor = rows[-1][0]
        yield cursor, ids, [meta_by_id.get(i, {}) for i in ids]


def _run_direct_sql_qampari(chroma_path: str, ckpt: str) -> None:
    """QAMPARI fast path: rewrite the `doc_id` metadata rows directly in the store's sqlite,
    batched by id range (bounded rollback journal), committing + checkpointing per batch so a
    killed run resumes where it left off. QAMPARI-only because its transform is pure
    string SQL (page_id = the id up to the first "__"; page_ids are numeric so the first "__" is
    THE separator). The chroma API path on the 25.9M-row store measured ~350 rows/s (~20 h) —
    per-row update overhead — vs ~5,700 rows/s (~75 min) here."""
    batch = 500_000
    db = sqlite3.connect(os.path.join(chroma_path, "chroma.sqlite3"), timeout=60)
    (max_id,) = db.execute("SELECT max(id) FROM embeddings").fetchone()
    t0 = time.time()
    total = 0
    lo = _load_cursor(ckpt) + 1
    if lo > 1:
        print(f"Resuming from id {lo} (checkpoint {ckpt}).")
    while lo <= max_id:
        hi = min(lo + batch - 1, max_id)
        cur = db.execute(
            """UPDATE embedding_metadata
               SET string_value = (SELECT substr(e.embedding_id, 1, instr(e.embedding_id, '__') - 1)
                                   FROM embeddings e WHERE e.id = embedding_metadata.id)
               WHERE key = 'doc_id' AND id BETWEEN ? AND ?""",
            (lo, hi),
        )
        db.commit()
        with open(ckpt, "w") as f:
            f.write(str(hi))
        total += cur.rowcount
        print(f"  ids {lo}-{hi}: {cur.rowcount} rows ({100 * hi / max_id:.1f}% done, "
              f"elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
        lo = hi + 1
    db.close()
    print(f"Done: {total} doc_id rows rewritten in {(time.time() - t0) / 60:.1f} min.")
    if os.path.exists(ckpt):
        os.remove(ckpt)


def _iter_batches_offset(collection, batch_size: int, cursor: int) -> Iterator[tuple[int, list[str], list[dict]]]:
    """(new_cursor, ids, metadatas) batches via chroma offset pagination (server mode only — the
    offset walk is O(offset) per call, so prefer the sqlite path when the store dir is local)."""
    while True:
        got = collection.get(limit=batch_size, offset=cursor, include=["metadatas"])
        ids = [str(i) for i in got.get("ids") or []]
        if not ids:
            return
        cursor += len(ids)
        yield cursor, ids, [(m or {}) for m in got.get("metadatas") or []]


def main() -> None:
    parser = argparse.ArgumentParser(description="In-place doc_id migration for QAMPARI / FreshStack collections.")
    parser.add_argument("--benchmark", required=True, choices=sorted(TRANSFORMS))
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--chroma-path", default=None, help="embedded PersistentClient dir (default mode)")
    parser.add_argument("--host", default=None, help="chroma server host (instead of --chroma-path)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--batch-size", type=int, default=20000, help="rows fetched per batch")
    parser.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    parser.add_argument("--direct-sql", action="store_true",
                        help="qampari + --chroma-path only: rewrite doc_id rows via batched SQL (~16x faster)")
    args = parser.parse_args()

    if bool(args.chroma_path) == bool(args.host):
        parser.error("exactly one of --chroma-path / --host is required")
    if args.direct_sql:
        if args.benchmark != "qampari" or not args.chroma_path or args.dry_run:
            parser.error("--direct-sql supports only --benchmark qampari with --chroma-path (and no --dry-run)")
        _run_direct_sql_qampari(args.chroma_path, _checkpoint_path(args))
        return
    client = (
        chromadb.HttpClient(host=args.host, port=args.port)
        if args.host
        else chromadb.PersistentClient(path=args.chroma_path)
    )
    collection = client.get_collection(name=args.collection_name)
    transform = TRANSFORMS[args.benchmark]
    total = collection.count()

    ckpt = _checkpoint_path(args)
    cursor = 0 if args.dry_run else _load_cursor(ckpt)
    if cursor:
        print(f"Resuming from cursor {cursor} (checkpoint {ckpt}).")
    batches = (
        _iter_batches_sqlite(args.chroma_path, collection, args.batch_size, cursor)
        if args.chroma_path
        else _iter_batches_offset(collection, args.batch_size, cursor)
    )
    print(f"Migrating {args.benchmark!r} collection {args.collection_name!r}: {total} rows"
          f"{' (dry run)' if args.dry_run else ''}.")

    n_seen = n_updated = n_skipped = 0
    t0 = time.time()
    last_log = 0.0
    for cursor, ids, metas in batches:
        upd_ids: list[str] = []
        upd_metas: list[dict] = []
        for cid, meta in zip(ids, metas):
            new_meta = transform(cid, dict(meta))
            if new_meta is None:
                n_skipped += 1
            else:
                upd_ids.append(cid)
                upd_metas.append(new_meta)
        if upd_ids and not args.dry_run:
            for i in range(0, len(upd_ids), UPDATE_BATCH):
                collection.update(ids=upd_ids[i : i + UPDATE_BATCH], metadatas=upd_metas[i : i + UPDATE_BATCH])
        n_updated += len(upd_ids)
        n_seen += len(ids)
        if not args.dry_run:
            with open(ckpt, "w") as f:
                f.write(str(cursor))
        now = time.time()
        if now - last_log >= 15 or n_seen >= total:
            last_log = now
            rate = n_seen / max(now - t0, 1e-9)
            eta = (total - n_seen) / max(rate, 1e-9)
            print(f"  {n_seen}/{total} rows ({100 * n_seen / total:.1f}%) — updated {n_updated}, "
                  f"already-migrated {n_skipped} — {rate:.0f} rows/s, ETA {eta / 60:.0f} min", flush=True)

    print(f"Done: {n_updated} rows updated, {n_skipped} already migrated"
          f"{' (dry run — nothing written)' if args.dry_run else ''}.")
    if not args.dry_run and os.path.exists(ckpt):
        os.remove(ckpt)


if __name__ == "__main__":
    main()
