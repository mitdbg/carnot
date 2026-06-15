"""Export a single ChromaDB collection into a fresh, slimmed-down PersistentClient store.

Reads `--collection` from the `--src` store and writes it into a new store at `--dest`,
rebuilding the HNSW index there. Use it to carve one collection out of a large multi-collection
store so the server can hold the whole thing in RAM with a clean, small sqlite (see
`scripts/run_chroma_server.sh`).

The SOURCE store must NOT be served by a running chroma server at the same time — a second
client on one directory is a cross-process lock that hangs both. Stop the server first.

  python -m skunk.search_agent.prep.export_chroma_collection \
      --src .chromadb --collection qwen-v2 --dest .chromadb-slim-officeqa
"""

from __future__ import annotations

import argparse

import chromadb

# ChromaDB rejects add()/get() batches larger than this.
CHROMA_MAX_BATCH_SIZE = 5000


def _copy_collection(src_collection, dest_collection, batch_size: int) -> int:
    """Page the whole source collection (embeddings + documents + metadatas) into the dest,
    which rebuilds its HNSW index as rows are added."""
    total = src_collection.count()
    copied = 0
    while copied < total:
        batch = src_collection.get(
            limit=batch_size,
            offset=copied,
            include=["embeddings", "documents", "metadatas"],
        )
        ids = batch["ids"]
        if not ids:
            break
        # chroma returns embeddings as a numpy array; add() wants plain lists.
        embeddings = [list(e) for e in batch["embeddings"]]
        dest_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )
        copied += len(ids)
        print(f"  copied {copied}/{total} rows", flush=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=".chromadb", help="source PersistentClient path")
    parser.add_argument("--collection", required=True, help="collection name to export")
    parser.add_argument("--dest", required=True, help="destination PersistentClient path (created)")
    parser.add_argument("--batch-size", type=int, default=CHROMA_MAX_BATCH_SIZE)
    args = parser.parse_args()

    src_client = chromadb.PersistentClient(path=args.src)
    src_collection = src_client.get_collection(name=args.collection)

    dest_client = chromadb.PersistentClient(path=args.dest)
    # Preserve the source's collection metadata (e.g. {"hnsw:space": "cosine"}) so the rebuilt
    # index uses the SAME distance function — otherwise search results would differ.
    dest_collection = dest_client.get_or_create_collection(
        name=args.collection,
        metadata=src_collection.metadata or None,
    )

    print(
        f"Exporting '{args.collection}': {args.src} -> {args.dest} "
        f"({src_collection.count()} rows)",
        flush=True,
    )
    n = _copy_collection(src_collection, dest_collection, args.batch_size)
    print(f"Done: copied {n} rows into {args.dest}", flush=True)


if __name__ == "__main__":
    main()
