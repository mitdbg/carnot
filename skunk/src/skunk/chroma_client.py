"""Shared ChromaDB client construction.

Read paths talk to a ChromaDB **server** (HttpClient), not the embedded `PersistentClient`.
The embedded engine deadlocks under the eval's 15-way in-process concurrency (worker threads
wedge inside ChromaDB's Rust core); the server process owns ChromaDB's concurrency, so many
clients can query it in parallel safely. Start the server with `scripts/run_chroma_server.sh`.

Build/write scripts (`create_vector_db.py`, `export_chroma_collection.py`, …) still use
`PersistentClient` directly — they run standalone, and must NOT touch the same on-disk store
while the server holds it (a second client on one directory is a cross-process lock).
"""

from __future__ import annotations

import chromadb


def make_chroma_client(host: str, port: int):
    """Connect to the ChromaDB server at `host:port`, failing fast with a clear message if
    it isn't reachable. Every read consumer (eval runtime, datagen, prep harnesses) builds
    its client through here so they behave identically."""
    try:
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()  # forces a real round-trip; HttpClient construction alone is lazy
    except Exception as e:
        raise RuntimeError(
            f"Could not reach the ChromaDB server at {host}:{port} ({e}). "
            "Start it in a long-lived tmux first:  ./scripts/run_chroma_server.sh "
            "(override host/port via SKUNK_CHROMA_SERVER_HOST / SKUNK_CHROMA_SERVER_PORT)."
        ) from e
    return client
