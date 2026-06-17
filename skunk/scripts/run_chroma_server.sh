#!/usr/bin/env bash
# Run the ChromaDB SERVER over the corpus store, in a long-lived tmux.
#
# Why: the embedded `PersistentClient` deadlocks under our mandated 15-way in-process
# concurrency (worker threads wedge inside ChromaDB's Rust core). The server process owns
# ChromaDB's concurrency, so many clients can query it in parallel safely. Every read path
# in skunk (eval runtime, datagen, prep harnesses) connects to this server via HttpClient.
#
#   tmux new -s chroma
#   ./scripts/run_chroma_server.sh qwen-v2            # warm one collection
#   ./scripts/run_chroma_server.sh qwen-v2 <other>    # warm several
#
# The collection(s) to warm are a REQUIRED argument (no default) so you can't forget to set
# it. Warm start: ChromaDB has no "preload on startup" flag — it loads a collection's HNSW
# index into memory lazily on first query, so the first real query eats a multi-minute
# cold-load on a large store. This script fires one warm-up query against each named
# collection right after the server is ready, so the index is resident before the solution
# starts. The warm-up uses a stored embedding as the query vector (no embedding API needed).
#
# Knobs (env): SKUNK_CHROMADB_DIR (store path, default .chromadb — the real store),
# SKUNK_CHROMA_SERVER_HOST (default 127.0.0.1), SKUNK_CHROMA_SERVER_PORT (default 8001).
# These names match config.py / the read paths.
#
# IMPORTANT: do NOT run the build/export scripts (create_vector_db.py, ...) against the same
# store while this server is up — a second client on one directory is the cross-process lock
# that hangs both. Stop this server first.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# --- Load skunk/.env without clobbering vars already in the environment ---
# Same idiom as run_solution.sh / run_practice_server.sh, so all three components share one
# config source (SKUNK_CHROMADB_DIR, SKUNK_CHROMA_SERVER_*). An explicit shell export still wins.
ENV_FILE="${SKUNK_ENV_FILE:-$(pwd)/.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$val" == \"*\" || "$val" == \'*\' ]]; then val="${val:1:${#val}-2}"; fi
    if ! eval "[ -n \"\${$key+x}\" ]"; then export "$key=$val"; fi
  done < "$ENV_FILE"
fi

# --- Required: the collection(s) to warm (fail fast, before starting the server) ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <collection> [<collection> ...]" >&2
  echo "  Starts the ChromaDB server and warm-loads the named collection(s) into memory." >&2
  echo "  Example: $0 qwen-v2" >&2
  exit 2
fi
WARM_COLLECTIONS="$(IFS=,; echo "$*")"   # comma-join the positional args

CHROMA_PATH="${SKUNK_CHROMADB_DIR:-.chromadb}"
HOST="${SKUNK_CHROMA_SERVER_HOST:-127.0.0.1}"
PORT="${SKUNK_CHROMA_SERVER_PORT:-8001}"
PY="./venv/bin/python"

if [[ ! -d "$CHROMA_PATH" ]]; then
  echo "ERROR: ChromaDB store '$CHROMA_PATH' not found. Build it first (src/skunk/search_agent/prep/)" >&2
  echo "or set SKUNK_CHROMADB_DIR to the store directory." >&2
  exit 1
fi

echo "Starting ChromaDB server on ${HOST}:${PORT} over '${CHROMA_PATH}'"
./venv/bin/chroma run --host "$HOST" --port "$PORT" --path "$CHROMA_PATH" &
SERVER_PID=$!

# Kill the server when this script exits (Ctrl-C in the tmux pane, etc.) so it never orphans.
cleanup() { trap '' INT TERM EXIT; kill -TERM "$SERVER_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

# Wait until the server answers a heartbeat (the HTTP layer comes up fast; this does NOT load
# any collection). Bail out if the server dies during startup.
echo "Waiting for the server to accept connections..."
ready=0
for _ in $(seq 1 120); do
  if "$PY" -c "from skunk.chroma_client import make_chroma_client; make_chroma_client('${HOST}', int('${PORT}'))" 2>/dev/null; then
    ready=1; break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: chroma server exited during startup" >&2; exit 1; }
  sleep 1
done
[[ "$ready" == 1 ]] || { echo "ERROR: chroma server not ready after 120s" >&2; exit 1; }

# Warm start: load each named collection's HNSW index into memory (best-effort; a missing
# collection or warm failure must NOT take the server down).
echo "Warming collection(s): ${WARM_COLLECTIONS} (loads the HNSW index; can take a while cold)"
"$PY" - "$HOST" "$PORT" "$WARM_COLLECTIONS" <<'PY' || echo "[warm] warning: warm-up failed (continuing; first real query will pay the cold-load)"
import sys
from skunk.chroma_client import make_chroma_client

host, port, names = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
client = make_chroma_client(host, port)
for name in (n.strip() for n in names if n.strip()):
    try:
        coll = client.get_collection(name)
        got = coll.get(limit=1, include=["embeddings"])
        embs = got.get("embeddings")
        if embs is None or len(embs) == 0:
            print(f"[warm] '{name}' is empty; nothing to load")
            continue
        # A real query forces the HNSW segment to load into the server's memory.
        coll.query(query_embeddings=[list(embs[0])], n_results=10)
        print(f"[warm] loaded HNSW index for '{name}'")
    except Exception as e:
        print(f"[warm] could not warm '{name}': {e}")
PY

echo "ChromaDB server ready on ${HOST}:${PORT}  (path=${CHROMA_PATH})"
# Keep the server in the foreground so this tmux pane shows its logs and stays tied to its life.
wait "$SERVER_PID"
