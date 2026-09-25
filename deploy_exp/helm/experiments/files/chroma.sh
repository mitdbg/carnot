#!/usr/bin/env bash
# sidecar: the chroma server over the pulled store, warming the cell's collection. Delegates to the same
# script the dev box uses (it raises the open-file limit, starts `chroma run`, warms, then waits on it).
set -euo pipefail
COLLECTION="$(python3 /scripts/cell.py collection)"
echo "[chroma] serving /data/chromadb on ${SKUNK_CHROMA_SERVER_HOST}:${SKUNK_CHROMA_SERVER_PORT}, warming $COLLECTION"
exec /app/skunk/scripts/run_chroma_server.sh "$COLLECTION"
