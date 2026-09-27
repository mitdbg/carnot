#!/usr/bin/env bash
# sidecar: the chroma server over the pulled store, warming the cell's collection. Delegates to the same
# script the dev box uses (it raises the open-file limit, starts `chroma run`, warms, then waits on it).
#
# Bind to 0.0.0.0, not the SKUNK_CHROMA_SERVER_HOST=127.0.0.1 the other containers connect to: the
# kubelet's httpGet startup/liveness probes hit the POD IP, and a server bound to loopback never answers
# them, so the runner would wait forever. The pod's network namespace is private and nothing exposes the
# port through a Service, so 0.0.0.0 is loopback-equivalent here. The warm-up client inside the script
# connects to the same HOST value, which works for 0.0.0.0 on Linux.
set -euo pipefail
COLLECTION="$(python3 /scripts/cell.py collection)"
export SKUNK_CHROMA_SERVER_HOST=0.0.0.0
echo "[chroma] serving /data/chromadb on 0.0.0.0:${SKUNK_CHROMA_SERVER_PORT} (probes use the pod IP), warming $COLLECTION"
exec /app/skunk/scripts/run_chroma_server.sh "$COLLECTION"
