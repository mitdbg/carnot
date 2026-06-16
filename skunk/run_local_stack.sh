#!/usr/bin/env bash
# Tiny local launcher for the whole practice stack.
# Edit the variables below when you want to switch modes.
set -euo pipefail

ROOT_DIR="/home/gerardo/carnot/skunk"
PYTHON_BIN="/home/gerardo/.local/share/mamba/envs/carnot/bin/python"
CHROMA_BIN="/home/gerardo/.local/share/mamba/envs/carnot/bin/chroma"

HOST="127.0.0.1"
CUP_PORT="8765"
BACKEND_PORT="8787"
WEB_PORT="8788"
CHROMA_PORT="8001"

CHROMA_STORE="$ROOT_DIR/cache/chromadb/qwen-v2-export"
CHROMA_COLLECTION="qwen-v2"
CONCURRENCY="15"

REASONER="dummy_agent:solve"
# REASONER="skunk_reasoner:solve"

ROUND_SECONDS="3600"
QUESTIONS_FILE="$ROOT_DIR/harness_ui/questions/practice_questions.json"

cd "$ROOT_DIR"

export PYTHON_BIN
export CHROMA_BIN
export SKUNK_CHROMADB_DIR="$CHROMA_STORE"
export SKUNK_CHROMADB_COLLECTION="$CHROMA_COLLECTION"
export SKUNK_CHROMA_SERVER_HOST="$HOST"
export SKUNK_CHROMA_SERVER_PORT="$CHROMA_PORT"
export SKUNK_CONSOLE_HOST="$HOST"
export SKUNK_CONSOLE_CUP_PORT="$CUP_PORT"
export SKUNK_CONSOLE_SERVER_PORT="$BACKEND_PORT"
export SKUNK_CONSOLE_WEB_PORT="$WEB_PORT"
export REASONER

pids=()

cleanup() {
  trap '' INT TERM EXIT
  echo
  echo "Stopping local stack..."
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

echo "Starting ChromaDB on ${HOST}:${CHROMA_PORT}"
"$ROOT_DIR/scripts/run_chroma_server.sh" "$CHROMA_COLLECTION" &
pids+=($!)
sleep 5

echo "Starting practice Cup on ${HOST}:${CUP_PORT}"
"$ROOT_DIR/harness_ui/run_practice_server.sh" \
  --host "$HOST" \
  --port "$CUP_PORT" \
  --round-seconds "$ROUND_SECONDS" \
  --questions "$QUESTIONS_FILE" &
pids+=($!)
sleep 3

echo "Starting solution backend + web UI with ${REASONER}"
"$ROOT_DIR/harness_ui/run_solution.sh" --concurrency "$CONCURRENCY" &
pids+=($!)

echo
echo "Local stack is starting."
echo "Cup:        http://${HOST}:${CUP_PORT}"
echo "Backend:    http://${HOST}:${BACKEND_PORT}"
echo "Console UI: http://${HOST}:${WEB_PORT}"
echo "ChromaDB:   http://${HOST}:${CHROMA_PORT}"
echo "Reasoner:   ${REASONER}"
echo
echo "Press Ctrl-C here to stop everything."

while :; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A stack process exited; shutting down the rest."
      exit 1
    fi
  done
  sleep 1
done
