#!/usr/bin/env bash
# Tiny local launcher for the whole practice stack.
# Edit the variables below when you want to switch modes.
set -euo pipefail

ROOT_DIR="/home/$USER/carnot/skunk"
PYTHON_BIN="python"

HOST="127.0.0.1"
CUP_PORT="8765"
BACKEND_PORT="8787"
WEB_PORT="8788"
CHROMA_PORT="8001"

# Corpus = DAIS. The exports below win over .env (run_solution.sh loads .env without clobbering
# already-exported vars), so the run doesn't depend on a stale treasury .env.
SKUNK_CHROMADB_DIR=/home/ubuntu/dais/.chromadb-dais-slim
# Table-corrected map (matches the corrected tables pushed into the dais-slim collection).
# Was: /home/ubuntu/dais/cleaned_pages_v1/clean_page_map.json  (raw parsed-JSON text)
CLEAN_PAGE_MAP=/home/ubuntu/dais/cleaned/clean_page_map.json
CHROMA_COLLECTION="dais-slim"
CONCURRENCY="15"

# REASONER="dummy_agent:solve"
REASONER="skunk_reasoner:solve"

ROUND_SECONDS="3600"
QUESTIONS_FILE="$ROOT_DIR/harness_ui/questions/dais_sample_questions.json"

cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs/local_stack/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export PYTHON_BIN
export CHROMA_BIN
export SKUNK_CHROMADB_DIR="$SKUNK_CHROMADB_DIR"
export SKUNK_CLEAN_PAGE_MAP="$CLEAN_PAGE_MAP"
export SKUNK_CHROMADB_COLLECTION="$CHROMA_COLLECTION"
export SKUNK_CHROMA_SERVER_HOST="$HOST"
export SKUNK_CHROMA_SERVER_PORT="$CHROMA_PORT"
export SKUNK_CONSOLE_HOST="$HOST"
export SKUNK_CONSOLE_CUP_PORT="$CUP_PORT"
export SKUNK_CONSOLE_SERVER_PORT="$BACKEND_PORT"
export SKUNK_CONSOLE_WEB_PORT="$WEB_PORT"
export REASONER

# --- DAIS corpus config (authoritative; overrides any stale values in .env) ---
export OFFICEQA_PDF_DIR=/home/ubuntu/dais/pdfs
export OFFICEQA_PARSED_JSON_DIR=/home/ubuntu/dais/parsed_json
export SKUNK_RETRIEVER=search_agent
export SKUNK_EMB_MODEL=qwen/qwen3-embedding-8b
export SKUNK_PROMPT_OVERRIDES=config/prompts/us_receipts_expenditures.yaml
export SKUNK_PAGE_INDEX_DIR=/home/ubuntu/dais/competition_page_index
export SKUNK_HUMAN_INTERVENTION=0   # bring-up: keep treasury-shaped HITL provenance off the path

pids=()
names=()
logs=()

start_stack_process() {
  local name="$1"
  local log="$2"
  shift 2
  echo "Starting ${name}; log: ${log}"
  "$@" >"$log" 2>&1 &
  pids+=($!)
  names+=("$name")
  logs+=("$log")
}

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

start_stack_process "chroma" "$LOG_DIR/chroma.log" \
  "$ROOT_DIR/scripts/run_chroma_server.sh" "$CHROMA_COLLECTION"
sleep 5

start_stack_process "practice-cup" "$LOG_DIR/practice-cup.log" \
  "$ROOT_DIR/harness_ui/run_practice_server.sh" \
  --host "$HOST" \
  --port "$CUP_PORT" \
  --round-seconds "$ROUND_SECONDS" \
  --questions "$QUESTIONS_FILE"
sleep 3

start_stack_process "solution" "$LOG_DIR/solution.log" \
  "$ROOT_DIR/harness_ui/run_solution.sh" --concurrency "$CONCURRENCY"

echo
echo "Local stack is starting."
echo "Cup:        http://${HOST}:${CUP_PORT}"
echo "Backend:    http://${HOST}:${BACKEND_PORT}"
echo "Console UI: http://${HOST}:${WEB_PORT}"
echo "ChromaDB:   http://${HOST}:${CHROMA_PORT}"
echo "Reasoner:   ${REASONER}"
echo "Logs:       ${LOG_DIR}"
echo
echo "Press Ctrl-C here to stop everything."

while :; do
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      exit_code=0
      wait "$pid" || exit_code=$?
      echo "Stack process exited: ${names[$i]} pid=${pid} exit=${exit_code}"
      echo "Log: ${logs[$i]}"
      echo "--- last 120 lines of ${names[$i]} log ---"
      tail -n 120 "${logs[$i]}" || true
      echo "--- end ${names[$i]} log ---"
      echo "Shutting down the rest."
      exit 1
    fi
  done
  sleep 1
done
