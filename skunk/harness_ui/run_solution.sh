#!/usr/bin/env bash
# Run OUR competition solution: the agent BACKEND (worker pool + Cup adapter + FileSink) and the
# browser-facing WEB process (FileTailer + SSE + command proxy). They share $SKUNK_STREAM_DIR; the
# backend connects to a Cup server and submits answers, the web only ever reads the filesystem +
# proxies commands. This is exactly what we run during the real Cup — the only thing that changes
# between rehearsal and competition is which Cup the backend points at.
#
#   REAL COMPETITION: point at the host's Cup and authenticate.
#       ./run_solution.sh --cup-base-url https://cup.databricks.example --team-token "$TOKEN"
#
#   LOCAL REHEARSAL (default): in another terminal start ./run_practice_server.sh, then run this
#       with no --cup-base-url — it defaults to the local practice server at http://127.0.0.1:8765.
#
# Corpus + LLM settings (GEMINI_API_KEY, OFFICEQA_PDF_DIR, SKUNK_CHROMADB_DIR, SKUNK_RETRIEVER, the
# SKUNK_HUMAN_* gates, ...) come from skunk/.env; anything already in your shell wins over the file.
#
# Knobs (flag or env): --cup-base-url/CUP_BASE_URL, --team-token/CUP_TEAM_TOKEN,
# --concurrency/SKUNK_CONCURRENCY, --reasoner/REASONER, --blocking. Ports via the SKUNK_CONSOLE_*
# env vars.
set -euo pipefail

cd "$(dirname "$0")"
HARNESS_DIR="$(pwd)"
SKUNK_DIR="$(cd .. && pwd)"

# --- Load skunk/.env without clobbering vars already in the environment ---
ENV_FILE="${SKUNK_ENV_FILE:-$SKUNK_DIR/.env}"
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

# --- Flags (override env) ---
EXTERNAL_CUP=""        # set => --cup-base-url supplied (a real Cup), not the local default
BLOCKING=""            # set => blocking human transport (branch suspends on the human in the UI)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cup-base-url) export CUP_BASE_URL="$2"; EXTERNAL_CUP=1; shift 2 ;;
    --team-token)   export CUP_TEAM_TOKEN="$2"; shift 2 ;;
    --concurrency)  export SKUNK_CONCURRENCY="$2"; shift 2 ;;
    --reasoner)     export REASONER="$2"; shift 2 ;;
    --blocking)     BLOCKING=1; export SKUNK_HUMAN_BLOCKING=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Blocking needs a human gate ON to actually consult anyone — default extract verification on if
# the caller set none (the transport decides HOW a review is serviced; the gate decides WHETHER).
if [[ -n "$BLOCKING" && -z "${SKUNK_HUMAN_VERIFY_EXTRACT:-}${SKUNK_HUMAN_FIGURE:-}${SKUNK_HUMAN_LOOKUP:-}" ]]; then
  export SKUNK_HUMAN_VERIFY_EXTRACT=1
fi

HOST="${SKUNK_CONSOLE_HOST:-127.0.0.1}"
CUP_PORT="${SKUNK_CONSOLE_CUP_PORT:-8765}"
SKUNK_SERVER_PORT="${SKUNK_CONSOLE_SERVER_PORT:-8787}"   # backend (agent) — localhost only
WEB_PORT="${SKUNK_CONSOLE_WEB_PORT:-8788}"               # web (browser-facing) — the UI
# Question-execution parallelism = questions released per round (the real Cup releases 15).
CONCURRENCY="${SKUNK_CONCURRENCY:-15}"
REASONER="${REASONER:-skunk_reasoner:solve}"

# Cup the backend submits to. Default: the local practice server (run_practice_server.sh) so you can
# rehearse end to end. In the real Cup, pass --cup-base-url + --team-token to point at the host.
export CUP_BASE_URL="${CUP_BASE_URL:-http://${HOST}:${CUP_PORT}}"
export CUP_TEAM_TOKEN="${CUP_TEAM_TOKEN:-anything}"
# Backend (agent) URL — the web process proxies command routes here; never browser-facing.
export SKUNK_SERVER_URL="http://${HOST}:${SKUNK_SERVER_PORT}"

# Python interpreter: explicit PYTHON_BIN, else the repo venv, else PATH python3.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$SKUNK_DIR/venv/bin/python" ]]; then PYTHON_BIN="$SKUNK_DIR/venv/bin/python"
  else PYTHON_BIN="$(command -v python3)"; fi
fi

# Per-run shared stream dir: the backend (FileSink) writes status.json + events.jsonl here and the
# web process (FileTailer) tails it back into its SSE hub. Override via SKUNK_STREAM_DIR.
if [[ -z "${SKUNK_STREAM_DIR:-}" ]]; then
  export SKUNK_STREAM_DIR="$SKUNK_DIR/logs/console/stream/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SKUNK_STREAM_DIR"

# Per-run trace dir: the reasoner dumps each question's event stream here. Set SKUNK_CONSOLE_TRACE_DIR
# empty in the env to disable, or to a fixed path to override the timestamped default.
if [[ -z "${SKUNK_CONSOLE_TRACE_DIR+x}" ]]; then
  export SKUNK_CONSOLE_TRACE_DIR="$SKUNK_DIR/logs/console/$(date +%Y%m%d_%H%M%S)"
fi
[[ -n "${SKUNK_CONSOLE_TRACE_DIR:-}" ]] && mkdir -p "$SKUNK_CONSOLE_TRACE_DIR"

# Relative paths in .env (e.g. SKUNK_CHROMADB_DIR=.chromadb) resolve against cwd, and the corpus
# caches live under the skunk dir — so run the python processes from there, harness on PYTHONPATH.
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$SKUNK_DIR"

# --- Preflight: the ChromaDB server must be up (read paths use HttpClient, not embedded) ---
# Probe with the same helper the solution uses, so a passing probe guarantees the real
# connection works. Hard-stop BEFORE spawning any worker if it's down.
CHROMA_HOST="${SKUNK_CHROMA_SERVER_HOST:-127.0.0.1}"
CHROMA_PORT="${SKUNK_CHROMA_SERVER_PORT:-8001}"
if ! "$PYTHON_BIN" -c "from skunk.chroma_client import make_chroma_client; make_chroma_client('${CHROMA_HOST}', int('${CHROMA_PORT}'))" 2>/dev/null; then
  echo "ERROR: ChromaDB server not reachable at ${CHROMA_HOST}:${CHROMA_PORT}." >&2
  echo "Start it first in a long-lived tmux:  ./scripts/run_chroma_server.sh" >&2
  exit 1
fi
echo "ChromaDB:   ${CHROMA_HOST}:${CHROMA_PORT}   (server reachable)"

pids=()

# Backend (agent) process: worker pool + command routes, bound to localhost, writes the browser
# firehose to $SKUNK_STREAM_DIR. Serves no browsers.
"$PYTHON_BIN" -m skunk_server.server --host "$HOST" --port "$SKUNK_SERVER_PORT" \
  --cup-base-url "$CUP_BASE_URL" --team-token "$CUP_TEAM_TOKEN" \
  --reasoner "$REASONER" --concurrency "$CONCURRENCY" \
  --stream-dir "$SKUNK_STREAM_DIR" ${BLOCKING:+--human-blocking} &
pids+=($!)

# Web (browser-facing) process: serves the UI + SSE by tailing $SKUNK_STREAM_DIR and proxies command
# routes to the backend. This is the only port a browser should hit.
"$PYTHON_BIN" -m skunk_server.web_app --host "$HOST" --port "$WEB_PORT" \
  --backend-url "$SKUNK_SERVER_URL" --stream-dir "$SKUNK_STREAM_DIR" &
pids+=($!)

# Tear both processes down on exit/Ctrl-C: SIGTERM (graceful uvicorn), then SIGKILL any survivor —
# the reasoner offloads blocking tool/code work onto non-daemon threads that can't be interrupted,
# so without the hard kill a wedged worker could keep a server alive.
shutdown() {
  trap '' INT TERM
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 8); do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break
    sleep 1
  done
  kill -KILL "${pids[@]}" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

echo "Cup:        $CUP_BASE_URL $([[ -n "$EXTERNAL_CUP" ]] && echo '(external host)' || echo '(local practice server)')"
echo "Console UI: http://${HOST}:${WEB_PORT}   (web process serves the monitoring UI)"
echo "Backend:    $SKUNK_SERVER_URL   (agent process, localhost only)"
echo "Stream dir: $SKUNK_STREAM_DIR"
echo "Reasoner:   $REASONER   concurrency=$CONCURRENCY"
echo "Human:      transport=$([[ -n "$BLOCKING" ]] && echo blocking || echo optimistic)  gate VERIFY_EXTRACT=${SKUNK_HUMAN_VERIFY_EXTRACT:-0} FIGURE=${SKUNK_HUMAN_FIGURE:-0} LOOKUP=${SKUNK_HUMAN_LOOKUP:-0}"

# Block until either process exits, then the trap tears the other down.
# (Poll loop instead of `wait -n` so this works on macOS's stock bash 3.2.)
while :; do
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || exit 0; done
  sleep 1
done
