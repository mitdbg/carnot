#!/usr/bin/env bash
# Launch the split Skunk console: the agent BACKEND (worker pool + Cup adapter + FileSink) and the
# browser-facing WEB process (FileTailer + SSE + command proxy). They share $SKUNK_STREAM_DIR; the
# backend connects to a Cup server and the web only ever reads the filesystem + proxies commands.
#
# Two ways to point the backend at a Cup:
#
#   DEV  (default): also boot the local practice harness (cup_kit.practice_server) on $CUP_PORT and
#                   feed it $QUESTIONS. Generate $QUESTIONS with make_practice_questions.py.
#                     ./run_console.sh --questions /tmp/q.json
#
#   EXTERNAL:       skip the practice server and connect to a real Cup. Supply its URL + token:
#                     ./run_console.sh --cup-base-url https://cup.example --team-token "$TOKEN"
#
# Knobs (flag or env): --questions/SKUNK_CONSOLE_QUESTIONS, --round-seconds/SKUNK_CONSOLE_ROUND_SECONDS,
# --concurrency/SKUNK_CONCURRENCY, --reasoner/REASONER. Human-review gates (SKUNK_HUMAN_VERIFY_EXTRACT
# etc.) and corpus/LLM settings come from skunk/.env. Anything already in your shell wins over .env.
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
EXTERNAL_CUP=""        # set => connect to a real Cup, don't boot the practice server
BLOCKING=""            # set => blocking human transport (branch suspends on the human in the UI)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cup-base-url) export CUP_BASE_URL="$2"; EXTERNAL_CUP=1; shift 2 ;;
    --team-token)   export CUP_TEAM_TOKEN="$2"; shift 2 ;;
    --questions)    export SKUNK_CONSOLE_QUESTIONS="$2"; shift 2 ;;
    --round-seconds) export SKUNK_CONSOLE_ROUND_SECONDS="$2"; shift 2 ;;
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
WEB_PORT="${SKUNK_CONSOLE_WEB_PORT:-8788}"               # web (browser-facing)
ROUND_SECONDS="${SKUNK_CONSOLE_ROUND_SECONDS:-1800}"     # long, so nothing auto-submits mid-demo
CONCURRENCY="${SKUNK_CONCURRENCY:-4}"
REASONER="${REASONER:-skunk_reasoner:solve}"
CUP_TEAM_TOKEN="${CUP_TEAM_TOKEN:-anything}"

if [[ -z "$EXTERNAL_CUP" ]]; then
  # DEV: the backend talks to the local practice server we boot below.
  export CUP_BASE_URL="http://${HOST}:${CUP_PORT}"
  QUESTIONS="${SKUNK_CONSOLE_QUESTIONS:-}"
  if [[ -z "$QUESTIONS" || ! -f "$QUESTIONS" ]]; then
    echo "ERROR: dev mode needs --questions <file> (generate with make_practice_questions.py)" >&2
    exit 2
  fi
else
  : "${CUP_BASE_URL:?--cup-base-url required for external mode}"
fi
export CUP_TEAM_TOKEN
export SKUNK_SERVER_URL="http://${HOST}:${SKUNK_SERVER_PORT}"

# Python interpreter: explicit PYTHON_BIN, else the repo venv, else PATH python3.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$SKUNK_DIR/venv/bin/python" ]]; then PYTHON_BIN="$SKUNK_DIR/venv/bin/python"
  else PYTHON_BIN="$(command -v python3)"; fi
fi

# Per-run shared stream dir (status.json + per-round events.<round>.jsonl) and trace dump dir.
if [[ -z "${SKUNK_STREAM_DIR:-}" ]]; then
  export SKUNK_STREAM_DIR="$SKUNK_DIR/logs/console/stream/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$SKUNK_STREAM_DIR"
if [[ -z "${SKUNK_CONSOLE_TRACE_DIR+x}" ]]; then
  export SKUNK_CONSOLE_TRACE_DIR="$SKUNK_DIR/logs/console/$(date +%Y%m%d_%H%M%S)"
fi
[[ -n "${SKUNK_CONSOLE_TRACE_DIR:-}" ]] && mkdir -p "$SKUNK_CONSOLE_TRACE_DIR"

# Relative caches in .env resolve against cwd, and the corpus lives under the skunk dir.
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$SKUNK_DIR"

pids=()
if [[ -z "$EXTERNAL_CUP" ]]; then
  "$PYTHON_BIN" "$HARNESS_DIR/practice_server.py" --host "$HOST" --port "$CUP_PORT" \
    --round-seconds "$ROUND_SECONDS" --questions "$QUESTIONS" &
  pids+=($!)
fi

"$PYTHON_BIN" -m skunk_server.server --host "$HOST" --port "$SKUNK_SERVER_PORT" \
  --cup-base-url "$CUP_BASE_URL" --team-token "$CUP_TEAM_TOKEN" \
  --reasoner "$REASONER" --concurrency "$CONCURRENCY" \
  --stream-dir "$SKUNK_STREAM_DIR" ${BLOCKING:+--human-blocking} &
pids+=($!)

"$PYTHON_BIN" -m skunk_server.web_app --host "$HOST" --port "$WEB_PORT" \
  --backend-url "$SKUNK_SERVER_URL" --stream-dir "$SKUNK_STREAM_DIR" &
pids+=($!)

# Tear all processes down on exit/Ctrl-C: SIGTERM (graceful uvicorn), then SIGKILL any survivor.
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

echo "Cup:        $CUP_BASE_URL $([[ -z "$EXTERNAL_CUP" ]] && echo '(local practice server)' || echo '(external)')"
echo "Console UI: http://${HOST}:${WEB_PORT}"
echo "Backend:    $SKUNK_SERVER_URL (localhost only)"
echo "Stream dir: $SKUNK_STREAM_DIR"
echo "Reasoner:   $REASONER   concurrency=$CONCURRENCY   round=${ROUND_SECONDS}s"
echo "Human:      transport=$([[ -n "$BLOCKING" ]] && echo blocking || echo optimistic)  gate VERIFY_EXTRACT=${SKUNK_HUMAN_VERIFY_EXTRACT:-0} FIGURE=${SKUNK_HUMAN_FIGURE:-0} LOOKUP=${SKUNK_HUMAN_LOOKUP:-0}"

# Block until any process exits, then the trap tears the rest down.
while :; do
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || exit 0; done
  sleep 1
done
