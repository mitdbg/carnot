#!/usr/bin/env bash
# Launch the local Skunk console: practice API + skunk server + web client.
#
# Configuration comes from the repo .env (skunk/.env): every entry there is
# exported so the reasoner subprocess sees the corpus + LLM settings
# (GEMINI_API_KEY, OFFICEQA_PDF_DIR, SKUNK_CHROMADB_DIR, SKUNK_RETRIEVER, the
# SKUNK_HUMAN_* gates, ...). Anything already set in your shell wins over the
# file, so you can override ad hoc, e.g.:
#   REASONER=dummy_agent:solve ./serve_local_console.sh        # no-LLM smoke test
#   SKUNK_HUMAN_VERIFY_EXTRACT=1 ./serve_local_console.sh      # try the HITL gates
#
# Launcher knobs (all optional, env or .env): REASONER, CONCURRENCY, PYTHON_BIN,
# SKUNK_CONSOLE_QUESTIONS, SKUNK_CONSOLE_HOST, and the three SKUNK_CONSOLE_*_PORT
# vars. Defaults are below. (Submission is manual + a deadline sweep that submits any
# still-READY answer just before the round closes — there is no auto-submit toggle.)
set -euo pipefail

cd "$(dirname "$0")"
HARNESS_DIR="$(pwd)"
SKUNK_DIR="$(cd .. && pwd)"

# --- Load skunk/.env, without clobbering vars already in the environment ---
ENV_FILE="${SKUNK_ENV_FILE:-$SKUNK_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                                  # tolerate CRLF
    [[ -z "$line" || "$line" == \#* ]] && continue        # skip blanks / comments
    line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # strip one layer of surrounding single/double quotes from the value
    if [[ "$val" == \"*\" || "$val" == \'*\' ]]; then val="${val:1:${#val}-2}"; fi
    # environment wins: only adopt the .env value when the var is unset (key is
    # validated above, so this eval cannot inject).
    if ! eval "[ -n \"\${$key+x}\" ]"; then
      export "$key=$val"
    fi
  done < "$ENV_FILE"
fi

# --- Launcher knobs (override via environment / .env) ---
HOST="${SKUNK_CONSOLE_HOST:-127.0.0.1}"
CUP_PORT="${SKUNK_CONSOLE_CUP_PORT:-8765}"
SKUNK_SERVER_PORT="${SKUNK_CONSOLE_SERVER_PORT:-8787}"
ROUND_SECONDS="${SKUNK_CONSOLE_ROUND_SECONDS:-3600}"
# Question-execution parallelism = questions released per round (always 15). Override
# with SKUNK_CONCURRENCY.
CONCURRENCY="${SKUNK_CONCURRENCY:-15}"
REASONER="${REASONER:-skunk_reasoner:solve}"
CUP_TEAM_TOKEN="${CUP_TEAM_TOKEN:-anything}"
QUESTIONS="${SKUNK_CONSOLE_QUESTIONS:-$SKUNK_DIR/officeqa-cup-kit-v0.1.5/practice_questions.json}"

# Python interpreter: explicit PYTHON_BIN, else the repo venv, else PATH python3.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$SKUNK_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$SKUNK_DIR/venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export CUP_BASE_URL="http://${HOST}:${CUP_PORT}"
export CUP_TEAM_TOKEN
export SKUNK_SERVER_URL="http://${HOST}:${SKUNK_SERVER_PORT}"

# Per-run trace dir: the reasoner dumps each question's event stream here (the UI path
# otherwise persists nothing — see skunk_reasoner._dump_console_trace). Set it to empty
# in the env to disable, or to a fixed path to override the timestamped default.
if [[ -z "${SKUNK_CONSOLE_TRACE_DIR+x}" ]]; then
  export SKUNK_CONSOLE_TRACE_DIR="$SKUNK_DIR/logs/console/$(date +%Y%m%d_%H%M%S)"
fi
# Create it up front (the reasoner also mkdirs lazily, but this makes the dir visible
# immediately and fails fast on a bad path). Skipped when set empty to disable tracing.
[[ -n "${SKUNK_CONSOLE_TRACE_DIR:-}" ]] && mkdir -p "$SKUNK_CONSOLE_TRACE_DIR"

# Relative paths in .env (e.g. SKUNK_CHROMADB_DIR=.chromadb) resolve against the
# process cwd, and the corpus caches live under the skunk repo dir — so run the
# python processes from there, with the harness modules importable on PYTHONPATH.
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$SKUNK_DIR"

"$PYTHON_BIN" "$HARNESS_DIR/practice_server.py" --host "$HOST" --port "$CUP_PORT" \
  --round-seconds "$ROUND_SECONDS" --questions "$QUESTIONS" &
practice_pid=$!

"$PYTHON_BIN" -m skunk_server.server --host "$HOST" --port "$SKUNK_SERVER_PORT" \
  --cup-base-url "$CUP_BASE_URL" --team-token "$CUP_TEAM_TOKEN" \
  --reasoner "$REASONER" --concurrency "$CONCURRENCY" &
server_pid=$!

# Tear both servers down on exit/Ctrl-C. SIGTERM first for a graceful uvicorn shutdown
# (it now has a 5s graceful-shutdown deadline), then SIGKILL any survivor — the reasoner
# offloads blocking tool/code work onto non-daemon thread-pool threads that can't be
# interrupted, so without the hard kill a wedged worker could still keep a server alive.
shutdown() {
  trap '' INT TERM                                          # don't re-enter while tearing down
  kill -TERM "$practice_pid" "$server_pid" 2>/dev/null || true
  for _ in $(seq 1 8); do
    kill -0 "$practice_pid" 2>/dev/null || kill -0 "$server_pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$practice_pid" "$server_pid" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

echo "Practice API: $CUP_BASE_URL"
echo "Console UI:   $SKUNK_SERVER_URL   (skunk_server serves the monitoring UI)"
echo "Reasoner:     $REASONER"
echo "Python:       $PYTHON_BIN"
echo "Questions:    $QUESTIONS"

# Block until either process exits, then the trap tears the other down.
# (Poll loop instead of `wait -n` so this works on macOS's stock bash 3.2.)
while kill -0 "$practice_pid" 2>/dev/null \
   && kill -0 "$server_pid" 2>/dev/null; do
  sleep 1
done
