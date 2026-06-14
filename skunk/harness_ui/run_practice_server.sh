#!/usr/bin/env bash
# Run the OfficeQA Cup PRACTICE SERVER — the question/scoring server.
#
# In the real competition this role is played by Databricks (or another third party): they host the
# Cup server, release rounds of questions, and score submissions. You do NOT run this during the
# real Cup — you point run_solution.sh at their --cup-base-url instead. This script exists so you
# can stand up an equivalent server LOCALLY and rehearse the full loop end to end.
#
# It serves a questions JSON over HTTP and scores submissions against each question's
# canonical_answer. It needs no corpus, LLM keys, or GPU — only a questions file (generate one with
# make_practice_questions.py; the held-out test UIDs are excluded automatically). Start our agent +
# UI from a second terminal with ./run_solution.sh.
#
# Knobs (flag or env): --host/SKUNK_CONSOLE_HOST, --port/SKUNK_CONSOLE_CUP_PORT,
# --round-seconds/SKUNK_CONSOLE_ROUND_SECONDS, --questions/SKUNK_CONSOLE_QUESTIONS.
set -euo pipefail

cd "$(dirname "$0")"
HARNESS_DIR="$(pwd)"
SKUNK_DIR="$(cd .. && pwd)"

# --- Load skunk/.env without clobbering vars already in the environment ---
# Mostly for shared port config (SKUNK_CONSOLE_*) so both launchers agree on host/ports.
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
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)          export SKUNK_CONSOLE_HOST="$2"; shift 2 ;;
    --port)          export SKUNK_CONSOLE_CUP_PORT="$2"; shift 2 ;;
    --round-seconds) export SKUNK_CONSOLE_ROUND_SECONDS="$2"; shift 2 ;;
    --questions)     export SKUNK_CONSOLE_QUESTIONS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HOST="${SKUNK_CONSOLE_HOST:-127.0.0.1}"
CUP_PORT="${SKUNK_CONSOLE_CUP_PORT:-8765}"
ROUND_SECONDS="${SKUNK_CONSOLE_ROUND_SECONDS:-3600}"

# Questions: explicit --questions/SKUNK_CONSOLE_QUESTIONS, else the bundled practice set.
QUESTIONS="${SKUNK_CONSOLE_QUESTIONS:-$HARNESS_DIR/questions/practice_questions.json}"
if [[ ! -f "$QUESTIONS" ]]; then
  echo "ERROR: questions file not found: $QUESTIONS" >&2
  echo "       generate one with: python3 $HARNESS_DIR/make_practice_questions.py --num 8 --out /tmp/q.json" >&2
  echo "       then run:          $0 --questions /tmp/q.json" >&2
  exit 2
fi

# Python interpreter: explicit PYTHON_BIN, else the repo venv, else PATH python3.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$SKUNK_DIR/venv/bin/python" ]]; then PYTHON_BIN="$SKUNK_DIR/venv/bin/python"
  else PYTHON_BIN="$(command -v python3)"; fi
fi

# cup_kit lives under the harness dir; make it importable and run from the skunk dir for parity
# with run_solution.sh (relative paths resolve the same way).
export PYTHONPATH="${HARNESS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$SKUNK_DIR"

echo "Practice server: http://${HOST}:${CUP_PORT}   (the role Databricks plays in the real Cup)"
echo "Questions:       $QUESTIONS"
echo "Round length:    ${ROUND_SECONDS}s"
echo "Python:          $PYTHON_BIN"
echo "Attach our solution from another terminal:  ./run_solution.sh"

# Single foreground process — Ctrl-C kills it directly, no teardown trap needed.
exec "$PYTHON_BIN" "$HARNESS_DIR/practice_server.py" --host "$HOST" --port "$CUP_PORT" \
  --round-seconds "$ROUND_SECONDS" --questions "$QUESTIONS"
