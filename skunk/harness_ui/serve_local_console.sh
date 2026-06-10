#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
CUP_PORT="8765"
CONSOLE_PORT="8787"
ROUND_SECONDS="3600"
QUESTIONS="questions/officeqa_2x5_questions.json"
CUP_TEAM_TOKEN="anything"
REASONER="${REASONER:-skunk_reasoner:solve}"
# REASONER="dummy_agent:solve"
CONCURRENCY="3"

cd "$(dirname "$0")"
export CUP_BASE_URL="http://${HOST}:${CUP_PORT}"
export CUP_TEAM_TOKEN

python practice_server.py --host "$HOST" --port "$CUP_PORT" \
  --round-seconds "$ROUND_SECONDS" --questions "$QUESTIONS" &
practice_pid=$!

python competition_console.py --host "$HOST" --port "$CONSOLE_PORT" \
  --cup-base-url "$CUP_BASE_URL" --team-token "$CUP_TEAM_TOKEN" \
  --reasoner "$REASONER" --concurrency "$CONCURRENCY" &
console_pid=$!

trap 'kill "$practice_pid" "$console_pid" 2>/dev/null || true' EXIT INT TERM

echo "Practice API: $CUP_BASE_URL"
echo "Console UI:   http://${HOST}:${CONSOLE_PORT}"
wait -n "$practice_pid" "$console_pid"
