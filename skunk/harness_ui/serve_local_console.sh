#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
CUP_PORT="8765"
SKUNK_SERVER_PORT="8787"
SKUNK_CLIENT_PORT="8790"
ROUND_SECONDS="3600"
QUESTIONS="questions/officeqa_2x5_questions.json"
CUP_TEAM_TOKEN="anything"
# REASONER="skunk_reasoner:solve"
# REASONER="dummy_agent:solve"
REASONER="cached_reasoner:solve"
CONCURRENCY="5"
AUTO_SUBMIT="${AUTO_SUBMIT:-false}"
PYTHON_BIN="${PYTHON_BIN:-/home/gerardo/.local/share/mamba/envs/carnot/bin/python}"

cd "$(dirname "$0")"
export CUP_BASE_URL="http://${HOST}:${CUP_PORT}"
export CUP_TEAM_TOKEN
export SKUNK_SERVER_URL="http://${HOST}:${SKUNK_SERVER_PORT}"
export SKUNK_PAGE_INDEX_DIR="${SKUNK_PAGE_INDEX_DIR:-$(cd .. && pwd)/cache/build_v3}"

"$PYTHON_BIN" practice_server.py --host "$HOST" --port "$CUP_PORT" \
  --round-seconds "$ROUND_SECONDS" --questions "$QUESTIONS" &
practice_pid=$!

auto_submit_flag="--no-auto-submit"
if [[ "$AUTO_SUBMIT" == "true" ]]; then
  auto_submit_flag="--auto-submit"
fi

"$PYTHON_BIN" -m skunk_server.server --host "$HOST" --port "$SKUNK_SERVER_PORT" \
  --cup-base-url "$CUP_BASE_URL" --team-token "$CUP_TEAM_TOKEN" \
  --reasoner "$REASONER" --concurrency "$CONCURRENCY" "$auto_submit_flag" &
server_pid=$!

"$PYTHON_BIN" -m skunk_client.client --host "$HOST" --port "$SKUNK_CLIENT_PORT" \
  --server-url "$SKUNK_SERVER_URL" &
client_pid=$!

trap 'kill "$practice_pid" "$server_pid" "$client_pid" 2>/dev/null || true' EXIT INT TERM

echo "Practice API: $CUP_BASE_URL"
echo "Skunk Server: $SKUNK_SERVER_URL"
echo "Skunk Client: http://${HOST}:${SKUNK_CLIENT_PORT}"
wait -n "$practice_pid" "$server_pid" "$client_pid"
