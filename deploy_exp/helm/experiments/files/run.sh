#!/usr/bin/env bash
# main container: wait for the chroma sidecar, run the cell's argv from /app/qatfd, keep /results mirrored
# to the results bucket while it runs, and leave a status json behind. Exit code == the cell's exit code, so
# the Job's podFailurePolicy sees real failures.
set -uo pipefail

CELL="python3 /scripts/cell.py"
LABEL="$($CELL label)"
COLLECTION="$($CELL collection)"
readarray -d '' ARGV < <($CELL --argv)
# EXTRA_OVERRIDES is newline-separated (from values runner.extraOverrides)
while IFS= read -r ov; do [[ -n "$ov" ]] && ARGV+=("$ov"); done <<< "${EXTRA_OVERRIDES:-}"

RESULTS_URI="s3://$RESULTS_BUCKET/${RESULTS_PREFIX%/}"
STATUS_URI="s3://$RESULTS_BUCKET/${JOBS_PREFIX%/}/${RELEASE_NAME}/$(printf '%04d' "$JOB_COMPLETION_INDEX")_${LABEL}.json"
CHROMA="http://${SKUNK_CHROMA_SERVER_HOST}:${SKUNK_CHROMA_SERVER_PORT}"
NODE="${NODE_NAME:-unknown}"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; t0=$(date +%s)

status() {  # status <state> <exit_code>
    python3 - "$1" "$2" <<'PY' > /tmp/status.json
import json, os, sys, datetime
print(json.dumps({
    "release": os.environ["RELEASE_NAME"], "index": int(os.environ["JOB_COMPLETION_INDEX"]),
    "label": os.environ["LABEL"], "state": sys.argv[1], "exit_code": int(sys.argv[2]),
    "node": os.environ["NODE"], "pod": os.environ.get("POD_NAME", ""), "started_at": os.environ["STARTED"],
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "elapsed_s": int(os.environ["ELAPSED"]),
}, indent=1))
PY
    aws s3 cp /tmp/status.json "$STATUS_URI" --only-show-errors || true
}
export RELEASE_NAME LABEL NODE ELAPSED=0 STARTED="$started"

sync_results() { aws s3 sync /results "$RESULTS_URI" --only-show-errors || echo "[run] WARN: results sync failed"; }

echo "[run] cell $JOB_COMPLETION_INDEX ($LABEL) on $NODE"
status starting 0

# --- wait for chroma: heartbeat, then the collection itself (the sidecar's startupProbe already gates on the
# heartbeat; the collection check catches a store that failed to pull or a wrong collection name early)
until curl -sf -m 5 "$CHROMA/api/v2/heartbeat" >/dev/null; do sleep 3; done
for _ in $(seq 1 200); do
    if python3 - "$COLLECTION" <<'PY' 2>/dev/null; then break; fi
import sys, os, chromadb
c = chromadb.HttpClient(host=os.environ["SKUNK_CHROMA_SERVER_HOST"], port=int(os.environ["SKUNK_CHROMA_SERVER_PORT"]))
print("[run] collection", sys.argv[1], "rows:", c.get_collection(sys.argv[1]).count())
PY
    sleep 3
done

# --- run the cell, mirroring /results every SYNC_INTERVAL seconds in the background
mkdir -p /results
echo "[run] argv: ${ARGV[*]}"
( while sleep "${SYNC_INTERVAL:-60}"; do sync_results; done ) &
SYNC_PID=$!
cd /app/qatfd
"${ARGV[@]}"
rc=$?
kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null

ELAPSED=$(( $(date +%s) - t0 ))
echo "[run] cell exited with $rc after ${ELAPSED}s; final results sync"
sync_results
if [[ $rc -eq 0 ]]; then status done 0; else status failed "$rc"; fi
exit "$rc"
