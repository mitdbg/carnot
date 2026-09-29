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
    aws s3 cp /tmp/status.json "$STATUS_URI" --only-show-errors \
        || { sleep 20; aws s3 cp /tmp/status.json "$STATUS_URI" --only-show-errors; } || true
}
export RELEASE_NAME LABEL NODE ELAPSED=0 STARTED="$started"

sync_results() { aws s3 sync /results "$RESULTS_URI" --only-show-errors || echo "[run] WARN: results sync failed"; }
# the last sync must not be lost to a transient network error (neighbouring pods' pulls saturate the node)
final_sync() {
    local attempt
    for attempt in 1 2 3 4 5 6; do
        aws s3 sync /results "$RESULTS_URI" --only-show-errors && return 0
        echo "[run] WARN: final results sync attempt $attempt failed; retrying in $((attempt * 20))s"
        sleep $((attempt * 20))
    done
    echo "[run] ERROR: final results sync failed; results remain on the pod's /results only"
    return 1
}

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

# --- codex sandbox (codex cells only; CODEX_SANDBOX_UID is set by the chart when codexSandbox.enabled): the codex
# subprocess runs as that uid (systems.sandbox_uid), which may read nothing the runner holds (the benchmark files
# incl. gold answers, the raw corpus, the chroma store, /results) and, via lockdown.sh's iptables rules, may only
# reach the MCP server and this egress proxy, which tunnels to the model provider alone.
mkdir -p /results
if [[ -n "${CODEX_SANDBOX_UID:-}" ]]; then
    chown root:root /data /results && chmod 700 /data /results
    mkdir -p /codex/home /codex/work
    chown -R "$CODEX_SANDBOX_UID:$CODEX_SANDBOX_UID" /codex && chmod 700 /codex
    python3 /scripts/egress_proxy.py --port "$CODEX_PROXY_PORT" --allow $CODEX_EGRESS_ALLOW &
    PROXY_PID=$!
    for _ in $(seq 1 50); do (exec 3<>"/dev/tcp/127.0.0.1/$CODEX_PROXY_PORT") 2>/dev/null && break; sleep 0.2; done
    # fail closed: prove the fence from the sandbox uid's side before any codex process exists
    if ! python3 /scripts/sandbox_check.py --uid "$CODEX_SANDBOX_UID" --proxy-port "$CODEX_PROXY_PORT" --mcp-port "$CODEX_MCP_PORT" --allow-target "${CODEX_EGRESS_ALLOW%% *}:443"; then
        echo "[run] ERROR: codex sandbox self-check failed; refusing to run the cell"
        ELAPSED=$(( $(date +%s) - t0 )); status failed 78
        exit 78   # EX_CONFIG
    fi
    # a shell cell whose codex shell sandbox (bubblewrap, workspace-write) cannot start would answer with every shell
    # command failing: probe it as the sandbox uid and refuse the cell instead (unless the cell opted out of bwrap)
    # (`codex sandbox` defaults to a READ-ONLY policy: the probe must ask for workspace-write, as `codex exec` does)
    if printf '%s\n' "${ARGV[@]}" | grep -qx 'systems.codex_shell=true' \
        && ! printf '%s\n' "${ARGV[@]}" | grep -qx 'systems.shell_sandbox_mode=danger-full-access'; then
        if ! (cd /codex/work && setpriv --reuid="$CODEX_SANDBOX_UID" --regid="$CODEX_SANDBOX_UID" --clear-groups \
                env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/codex/home CODEX_HOME=/codex/home \
                codex sandbox -c 'sandbox_mode="workspace-write"' -- bash -c 'touch /codex/work/.bwrap_probe && rm /codex/work/.bwrap_probe'); then
            echo "[run] ERROR: codex's shell sandbox (bubblewrap) cannot start in this pod; refusing a shell cell."
            echo "[run]        Fix the node (unprivileged user namespaces) or pass systems.shell_sandbox_mode=danger-full-access"
            echo "[run]        (the pod's uid + iptables fence still applies)."
            ELAPSED=$(( $(date +%s) - t0 )); status failed 78
            exit 78
        fi
        echo "[run] codex shell sandbox (bubblewrap) ok"
    fi
fi

# --- run the cell, mirroring /results every SYNC_INTERVAL seconds in the background
echo "[run] argv: ${ARGV[*]}"
( while sleep "${SYNC_INTERVAL:-60}"; do sync_results; done ) &
SYNC_PID=$!
cd /app/qatfd
"${ARGV[@]}"
rc=$?
kill "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null

# codex sandbox: its home (config, sessions, state db: the model / thread records) and work dir (the shell
# scenarios' notes / AGENTS.md; what the cheat check scans) live outside /results; file them into the run dir
if [[ -n "${CODEX_SANDBOX_UID:-}" ]]; then
    kill "${PROXY_PID:-}" 2>/dev/null
    for run_dir in /results/*/*/"${LABEL}"_[0-9]*/; do
        [[ -d "$run_dir" ]] || continue
        mkdir -p "$run_dir/codex_home" "$run_dir/codex_scratch"
        tar -C /codex/home --exclude=./.tmp --exclude=./tmp -cf - . | tar -C "$run_dir/codex_home" -xf - || echo "[run] WARN: codex_home copy failed"
        tar -C /codex/work -cf - . | tar -C "$run_dir/codex_scratch" -xf - || echo "[run] WARN: codex_scratch copy failed"
        echo "[run] filed /codex/{home,work} into $run_dir"
    done
fi

ELAPSED=$(( $(date +%s) - t0 ))
echo "[run] cell exited with $rc after ${ELAPSED}s; final results sync"
final_sync || { [[ $rc -eq 0 ]] && rc=75; }   # EX_TEMPFAIL: the cell ran but its results did not land
if [[ $rc -eq 0 ]]; then status done 0; else status failed "$rc"; fi
exit "$rc"
