#!/usr/bin/env bash
# Chroma server management shared by the sweep drivers (source this file; do not execute it).
#
# chroma 1.5.x never releases the HNSW index (RAM), file handles, or segment directory (disk) of a deleted
# collection, so a long sweep that creates and drops many working-set collections leaks a few GB of server
# RSS per run until the kernel OOM-kills the server (2026-09-23: 47 GB RSS on a 61 GB box, after which every
# later run failed in 2s with "connection refused"). With MANAGE_CHROMA=1 the driver owns the server: it
# starts skunk/scripts/run_chroma_server.sh $BASE_COLLECTION if none answers, checks the server's RSS before
# every run (chroma_ensure), and restarts it (stop, start, warm the corpus) when RSS exceeds CHROMA_MAX_RSS_GB,
# every CHROMA_RESTART_EVERY runs (0 = off), or when it stops answering. A server started by hand on the same
# host:port is adopted, i.e. it is the one that gets stopped on the first restart. The driver stops the server
# it started when it exits (CHROMA_KEEP_ON_EXIT=1 keeps it). MANAGE_CHROMA=0 only waits (CHROMA_WAIT_MIN) for
# a hand-run server to come back before aborting the whole sweep. The disk side of the leak is handled by
# scripts/clean_chroma_orphans.py (call it after every run).
#
# The sourcing script must define: log(), CHROMA_HOST, CHROMA_PORT, CHROMA_STORE, CHROMA_LOG, CHROMA_SERVER_SCRIPT,
# BASE_COLLECTION, LOG_DIR, MANAGE_CHROMA, CHROMA_MAX_RSS_GB, CHROMA_RESTART_EVERY, CHROMA_KEEP_ON_EXIT, CHROMA_WAIT_MIN.
# It should call `chroma_ensure` before each run and bump `runs_since_restart` after each run.

# ---------------------------------------------------------------------------
# chroma server management (see header)
# ---------------------------------------------------------------------------
chroma_alive() { curl -sf -m 5 "http://$CHROMA_HOST:$CHROMA_PORT/api/v2/heartbeat" >/dev/null; }

# pid of the `chroma run` process serving CHROMA_HOST:CHROMA_PORT (the [c] keeps pgrep from matching itself)
chroma_pid() { pgrep -f "[c]hroma run --host $CHROMA_HOST --port $CHROMA_PORT( |$)" | head -n 1; }
chroma_rss_gb() {
    local pid; pid="$(chroma_pid)"
    if [[ -n "$pid" && -r "/proc/$pid/status" ]]; then awk '/^VmRSS:/ {printf "%.1f", $2 / 1048576}' "/proc/$pid/status"; else echo 0; fi
}
chroma_owned=0
runs_since_restart=0

chroma_stop() {
    local pid; pid="$(chroma_pid)"
    [[ -n "$pid" ]] || return 0
    log "CHROMA stopping server pid $pid (rss=$(chroma_rss_gb) GB)"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then log "CHROMA server ignored TERM; sending KILL"; kill -KILL "$pid" 2>/dev/null || true; sleep 2; fi
}

chroma_start() {
    [[ -x "$CHROMA_SERVER_SCRIPT" ]] || { log "ABORT: chroma server script not found at $CHROMA_SERVER_SCRIPT"; exit 1; }
    [[ -f "$CHROMA_STORE/chroma.sqlite3" ]] || { log "ABORT: no chroma store at $CHROMA_STORE (set CHROMA_STORE)"; exit 1; }
    mkdir -p "$LOG_DIR"
    local before=0; [[ -f "$CHROMA_LOG" ]] && before="$(wc -l < "$CHROMA_LOG")"   # all lines: the offset feeds tail -n +N
    log "CHROMA starting server on $CHROMA_HOST:$CHROMA_PORT over $CHROMA_STORE, warming $BASE_COLLECTION (log: $CHROMA_LOG)"
    (
        cd "$(dirname "$CHROMA_SERVER_SCRIPT")/.." || exit 1
        SKUNK_CHROMADB_DIR="$CHROMA_STORE" SKUNK_CHROMA_SERVER_HOST="$CHROMA_HOST" SKUNK_CHROMA_SERVER_PORT="$CHROMA_PORT" \
            setsid nohup "$CHROMA_SERVER_SCRIPT" "$BASE_COLLECTION" >> "$CHROMA_LOG" 2>&1 < /dev/null &
    )
    chroma_owned=1
    local up=0
    for _ in $(seq 1 180); do if chroma_alive; then up=1; break; fi; sleep 1; done
    [[ $up -eq 1 ]] || { log "ABORT: chroma server did not answer within 180s (see $CHROMA_LOG)"; exit 1; }
    # wait for the warm-up (HNSW load of the corpus) so the first run does not pay it inside its timings
    local warm=0
    for _ in $(seq 1 900); do
        if tail -n +"$(( before + 1 ))" "$CHROMA_LOG" 2>/dev/null | grep -q "ChromaDB server ready"; then warm=1; break; fi
        chroma_alive || { log "ABORT: chroma server died during warm-up (see $CHROMA_LOG)"; exit 1; }
        sleep 1
    done
    [[ $warm -eq 1 ]] || log "WARN: chroma warm-up still running after 900s; continuing"
    # the warm-up is best-effort inside run_chroma_server.sh, but a corpus that cannot be loaded ("Error loading
    # hnsw index": the index does not fit in RAM, e.g. the 9 GB officeqa index on the 7.5 GB dev-shape box) would
    # fail every run of the sweep, so stop here instead
    local warm_err; warm_err="$(tail -n +"$(( before + 1 ))" "$CHROMA_LOG" 2>/dev/null | grep -m1 "\[warm\] could not warm")"
    if [[ -n "$warm_err" ]]; then
        log "ABORT: the chroma server cannot load $BASE_COLLECTION ($warm_err); free RAM: $(free -g | awk '/^Mem:/ {print $7}') GB"
        exit 1
    fi
    log "CHROMA server up (pid $(chroma_pid), rss=$(chroma_rss_gb) GB)"
}

# called before every run: (re)start the server when it is down, too big, or due for a scheduled restart
chroma_ensure() {
    if [[ "$MANAGE_CHROMA" != "1" ]]; then
        chroma_alive && return 0
        log "chroma server at $CHROMA_HOST:$CHROMA_PORT is not answering; waiting up to $CHROMA_WAIT_MIN min for it (MANAGE_CHROMA=0)"
        for _ in $(seq 1 "$(( CHROMA_WAIT_MIN * 6 ))"); do sleep 10; chroma_alive && { log "chroma server is back"; return 0; }; done
        log "ABORT: chroma server still down after $CHROMA_WAIT_MIN min"; exit 1
    fi
    local reason=""
    if ! chroma_alive; then
        reason="not answering"
    else
        local rss; rss="$(chroma_rss_gb)"
        if awk -v r="$rss" -v m="$CHROMA_MAX_RSS_GB" 'BEGIN { exit !(r + 0 > m + 0) }'; then reason="rss ${rss} GB > CHROMA_MAX_RSS_GB=$CHROMA_MAX_RSS_GB"; fi
        if [[ "$CHROMA_RESTART_EVERY" -gt 0 && "$runs_since_restart" -ge "$CHROMA_RESTART_EVERY" ]]; then reason="${reason:+$reason; }$runs_since_restart runs since last restart"; fi
    fi
    [[ -n "$reason" ]] || return 0
    log "CHROMA restart: $reason"
    chroma_stop
    chroma_start
    runs_since_restart=0
}

chroma_on_exit() {
    if [[ "$chroma_owned" == "1" && "$CHROMA_KEEP_ON_EXIT" != "1" ]]; then
        log "CHROMA stopping the server this sweep started (CHROMA_KEEP_ON_EXIT=1 to keep it)"
        chroma_stop
    fi
}
trap chroma_on_exit EXIT
