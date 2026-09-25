#!/usr/bin/env bash
# Upper-bound experiment for bootstrap / enrich: for X in XS, the SAME X OfficeQA dev questions are answered
# twice, in sequence, by one long-lived search_agent system, under the cells below (default: the five
# marked *; the three ws+agent combinations stay available via CELLS=...). The second pass is
# the ideal case for the idea (the exact questions the collections were built / curated for come back), so
# the pass-2 vs pass-1 delta bounds what bootstrap / enrich / trajectory working sets can buy.
#
#   cell             trajectory ws   hide ws   bootstrap   enrich    (search agent prompt)
# * exp1_baseline    off             yes       off         off       base collection only (working_set_collection_off=true)
# * exp2_ws          on              NO        off         off       base + earlier questions' trajectory working sets
# * exp3_bs          off             yes       on          off       base + bootstrap collections
#   exp4_bs_ws       on              yes       on          off       base + bootstrap collections
# * exp5_en          off             yes       off         on        base + enrich collections
#   exp6_en_ws       on              yes       off         on        base + enrich collections
# * exp7_bs_en       off             yes       on          on        base + bootstrap / enrich collections
#   exp8_bs_en_ws    on              yes       on          on        base + bootstrap / enrich collections
#
# "trajectory ws" = whether each search agent persists the chunks it searched / fetched as a docs_for_q*
# collection that later agents (and the EnrichAgent) can see (+ub.trajectory_working_sets, see
# scripts/bootstrap_enrich_upper_bound.py). "hide ws" = systems.retrieve.hide_and_clear_working_sets: with it
# on (the config default) later search agents never see those trajectory working sets, only the base collection
# plus what the Bootstrap / Enrich agents curated. exp2_ws is the one cell that turns it off — otherwise it
# would be indistinguishable from exp1_baseline. Bootstrap = enrich_working_sets=before (the BootstrapAgent
# builds collections once, before the first question); enrich = enrich_working_sets=after (the EnrichAgent
# curates the collections after every ENRICH_BATCH questions; default X, i.e. once at the end of each pass,
# so pass 2 runs against collections curated for exactly its questions); both = enrich_working_sets=both.
# Every cell but exp1 keeps working_set_collection_off=false so the agent is told about the managed
# collections; exp1 uses the base-only prompt (BASELINE_COLLECTION_OFF=false to give it the same prompt as
# the other cells, in which case it is told about smaller collections and sees none).
#
# Questions: the first X of a seeded permutation of the dev split, so X=1 ⊂ X=5 ⊂ X=10 and every cell at one
# (X, seed) answers the same questions in the same order. Each (X, cell) is run once per seed in SEEDS
# (default 0 1 2) for some diversity in the questions drawn. QIDS=... runs an explicit list instead.
#
# Isolation: every collection on the chroma server except the corpus collections (BASE_COLLECTION plus
# KEEP_COLLECTIONS) is deleted before each run. Runs are serial (collections live on the shared server).
# The corpus itself is reset as well: nothing stops the Bootstrap / Enrich agents from running map /
# semantic_map over BASE_COLLECTION, which writes new metadata keys onto every chunk, so before each run
# engaging-scripts/reset_corpus_metadata.py strips every chunk key other than doc_id / chunk_id / element_id
# from it — ONLY when BASE_COLLECTION is one of the -v1 copies (the script's explicit allowlist; the default
# here). An original base collection is only checked, and the run is skipped if it turns out to be polluted.
#
# Chroma server: chroma 1.5.x never releases the HNSW index (RAM), file handles, or segment directory (disk)
# of a deleted collection, so a long sweep that creates and drops many working-set collections leaks a few
# GB of server RSS per run until the kernel OOM-kills the server (2026-09-23: 47 GB RSS on a 61 GB box, and
# every later cell failed in 2s with "connection refused"). With MANAGE_CHROMA=1 (default) the driver owns
# the server: it starts skunk/scripts/run_chroma_server.sh $BASE_COLLECTION if none answers, checks the
# server's RSS before every run, and restarts it (stop, start, warm the corpus) when RSS exceeds
# CHROMA_MAX_RSS_GB, every CHROMA_RESTART_EVERY runs (0 = off), or when it stops answering. A server that was
# started by hand on the same host:port is adopted, i.e. it is the one that gets stopped on the first restart.
# The driver stops the server it started when it exits (CHROMA_KEEP_ON_EXIT=1 keeps it). MANAGE_CHROMA=0
# only waits (CHROMA_WAIT_MIN) for a hand-run server to come back before aborting the whole sweep. The disk
# side of the leak is handled by scripts/clean_chroma_orphans.py after every run (CLEAN_ORPHANS=1).
#
# Run dirs: $RESULTS_ROOT/$BENCHMARK/search_agent_ub/ub_x<X>_<cell>_s<seed>_<timestamp>/ with results.jsonl,
# report.csv (a `pass` column ahead of the stock columns), summary.json (per-pass means + per-question
# pass-over-pass), traces/pass<p>/, collections_after_pass<p>.json. A cell with a complete run dir (rows ==
# X * PASSES) is skipped, so the sweep can be relaunched after a failure.
#
# Usage (from anywhere; use tmux/nohup):
#   scripts/run_bootstrap_enrich_upper_bound.sh 2>&1 | tee ub_sweep.out        # X = 1 5 10, all 8 cells
#   XS="1" CELLS="exp1_baseline exp3_bs" SEEDS="0" ...                          # subsets (any of the 8 cells)
#   QIDS=UID0001 XS=1 RUN_PREFIX=smoke ...                                      # smoke test on one question
#   DRY_RUN=1 ...                                                               # print the commands only
#   MANAGE_CHROMA=0 ...                                                         # use (and never restart) a hand-run server
#   MODEL=... AGENT_MODEL=... MAP_MODEL=... PASSES=... ENRICH_BATCH=... EXTRA_OVERRIDES="..." ...
#
# Models: MODEL runs the search agent, COMPUTE_MODEL the shared compute answerer, AGENT_MODEL the Bootstrap /
# Enrich agents' own steps, MAP_MODEL their semantic_map judge (same defaults as run_collection_agent_ablation.sh).
set -uo pipefail

QATFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/carnot/skunk/venv/bin/python3}"
BENCHMARK="${BENCHMARK:-officeqa}"
SPLIT="${SPLIT:-dev}"
MODEL="${MODEL:-openai/gpt-5.6-luna}"                   # search agent model
COMPUTE_MODEL="${COMPUTE_MODEL:-openai/gpt-5.6-terra}"  # the shared compute answerer's model
AGENT_MODEL="${AGENT_MODEL:-openai/gpt-5.6-terra}"       # Bootstrap / Enrich agents' own model
MAP_MODEL="${MAP_MODEL:-openai/gpt-5.6-luna}"            # semantic_map judge model (bootstrap / enrich)
BASE_COLLECTION="${BASE_COLLECTION:-officeqa-qwen-8b-v1}"  # the v1 (doc_id/chunk_id/element_id-only) corpus copy
KEEP_COLLECTIONS="${KEEP_COLLECTIONS:-officeqa-qwen-8b officeqa-qwen-8b-v1}"  # never deleted by the wipe
XS="${XS:-1 5 10}"                      # values of X (questions per pass)
CELLS="${CELLS:-exp1_baseline exp2_ws exp3_bs exp5_en exp7_bs_en}"  # see the table in the header for all 8
SEEDS="${SEEDS:-0 1 2}"                 # seeds of the dev-split permutation the X questions are the head of; one run per seed
PASSES="${PASSES:-2}"                   # times the X-question sequence is answered
ENRICH_BATCH="${ENRICH_BATCH:-}"        # EnrichAgent cadence in questions; empty = X (once per pass)
QIDS="${QIDS:-}"                        # comma-separated explicit qids (overrides X / SEED; XS is then a label only)
BASELINE_COLLECTION_OFF="${BASELINE_COLLECTION_OFF:-true}"  # exp1's working_set_collection_off (see header)
RUN_PREFIX="${RUN_PREFIX:-ub}"          # run labels are ${RUN_PREFIX}_x<X>_<cell>_s<seed>
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # extra hydra overrides appended to every run
RESULTS_ROOT="${RESULTS_ROOT:-$QATFD_DIR/results}"
RUN_ROOT="$RESULTS_ROOT/$BENCHMARK/search_agent_ub"
LOG_DIR="$RUN_ROOT/logs"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
CHROMA_STORE="${CHROMA_STORE:-$QATFD_DIR/benchmarks/officeqa/chromadb}"  # on-disk store behind the server (orphan cleanup)
CLEAN_ORPHANS="${CLEAN_ORPHANS:-1}"  # 1 = after every run, delete HNSW segment dirs that chroma left behind for dropped collections
MANAGE_CHROMA="${MANAGE_CHROMA:-1}"     # 1 = driver starts / restarts the chroma server itself (see header)
CHROMA_MAX_RSS_GB="${CHROMA_MAX_RSS_GB:-30}"       # restart the server before a run once its RSS exceeds this
CHROMA_RESTART_EVERY="${CHROMA_RESTART_EVERY:-0}"  # also restart every N runs (0 = only on RSS / liveness)
CHROMA_KEEP_ON_EXIT="${CHROMA_KEEP_ON_EXIT:-0}"    # 1 = leave a driver-started server running when the sweep ends
CHROMA_WAIT_MIN="${CHROMA_WAIT_MIN:-30}"           # MANAGE_CHROMA=0: minutes to wait for a down server before aborting
CHROMA_SERVER_SCRIPT="${CHROMA_SERVER_SCRIPT:-$QATFD_DIR/../skunk/scripts/run_chroma_server.sh}"
CHROMA_LOG="$LOG_DIR/chroma_server.log"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
has_openrouter_key() {
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || grep -q '^OPENROUTER_API_KEY=' "$QATFD_DIR/../skunk/.env" 2>/dev/null
}

# ---------------------------------------------------------------------------
# chroma server management: scripts/lib/chroma_manage.sh (chroma_ensure / chroma_start / chroma_stop, see its header)
# ---------------------------------------------------------------------------
source "$QATFD_DIR/scripts/lib/chroma_manage.sh"

# strip agent-written chunk metadata (a map / semantic_map over the corpus) from BASE_COLLECTION: a -v1
# copy is stripped, any other known collection is only checked (see engaging-scripts/reset_corpus_metadata.py)
reset_corpus_metadata() {
    "$PYTHON" engaging-scripts/reset_corpus_metadata.py --collection "$BASE_COLLECTION" --server "$CHROMA_HOST:$CHROMA_PORT"
}

preflight() {
    local ok=1
    [[ -x "$PYTHON" ]] || { log "ABORT: python not found at $PYTHON"; ok=0; }
    has_openrouter_key || { log "ABORT: OPENROUTER_API_KEY is unset (env or skunk/.env)"; ok=0; }
    if ! chroma_alive; then
        if [[ "$MANAGE_CHROMA" == "1" ]]; then
            log "no chroma server answering at $CHROMA_HOST:$CHROMA_PORT; starting one (MANAGE_CHROMA=1)"
            chroma_start
        else
            log "ABORT: no chroma server answering at $CHROMA_HOST:$CHROMA_PORT"; ok=0
        fi
    fi
    # another search-agent run would share (and have its collections wiped by) this sweep
    if ps -eo args | grep -E '^[^ ]*python[0-9.]* (-m qatfd\.runner|[^ ]*bootstrap_enrich_upper_bound\.py)' | grep -Eq 'systems=search_agent|bootstrap_enrich_upper_bound'; then
        log "ABORT: a search_agent run is already in progress (collections are shared on the chroma server)"; ok=0
    fi
    [[ $ok -eq 1 ]] || exit 1
}

# ---------------------------------------------------------------------------
# cells: trajectory_working_sets | working_set_collection_off | enrich_working_sets | hide_and_clear_working_sets
# ---------------------------------------------------------------------------
cell_settings() {
    case "$1" in
        exp1_baseline) echo "false $BASELINE_COLLECTION_OFF null true" ;;
        exp2_ws)       echo "true  false null   false" ;;
        exp3_bs)       echo "false false before true" ;;
        exp4_bs_ws)    echo "true  false before true" ;;
        exp5_en)       echo "false false after  true" ;;
        exp6_en_ws)    echo "true  false after  true" ;;
        exp7_bs_en)    echo "false false both   true" ;;
        exp8_bs_en_ws) echo "true  false both   true" ;;
        *) log "ABORT: unknown cell '$1'"; exit 1 ;;
    esac
}

result_rows() { if [[ -f "$1" ]]; then grep -c . "$1" || true; else echo 0; fi; }
find_run_dir() { ls -d "$RUN_ROOT/${1}_"[0-9]* 2>/dev/null | sort | tail -n 1; }

# the search agent's fixed tool configuration (= the baseline sweeps' `seq` scenario, no semantic filter)
SYSTEM_OVERRIDES="systems=search_agent inference.llm_model=$MODEL systems.compute.llm_model=$COMPUTE_MODEL \
systems.retrieve.include_search_corpus=true systems.retrieve.include_grep_corpus=true systems.retrieve.include_semantic_filter=false \
systems.retrieve.id_tracking_off=false systems.retrieve.fetch_related_working_sets=true \
systems.retrieve.bootstrap_config.llm_model=$AGENT_MODEL systems.retrieve.bootstrap_config.semantic_map_llm_model=$MAP_MODEL \
systems.retrieve.enrich_config.llm_model=$AGENT_MODEL systems.retrieve.enrich_config.semantic_map_llm_model=$MAP_MODEL \
benchmarks=$BENCHMARK benchmarks.collection_name=$BASE_COLLECTION \
benchmarks.chroma_server_host=$CHROMA_HOST benchmarks.chroma_server_port=$CHROMA_PORT \
experiments.split=$SPLIT experiments.run_mode=sequential experiments.workers=1"
[[ "$RESULTS_ROOT" == "$QATFD_DIR/results" ]] || SYSTEM_OVERRIDES="$SYSTEM_OVERRIDES results_root=$RESULTS_ROOT"

keep_list="$(echo $BASE_COLLECTION $KEEP_COLLECTIONS | tr ' ' '\n' | sort -u | paste -sd,)"

# ---------------------------------------------------------------------------
# main loop: X-major, then seed, then cell — every cell at a small X finishes before the expensive X=10
# cells start, and within one X a complete 8-cell comparison for one seed lands before the next seed begins
# ---------------------------------------------------------------------------
[[ "$DRY_RUN" == "1" ]] || preflight
mkdir -p "$LOG_DIR"
cd "$QATFD_DIR"
log "benchmark=$BENCHMARK collection=$BASE_COLLECTION split=$SPLIT model=$MODEL compute_model=$COMPUTE_MODEL agent_model=$AGENT_MODEL map_model=$MAP_MODEL xs='$XS' cells='$CELLS' seeds='$SEEDS' passes=$PASSES enrich_batch=${ENRICH_BATCH:-X}${QIDS:+ qids=$QIDS} run_root=$RUN_ROOT"

declare -a summary=()
for X in $XS; do
  for SEED in $SEEDS; do
    for cell in $CELLS; do
        read -r traj coll_off mode hide <<< "$(cell_settings "$cell")"
        label="${RUN_PREFIX}_x${X}_${cell}_s${SEED}"
        batch="${ENRICH_BATCH:-$X}"
        n_q="$X"
        [[ -z "$QIDS" ]] || n_q=$(( $(tr -cd ',' <<< "$QIDS" | wc -c) + 1 ))
        expected=$(( n_q * PASSES ))

        overrides="$SYSTEM_OVERRIDES \
systems.retrieve.working_set_collection_off=$coll_off systems.retrieve.hide_and_clear_working_sets=$hide \
systems.retrieve.enrich_working_sets=$mode systems.retrieve.enrich_query_batch_size=$batch \
experiments.run_name=$label \
+ub.label=$label +ub.num_queries=$X +ub.sample_seed=$SEED +ub.passes=$PASSES \
+ub.trajectory_working_sets=$traj +ub.wipe_collections=true +ub.keep_collections=[$keep_list]"
        [[ -z "$QIDS" ]] || overrides="$overrides experiments.qids=[$QIDS]"
        overrides="$overrides $EXTRA_OVERRIDES"

        existing="$(find_run_dir "$label")"
        if [[ -n "$existing" ]]; then
            n="$(result_rows "$existing/results.jsonl")"
            if [[ "$n" -ge "$expected" ]]; then
                log "SKIP  $label: complete ($n rows) at $existing"
                summary+=("skip   $label")
                continue
            fi
            log "REDO  $label: incomplete ($n/$expected rows) at $existing; running again into a fresh dir"
        fi

        cmd=("$PYTHON" scripts/bootstrap_enrich_upper_bound.py $overrides)
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  (reset $BASE_COLLECTION's chunk metadata on $CHROMA_HOST:$CHROMA_PORT)"
            printf '  %q' "${cmd[@]}"; printf '\n'
            summary+=("dry    $label")
            continue
        fi

        chroma_ensure
        if ! reset_corpus_metadata; then
            log "FAILED $label: could not reset $BASE_COLLECTION's chunk metadata; skipping"
            summary+=("FAILED(reset) $label")
            continue
        fi
        log "START $label (trajectory_ws=$traj hide_ws=$hide collection_off=$coll_off enrich_working_sets=$mode batch=$batch chroma_rss=$(chroma_rss_gb)GB)"
        t0=$(date +%s)
        if "${cmd[@]}" > "$LOG_DIR/$label.log" 2>&1; then
            status="ok"
        else
            status="FAILED (exit $?)"
        fi
        dt=$(( $(date +%s) - t0 ))
        run_dir="$(find_run_dir "$label")"
        runs_since_restart=$(( runs_since_restart + 1 ))
        log "DONE  $label: $status in ${dt}s chroma_rss=$(chroma_rss_gb)GB -> $run_dir (log: $LOG_DIR/$label.log)"
        [[ -f "$run_dir/summary.json" ]] && grep -E '^\[ub\] (metric|mean_score|mean_retrieve_cost|mean_retrieve_wall_s|sum_precompute_cost|sum_enrich_cost)' "$LOG_DIR/$label.log"
        summary+=("$status $label ${dt}s")
        if [[ "$CLEAN_ORPHANS" == "1" && -f "$CHROMA_STORE/chroma.sqlite3" ]]; then
            python3 "$QATFD_DIR/scripts/clean_chroma_orphans.py" --store "$CHROMA_STORE" --min-age-min 5 --quiet || log "WARN: orphan cleanup failed"
        fi
    done
  done
done

log "summary:"
printf '  %s\n' "${summary[@]}"
