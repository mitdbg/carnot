#!/usr/bin/env bash
# Collection-agent ablation: the SearchAgentSystem with the Bootstrap / Enrich agents building and
# curating collections, on the same model, seeds and question orders as scripts/run_codex_ablation.sh
# and scripts/run_search_agent_baseline.sh (so the runs land in the same plots). Four scenarios:
#
#   working_sets       enrich_working_sets=null    no collection agents; the search agents see (and reuse) each
#                                                  other's docs_for_q* trajectory working sets (hide_and_clear_working_sets=false)
#   bootstrap          enrich_working_sets=before  BootstrapAgent builds collections before the first question
#   enrich             enrich_working_sets=after   EnrichAgent curates collections after every ENRICH_BATCH questions
#   bootstrap_enrich   enrich_working_sets=both    both of the above
#
# In the three collection-agent scenarios the search agents only see the base collection plus the collections the
# Bootstrap / Enrich agents curated (QATFDSearchAgentConfig.hide_and_clear_working_sets, default true: trajectory
# working sets are hidden from later search agents and deleted after each Enrich run). working_sets is the one
# scenario that turns that off, so it measures trajectory reuse on its own.
#
# each for experiments.shuffle_seed in SEEDS, always sequential (workers=1) with the search agent's full
# working set + related-working-set reuse on, like the baseline's `seq` scenario — the collections the
# agents build are meant to be reused by later questions. ORDER=scenario (default) runs every seed of a
# scenario before moving to the next one (all bootstrap runs, then all enrich, then all bootstrap_enrich);
# ORDER=seed is the codex / baseline sweeps' seed-major order.
#
# Isolation: every collection on the chroma server EXCEPT the corpus collections (BASE_COLLECTION plus
# KEEP_COLLECTIONS) is deleted before each run (the search agents' ws_* working sets, the docs_for_q*
# collections, and everything the Bootstrap / Enrich agents created), so each run only ever sees its
# own collections. Legacy collections without metadata are deleted too.
# The corpus itself is reset as well: nothing stops the Bootstrap / Enrich agents from running map /
# semantic_map over BASE_COLLECTION, which writes new metadata keys onto every chunk, so before each
# (non-resumed) run engaging-scripts/reset_corpus_metadata.py strips every chunk key other than doc_id /
# chunk_id / element_id from it — ONLY when BASE_COLLECTION is one of the -v1 copies (the script's explicit
# allowlist). An original base collection is only checked, and the run is skipped if it turns out to be
# polluted (its native fields cannot be told apart from agent-written ones).
#
# Corpus: BASE_COLLECTION is the collection the eval opens (passed as benchmarks.collection_name) AND the
# one the wipe protects. To run on the metadata-stripped copy (doc_id / chunk_id / element_id only):
#   BASE_COLLECTION=officeqa-qwen-8b-v1 RUN_PREFIX=sa_ca_v1 scripts/run_collection_agent_ablation.sh
# (set RUN_PREFIX so the run dirs say which corpus they used; the base and the v1 copy live in the same
# store, so the same chroma server serves both and KEEP_COLLECTIONS shields whichever is not active).
#
# Chroma server: MANAGE_CHROMA=1 (default) makes the driver own / restart the server on an RSS threshold and
# run scripts/clean_chroma_orphans.py after every run; see scripts/lib/chroma_manage.sh for the why and the knobs
# (CHROMA_STORE, CHROMA_MAX_RSS_GB, CHROMA_RESTART_EVERY, CHROMA_KEEP_ON_EXIT, MANAGE_CHROMA=0 to use a hand-run server).
#
# Run dirs: $RESULTS_ROOT/$BENCHMARK/search_agent/${RUN_PREFIX}_<scenario>_<split>_s<seed>_<timestamp>.
# Idempotent like the baseline sweep: a run with EXPECTED_N rows is skipped, an incomplete run dir is
# resumed via experiments.resume_dir (its collections are NOT wiped, so reuse keeps working).
#
# Usage (from anywhere; use tmux/nohup):
#   scripts/run_collection_agent_ablation.sh 2>&1 | tee sa_collection_ablation.out    # officeqa dev, seeds 0 1 2
#   QIDS=UID0001 SEEDS=0 RUN_PREFIX=smoke ENRICH_BATCH=1 scripts/run_collection_agent_ablation.sh   # smoke test
#   DRY_RUN=1 ...                              # print the commands only
#   SEEDS="0" SCENARIOS="bootstrap" ...        # subsets
#   COMPUTE_MODEL=... ...                      # the shared compute answerer (default openai/gpt-5.6-terra)
#   ORDER=seed ...                             # seed-major order instead of scenario-major
#   MODEL=... AGENT_MODEL=... MAP_MODEL=... ENRICH_BATCH=... EXTRA_OVERRIDES="systems.retrieve.bootstrap_config.max_steps=10" ...
#   BASE_COLLECTION=officeqa-qwen-8b-v1 RUN_PREFIX=sa_ca_v1 ...   # no-metadata corpus copy (see above)
#
# Models: MODEL runs the search agent, COMPUTE_MODEL the shared compute answerer, AGENT_MODEL the Bootstrap /
# Enrich agents' own steps, MAP_MODEL the semantic_map judge (enrich) and the search agent's semantic_filter judge.
set -uo pipefail

QATFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/carnot/skunk/venv/bin/python3}"
BENCHMARK="${BENCHMARK:-officeqa}"
SPLIT="${SPLIT:-dev}"
MODEL="${MODEL:-openai/gpt-5.6-luna}"                  # search agent model
COMPUTE_MODEL="${COMPUTE_MODEL:-openai/gpt-5.6-terra}"  # the shared compute answerer's model
AGENT_MODEL="${AGENT_MODEL:-openai/gpt-5.6-terra}"      # Bootstrap / Enrich agents' own model
MAP_MODEL="${MAP_MODEL:-openai/gpt-5.6-luna}"           # semantic_map (bootstrap/enrich) + semantic_filter (search agent) judge model
BASE_COLLECTION="${BASE_COLLECTION:-officeqa-qwen-8b}"   # the corpus collection the eval opens; never deleted
KEEP_COLLECTIONS="${KEEP_COLLECTIONS:-officeqa-qwen-8b officeqa-qwen-8b-v1}"  # also never deleted (the other corpus copies in the store)
EXPECTED_N="${EXPECTED_N:-}"            # questions in the split; derived below
QIDS="${QIDS:-}"                        # comma-separated explicit qids (e.g. a smoke test); empty = the whole split
ENRICH_BATCH="${ENRICH_BATCH:-10}"      # EnrichAgent runs after every ENRICH_BATCH questions (enrich / bootstrap_enrich)
RUN_PREFIX="${RUN_PREFIX:-sa_ca}"       # run_name prefix; labels are ${RUN_PREFIX}_<scenario>_<split>_s<seed>
JUDGE_MODEL="${JUDGE_MODEL:-}"          # LLM judge for nugget-recall benchmarks (officeqa_synth); unused by officeqa
QA_PAIRS_PATH="${QA_PAIRS_PATH:-}"      # officeqa_synth only: qa_pairs file; empty = the yaml's
SHUFFLE_GROUP_KEY="${SHUFFLE_GROUP_KEY-__default__}"  # Question.meta key the seed shuffles by; empty = single questions
SEEDS="${SEEDS:-0 1 2}"
SCENARIOS="${SCENARIOS:-working_sets bootstrap enrich bootstrap_enrich}"
ORDER="${ORDER:-scenario}"              # scenario | seed (see header)
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # extra hydra overrides appended to every run
RESULTS_ROOT="${RESULTS_ROOT:-$QATFD_DIR/results}"
RUN_ROOT="$RESULTS_ROOT/$BENCHMARK/search_agent"
LOG_DIR="$RUN_ROOT/logs"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
CHROMA_STORE="${CHROMA_STORE:-$QATFD_DIR/benchmarks/officeqa/chromadb}"  # on-disk store behind the server (orphan cleanup / restarts)
CLEAN_ORPHANS="${CLEAN_ORPHANS:-1}"  # 1 = after every run, delete HNSW segment dirs that chroma left behind for dropped collections
MANAGE_CHROMA="${MANAGE_CHROMA:-1}"     # 1 = driver starts / restarts the chroma server itself (see scripts/lib/chroma_manage.sh)
CHROMA_MAX_RSS_GB="${CHROMA_MAX_RSS_GB:-30}"       # restart the server before a run once its RSS exceeds this
CHROMA_RESTART_EVERY="${CHROMA_RESTART_EVERY:-0}"  # also restart every N runs (0 = only on RSS / liveness)
CHROMA_KEEP_ON_EXIT="${CHROMA_KEEP_ON_EXIT:-0}"    # 1 = leave a driver-started server running when the sweep ends
CHROMA_WAIT_MIN="${CHROMA_WAIT_MIN:-30}"           # MANAGE_CHROMA=0: minutes to wait for a down server before aborting
CHROMA_SERVER_SCRIPT="${CHROMA_SERVER_SCRIPT:-$QATFD_DIR/../skunk/scripts/run_chroma_server.sh}"
CHROMA_LOG="$LOG_DIR/chroma_server.log"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

result_rows() { if [[ -f "$1" ]]; then grep -c . "$1" || true; else echo 0; fi; }
source "$QATFD_DIR/scripts/lib/chroma_manage.sh"

# ---------------------------------------------------------------------------
# per-benchmark defaults (same rules as run_search_agent_baseline.sh)
# ---------------------------------------------------------------------------
synth_expected_n() {
    "$PYTHON" - "$QATFD_DIR/configs/benchmarks/officeqa_synth.yaml" "$QATFD_DIR/benchmarks" "$QA_PAIRS_PATH" <<'PY'
import json, pathlib, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
path = pathlib.Path(sys.argv[3] or cfg["qa_pairs_path"])
if not path.is_absolute():
    path = pathlib.Path(sys.argv[2]) / path
records = json.load(open(path))
include_seeds = bool(cfg.get("include_seeds", False))
print(sum(1 for r in records if include_seeds or r.get("idx") is not None))
PY
}

BENCH_OVERRIDES=""
case "$BENCHMARK" in
    officeqa)
        EXPECTED_N="${EXPECTED_N:-33}"
        [[ "$SHUFFLE_GROUP_KEY" == "__default__" ]] && SHUFFLE_GROUP_KEY=""
        ;;
    officeqa_synth)
        EXPECTED_N="${EXPECTED_N:-$(synth_expected_n)}" || { log "ABORT: could not derive EXPECTED_N from the officeqa_synth qa_pairs file"; exit 1; }
        [[ "$SHUFFLE_GROUP_KEY" == "__default__" ]] && SHUFFLE_GROUP_KEY="loi_idx"
        [[ -n "$JUDGE_MODEL" ]] || { log "ABORT: officeqa_synth is scored by an LLM nugget judge; set JUDGE_MODEL=<openrouter model id>"; exit 1; }
        [[ -z "$QA_PAIRS_PATH" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES benchmarks.qa_pairs_path=$QA_PAIRS_PATH"
        ;;
    *)
        [[ -n "$EXPECTED_N" ]] || { log "ABORT: set EXPECTED_N for benchmark '$BENCHMARK'"; exit 1; }
        [[ "$SHUFFLE_GROUP_KEY" == "__default__" ]] && SHUFFLE_GROUP_KEY=""
        ;;
esac
if [[ -n "$QIDS" ]]; then
    # an explicit qid list overrides the split: a run is complete once every listed qid has a row
    EXPECTED_N=$(( $(tr -cd ',' <<< "$QIDS" | wc -c) + 1 ))
    BENCH_OVERRIDES="$BENCH_OVERRIDES experiments.qids=[$QIDS]"
fi
[[ -z "$JUDGE_MODEL" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES benchmarks.judge_model=$JUDGE_MODEL"
[[ "$RESULTS_ROOT" == "$QATFD_DIR/results" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES results_root=$RESULTS_ROOT"
[[ -z "$SHUFFLE_GROUP_KEY" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES experiments.shuffle_group_key=$SHUFFLE_GROUP_KEY"

# the search agent's fixed tool / working-set configuration (= the baseline's `seq` scenario)
SYSTEM_OVERRIDES="systems=search_agent inference.llm_model=$MODEL systems.compute.llm_model=$COMPUTE_MODEL \
systems.retrieve.include_search_corpus=true systems.retrieve.include_grep_corpus=true systems.retrieve.include_semantic_filter=false \
systems.retrieve.working_set_collection_off=false systems.retrieve.id_tracking_off=false \
systems.retrieve.fetch_related_working_sets=true experiments.run_mode=sequential experiments.workers=1 \
systems.retrieve.semantic_filter_llm_model=$MAP_MODEL \
systems.retrieve.bootstrap_config.llm_model=$AGENT_MODEL systems.retrieve.bootstrap_config.semantic_map_llm_model=$MAP_MODEL \
systems.retrieve.enrich_config.llm_model=$AGENT_MODEL systems.retrieve.enrich_config.semantic_map_llm_model=$MAP_MODEL \
benchmarks.collection_name=$BASE_COLLECTION \
benchmarks.chroma_server_host=$CHROMA_HOST benchmarks.chroma_server_port=$CHROMA_PORT"

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
has_openrouter_key() {
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || grep -q '^OPENROUTER_API_KEY=' "$QATFD_DIR/../skunk/.env" 2>/dev/null
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
    # another SEARCH AGENT run would share (and have its collections wiped by) this sweep
    if ps -eo args | grep -E '^[^ ]*python[0-9.]* -m qatfd\.runner' | grep -q 'systems=search_agent'; then
        log "ABORT: a search_agent qatfd.runner is already running (collections are shared on the chroma server)"; ok=0
    fi
    [[ $ok -eq 1 ]] || exit 1
}

# delete every collection except the corpus collections, so a run only sees the collections it builds itself
wipe_collections() {
    "$PYTHON" - "$CHROMA_HOST" "$CHROMA_PORT" "$BASE_COLLECTION" $KEEP_COLLECTIONS <<'PY'
import sys, time
from skunk.chroma_client import make_chroma_client
from chromadb.errors import NotFoundError
host, port, base = sys.argv[1], int(sys.argv[2]), sys.argv[3]
keep = {base, *sys.argv[4:]}  # the active corpus + every other corpus copy served from this store
client = make_chroma_client(host, port)
client.get_collection(base)  # raises if the base corpus is not served here
names, offset = [], 0
while True:
    page = client.list_collections(limit=100, offset=offset)
    names.extend(c.name for c in page if c.name not in keep)
    if len(page) < 100:
        break
    offset += 100
failed = []
for name in names:
    for attempt in range(3):  # a delete can 500 transiently (e.g. the server is still compacting it)
        try:
            client.delete_collection(name)
            break
        except NotFoundError:
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                failed.append((name, f"{type(e).__name__}: {e}"))
            else:
                time.sleep(5)
left = [c.name for c in client.list_collections(limit=100) if c.name not in keep]
print(f"[sweep] wiped {len(names) - len(failed)} collection(s); kept {sorted(keep)}; {len(left)} other collection(s) remain")
for name, err in failed:
    print(f"[sweep] delete failed for {name!r}: {err}", file=sys.stderr)
sys.exit(1 if left else 0)
PY
}

# strip agent-written chunk metadata (a map / semantic_map over the corpus) from BASE_COLLECTION: a -v1
# copy is stripped, any other known collection is only checked (see engaging-scripts/reset_corpus_metadata.py)
reset_corpus_metadata() {
    "$PYTHON" engaging-scripts/reset_corpus_metadata.py --collection "$BASE_COLLECTION" --server "$CHROMA_HOST:$CHROMA_PORT"
}

scenario_overrides() {
    case "$1" in
        working_sets)     echo "systems.retrieve.enrich_working_sets=null systems.retrieve.hide_and_clear_working_sets=false" ;;
        bootstrap)        echo "systems.retrieve.enrich_working_sets=before" ;;
        enrich)           echo "systems.retrieve.enrich_working_sets=after systems.retrieve.enrich_query_batch_size=$ENRICH_BATCH" ;;
        bootstrap_enrich) echo "systems.retrieve.enrich_working_sets=both systems.retrieve.enrich_query_batch_size=$ENRICH_BATCH" ;;
        *) log "ABORT: unknown scenario '$1'"; exit 1 ;;
    esac
}

find_run_dir() { ls -d "$RUN_ROOT/${1}_"[0-9]* 2>/dev/null | sort | tail -n 1; }

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
[[ "$DRY_RUN" == "1" ]] || preflight
mkdir -p "$LOG_DIR"
cd "$QATFD_DIR"
log "benchmark=$BENCHMARK collection=$BASE_COLLECTION split=$SPLIT model=$MODEL compute_model=$COMPUTE_MODEL agent_model=$AGENT_MODEL map_model=$MAP_MODEL expected_n=$EXPECTED_N seeds='$SEEDS' scenarios='$SCENARIOS' enrich_batch=$ENRICH_BATCH${QIDS:+ qids=$QIDS}${JUDGE_MODEL:+ judge=$JUDGE_MODEL}${SHUFFLE_GROUP_KEY:+ shuffle_by=$SHUFFLE_GROUP_KEY} run_root=$RUN_ROOT"

# (scenario, seed) pairs in the requested order
pairs=()
case "$ORDER" in
    scenario) for scenario in $SCENARIOS; do for seed in $SEEDS; do pairs+=("$scenario $seed"); done; done ;;
    seed)     for seed in $SEEDS; do for scenario in $SCENARIOS; do pairs+=("$scenario $seed"); done; done ;;
    *) log "ABORT: unknown ORDER '$ORDER' (scenario | seed)"; exit 1 ;;
esac

declare -a summary=()
for pair in "${pairs[@]}"; do
    read -r scenario seed <<< "$pair"
    {
        label="${RUN_PREFIX}_${scenario}_${SPLIT}_s${seed}"
        overrides="$SYSTEM_OVERRIDES benchmarks=$BENCHMARK experiments.split=$SPLIT experiments.shuffle_seed=$seed \
experiments.run_name=$label $(scenario_overrides "$scenario")$BENCH_OVERRIDES $EXTRA_OVERRIDES"

        existing="$(find_run_dir "$label")"
        wipe=1
        if [[ -n "$existing" ]]; then
            n="$(result_rows "$existing/results.jsonl")"
            if [[ "$n" -ge "$EXPECTED_N" ]]; then
                log "SKIP  $label: complete ($n rows) at $existing"
                summary+=("skip   $label")
                continue
            fi
            log "RESUME $label: $n/$EXPECTED_N rows at $existing (collections kept)"
            overrides="$overrides experiments.resume_dir=$existing"
            wipe=0
        else
            log "START $label"
        fi

        cmd=("$PYTHON" -m qatfd.runner $overrides)
        if [[ "$DRY_RUN" == "1" ]]; then
            [[ $wipe -eq 1 ]] && echo "  (wipe all non-base collections on $CHROMA_HOST:$CHROMA_PORT; reset $BASE_COLLECTION's chunk metadata)"
            printf '  %q' "${cmd[@]}"; printf '\n'
            summary+=("dry    $label")
            continue
        fi

        chroma_ensure
        if [[ $wipe -eq 1 ]] && ! wipe_collections; then
            log "FAILED $label: could not wipe collections; skipping"
            summary+=("FAILED(wipe) $label")
            continue
        fi
        if [[ $wipe -eq 1 ]] && ! reset_corpus_metadata; then
            log "FAILED $label: could not reset $BASE_COLLECTION's chunk metadata; skipping"
            summary+=("FAILED(reset) $label")
            continue
        fi

        log "RUN   $label (chroma_rss=$(chroma_rss_gb)GB)"
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
        summary+=("$status $label ${dt}s")
        if [[ "$CLEAN_ORPHANS" == "1" && -f "$CHROMA_STORE/chroma.sqlite3" ]]; then
            python3 "$QATFD_DIR/scripts/clean_chroma_orphans.py" --store "$CHROMA_STORE" --min-age-min 5 --quiet || log "WARN: orphan cleanup failed"
        fi
    }
done

log "summary:"
printf '  %s\n' "${summary[@]}"
