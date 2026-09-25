#!/usr/bin/env bash
# SearchAgent baseline to set against the Codex ablation: grep + vector search + read_document with the
# full working set (intermediate collection + id tracking) ON, on the same model, seeds and question
# orders as scripts/run_codex_ablation.sh, so the two systems land in the same plots.
#
# Runs, serially (working sets live on the shared chroma server, so runs must not overlap):
#   par   parallel   workers=$WORKERS  fetch_related_working_sets=false   (isolation, like codex `par`)
#   seq   sequential workers=1         fetch_related_working_sets=true    (reuses earlier questions' working
#                                                                          sets, the search agent's analogue of
#                                                                          codex's session-resume scenario)
# each for experiments.shuffle_seed in SEEDS, seed-major. Every ws_* collection on the chroma server is
# deleted before each run: the registry rehydrates all of them, so leftovers would leak across runs.
#
# Run dirs: $RESULTS_ROOT/$BENCHMARK/search_agent/sa_<scenario>_<split>_s<seed>_<timestamp>. The codex plot
# scripts (plot_codex_ablation.py, plot_codex_by_index.py) pick these up automatically next to the codex
# runs when pointed at results/<benchmark>/codex.
#
# Idempotent like the codex sweep: a run with EXPECTED_N rows is skipped, an incomplete run dir is resumed
# via experiments.resume_dir (its working sets are NOT wiped, so reuse keeps working across the resume).
#
# Usage (from anywhere; use tmux/nohup):
#   scripts/run_search_agent_baseline.sh 2>&1 | tee sa_baseline.out                # officeqa dev, seeds 0 1 2
#   BENCHMARK=officeqa_synth JUDGE_MODEL=openai/gpt-5.6-terra \
#       QA_PAIRS_PATH=officeqa/synth_dev_v3_qa_pairs.json RESULTS_ROOT=results/v3 \
#       scripts/run_search_agent_baseline.sh 2>&1 | tee sa_baseline_v3.out         # synth v3 (matches the codex v3 runs)
#   DRY_RUN=1 ...                    # print the commands only
#   SEEDS="0" SCENARIOS="seq" ...    # subsets
#   MODEL=... WORKERS=... REUSE_IN_PAR=true ...
set -uo pipefail

QATFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/carnot/skunk/venv/bin/python3}"
BENCHMARK="${BENCHMARK:-officeqa}"
SPLIT="${SPLIT:-dev}"
MODEL="${MODEL:-openai/gpt-5.6-luna}"
EXPECTED_N="${EXPECTED_N:-}"            # questions in the split; derived below
WORKERS="${WORKERS:-4}"                 # parallel-mode concurrency (chroma grep/search saturate ~4)
REUSE_IN_PAR="${REUSE_IN_PAR:-false}"   # fetch_related_working_sets in the parallel scenario (racy across workers)
JUDGE_MODEL="${JUDGE_MODEL:-}"          # LLM judge for nugget-recall benchmarks (officeqa_synth); unused by officeqa
QA_PAIRS_PATH="${QA_PAIRS_PATH:-}"      # officeqa_synth only: qa_pairs file (under qatfd/benchmarks/, or absolute); empty = the yaml's
SHUFFLE_GROUP_KEY="${SHUFFLE_GROUP_KEY-__default__}"  # Question.meta key the seed shuffles by; empty = single questions
SEEDS="${SEEDS:-0 1 2}"
SCENARIOS="${SCENARIOS:-par seq}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # extra hydra overrides appended to every run
RESULTS_ROOT="${RESULTS_ROOT:-$QATFD_DIR/results}"
RUN_ROOT="$RESULTS_ROOT/$BENCHMARK/search_agent"
LOG_DIR="$RUN_ROOT/logs"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

# ---------------------------------------------------------------------------
# per-benchmark defaults (same rules as run_codex_ablation.sh)
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
[[ -z "$JUDGE_MODEL" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES benchmarks.judge_model=$JUDGE_MODEL"
[[ "$RESULTS_ROOT" == "$QATFD_DIR/results" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES results_root=$RESULTS_ROOT"
[[ -z "$SHUFFLE_GROUP_KEY" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES experiments.shuffle_group_key=$SHUFFLE_GROUP_KEY"

# the search agent's fixed tool / working-set configuration for this baseline
SYSTEM_OVERRIDES="systems=search_agent inference.llm_model=$MODEL \
systems.retrieve.include_search_corpus=true systems.retrieve.include_grep_corpus=true systems.retrieve.include_semantic_filter=false \
systems.retrieve.working_set_collection_off=false systems.retrieve.id_tracking_off=false"

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
has_openrouter_key() {
    [[ -n "${OPENROUTER_API_KEY:-}" ]] || grep -q '^OPENROUTER_API_KEY=' "$QATFD_DIR/../skunk/.env" 2>/dev/null
}

preflight() {
    local ok=1
    [[ -x "$PYTHON" ]] || { log "ABORT: python not found at $PYTHON"; ok=0; }
    # the search agent (skunk LLMClient + embeddings + judge) authenticates with OPENROUTER_API_KEY, not the codex key
    has_openrouter_key || { log "ABORT: OPENROUTER_API_KEY is unset (env or skunk/.env)"; ok=0; }
    if ! curl -sf -m 5 "http://$CHROMA_HOST:$CHROMA_PORT/api/v2/heartbeat" >/dev/null; then
        log "ABORT: no chroma server answering at $CHROMA_HOST:$CHROMA_PORT"; ok=0
    fi
    # another SEARCH AGENT run would share (and have its ws_* collections wiped by) this sweep; a codex run
    # on the same chroma server is fine, it never touches working sets
    if ps -eo args | grep -E '^[^ ]*python[0-9.]* -m qatfd\.runner' | grep -q 'systems=search_agent'; then
        log "ABORT: a search_agent qatfd.runner is already running (working sets are shared on the chroma server)"; ok=0
    fi
    [[ $ok -eq 1 ]] || exit 1
}

# delete every ws_* collection so a run's working-set reuse only sees its own questions
wipe_working_sets() {
    "$PYTHON" - "$CHROMA_HOST" "$CHROMA_PORT" <<'PY'
import sys
from skunk.chroma_client import make_chroma_client
client = make_chroma_client(sys.argv[1], int(sys.argv[2]))
names = [c.name for c in client.list_collections() if c.metadata.get("is_working_set")]
for name in names:
    client.delete_collection(name)
print(f"[sweep] wiped {len(names)} working-set collection(s)")
PY
}

scenario_overrides() {
    case "$1" in
        par) echo "experiments.run_mode=parallel experiments.workers=$WORKERS systems.retrieve.fetch_related_working_sets=$REUSE_IN_PAR" ;;
        # hide_and_clear_working_sets (default true) hides earlier questions' trajectory working sets from later
        # search agents; the seq scenario exists to measure exactly that reuse, so turn it off
        seq) echo "experiments.run_mode=sequential experiments.workers=1 systems.retrieve.fetch_related_working_sets=true systems.retrieve.hide_and_clear_working_sets=false" ;;
        *) log "ABORT: unknown scenario '$1'"; exit 1 ;;
    esac
}

result_rows() { if [[ -f "$1" ]]; then grep -c . "$1" || true; else echo 0; fi; }
find_run_dir() { ls -d "$RUN_ROOT/${1}_"[0-9]* 2>/dev/null | sort | tail -n 1; }

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
[[ "$DRY_RUN" == "1" ]] || preflight
mkdir -p "$LOG_DIR"
cd "$QATFD_DIR"
log "benchmark=$BENCHMARK split=$SPLIT model=$MODEL expected_n=$EXPECTED_N seeds='$SEEDS' scenarios='$SCENARIOS'${JUDGE_MODEL:+ judge=$JUDGE_MODEL}${SHUFFLE_GROUP_KEY:+ shuffle_by=$SHUFFLE_GROUP_KEY}${QA_PAIRS_PATH:+ qa_pairs=$QA_PAIRS_PATH} run_root=$RUN_ROOT"

declare -a summary=()
for seed in $SEEDS; do
    for scenario in $SCENARIOS; do
        label="sa_${scenario}_${SPLIT}_s${seed}"
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
            log "RESUME $label: $n/$EXPECTED_N rows at $existing (working sets kept)"
            overrides="$overrides experiments.resume_dir=$existing"
            wipe=0
        else
            log "START $label"
        fi

        cmd=("$PYTHON" -m qatfd.runner $overrides)
        if [[ "$DRY_RUN" == "1" ]]; then
            [[ $wipe -eq 1 ]] && echo "  (wipe ws_* collections on $CHROMA_HOST:$CHROMA_PORT)"
            printf '  %q' "${cmd[@]}"; printf '\n'
            summary+=("dry    $label")
            continue
        fi

        if [[ $wipe -eq 1 ]] && ! wipe_working_sets; then
            log "FAILED $label: could not wipe working-set collections; skipping"
            summary+=("FAILED(wipe) $label")
            continue
        fi

        t0=$(date +%s)
        if "${cmd[@]}" > "$LOG_DIR/$label.log" 2>&1; then
            status="ok"
        else
            status="FAILED (exit $?)"
        fi
        dt=$(( $(date +%s) - t0 ))
        run_dir="$(find_run_dir "$label")"
        log "DONE  $label: $status in ${dt}s -> $run_dir (log: $LOG_DIR/$label.log)"
        summary+=("$status $label ${dt}s")
    done
done

log "summary:"
printf '  %s\n' "${summary[@]}"
