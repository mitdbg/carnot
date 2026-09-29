#!/usr/bin/env bash
# Codex session-resume ablation on the OfficeQA dev split, or on the synthetic OfficeQA
# v2 set (BENCHMARK=officeqa_synth).
#
# Runs, serially (the runner hosts the MCP server on one fixed port, so runs cannot overlap):
#   parallel                          x3   (isolation baseline: fresh session per question)
#   sequential  resume=on             x3   (one thread carried across all questions)
#   sequential  shell=on  resume=off  x3   (opt-in: SCENARIOS="seq_shell seq_shell_resume"; the agent gets a
#   sequential  shell=on  resume=on   x3    sandboxed shell and a persistent workspace with AGENTS.md)
#   sequential  isolation             x3   (opt-in: SCENARIOS="seq_iso seq_iso_bs seq_iso_en seq_iso_bsen"; every
#   sequential  isolation + Bootstrap x3    question is a fresh ephemeral codex session with no shell, like `par`, but
#   sequential  isolation + Enrich    x3    run in arrival order so the Bootstrap / Enrich collection agents can build
#   sequential  isolation + both      x3    collections between questions; mirrors the search agent's
#                                           bootstrap_enrich_dev cells exp1 / exp3 / exp5 / exp7)
# in seed-major order (every scenario at seed 0, then seed 1, ...) so the scenarios can be compared
# after the first |SCENARIOS| runs; each for experiments.shuffle_seed in SEEDS: the seed fixes the ORDER the questions are asked in, which is
# what the session-resume / shell scenarios are sensitive to. The parallel (isolation) baseline gets the
# same seeds so every scenario is labelled and snapshotted identically (run_name suffix _s<N>), even
# though order cannot matter there.
#
# officeqa_synth (BENCHMARK=officeqa_synth): the questions come in ~5-question lines of inquiry that must
# stay consecutive for the sequential scenarios to mean anything, so the seed shuffles the ORDER OF THE
# LINES OF INQUIRY (experiments.shuffle_group_key=loi_idx), not single questions. Scoring is LLM-judged
# nugget recall, so JUDGE_MODEL is required (-> benchmarks.judge_model) and the judge needs
# OPENROUTER_API_KEY (env or skunk/.env). EXPECTED_N is derived from the benchmark's qa_pairs file.
#
# Idempotent: a run whose results.jsonl already has EXPECTED_N rows is skipped; a run dir that exists
# but is incomplete is resumed via experiments.resume_dir (for sequential+resume this also picks up the
# persisted codex thread id). Per-run stdout/stderr goes to $LOG_DIR/<label>.log.
#
# seq_iso* scenarios are NEVER resumed: the collection agents' state (question count, workload window, the
# collections themselves) is not persisted, so an incomplete seq_iso* run is abandoned and started over in a
# new run dir. Before every seq_iso* run (the no-agent baseline included) every collection on the chroma
# server except the corpus copies (BASE_COLLECTION + KEEP_COLLECTIONS) is deleted and BASE_COLLECTION's
# agent-written chunk metadata is stripped (scripts/lib/collection_reset.sh), so each run starts from the
# pristine corpus. That requires BASE_COLLECTION to be a -v1 copy (the only collections the reset may strip),
# and no other collection-building run (search_agent or codex) may share the chroma server.
# Collection agents: AGENT_MODEL runs the Bootstrap / Enrich agents' own steps, MAP_MODEL their semantic_map
# judge (defaults match the search agent's bootstrap_enrich_dev sweep), ENRICH_BATCH / MAX_PREVIOUS_QUERIES the
# Enrich cadence and workload window. They bill through OPENROUTER_API_KEY (env or skunk/.env).
#
# Usage (from anywhere; runs for many hours — use nohup/tmux):
#   nohup scripts/run_codex_ablation.sh > codex_ablation.out 2>&1 &
#   DRY_RUN=1 scripts/run_codex_ablation.sh        # print the commands only
#   SEEDS="0" scripts/run_codex_ablation.sh         # subset of seeds
#   SCENARIOS="par" scripts/run_codex_ablation.sh   # subset of scenarios
#   VARIANT=c40 EXTRA_OVERRIDES="systems.auto_compact_token_limit=420000" \
#       SCENARIOS="seq_resume" scripts/run_codex_ablation.sh
#     # a tagged variant: run_name becomes codex_<scenario>_<VARIANT>_<split>_s<seed> so it never collides
#     # with (or silently supersedes) the untagged baseline in the plot scripts; VARIANT must be [a-z0-9]+
#   BENCHMARK=officeqa_synth JUDGE_MODEL=<judge-model> nohup scripts/run_codex_ablation.sh > codex_synth.out 2>&1 &
#     # the synthetic v2 set: results under results/officeqa_synth/codex (plot with
#     # --results-root results/officeqa_synth/codex)
#   BENCHMARK=officeqa_synth JUDGE_MODEL=<judge-model> QA_PAIRS_PATH=officeqa/synth_dev_v3_qa_pairs.json \
#       RESULTS_ROOT=results/v3 nohup scripts/run_codex_ablation.sh > codex_synth_v3.out 2>&1 &
#     # another qa_pairs file (e.g. the v3 sample): EXPECTED_N is derived from that file, and RESULTS_ROOT keeps
#     # its runs under results/v3/officeqa_synth/codex so the plot scripts never mix them with the v2 runs
#     # (plot with --results-root results/v3/officeqa_synth/codex; the figures are labelled officeqa_synth_v3)
#   BASE_COLLECTION=officeqa-qwen-8b-v1 VARIANT=v1 SCENARIOS="seq_iso seq_iso_bs seq_iso_en seq_iso_bsen" \
#       nohup scripts/run_codex_ablation.sh > codex_collection_agents.out 2>&1 &
#     # sequential isolation +- Bootstrap / Enrich on the no-metadata corpus copy (plot with
#     #   --scenarios par seq_iso seq_iso_bs seq_iso_en seq_iso_bsen --variants v1 --tag collection_agents)
set -uo pipefail

QATFD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/carnot/skunk/venv/bin/python3}"
BENCHMARK="${BENCHMARK:-officeqa}"
SPLIT="${SPLIT:-dev}"
EXPECTED_N="${EXPECTED_N:-}"            # questions in the split; a run with this many rows is complete (derived below)
WORKERS="${WORKERS:-4}"                 # parallel-mode concurrency (chroma grep/search saturate ~4)
JUDGE_MODEL="${JUDGE_MODEL:-}"          # LLM judge for nugget-recall benchmarks (officeqa_synth); unused by officeqa
QA_PAIRS_PATH="${QA_PAIRS_PATH:-}"      # officeqa_synth only: qa_pairs file (under qatfd/benchmarks/, or absolute); empty = the yaml's
SHUFFLE_GROUP_KEY="${SHUFFLE_GROUP_KEY-__default__}"  # Question.meta key the seed shuffles by; empty = single questions
SEEDS="${SEEDS:-0 1 2}"
SCENARIOS="${SCENARIOS:-par seq_resume}"
VARIANT="${VARIANT:-}"                  # optional run_name tag (e.g. c40); empty = untagged baseline
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # extra hydra overrides appended to every run (e.g. the variant's knob)
WORKSPACES_ROOT="${WORKSPACES_ROOT:-/home/ubuntu/codex_workspaces}"  # codex cwd per run: $WORKSPACES_ROOT/<run label>
RESULTS_ROOT="${RESULTS_ROOT:-$QATFD_DIR/results}"
RUN_ROOT="$RESULTS_ROOT/$BENCHMARK/codex"
LOG_DIR="$RUN_ROOT/logs"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
MCP_PORT="${MCP_PORT:-8765}"
DRY_RUN="${DRY_RUN:-0}"
BASE_COLLECTION="${BASE_COLLECTION:-}"  # corpus collection (benchmarks.collection_name); empty = the benchmark yaml's
KEEP_COLLECTIONS="${KEEP_COLLECTIONS:-officeqa-qwen-8b officeqa-qwen-8b-v1}"  # corpus copies the seq_iso* wipe never deletes
MODEL="${MODEL:-openai/gpt-5.6-luna}"               # inference.llm_model for seq_iso*: the fallback for any agent model left null
AGENT_MODEL="${AGENT_MODEL:-openai/gpt-5.6-terra}"  # Bootstrap / Enrich agents' own steps
MAP_MODEL="${MAP_MODEL:-openai/gpt-5.6-luna}"       # Bootstrap / Enrich semantic_map judge
ENRICH_BATCH="${ENRICH_BATCH:-5}"                   # EnrichAgent runs after every ENRICH_BATCH questions
MAX_PREVIOUS_QUERIES="${MAX_PREVIOUS_QUERIES:-50}"  # EnrichAgent workload window (questions shown to it)

export PATH="$HOME/.local/bin:$PATH"

source "$QATFD_DIR/scripts/lib/collection_reset.sh"   # wipe_collections / reset_corpus_metadata

log() { printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }

# ---------------------------------------------------------------------------
# per-benchmark defaults
# ---------------------------------------------------------------------------
# officeqa_synth: the qa_pairs file (QA_PAIRS_PATH, else the benchmark yaml's), minus the seed records unless
# the yaml includes them (mirrors OfficeQASynthBenchmark.load_questions).
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

BENCH_OVERRIDES=""   # benchmark-specific hydra overrides appended to every run
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
# a non-default RESULTS_ROOT must reach the runner too, or it would write under results/ while this script
# looks for (and resumes) runs under RESULTS_ROOT
[[ "$RESULTS_ROOT" == "$QATFD_DIR/results" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES results_root=$RESULTS_ROOT"
[[ -z "$SHUFFLE_GROUP_KEY" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES experiments.shuffle_group_key=$SHUFFLE_GROUP_KEY"
[[ -z "$BASE_COLLECTION" ]] || BENCH_OVERRIDES="$BENCH_OVERRIDES benchmarks.collection_name=$BASE_COLLECTION"

# seq_iso* = sequential isolation (+- the collection agents); they get the wipe / reset / no-resume treatment
is_iso_scenario() { [[ "$1" == seq_iso* ]]; }
ANY_ISO=0
for s in $SCENARIOS; do is_iso_scenario "$s" && ANY_ISO=1; done

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
preflight() {
    local ok=1
    [[ -x "$PYTHON" ]] || { log "ABORT: python not found at $PYTHON"; ok=0; }
    command -v codex >/dev/null || { log "ABORT: codex CLI not on PATH"; ok=0; }
    [[ -n "${OPENROUTER_CODEX_API_KEY:-}" ]] || { log "ABORT: OPENROUTER_CODEX_API_KEY is unset (codex auth)"; ok=0; }
    [[ -n "${OPENROUTER_MGMT_API_KEY:-}" ]] || { log "ABORT: OPENROUTER_MGMT_API_KEY is unset (cost metering)"; ok=0; }
    if [[ -n "$JUDGE_MODEL" && -z "${OPENROUTER_API_KEY:-}" ]] && ! grep -q '^OPENROUTER_API_KEY=' "$QATFD_DIR/../skunk/.env" 2>/dev/null; then
        log "ABORT: OPENROUTER_API_KEY is unset (env or skunk/.env) — the nugget judge needs it"; ok=0
    fi
    if ! curl -sf -m 5 "http://$CHROMA_HOST:$CHROMA_PORT/api/v2/heartbeat" >/dev/null; then
        log "ABORT: no chroma server answering at $CHROMA_HOST:$CHROMA_PORT"; ok=0
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$MCP_PORT") 2>/dev/null; then
        log "ABORT: something is already listening on the MCP port $MCP_PORT (stale runner?)"; ok=0
    fi
    if [[ $ANY_ISO -eq 1 ]]; then
        if [[ -z "${OPENROUTER_API_KEY:-}" ]] && ! grep -q '^OPENROUTER_API_KEY=' "$QATFD_DIR/../skunk/.env" 2>/dev/null; then
            log "ABORT: OPENROUTER_API_KEY is unset (env or skunk/.env); the Bootstrap / Enrich agents bill through it"; ok=0
        fi
        # the wipe + metadata reset before each seq_iso* run may only strip a -v1 copy
        if [[ "$BASE_COLLECTION" != *-v1 ]]; then
            log "ABORT: seq_iso* scenarios need BASE_COLLECTION=<a -v1 corpus copy> (got '${BASE_COLLECTION:-<yaml default>}')"; ok=0
        fi
        # any other runner on this chroma server would have its collections wiped by (or leak into) this sweep
        if ps -eo args | grep -E '^[^ ]*python[0-9.]* -m qatfd\.runner' | grep -q 'systems='; then
            log "ABORT: another qatfd.runner is running (collections are shared on the chroma server)"; ok=0
        fi
    fi
    [[ $ok -eq 1 ]] || exit 1
}

# ---------------------------------------------------------------------------
# scenario -> hydra overrides
# ---------------------------------------------------------------------------
# systems.isolation (default true in configs/systems/codex.yaml) is set explicitly everywhere: true = a fresh
# ephemeral session with its own work dir per question; the resume / shell scenarios carry state across questions
ISO="experiments.run_mode=sequential experiments.workers=1 systems.isolation=true systems.codex_shell=false \
systems.session_resume=false inference.llm_model=$MODEL"
COLLECTION_AGENTS="systems.bootstrap_config.llm_model=$AGENT_MODEL systems.bootstrap_config.semantic_map_llm_model=$MAP_MODEL \
systems.enrich_config.llm_model=$AGENT_MODEL systems.enrich_config.semantic_map_llm_model=$MAP_MODEL \
systems.enrich_config.max_previous_queries=$MAX_PREVIOUS_QUERIES systems.enrich_query_batch_size=$ENRICH_BATCH"
scenario_overrides() {
    case "$1" in
        par)              echo "experiments.run_mode=parallel experiments.workers=$WORKERS systems.isolation=true systems.codex_shell=false systems.session_resume=false" ;;
        seq_resume)       echo "experiments.run_mode=sequential systems.isolation=false systems.codex_shell=false systems.session_resume=true" ;;
        seq_shell)        echo "experiments.run_mode=sequential systems.isolation=false systems.codex_shell=true  systems.session_resume=false" ;;
        seq_shell_resume) echo "experiments.run_mode=sequential systems.isolation=false systems.codex_shell=true  systems.session_resume=true" ;;
        seq_iso)          echo "$ISO systems.enrich_working_sets=null" ;;
        seq_iso_bs)       echo "$ISO $COLLECTION_AGENTS systems.enrich_working_sets=before" ;;
        seq_iso_en)       echo "$ISO $COLLECTION_AGENTS systems.enrich_working_sets=after" ;;
        seq_iso_bsen)     echo "$ISO $COLLECTION_AGENTS systems.enrich_working_sets=both" ;;
        *) log "ABORT: unknown scenario '$1'"; exit 1 ;;
    esac
}

# Codex's working directory for a run. Kept OUTSIDE the carnot checkout: under workspace-write the agent
# can read the whole disk, and a cwd inside qatfd/results/ puts benchmarks/officeqa/officeqa_pro.csv (the
# gold answers) three `..` hops away. Out here nothing of interest is reachable by relative path, and
# Codex only loads AGENTS.md from the cwd itself when the cwd is not inside a git repo.
workspace_dir() { echo "$WORKSPACES_ROOT/$1"; }

# rows in a results.jsonl (0 if missing)
result_rows() { if [[ -f "$1" ]]; then grep -c . "$1" || true; else echo 0; fi; }

# newest existing run dir for a label, or empty
find_run_dir() {
    ls -d "$RUN_ROOT/${1}_"[0-9]* 2>/dev/null | sort | tail -n 1
}

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
if [[ -n "$VARIANT" && ! "$VARIANT" =~ ^[a-z0-9]+$ ]]; then
    log "ABORT: VARIANT='$VARIANT' must match [a-z0-9]+ (the plot scripts parse it out of the run name)"; exit 1
fi
[[ "$DRY_RUN" == "1" ]] || preflight
if [[ " $SCENARIOS " == *" seq_shell"* ]]; then
    log "WARNING: shell scenarios on this box are NOT sandboxed (codex runs as $(id -un) and can read the disk, incl. the gold"
    log "         answers); run them on the cluster (python -m qatfd.k8s ... --sweep codex_ablation) or scan the traces afterwards"
fi
mkdir -p "$LOG_DIR"
cd "$QATFD_DIR"
log "benchmark=$BENCHMARK collection=${BASE_COLLECTION:-<yaml default>} split=$SPLIT expected_n=$EXPECTED_N seeds='$SEEDS' scenarios='$SCENARIOS'${VARIANT:+ variant=$VARIANT}$( [[ $ANY_ISO -eq 1 ]] && echo " agent_model=$AGENT_MODEL map_model=$MAP_MODEL enrich_batch=$ENRICH_BATCH window=$MAX_PREVIOUS_QUERIES" )${JUDGE_MODEL:+ judge=$JUDGE_MODEL}${SHUFFLE_GROUP_KEY:+ shuffle_by=$SHUFFLE_GROUP_KEY}${QA_PAIRS_PATH:+ qa_pairs=$QA_PAIRS_PATH} run_root=$RUN_ROOT"

# seed-major order: every scenario at seed 0 first, then seed 1, ... so a full cross-scenario comparison
# exists after the first |SCENARIOS| runs rather than only at the end of the sweep.
declare -a summary=()
for seed in $SEEDS; do
    for scenario in $SCENARIOS; do
        label="codex_${scenario}${VARIANT:+_$VARIANT}_${SPLIT}_s${seed}"
        workspace="$(workspace_dir "$label")"
        overrides="systems=codex benchmarks=$BENCHMARK experiments.split=$SPLIT experiments.shuffle_seed=$seed \
experiments.run_name=$label systems.codex_scratch_dir=$workspace $(scenario_overrides "$scenario")$BENCH_OVERRIDES $EXTRA_OVERRIDES"

        existing="$(find_run_dir "$label")"
        fresh=1
        if [[ -n "$existing" ]]; then
            n="$(result_rows "$existing/results.jsonl")"
            if [[ "$n" -ge "$EXPECTED_N" ]]; then
                log "SKIP  $label: complete ($n rows) at $existing"
                summary+=("skip   $label")
                continue
            fi
            if ! is_iso_scenario "$scenario"; then
                log "RESUME $label: $n/$EXPECTED_N rows at $existing (workspace kept: $workspace)"
                overrides="$overrides experiments.resume_dir=$existing"
                fresh=0
            else
                log "RESTART $label: abandoning incomplete $existing ($n/$EXPECTED_N rows); seq_iso* runs are never resumed"
            fi
        fi
        if [[ $fresh -eq 1 ]]; then
            # a fresh run must not inherit a previous attempt's files / AGENTS.md from the shared workspace path
            if [[ -d "$workspace" && "$workspace" == "$WORKSPACES_ROOT/codex_"* ]]; then
                log "START $label (clearing stale workspace $workspace)"
                [[ "$DRY_RUN" == "1" ]] || rm -rf "$workspace"
            else
                log "START $label"
            fi
        fi

        cmd=("$PYTHON" -m qatfd.runner $overrides)
        if [[ "$DRY_RUN" == "1" ]]; then
            is_iso_scenario "$scenario" && echo "  (wipe all non-corpus collections on $CHROMA_HOST:$CHROMA_PORT; reset $BASE_COLLECTION's chunk metadata)"
            printf '  %q' "${cmd[@]}"; printf '\n'
            summary+=("dry    $label")
            continue
        fi

        # seq_iso*: start from the pristine corpus (no collections from an earlier run, no agent-written metadata)
        if is_iso_scenario "$scenario"; then
            if ! wipe_collections; then
                log "FAILED $label: could not wipe collections; skipping"
                summary+=("FAILED(wipe) $label")
                continue
            fi
            if ! reset_corpus_metadata; then
                log "FAILED $label: could not reset $BASE_COLLECTION's chunk metadata; skipping"
                summary+=("FAILED(reset) $label")
                continue
            fi
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
