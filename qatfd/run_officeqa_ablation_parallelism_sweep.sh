#!/usr/bin/env bash
# OfficeQA search-agent ablation × question-level parallelism sweep.
#
# 4 tool sets × 4 worker counts = 16 runs, all on the SAME 10 deterministically-sampled
# OfficeQA dev questions (experiments.sample=10 + a fixed experiments.seed, so every run
# scores the identical subset). Agent model (qwen3.6-35b-a3b) and query embedder
# (qwen3-embedding-8b) are both served by vLLM on engaging and reached over an SSH tunnel
# on :8100 / :8101 (see engaging-scripts/run_vllm_officeqa_ablation.slurm).
#
# Tool sets (read_document + prune are always on; these toggle discovery/narrowing tools):
#   grep_read          grep + read                 (vector off, grep on,  sem off)
#   vector_read        vector-search + read        (vector on,  grep off, sem off)
#   grep_vector_read   grep + vector + read        (vector on,  grep on,  sem off)
#   sem_read           semantic-filter + read      (vector off, grep off, sem on)
#
# Worker counts (experiments.workers = question-level concurrency): 1, 2, 4, 10.
# NOTE on sem_read: the semantic_filter tool judges its candidate docs on its OWN internal
# ThreadPool (skunk filter_docs max_workers=8), which is INDEPENDENT of experiments.workers.
# So it keeps its default 8-way judge parallelism at every worker count — nothing to set.
#
# Idempotent: a run whose results dir (…/<label>_w<N>_<timestamp>/) already holds report.csv
# is treated as complete and skipped. Delete that dir (or its report.csv) to force a rerun.
#
# Env overrides:
#   SEED=0                     RNG seed for the 10-question draw (keep fixed across the sweep)
#   VLLM_HOST=127.0.0.1        host the SSH tunnel forwards from
#   CHAT_PORT=8100 EMB_PORT=8101
#   CHROMA_HOST=127.0.0.1 CHROMA_PORT=8001
#   PYTHON=<...>/venv/bin/python3
#   WORKERS="1 2 4 10"         worker counts to sweep
set -euo pipefail
cd "$(dirname "$0")"

# Absolute path with `..` resolved: a relative `../skunk/...` interpreter makes the venv's
# site.py emit a harmless sys.prefix RuntimeWarning (unresolved `..`). Normalize the DIR via
# cd/pwd (NOT realpath — that would follow the python3 symlink out of the venv to the system
# interpreter and lose the venv's site-packages), keeping the venv's python3 symlink intact.
PYTHON="${PYTHON:-$(cd ../skunk/venv/bin && pwd)/python3}"
SEED="${SEED:-0}"
SAMPLE=10
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
CHAT_PORT="${CHAT_PORT:-8100}"
EMB_PORT="${EMB_PORT:-8101}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
WORKERS="${WORKERS:-1 2 4 10}"

CHAT_MODEL="qwen/qwen3.6-35b-a3b"
EMB_MODEL="Qwen/Qwen3-Embedding-8B"
VLLM_URLS="{${CHAT_MODEL}: \"http://${VLLM_HOST}:${CHAT_PORT}/v1\", ${EMB_MODEL}: \"http://${VLLM_HOST}:${EMB_PORT}/v1\"}"

RESULTS_DIR="results/officeqa/search_agent"

# label | include_search_corpus | include_grep_corpus | include_semantic_filter
CONFIGS=(
  "grep_read|false|true|false"
  "vector_read|true|false|false"
  "grep_vector_read|true|true|false"
  "sem_read|false|false|true"
)

# A run is "done" if a matching results dir already contains report.csv (written only after
# every question finishes). Partial runs (results.jsonl but no report.csv) are NOT skipped.
already_done() {
  local run_name="$1"
  for d in "${RESULTS_DIR}/${run_name}"_*/; do
    [ -f "${d}report.csv" ] && return 0
  done
  return 1
}

echo "Sweep: seed=${SEED} sample=${SAMPLE} | chat=${VLLM_HOST}:${CHAT_PORT} emb=${VLLM_HOST}:${EMB_PORT} | chroma=${CHROMA_HOST}:${CHROMA_PORT}"

# Outer loop = worker count (each system at w=1, then all at w=2, ...); inner = tool set.
for w in $WORKERS; do
  for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label vec grep sem <<< "$entry"
    run_name="${label}_w${w}"

    if already_done "$run_name"; then
      echo "SKIP  ${run_name}  (report.csv already on disk)"
      continue
    fi

    echo "==================================================================="
    echo "=== RUN  ${run_name}  (vector=$vec grep=$grep sem=$sem, workers=$w)"
    echo "==================================================================="
    "$PYTHON" -m qatfd.runner \
      benchmarks=officeqa \
      systems=search_agent \
      experiments.split=dev \
      experiments.sample="$SAMPLE" \
      experiments.seed="$SEED" \
      experiments.workers="$w" \
      inference.llm_provider=vllm \
      inference.llm_model="$CHAT_MODEL" \
      inference.emb_provider=vllm \
      inference.emb_model_id="$EMB_MODEL" \
      inference.llm_default_rpm=100000 \
      "++inference.vllm_base_urls=${VLLM_URLS}" \
      systems.include_search_corpus="$vec" \
      systems.include_grep_corpus="$grep" \
      systems.include_semantic_filter="$sem" \
      benchmarks.chroma_server_host="$CHROMA_HOST" \
      benchmarks.chroma_server_port="$CHROMA_PORT" \
      experiments.run_name="$run_name"
  done
done

echo
echo "Sweep complete. Reports: ${RESULTS_DIR}/<label>_w<N>_<timestamp>/report.csv"
