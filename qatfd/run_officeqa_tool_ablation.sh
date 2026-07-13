#!/usr/bin/env bash
# Tool-ablation experiment: qwen3.6-35b-a3b + the SearchAgent on OfficeQA (dev split), sweeping
# the six retrieval tool sets below. Every run uses the `ablation_search_agent` system, which
# always has read_document + prune and toggles vector / grep / semantic_filter by config; the
# config label is the run-name, so the six runs land in distinct dirs under
#   results/officeqa/ablation_search_agent/<label>_<timestamp>/
#
#   grep_read           grep + read                        (vector off, grep on,  sem off)
#   vector_read         vector-search + read               (vector on,  grep off, sem off)
#   sem_read            semantic-filter + read             (vector off, grep off, sem on)
#   grep_vector_read    grep + vector + read  (= SearchAgent) (vector on, grep on, sem off)
#   grep_sem_read       grep + sem + read     (= QATFD)      (vector off, grep on, sem on)
#   all_tools           grep + sem + vector + read           (vector on, grep on, sem on)
#
# Default worker count. Reads from the already-running ChromaDB server (host/port below).
#
# Env overrides:
#   MODEL=qwen/qwen3.6-35b-a3b     # full OpenRouter model id
#   PROVIDER=parasail             # OpenRouter provider pin; empty string = no pin (let OR pick)
#   CHROMA_HOST=127.0.0.1  CHROMA_PORT=8001  PYTHON=python3
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"

# Model defaults to the qwen run; override MODEL/PROVIDER for another. PROVIDER="" pins nothing.
MODEL="${MODEL:-qwen/qwen3.6-35b-a3b}"
PROVIDER="${PROVIDER-parasail}"

# label | tool_vector | tool_grep | tool_semantic_filter
CONFIGS=(
  "grep_read|false|true|false"
  "vector_read|true|false|false"
  "sem_read|false|false|true"
  "grep_vector_read|true|true|false"
  "grep_sem_read|false|true|true"
  "all_tools|true|true|true"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label vec grep sem <<< "$entry"
  provider_ovr=()
  [ -n "$PROVIDER" ] && provider_ovr+=( "systems.llm_provider_order=[${PROVIDER}]" )
  echo "==================================================================="
  echo "=== officeqa | ablation_search_agent | ${label}  (vector=$vec grep=$grep sem=$sem) | model=$MODEL ${PROVIDER:+provider=$PROVIDER}"
  echo "==================================================================="
  "$PYTHON" -m qatfd.runner \
    benchmarks=officeqa \
    systems=ablation_search_agent \
    systems.llm_model="$MODEL" \
    "${provider_ovr[@]}" \
    systems.tool_vector="$vec" \
    systems.tool_grep="$grep" \
    systems.tool_semantic_filter="$sem" \
    benchmarks.chromadb_host="$CHROMA_HOST" \
    benchmarks.chromadb_port="$CHROMA_PORT" \
    experiments.run_name="$label"
done

echo
echo "All 6 ablation runs complete."
echo "Reports: results/officeqa/ablation_search_agent/<label>_<timestamp>/report.csv"
echo "Tool breakdown: python3 <scratchpad>/tool_metrics_ablation.py"
