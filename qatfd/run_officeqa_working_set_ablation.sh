#!/usr/bin/env bash
# Working-set tool ablation: the SearchAgent on OfficeQA (dev split), sweeping the four
# retrieval tool sets below. Every run uses the `ablation_search_agent` system, which always
# has read_document + prune and toggles vector / grep / semantic_filter by config; the config
# label is the run-name, so the four runs land in distinct dirs under
#   results/officeqa/ablation_search_agent/<label>_<timestamp>/
#
#   grep_read           grep + read              (vector off, grep on,  sem off)
#   vector_read         vector-search + read     (vector on,  grep off, sem off)
#   grep_vector_read    grep + vector + read     (vector on,  grep on,  sem off)  = SearchAgent
#   sem_read            semantic-filter + read   (vector off, grep off, sem on)
#
# All model calls go through OpenRouter (needs OPENROUTER_API_KEY in the environment) and read
# from the already-running ChromaDB server (host/port below). Run with the skunk venv active.
#
# Env overrides:
#   MODEL=qwen/qwen3.6-35b-a3b    # full OpenRouter model id
#   PROVIDER=parasail             # OpenRouter provider pin; empty string = no pin (let OR pick)
#   JUDGE_PROVIDERS=parasail,akashml,deepinfra  # semantic_filter judge provider order; "" = client default
#   CHROMA_HOST=127.0.0.1  CHROMA_PORT=8001  PYTHON=python3
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"

# Model defaults to the qwen run; override MODEL/PROVIDER for another. PROVIDER="" pins nothing.
MODEL="${MODEL:-qwen/qwen3.6-35b-a3b}"
PROVIDER="${PROVIDER-parasail}"
# Provider order for the semantic_filter judge calls only (sem_read arm); "" = client default.
JUDGE_PROVIDERS="${JUDGE_PROVIDERS-parasail,akashml,deepinfra}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set in the environment}"

# label | tool_vector | tool_grep | tool_semantic_filter
CONFIGS=(
  "grep_read|false|true|false"
  "vector_read|true|false|false"
  "grep_vector_read|true|true|false"
  "sem_read|false|false|true"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label vec grep sem <<< "$entry"
  provider_ovr=()
  [ -n "$PROVIDER" ] && provider_ovr+=( "inference.llm_provider_order=[${PROVIDER}]" )
  [ -n "$JUDGE_PROVIDERS" ] && provider_ovr+=( "systems.semantic_filter_provider_order=[${JUDGE_PROVIDERS}]" )
  echo "==================================================================="
  echo "=== officeqa | ablation_search_agent | ${label}  (vector=$vec grep=$grep sem=$sem) | model=$MODEL ${PROVIDER:+provider=$PROVIDER}"
  echo "==================================================================="
  "$PYTHON" -m qatfd.runner \
    benchmarks=officeqa \
    systems=ablation_search_agent \
    inference.llm_model="$MODEL" \
    "${provider_ovr[@]}" \
    systems.tool_vector="$vec" \
    systems.tool_grep="$grep" \
    systems.tool_semantic_filter="$sem" \
    benchmarks.chroma_server_host="$CHROMA_HOST" \
    benchmarks.chroma_server_port="$CHROMA_PORT" \
    experiments.run_name="$label"
done

echo
echo "All 4 ablation runs complete."
echo "Reports: results/officeqa/ablation_search_agent/<label>_<timestamp>/report.csv"
