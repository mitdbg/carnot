#!/usr/bin/env bash
# Sweep {RAG-LLM, SearchAgent, QATFD} x {3 models} on OfficeQA (dev split).
#
#   - rag_llm uses top_k=10 (the agent arms take no top_k).
#   - SearchAgent and QATFD are the same `search_agent` system with different retrieval tool
#     sets (read_document + prune are always present):
#       search_agent   grep + vector + read
#       qatfd          grep + semantic-filter + read
#   - Default worker count (ExperimentConfig.workers = 32); not overridden here.
#   - Reads from the already-running ChromaDB server (host/port below); OfficeQA's scorer
#     is the deterministic numeric cup scorer, so no judge model is needed.
#
# Runs are written to results/officeqa/<system>/<arm>_<model-tag>_<timestamp>/. Both agent arms
# share the `search_agent` dir, so the arm label leads the run-name to keep them apart; the
# model id (slashes replaced with dashes) follows so each model is distinguishable. The eval
# scripts recover the arm from each run's config.yaml (eval/variants.py), not from this name.
#
# Env overrides:
#   ARMS=rag_llm,search_agent,qatfd          # comma-separated subset of arms to run
#   CHROMA_HOST=127.0.0.1 CHROMA_PORT=8001   # the tmux ChromaDB server endpoint
#   PYTHON=python3                            # interpreter (must have qatfd importable)
set -euo pipefail

# Run from the qatfd repo root so Hydra's results/ and the skunk/.env lookup resolve.
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"

# model|provider-pin — pin the two Qwen models to the provider whose prompt-cache pricing we
# costed (Parasail / io.net); empty pin lets OpenRouter pick (gemini-3.1-flash-lite).
MODELS=(
  "google/gemini-3.1-flash-lite|"
  "qwen/qwen3.6-35b-a3b|parasail"
  "qwen/qwen3.6-27b|io-net"
)

# arm | systems group | arm-specific overrides (space-separated)
ARM_CONFIGS=(
  "rag_llm|rag_llm|systems.top_k=10"
  "search_agent|search_agent|systems.include_search_corpus=true systems.include_grep_corpus=true systems.include_semantic_filter=false"
  "qatfd|search_agent|systems.include_search_corpus=false systems.include_grep_corpus=true systems.include_semantic_filter=true"
)

ARMS="${ARMS:-rag_llm,search_agent,qatfd}"
n_runs=0

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  provider="${entry##*|}"
  tag="${model//\//-}"   # results-dir-safe label (no slashes)
  for arm_entry in "${ARM_CONFIGS[@]}"; do
    IFS='|' read -r arm system arm_overrides <<< "$arm_entry"
    if ! [[ ",$ARMS," == *",$arm,"* ]]; then
      continue
    fi
    extra=()
    # shellcheck disable=SC2206 — deliberate word-splitting of the override list
    extra+=( ${arm_overrides} )
    [ -n "$provider" ] && extra+=( "inference.llm_provider_order=[${provider}]" )
    echo "==================================================================="
    echo "=== officeqa | ${arm} | ${model} ${provider:+(provider=$provider)}"
    echo "==================================================================="
    "$PYTHON" -m qatfd.runner \
      benchmarks=officeqa \
      systems="$system" \
      inference.llm_model="$model" \
      benchmarks.chroma_server_host="$CHROMA_HOST" \
      benchmarks.chroma_server_port="$CHROMA_PORT" \
      experiments.run_name="${arm}_${tag}" \
      "${extra[@]}"
    n_runs=$((n_runs + 1))
  done
done

echo
echo "All ${n_runs} runs complete. Reports: results/officeqa/<system>/<arm>_<model-tag>_<timestamp>/report.csv"
