#!/usr/bin/env bash
# Sweep {rag_llm, search_agent, qatfd_search_agent} x {3 models} on OfficeQA (dev split).
#
#   - rag_llm uses top_k=10 (the agent systems take no top_k).
#   - Default worker count (ExperimentConfig.workers = 32); not overridden here.
#   - Reads from the already-running ChromaDB server (host/port below); OfficeQA's scorer
#     is the deterministic numeric cup scorer, so no judge model is needed.
#
# Runs are written to results/officeqa/<system>/<model-tag>_<timestamp>/ — the model id
# (slashes replaced with dashes) is the run-name prefix so each model is distinguishable.
#
# Env overrides:
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
SYSTEMS=(rag_llm search_agent qatfd_search_agent)

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  provider="${entry##*|}"
  tag="${model//\//-}"   # results-dir-safe label (no slashes)
  for system in "${SYSTEMS[@]}"; do
    extra=()
    [ "$system" = "rag_llm" ] && extra+=( "systems.top_k=10" )
    [ -n "$provider" ] && extra+=( "systems.llm_provider_order=[${provider}]" )
    echo "==================================================================="
    echo "=== officeqa | ${system} | ${model} ${provider:+(provider=$provider)}"
    echo "==================================================================="
    "$PYTHON" -m qatfd.runner \
      benchmarks=officeqa \
      systems="$system" \
      systems.llm_model="$model" \
      benchmarks.chroma_server_host="$CHROMA_HOST" \
      benchmarks.chroma_server_port="$CHROMA_PORT" \
      experiments.run_name="$tag" \
      "${extra[@]}"
  done
done

echo
echo "All 9 runs complete. Reports: results/officeqa/<system>/<model-tag>_<timestamp>/report.csv"
