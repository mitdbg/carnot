#!/usr/bin/env bash
# Matrix driver over {benchmarks} x {arms} — one Hydra multirun per arm.
#
# An "arm" is one row of the main results table. `rag_llm` is its own system; the rest are
# retrieval tool sets of the single `search_agent` system (read_document + prune are always
# present), so they are selected by config flags rather than by a system name:
#
#   rag_llm        RAG-LLM, top_k=10
#   search_agent   grep + vector + read              (the vanilla SearchAgent)
#   qatfd          grep + semantic-filter + read     (QATFD)
#
# Because every agent arm writes under results/<benchmark>/search_agent/, the arm label is
# used as the run-name prefix so its runs stay identifiable:
#   results/<benchmark>/search_agent/<arm>_<timestamp>/
# The eval scripts do not rely on that name — they recover each run's arm from its config.yaml
# (see eval/variants.py) — it is purely for humans reading the results tree.
#
# Each axis is a comma-separated list; override any axis or knob via env vars:
#   BENCHMARKS="officeqa"                       ./run_all.sh   # one benchmark
#   ARMS="rag_llm,search_agent"                 ./run_all.sh   # subset of arms
#   SAMPLE=20 WORKERS=16                        ./run_all.sh   # 20-question smoke
#   SPLIT=test                                  ./run_all.sh   # held-out test split
#   EMB_PROVIDER=vllm                           ./run_all.sh   # query-embedding backend
#
# Anything after `--` is forwarded verbatim as extra Hydra overrides, e.g.:
#   ./run_all.sh -- systems.top_k=30 inference.llm_model=google/gemini-3.5-flash
#
# Uses python3 (no `python` on PATH in this repo).
set -euo pipefail

# Run from the qatfd repo root so Hydra's run/sweep dirs (configs/config.yaml)
# and qatfd.env's skunk/.env lookup resolve consistently.
cd "$(dirname "$0")"

BENCHMARKS="${BENCHMARKS:-officeqa,browsecomp_plus}"
ARMS="${ARMS:-rag_llm,search_agent,qatfd}"
SAMPLE="${SAMPLE:-}"
WORKERS="${WORKERS:-32}"
SPLIT="${SPLIT:-dev}"
RUN_NAME="${RUN_NAME:-}"
EMB_PROVIDER="${EMB_PROVIDER:-openrouter}"

# arm | systems group | arm-specific overrides (space-separated)
ARM_CONFIGS=(
  "rag_llm|rag_llm|systems.top_k=10"
  "search_agent|search_agent|systems.include_search_corpus=true systems.include_grep_corpus=true systems.include_semantic_filter=false"
  "qatfd|search_agent|systems.include_search_corpus=false systems.include_grep_corpus=true systems.include_semantic_filter=true"
)

for entry in "${ARM_CONFIGS[@]}"; do
  IFS='|' read -r arm system arm_overrides <<< "$entry"
  if ! [[ ",$ARMS," == *",$arm,"* ]]; then
    echo "--- skipping arm $arm (not in ARMS=$ARMS)"
    continue
  fi
  # RUN_NAME, when set, is a prefix shared by every arm of this sweep.
  run_name="${RUN_NAME:+${RUN_NAME}_}${arm}"
  overrides=(
    --multirun
    "benchmarks=${BENCHMARKS}"
    "systems=${system}"
    "experiments.split=${SPLIT}"
    "experiments.workers=${WORKERS}"
    "experiments.run_name=${run_name}"
    "inference.emb_provider=${EMB_PROVIDER}"
  )
  # shellcheck disable=SC2206 — deliberate word-splitting of the override list
  overrides+=( ${arm_overrides} )
  [ -n "$SAMPLE" ] && overrides+=( "experiments.sample=${SAMPLE}" )

  echo "==================================================================="
  echo "=== arm=${arm} | systems=${system} | benchmarks=${BENCHMARKS}"
  echo "==================================================================="
  ( set -x; python3 -m qatfd.runner "${overrides[@]}" "$@" )
done
