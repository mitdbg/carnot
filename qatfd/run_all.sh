#!/usr/bin/env bash
# Matrix driver over {benchmarks} x {systems} via Hydra multirun.
#
# Each axis is a comma-separated list of config-group options (file stems under
# configs/benchmarks and configs/systems). Defaults to the full matrix; override
# any axis or knob via env vars:
#   BENCHMARKS="officeqa"                       ./run_all.sh   # one benchmark
#   SYSTEMS="rag_llm,search_agent"              ./run_all.sh   # subset of systems
#   SAMPLE=20 WORKERS=16                        ./run_all.sh   # 20-question smoke
#   SPLIT=test                                  ./run_all.sh   # held-out test split
#   EMB_PROVIDER=vllm                           ./run_all.sh   # query-embedding backend
#
# Anything after `--` is forwarded verbatim as extra Hydra overrides, e.g.:
#   ./run_all.sh -- systems.top_k=30 skunk.llm_model=gemini-3.5-pro
#
# Uses python3 (no `python` on PATH in this repo).
set -euo pipefail

# Run from the qatfd repo root so Hydra's run/sweep dirs (configs/config.yaml)
# and qatfd.env's skunk/.env lookup resolve consistently.
cd "$(dirname "$0")"

BENCHMARKS="${BENCHMARKS:-officeqa,browsecomp_plus}"
SYSTEMS="${SYSTEMS:-rag_llm,search_agent,qatfd_search_agent}"
SAMPLE="${SAMPLE:-}"
WORKERS="${WORKERS:-32}"
SPLIT="${SPLIT:-dev}"
RUN_NAME="${RUN_NAME:-sweep}"
EMB_PROVIDER="${EMB_PROVIDER:-openrouter}"

overrides=(
  --multirun
  "benchmarks=${BENCHMARKS}"
  "systems=${SYSTEMS}"
  "experiments.split=${SPLIT}"
  "experiments.workers=${WORKERS}"
  "experiments.run_name=${RUN_NAME}"
  "systems.emb_provider=${EMB_PROVIDER}"
)
[ -n "$SAMPLE" ] && overrides+=( "experiments.sample=${SAMPLE}" )

set -x
python3 -m qatfd.runner "${overrides[@]}" "$@"
