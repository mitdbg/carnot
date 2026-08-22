#!/usr/bin/env bash
# Working-set tool ablation: the SearchAgent on OfficeQA (dev split), sweeping the four
# retrieval tool sets below. Every run uses the `search_agent` system, which always
# has read_document + prune and toggles vector / grep / semantic_filter by config; the config
# label is the run-name, so the four runs land in distinct dirs under
#   results/officeqa/search_agent/<label>_<timestamp>/
#
#   grep_read           grep + read                  (vector off, grep on,  sem off)
#   vector_read         vector-search + read         (vector on,  grep off, sem off)
#   grep_vector_read    grep + vector + read         (vector on,  grep on,  sem off)  = SearchAgent
#   sem_read            semantic-filter + read       (vector off, grep off, sem on)
#   all_tools           grep + vector + sem + read   (vector on,  grep on,  sem on)
#
# All model calls go through OpenRouter (needs OPENROUTER_API_KEY in the environment) and read
# from the already-running ChromaDB server (host/port below). Run with the skunk venv active.
#
# Env overrides:
#   MODEL=google/gemini-3.5-flash        # agent model (full OpenRouter id)
#   PROVIDER=                            # agent provider pin; empty string = no pin (let OR pick)
#   JUDGE_MODEL=qwen/qwen3.6-35b-a3b     # semantic_filter judge model; "" = use the agent model
#   JUDGE_PROVIDERS=parasail,akashml,deepinfra  # semantic_filter judge provider order; "" = client default
#   ARMS=vector_read,grep_vector_read,sem_read,all_tools  # comma-separated labels to run; "" = all
#   WORKING_SET_OFF=true          # disable the working-set abstraction (tools force fetch+read);
#                                 # run labels get a _ws_off suffix so runs land in distinct dirs
#   CHROMA_HOST=127.0.0.1  CHROMA_PORT=8001  PYTHON=python3
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"

# Agent model defaults to gemini-3.5-flash, unpinned: PROVIDER defaults to "" because the qwen
# pin (parasail) does not serve Gemini — set PROVIDER only when it actually serves MODEL.
MODEL="${MODEL:-google/gemini-3.5-flash}"
PROVIDER="${PROVIDER-}"
# Judge model + provider order for the semantic_filter judge calls only (sem_read/all_tools arms).
# JUDGE_MODEL="" = fall back to the agent model; JUDGE_PROVIDERS="" = client default routing.
# NOTE: a new JUDGE_MODEL needs an inference.llm_context_limits entry (configs/inference/base.yaml)
# or the semantic filter won't head-truncate oversized candidate docs for it.
JUDGE_MODEL="${JUDGE_MODEL-qwen/qwen3.6-35b-a3b}"
JUDGE_PROVIDERS="${JUDGE_PROVIDERS-parasail,akashml,deepinfra}"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set in the environment}"

# label | include_search_corpus | include_grep_corpus | include_semantic_filter
CONFIGS=(
  "grep_read|false|true|false"
  "vector_read|true|false|false"
  "grep_vector_read|true|true|false"
  "sem_read|false|false|true"
  "all_tools|true|true|true"
)

# ARMS: comma-separated subset of labels to run (e.g. resume a partial sweep); "" = all.
ARMS="${ARMS-}"
# WORKING_SET_OFF=true disables the working-set abstraction (systems.working_set_off).
WORKING_SET_OFF="${WORKING_SET_OFF:-false}"

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label vec grep sem <<< "$entry"
  if [ -n "$ARMS" ] && ! [[ ",$ARMS," == *",$label,"* ]]; then
    echo "--- skipping $label (not in ARMS=$ARMS)"
    continue
  fi
  provider_ovr=()
  [ -n "$PROVIDER" ] && provider_ovr+=( "inference.llm_provider_order=[${PROVIDER}]" )
  [ -n "$JUDGE_MODEL" ] && provider_ovr+=( "systems.semantic_filter_model=${JUDGE_MODEL}" )
  [ -n "$JUDGE_PROVIDERS" ] && provider_ovr+=( "systems.semantic_filter_provider_order=[${JUDGE_PROVIDERS}]" )
  run_name="$label"
  if [ "$WORKING_SET_OFF" = "true" ]; then
    provider_ovr+=( "systems.working_set_off=true" )
    run_name="${label}_ws_off"
  fi
  echo "==================================================================="
  echo "=== officeqa | search_agent | ${run_name}  (vector=$vec grep=$grep sem=$sem working_set_off=$WORKING_SET_OFF)"
  echo "===   agent=$MODEL${PROVIDER:+ (pin: $PROVIDER)} | judge=${JUDGE_MODEL:-$MODEL}${JUDGE_PROVIDERS:+ (pin: $JUDGE_PROVIDERS)}"
  echo "==================================================================="
  "$PYTHON" -m qatfd.runner \
    benchmarks=officeqa \
    systems=search_agent \
    inference.llm_model="$MODEL" \
    "${provider_ovr[@]}" \
    systems.include_search_corpus="$vec" \
    systems.include_grep_corpus="$grep" \
    systems.include_semantic_filter="$sem" \
    benchmarks.chroma_server_host="$CHROMA_HOST" \
    benchmarks.chroma_server_port="$CHROMA_PORT" \
    experiments.run_name="$run_name"
done

echo
echo "All ${#CONFIGS[@]} ablation runs complete."
echo "Reports: results/officeqa/search_agent/<label>_<timestamp>/report.csv"
