#!/usr/bin/env bash
# Strong default retrieval pipeline (page FTS + rows + LateOn + LLM planner + decompose/retry).
#
# Usage, from skunk/retrieval:
#   export OPENROUTER_API_KEY=...   # or GEMINI_API_KEY + SKUNK_LLM_PROVIDER=gemini
#   ./run.sh
#   ./run.sh --limit 10
#
# Environment:
#   SKUNK_DATA_DIR              corpus root (default: /tmp/officeqa)
#   SKUNK_LLM_PROVIDER          openrouter | gemini (default: openrouter)
#   SKUNK_LLM_MODEL             model id (default: google/gemini-2.5-flash)
#   EVAL_K                      final top-k (default: 500)
#   SKUNK_DISABLE               comma list: lateon,llm,decompose,retry,rows
#   OFFICEQA_*                  legacy overrides still supported

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKUNK_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$SKUNK_DIR/.venv/bin/python"

DATA_DIR="${SKUNK_DATA_DIR:-${OFFICEQA_DATA_DIR:-/tmp/officeqa}}"
RESULTS_DIR="${OFFICEQA_RESULTS_DIR:-$DATA_DIR/results}"
EVAL_K="${EVAL_K:-500}"
TOP_K_SUFFIX="top_k${EVAL_K}"

PAGE_TABLE_INDEX="${OFFICEQA_PAGE_TABLE_INDEX:-$DATA_DIR/page_table.sqlite}"
LATEON_DIR="${OFFICEQA_LATEON_DIR:-$DATA_DIR/lateon}"
LATEON_SENTINEL="$LATEON_DIR/officeqa_lateon_records.jsonl"
BENCHMARK_CSV="$DATA_DIR/officeqa_pro.csv"

LLM_PROVIDER="${SKUNK_LLM_PROVIDER:-openrouter}"
LLM_MODEL="${SKUNK_LLM_MODEL:-${OFFICEQA_OPENROUTER_MODEL:-google/gemini-2.5-flash}}"
LATEON_DEVICE="${OFFICEQA_LATEON_DEVICE:-}"

PAGE_K="${OFFICEQA_PAGE_K:-800}"
TABLE_K="${OFFICEQA_TABLE_K:-800}"
ROW_K="${OFFICEQA_ROW_K:-200}"
FILE_K="${OFFICEQA_FILE_K:-80}"
LATEON_CANDIDATE_K="${OFFICEQA_LATEON_CANDIDATE_K:-500}"
LATEON_RERANK_K="${OFFICEQA_LATEON_RERANK_K:-1000}"

PAGE_TABLE_RESULT_NAME="strong_${LLM_PROVIDER}_${TOP_K_SUFFIX}"
PAGE_TABLE_ROWS="${OFFICEQA_PAGE_TABLE_ROWS:-$RESULTS_DIR/${PAGE_TABLE_RESULT_NAME}_rows.jsonl}"
PAGE_TABLE_SUMMARY="${OFFICEQA_PAGE_TABLE_SUMMARY:-$RESULTS_DIR/${PAGE_TABLE_RESULT_NAME}_summary.json}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$LLM_PROVIDER" == "openrouter" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY required for SKUNK_LLM_PROVIDER=openrouter" >&2
  exit 2
fi
if [[ "$LLM_PROVIDER" == "gemini" && -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY or GOOGLE_API_KEY required for SKUNK_LLM_PROVIDER=gemini" >&2
  exit 2
fi

LIMIT_ARGS=()
while (($#)); do
  case "$1" in
    --limit=*) LIMIT_ARGS=(--limit "${1#*=}") ;;
    --limit)
      shift
      LIMIT_ARGS=(--limit "${1:?--limit requires a value}")
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift || true
done

LATEON_ARGS=(--lateon-search-batch-size "${OFFICEQA_LATEON_SEARCH_BATCH_SIZE:-2048}")
if [[ -n "$LATEON_DEVICE" ]]; then
  LATEON_ARGS+=(--lateon-device "$LATEON_DEVICE")
fi

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --llm-provider "$LLM_PROVIDER"
  --openrouter-model "$LLM_MODEL"
  --page-k "$PAGE_K"
  --table-k "$TABLE_K"
  --row-k "$ROW_K"
  --file-k "$FILE_K"
  --lateon-folder "$LATEON_DIR"
  --lateon-candidate-k "$LATEON_CANDIDATE_K"
  --lateon-rerank-k "$LATEON_RERANK_K"
  --lateon-expand-page-records
  --lateon-include-row-records
)

hr()   { echo "----------------------------------------" >&2; }
step() { hr; echo "  $*" >&2; hr; }
skip() { echo "  $* already exists" >&2; }
run()  { PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -m skunk_retrieval "$@"; }

step "Corpus"
if [[ -f "$BENCHMARK_CSV" ]] && [[ -d "$DATA_DIR/treasury_bulletins_parsed/jsons" || -d "$DATA_DIR/jsons" ]]; then
  skip "$DATA_DIR"
else
  echo "  Downloading to $DATA_DIR" >&2
  OFFICEQA_DATA_DIR="$DATA_DIR" "$PYTHON" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="databricks/officeqa",
    repo_type="dataset",
    allow_patterns=[
        "treasury_bulletins_parsed/jsons/*.json",
        "treasury_bulletins_parsed/transformed_page_level/*.txt",
        "treasury_bulletins_parsed/transformed/*.txt",
        "officeqa_pro.csv",
    ],
    local_dir=os.environ["OFFICEQA_DATA_DIR"],
)
PYEOF
fi

CANONICAL_FILE="$DATA_DIR/.skunk/canonical_source.json"
if [[ ! -f "$CANONICAL_FILE" ]]; then
  step "Format selection (LLM judges; auto-accept recommendation)"
  run choose-format --data-dir "$DATA_DIR" --llm-provider "$LLM_PROVIDER" --llm-model "$LLM_MODEL" --non-interactive --rich
else
  skip "$CANONICAL_FILE"
fi

step "Offline preprocess (FTS + LateOn)"
if [[ -f "$PAGE_TABLE_INDEX" && -f "$LATEON_SENTINEL" ]]; then
  skip "indexes at $DATA_DIR"
else
  run preprocess --data-dir "$DATA_DIR" --skip-format-choice --llm-provider "$LLM_PROVIDER" --llm-model "$LLM_MODEL" "${LATEON_ARGS[@]}" --rich
fi

step "Eval  k=$EVAL_K  provider=$LLM_PROVIDER"
run eval-page-table \
  --data-dir "$DATA_DIR" \
  --index-file "$PAGE_TABLE_INDEX" \
  --benchmark-csv "$BENCHMARK_CSV" \
  --k "$EVAL_K" \
  --rows-jsonl "$PAGE_TABLE_ROWS" \
  "${COMMON_ARGS[@]}" \
  "${LATEON_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --rich | tee "$PAGE_TABLE_SUMMARY"

hr
echo "  Done." >&2
echo "  Rows:    $PAGE_TABLE_ROWS" >&2
echo "  Summary: $PAGE_TABLE_SUMMARY" >&2
hr
