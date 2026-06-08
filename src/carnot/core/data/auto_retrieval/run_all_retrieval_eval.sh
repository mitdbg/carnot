#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_retrieval_eval.sh
#   ./run_all_retrieval_eval.sh "1,2,3" 20
#   ./run_all_retrieval_eval.sh "1,2,3" 20 eval_results_all
#
# Args:
#   $1 subsets    Comma-separated subset ids (default: 1,2,3)
#   $2 k          Top-K cutoff (default: 20)
#   $3 output_dir Optional output directory passed to runner

SUBSETS="${1:-1,2,3}"
K="${2:-20}"
OUTPUT_DIR="${3:-}"

SETUPS=(
  dense
  meta_dense
  splade
  meta_splade
  colbert
  meta_colbert
  dense_rerank
  splade_rerank
  colbert_rerank
  meta_rerank
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running retrieval evals"
echo "  subsets: $SUBSETS"
echo "  k:       $K"
if [[ -n "$OUTPUT_DIR" ]]; then
  echo "  out:     $OUTPUT_DIR"
else
  echo "  out:     (default from runner.py)"
fi

echo
for setup in "${SETUPS[@]}"; do
  echo "============================================================"
  echo "Setup: $setup"
  if [[ -n "$OUTPUT_DIR" ]]; then
    python -m retrieval_eval.runner --setup "$setup" --subsets "$SUBSETS" --k "$K" --output-dir "$OUTPUT_DIR"
  else
    python -m retrieval_eval.runner --setup "$setup" --subsets "$SUBSETS" --k "$K"
  fi
done

echo
echo "All setups completed."
