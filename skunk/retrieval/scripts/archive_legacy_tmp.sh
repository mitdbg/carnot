#!/usr/bin/env bash
# Move legacy /tmp artifacts into _old/ subfolders (safe to re-run).
set -euo pipefail

DATA_DIR="${SKUNK_DATA_DIR:-/tmp/officeqa}"
OLD_DIR="$DATA_DIR/_old"
RESULTS_OLD="$OLD_DIR/results"

mkdir -p "$OLD_DIR" "$RESULTS_OLD"

if [[ -f "$DATA_DIR/bm25.pkl" ]]; then
  mv -f "$DATA_DIR/bm25.pkl" "$OLD_DIR/"
  echo "moved bm25.pkl -> $OLD_DIR/"
fi

if [[ -d "$DATA_DIR/lateon_page_table" ]]; then
  mv -f "$DATA_DIR/lateon_page_table" "$OLD_DIR/"
  echo "moved lateon_page_table -> $OLD_DIR/"
fi

for pattern in bm25_ hybrid_ page_table_lateon_openrouter page_table_rows page_table_lateon_page_table page_table_lateon_summary; do
  for f in "$DATA_DIR/results"/${pattern}*; do
    [[ -e "$f" ]] || continue
    mv -f "$f" "$RESULTS_OLD/"
    echo "moved $(basename "$f") -> $RESULTS_OLD/"
  done
done

for f in /tmp/officeqa_hybrid_smoke*.jsonl; do
  [[ -e "$f" ]] || continue
  mv -f "$f" "$OLD_DIR/"
  echo "moved $(basename "$f") -> $OLD_DIR/"
done

echo "Done. Active results should use results/strong_* from the current pipeline."
