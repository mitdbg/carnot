#!/usr/bin/env bash
# init container: pull the cell's chroma store and benchmark files from the data bucket onto the pod's
# NVMe-backed /data volume. Idempotent (aws s3 sync), so a restarted init container resumes.
set -euo pipefail

CELL="python3 /scripts/cell.py"
LABEL="$($CELL label)"
STORE_PREFIX="$($CELL store_prefix)"
echo "[fetch] cell $JOB_COMPLETION_INDEX ($LABEL): store s3://$DATA_BUCKET/$STORE_PREFIX -> /data/chromadb"

# more parallel requests than the default 10: the cleaned-pages tree is tens of thousands of small objects
aws configure set default.s3.max_concurrent_requests "${MAX_CONCURRENT_REQUESTS:-64}"
aws configure set default.s3.max_queue_size 10000

t0=$(date +%s)
aws s3 sync "s3://$DATA_BUCKET/${STORE_PREFIX%/}/" /data/chromadb --only-show-errors
echo "[fetch] store done in $(( $(date +%s) - t0 ))s: $(du -sh /data/chromadb | cut -f1)"

while IFS=$'\t' read -r src dest include exclude; do
    [[ -n "$src" ]] || continue
    if [[ "$src" == */ ]]; then
        # a prefix: whole thing, or only `include` (e.g. metadata_rank*.json next to GBs of embeddings), minus `exclude`
        filters=()
        [[ -z "$include" ]] || filters+=(--exclude "*" --include "$include")
        [[ -z "$exclude" ]] || filters+=(--exclude "$exclude")
        mkdir -p "/data/benchmarks/$dest"
        aws s3 sync "s3://$DATA_BUCKET/$src" "/data/benchmarks/${dest%/}/" --only-show-errors "${filters[@]}"
    else
        mkdir -p "$(dirname "/data/benchmarks/$dest")"
        aws s3 cp "s3://$DATA_BUCKET/$src" "/data/benchmarks/$dest" --only-show-errors
    fi
    echo "[fetch] $src -> /data/benchmarks/$dest"
done < <($CELL --data)

echo "[fetch] done in $(( $(date +%s) - t0 ))s: $(du -sh /data | cut -f1) on /data"
