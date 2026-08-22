#!/usr/bin/env bash
# Overnight working-set ablation sweep: the SearchAgent (vector + grep + read + prune; NO
# semantic filter) across the four working-set ablation cells, in four phases ordered so
# finished results accumulate as early as possible:
#
#   p1_parallel_noreuse   officeqa dev (33 q)      workers=33  fetch_related=false
#   p2_seq_reuse          officeqa dev (33 q)      workers=1   fetch_related=true
#   p3_loi24_noreuse      officeqa_synth 24-q sub  workers=1   fetch_related=false
#   p4_loi24_reuse        officeqa_synth 24-q sub  workers=1   fetch_related=true
#
# The synth subset is the first 4 lines of inquiry x 6 questions (qids computed from the
# qa_pairs file at launch, preserving file order so LOI blocks run consecutively).
#
# Cells per phase (label | working_set_collection_off | id_tracking_off):
#
#   ws_coll_on_id_on     false | false   full working set (collection + id tracking)
#   ws_coll_on_id_off    false | true    collection only (no inclusion/exclusion filters)
#   ws_coll_off_id_on    true  | false   id tracking only (virtual working set over the corpus)
#   ws_coll_off_id_off   true  | true    no working set (every call is fetch+read on the corpus)
#
# Working-set isolation: the registry rehydrates EVERY ws_* collection on the chroma server,
# so this script DELETES all ws_* collections before each run — each run's reuse (and each
# agent's registry-rehydration cost) stays self-contained. The corpus collection is untouched.
#
# The ChromaDB server (host/port below) must serve the officeqa collection; if no server is
# up, this script starts one via skunk/scripts/run_chroma_server.sh over the qatfd officeqa
# store and leaves it running. NOTE: the officeqa HNSW index needs ~9GB+ of free RAM to load —
# run this on the experiment box, not the small dev box.
#
# Env overrides:
#   MODEL=openai/gpt-5.6-luna     # agent model (full OpenRouter id)
#   PROVIDER=                     # agent provider pin; empty = let OpenRouter pick
#   PHASES=p1,p2,p3,p4            # comma-separated subset of phase labels to run
#   CELLS=ws_coll_on_id_on,...    # comma-separated subset of cell labels to run
#   CHROMA_HOST=127.0.0.1  CHROMA_PORT=8001
#   PYTHON=/home/ubuntu/carnot/skunk/venv/bin/python3
set -uo pipefail   # no -e: one failed run must not kill the rest of the sweep
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/ubuntu/carnot/skunk/venv/bin/python3}"
CHROMA_HOST="${CHROMA_HOST:-127.0.0.1}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
MODEL="${MODEL:-openai/gpt-5.6-luna}"
PROVIDER="${PROVIDER-}"
PHASES="${PHASES-}"
CELLS="${CELLS-}"
OFFICEQA_CHROMA_DIR="$(pwd)/benchmarks/officeqa/chromadb"
OFFICEQA_COLLECTION="officeqa-qwen-8b"
SYNTH_QA_PAIRS="$(pwd)/benchmarks/officeqa/synth_dev_qa_pairs.json"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set in the environment}"

LOG_DIR="logs/ws_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[sweep] logs -> $LOG_DIR"

# --- Synth subset: qids of the first 4 LOI blocks (6 questions each), in file order ---------
SYNTH24_QIDS="$("$PYTHON" - "$SYNTH_QA_PAIRS" <<'PY'
import json, sys
records = json.load(open(sys.argv[1]))
# Mirror the officeqa_synth loader's qid scheme: per-source-uid counter, file order.
qids, counts, lois = [], {}, []
for rec in records:
    loi = rec["line_of_inquiry"]["line_of_inquiry"]
    if loi not in lois:
        if len(lois) == 4:
            break
        lois.append(loi)
    uid = str(rec["qid"])
    j = counts.get(uid, 0); counts[uid] = j + 1
    qids.append(f"{uid}_q{j}")
print(",".join(qids))
PY
)"
n_synth=$(( $(tr -cd ',' <<< "$SYNTH24_QIDS" | wc -c) + 1 ))
echo "[sweep] synth subset: $n_synth qids (first 4 LOIs): $SYNTH24_QIDS"

# --- ChromaDB server: reuse a live one, else start one over the officeqa store -------------
chroma_up() {
  "$PYTHON" -c "
from skunk.chroma_client import make_chroma_client
make_chroma_client('${CHROMA_HOST}', ${CHROMA_PORT})
" 2>/dev/null
}

if chroma_up; then
  echo "[sweep] chroma server already up on ${CHROMA_HOST}:${CHROMA_PORT}"
else
  echo "[sweep] no chroma server on ${CHROMA_HOST}:${CHROMA_PORT} — starting one over ${OFFICEQA_CHROMA_DIR}"
  [ -d "$OFFICEQA_CHROMA_DIR" ] || { echo "[sweep] ERROR: no chroma store at $OFFICEQA_CHROMA_DIR" >&2; exit 1; }
  SKUNK_CHROMADB_DIR="$OFFICEQA_CHROMA_DIR" \
  SKUNK_CHROMA_SERVER_HOST="$CHROMA_HOST" \
  SKUNK_CHROMA_SERVER_PORT="$CHROMA_PORT" \
    nohup bash ../skunk/scripts/run_chroma_server.sh "$OFFICEQA_COLLECTION" \
    > "$LOG_DIR/chroma_server.log" 2>&1 &
  ready=0
  for _ in $(seq 1 180); do
    if chroma_up; then ready=1; break; fi
    sleep 1
  done
  [ "$ready" = 1 ] || { echo "[sweep] ERROR: chroma server not ready after 180s (see $LOG_DIR/chroma_server.log)" >&2; exit 1; }
  echo "[sweep] chroma server up (warm-up continues in the background; see $LOG_DIR/chroma_server.log)"
fi

# --- Working-set cleanup: delete every ws_* collection so each run starts clean ------------
wipe_working_sets() {
  "$PYTHON" -c "
from skunk.chroma_client import make_chroma_client
from skunk.search_state.working_set_registry import WS_PREFIX
client = make_chroma_client('${CHROMA_HOST}', ${CHROMA_PORT})
names = [c.name for c in client.list_collections() if c.name.startswith(WS_PREFIX)]
for name in names:
    client.delete_collection(name)
print(f'[sweep] wiped {len(names)} working-set collection(s)')
"
}

# phase | benchmark | workers | fetch_related_working_sets | qids ("" = whole dev split)
PHASE_CONFIGS=(
  "p1_parallel_noreuse|officeqa|33|false|"
  "p2_seq_reuse|officeqa|1|true|"
  "p3_loi24_noreuse|officeqa_synth|1|false|${SYNTH24_QIDS}"
  "p4_loi24_reuse|officeqa_synth|1|true|${SYNTH24_QIDS}"
)

# label | working_set_collection_off | id_tracking_off
CELL_CONFIGS=(
  "ws_coll_on_id_on|false|false"
  "ws_coll_on_id_off|false|true"
  "ws_coll_off_id_on|true|false"
  "ws_coll_off_id_off|true|true"
)

declare -a SUMMARY=()
for phase_entry in "${PHASE_CONFIGS[@]}"; do
  IFS='|' read -r phase benchmark workers fetch_related qids <<< "$phase_entry"
  phase_key="${phase%%_*}"   # p1 / p2 / p3 / p4
  if [ -n "$PHASES" ] && ! [[ ",$PHASES," == *",$phase_key,"* ]] && ! [[ ",$PHASES," == *",$phase,"* ]]; then
    echo "--- skipping phase $phase (not in PHASES=$PHASES)"
    continue
  fi

  for cell_entry in "${CELL_CONFIGS[@]}"; do
    IFS='|' read -r label coll_off id_off <<< "$cell_entry"
    if [ -n "$CELLS" ] && ! [[ ",$CELLS," == *",$label,"* ]]; then
      echo "--- skipping $phase/$label (not in CELLS=$CELLS)"
      continue
    fi

    extra_ovr=()
    [ -n "$PROVIDER" ] && extra_ovr+=( "inference.llm_provider_order=[${PROVIDER}]" )
    [ -n "$qids" ] && extra_ovr+=( "experiments.qids=[${qids}]" )
    run_name="${phase}_${label}"

    echo "==================================================================="
    echo "=== $benchmark | search_agent | $run_name"
    echo "===   collection_off=$coll_off id_tracking_off=$id_off fetch_related=$fetch_related"
    echo "===   model=$MODEL | workers=$workers${qids:+ | qids=$n_synth-question synth subset}"
    echo "==================================================================="
    if ! wipe_working_sets; then
      echo "[sweep] ERROR: working-set wipe failed for $run_name; skipping run" >&2
      SUMMARY+=( "FAILED(wipe)  $benchmark/$run_name" )
      continue
    fi

    run_log="$LOG_DIR/${benchmark}_${run_name}.log"
    if "$PYTHON" -m qatfd.runner \
      benchmarks="$benchmark" \
      systems=search_agent \
      inference.llm_model="$MODEL" \
      "${extra_ovr[@]}" \
      systems.include_search_corpus=true \
      systems.include_grep_corpus=true \
      systems.include_semantic_filter=false \
      systems.working_set_collection_off="$coll_off" \
      systems.id_tracking_off="$id_off" \
      systems.fetch_related_working_sets="$fetch_related" \
      benchmarks.chroma_server_host="$CHROMA_HOST" \
      benchmarks.chroma_server_port="$CHROMA_PORT" \
      experiments.workers="$workers" \
      experiments.run_name="$run_name" \
      > "$run_log" 2>&1; then
      SUMMARY+=( "ok            $benchmark/$run_name" )
    else
      SUMMARY+=( "FAILED        $benchmark/$run_name (see $run_log)" )
      echo "[sweep] run FAILED: $benchmark/$run_name — continuing (log: $run_log)" >&2
    fi
  done
done

echo
echo "=================== sweep summary ==================="
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo "Reports: results/<benchmark>/search_agent/<phase>_<cell>_<timestamp>/report.csv"
echo "Logs:    $LOG_DIR/"
