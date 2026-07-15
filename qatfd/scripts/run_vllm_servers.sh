#!/usr/bin/env bash
# Launch one vLLM OpenAI-compatible server PER MODEL for a qatfd experiment, in a
# long-lived tmux on the GPU box. vLLM serves a single model per process, so a run that
# mixes models (agent + semantic filter + embedder) needs several servers on distinct
# ports; this script starts them all, waits for health, writes a JSON manifest of
# model -> base URL, and prints the paste-ready Hydra override for
# `systems.vllm_base_urls`. skunk's LLMClient routes any model listed in that map to its
# server; unlisted models (e.g. benchmarks.judge_model) stay on OpenRouter.
#
#   tmux new -s vllm
#   ./scripts/run_vllm_servers.sh 'Qwen/Qwen3-32B:gpus=0:mem=0.9' \
#                                 'Qwen/Qwen3-8B:gpus=1:mem=0.45' \
#                                 'Qwen/Qwen3-Embedding-0.6B:gpus=1:mem=0.2:task=embed'
#
# Each positional arg is a model spec: <model-id>[:key=val ...] with keys
#   gpus=0,1    CUDA_VISIBLE_DEVICES for this server (unset = vLLM's default visibility)
#   tp=2        --tensor-parallel-size
#   mem=0.45    --gpu-memory-utilization (fraction; lets several servers share one GPU)
#   len=32768   --max-model-len
#   task=embed  embedding server (--task embed; newer vLLM: swap for --runner pooling)
#   name=<id>   --served-model-name (DEFAULT: the model id itself, so the manifest keys,
#               systems.llm_model / semantic_filter_model / emb_model_id, and the server
#               all agree — see skunk config.py `vllm_base_urls`)
#   port=8105   explicit port (default: BASE_PORT + arg index)
# Model ids never contain ':', so the colon-split is unambiguous.
#
# Knobs (env): HOST (bind address, default 0.0.0.0), ADVERTISE_HOST (host written into the
# manifest URLs, default first `hostname -I` address), BASE_PORT (default 8100), MANIFEST
# (default scripts/vllm_manifest.json), LOG_DIR (default logs/vllm), HEALTH_TIMEOUT_S
# (default 1800 — big models load slowly), VLLM_BIN (default vllm), VLLM_API_KEY (adds
# --api-key and is what skunk's client sends; unset = server open, client sends "EMPTY"),
# VLLM_EXTRA_ARGS (appended verbatim to every server).
#
# --dry-run (first arg) prints each `vllm serve` command and the manifest without needing
# vllm or a GPU — sanity-checkable on the dev box.
set -euo pipefail

cd "$(dirname "$0")/.."   # qatfd repo root

# --- Load skunk/.env without clobbering vars already in the environment ---
# Same idiom as skunk/scripts/run_chroma_server.sh; brings in HF_TOKEN / VLLM_API_KEY.
ENV_FILE="${SKUNK_ENV_FILE:-$(pwd)/../skunk/.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    line="${line#export }"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$val" == \"*\" || "$val" == \'*\' ]]; then val="${val:1:${#val}-2}"; fi
    if ! eval "[ -n \"\${$key+x}\" ]"; then export "$key=$val"; fi
  done < "$ENV_FILE"
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--dry-run] <model-spec> [<model-spec> ...]" >&2
  echo "  model-spec: <model-id>[:key=val ...]  keys: gpus= tp= mem= len= task=embed name= port=" >&2
  echo "  Example: $0 'Qwen/Qwen3-32B:gpus=0:mem=0.9' 'Qwen/Qwen3-Embedding-0.6B:gpus=1:task=embed'" >&2
  exit 2
fi

HOST="${HOST:-0.0.0.0}"
ADVERTISE_HOST="${ADVERTISE_HOST:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
ADVERTISE_HOST="${ADVERTISE_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8100}"
MANIFEST="${MANIFEST:-scripts/vllm_manifest.json}"
LOG_DIR="${LOG_DIR:-logs/vllm}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"
VLLM_BIN="${VLLM_BIN:-vllm}"
# Poll health on the bind address, except the wildcard bind (poll loopback instead).
POLL_HOST="$HOST"; [[ "$HOST" == "0.0.0.0" ]] && POLL_HOST="127.0.0.1"

if [[ "$DRY_RUN" == 0 ]] && ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$VLLM_BIN' not on PATH. Install vLLM on this (GPU) box: pip3 install vllm" >&2
  echo "       (or use --dry-run to preview the commands without vllm)" >&2
  exit 1
fi

# One tag per invocation so per-model logs don't overwrite across runs; under Slurm the
# job id is the natural tag, elsewhere a timestamp. Override with RUN_TAG=... if desired.
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

PIDS=()
NAMES=()
PORTS=()
LOGS=()

cleanup() {
  trap '' INT TERM EXIT
  echo "Shutting down ${#PIDS[@]} vLLM server(s)..."
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 5
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
}
[[ "$DRY_RUN" == 0 ]] && trap cleanup INT TERM EXIT

mkdir -p "$LOG_DIR" "$(dirname "$MANIFEST")"

i=0
for spec in "$@"; do
  IFS=':' read -r -a fields <<< "$spec"
  model="${fields[0]}"
  gpus="" tp="" mem="" len="" task="" name="$model" port=""
  for kv in "${fields[@]:1}"; do
    case "$kv" in
      gpus=*) gpus="${kv#gpus=}" ;;
      tp=*)   tp="${kv#tp=}" ;;
      mem=*)  mem="${kv#mem=}" ;;
      len=*)  len="${kv#len=}" ;;
      task=*) task="${kv#task=}" ;;
      name=*) name="${kv#name=}" ;;
      port=*) port="${kv#port=}" ;;
      *) echo "ERROR: unknown key in model spec '$spec': '$kv'" >&2; exit 2 ;;
    esac
  done
  port="${port:-$((BASE_PORT + i))}"

  cmd=("$VLLM_BIN" serve "$model" --host "$HOST" --port "$port" --served-model-name "$name")
  [[ -n "$tp" ]] && cmd+=(--tensor-parallel-size "$tp")
  [[ -n "$mem" ]] && cmd+=(--gpu-memory-utilization "$mem")
  [[ -n "$len" ]] && cmd+=(--max-model-len "$len")
  [[ -n "$task" ]] && cmd+=(--task "$task")
  [[ -n "${VLLM_API_KEY:-}" ]] && cmd+=(--api-key "$VLLM_API_KEY")
  # shellcheck disable=SC2206 — word-splitting VLLM_EXTRA_ARGS is the point
  [[ -n "${VLLM_EXTRA_ARGS:-}" ]] && cmd+=(${VLLM_EXTRA_ARGS})

  log="$LOG_DIR/$(echo "$name" | tr '/:' '--').$RUN_TAG.log"
  LOGS+=("$log")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] ${gpus:+CUDA_VISIBLE_DEVICES=$gpus }${cmd[*]}  # log: $log"
  else
    echo "Starting vLLM for '$name' on :$port ${gpus:+(GPUs $gpus) }-> $log"
    if [[ -n "$gpus" ]]; then
      CUDA_VISIBLE_DEVICES="$gpus" "${cmd[@]}" >"$log" 2>&1 &
    else
      "${cmd[@]}" >"$log" 2>&1 &
    fi
    PIDS+=($!)
  fi
  NAMES+=("$name")
  PORTS+=("$port")
  i=$((i + 1))
done

# --- Health: poll each server's /v1/models until it answers (model weights loaded) ---
if [[ "$DRY_RUN" == 0 ]]; then
  auth=()
  [[ -n "${VLLM_API_KEY:-}" ]] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
  for idx in "${!PORTS[@]}"; do
    port="${PORTS[$idx]}"; name="${NAMES[$idx]}"; pid="${PIDS[$idx]}"
    echo "Waiting for '$name' on :$port (up to ${HEALTH_TIMEOUT_S}s — first run downloads + loads weights)..."
    deadline=$((SECONDS + HEALTH_TIMEOUT_S))
    until curl -sf "${auth[@]}" -o /dev/null "http://$POLL_HOST:$port/v1/models"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: vLLM server for '$name' exited during startup; last log lines:" >&2
        tail -n 30 "${LOGS[$idx]}" >&2 || true
        exit 1
      fi
      if (( SECONDS >= deadline )); then
        echo "ERROR: '$name' not healthy after ${HEALTH_TIMEOUT_S}s (see $LOG_DIR)" >&2
        exit 1
      fi
      sleep 5
    done
    echo "  '$name' ready."
  done
fi

# --- Manifest + the paste-ready Hydra override ---
manifest_json="{"
override="{"
for idx in "${!PORTS[@]}"; do
  sep=$([[ $idx -gt 0 ]] && echo ", " || echo "")
  url="http://$ADVERTISE_HOST:${PORTS[$idx]}/v1"
  manifest_json+="$sep\"${NAMES[$idx]}\": \"$url\""
  override+="$sep${NAMES[$idx]}: \"$url\""
done
manifest_json+="}"
override+="}"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] manifest ($MANIFEST): $manifest_json"
  echo "[dry-run] Hydra override: '++systems.vllm_base_urls=$override'"
  exit 0
fi

printf '%s\n' "$manifest_json" > "$MANIFEST"
echo "Wrote $MANIFEST"
echo
echo "All servers healthy. Point qatfd at them with:"
echo "  '++systems.vllm_base_urls=$override'"
echo "(++ because Hydra's struct mode rejects new dict keys under a plain override)"
echo "(add systems.llm_provider=vllm to route EVERY model locally, or leave it on"
echo " openrouter so only the mapped models — not e.g. the judge — go local)"
echo
# Keep the servers in the foreground so this tmux pane owns their lifetime.
wait
