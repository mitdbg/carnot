#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=vllm-serve
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -G h100:2
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o logs/vllm_serve_%j.out
#SBATCH -e logs/vllm_serve_%j.err

# Serves the two-model qatfd experiment (agent + semantic filter) from one GPU node via
# scripts/run_vllm_servers.sh — one vLLM server per GPU, OpenAI API on ports 8100/8101.
# BOTH GPUs must land on ONE node (-N 1): the launcher runs both servers in this job.
#
# GPU choice: h100 (80 GB) — the more available pool (sinfo 2026-07-15 showed several
# preemptable nodes with 2-4 idle h100s while h200s ran mostly 8/8 used). On 80 GB the
# 27B (~54 GB bf16) is comfortable; the 35B-A3B (~70 GB bf16) is TIGHT, hence the len=
# caps below. If the 35B OOMs at load, switch its spec to an FP8 checkpoint (~35 GB;
# native FP8 on h100) rather than shrinking len further. h200:2 (140 GB, no len cap
# needed) when the queue allows; FP8-on-l40s:2 works on the PI partition in a pinch.
# mit_normal_gpu is the non-preemptable alternative (per-user GRES caps, no mid-run kills).
#
# PREEMPTION: on mit_preemptable this job can be killed anytime, which fails any qatfd run
# pointed at it mid-flight. Resubmit this job, re-tunnel, and resume the qatfd run with
# experiments.resume_dir=<run-dir> (finished questions are skipped).
#
# Connectivity: compute nodes are only reachable inside the cluster. From the machine that
# runs qatfd (holds the chroma stores + OPENROUTER_API_KEY), tunnel through the login node
# using the node name this job prints below:
#   ssh -N -L 8100:<node>:8100 -L 8101:<node>:8101 <user>@orcd-login001.mit.edu
# then point the run at the tunnel:
#   '++systems.vllm_base_urls={qwen/qwen3.6-27b: "http://127.0.0.1:8100/v1",
#                              qwen/qwen3.6-35b-a3b: "http://127.0.0.1:8101/v1"}'
#
# One-time env setup (login node):
#   module load miniforge && mamba create -n vllm python=3.12 -y
#   mamba activate vllm && pip install vllm
# (~20 GB in ~/.conda — watch the 200 GB home quota; `pip cache purge` after.)

# (-e/-o pipefail but NOT -u: conda's activate scripts reference unbound variables.)
set -eo pipefail

# sbatch runs a COPY of this script from a spool dir, so $0 can't locate the repo. Submit
# from the qatfd repo root (`sbatch engaging-scripts/run_vllm_servers.slurm`) and anchor on
# the submit dir instead.
cd "${SLURM_SUBMIT_DIR:?}"
if [[ ! -x scripts/run_vllm_servers.sh ]]; then
  echo "ERROR: scripts/run_vllm_servers.sh not found in $PWD — submit from the qatfd repo root" >&2
  exit 1
fi

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
nvidia-smi

# ── Environment setup ──
module purge
module load miniforge
mamba activate vllm
export PYTHONNOUSERSITE=True
# Weights cache on node-local scratch (~/orcd counts against the 200 GB home quota; /tmp is
# node-local NVMe). ~125 GB for the two models — re-downloaded per job/node, so budget the
# first ~15 min of the walltime for the pull. export HF_TOKEN before sbatch if repos are gated.
export HF_HOME=/tmp/$USER/hf_cache

# Qwen3.6's GDN linear-attention prefill kernel is JIT-compiled by FlashInfer, which needs
# nvcc — and compute nodes have no /usr/local/cuda (pip vLLM ships only the CUDA runtime).
# The Triton backend bundles its own compiler, so it needs nothing from the node. Extra
# args passed by the caller are appended after and still apply.
export VLLM_EXTRA_ARGS="--gdn-prefill-backend triton ${VLLM_EXTRA_ARGS:-}"

# ── Launch both servers (blocks until the job ends; servers die with the job) ──
# Inside the allocation Slurm renumbers the granted GPUs 0..N-1, so gpus=0/gpus=1 pin one
# server per GPU. ADVERTISE_HOST is the node's cluster-internal name — reachable from other
# cluster nodes directly, from outside via the SSH tunnel above.
# len= caps the context window so KV cache + CUDA graphs fit next to bf16 weights on 80 GB
# (vLLM otherwise reserves KV for the model's full advertised context, e.g. 262K). 64K/32K
# comfortably cover the search-agent and semantic-filter prompt sizes.
HOST=0.0.0.0 ADVERTISE_HOST=$(hostname) VLLM_BIN=$(command -v vllm) \
  ./scripts/run_vllm_servers.sh \
    'Qwen/Qwen3.6-27B:gpus=0:mem=0.92:len=65536:name=qwen/qwen3.6-27b' \
    'Qwen/Qwen3.6-35B-A3B:gpus=1:mem=0.95:len=32768:name=qwen/qwen3.6-35b-a3b'
