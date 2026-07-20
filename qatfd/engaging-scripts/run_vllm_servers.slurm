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
#   ssh -N -L 8100:<node>:8100 -L 8101:<node>:8101 <user>@orcd-login.mit.edu
# (with a ControlMaster/ControlPersist ssh-config alias, `ssh <alias>` once for
# password+2FA, then the -N tunnel multiplexes over the socket with no re-auth)
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
# CUDA toolkit for FlashInfer's JIT-compiled kernels (GDN prefill, sampling, ...): puts
# nvcc on PATH + headers under CUDA_HOME. Pinned to the version matching the node driver
# and the env's cu13 torch/vLLM wheels; compiled kernels are cached in ~/.cache/flashinfer,
# so only the first run per kernel pays the compile.
module load cuda/13.1.0
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
# nvcc compiles kernels through a HOST g++, and the nodes' system gcc is too old for the
# C++17/20 that FlashInfer's kernels require ("-std=c++20 ... not supported with the
# configured host compiler"). gcc 12 fully supports C++20 and is inside nvcc 13.1's
# supported host-compiler range; nvcc picks it up from PATH.
module load gcc/12.2.0
# The JIT-built kernels link against gcc 12's libstdc++, but dlopen resolves
# libstdc++.so.6 from the node's OLD /lib64 copy ("GLIBCXX_3.4.26 not found"). The gcc
# module sets PATH but not the runtime library path, so put its libstdc++ first.
export LD_LIBRARY_PATH="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mamba activate vllm
export PYTHONNOUSERSITE=True
# Weights cache on node-local scratch (~/orcd counts against the 200 GB home quota; /tmp is
# node-local NVMe). ~125 GB for the two models — re-downloaded per job/node, so budget the
# first ~15 min of the walltime for the pull. export HF_TOKEN before sbatch if repos are gated.
export HF_HOME=/tmp/$USER/hf_cache

# ── Launch both servers (blocks until the job ends; servers die with the job) ──
# Inside the allocation Slurm renumbers the granted GPUs 0..N-1, so gpus=0/gpus=1 pin one
# server per GPU. ADVERTISE_HOST is the node's cluster-internal name — reachable from other
# cluster nodes directly, from outside via the SSH tunnel above.
# len= caps the context window so KV cache + CUDA graphs fit next to bf16 weights on 80 GB
# (vLLM otherwise reserves KV for the model's full advertised context, e.g. 262K). 64K/32K
# comfortably cover the search-agent and semantic-filter prompt sizes.
# seqs= on BOTH: the Qwen3.6 models are hybrids whose GDN/mamba layers need one
# fixed-size cache block per in-flight sequence, and the memory left after weights holds
# only ~345 (27B) / ~229 (35B) blocks — vLLM's default --max-num-seqs=1024 therefore fails
# CUDA-graph capture ("exceeds available Mamba cache blocks"). 128 concurrent decodes is
# far above what one qatfd sweep drives at once.
HOST=0.0.0.0 ADVERTISE_HOST=$(hostname) VLLM_BIN=$(command -v vllm) \
  ./scripts/run_vllm_servers.sh \
    'Qwen/Qwen3.6-27B:gpus=0:mem=0.92:len=65536:seqs=128:name=qwen/qwen3.6-27b' \
    'Qwen/Qwen3.6-35B-A3B:gpus=1:mem=0.95:len=32768:seqs=128:name=qwen/qwen3.6-35b-a3b'
