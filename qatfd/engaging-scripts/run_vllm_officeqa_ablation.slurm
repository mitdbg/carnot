#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=vllm-officeqa-abl
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -G h100:2
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o logs/vllm_officeqa_abl_%j.out
#SBATCH -e logs/vllm_officeqa_abl_%j.err

# Serves the two models the OfficeQA search-agent ablation needs, from one GPU node via
# scripts/run_vllm_servers.sh — one vLLM server per GPU, OpenAI API on ports 8100/8101:
#   GPU0 :8100  Qwen/Qwen3.6-35B-A3B   (agent model + semantic_filter judge)  name=qwen/qwen3.6-35b-a3b
#   GPU1 :8101  Qwen/Qwen3-Embedding-8B (query embedder for vector search + semantic_filter top_k)
# Both GPUs must land on ONE node (-N 1): the launcher runs both servers in this job.
#
# GPU choice: h100 (80 GB) — the 35B-A3B (~70 GB bf16) is TIGHT on 80 GB, hence len=32768.
# If it OOMs at load, switch its spec to an FP8 checkpoint (~35 GB) rather than shrinking len.
# The 8B embedder is tiny (only ever embeds short queries — the corpus is already in Chroma).
#
# PREEMPTION: on mit_preemptable this job can be killed anytime, failing any qatfd run pointed
# at it mid-flight. Resubmit, re-tunnel, and resume the qatfd run with
# experiments.resume_dir=<run-dir> (finished questions in results.jsonl are skipped).
#
# Connectivity: compute nodes are only reachable inside the cluster. From the machine that
# runs qatfd (holds the Chroma store), tunnel through the login node using the node name this
# job prints below:
#   ssh -N -L 8100:<node>:8100 -L 8101:<node>:8101 <kerb>@orcd-login.mit.edu
# then point the run at the tunnel (see run_officeqa_ablation_parallelism_sweep.sh).
#
# One-time env setup (login node): see run_vllm_servers.slurm.

# (-e/-o pipefail but NOT -u: conda's activate scripts reference unbound variables.)
set -eo pipefail

# sbatch runs a COPY of this script from a spool dir, so $0 can't locate the repo. Submit
# from the qatfd repo root (`sbatch engaging-scripts/run_vllm_officeqa_ablation.slurm`).
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
echo "Tunnel: ssh -N -L 8100:$(hostname):8100 -L 8101:$(hostname):8101 <kerb>@orcd-login.mit.edu"
nvidia-smi

# ── Environment setup (same module stack as run_vllm_servers.slurm) ──
module purge
module load miniforge
module load cuda/13.1.0
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
module load gcc/12.2.0
export LD_LIBRARY_PATH="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mamba activate vllm
export PYTHONNOUSERSITE=True
# Weights cache on node-local scratch. ~70 GB for the 35B (re-pulled per job/node) — budget
# the first ~10-15 min of walltime for the download. export HF_TOKEN before sbatch if gated.
export HF_HOME=/tmp/$USER/hf_cache

# ── Launch both servers (blocks until the job ends; servers die with the job) ──
# Inside the allocation Slurm renumbers the granted GPUs 0..N-1, so gpus=0/gpus=1 pin one
# server per GPU. ADVERTISE_HOST is the node's cluster-internal name — reachable via the tunnel.
# len= caps the KV context so it fits next to bf16 weights on 80 GB; seqs= because the Qwen3.6
# hybrid's GDN/mamba layers cap concurrent sequences by cache blocks (default 1024 fails
# CUDA-graph capture). The embedder defaults its served name to the model id, so it matches
# inference.emb_model_id / the vllm_base_urls key.
HOST=0.0.0.0 ADVERTISE_HOST=$(hostname) VLLM_BIN=$(command -v vllm) \
  ./scripts/run_vllm_servers.sh \
    'Qwen/Qwen3.6-35B-A3B:gpus=0:mem=0.95:len=32768:seqs=128:name=qwen/qwen3.6-35b-a3b' \
    'Qwen/Qwen3-Embedding-8B:gpus=1:mem=0.5:runner=pooling'
