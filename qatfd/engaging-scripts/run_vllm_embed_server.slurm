#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=vllm-embed
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -G h100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/vllm_embed_%j.out
#SBATCH -e logs/vllm_embed_%j.err

# Serves ONE embedding model (Qwen3-Embedding-0.6B — the TREC-BioGen / QAMPARI query
# embedder) via scripts/run_vllm_servers.sh, OpenAI API on port 8100. A single h100 is
# overkill for a 0.6B embedder but it's the available preemptable pool; l40s:1 on the PI
# partition works just as well if the h100 queue is deep.
#
# PREEMPTION: on mit_preemptable this job can be killed anytime, which fails any qatfd run
# pointed at it mid-flight. Resubmit, re-tunnel, and resume the qatfd run with
# experiments.resume_dir=<run-dir> (finished questions are skipped).
#
# Connectivity: compute nodes are only reachable inside the cluster. From the machine that
# runs qatfd, tunnel through the login node using the node name this job prints below:
#   ssh -N -L 8100:<node>:8100 <user>@orcd-login.mit.edu
# (with a ControlMaster/ControlPersist ssh-config alias, `ssh <alias>` once for
# password+2FA, then the -N tunnel multiplexes over the socket with no re-auth)
# then point the run at the tunnel:
#   systems.emb_provider=vllm systems.emb_model_id=Qwen/Qwen3-Embedding-0.6B \
#   '++systems.vllm_base_urls={Qwen/Qwen3-Embedding-0.6B: "http://127.0.0.1:8100/v1"}'
#
# One-time env setup (login node): see run_vllm_servers.slurm.

# (-e/-o pipefail but NOT -u: conda's activate scripts reference unbound variables.)
set -eo pipefail

# sbatch runs a COPY of this script from a spool dir, so $0 can't locate the repo. Submit
# from the qatfd repo root (`sbatch engaging-scripts/run_vllm_embed_server.slurm`).
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

# ── Environment setup (same module stack as run_vllm_servers.slurm) ──
module purge
module load miniforge
module load cuda/13.1.0
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
module load gcc/12.2.0
export LD_LIBRARY_PATH="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mamba activate vllm
export PYTHONNOUSERSITE=True
# Weights cache on node-local scratch; the 0.6B pull is ~1.2 GB, so no walltime budget needed.
export HF_HOME=/tmp/$USER/hf_cache

# ── Launch the embedding server (blocks until the job ends; server dies with the job) ──
# mem=0.3 leaves the rest of the GPU free (0.6B weights + pooling need little); default
# served name = the model id, so emb_model_id and the manifest key agree. runner=pooling
# (not task=embed): the cluster's vllm is new enough that --task was removed.
HOST=0.0.0.0 ADVERTISE_HOST=$(hostname) VLLM_BIN=$(command -v vllm) \
  ./scripts/run_vllm_servers.sh \
    'Qwen/Qwen3-Embedding-0.6B:gpus=0:mem=0.3:runner=pooling'
