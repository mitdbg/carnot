#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=qwen3-biogen-embed
#SBATCH -p mit_preemptable
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH -c 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 12:00:00
#SBATCH --exclude=node2901
#SBATCH -o logs/biogen_embed_%j.out
#SBATCH -e logs/biogen_embed_%j.err

# 26.8M abstracts with Qwen3-0.6B (fits on ANY GPU >=4GB, so --gpus-per-task=1 takes whatever's
# free). NO single-node constraint: the 4 ranks are independent — each streams its file-shard
# to/from S3 — so they SPREAD across free GPUs cluster-wide, far easier to schedule than
# 4-on-one-node. RESUMABLE (finished partition .npz are skipped) -> on mit_preemptable just
# resubmit until done. Keep --ntasks=4 fixed across resubmits (the file-shard/resume keys on it);
# to pin a GPU type instead, use e.g. --gpus-per-task=l40s:1.

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
nvidia-smi

# ── Environment setup ──
module purge
module load miniforge
mamba activate officeqa
export PYTHONNOUSERSITE=True
# Qwen3-Embedding-0.6B HF cache (small model, ~1.2GB) — keep it off the 200 GB home quota.
# (~/orcd is NOT a separate filesystem on this account — it counts against home — so use /tmp,
#  which is node-local scratch and ample for a 1.2 GB model.)
export HF_HOME=/tmp/$USER/hf_cache
# Reduce CUDA allocator fragmentation (the embedder's variable batch sizes leave reserved-but-
# unallocated memory; this lets the allocator grow segments instead of OOMing around the gaps).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# boto3 for S3 streaming (one-time; harmless if already installed). No `aws` CLI needed.
pip install --quiet boto3

# ── AWS credentials for s3://carnot-research ──
# Use the `mit` profile from ~/.aws/credentials (srun inherits this env). No secrets in this file.
export AWS_PROFILE=mit
# Set the bucket's region only if the `mit` profile in ~/.aws/config doesn't already specify one:
# export AWS_DEFAULT_REGION=us-east-1

# ── Paths: stream corpus in / embeddings out via S3, so the cluster needs ~zero local disk ──
COLLECTION_DIR=s3://carnot-research/biogen/jsonl_collection
OUTPUT_DIR=s3://carnot-research/biogen/biogen-element-embeddings

# ── Run embedding computation (one worker per GPU; each reads its $SLURM_PROCID/$SLURM_NTASKS) ──
srun python compute_biogen_embeddings.py \
    --collection_dir "$COLLECTION_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --n_partitions   2000

echo "End: $(date)"

# ── Then, on a machine WITH disk (your local 2.8 TiB box) build the Chroma index from S3 output ──
# 1. pull the embeddings + metadata (aws CLI or boto3):
#      aws s3 sync s3://carnot-research/biogen/biogen-element-embeddings ./biogen-element-embeddings
# 2. build the collection:
#      python -m skunk.search_agent.prep.create_vector_db \
#          --embeddings-dir ./biogen-element-embeddings \
#          --collection-name qwen-biogen-0.6b \
#          --chroma-path ./.chromadb \
#          --benchmark trec_biogen
# 3. run the benchmark locally (metadata glob points at the pulled dir):
#      python3 -m qatfd.runner benchmarks=trec-biogen systems=search_agent experiments.split=test \
#          benchmarks.chromadb_dir=./.chromadb \
#          benchmarks.biogen_metadata_glob='./biogen-element-embeddings/metadata_rank*.json' \
#          systems.emb_model_id=qwen/qwen3-embedding-0.6b
