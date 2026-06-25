#!/bin/bash

# ── SLURM Job Flags ──
# QAMPARI corpus embeddings = the FULL chunked Wikipedia (~25.9M ~100-word passages, no filtering).
# ONE index serves BOTH the dev and test question splits (the split is purely at the question level,
# like TREC-BioGen), so there is a single embedding job + a single Chroma collection.
#SBATCH --job-name=qwen3-qampari-embed
#SBATCH -p mit_preemptable
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH -c 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 12:00:00
#SBATCH --exclude=node2901
#SBATCH -o logs/qampari_embed_%j.out
#SBATCH -e logs/qampari_embed_%j.err

# ~25.9M chunks with Qwen3-0.6B (fits on ANY GPU >=4GB, so --gpus-per-task=1 takes whatever's free).
# Same scale as TREC-BioGen's 26.8M abstracts. The 4 ranks are independent — each streams its
# file-shard to/from S3 — so they SPREAD across free GPUs cluster-wide. RESUMABLE (finished partition
# .npz are skipped) -> on mit_preemptable just resubmit until done. Keep --ntasks=4 fixed across
# resubmits (the file-shard/resume keys on it); to pin a GPU type, use e.g. --gpus-per-task=l40s:1.

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
# Qwen3-Embedding-0.6B HF cache (small model, ~1.2GB) — keep it off the home quota (use node-local /tmp).
export HF_HOME=/tmp/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# boto3 for S3 streaming (one-time; harmless if already installed). No `aws` CLI needed.
pip install --quiet boto3

# ── AWS credentials for s3://carnot-research ──
export AWS_PROFILE=mit
# export AWS_DEFAULT_REGION=us-east-1   # only if the `mit` profile doesn't already set a region

# ── Paths: stream the full chunked-Wikipedia corpus in / embeddings out via S3 (~zero local disk) ──
COLLECTION_DIR=s3://carnot-research/qampari/wikipedia_chunks/chunks_v5
OUTPUT_DIR=s3://carnot-research/qampari/qampari-element-embeddings

# ── Run embedding computation (one worker per GPU; each reads its $SLURM_PROCID/$SLURM_NTASKS) ──
srun python compute_qampari_embeddings.py \
    --collection_dir "$COLLECTION_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --n_partitions   2000

echo "End: $(date)"

# ── Then, on a machine WITH disk, build the (single) Chroma collection from the S3 output ──
# 1. pull the embeddings + metadata:
#      aws s3 sync s3://carnot-research/qampari/qampari-element-embeddings ./qampari-element-embeddings --profile mit
# 2. build the collection (rank-sharded metadata load keeps peak RAM to one shard):
#      python -m skunk.search_agent.prep.create_vector_db \
#          --embeddings-dir ./qampari-element-embeddings \
#          --collection-name qwen-qampari-0.6b \
#          --chroma-path ./.chromadb \
#          --benchmark qampari
# 3. run the benchmark locally (the SAME collection serves both splits; pick the question file):
#      # TEST (the 1000-question KARL set):
#      python3 -m qatfd.runner benchmarks=qampari systems=search_agent experiments.split=test \
#          benchmarks.chromadb_dir=./.chromadb \
#          systems.emb_model_id=qwen/qwen3-embedding-0.6b benchmarks.judge_model=<judge-model>
#      # DEV (iterate here; just point at dev_data.jsonl):
#      python3 -m qatfd.runner benchmarks=qampari systems=search_agent experiments.split=test \
#          benchmarks.questions_path=qampari/dev_data.jsonl \
#          benchmarks.chromadb_dir=./.chromadb \
#          systems.emb_model_id=qwen/qwen3-embedding-0.6b benchmarks.judge_model=<judge-model> \
#          experiments.sample=50 experiments.seed=0   # optional: fast reproducible subset
