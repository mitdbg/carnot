#!/bin/bash

# ── SLURM Job Flags ──
# QAMPARI **DEV** corpus embeddings (the 1000-question dev split; for iteration, not the held-out test).
#SBATCH --job-name=qwen3-qampari-embed-dev
#SBATCH -p mit_preemptable
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH -c 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 6:00:00
#SBATCH --exclude=node2901
#SBATCH -o logs/qampari_embed_dev_%j.out
#SBATCH -e logs/qampari_embed_dev_%j.err

# Same Qwen3-0.6B embedder as test, just over the DEV-filtered corpus. The 4 ranks are independent —
# each streams its file-shard to/from S3 — so they SPREAD across free GPUs cluster-wide. RESUMABLE
# (finished partition .npz are skipped) -> on mit_preemptable just resubmit until done. Keep
# --ntasks=4 fixed across resubmits (the file-shard/resume keys on it).

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

# ── Paths (DEV): stream the filtered DEV corpus in / embeddings out via S3, so the cluster needs ~zero disk ──
COLLECTION_DIR=s3://carnot-research/qampari/qampari-corpus-dev
OUTPUT_DIR=s3://carnot-research/qampari/qampari-element-embeddings-dev

# ── Run embedding computation (one worker per GPU; each reads its $SLURM_PROCID/$SLURM_NTASKS) ──
srun python compute_qampari_embeddings.py \
    --collection_dir "$COLLECTION_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --n_partitions   200

echo "End: $(date)"

# ── Then, on a machine WITH disk, build the DEV Chroma collection from the S3 output ──
# 1. pull the embeddings + metadata:
#      aws s3 sync s3://carnot-research/qampari/qampari-element-embeddings-dev ./qampari-element-embeddings-dev
# 2. build the collection (note the -dev collection name):
#      python -m skunk.search_agent.prep.create_vector_db \
#          --embeddings-dir ./qampari-element-embeddings-dev \
#          --collection-name qwen-qampari-0.6b-dev \
#          --chroma-path ./.chromadb \
#          --benchmark qampari
# 3. run the benchmark on DEV locally (point questions + collection + metadata glob at the dev set;
#    all 1000 dev questions are "the set" for this file, so run with experiments.split=test):
#      python3 -m qatfd.runner benchmarks=qampari systems=search_agent experiments.split=test \
#          benchmarks.questions_path=qampari/dev_data.jsonl \
#          benchmarks.chromadb_collection=qwen-qampari-0.6b-dev \
#          benchmarks.chromadb_dir=./.chromadb \
#          benchmarks.qampari_metadata_glob='./qampari-element-embeddings-dev/metadata_rank*.json' \
#          systems.emb_model_id=qwen/qwen3-embedding-0.6b \
#          benchmarks.judge_model=<judge-model>
#    (fast iteration: add experiments.sample=N experiments.seed=0 to run a reproducible subset.)
