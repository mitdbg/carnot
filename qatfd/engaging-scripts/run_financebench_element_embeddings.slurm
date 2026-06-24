#!/bin/bash

# ── SLURM Job Flags ──
# Embeds the FinanceBench element JSONs with Qwen3-Embedding-8B. RESUMABLE (finished partition
# .npz are skipped) and STREAMS element JSONs in / embeddings+metadata out of S3, so it keeps
# ~zero local disk beyond the model cache -> on a preemptable partition just resubmit until done.
# Keep --ntasks fixed across resubmits (the file-shard/resume keys on $SLURM_PROCID/$SLURM_NTASKS).
#SBATCH --job-name=qwen3-financebench-embed
#SBATCH -p mit_preemptable
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/fb_embed_%j.out
#SBATCH -e logs/fb_embed_%j.err

mkdir -p logs

# ── Print job info ──
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
# Qwen3-Embedding-8B HF cache (~16 GB) -> node-local /tmp scratch, NOT the 200 GB home quota.
export HF_HOME=/tmp/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# boto3 for S3 streaming (one-time; harmless if already installed). No `aws` CLI needed.
pip install --quiet boto3

# ── AWS credentials for s3://carnot-research (the `mit` profile from ~/.aws/credentials) ──
export AWS_PROFILE=mit
# export AWS_DEFAULT_REGION=us-east-1   # only if the profile doesn't already set a region

# ── Paths: stream element JSONs in / embeddings out via S3 (zero local disk beyond the model) ──
INPUT_DIR=s3://carnot-research/financebench/financebench-elements
OUTPUT_DIR=s3://carnot-research/financebench/financebench-element-embeddings

# ── Run embedding computation (one worker per GPU; each reads its $SLURM_PROCID/$SLURM_NTASKS) ──
srun python compute_financebench_element_embeddings.py \
    --input_dir  "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "End: $(date)"

# ── Then, on a machine WITH disk, build the Chroma index from the S3 output ──
# 1. pull the embeddings + metadata:
#      aws s3 sync s3://carnot-research/financebench/financebench-element-embeddings ./fb-embeddings
# 2. build the collection:
#      python -m skunk.search_agent.prep.create_vector_db \
#          --embeddings-dir ./fb-embeddings --collection-name qwen-financebench \
#          --chroma-path ./.chromadb --benchmark finance_bench
