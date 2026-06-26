#!/bin/bash

# ── SLURM Job Flags ──
# FreshStack corpus embeddings for ONE topic (pass the topic as the first arg; default langchain).
# Each topic is a SEPARATE corpus -> a separate collection, so run this once per topic:
#     sbatch run_freshstack_embeddings.slurm langchain   # test set (203 q / 49,514 docs)
#     sbatch run_freshstack_embeddings.slurm laravel     # dev  set (184 q / 52,351 docs)
#SBATCH --job-name=qwen3-freshstack-embed
#SBATCH -p mit_preemptable
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=a100:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 2:00:00
#SBATCH --requeue
#SBATCH -o logs/freshstack_embed_%j.out
#SBATCH -e logs/freshstack_embed_%j.err

# ~50K chunks with Qwen3-0.6B — tiny next to TREC-BioGen/QAMPARI's 25-27M, so this finishes in minutes.
# --ntasks=4 a100 because mit_preemptable caps GPUs-per-user (QOSMaxGRESPerUser=4); the 4 ranks are
# independent (each embeds its `records[rank::world_size]` stripe of the single corpus.jsonl) so SLURM
# spreads them across free a100s. We PIN a100 via gres to avoid the Blackwell sm_120 nodes (e.g.
# rtx_pro_6000) whose kernels the officeqa env's torch can't run (cf. financebench commit e56f451).
# RESUMABLE (finished partition .npz are skipped) -> on mit_preemptable just resubmit until done. Keep
# --ntasks=4 FIXED across resubmits (the record-stripe + the embeddings_{rank}_{p}.npz resume key
# depend on it); changing it re-shards and orphans partial output.

TOPIC="${1:-langchain}"
echo "Topic:  $TOPIC"

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

# ── Paths: stream the topic's corpus in / embeddings out via S3 (~zero local disk) ──
# (upload corpus.jsonl first, e.g. download_freshstack.py then `aws s3 cp` to this prefix)
CORPUS_FILE=s3://carnot-research/freshstack/$TOPIC/corpus.jsonl
OUTPUT_DIR=s3://carnot-research/freshstack/$TOPIC/embeddings

# ── Run embedding computation (one worker per GPU; each reads its $SLURM_PROCID/$SLURM_NTASKS) ──
srun python compute_freshstack_embeddings.py \
    --corpus_file "$CORPUS_FILE" \
    --output_dir  "$OUTPUT_DIR" \
    --n_partitions 50

echo "End: $(date)"

# ── Then, on a machine WITH disk, build the topic's Chroma collection from the S3 output ──
# 1. pull the embeddings + metadata:
#      aws s3 sync s3://carnot-research/freshstack/$TOPIC/embeddings ./freshstack/$TOPIC/embeddings --profile mit
# 2. build the collection (rank-sharded metadata load keeps peak RAM to one shard):
#      python -m skunk.search_agent.prep.create_vector_db \
#          --embeddings-dir ./freshstack/$TOPIC/embeddings \
#          --collection-name qwen-freshstack-$TOPIC-0.6b \
#          --chroma-path ./.chromadb \
#          --benchmark freshstack
# 3. run the benchmark locally (KARL retrieves FreshStack with Qwen3-0.6B at k=10 — set
#    systems.top_k=10 for the rag_llm baseline; the search_agent controls its own retrieval breadth):
#      # TEST (langchain — the 203-question KARL set):
#      python3 -m qatfd.runner benchmarks=freshstack systems=search_agent experiments.split=test \
#          benchmarks.chromadb_dir=./.chromadb \
#          systems.emb_model_id=qwen/qwen3-embedding-0.6b benchmarks.judge_model=<judge-model>
#      # DEV (laravel — switch the whole topic with one override):
#      python3 -m qatfd.runner benchmarks=freshstack benchmarks.topic=laravel systems=search_agent \
#          experiments.split=test benchmarks.chromadb_dir=./.chromadb \
#          systems.emb_model_id=qwen/qwen3-embedding-0.6b benchmarks.judge_model=<judge-model>
