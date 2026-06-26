#!/bin/bash

# ── SLURM Job Flags ──
# QAMPARI corpus embeddings = the FULL chunked Wikipedia (~25.9M ~100-word passages, no filtering).
# ONE index serves BOTH the dev and test question splits (the split is purely at the question level,
# like TREC-BioGen), so there is a single embedding job + a single Chroma collection.
#SBATCH --job-name=qwen3-qampari-embed
#SBATCH -p mit_preemptable
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=a100:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH -o logs/qampari_embed_%j.out
#SBATCH -e logs/qampari_embed_%j.err

# ~25.9M chunks with Qwen3-0.6B (fits on any GPU >=4GB; the model is tiny, so embedding is the work).
# Same scale as TREC-BioGen's 26.8M abstracts. --ntasks=4 because mit_preemptable caps GPUs-per-user
# (QOSMaxGRESPerUser) — a single gang allocation can't exceed the cap (4), and the cap bounds your
# CONCURRENT GPUs regardless of how the job is sliced, so 4 a100 is the max throughput here. The 4
# ranks are independent — each streams its file-shard to/from S3 — so SLURM spreads them across free
# a100s cluster-wide. We PIN a100 via gres (plenty free, and it avoids the Blackwell sm_120 nodes —
# e.g. rtx_pro_6000 — whose kernels the officeqa env's torch can't run; cf. financebench commit
# e56f451). RESUMABLE (finished partition .npz are skipped) -> on mit_preemptable just resubmit until
# done. Keep --ntasks=4 fixed across resubmits (the file-shard + the embeddings_{rank}_{p}.npz resume
# key on it); changing it re-shards and orphans partial output. If your QOS gpu cap is >4, you may
# raise --ntasks to it; to use a card with more idle capacity instead, switch gres, e.g. l40s:1.
#
# SURVIVING PREEMPTION: `--requeue` (above) puts the job back in the queue if preempted (when the
# partition's PreemptMode=REQUEUE; check `scontrol show partition mit_preemptable | grep -i preempt`).
# Since the job is resumable, a requeue/restart just skips finished partitions. For preempt modes that
# CANCEL (or to also survive hitting the 8h wall), submit a DEPENDENCY CHAIN — each link resumes, and
# once everything's embedded the remaining links no-op in seconds:
#   # simplest (serialize all same-name jobs for this user):
#   for i in $(seq 1 6); do sbatch --dependency=singleton run_qampari_embeddings.slurm; done
#   # or an explicit afterany chain (next starts after prev ends in ANY state):
#   JOB=$(sbatch --parsable run_qampari_embeddings.slurm)
#   for i in $(seq 1 5); do JOB=$(sbatch --parsable --dependency=afterany:$JOB run_qampari_embeddings.slurm); done

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
