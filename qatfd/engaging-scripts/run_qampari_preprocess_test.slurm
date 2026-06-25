#!/bin/bash

# ── SLURM Job Flags ──
# QAMPARI **TEST** corpus filter (the 1000-question KARL comparison set; test_data.jsonl).
# CPU-ONLY job: filter QAMPARI's chunked Wikipedia (~21M chunks) down to the ~256,680 chunks that
# mention a gold answer entity. Fans the chunk files across a process pool (one per allocated core).
# Streams in/out of S3, so it needs ~zero local disk. RESUMABLE (an input file whose output already
# exists is skipped) -> on a preemptable partition just resubmit until done.
#SBATCH --job-name=qampari-filter-test
#SBATCH -p mit_preemptable
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o logs/qampari_filter_test_%j.out
#SBATCH -e logs/qampari_filter_test_%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# ── Environment setup ──
module purge
module load miniforge
mamba activate officeqa
export PYTHONNOUSERSITE=True
# Use the conda env's libstdc++ (GLIBCXX_3.4.29, newer than the system /lib64 one).
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# flashtext = fast O(text) whole-word multi-keyword matcher (the script falls back to `re` if it's
# missing, but flashtext is much faster over 21M chunks). boto3 for S3 streaming.
pip install --quiet flashtext boto3

# ── AWS credentials for s3://carnot-research (the `mit` profile from ~/.aws/credentials) ──
export AWS_PROFILE=mit
# export AWS_DEFAULT_REGION=us-east-1   # only if the profile doesn't already set a region

# ── Paths (TEST) ──
# CHUNKS_DIR: QAMPARI's released chunked Wikipedia, the `wikipedia_chunks/chunks_v5` dir of
#   wikipedia_chunks_*.jsonl (download chunked_wikipedia.tar.gz, untar, upload once to S3).
# QUESTIONS:  the QAMPARI test_data.jsonl (gold entities source).
# OUTPUT_DIR: the filtered TEST corpus (qampari_chunks_*.jsonl), consumed by the embedding job.
CHUNKS_DIR=s3://carnot-research/qampari/wikipedia_chunks/chunks_v5
QUESTIONS=s3://carnot-research/qampari/test_data.jsonl
OUTPUT_DIR=s3://carnot-research/qampari/qampari-corpus

# To split across array tasks instead of one big node, add `#SBATCH --array=0-3` and pass
# --rank $SLURM_ARRAY_TASK_ID --world_size 4 (the script also reads SLURM_PROCID/SLURM_NTASKS).
srun python preprocess_qampari_corpus.py \
    --chunks_dir "$CHUNKS_DIR" \
    --questions  "$QUESTIONS" \
    --output_dir "$OUTPUT_DIR" \
    --processes  "${SLURM_CPUS_PER_TASK:-32}"

echo "End: $(date)"

# ── Next: EMBED (GPU) reads the filtered TEST corpus (run_qampari_embeddings_test.slurm), then
#    create_vector_db on a box with disk (--benchmark qampari). See configs/benchmarks/qampari.yaml. ──
