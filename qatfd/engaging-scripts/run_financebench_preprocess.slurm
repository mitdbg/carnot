#!/bin/bash

# ── SLURM Job Flags ──
# CPU-ONLY + network job: no GPU, no model download (the gate/extract models run on OpenRouter).
# It streams PDFs in and element JSONs out of S3, so it needs ~zero local disk. RESUMABLE
# (a doc whose {doc_name}.json already exists in the output prefix is skipped) -> on a
# preemptable partition just resubmit until done.
#SBATCH --job-name=fb-preprocess
#SBATCH -p mit_preemptable
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/fb_preprocess_%j.out
#SBATCH -e logs/fb_preprocess_%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# ── Environment setup ──
module purge
module load miniforge
mamba activate officeqa
export PYTHONNOUSERSITE=True

# boto3 for S3 streaming (one-time; harmless if already installed). No `aws` CLI needed.
pip install --quiet boto3

# ── AWS credentials for s3://carnot-research (the `mit` profile from ~/.aws/credentials) ──
export AWS_PROFILE=mit
# export AWS_DEFAULT_REGION=us-east-1   # only if the profile doesn't already set a region

# ── OpenRouter key for the gate/extract calls ──
# sbatch exports your submit-time env by default (--export=ALL), so OPENROUTER_API_KEY set in your
# shell / ~/.bashrc propagates here. Fail fast if it didn't.
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in your environment before sbatch}"

# ── Paths: stream PDFs in / element JSONs out via S3, so the cluster needs ~zero local disk ──
# (Upload the corpus once from a machine with disk: `aws s3 sync skunk/financebench/pdfs \
#   s3://carnot-research/financebench/pdfs`.)
INPUT_DIR=s3://carnot-research/financebench/pdfs
OUTPUT_DIR=s3://carnot-research/financebench/financebench-elements

# Single asyncio process; the OpenRouter API (not local CPU) is the bottleneck, so high
# --concurrency is fine. To split across array tasks instead, add `#SBATCH --array=0-3`, set
# --ntasks per task to 1, and pass --rank $SLURM_ARRAY_TASK_ID --world_size 4.
srun python preprocess_financebench_pdfs.py \
    --input_dir  "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --concurrency 32

echo "End: $(date)"

# ── Next: EMBED (GPU) reads the element JSONs (pull from S3 or point --input_dir at the S3 prefix
#    once compute_financebench_element_embeddings.py is S3-aware), then create_vector_db on a box
#    with disk. See run_financebench_element_embeddings.slurm and qatfd/README.md. ──
