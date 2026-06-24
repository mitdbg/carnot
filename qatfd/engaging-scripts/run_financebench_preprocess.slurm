#!/bin/bash

# ── SLURM Job Flags ──
# CPU-ONLY + network job: no GPU, no model download (the gate/extract models run on OpenRouter).
# Three phases run in-process: RENDER (ProcessPool, CPU-bound — uses the allocated cores), LLM
# (ThreadPool, network-bound), ASSEMBLE. Everything streams in/out of S3, so it needs ~zero local
# disk. PAGE-LEVEL RESUMABLE (each phase skips work already persisted: rendered pages, done pages,
# assembled docs) -> on a preemptable partition just resubmit until done.
#SBATCH --job-name=fb-preprocess
#SBATCH -p mit_preemptable
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G
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
# The script also writes two intermediate S3 prefixes derived from --output_dir (override with
# --renders_dir / --pages_dir): *-renders (phase-1 PNG + text cache, ~25-35 GB; delete after) and
# *-pages (phase-2 per-page element artifacts). Only OUTPUT_DIR holds the final {doc}.json.
INPUT_DIR=s3://carnot-research/financebench/pdfs
OUTPUT_DIR=s3://carnot-research/financebench/financebench-elements

# --render-procs uses the allocated cores for phase 1; --concurrency bounds phase-2/3 network
# threads (the OpenRouter API, not local CPU, is the bottleneck there). To split across array tasks,
# add `#SBATCH --array=0-3` and pass --rank $SLURM_ARRAY_TASK_ID --world_size 4.
srun python preprocess_financebench_pdfs.py \
    --input_dir  "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --render-procs "${SLURM_CPUS_PER_TASK:-16}" \
    --concurrency 32

echo "End: $(date)"

# ── Next: EMBED (GPU) reads the element JSONs (--input_dir s3://.../financebench-elements), then
#    create_vector_db on a box with disk. See run_financebench_element_embeddings.slurm + README. ──
