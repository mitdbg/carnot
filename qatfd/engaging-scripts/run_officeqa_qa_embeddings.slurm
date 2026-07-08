#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=qa-embed-officeqa
#SBATCH -p pi_srmadden
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/qa_embed_officeqa_%j.out
#SBATCH -e logs/qa_embed_officeqa_%j.err

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
export HF_HOME=/home/mdrusso/orcd/scratch/.cache/huggingface

# ── Run embedding computation ──
# OfficeQA Pro has ~1,200 questions so a single GPU is sufficient.
# The script reads $SLURM_PROCID / $SLURM_NTASKS from inside Python;
# do NOT expand them here in the parent shell.
srun python /home/mdrusso/carnot/skunk/src/skunk/datagen/compute_qa_embeddings.py \
    --benchmark officeqa \
    --output_dir /home/mdrusso/carnot/benchmark-question-answer-embeddings/officeqa-qa-embeddings/ \
    --officeqa_path /home/mdrusso/carnot/qatfd/benchmarks/officeqa/officeqa_pro.csv

echo "End: $(date)"

# ── After the job completes ──
#   1. Back up the raw npz shards:
#      aws s3 sync /home/mdrusso/carnot/benchmark-question-answer-embeddings/officeqa-qa-embeddings \
#          s3://carnot-research/qa-embeddings/benchmark-question-answer-embeddings/officeqa-qa-embeddings --profile mit
#   2. Load them into the shared QA-embeddings chroma store (both benchmarks'
#      collections live in one store, backed up at s3://carnot-research/qa-embeddings/chromadb/):
#      python -m skunk.datagen.create_qa_vector_db  (see its --help for paths)
