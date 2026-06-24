#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=qwen3-element-embed
#SBATCH -p pi_srmadden
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/embed_%j.out
#SBATCH -e logs/embed_%j.err

mkdir -p logs

# ── Print job info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
nvidia-smi

# ── Environment setup ──
# Start with a clean environment, then load miniforge (Engaging's Python module).
# See: https://orcd-docs.mit.edu/software/modules/
module purge
module load miniforge

# activate the conda environment
mamba activate officeqa

# Avoid conflicts with packages in ~/.local
export PYTHONNOUSERSITE=True

# Use scratch for the large HuggingFace model cache (~16GB for 8B model)
export HF_HOME=/home/mdrusso/orcd/scratch/.cache/huggingface

# ── Run embedding computation ──
# Launch one worker per GPU. srun spawns --ntasks=2 processes; each gets its
# own GPU via --gpus-per-task=1. Each task reads its own $SLURM_PROCID /
# $SLURM_NTASKS from the env *inside* Python -- DO NOT expand them here in
# the parent shell, or every task gets the parent's (empty) values.
srun python compute_element_embeddings.py \
    --input_dir /home/mdrusso/carnot/skunk/treasury_bulletins_parsed/jsons/ \
    --output_dir /home/mdrusso/carnot/skunk/treasury-bulletin-element-embeddings/qwen-strip-years/
    --strip_years

echo "End: $(date)"
