#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=qwen3-element-embed
#SBATCH -p mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=h100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
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
# Launch one worker per GPU. srun spawns --ntasks=4 processes; each gets its
# own GPU via --gpus-per-task=1. Each task reads its own $SLURM_PROCID /
# $SLURM_NTASKS from the env *inside* Python -- DO NOT expand them here in
# the parent shell, or every task gets the parent's (empty) values.
srun python compute_element_embeddings.py \
    --output_dir /home/mdrusso/carnot/skunk/browsecomp-plus-embedding-gen/browsecomp-plus-element-embeddings \
    --n_partitions 200

echo "End: $(date)"
