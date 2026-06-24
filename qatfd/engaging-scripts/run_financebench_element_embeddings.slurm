#!/bin/bash

# ── SLURM Job Flags ──
#SBATCH --job-name=qwen3-financebench-embed
#SBATCH -p pi_srmadden
#SBATCH --nodes=1
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
export HF_HOME=/home/mdrusso/orcd/scratch/.cache/huggingface

# ── Run embedding computation ──
# One worker per GPU; each task reads its own $SLURM_PROCID / $SLURM_NTASKS inside Python.
# Input is the element JSONs from preprocess_financebench_pdfs.py (run that on a CPU+network box
# first; it needs OPENROUTER_API_KEY, not a GPU).
srun python compute_financebench_element_embeddings.py \
    --input_dir /home/mdrusso/carnot/skunk/financebench/financebench-elements \
    --output_dir /home/mdrusso/carnot/skunk/financebench/financebench-element-embeddings

echo "End: $(date)"
