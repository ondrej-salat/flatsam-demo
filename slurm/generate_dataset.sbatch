#!/bin/bash
# SLURM batch script for pre-generating the NN training dataset (CPU-only).
#
# Not meant to be run directly.  Use slurm/submit_generate.py.
#
# Manual one-off:
#   FLATSAM2_GEN_ARGS="--output-dir /scratch/nn_cache --n-samples 500000 --num-workers 32" \
#   FLATSAM2_REPO_ROOT=$(pwd) \
#   sbatch --time=04:00:00 --mem=64G --cpus-per-task=32 slurm/generate_dataset.sbatch

#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${FLATSAM2_REPO_ROOT}"
source .envrc

echo "Dataset generation job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "CPUs available: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Args: ${FLATSAM2_GEN_ARGS}"

export PYTHONUNBUFFERED=1
read -ra ARGS <<< "${FLATSAM2_GEN_ARGS}"
python -u -m flatsam2.nn.generate_dataset "${ARGS[@]}"

echo "Done — $(date)"
