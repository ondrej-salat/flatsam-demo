#!/bin/bash
# SLURM batch script for NN homography quality estimation training.
#
# Not meant to be run directly.  Use slurm/submit_train.py, which sets
# FLATSAM2_TRAIN_ARGS and calls sbatch with the right resource flags.
#
# Manual one-off usage:
#   FLATSAM2_TRAIN_ARGS="--arch option_e --output-dir checkpoints/option_e" \
#   FLATSAM2_REPO_ROOT=$(pwd) \
#   sbatch --time=24:00:00 --mem=32G --gres=gpu:1 slurm/train.sbatch

#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${FLATSAM2_REPO_ROOT}"
source .envrc

echo "Training job ${SLURM_JOB_ID} on $(hostname) — $(date)"
echo "Args: ${FLATSAM2_TRAIN_ARGS}"

export PYTHONUNBUFFERED=1
read -ra ARGS <<< "${FLATSAM2_TRAIN_ARGS}"
python -u -m flatsam2.nn.train "${ARGS[@]}"

echo "Done — $(date)"
