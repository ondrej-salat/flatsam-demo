#!/bin/bash
# SLURM batch script for LoMa feature matching experiments.
#
# Not meant to be called directly — use slurm/submit_loma.py which sets all
# LOMA_* environment variables and submits with the right resource flags.
#
# Manual one-off usage:
#   LOMA_ARGS="pot --model LoMaB --output-dir /mnt/.../LoMa/" \
#   LOMA_SUBSET_FILES=/path/to/mylist.txt \
#   FLATSAM2_REPO_ROOT=$(pwd) \
#   sbatch --time=8:00:00 --gpus=1 slurm/loma.sbatch

#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# ---------------------------------------------------------------------------
# Pick the subset file for this array task
# LOMA_SUBSET_FILES is a colon-separated list of paths, one per task.
# ---------------------------------------------------------------------------
IFS=':' read -ra _SUBSET_FILES <<< "${LOMA_SUBSET_FILES}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUBSET_FILE="${_SUBSET_FILES[$((TASK_ID - 1))]}"

if [ -z "${SUBSET_FILE}" ] || [ ! -f "${SUBSET_FILE}" ]; then
    echo "ERROR: subset file not found for task ${TASK_ID}: '${SUBSET_FILE}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Change to repo root, activate environment, and run the experiment
# ---------------------------------------------------------------------------
cd "${FLATSAM2_REPO_ROOT}"
source .envrc

read -ra ARGS <<< "${LOMA_ARGS}"
ARGS+=("--subset-file" "${SUBSET_FILE}")

echo "Task ${TASK_ID}: running loma_experiment.py ${ARGS[*]}"
export PYTHONUNBUFFERED=1
python -u experiments/loma_experiment.py "${ARGS[@]}"
