#!/bin/bash
# SLURM batch script for masked LoMa v2 tracking experiment.
#
# Not meant to be called directly — use slurm/submit_loma_v2.py which sets
# all LOMA_V2_* environment variables and submits with the right resource flags.
#
# Manual one-off usage:
#   LOMA_V2_ARGS="pot --config configs/loma/loma_b_default.py \
#       --precomputed-masks-dir /mnt/.../masks \
#       --output-dir /mnt/.../loma_v2/" \
#   LOMA_V2_SUBSET_FILES=/path/to/mylist.txt \
#   FLATSAM2_REPO_ROOT=$(pwd) \
#   sbatch --time=8:00:00 --gres=gpu:1 slurm/loma_v2.sbatch

#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# ---------------------------------------------------------------------------
# Pick the subset file for this array task.
# LOMA_V2_SUBSET_FILES is a colon-separated list of paths, one per task.
# ---------------------------------------------------------------------------
IFS=':' read -ra _SUBSET_FILES <<< "${LOMA_V2_SUBSET_FILES}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUBSET_FILE="${_SUBSET_FILES[$((TASK_ID - 1))]}"

if [ -z "${SUBSET_FILE}" ] || [ ! -f "${SUBSET_FILE}" ]; then
    echo "ERROR: subset file not found for task ${TASK_ID}: '${SUBSET_FILE}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Change to repo root, activate environment, and run the experiment.
# ---------------------------------------------------------------------------
cd "${FLATSAM2_REPO_ROOT}"
source .envrc

read -ra ARGS <<< "${LOMA_V2_ARGS}"
ARGS+=("--subset-file" "${SUBSET_FILE}")

echo "Task ${TASK_ID}: running loma_experiment_v2.py ${ARGS[*]}"
export PYTHONUNBUFFERED=1
python -u experiments/loma_experiment_v2.py "${ARGS[@]}"
