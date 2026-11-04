#!/bin/bash
# SLURM batch script for LoMa-tracker (RANSAC on pre-computed correspondences).
#
# Not meant to be called directly — use slurm/submit_loma_tracker.py which
# sets all LOMA_TRACKER_* environment variables and submits with the right
# resource flags.
#
# Manual one-off usage:
#   LOMA_TRACKER_ARGS="pot --model LoMaB --ransac RANSAC \
#       --loma-dir /mnt/.../LoMa/ --output-dir /mnt/.../loma_tracker/" \
#   LOMA_TRACKER_SUBSET_FILES=/path/to/mylist.txt \
#   FLATSAM2_REPO_ROOT=$(pwd) \
#   sbatch --time=2:00:00 slurm/loma_tracker.sbatch

#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Pick the subset file for this array task.
# LOMA_TRACKER_SUBSET_FILES is a colon-separated list of paths, one per task.
# ---------------------------------------------------------------------------
IFS=':' read -ra _SUBSET_FILES <<< "${LOMA_TRACKER_SUBSET_FILES}"
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

read -ra ARGS <<< "${LOMA_TRACKER_ARGS}"
ARGS+=("--subset-file" "${SUBSET_FILE}")

echo "Task ${TASK_ID}: running loma_tracker.py ${ARGS[*]}"
export PYTHONUNBUFFERED=1
python -u experiments/loma_tracker.py "${ARGS[@]}"
