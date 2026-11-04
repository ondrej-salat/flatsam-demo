#!/bin/bash
# SLURM batch script for flatsam2 evaluation, submitted as a dependency job
# after the run.py array job finishes successfully.
#
# This file is NOT meant to be edited directly.  Use slurm/submit.py to
# generate and submit jobs — it sets all FLATSAM2_* environment variables and
# submits this script with --dependency=afterok:<array_job_id>.

#SBATCH --output=slurm/logs/%x_%j_eval.out
#SBATCH --error=slurm/logs/%x_%j_eval.err
#SBATCH --mail-type=FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Change to repo root and activate the environment
# ---------------------------------------------------------------------------
cd "${FLATSAM2_REPO_ROOT}"
source .envrc

# FLATSAM2_EVAL_ARGS is a shell-quoted argument string built by submit.py.
read -ra EVAL_ARGS <<< "${FLATSAM2_EVAL_ARGS}"

echo "Running evaluate.py ${EVAL_ARGS[*]}"
EVAL_TMP=$(mktemp)
python evaluate.py "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_TMP}"
EVAL_EXIT="${PIPESTATUS[0]}"
EVAL_OUTPUT=$(cat "${EVAL_TMP}")
rm -f "${EVAL_TMP}"

if [ "${EVAL_EXIT}" -ne 0 ]; then
    echo "ERROR: evaluate.py exited with code ${EVAL_EXIT}" >&2
    exit "${EVAL_EXIT}"
fi

# ---------------------------------------------------------------------------
# Parse P@5 and P@15 from the markdown table row printed by evaluate.py:
#   | method | p@5 | p@15 |
#   | | 52.3 | 71.8 |
# ---------------------------------------------------------------------------
P5=$(echo "${EVAL_OUTPUT}"  | awk -F'|' '/^\|[[:space:]]*\|[[:space:]]*[0-9]/ {gsub(/ /,"",$2); gsub(/ /,"",$3); print $3}' | head -1)
P15=$(echo "${EVAL_OUTPUT}" | awk -F'|' '/^\|[[:space:]]*\|[[:space:]]*[0-9]/ {gsub(/ /,"",$2); gsub(/ /,"",$4); print $4}' | head -1)

# Fall back to "N/A" if parsing failed
P5="${P5:-N/A}"
P15="${P15:-N/A}"

# ---------------------------------------------------------------------------
# Build experiment description from env vars set by submit.py
# ---------------------------------------------------------------------------
JOB_NAME="${FLATSAM2_JOB_NAME:-flatsam2}"
DATASET="${FLATSAM2_DATASET:-unknown}"
CONFIG="${FLATSAM2_CONFIG:-default}"
RUN_NAME="${SLURM_JOB_NAME:-${JOB_NAME}}"
MAIL_TO="${FLATSAM2_MAIL_TO:-}"
ARRAY_JOB_ID="${FLATSAM2_ARRAY_JOB_ID:-unknown}"
EVAL_JOB_ID="${SLURM_JOB_ID:-unknown}"

SUBJECT="[flatsam2] ${RUN_NAME} — ${DATASET} — done"

BODY=$(cat <<EOF
Experiment finished successfully.

  Job name   : ${RUN_NAME}
  Dataset    : ${DATASET}
  Config     : ${CONFIG}
  Run job ID : ${ARRAY_JOB_ID}
  Eval job ID: ${EVAL_JOB_ID}

Results
-------
  P@5  : ${P5}%
  P@15 : ${P15}%

Full evaluate.py output
-----------------------
${EVAL_OUTPUT}
EOF
)

# mail is not available on compute nodes; pipe the body over SSH to the login
# node that submitted this job (SLURM_SUBMIT_HOST is set automatically).
# echo "${BODY}" | ssh "${SLURM_SUBMIT_HOST}" "mail -s '${SUBJECT}' '${MAIL_TO}'"
# echo "Success email sent to ${MAIL_TO} via ${SLURM_SUBMIT_HOST}"
