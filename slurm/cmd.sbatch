#!/bin/bash
# Generic one-off SLURM job: activate the flatsam2 environment, then run an
# arbitrary command. Submit via slurm/submit_cmd.py, which sets FLATSAM2_CMD
# and FLATSAM2_REPO_ROOT. Unlike run.sbatch this does NO sequence splitting /
# array packing — it is for aggregate or utility jobs (reports, the
# sam_overseg_correction dump/report/compare, ad-hoc scripts).

#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --mail-type=FAIL

set -uo pipefail

cd "${FLATSAM2_REPO_ROOT}"
source .envrc          # module load PyTorch/CUDA + activate venv
mkdir -p slurm/logs

echo "Host    : $(hostname)"
echo "Workdir : $(pwd)"
echo "Command : ${FLATSAM2_CMD}"
echo "=========================================================================="
export PYTHONUNBUFFERED=1
eval "${FLATSAM2_CMD}"
rc=$?
echo "=========================================================================="
echo "Exit code: ${rc}"
exit ${rc}
