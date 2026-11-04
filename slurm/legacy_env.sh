# Environment for the legacy flatsam (WOFTSAM / SAM-H) trackers.
#
# Sourced instead of flatsam2's .envrc by slurm/run.sbatch when
# FLATSAM2_ENV_FILE points here (slurm/submit.py --legacy-config does that).
#
# The legacy venv is the one that has `sam2` and `kornia`, which flatsam2's venv
# does not; it also satisfies flatsam2's own dependencies, so experiments/
# run_legacy.py can import both packages at once. Its own .envrc activates the
# venv by a relative path, so it cannot be sourced from flatsam2's working
# directory — the module loads are replicated here instead (same PyTorch/CUDA
# modules as flatsam2's .envrc).

FLATSAM1_ROOT="${FLATSAM1_ROOT:-${HOME}/git/flatsam}"

module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load torchvision/0.22.1-foss-2024a-CUDA-12.6.0
module load Triton/3.5.0-foss-2024a-CUDA-12.6.0

source "${FLATSAM1_ROOT}/venv/bin/activate"
export PYTHONPATH="${FLATSAM1_ROOT}/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export FLATSAM1_ROOT
