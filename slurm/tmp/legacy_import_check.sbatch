#!/bin/bash
#SBATCH --job-name=legacy-import-check
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --mail-type=NONE
#SBATCH --output=/home/serycjon/git/flatsam2/slurm/logs/%x_%j.out
#SBATCH --error=/home/serycjon/git/flatsam2/slurm/logs/%x_%j.err

set -uo pipefail
cd /home/serycjon/git/flatsam2
source slurm/legacy_env.sh
export PYTHONUNBUFFERED=1

echo "python: $(which python)"
python experiments/run_legacy.py --help >/dev/null && echo "HELP OK"
python - <<'PY'
import sys
sys.path.insert(0, "/home/serycjon/git/flatsam2")
import flatsam2, cv2, numpy  # flatsam2 side
from flatsam2.datasets import get_sequences, make_gt_loader, make_image_paths
sys.path.insert(0, "/home/serycjon/git/flatsam/src")
import flatsam
from flatsam.config import load_config
import os
os.chdir("/home/serycjon/git/flatsam")
conf = load_config("configs/woftsam_samfallback_nl.py")
print("legacy config loaded:", conf.name, "| sam:", conf.sam.size, conf.sam.memory_stride)
print("track_function:", conf.track_function)
print("BOTH SIDES OK")
PY
echo "exit: $?"
