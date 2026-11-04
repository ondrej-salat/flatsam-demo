"""Submit LoMa-tracker (RANSAC homography) jobs to a SLURM cluster.

Collects the requested sequences, splits them into N equal parts, writes
temporary subset files, and submits a SLURM array job where each task
processes one part via experiments/loma_tracker.py.

Usage examples
--------------
# 20-way parallel over all POT-210 sequences, LoMaB + RANSAC:
    python slurm/submit_loma_tracker.py pot --model LoMaB --ransac RANSAC --jobs 20

# PTI test split, LoMaG, MAGSAC, dry-run:
    python slurm/submit_loma_tracker.py pti --model LoMaG --ransac USAC_MAGSAC \\
        --jobs 10 --dry-run

# All models × all RANSAC methods, each as a separate array job:
    python slurm/submit_loma_tracker.py pot --model all --ransac all --jobs 20
"""

import argparse
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

_DEFAULT_LOMA_DIR = Path("/mnt/personal/serycjon/flatsam2/LoMa/")
_DEFAULT_OUTPUT_DIR = Path("/mnt/personal/serycjon/flatsam2/loma_tracker/")
_DEFAULT_THRESHOLD = 5.0
_LOMA_MODELS = ["LoMaB", "LoMaB128", "LoMaL", "LoMaG", "LoMaR"]

# Keep in sync with experiments/loma_tracker.py RANSAC_METHODS.
# Listed explicitly so the submitter has no cv2 dependency.
_RANSAC_METHODS = [
    "RANSAC",
    "USAC_DEFAULT",
    "USAC_MAGSAC",
    # "USAC_FAST",
    # "USAC_ACCURATE",
    # "USAC_PROSAC",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sequences(sequences: list[str], n_jobs: int) -> list[list[str]]:
    n_jobs = min(n_jobs, len(sequences))
    chunk = math.ceil(len(sequences) / n_jobs)
    return [sequences[i:i + chunk] for i in range(0, len(sequences), chunk)]


def _write_subset_files(parts: list[list[str]], tmp_dir: Path, stamp: str) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths_out = []
    for idx, part in enumerate(parts, start=1):
        p = tmp_dir / f"loma_tracker_job_{stamp}_part{idx}.txt"
        p.write_text("\n".join(part) + "\n")
        paths_out.append(p)
    return paths_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit LoMa-tracker (RANSAC homography) experiments as SLURM array jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to process.")
    parser.add_argument(
        "--model",
        choices=_LOMA_MODELS + ["all"],
        required=True,
        help="LoMa model variant, or 'all' to submit a separate array job per variant.",
    )
    parser.add_argument(
        "--ransac",
        choices=_RANSAC_METHODS + ["all"],
        default="RANSAC",
        help="RANSAC method, or 'all' to submit a separate array job per method.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        metavar="PX",
        help="RANSAC inlier pixel threshold (forwarded).",
    )

    # Sequence selection (consumed here, translated into --subset-file per task).
    parser.add_argument("--sequences", nargs="+", metavar="SEQ", help="Explicit sequence names.")
    parser.add_argument("--subset", metavar="NAME", help="configs/subsets/<name>.txt subset.")
    parser.add_argument("--split", choices=["train", "test"], help="Train/test split.")

    # Forwarded to loma_tracker.py verbatim.
    parser.add_argument("--base-dir", type=Path, help="Dataset root.")
    parser.add_argument(
        "--loma-dir",
        type=Path,
        default=_DEFAULT_LOMA_DIR,
        help="Base directory of pre-computed LoMa NPZs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base output directory for predicted corners.",
    )
    parser.add_argument("--initfix-dir", type=Path, help="Initfix dir for PTI.")
    parser.add_argument("--force", action="store_true", help="Recompute if output already exists.")

    # SLURM-specific options (NOT forwarded to loma_tracker.py).
    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument("--jobs", type=int, default=10, metavar="N",
                       help="Number of parallel array tasks.")
    slurm.add_argument("--time", default="02:00:00", help="Wall-clock time limit.")
    slurm.add_argument("--mem", default="16G", help="Memory per node.")
    slurm.add_argument("--gpus", default="0", help="GPUs per task (0 = CPU-only).")
    slurm.add_argument("--partition", default="cpu", help="SLURM partition.")
    slurm.add_argument("--qos", default=None, help="SLURM QOS.")
    slurm.add_argument("--job-name", help="SLURM job name.")
    slurm.add_argument("--mail-to", default=os.environ.get("SLURM_MAIL_USER", ""), metavar="EMAIL",
                       help="Notification email.")
    slurm.add_argument("--dry-run", action="store_true",
                       help="Print commands without submitting.")

    return parser


def build_experiment_args(args: argparse.Namespace, model: str, ransac: str) -> str:
    """Build shell-quoted loma_tracker.py arguments (without --subset-file).

    Args:
        args: Parsed namespace from this script's parser.
        model: Concrete LoMa model name.
        ransac: Concrete RANSAC method name.

    Returns:
        Shell-quoted argument string to store in LOMA_TRACKER_ARGS env var.
    """
    tokens: list[str] = [args.dataset, "--model", model, "--ransac", ransac]

    if abs(args.ransac_threshold - _DEFAULT_THRESHOLD) > 1e-6:
        tokens += ["--ransac-threshold", str(args.ransac_threshold)]
    if args.base_dir is not None:
        tokens += ["--base-dir", str(args.base_dir)]
    tokens += ["--loma-dir", str(args.loma_dir)]
    tokens += ["--output-dir", str(args.output_dir)]
    if args.initfix_dir is not None:
        tokens += ["--initfix-dir", str(args.initfix_dir)]
    if args.split is not None:
        tokens += ["--split", args.split]
    if args.force:
        tokens.append("--force")

    return shlex.join(tokens)


def _submit_one_combo(
    args: argparse.Namespace,
    model: str,
    ransac: str,
    sequences: list[str],
    stamp: str,
) -> None:
    """Submit a SLURM array job for one (model, ransac) combination.

    Args:
        args: Parsed namespace.
        model: Concrete LoMa model name.
        ransac: Concrete RANSAC method name.
        sequences: Sequence names to distribute across array tasks.
        stamp: Timestamp string for unique subset-file names.
    """
    n_jobs = min(args.jobs, len(sequences))
    if n_jobs != args.jobs:
        print(
            f"Note: --jobs {args.jobs} reduced to {n_jobs} "
            f"(only {len(sequences)} sequences available)."
        )

    parts = _split_sequences(sequences, n_jobs)
    tmp_dir = _REPO_ROOT / "slurm" / "tmp"
    subset_files = _write_subset_files(parts, tmp_dir, stamp)

    print(f"\nModel: {model}  RANSAC: {ransac}")
    print(f"Splitting into {n_jobs} job(s):")
    for i, (part, sf) in enumerate(zip(parts, subset_files), start=1):
        print(f"  [{i}/{n_jobs}] {len(part):3d} sequences → {sf.relative_to(_REPO_ROOT)}")

    sbatch_script = _REPO_ROOT / "slurm" / "loma_tracker.sbatch"
    subset_files_str = ":".join(str(p) for p in subset_files)

    job_name = args.job_name or f"loma_tracker_{model}_{ransac}_{args.dataset}"
    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--array=1-{n_jobs}",
    ]
    if args.gpus != "0":
        sbatch_cmd.append(f"--gres=gpu:{args.gpus}")
    if args.partition:
        sbatch_cmd.append(f"--partition={args.partition}")
    if args.qos:
        sbatch_cmd.append(f"--qos={args.qos}")

    env = os.environ.copy()
    env["LOMA_TRACKER_ARGS"] = build_experiment_args(args, model, ransac)
    env["LOMA_TRACKER_SUBSET_FILES"] = subset_files_str
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)

    sbatch_cmd.append(str(sbatch_script))

    print(f"Submitting: {' '.join(sbatch_cmd)}")

    if args.dry_run:
        print("(dry-run: not submitting)")
        print("Environment variables that would be set:")
        for k in ("LOMA_TRACKER_ARGS", "LOMA_TRACKER_SUBSET_FILES", "FLATSAM2_REPO_ROOT"):
            print(f"  {k}={env[k]}")
        return

    result = subprocess.run(sbatch_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting job:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ds_info = DATASETS[args.dataset]
    if ds_info.uses_split and args.split is None:
        args.split = "test"

    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)

    # Collect sequences once; shared across all combos.
    try:
        sequences = get_sequences(
            args.dataset,
            args.base_dir,
            sequences=args.sequences,
            subset=args.subset,
            split=args.split,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(sequences)} sequences in {args.dataset!r} ({ds_info.label})")

    models = _LOMA_MODELS if args.model == "all" else [args.model]
    ransacs = _RANSAC_METHODS if args.ransac == "all" else [args.ransac]

    combos = [(m, r) for m in models for r in ransacs]
    if len(combos) > 1:
        print(f"Submitting {len(combos)} array job(s): "
              f"{len(models)} model(s) × {len(ransacs)} RANSAC method(s)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model, ransac in combos:
        _submit_one_combo(args, model, ransac, sequences, stamp)


if __name__ == "__main__":
    main()
