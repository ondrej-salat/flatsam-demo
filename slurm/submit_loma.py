"""Submit LoMa feature matching jobs to a SLURM cluster.

Collects the requested sequences, splits them into N equal parts, writes
temporary subset files, and submits a SLURM array job where each task
processes one part via experiments/loma_experiment.py.

Usage examples
--------------
# 20-way parallel over all POT-210 sequences, LoMaB:
    python slurm/submit_loma.py pot --model LoMaB --jobs 20

# PTI test split, LoMaG, dry-run:
    python slurm/submit_loma.py pti --model LoMaG --jobs 10 --dry-run

# Only run analysis (assumes matching already done):
    python slurm/submit_loma.py pot --model LoMaB --jobs 20 --skip-matching
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

_DEFAULT_OUTPUT_DIR = Path("/mnt/personal/serycjon/flatsam2/LoMa/")
_LOMA_MODELS = ["LoMaB", "LoMaB128", "LoMaL", "LoMaG", "LoMaR"]


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
        p = tmp_dir / f"loma_job_{stamp}_part{idx}.txt"
        p.write_text("\n".join(part) + "\n")
        paths_out.append(p)
    return paths_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit LoMa experiments as a SLURM array job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to process.")
    parser.add_argument(
        "--model",
        choices=_LOMA_MODELS + ["all"],
        required=True,
        help="LoMa model variant, or 'all' to submit a separate array job per variant.",
    )

    # Sequence selection (consumed here, translated into --subset-file per task).
    parser.add_argument("--sequences", nargs="+", metavar="SEQ", help="Explicit sequence names.")
    parser.add_argument("--subset", metavar="NAME", help="configs/subsets/<name>.txt subset.")
    parser.add_argument("--split", choices=["train", "test"], help="Train/test split.")

    # Forwarded to loma_experiment.py verbatim.
    parser.add_argument("--base-dir", type=Path, help="Dataset root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base output directory for LoMa results.",
    )
    parser.add_argument("--initfix-dir", type=Path, help="Initfix dir for PTI.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        metavar="PX",
        help="Pixel thresholds for analysis (forwarded).",
    )
    parser.add_argument("--force-matching", action="store_true")
    parser.add_argument("--skip-matching", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")

    # SLURM-specific options (NOT forwarded to loma_experiment.py).
    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument("--jobs", type=int, default=10, metavar="N",
                       help="Number of parallel array tasks.")
    slurm.add_argument("--time", default="02:00:00", help="Wall-clock time limit.")
    slurm.add_argument("--mem", default="32G", help="Memory per node.")
    slurm.add_argument("--gpus", default="1", help="GPUs per task.")
    slurm.add_argument("--partition", default="amdgpu", help="SLURM partition.")
    slurm.add_argument("--qos", default=None, help="SLURM QOS.")
    slurm.add_argument("--job-name", help="SLURM job name.")
    slurm.add_argument("--mail-to", default=os.environ.get("SLURM_MAIL_USER", ""), metavar="EMAIL",
                       help="Notification email.")
    slurm.add_argument("--dry-run", action="store_true",
                       help="Print commands without submitting.")

    return parser


def build_experiment_args(args: argparse.Namespace) -> str:
    """Build shell-quoted loma_experiment.py arguments (without --subset-file).

    Args:
        args: Parsed namespace from this script's parser.

    Returns:
        Shell-quoted argument string to store in LOMA_ARGS env var.
    """
    tokens: list[str] = [args.dataset, "--model", args.model]

    if args.base_dir is not None:
        tokens += ["--base-dir", str(args.base_dir)]
    tokens += ["--output-dir", str(args.output_dir)]
    if args.initfix_dir is not None:
        tokens += ["--initfix-dir", str(args.initfix_dir)]
    if args.split is not None:
        tokens += ["--split", args.split]
    if args.thresholds:
        tokens += ["--thresholds"] + [str(t) for t in args.thresholds]
    if args.force_matching:
        tokens.append("--force-matching")
    if args.skip_matching:
        tokens.append("--skip-matching")
    if args.skip_analysis:
        tokens.append("--skip-analysis")

    return shlex.join(tokens)


def _submit_one_model(args: argparse.Namespace, sequences: list[str], stamp: str) -> None:
    """Submit a SLURM array job for a single model variant.

    Args:
        args: Parsed namespace; args.model must be a concrete model name.
        sequences: Sequence names to distribute across array tasks.
        stamp: Timestamp string for unique subset-file names.
    """
    ds_info = DATASETS[args.dataset]

    n_jobs = min(args.jobs, len(sequences))
    if n_jobs != args.jobs:
        print(
            f"Note: --jobs {args.jobs} reduced to {n_jobs} "
            f"(only {len(sequences)} sequences available)."
        )

    parts = _split_sequences(sequences, n_jobs)
    tmp_dir = _REPO_ROOT / "slurm" / "tmp"
    subset_files = _write_subset_files(parts, tmp_dir, stamp)

    print(f"\nModel: {args.model}")
    print(f"Splitting into {n_jobs} job(s):")
    for i, (part, sf) in enumerate(zip(parts, subset_files), start=1):
        print(f"  [{i}/{n_jobs}] {len(part):3d} sequences → {sf.relative_to(_REPO_ROOT)}")

    sbatch_script = _REPO_ROOT / "slurm" / "loma.sbatch"
    subset_files_str = ":".join(str(p) for p in subset_files)

    job_name = args.job_name or f"loma_{args.model}_{args.dataset}"
    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--gres=gpu:{args.gpus}",
        f"--array=1-{n_jobs}",
    ]
    if args.partition:
        sbatch_cmd.append(f"--partition={args.partition}")
    if args.qos:
        sbatch_cmd.append(f"--qos={args.qos}")

    env = os.environ.copy()
    env["LOMA_ARGS"] = build_experiment_args(args)
    env["LOMA_SUBSET_FILES"] = subset_files_str
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)

    sbatch_cmd.append(str(sbatch_script))

    print(f"Submitting: {' '.join(sbatch_cmd)}")

    if args.dry_run:
        print("(dry-run: not submitting)")
        print("Environment variables that would be set:")
        for k in ("LOMA_ARGS", "LOMA_SUBSET_FILES", "FLATSAM2_REPO_ROOT"):
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

    # Collect sequences once; shared across all model variants.
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
    if args.model == "all":
        print(f"Submitting array jobs for all models: {', '.join(models)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_name in models:
        args.model = model_name
        _submit_one_model(args, sequences, stamp)


if __name__ == "__main__":
    main()
