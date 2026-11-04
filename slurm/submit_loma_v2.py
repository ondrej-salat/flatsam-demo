"""Submit masked LoMa v2 tracking jobs to a SLURM cluster.

Collects the requested sequences, splits them into N equal parts, writes
temporary subset files, and submits a SLURM array job where each task
processes one part via experiments/loma_experiment_v2.py.

Usage examples
--------------
# 20-way parallel over all POT-210 sequences:
    python slurm/submit_loma_v2.py pot \\
        --config configs/loma/loma_b_default.py \\
        --precomputed-masks-dir /mnt/.../results/fast_kitti/POT-210/masks \\
        --jobs 20

# PTI test split, dry-run:
    python slurm/submit_loma_v2.py pti \\
        --config configs/loma/loma_b_default.py \\
        --precomputed-masks-dir /mnt/.../masks \\
        --jobs 10 --dry-run
"""

import argparse
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

_DEFAULT_OUTPUT_DIR = Path("/mnt/personal/serycjon/flatsam2/loma_v2/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sequences(sequences: list[str], n_jobs: int) -> list[list[str]]:
    n_jobs = min(n_jobs, len(sequences))
    chunk = math.ceil(len(sequences) / n_jobs)
    return [sequences[i:i + chunk] for i in range(0, len(sequences), chunk)]


def _write_subset_files(
    parts: list[list[str]],
    tmp_dir: Path,
    stamp: str,
    dataset: str,
) -> list[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths_out = []
    for idx, part in enumerate(parts, start=1):
        p = tmp_dir / f"loma_v2_job_{dataset}_{stamp}_part{idx}.txt"
        p.write_text("\n".join(part) + "\n")
        paths_out.append(p)
    return paths_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit masked LoMa v2 experiments as a SLURM array job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    dataset_choices = sorted(set(CLI_CHOICES + ["both"]))
    parser.add_argument(
        "dataset",
        choices=dataset_choices,
        help="Dataset to process (`both` submits POT-210 and PTi).",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a LoMaTrackerConfig Python file (configs/loma/*.py).",
    )
    parser.add_argument(
        "--precomputed-masks-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory with pre-computed SAM masks (<DIR>/<seq>/{frame+1:05d}.png).",
    )
    parser.add_argument(
        "--precomputed-masks",
        action="store_true",
        help=(
            "Use built-in default precomputed mask paths "
            f"(POT: {paths.SAM3_PRECOMPUTED_MASKS_POT}, "
            f"PTi: {paths.SAM3_PRECOMPUTED_MASKS_PTI}). "
            "--precomputed-masks-dir overrides this per submission."
        ),
    )

    # Sequence selection (consumed here, translated into --subset-file per task).
    parser.add_argument("--sequences", nargs="+", metavar="SEQ", help="Explicit sequence names.")
    parser.add_argument("--subset", metavar="NAME", help="configs/subsets/<name>.txt subset.")
    parser.add_argument("--split", choices=["train", "test"], help="Train/test split.")

    # Forwarded to loma_experiment_v2.py verbatim.
    parser.add_argument("--base-dir", type=Path, help="Dataset root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base output directory for v2 results.",
    )
    parser.add_argument("--initfix-dir", type=Path, help="Initfix dir for PTI.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output already exists (forwarded).",
    )

    # SLURM-specific options (NOT forwarded to loma_experiment_v2.py).
    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument("--jobs", type=int, default=10, metavar="N",
                       help="Number of parallel array tasks.")
    slurm.add_argument("--time", default="02:30:00", help="Wall-clock time limit.")
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
    """Build shell-quoted loma_experiment_v2.py arguments (without --subset-file).

    Args:
        args: Parsed namespace from this script's parser.

    Returns:
        Shell-quoted argument string to store in LOMA_V2_ARGS env var.
    """
    tokens: list[str] = [args.dataset, "--config", str(args.config)]
    tokens += ["--output-dir", str(args.output_dir)]

    if args.precomputed_masks_dir is not None:
        tokens += ["--precomputed-masks-dir", str(args.precomputed_masks_dir)]
    if args.base_dir is not None:
        tokens += ["--base-dir", str(args.base_dir)]
    if args.initfix_dir is not None:
        tokens += ["--initfix-dir", str(args.initfix_dir)]
    if args.split is not None:
        tokens += ["--split", args.split]
    if args.force:
        tokens.append("--force")

    return shlex.join(tokens)


def _default_masks_dir_for_dataset(dataset: str) -> Path:
    ds_info = DATASETS[dataset]
    if ds_info.gt_loader == "pot":
        return paths.SAM3_PRECOMPUTED_MASKS_POT
    if ds_info.gt_loader == "pti":
        return paths.SAM3_PRECOMPUTED_MASKS_PTI
    raise ValueError(
        f"--precomputed-masks defaults unavailable for dataset {dataset!r} ({ds_info.label}). "
        "Provide --precomputed-masks-dir explicitly."
    )


def _submit(args: argparse.Namespace, sequences: list[str], stamp: str) -> None:
    """Submit a SLURM array job for the given sequences.

    Args:
        args: Parsed namespace.
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
    subset_files = _write_subset_files(parts, tmp_dir, stamp, args.dataset)

    config_stem = Path(args.config).stem
    print(f"\nConfig: {config_stem}")
    print(f"Splitting into {n_jobs} job(s):")
    for i, (part, sf) in enumerate(zip(parts, subset_files), start=1):
        print(f"  [{i}/{n_jobs}] {len(part):3d} sequences → {sf.relative_to(_REPO_ROOT)}")

    sbatch_script = _REPO_ROOT / "slurm" / "loma_v2.sbatch"
    subset_files_str = ":".join(str(p) for p in subset_files)

    job_name = args.job_name or f"loma_v2_{config_stem}_{args.dataset}"
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
    env["LOMA_V2_ARGS"] = build_experiment_args(args)
    env["LOMA_V2_SUBSET_FILES"] = subset_files_str
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)

    sbatch_cmd.append(str(sbatch_script))

    print(f"Submitting: {' '.join(sbatch_cmd)}")

    if args.dry_run:
        print("(dry-run: not submitting)")
        print("Environment variables that would be set:")
        for k in ("LOMA_V2_ARGS", "LOMA_V2_SUBSET_FILES", "FLATSAM2_REPO_ROOT"):
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_to_submit = ["pot", "pti"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets_to_submit:
        submit_args = argparse.Namespace(**vars(args))
        submit_args.dataset = dataset

        ds_info = DATASETS[submit_args.dataset]
        if ds_info.uses_split:
            if submit_args.split is None:
                submit_args.split = "test"
        else:
            submit_args.split = None

        if submit_args.base_dir is None:
            submit_args.base_dir = getattr(paths, ds_info.base_dir_attr)

        if submit_args.precomputed_masks_dir is None and submit_args.precomputed_masks:
            try:
                submit_args.precomputed_masks_dir = _default_masks_dir_for_dataset(
                    submit_args.dataset
                )
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)

        try:
            sequences = get_sequences(
                submit_args.dataset,
                submit_args.base_dir,
                sequences=submit_args.sequences,
                subset=submit_args.subset,
                split=submit_args.split,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print(
            f"Discovered {len(sequences)} sequences in "
            f"{submit_args.dataset!r} ({ds_info.label})"
        )
        _submit(submit_args, sequences, stamp)


if __name__ == "__main__":
    main()
