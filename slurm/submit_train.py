"""Submit a NN homography quality estimation training job to SLURM.

Usage examples
--------------
# Train option_e with defaults:
    python slurm/submit_train.py --arch option_e --output-dir checkpoints/option_e

# Quick smoke run (few epochs) to verify the job starts:
    python slurm/submit_train.py --arch option_e --output-dir checkpoints/test \\
        --epochs 2 --steps-per-epoch 500 --val-steps 200 --time 00:30:00

# Dry-run (print the sbatch command without submitting):
    python slurm/submit_train.py --arch option_e --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit flatsam2 NN training as a SLURM job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- training args (forwarded verbatim to flatsam2.nn.train) ------------
    train = parser.add_argument_group("Training options (forwarded to nn.train)")
    train.add_argument("--arch", default="option_e", choices=["option_e", "option_c", "point_baseline"])
    train.add_argument("--output-dir", default="checkpoints/nn_quality")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--steps-per-epoch", type=int, default=10_000)
    train.add_argument("--val-steps", type=int, default=5_000)
    train.add_argument("--num-workers", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--sdf-downsample", type=int, default=1,
                       help="SDF downsample factor (1 = sparse path, no downsampling needed).")
    train.add_argument("--cache-dir", default=None,
                       help="Pre-generated dataset directory. Skips on-the-fly generation.")

    # ---- SLURM options (consumed here, not forwarded) -----------------------
    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument("--time", default="24:00:00", help="Wall-clock time limit.")
    slurm.add_argument("--mem", default="32G", help="Memory per node.")
    slurm.add_argument("--gpus", default="1", help="Number of GPUs.")
    slurm.add_argument("--cpus-per-task", type=int, default=None,
                       help="CPUs to request.  Defaults to num_workers + 1 "
                            "(one per DataLoader worker plus the main process).")
    slurm.add_argument("--partition", default="amdgpu", help="SLURM partition.")
    slurm.add_argument("--qos", default=None, help="SLURM QOS.")
    slurm.add_argument("--job-name", default="flatsam2_train", help="SLURM job name.")
    slurm.add_argument("--dry-run", action="store_true",
                       help="Print the sbatch command without submitting.")

    return parser


def build_train_args(args: argparse.Namespace) -> str:
    """Build a shell-quoted argument string for flatsam2.nn.train."""
    tokens = [
        "--arch", args.arch,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--val-steps", str(args.val_steps),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--patience", str(args.patience),
        "--sdf-downsample", str(args.sdf_downsample),
    ]
    if args.cache_dir is not None:
        tokens += ["--cache-dir", args.cache_dir]
    return shlex.join(tokens)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_args_str = build_train_args(args)

    sbatch_script = _REPO_ROOT / "slurm" / "train.sbatch"
    cpus = args.cpus_per_task if args.cpus_per_task is not None else args.num_workers + 1

    sbatch_cmd = [
        "sbatch",
        f"--job-name={args.job_name}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--gres=gpu:{args.gpus}",
        f"--cpus-per-task={cpus}",
    ]
    if args.partition:
        sbatch_cmd.append(f"--partition={args.partition}")
    if args.qos:
        sbatch_cmd.append(f"--qos={args.qos}")
    sbatch_cmd.append(str(sbatch_script))

    env = os.environ.copy()
    env["FLATSAM2_TRAIN_ARGS"] = train_args_str
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)

    print(f"Training args: {train_args_str}")
    print(f"Submitting: {' '.join(sbatch_cmd)}")

    if args.dry_run:
        print("(dry-run: not submitting)")
        print(f"  FLATSAM2_TRAIN_ARGS={train_args_str}")
        print(f"  FLATSAM2_REPO_ROOT={_REPO_ROOT}")
        return

    result = subprocess.run(sbatch_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting job:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
