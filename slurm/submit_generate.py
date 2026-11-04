"""Submit a dataset pre-generation job to SLURM (CPU-only).

Usage examples
--------------
    # Generate 500 K samples, 32 workers, 4h time limit:
    python slurm/submit_generate.py --output-dir /scratch/nn_cache --n-samples 500000

    # Dry-run:
    python slurm/submit_generate.py --output-dir /scratch/nn_cache --dry-run
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
        description="Submit dataset pre-generation as a SLURM job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    gen = parser.add_argument_group("Generation options (forwarded to generate_dataset.py)")
    gen.add_argument("--output-dir", required=True,
                     help="Where to write the .npy cache files.")
    gen.add_argument("--n-samples", type=int, default=500_000)
    gen.add_argument("--arch", default="option_e", choices=["option_e", "option_c", "both"])
    gen.add_argument("--n-points", type=int, default=128)
    gen.add_argument("--seed", type=int, default=0)

    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument("--num-workers", type=int, default=32,
                       help="Worker processes — also sets --cpus-per-task.")
    slurm.add_argument("--time", default="06:00:00")
    slurm.add_argument("--mem", default="64G")
    slurm.add_argument("--partition", default="cpu")
    slurm.add_argument("--qos", default=None)
    slurm.add_argument("--job-name", default="flatsam2_gen")
    slurm.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gen_args = shlex.join([
        "--output-dir", args.output_dir,
        "--n-samples",  str(args.n_samples),
        "--num-workers", str(args.num_workers),
        "--arch",       args.arch,
        "--n-points",   str(args.n_points),
        "--seed",       str(args.seed),
    ])

    sbatch_script = _REPO_ROOT / "slurm" / "generate_dataset.sbatch"
    sbatch_cmd = [
        "sbatch",
        f"--job-name={args.job_name}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.num_workers}",
    ]
    if args.partition:
        sbatch_cmd.append(f"--partition={args.partition}")
    if args.qos:
        sbatch_cmd.append(f"--qos={args.qos}")
    sbatch_cmd.append(str(sbatch_script))

    env = os.environ.copy()
    env["FLATSAM2_GEN_ARGS"] = gen_args
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)

    print(f"Generation args: {gen_args}")
    print(f"Submitting: {' '.join(sbatch_cmd)}")

    if args.dry_run:
        print("(dry-run: not submitting)")
        return

    result = subprocess.run(sbatch_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting job:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
