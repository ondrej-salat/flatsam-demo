"""Submit an arbitrary command to SLURM with the flatsam2 environment loaded.

Generic one-off submitter: it activates ``.envrc`` (module load PyTorch/CUDA +
venv) on the node, then runs whatever command you pass after ``--``. Unlike
``slurm/submit.py`` it does no sequence splitting or array packing — use it for
aggregate / utility jobs (reports, the ``sam_overseg_correction``
dump/report/compare, ad-hoc scripts).

Examples
--------
    # the sam_overseg_correction mask comparison (CPU-only, no GPU needed):
    python slurm/submit_cmd.py --job-name overseg-compare -- \
        python experiments/sam_overseg_correction.py compare \
            --config configs/tracker/loma_samh_correct_decoupled_excessdepth.py

    # request a GPU and a longer wall-clock limit:
    python slurm/submit_cmd.py --gpus 1 --time 8:00:00 -- python some_script.py --flag

    # preview the sbatch invocation without submitting:
    python slurm/submit_cmd.py --dry-run -- python foo.py

Everything after ``--`` is the command, forwarded verbatim (quoting preserved).
Outputs land wherever the command writes them (e.g. the compare CSV/montage go
to the shared OUTPUT_DIRECTORY), so no copy-back step is needed.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-name", default="flatsam2-cmd", help="SLURM job name.")
    # Defaults sized for what this wrapper is actually used for: probes, offline
    # read-outs and test runs, which finish in seconds to minutes. The old
    # 04:00:00 / 32G / 8-core default asked for a slot big enough that the
    # backfill scheduler could not place it on a busy cluster — two such jobs sat
    # PENDING for ~1h45 on 2026-08-19 to run a 25-second pytest. Heavy callers
    # pass explicit larger values; nobody should pay for them by default.
    p.add_argument("--time", default="00:30:00", help="Wall-clock limit (HH:MM:SS).")
    p.add_argument("--mem", default="8G", help="Memory request.")
    p.add_argument("--cpus", type=int, default=2, help="CPU cores (--cpus-per-task).")
    p.add_argument("--gpus", type=int, default=0,
                   help="GPUs to request; 0 omits --gres entirely (CPU-only job).")
    p.add_argument("--partition", default=None,
                   help="SLURM partition (default: 'cpu' when --gpus 0, else 'amdgpu').")
    p.add_argument("--qos", default=None, help="SLURM QOS.")
    p.add_argument("--no-mail", action="store_true",
                   help="Suppress SLURM notification mail (overrides the sbatch script's "
                        "--mail-type). Use for throwaway/agent-submitted probe jobs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the sbatch command without submitting.")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="Command to run on the node (put it after `--`).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cmd = args.command
    if cmd and cmd[0] == "--":  # argparse leaves the leading `--` in REMAINDER
        cmd = cmd[1:]
    if not cmd:
        print("ERROR: no command given. Put it after `--`, e.g.\n"
              "  python slurm/submit_cmd.py -- python myscript.py --flag",
              file=sys.stderr)
        sys.exit(1)
    cmd_str = shlex.join(cmd)

    # GPU jobs go on the GPU partition; CPU-only jobs must use the cpu partition.
    partition = args.partition or (
        os.environ.get("SLURM_GPU_PARTITION", "amdgpu") if args.gpus > 0
        else os.environ.get("SLURM_CPU_PARTITION", "cpu")
    )

    sbatch_cmd = [
        "sbatch",
        f"--job-name={args.job_name}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus}",
    ]
    if args.gpus > 0:
        sbatch_cmd.append(f"--gres=gpu:{args.gpus}")
    if partition:
        sbatch_cmd.append(f"--partition={partition}")
    if args.qos:
        sbatch_cmd.append(f"--qos={args.qos}")
    if args.no_mail:
        sbatch_cmd.append("--mail-type=NONE")
    elif os.environ.get("SLURM_MAIL_USER"):
        sbatch_cmd.append(f"--mail-user={os.environ['SLURM_MAIL_USER']}")
    sbatch_cmd.append(str(_REPO_ROOT / "slurm" / "cmd.sbatch"))

    env = os.environ.copy()
    env["FLATSAM2_CMD"] = cmd_str
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)

    print(f"Command : {cmd_str}")
    print(f"Submit  : {' '.join(sbatch_cmd)}")
    if args.dry_run:
        print("(dry-run: not submitting)")
        print(f"FLATSAM2_CMD={cmd_str}")
        print(f"FLATSAM2_REPO_ROOT={env['FLATSAM2_REPO_ROOT']}")
        return

    result = subprocess.run(sbatch_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting job:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())

    # sbatch --mail-type=NONE does not win against cmd.sbatch's own #SBATCH
    # directive on this cluster, so --no-mail is enforced after submission.
    # These are throwaway probe jobs; they must never reach the user's inbox.
    if args.no_mail:
        job_id = next((tok for tok in result.stdout.split() if tok.isdigit()), None)
        if job_id is not None:
            subprocess.run(
                ["scontrol", "update", f"jobid={job_id}", "MailType=NONE"],
                capture_output=True, text=True,
            )


if __name__ == "__main__":
    main()
