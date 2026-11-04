"""Submit flatsam2 tracking jobs to a SLURM cluster.

Collects the requested sequences (applying the same --sequences / --subset /
--split logic as run.py), splits them into N equal parts, writes temporary
subset files, and submits a SLURM array job where each task processes one part.

Usage examples
--------------
# 4-way parallel over all POT-210 sequences:
    python slurm/submit.py pot --jobs 4

# Both POT-210 and PTi in one command:
    python slurm/submit.py both --config configs/tracker/default.py --jobs 8

# Custom config, 8-way split, longer time limit:
    python slurm/submit.py pot --config configs/tracker/default.py --jobs 8 --time 8:00:00

# PlanarTrack test split:
    python slurm/submit.py planartrack --split test --jobs 4

# Dry-run (print the sbatch command without submitting):
    python slurm/submit.py pot --jobs 4 --dry-run

# Legacy tracker (WOFTSAM / SAM-H from ~/git/flatsam): same splitting and
# packing, but the tasks run experiments/run_legacy.py under the legacy venv:
    python slurm/submit.py tmt --legacy-config configs/woftsam_samfallback_nl.py --jobs 8
"""

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure the repo root is on the path so flatsam2 is importable
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sequences(sequences: list[str], n_jobs: int) -> list[list[str]]:
    """Divide *sequences* into *n_jobs* roughly equal parts (no empty parts).

    Args:
        sequences: Full ordered list of sequence names.
        n_jobs: Desired number of parts.

    Returns:
        List of non-empty sub-lists. Length is at most ``min(n_jobs,
        len(sequences))`` but may be fewer: ``ceil``-chunk rounding can pack the
        work into fewer groups than requested (e.g. 210 seqs, n_jobs=40 →
        chunk=6 → 35 parts). Callers must use ``len(...)`` of the result, never
        the requested ``n_jobs``, for downstream sizing.
    """
    n_jobs = min(n_jobs, len(sequences))
    chunk = math.ceil(len(sequences) / n_jobs)
    return [sequences[i:i + chunk] for i in range(0, len(sequences), chunk)]


def _scale_mem(mem: str, factor: int) -> str:
    """Multiply a SLURM memory string (e.g. ``16G``, ``32000M``) by *factor*.

    Args:
        mem: Memory string with an optional K/M/G/T unit suffix.
        factor: Positive integer multiplier (the pack size).

    Returns:
        Scaled memory string with the original unit, rounded up to an integer.

    Raises:
        ValueError: If ``mem`` is not a recognised SLURM memory string.
    """
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)\s*", mem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Unrecognised --mem value: {mem!r}")
    value = float(match.group(1)) * factor
    return f"{math.ceil(value)}{match.group(2).upper()}"


def _write_subset_files(
    parts: list[list[str]], tmp_dir: Path, stamp: str, dataset: str
) -> list[Path]:
    """Write each part to a temporary subset text file.

    Args:
        parts: List of sequence-name sub-lists.
        tmp_dir: Directory to write files into (created if absent).
        stamp: Timestamp string used in file names.
        dataset: Dataset key included in file names to avoid collisions when
            submitting multiple datasets (e.g. ``both``) with the same stamp.

    Returns:
        List of paths to the written files, one per part.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths_out = []
    for idx, part in enumerate(parts, start=1):
        p = tmp_dir / f"job_{stamp}_{dataset}_part{idx}.txt"
        p.write_text("\n".join(part) + "\n")
        paths_out.append(p)
    return paths_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit flatsam2 run.py as a SLURM array job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- positional (forwarded to run.py) -----------------------------------
    dataset_choices = sorted(set(CLI_CHOICES + ["both"]))
    parser.add_argument(
        "dataset",
        choices=dataset_choices,
        help="Dataset to run experiments on (`both` submits POT-210 and PTi).",
    )

    # ---- sequence selection (forwarded / replaced) --------------------------
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQ",
        help="Explicit sequence names.  The combined list is then split across jobs.",
    )
    parser.add_argument(
        "--subset",
        metavar="NAME",
        help="Subset file name (configs/subsets/<name>.txt) to select sequences from.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="PlanarTrack split (forwarded to run.py).",
    )

    # ---- run.py pass-through ------------------------------------------------
    parser.add_argument("--base-dir", type=Path, help="Dataset base directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(paths.OUTPUT_DIRECTORY),
        help="Top-level output directory.",
    )
    parser.add_argument("--config", type=Path, help="Tracker config file path.")
    parser.add_argument(
        "--legacy-config",
        type=Path,
        help=(
            "Run the legacy flatsam tracker (WOFTSAM / SAM-H) named by this legacy "
            "config instead of flatsam2's own tracker: the array tasks run "
            "experiments/run_legacy.py under slurm/legacy_env.sh. Path is absolute "
            "or relative to the legacy repo root."
        ),
    )
    parser.add_argument(
        "--flatsam1-root",
        type=Path,
        default=None,
        help="Legacy flatsam repository root (default: run_legacy.py's own default).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Results directory name (default: the config stem).",
    )
    parser.add_argument(
        "--guard-autocast",
        action="store_true",
        help="Legacy runs only: pass --guard-autocast to run_legacy.py.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Legacy runs only: pass --disable-tf32 to run_legacy.py.",
    )
    parser.add_argument(
        "--precomputed-masks",
        type=Path,
        default=None,
        help=(
            "Legacy runs only: replay masks dumped by an earlier run instead of "
            "segmenting (passed through to run_legacy.py)."
        ),
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Pass --save-masks to run.py.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Pass --no-skip-completed to run.py.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Do not submit the dependent evaluate.py job. Use for runs that "
            "produce no corner predictions to score, e.g. the SAM3 candidate-mask "
            "dump (configs/tracker/sam3_candidates.py)."
        ),
    )

    # ---- SLURM-specific (consumed here, NOT forwarded) ----------------------
    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Number of work parts to split sequences across (one python invocation each).",
    )
    slurm.add_argument(
        "--pack",
        type=int,
        default=1,
        metavar="P",
        help="How many work parts to run concurrently inside a single array task "
             "(shared GPU). Total array width = ceil(jobs/pack). Inspect resource "
             "usage afterwards with slurm/summarize_resources.py to tune this.",
    )
    slurm.add_argument("--time", default="05:00:00", help="Wall-clock time limit (HH:MM:SS).")
    slurm.add_argument(
        "--mem",
        default="16G",
        help="Memory per packed work part; the node total is this × --pack.",
    )
    slurm.add_argument(
        "--cpus-per-part",
        type=int,
        default=4,
        metavar="C",
        help="CPU cores per packed work part; --cpus-per-task is this × --pack.",
    )
    slurm.add_argument("--gpus", default="1", help="Number of GPUs per task (passed as --gres=gpu:N).")
    slurm.add_argument("--partition", default=os.environ.get("SLURM_GPU_PARTITION", "amdgpu"),
                       help="SLURM partition name.")
    slurm.add_argument("--qos", default=None, help="SLURM QOS name.")
    slurm.add_argument(
        "--job-name",
        default=None,
        help="SLURM job name (default: <config_stem>_<dataset>).",
    )
    slurm.add_argument(
        "--mail-to",
        default=os.environ.get("SLURM_MAIL_USER", ""),
        metavar="EMAIL",
        help="Email address for the success notification after evaluation.",
    )
    slurm.add_argument(
        "--no-mail",
        action="store_true",
        help="Suppress all SLURM notification mail for this submission: overrides the "
             "sbatch scripts' --mail-type and clears the summary email address. Use for "
             "throwaway/agent-submitted probe jobs.",
    )
    slurm.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without actually submitting.",
    )

    return parser


def results_name(args: argparse.Namespace) -> str:
    """Name of the results sub-directory this submission writes into.

    Args:
        args: Parsed namespace from this script's parser.

    Returns:
        ``--name`` if given, else the (legacy) config's stem, else ``'default'``.
    """
    if args.name is not None:
        return args.name
    config = args.legacy_config or args.config
    return Path(config).stem if config is not None else "default"


def build_run_py_args(args: argparse.Namespace) -> str:
    """Build a shell-quoted run.py / run_legacy.py argument string.

    Sequence selection (``--sequences`` / ``--subset``) is intentionally
    omitted; the batch script injects ``--subset-file <path>`` at runtime for
    each array task.

    Args:
        args: Parsed namespace from this script's parser.

    Returns:
        Shell-quoted string of arguments, ready to be stored in an environment
        variable and expanded by the batch script.
    """
    tokens: list[str] = [args.dataset]

    if args.base_dir is not None:
        tokens += ["--base-dir", str(args.base_dir)]
    tokens += ["--output-dir", str(args.output_dir)]
    if args.legacy_config is not None:
        tokens += ["--legacy-config", str(args.legacy_config)]
        if args.flatsam1_root is not None:
            tokens += ["--flatsam1-root", str(args.flatsam1_root)]
        if args.name is not None:
            tokens += ["--name", args.name]
        if args.precomputed_masks is not None:
            tokens += ["--precomputed-masks", str(args.precomputed_masks)]
        if args.disable_tf32:
            tokens.append("--disable-tf32")
        if args.guard_autocast:
            tokens.append("--guard-autocast")
    elif args.config is not None:
        tokens += ["--config", str(args.config)]
    if args.split is not None:
        tokens += ["--split", args.split]
    if args.save_masks:
        tokens.append("--save-masks")
    if args.no_skip_completed:
        tokens.append("--no-skip-completed")

    return shlex.join(tokens)


def build_evaluate_py_args(args: argparse.Namespace) -> str:
    """Build a shell-quoted evaluate.py argument string.

    Mirrors the run.py arguments that are relevant for evaluation so that
    ``evaluate.py`` resolves the same results directory.

    Args:
        args: Parsed namespace from this script's parser.

    Returns:
        Shell-quoted string of evaluate.py arguments, ready to be stored in
        ``FLATSAM2_EVAL_ARGS`` and expanded by the batch script.
    """
    tokens: list[str] = [args.dataset]

    if args.base_dir is not None:
        tokens += ["--base-dir", str(args.base_dir)]
    tokens += ["--results-dir", str(args.output_dir)]
    if args.config is not None:
        tokens += ["--config", str(args.config)]
    tokens += ["--results-name", results_name(args)]
    if args.split is not None:
        tokens += ["--split", args.split]

    return shlex.join(tokens)


def _submit(args: argparse.Namespace, sequences: list[str], stamp: str) -> None:
    """Submit a SLURM array job + dependent evaluation job for one dataset.

    Args:
        args: Parsed namespace with `dataset` already resolved to a concrete key.
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
    # _split_sequences may yield fewer parts than n_jobs (ceil-chunk rounding can
    # leave trailing parts empty). The array width and all prints below must use
    # the real part count, else high task indices map past the subset-file list.
    n_jobs = len(parts)
    tmp_dir = _REPO_ROOT / "slurm" / "tmp"
    subset_files = _write_subset_files(parts, tmp_dir, stamp, dataset=args.dataset)

    config_stem = results_name(args)
    job_name = args.job_name or f"{config_stem}_{args.dataset}"

    print(f"\nDataset: {args.dataset!r}  config: {config_stem}")
    print(f"Splitting into {n_jobs} job(s):")
    for i, (part, sf) in enumerate(zip(parts, subset_files), start=1):
        print(f"  [{i}/{n_jobs}] {len(part):3d} sequences → {sf.relative_to(_REPO_ROOT)}")

    sbatch_script = _REPO_ROOT / "slurm" / "run.sbatch"
    subset_files_str = ":".join(str(p) for p in subset_files)

    if args.pack < 1:
        print(f"Warning: --pack {args.pack} is invalid; clamping to 1.")
    pack = max(1, args.pack)
    n_array_tasks = math.ceil(n_jobs / pack)
    # Packed parts share one task's allocation, so scale CPUs and RAM by pack.
    total_mem = _scale_mem(args.mem, pack)
    total_cpus = max(1, args.cpus_per_part) * pack
    if pack > 1:
        print(
            f"Packing: {pack} parts per array task → {n_array_tasks} array task(s) "
            f"(each task runs up to {pack} python invocations in parallel on one GPU). "
            f"Per-task alloc: {total_cpus} CPUs, {total_mem} RAM "
            f"({args.cpus_per_part} CPUs / {args.mem} per part). "
            "Check slurm/summarize_resources.py output afterwards to tune --pack."
        )

    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--time={args.time}",
        f"--mem={total_mem}",
        f"--cpus-per-task={total_cpus}",
        f"--gres=gpu:{args.gpus}",
        f"--array=1-{n_array_tasks}",
    ]
    if args.partition:
        sbatch_cmd.append(f"--partition={args.partition}")
    if args.qos:
        sbatch_cmd.append(f"--qos={args.qos}")
    if args.no_mail:
        sbatch_cmd.append("--mail-type=NONE")
    elif args.mail_to:
        sbatch_cmd.append(f"--mail-user={args.mail_to}")

    env = os.environ.copy()
    env["FLATSAM2_RUN_ARGS"] = build_run_py_args(args)
    env["FLATSAM2_SUBSET_FILES"] = subset_files_str
    env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)
    env["FLATSAM2_PACK"] = str(pack)
    env["FLATSAM2_MAIL_TO"] = "" if args.no_mail else args.mail_to
    if args.legacy_config is not None:
        # Legacy trackers need the legacy venv (sam2 + kornia) and their own entry point.
        env["FLATSAM2_ENV_FILE"] = "slurm/legacy_env.sh"
        env["FLATSAM2_ENTRY"] = "experiments/run_legacy.py"

    sbatch_cmd.append(str(sbatch_script))

    eval_script = _REPO_ROOT / "slurm" / "evaluate.sbatch"
    eval_env = os.environ.copy()
    eval_env["FLATSAM2_EVAL_ARGS"] = build_evaluate_py_args(args)
    eval_env["FLATSAM2_REPO_ROOT"] = str(_REPO_ROOT)
    eval_env["FLATSAM2_JOB_NAME"] = job_name
    eval_env["FLATSAM2_DATASET"] = args.dataset
    eval_env["FLATSAM2_CONFIG"] = str(args.config) if args.config else results_name(args)
    eval_env["FLATSAM2_MAIL_TO"] = "" if args.no_mail else args.mail_to
    eval_env["FLATSAM2_ARRAY_JOB_ID"] = "<array_job_id>"

    print(f"Submitting: {' '.join(sbatch_cmd)}")

    if args.dry_run:
        print("(dry-run: not submitting)")
        print("Environment variables that would be set:")
        for k in ("FLATSAM2_RUN_ARGS", "FLATSAM2_SUBSET_FILES", "FLATSAM2_REPO_ROOT",
                  "FLATSAM2_PACK", "FLATSAM2_MAIL_TO",
                  "FLATSAM2_ENV_FILE", "FLATSAM2_ENTRY"):
            if k in env:
                print(f"  {k}={env[k]}")

        if args.no_eval:
            print("(--no-eval: evaluation job would be skipped)")
            return

        eval_sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            "--dependency=afterok:<array_job_id>",
            *(["--mail-type=NONE"] if args.no_mail else []),
            str(eval_script),
        ]
        print(f"Evaluation job (would submit after array job): {' '.join(eval_sbatch_cmd)}")
        print("Environment variables that would be set for evaluation job:")
        for k in ("FLATSAM2_EVAL_ARGS", "FLATSAM2_REPO_ROOT", "FLATSAM2_JOB_NAME",
                  "FLATSAM2_DATASET", "FLATSAM2_CONFIG", "FLATSAM2_MAIL_TO",
                  "FLATSAM2_ARRAY_JOB_ID"):
            print(f"  {k}={eval_env[k]}")
        return

    result = subprocess.run(sbatch_cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting job:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())

    array_id_for_mail = next(
        (tok for tok in result.stdout.split() if tok.isdigit()), None
    )
    if args.no_mail and array_id_for_mail is not None:
        enforce_no_mail(array_id_for_mail)

    if args.no_eval:
        print("(--no-eval: skipping evaluation job submission)")
        return

    array_job_id: Optional[str] = None
    for token in result.stdout.split():
        if token.isdigit():
            array_job_id = token
            break

    if array_job_id is None:
        print("WARNING: could not parse array job ID from sbatch output; skipping evaluation job.")
        return

    eval_env["FLATSAM2_ARRAY_JOB_ID"] = array_job_id

    eval_sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--dependency=afterok:{array_job_id}",
        "--kill-on-invalid-dep=yes",
    ]
    if args.no_mail:
        eval_sbatch_cmd.append("--mail-type=NONE")
    elif args.mail_to:
        eval_sbatch_cmd.append(f"--mail-user={args.mail_to}")
    eval_sbatch_cmd.append(str(eval_script))
    print(f"Submitting evaluation job: {' '.join(eval_sbatch_cmd)}")
    eval_result = subprocess.run(eval_sbatch_cmd, env=eval_env, capture_output=True, text=True)
    if eval_result.returncode != 0:
        print(f"ERROR submitting evaluation job:\n{eval_result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(eval_result.stdout.strip())
    eval_id = next((tok for tok in eval_result.stdout.split() if tok.isdigit()), None)
    if args.no_mail and eval_id is not None:
        enforce_no_mail(eval_id)


def enforce_no_mail(job_id: str) -> None:
    """Force ``MailType=NONE`` on an already-submitted job, and verify it stuck.

    ``sbatch --mail-type=NONE`` is *not* sufficient on this cluster: the value
    recorded is the sbatch script's own ``#SBATCH --mail-type`` directive
    (observed 2026-08-18 -- every job submitted with ``--no-mail`` still showed
    ``MailType=FAIL``). Since ``--no-mail`` exists precisely so that
    agent-submitted probe jobs never reach the user's inbox, and short time
    limits make wall-clock kills routine, the flag has to be enforced after the
    fact rather than requested.

    Args:
        job_id: The submitted job (array jobs: the parent id).
    """
    subprocess.run(
        ["scontrol", "update", f"jobid={job_id}", "MailType=NONE"],
        capture_output=True, text=True,
    )
    check = subprocess.run(
        ["scontrol", "show", "job", job_id], capture_output=True, text=True
    )
    if "MailType=FAIL" in check.stdout or "MailType=END" in check.stdout:
        print(
            f"WARNING: could not silence notification mail for job {job_id}; "
            "cancel it rather than letting it mail the user.",
            file=sys.stderr,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.legacy_config is not None:
        if args.config is not None:
            parser.error("--config and --legacy-config are mutually exclusive.")
        if args.save_masks:
            parser.error("--save-masks is not supported for legacy runs.")
    elif args.precomputed_masks is not None:
        parser.error("--precomputed-masks requires --legacy-config.")
    elif args.disable_tf32:
        parser.error("--disable-tf32 requires --legacy-config.")
    elif args.guard_autocast:
        parser.error("--guard-autocast requires --legacy-config.")
    elif args.name is not None:
        parser.error("--name is only supported together with --legacy-config.")

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
