"""Migrate legacy FlatSAM result files to the FlatSAM2 directory layout.

Legacy layout (per file):

    SRC/[config_name]/masks/[sequence_name]/[sequence_name]_FLATSAM.txt

Special case for POT-210 in the legacy layout:
    the config directory is prefixed with ``pot_``.

New layout (per file):

    DST/[config_name]/[dataset]/[sequence_name].txt

By default this script maps:
    - ``pot_*`` config directories -> ``POT-210`` dataset,
      with the ``pot_`` prefix removed from config_name.
    - ``initfix_*`` config directories -> ``PlanarTrackInitfix`` dataset,
      with the ``initfix_`` prefix removed from config_name.
    - all other config directories -> ``PlanarTrack`` dataset.
"""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import flatsam2.path_config  # noqa: F401 -- loads .env (SLURM_MAIL_USER)


@dataclass(frozen=True)
class MigrationEntry:
    """Single file migration mapping.

    Args:
        source_file: Legacy result file path.
        destination_file: FlatSAM2-style destination path.
        config_name: Destination config directory name.
        dataset_key: Evaluate.py dataset key.
    """

    source_file: Path
    destination_file: Path
    config_name: str
    dataset_key: str


def _parse_sequence_name(file_path: Path) -> str:
    """Extract sequence name from a legacy result filename.

    Args:
        file_path: Legacy result file path ending with ``_FLATSAM.txt``.

    Returns:
        Sequence name without the ``_FLATSAM`` suffix.

    Raises:
        ValueError: If the filename does not match the expected suffix.
    """
    suffix = "_FLATSAM.txt"
    if not file_path.name.endswith(suffix):
        raise ValueError(f"Legacy filename does not end with {suffix!r}: {file_path}")
    return file_path.name[: -len(suffix)]


def _map_config_and_dataset(
    config_dir_name: str,
    planartrack_dataset: str,
    initfix_dataset: str,
    pot_dataset: str,
) -> tuple[str, str, str]:
    """Map legacy config directory naming to flatsam2 config+dataset.

    Args:
        config_dir_name: Legacy config directory name.
        planartrack_dataset: Destination dataset label for non-POT configs.
        initfix_dataset: Destination dataset label for initfix-prefixed configs.
        pot_dataset: Destination dataset label for POT configs.

    Returns:
        Tuple ``(config_name, dataset_label, dataset_key)``.
    """
    if config_dir_name.startswith("pot_"):
        return config_dir_name[len("pot_") :], pot_dataset, "pot"
    if config_dir_name.startswith("initfix_"):
        return (
            config_dir_name[len("initfix_") :],
            initfix_dataset,
            "planartrack-initfix",
        )
    return config_dir_name, planartrack_dataset, "planartrack"


def discover_migrations(
    source_root: Path,
    destination_root: Path,
    planartrack_dataset: str,
    initfix_dataset: str,
    pot_dataset: str,
    selected_config_patterns: Optional[list[str]],
    config_name_prefix: str = "",
) -> list[MigrationEntry]:
    """Discover all legacy files and compute destination paths.

    Args:
        source_root: Root of legacy outputs.
        destination_root: Root of new outputs.
        planartrack_dataset: Destination dataset label for planartrack files.
        initfix_dataset: Destination dataset label for planartrack-initfix files.
        pot_dataset: Destination dataset label for POT files.
        selected_config_patterns: Optional list of legacy config directory name
            patterns to include (supports ``*`` wildcard). If None, include all
            config directories.
        config_name_prefix: String prepended to every destination config name.

    Returns:
        Sorted list of migration entries.
    """
    entries: list[MigrationEntry] = []
    for config_dir in sorted(source_root.iterdir()):
        if not config_dir.is_dir():
            continue

        if selected_config_patterns is not None and not any(
            fnmatch.fnmatch(config_dir.name, pattern)
            for pattern in selected_config_patterns
        ):
            continue

        config_name, dataset, dataset_key = _map_config_and_dataset(
            config_dir.name,
            planartrack_dataset=planartrack_dataset,
            initfix_dataset=initfix_dataset,
            pot_dataset=pot_dataset,
        )
        config_name = config_name_prefix + config_name

        masks_dir = config_dir / "masks"
        if not masks_dir.is_dir():
            continue

        for legacy_file in sorted(masks_dir.glob("*/*_FLATSAM.txt")):
            sequence_name = _parse_sequence_name(legacy_file)
            destination_file = (
                destination_root / config_name / dataset / f"{sequence_name}.txt"
            )
            entries.append(
                MigrationEntry(
                    source_file=legacy_file,
                    destination_file=destination_file,
                    config_name=config_name,
                    dataset_key=dataset_key,
                )
            )

    return entries


def _apply_single_migration(
    entry: MigrationEntry,
    overwrite: bool,
    move: bool,
    dry_run: bool,
) -> Optional[str]:
    """Apply one file migration.

    Args:
        entry: Migration mapping to apply.
        overwrite: Whether to overwrite existing destination files.
        move: Whether to move instead of copy.
        dry_run: Whether to only print planned actions.

    Returns:
        Optional warning message if the file was skipped.
    """
    destination_parent = entry.destination_file.parent
    if entry.destination_file.exists() and not overwrite:
        return None
        # return f"SKIP exists: {entry.destination_file}"

    if dry_run:
        action = "MOVE" if move else "COPY"
        print(f"{action}: {entry.source_file} -> {entry.destination_file}")
        return None

    destination_parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(entry.source_file), str(entry.destination_file))
    else:
        shutil.copy2(entry.source_file, entry.destination_file)
    return None


def _collect_eval_targets(entries: list[MigrationEntry]) -> list[tuple[str, str]]:
    """Collect unique ``(config_name, dataset_key)`` evaluation targets."""
    targets = {(entry.config_name, entry.dataset_key) for entry in entries}
    return sorted(targets)


def _build_eval_args(
    destination_root: Path, dataset_key: str, config_name: str
) -> tuple[str, str]:
    """Build evaluate.py args and fake config path string.

    Args:
        destination_root: New result root.
        dataset_key: Evaluate.py dataset key.
        config_name: Destination config directory name.

    Returns:
        Tuple ``(eval_args, fake_config_path)`` where ``eval_args`` is shell-quoted.
    """
    fake_config_path = Path("configs") / "tracker" / f"{config_name}.py"
    tokens = [
        dataset_key,
        "--results-dir",
        str(destination_root),
        "--config",
        str(fake_config_path),
    ]
    return shlex.join(tokens), str(fake_config_path)


def enqueue_slurm_evaluations(
    entries: list[MigrationEntry],
    destination_root: Path,
    mail_to: str,
    job_prefix: str,
    partition: Optional[str],
    qos: Optional[str],
    dry_run: bool,
) -> None:
    """Enqueue one SLURM evaluate job per migrated config+dataset.

    Args:
        entries: Discovered migration entries.
        destination_root: New result root used by evaluate.py.
        mail_to: Recipient email for evaluate.sbatch notifications.
        job_prefix: Prefix for SLURM job names.
        partition: Optional SLURM partition.
        qos: Optional SLURM QoS.
        dry_run: If True, only print planned sbatch submissions.
    """
    repo_root = Path(__file__).resolve().parent
    eval_script = repo_root / "slurm" / "evaluate.sbatch"
    targets = _collect_eval_targets(entries)
    if not targets:
        print("No evaluation targets to enqueue.")
        return

    for config_name, dataset_key in targets:
        eval_args, fake_config_path = _build_eval_args(
            destination_root, dataset_key, config_name
        )
        job_name = f"{job_prefix}_{config_name}_{dataset_key}".replace("/", "_")

        cmd = ["sbatch", f"--job-name={job_name}"]
        if partition:
            cmd.append(f"--partition={partition}")
        if qos:
            cmd.append(f"--qos={qos}")
        cmd.append(str(eval_script))

        env = os.environ.copy()
        env["FLATSAM2_EVAL_ARGS"] = eval_args
        env["FLATSAM2_REPO_ROOT"] = str(repo_root)
        env["FLATSAM2_JOB_NAME"] = job_name
        env["FLATSAM2_DATASET"] = dataset_key
        env["FLATSAM2_CONFIG"] = fake_config_path
        env["FLATSAM2_MAIL_TO"] = mail_to
        env["FLATSAM2_ARRAY_JOB_ID"] = "migration"

        if dry_run:
            print(f"EVAL DRY-RUN: {' '.join(cmd)}")
            print(f"  FLATSAM2_EVAL_ARGS={eval_args}")
            continue

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(
                f"Failed to submit evaluation job for {config_name}/{dataset_key}:\n"
                f"{result.stderr}"
            )
        print(f"EVAL SUBMITTED [{config_name}/{dataset_key}]: {result.stdout.strip()}")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for legacy result migration.

    Args:
        argv: Optional CLI argv override for testing.

    Returns:
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Migrate legacy FlatSAM result files to FlatSAM2 layout"
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help=(
            "Legacy result root "
            "(contains <config>/masks/<sequence>/<sequence>_FLATSAM.txt)"
        ),
    )
    parser.add_argument(
        "destination_root",
        type=Path,
        help="New result root (writes to <config>/<dataset>/<sequence>.txt)",
    )
    parser.add_argument(
        "--planartrack-dataset",
        default="PlanarTrack",
        help="Destination dataset label for non-pot_* configs (default: PlanarTrack)",
    )
    parser.add_argument(
        "--pot-dataset",
        default="POT-210",
        help="Destination dataset label for pot_* configs (default: POT-210)",
    )
    parser.add_argument(
        "--initfix-dataset",
        default="PlanarTrackInitfix",
        help=(
            "Destination dataset label for initfix_* configs "
            "(default: PlanarTrackInitfix)"
        ),
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Prefix prepended to every destination config directory name",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help=(
            "Only migrate these legacy config directory names under source_root "
            "(supports * wildcard). Example: --configs pot_* initfix_cfgB cfgC"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without modifying files",
    )
    parser.add_argument(
        "--enqueue-slurm-eval",
        action="store_true",
        help="Submit one evaluate.sbatch job per migrated config+dataset",
    )
    parser.add_argument(
        "--eval-mail-to",
        default=os.environ.get("SLURM_MAIL_USER", ""),
        help="Email used by evaluate.sbatch notifications (default: SLURM_MAIL_USER from .env)",
    )
    parser.add_argument(
        "--eval-job-prefix",
        default="migrateeval",
        help="Prefix for evaluation SLURM job names",
    )
    parser.add_argument(
        "--eval-partition",
        default="cpu",
        help="Optional SLURM partition for evaluation jobs",
    )
    parser.add_argument(
        "--eval-qos",
        default=None,
        help="Optional SLURM QoS for evaluation jobs",
    )
    args = parser.parse_args(argv)

    source_root: Path = args.source_root
    destination_root: Path = args.destination_root

    if not source_root.exists() or not source_root.is_dir():
        raise ValueError(f"source_root must be an existing directory: {source_root}")

    entries = discover_migrations(
        source_root=source_root,
        destination_root=destination_root,
        planartrack_dataset=args.planartrack_dataset,
        initfix_dataset=args.initfix_dataset,
        pot_dataset=args.pot_dataset,
        selected_config_patterns=args.configs if args.configs else None,
        config_name_prefix=args.name_prefix,
    )

    if not entries:
        print(f"No legacy files found under {source_root}")
        return 0

    n_skipped = 0
    for entry in entries:
        warning = _apply_single_migration(
            entry,
            overwrite=args.overwrite,
            move=args.move,
            dry_run=args.dry_run,
        )
        if warning is not None:
            n_skipped += 1
            print(warning)

    n_total = len(entries)
    n_done = n_total - n_skipped
    mode = "dry-run" if args.dry_run else ("move" if args.move else "copy")
    print(
        f"Done ({mode}): migrated={n_done}, skipped={n_skipped}, total_discovered={n_total}"
    )

    if args.enqueue_slurm_eval:
        enqueue_slurm_evaluations(
            entries=entries,
            destination_root=destination_root,
            mail_to=args.eval_mail_to,
            job_prefix=args.eval_job_prefix,
            partition=args.eval_partition,
            qos=args.eval_qos,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
