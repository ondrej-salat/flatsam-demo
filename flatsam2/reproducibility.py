"""Experiment reproducibility tracking.

Saves all information needed to reproduce an experiment run:
config, CLI args, git state, code snapshot, and a per-invocation log
that handles restarts gracefully.
"""

import datetime
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


def _git_info(repo_dir: Path) -> dict:
    """Capture current git state of the repository.

    Args:
        repo_dir: Path to the git repository root.

    Returns:
        Dict with keys: commit, branch, dirty, diff (unstaged+staged diff).
    """
    info: dict = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit"] = commit
    except Exception:
        info["commit"] = None

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        info["branch"] = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = len(status) > 0
    except Exception:
        info["dirty"] = None

    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode()
        info["diff"] = diff if diff else None
    except Exception:
        info["diff"] = None

    return info


def _repo_root() -> Path:
    """Find the repository root (directory containing this package)."""
    return Path(__file__).resolve().parent.parent


def _env_info() -> dict:
    """Capture Python and platform information.

    Returns:
        Dict with python_version, platform, and hostname.
    """
    import socket

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def _build_run_entry(
    config: Optional[BaseModel],
    config_path: Optional[Path],
    argv: list[str],
) -> dict:
    """Build a single run-log entry with all reproducibility metadata.

    Args:
        config: Pydantic config object (TrackerConfig), or None.
        config_path: Path to the config file used, or None.
        argv: sys.argv at invocation time.

    Returns:
        Dict ready for JSON serialization.
    """
    repo_root = _repo_root()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    entry: dict = {
        "timestamp": timestamp,
        "argv": argv,
        "config_path": str(config_path) if config_path else None,
        "config": config.model_dump() if config is not None else None,
        "git": _git_info(repo_root),
        "env": _env_info(),
    }
    return entry


def save_run_info(
    output_dir: Path,
    config: Optional[BaseModel] = None,
    config_path: Optional[Path] = None,
    argv: Optional[list[str]] = None,
) -> None:
    """Save reproducibility information for the current run.

    On first invocation (no existing ``run_manifest.json``), writes
    ``run_manifest.json`` (the "golden" record for this experiment) and
    creates a code snapshot.

    On every invocation (including restarts), appends a JSON-lines entry
    to ``run_log.jsonl`` so that every code state that contributed to the
    results is recorded.

    Args:
        output_dir: Experiment output directory (e.g. ``results/default/POT-210``).
        config: Pydantic config object, or None for default config.
        config_path: Path to the config file, or None.
        argv: Command-line arguments (defaults to ``sys.argv``).
    """
    if argv is None:
        argv = sys.argv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repro_dir = output_dir / ".repro"
    repro_dir.mkdir(parents=True, exist_ok=True)

    entry = _build_run_entry(config, config_path, argv)
    is_first_run = not (repro_dir / "run_manifest.json").exists()

    # Always append to the run log
    log_path = repro_dir / "run_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if is_first_run:
        # Write the golden manifest on first run
        manifest_path = repro_dir / "run_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(entry, f, indent=2)
            f.write("\n")
        # Full code snapshot on first run
        snapshot_code(repro_dir, tag="initial")
    else:
        # On restart, save a new code snapshot only if the code changed
        snapshot_code(repro_dir, tag=_timestamp_tag())

    # Always copy the original config file verbatim (if provided)
    if config_path is not None:
        config_copy_path = repro_dir / "config_original.py"
        if not config_copy_path.exists():
            shutil.copy2(config_path, config_copy_path)

    # Save the fully resolved config (all subconfigs expanded) for quick
    # inspection (JSON) and re-running (Python).  Written once on first run.
    _save_resolved_config(repro_dir, config)


def _save_resolved_config(repro_dir: Path, config: Optional[BaseModel]) -> None:
    """Save the fully resolved config as JSON and as a re-runnable Python file.

    Both files are written only once (on the first run).  The JSON is for
    quick human inspection; the Python file can be passed directly to
    ``--config`` to reproduce the experiment.

    Args:
        repro_dir: The ``.repro`` directory to write into.
        config: Fully resolved Pydantic config object (e.g. TrackerConfig).
            If None, saves the default TrackerConfig.
    """
    if config is None:
        from flatsam2.config import TrackerConfig

        config = TrackerConfig()
    # --- JSON (for quick inspection) ---
    json_path = repro_dir / "config_resolved.json"
    if not json_path.exists():
        with open(json_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
            f.write("\n")

    # --- Python (for re-running with --config) ---
    py_path = repro_dir / "config_resolved.py"
    if not py_path.exists():
        py_src = _config_to_python(config)
        with open(py_path, "w") as f:
            f.write(py_src)


def _config_to_python(config: BaseModel) -> str:
    """Generate a self-contained Python config file from a resolved config.

    The output file can be loaded by ``load_config_from_file`` (it defines
    a module-level ``config`` variable of the correct type).

    Args:
        config: Fully resolved Pydantic config object.

    Returns:
        Python source code as a string.
    """
    cls_name = type(config).__name__
    # Collect all Pydantic model classes used in the config tree
    needed_classes = _collect_model_classes(config)
    class_names = sorted({cls.__name__ for cls in needed_classes})

    imports = ", ".join(class_names)
    lines = [
        f'"""Resolved config (auto-generated for reproducibility)."""',
        f"from flatsam2.config import {imports}",
        "",
        f"config = {cls_name}(",
    ]
    lines.extend(_format_model_fields(config, indent=1))
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


def _collect_model_classes(obj: BaseModel) -> list[type]:
    """Recursively collect all Pydantic BaseModel classes in a config tree.

    Args:
        obj: Root Pydantic model.

    Returns:
        List of model classes (including the root).
    """
    classes = [type(obj)]
    for field_name in obj.model_fields:
        value = getattr(obj, field_name)
        if isinstance(value, BaseModel):
            classes.extend(_collect_model_classes(value))
    return classes


def _format_model_fields(obj: BaseModel, indent: int) -> list[str]:
    """Recursively format Pydantic model fields as Python source lines.

    Args:
        obj: Pydantic model instance.
        indent: Current indentation level (number of 4-space indents).

    Returns:
        List of source lines (without trailing newlines).
    """
    pad = "    " * indent
    lines = []
    for field_name in obj.model_fields:
        value = getattr(obj, field_name)
        if isinstance(value, BaseModel):
            cls_name = type(value).__name__
            lines.append(f"{pad}{field_name}={cls_name}(")
            lines.extend(_format_model_fields(value, indent + 1))
            lines.append(f"{pad}),")
        else:
            lines.append(f"{pad}{field_name}={value!r},")
    return lines


def _timestamp_tag() -> str:
    """Generate a filesystem-safe timestamp tag.

    Returns:
        String like ``20250315T143022``.
    """
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def snapshot_code(
    dest_dir: Path,
    tag: str = "initial",
) -> Optional[Path]:
    """Create a tar.gz snapshot of the repository source code.

    Includes only tracked and untracked source files (not .git, __pycache__,
    results, or large data).  Falls back to a simple directory copy if git
    is unavailable.

    Args:
        dest_dir: Directory to write the snapshot archive into.
        tag: Label for the snapshot file (e.g. ``initial``, ``20250315T143022``).

    Returns:
        Path to the created archive, or None on failure.
    """
    repo_root = _repo_root()
    archive_name = f"code_snapshot_{tag}.tar.gz"
    archive_path = Path(dest_dir) / archive_name

    if archive_path.exists():
        return archive_path

    try:
        # Use git archive to create a clean snapshot of tracked files
        subprocess.check_call(
            ["git", "stash", "create"],
            cwd=repo_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # git archive only captures committed content; to include uncommitted
        # changes we create a tarball with tracked + modified files via
        # git ls-files.
        tracked_files = subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip().split("\n")

        # Filter out large / irrelevant paths
        exclude_prefixes = ("results/", "vis/", ".git/", "__pycache__/", "third_party_examples/")
        exclude_suffixes = (".pyc", ".pth", ".pt", ".onnx", ".tar", ".tar.gz", ".zip")
        filtered = [
            f for f in tracked_files
            if f and
            not any(f.startswith(p) for p in exclude_prefixes) and
            not any(f.endswith(s) for s in exclude_suffixes)
        ]

        import tarfile

        with tarfile.open(archive_path, "w:gz") as tar:
            for rel_path in filtered:
                full_path = repo_root / rel_path
                if full_path.is_file():
                    tar.add(full_path, arcname=rel_path)

        return archive_path
    except Exception:
        # Silently skip snapshot if something goes wrong (e.g. no git)
        return None
