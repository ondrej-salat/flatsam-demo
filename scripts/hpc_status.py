#!/usr/bin/env python3
"""hpc_status.py — Check experiment computation status via pipeline.yml manifest.

Reads the pipeline manifest (experiments/pipeline.yml by default), discovers
sequence lists from disk, scans output directories, and reports which runs are
complete, partial, or not started.

Works on any repo that has a pipeline.yml following the schema — repo-agnostic.

Usage:
    python scripts/hpc_status.py
    python scripts/hpc_status.py --pipeline loma_v2
    python scripts/hpc_status.py --manifest /path/to/pipeline.yml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# YAML loading (requires PyYAML; available in all scientific Python envs)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.exit(
            "PyYAML not found.\n"
            "Install: pip install pyyaml\n"
            "Or activate your project virtual environment."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Path variable expansion
# ---------------------------------------------------------------------------

def _expand(template: str, variables: dict) -> str:
    """Expand {VAR} placeholders. Unknown variables left as-is."""
    result = template
    for k, v in variables.items():
        result = result.replace(f"{{{k}}}", str(v))
    return result


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def _discover_sequences(dataset: dict, path_vars: dict) -> list[str]:
    """Return sorted list of sequence names based on dataset discovery config.

    Types:
        directories    — subdirectory names under `path`
        text_file      — one sequence name per line in `path`
        nested_csv_ids — `path` lists parent sequences (one per line); each one's
            `child_path` is a CSV whose `id_column` holds the ids of the tasks
            inside it, joined as ``<parent><separator><id>``.  This is how a
            multi-object dataset (MPOT-3K) expands one video into one tracking
            task per planar instance.

    An optional `require_file` template (with ``{seq_name}``) drops sequences
    whose companion file is missing — e.g. UCSB keeps auxiliary directories
    next to its 96 real sequences, and only the latter have annotations.
    """
    seq_cfg = dataset.get("sequences", {})
    seq_type = seq_cfg.get("type", "")
    raw_path = seq_cfg.get("path", "")
    resolved = Path(_expand(raw_path, path_vars))

    names: list[str] = []
    if seq_type == "directories":
        if not resolved.exists():
            return []
        names = sorted(d.name for d in resolved.iterdir() if d.is_dir())

    elif seq_type == "text_file":
        if not resolved.exists():
            return []
        names = [ln.strip() for ln in resolved.read_text().splitlines() if ln.strip()]

    elif seq_type == "nested_csv_ids":
        if not resolved.exists():
            return []
        child_template = seq_cfg.get("child_path", "")
        id_column = int(seq_cfg.get("id_column", 1))
        separator = seq_cfg.get("separator", "__")
        for parent in [ln.strip() for ln in resolved.read_text().splitlines() if ln.strip()]:
            child = Path(_expand(child_template, {**path_vars, "seq_name": parent}))
            if not child.exists():
                continue
            ids = set()
            for line in child.read_text().splitlines():
                fields = [f for f in line.strip().split(",") if f != ""]
                if len(fields) > id_column:
                    ids.add(int(float(fields[id_column])))
            names.extend(f"{parent}{separator}{i}" for i in sorted(ids))

    else:
        return []

    require_file = seq_cfg.get("require_file")
    if require_file:
        names = [
            name
            for name in names
            if Path(_expand(require_file, {**path_vars, "seq_name": name})).exists()
        ]

    return names


# ---------------------------------------------------------------------------
# Pipeline status check
# ---------------------------------------------------------------------------

def _check_pipeline(pipeline: dict, manifest: dict) -> None:
    name = pipeline["name"]
    desc = pipeline.get("description", "")
    path_vars = manifest.get("path_variables", {})
    all_datasets = manifest.get("datasets", {})

    ds_keys = pipeline.get("datasets") or list(all_datasets.keys())
    active_configs = pipeline.get("active_configs") or [None]
    completeness_templates = pipeline.get("completeness") or []
    evaluator = pipeline.get("evaluator")
    # A pipeline may run only part of a dataset (a probe set, a train-split
    # sample). Without this its status reads as 20/300 forever, which is worse
    # than no status at all.
    subset_path = pipeline.get("sequence_subset")
    subset_names: Optional[set[str]] = None
    if subset_path:
        resolved_subset = Path(_expand(str(subset_path), path_vars))
        if not resolved_subset.is_absolute():
            resolved_subset = _REPO_ROOT / resolved_subset
        if resolved_subset.exists():
            subset_names = {
                ln.strip() for ln in resolved_subset.read_text().splitlines() if ln.strip()
            }
        else:
            print(f"  (sequence_subset not found: {resolved_subset})")

    print(f"\n{'=' * 66}")
    print(f"  {name}")
    if desc:
        print(f"  {desc}")
    print(f"{'=' * 66}")

    for config_path in active_configs:
        config_stem = Path(config_path).stem if config_path else None

        if len(active_configs) > 1 or (len(active_configs) == 1 and config_path):
            print(f"\n  config: {config_stem or '(none)'}")

        for ds_key in ds_keys:
            ds = all_datasets.get(ds_key, {})
            ds_label = ds.get("label", ds_key)

            sequences = _discover_sequences(ds, path_vars)
            if subset_names is not None:
                sequences = [s for s in sequences if s in subset_names]
            n_total = len(sequences)

            # Build per-pipeline variable set
            render_vars = dict(path_vars)
            if config_stem:
                render_vars["config_stem"] = config_stem
                render_vars["config"] = config_path or ""
            render_vars["dataset_label"] = ds_label
            render_vars["dataset"] = ds_key

            if n_total == 0:
                seq_path = _expand(ds.get("sequences", {}).get("path", "?"), path_vars)
                print(f"  {ds_label:30s}  INACCESSIBLE  ({seq_path})")
                continue

            # Count complete sequences (all completeness files must exist)
            n_done = 0
            missing_seqs: list[str] = []
            for seq in sequences:
                seq_vars = dict(render_vars)
                seq_vars["seq_name"] = seq
                all_present = all(
                    Path(_expand(tmpl, seq_vars)).exists()
                    for tmpl in completeness_templates
                )
                if all_present:
                    n_done += 1
                else:
                    missing_seqs.append(seq)

            pct = 100 * n_done / n_total if n_total else 0
            if n_done == n_total:
                tag = "DONE   "
            elif n_done == 0:
                tag = "NONE   "
            else:
                tag = "PARTIAL"

            print(f"  {ds_label:30s}  [{tag}]  {n_done:4d}/{n_total}  ({pct:.0f}%)")

            if missing_seqs and n_done < n_total:
                show = min(4, len(missing_seqs))
                preview = ", ".join(missing_seqs[:show])
                if len(missing_seqs) > show:
                    preview += f"  … (+{len(missing_seqs) - show} more)"
                print(f"    missing: {preview}")

            # Evaluator output
            if evaluator and evaluator.get("output"):
                eval_path = Path(_expand(evaluator["output"], render_vars))
                if eval_path.exists():
                    mtime = datetime.fromtimestamp(eval_path.stat().st_mtime)
                    print(f"    eval:    {eval_path.name}  (updated {mtime.strftime('%Y-%m-%d %H:%M')})")
                elif n_done == n_total:
                    print(f"    eval:    NOT RUN  → {eval_path}")

    # Hint: what to run
    runner = pipeline.get("runner", "")
    slurm = pipeline.get("slurm", "")
    if runner or slurm:
        print()
    if runner:
        print(f"  run locally : {runner}")
    if slurm:
        print(f"  submit SLURM: {slurm}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest", default="experiments/pipeline.yml",
        help="Path to pipeline.yml (default: experiments/pipeline.yml)",
    )
    parser.add_argument(
        "--pipeline", default=None,
        help="Filter: only show pipelines whose name contains this substring",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(
            f"Manifest not found: {manifest_path}\n"
            "Create experiments/pipeline.yml or pass --manifest PATH.\n"
            "See ~/.claude/skills/hpc-status/SKILL.md for the schema."
        )

    manifest = _load_yaml(manifest_path)

    print(f"HPC Status Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Manifest: {manifest_path}")

    pipelines = manifest.get("pipelines", [])
    if args.pipeline:
        pipelines = [p for p in pipelines if args.pipeline.lower() in p["name"].lower()]
        if not pipelines:
            sys.exit(f"No pipelines matching {args.pipeline!r}")

    for pl in pipelines:
        _check_pipeline(pl, manifest)

    print(f"\n{'=' * 66}")


if __name__ == "__main__":
    main()
