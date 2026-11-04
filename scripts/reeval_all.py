"""Re-run evaluate.py over every existing result to refresh metrics in place.

Evaluation reads predictions + ground truth only (no re-tracking), so this is
cheap. Use it to backfill best-rotation (``br_p@k``) metrics into older
``eval_results.json`` files produced before best-rotation was added.

Scans ``<results_dir>/<config>/<dataset>/eval_results.json`` and re-runs
evaluate.py for each (config, dataset) pair. ``compute_output_dir`` only needs
the config file *stem*, so a synthetic ``<config>.py`` path reproduces the
existing directory without requiring the real config file to exist.

Caveat: re-running rewrites eval_results.json and only re-emits
``attribute_aggregates`` when ``--attributes-file`` is given. Pass it to keep
attribute breakdowns; otherwise they are dropped.

Usage examples::

    python scripts/reeval_all.py --dry-run
    python scripts/reeval_all.py
    python scripts/reeval_all.py --results-dir ~/data/flatsam2/results/
    python scripts/reeval_all.py --attributes-file annotations/pt_attributes.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# evaluate.py lives at the repo root (not inside the installed package), so make
# the repo root importable regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.path_config import paths
from flatsam2.registry import DATASETS

import evaluate

# Map output-dir label (e.g. "POT-210", "PTi") → canonical dataset key.
_KEY_BY_LABEL: dict[str, str] = {ds.label: ds.key for ds in set(DATASETS.values())}


def find_pairs(results_dirs: tuple[Path, ...]) -> list[tuple[str, str, Path]]:
    """Find (config_name, dataset_label, results_root) triples to re-evaluate.

    Args:
        results_dirs: Top-level result roots laid out as
            ``<root>/<config>/<dataset>/eval_results.json``.

    Returns:
        List of ``(config_name, dataset_label, results_root)`` tuples for every
        existing eval_results.json, de-duplicated across roots (first root wins).
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, Path]] = []
    for root in results_dirs:
        if not root.is_dir():
            continue
        for config_dir in sorted(root.iterdir()):
            if not config_dir.is_dir():
                continue
            for ds_dir in sorted(config_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                if not (ds_dir / "eval_results.json").exists():
                    continue
                pair = (config_dir.name, ds_dir.name)
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append((config_dir.name, ds_dir.name, root))
    return pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-run evaluate.py over all existing results to refresh metrics"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Top-level results directory. May be passed multiple times. "
            "Defaults to path_config.py OUTPUT_DIRECTORY."
        ),
    )
    parser.add_argument(
        "--attributes-file",
        type=Path,
        default=None,
        help=(
            "Forwarded to evaluate.py so attribute_aggregates are preserved. "
            "Without it, existing attribute breakdowns are dropped on rewrite."
        ),
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=None,
        help="Forwarded to evaluate.py for config display-name aliases.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the (config, dataset) pairs that would be re-evaluated and exit.",
    )
    args = parser.parse_args(argv)

    roots: list[Path] = [paths.OUTPUT_DIRECTORY.expanduser()]
    if args.results_dir is not None:
        roots.extend(p.expanduser() for p in args.results_dir)
    results_dirs = tuple(dict.fromkeys(roots))

    pairs = find_pairs(results_dirs)
    if not pairs:
        joined = ", ".join(str(p) for p in results_dirs)
        print(f"No eval_results.json found under: {joined}")
        return 0

    print(f"Found {len(pairs)} (config, dataset) result(s) to re-evaluate.")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for config_name, dataset_label, root in pairs:
        ds_key: Optional[str] = _KEY_BY_LABEL.get(dataset_label)
        if ds_key is None:
            print(f"SKIP  {config_name}/{dataset_label}: unknown dataset label")
            n_skip += 1
            continue

        # compute_output_dir only uses .stem, so a synthetic path is enough.
        synthetic_config = f"{config_name}.py"
        eval_argv = [
            ds_key,
            "--config",
            synthetic_config,
            "--results-dir",
            str(root),
        ]
        if args.attributes_file is not None:
            eval_argv += ["--attributes-file", str(args.attributes_file)]
        if args.alias_config is not None:
            eval_argv += ["--alias-config", str(args.alias_config)]

        print(f"\n=== {config_name} / {dataset_label} ({ds_key}) ===")
        if args.dry_run:
            print("  evaluate.py " + " ".join(eval_argv))
            n_ok += 1
            continue

        try:
            rc = evaluate.main(eval_argv)
        except Exception as e:
            print(f"FAIL  {config_name}/{dataset_label}: {e}")
            n_fail += 1
            continue
        if rc == 0:
            n_ok += 1
        else:
            print(f"FAIL  {config_name}/{dataset_label}: evaluate.py rc={rc}")
            n_fail += 1

    verb = "would re-evaluate" if args.dry_run else "re-evaluated"
    print(f"\nDone: {verb} {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
