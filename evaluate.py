"""Evaluate tracking results on POT-210 and PlanarTrack datasets.

This script evaluates tracking results against ground truth and computes
various metrics including alignment error, P@5, P@10, P@15.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from einops import rearrange

from flatsam2.config_aliases import (
    get_config_display_name,
    load_config_aliases_from_file,
)
from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    make_gt_loader,
)
from flatsam2.metrics import aggregate_results, evaluate_sequence
from flatsam2.path_config import paths


def load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise ValueError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    pred = rearrange(
        pred_flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2
    )
    return pred


def print_sequence_results(seq_name: str, results: Dict) -> None:
    print(f"\n{seq_name}:")
    print(f"  Valid frames: {results['num_valid']}")
    print(f"  Mean error: {results['mean_error']:.2f} px")
    print(f"  Median error: {results['median_error']:.2f} px")
    print(f"  P@5:  {results['p@5']:.1f}%")
    print(f"  P@10: {results['p@10']:.1f}%")
    print(f"  P@15: {results['p@15']:.1f}%")
    if "br_mean_error" in results:
        print(f"  Best Rotation:")
        print(f"    Mean error: {results['br_mean_error']:.2f} px")
        print(f"    Median error: {results['br_median_error']:.2f} px")
        print(f"    P@5:  {results['br_p@5']:.1f}%")
        print(f"    P@10: {results['br_p@10']:.1f}%")
        print(f"    P@15: {results['br_p@15']:.1f}%")


def print_aggregate_results(
    results: Dict,
    use_best_rotation: bool = False,
    verbose: bool = False,
    method_name: str = "",
) -> None:
    if verbose:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Number of sequences: {results['num_sequences']}")
        print(f"Total frames: {results['total_frames']}")
        print()
        print("Overall (all frames):")
        print(f"  Mean error: {results['overall_mean_error']:.2f} px")
        print(f"  Median error: {results['overall_median_error']:.2f} px")
        print(f"  P@5:  {results['overall_p@5']:.1f}%")
        print(f"  P@15: {results['overall_p@15']:.1f}%")
        if use_best_rotation and "mean_br_p@5" in results:
            print()
            print("Best Rotation - Mean across sequences:")
            print(f"  P@5:  {results['mean_br_p@5']:.1f}%")
            print(f"  P@15: {results['mean_br_p@15']:.1f}%")
            print()
            print("Best Rotation - Overall (all frames):")
            print(f"  Mean error: {results['overall_br_mean_error']:.2f} px")
            print(f"  Median error: {results['overall_br_median_error']:.2f} px")
            print(f"  P@5:  {results['overall_br_p@5']:.1f}%")
            print(f"  P@15: {results['overall_br_p@15']:.1f}%")

    print("| method | p@5 | p@15 |")
    print(
        f"| {method_name} | {results['overall_p@5']:.1f} | {results['overall_p@15']:.1f} |"
    )


def compute_attribute_aggregates(
    sequence_results: List[Dict],
    annotation_file: Path,
) -> Dict[str, Dict]:
    """Compute aggregate metrics stratified by sequence-level binary attributes.

    Args:
        sequence_results: Per-sequence results dicts (must have 'sequence' key and 'errors' array).
        annotation_file: Path to annotation JSON produced by planartrack_sequence_annotator.py.

    Returns:
        Dict mapping attribute name -> {
            "num_with": int,
            "num_without": int,
            "with": {"p@5": float, "p@15": float},   # present only when num_with > 0
            "without": {"p@5": float, "p@15": float}, # present only when num_without > 0
        }
    """
    annotation_data = json.loads(annotation_file.read_text(encoding="utf-8"))
    annotations: Dict[str, Dict[str, bool]] = annotation_data.get("annotations", {})
    attributes: List[str] = annotation_data.get("attributes", [])

    result: Dict[str, Dict] = {}
    for attr in attributes:
        with_attr = [
            r for r in sequence_results
            if bool(annotations.get(r["sequence"], {}).get(attr, False))
        ]
        without_attr = [
            r for r in sequence_results
            if not bool(annotations.get(r["sequence"], {}).get(attr, False))
        ]

        entry: Dict = {
            "num_with": len(with_attr),
            "num_without": len(without_attr),
        }
        if with_attr:
            agg = aggregate_results(with_attr)
            entry["with"] = {
                "p@5": float(agg["overall_p@5"]),
                "p@15": float(agg["overall_p@15"]),
                "num_sequences": len(with_attr),
            }
        if without_attr:
            agg = aggregate_results(without_attr)
            entry["without"] = {
                "p@5": float(agg["overall_p@5"]),
                "p@15": float(agg["overall_p@15"]),
                "num_sequences": len(without_attr),
            }

        result[attr] = entry

    return result


def print_attribute_aggregates(attr_aggregates: Dict[str, Dict]) -> None:
    """Print per-attribute p@5/p@15 breakdown as an org-mode table.

    Args:
        attr_aggregates: Output of compute_attribute_aggregates().
    """
    if not attr_aggregates:
        return

    print("\nAttribute breakdown:")
    headers = ["attribute", "group", "n", "p@5", "p@15"]
    rows = []
    for attr, entry in attr_aggregates.items():
        for group in ("with", "without"):
            if group in entry:
                g = entry[group]
                rows.append([attr, group, str(g["num_sequences"]),
                              f"{g['p@5']:.1f}", f"{g['p@15']:.1f}"])

    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        parts = []
        for i, (cell, w) in enumerate(zip(cells, col_w)):
            if i <= 1:
                parts.append(f" {cell:<{w}} ")
            else:
                parts.append(f" {cell:>{w}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|-" + "-+-".join("-" * w for w in col_w) + "-|"
    print(fmt_row(headers))
    print(sep)
    prev_attr = None
    for row in rows:
        if prev_attr is not None and row[0] != prev_attr:
            print(sep)
        print(fmt_row(row))
        prev_attr = row[0]


def evaluate_dataset(
    dataset: str,
    base_dir: Path,
    base_results_dir: Path,
    sequences: List[str],
    split: Optional[str] = None,
    output_file: Optional[Path] = None,
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
    config_aliases: Optional[Dict[str, str]] = None,
    attributes_file: Optional[Path] = None,
    results_name: Optional[str] = None,
) -> Dict:
    ds_info = DATASETS[dataset]
    load_gt_fn = make_gt_loader(dataset, initfix_dir)
    # Results normally live under the config's stem; ``results_name`` names the
    # directory directly, for runs whose directory is not a flatsam2 config
    # (e.g. the legacy trackers driven by experiments/run_legacy.py --name).
    if results_name is not None:
        results_dir = base_results_dir / results_name / ds_info.label
    else:
        results_dir = compute_output_dir(base_results_dir, dataset, config_path)

    if output_file is None:
        output_file = results_dir / "eval_results.json"

    config_name = results_name or (config_path.stem if config_path is not None else "default")
    method_name = get_config_display_name(config_name, config_aliases or {})

    print(f"Evaluating {len(sequences)} {ds_info.label} sequences")
    if split:
        print(f"Split: {split}")
    print(f"Results directory: {results_dir}")

    sequence_results = []
    failed_sequences: List[str] = []
    # Each entry: (seq_name, error_message, traceback_str)
    error_log: List[Tuple[str, str, str]] = []

    for seq_name in sequences:
        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            pred_corners = load_predictions(results_dir, seq_name)

            if len(gt_corners) != len(pred_corners):
                print(
                    f"WARNING: {seq_name} - frame count mismatch "
                    f"(GT: {len(gt_corners)}, Pred: {len(pred_corners)})"
                )
                min_len = min(len(gt_corners), len(pred_corners))
                gt_corners = gt_corners[:min_len]
                pred_corners = pred_corners[:min_len]
                valid = valid[:min_len]

            results = evaluate_sequence(
                gt_corners, pred_corners, valid, use_best_rotation=True
            )
            results["sequence"] = seq_name
            sequence_results.append(results)
        except Exception as e:
            failed_sequences.append(seq_name)
            error_log.append((seq_name, str(e), traceback.format_exc()))
            continue

    n_ok = len(sequence_results)
    n_fail = len(failed_sequences)

    if sequence_results:
        aggregate = aggregate_results(sequence_results)
        print_aggregate_results(
            aggregate,
            use_best_rotation=True,
            method_name=method_name,
        )

        attr_aggregates: Optional[Dict] = None
        if attributes_file is not None:
            if not attributes_file.exists():
                print(f"WARNING: attributes file not found, skipping: {attributes_file}")
            else:
                attr_aggregates = compute_attribute_aggregates(sequence_results, attributes_file)
                print_attribute_aggregates(attr_aggregates)

        if output_file:
            output_data: Dict = {
                "aggregate": aggregate,
                "sequences": [
                    {
                        "sequence": res["sequence"],
                        "num_valid": int(res["num_valid"]),
                        "mean_error": float(res["mean_error"]),
                        "median_error": float(res["median_error"]),
                        "p@5": float(res["p@5"]),
                        "p@10": float(res["p@10"]),
                        "p@15": float(res["p@15"]),
                        "br_p@5": float(res["br_p@5"]),
                        "br_p@10": float(res["br_p@10"]),
                        "br_p@15": float(res["br_p@15"]),
                    }
                    for res in sequence_results
                ],
            }
            if attr_aggregates is not None:
                output_data["attribute_aggregates"] = attr_aggregates
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")
    else:
        print("No sequences were successfully evaluated!")

    # --- evaluation summary ---
    print(f"\nEvaluated: {n_ok} ok, {n_fail} failed  (total {n_ok + n_fail})")
    if failed_sequences:
        print(f"Failed sequences ({n_fail}):")
        for name in failed_sequences:
            print(f"{name},", end="")
        print("")

    # --- deferred error messages (capped at 3) ---
    MAX_SHOWN = 3
    if error_log:
        print("\n" + "=" * 60)
        print("ERROR DETAILS")
        print("=" * 60)
        for seq_name, msg, tb in error_log[:MAX_SHOWN]:
            print(f"\nERROR evaluating {seq_name}: {msg}")
            print(tb, end="")
        if n_fail > MAX_SHOWN:
            print(f"\n… and {n_fail - MAX_SHOWN} more error(s) not shown.")

    return aggregate if sequence_results else {}


def main(argv=None) -> int:
    default_alias_config = Path("configs/aliases/default.py")
    default_alias_config = (
        default_alias_config if default_alias_config.exists() else None
    )

    parser = argparse.ArgumentParser(
        description="Evaluate tracking results on POT-210 or PlanarTrack datasets"
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to evaluate on")
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Dataset base directory (uses path_config.py default if not specified)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=False,
        help="Top-level directory containing tracking results (computed automatically if not provided)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to evaluate (if not specified, evaluate all)",
    )
    parser.add_argument(
        "--subset",
        help=(
            "Name of a subset to load from configs/subsets/<name>.txt "
            "(one sequence per line). If provided and --sequences is not given the "
            "subset will be used. If both are provided the intersection will be used."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Dataset split (train or test) — only applicable for datasets that support splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output file for detailed results (JSON format). "
            "Defaults to <results_dir>/<config>/<dataset>/eval_results.json"
        ),
    )
    parser.add_argument(
        "--config", type=Path, help="Path to config file (for directory structure)"
    )
    parser.add_argument(
        "--results-name",
        default=None,
        help=(
            "Name of the results sub-directory under --results-dir, used instead "
            "of the config stem. For runs not produced by a flatsam2 config, e.g. "
            "the legacy trackers (experiments/run_legacy.py)."
        ),
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-sequence initfix files for planartrack-initfix / pti "
            "(defaults to PT_INITFIX_DIR from path_config.py)"
        ),
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=default_alias_config,
        help=(
            "Python file defining config-name display aliases (expects 'aliases' "
            "mapping). Defaults to configs/aliases/default.py when present."
        ),
    )
    parser.add_argument(
        "--attributes-file",
        type=Path,
        default=None,
        help=(
            "JSON file with sequence-level binary attributes produced by "
            "planartrack_sequence_annotator.py. When provided, metrics are also "
            "reported per attribute (sequences with / without each attribute)."
        ),
    )
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY

    if not args.base_dir.exists():
        print(f"ERROR: Dataset directory not found: {args.base_dir}")
        return 1
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        return 1

    try:
        config_aliases = load_config_aliases_from_file(args.alias_config)
    except Exception as e:
        print(f"ERROR: Could not load alias config: {e}")
        return 1

    try:
        sequences = get_sequences(
            args.dataset,
            args.base_dir,
            sequences=args.sequences,
            subset=args.subset,
            split=args.split,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    evaluate_dataset(
        args.dataset,
        args.base_dir,
        args.results_dir,
        sequences,
        args.split,
        args.output,
        args.config,
        initfix_dir=args.initfix_dir,
        config_aliases=config_aliases,
        attributes_file=args.attributes_file,
        results_name=args.results_name,
    )
    return 0


if __name__ == "__main__":
    # The aggregate is written even when sequences failed, so the exit status is
    # the only signal that the JSON describes a partial grid. slurm/submit.py
    # chains this with `afterok`; swallowing the status makes an incomplete
    # evaluation look like a finished one.
    sys.exit(main())
