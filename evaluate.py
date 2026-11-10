"""Evaluate tracking results on POT-210 and PlanarTrack datasets.

This script evaluates tracking results against ground truth and computes
various metrics including alignment error, P@5, P@10, P@15.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from einops import rearrange

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


def evaluate_dataset(
    dataset: str,
    base_dir: Path,
    base_results_dir: Path,
    sequences: List[str],
    split: Optional[str] = None,
    output_file: Optional[Path] = None,
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
    results_name: Optional[str] = None,
) -> Dict:
    ds_info = DATASETS[dataset]
    load_gt_fn = make_gt_loader(dataset, initfix_dir)
    # Results normally live under the config's stem; ``results_name`` names the
    # directory directly, for runs whose directory is not a flatsam2 config.
    if results_name is not None:
        results_dir = base_results_dir / results_name / ds_info.label
    else:
        results_dir = compute_output_dir(base_results_dir, dataset, config_path)

    if output_file is None:
        output_file = results_dir / "eval_results.json"

    method_name = results_name or (config_path.stem if config_path is not None else "default")

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
        "--subset-file",
        type=Path,
        help=(
            "Path to a subset file (one sequence name per line). Combined with "
            "--sequences, the intersection is used."
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
            "of the config stem. For runs not produced by a flatsam2 config."
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
        sequences = get_sequences(
            args.dataset,
            args.base_dir,
            sequences=args.sequences,
            subset_file=args.subset_file,
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
        results_name=args.results_name,
    )
    return 0


if __name__ == "__main__":
    # The aggregate is written even when sequences failed, so the exit status is
    # the only signal that the JSON describes a partial result.
    sys.exit(main())
