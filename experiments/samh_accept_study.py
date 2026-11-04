#!/usr/bin/env python3
"""Why is an *accepted* mask-align fit wrong? (SAM-H branch failure anatomy)

The coast study (``experiments/coast_covariance_study.py``) killed the
"coasted-but-good SAM-H gets discarded" theory: on the POT sequences where
corner fusion regresses, the SAM-H branch is not good-but-stale, it is simply
wrong — including on frames its own acceptance test *passed* (median boundary
residual <= ``align_residual_ok_px``). A fit that matches the mask boundary to
within a few pixels while sitting hundreds of pixels from the truth is the
signature of a **pose alias**: for a symmetric silhouette (every POT/PT target
is a rectangle), the boundary signed-distance objective has near-identical
minima at the 90/180/270-degree relabelings, and Gauss-Newton has no reason to
prefer the right one.

This script separates the candidate explanations on existing debug npz, no GPU:

    alias        — error under the best corner relabeling (``best_rotation_error``)
                   is small while the direct error is large
    jump         — the accepted fit is far from the previous frame's estimate
                   (the aliases are far away; genuine tracking steps are small)
    weak evidence— accepted on few boundary samples (``maskh_align_n_ok``)

Each has a different fix, so the shares here decide the next experiment:
an alias needs a correspondence/symmetry constraint, a jump needs a temporal
acceptance guard, weak evidence needs a sample-count floor.

Imports cv2 via flatsam2.datasets — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/samh_accept_study.py \\
        pot --config configs/tracker/corner_fusion_alignh_calibrated.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error, best_rotation_error
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

_WRONG_PX = 15.0  # a fit worse than this is "wrong" (the p@15 threshold)


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(f"samh_accept_study supports pot/pti GT loaders, got {loader!r}")


def _describe(name: str, values: np.ndarray) -> str:
    """One-line median/p90 summary of a sample (or a placeholder when empty)."""
    if values.size == 0:
        return f"{name}: (no frames)"
    return (
        f"{name}: n={values.size}  median={np.median(values):.2f}  "
        f"p90={np.percentile(values, 90):.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=None,
        help="Second run whose samh_H is scored on the same frames — e.g. the "
        "Hough-quad run, to ask whether the frames the aligner gets wrong are "
        "wrong because the mask is bad (both branches fail) or because the "
        "aligner cannot recover from its own drift (the stateless quad branch "
        "is fine on the same frame).",
    )
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)
    sequences = get_sequences(
        args.dataset, base_dir, sequences=args.sequences, subset=args.subset, split=args.split
    )
    load_gt_fn = _load_gt_fn(args.dataset)

    print(
        f"Accepted-fit study: config={args.config.stem}  dataset={ds_info.label}  "
        f"n_sequences={len(sequences)}"
    )

    # Per accepted GT frame: direct error, best-relabeling error, step from the
    # previous frame's SAM-H estimate, the residual that accepted it, n_ok.
    err: list[float] = []
    br_err: list[float] = []
    jump: list[float] = []
    residual: list[float] = []
    n_ok: list[float] = []
    ref_err: list[float] = []  # reference run's samh_H error, accepted frames
    # Same, on the frames the aligner *rejected* (where it coasts).
    rej_err: list[float] = []
    rej_ref_err: list[float] = []
    n_seqs = 0
    ref_dir = (
        compute_output_dir(results_dir, args.dataset, args.reference_config)
        if args.reference_config is not None
        else None
    )

    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "maskh_align_accepted" not in data:
                continue
            n_seqs += 1
            frame_idx = data["frame_idx"]
            accepted = data["maskh_align_accepted"]
            samh_H = data["samh_H"]
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]

            ref_H = None
            if ref_dir is not None:
                ref_path = ref_dir / f"{seq_name}_debug.npz"
                if ref_path.exists():
                    with np.load(ref_path, allow_pickle=False) as ref_data:
                        if len(ref_data["frame_idx"]) == len(frame_idx):
                            ref_H = ref_data["samh_H"]

            for t, fidx in enumerate(frame_idx):
                if fidx >= len(valid) or not valid[fidx]:
                    continue
                if not np.all(np.isfinite(samh_H[t])):
                    continue
                pred = warp_corners(samh_H[t], init_corners)
                gt = gt_corners[fidx]
                this_err = float(alignment_error(gt, pred))
                this_ref = (
                    float(alignment_error(gt, warp_corners(ref_H[t], init_corners)))
                    if ref_H is not None and np.all(np.isfinite(ref_H[t]))
                    else float("nan")
                )
                if not accepted[t]:
                    rej_err.append(this_err)
                    rej_ref_err.append(this_ref)
                    continue
                err.append(this_err)
                ref_err.append(this_ref)
                br_err.append(float(best_rotation_error(gt, pred)[0]))
                prev = samh_H[t - 1] if t > 0 else None
                jump.append(
                    float(alignment_error(warp_corners(prev, init_corners), pred))
                    if prev is not None and np.all(np.isfinite(prev))
                    else float("nan")
                )
                residual.append(float(data["maskh_align_residual_px"][t]))
                n_ok.append(float(data["maskh_align_n_ok"][t]))

    if n_seqs == 0:
        print("No debug npz with align telemetry — nothing to study.")
        return 1

    err_a = np.array(err)
    br_a = np.array(br_err)
    jump_a = np.array(jump)
    res_a = np.array(residual)
    nok_a = np.array(n_ok)
    if err_a.size == 0:
        print("No accepted align fit landed on a GT frame.")
        return 1

    wrong = err_a > _WRONG_PX
    right = ~wrong
    print(
        f"\nAccepted align fits on GT frames: {err_a.size} "
        f"({float(np.mean(wrong)):.1%} wrong at >{_WRONG_PX:.0f}px)"
    )

    # Alias test: wrong by direct labeling, right under a corner relabeling.
    aliased = wrong & (br_a <= _WRONG_PX)
    print(
        f"\n  of the wrong ones, corner-relabeling alias "
        f"(best-rotation error <= {_WRONG_PX:.0f}px): "
        f"{float(np.sum(aliased))}/{int(np.sum(wrong))} "
        f"({float(np.mean(aliased[wrong])) if wrong.any() else 0.0:.1%})"
    )
    print("  " + _describe("wrong-fit direct error (px)", err_a[wrong]))
    print("  " + _describe("wrong-fit best-relabeling error (px)", br_a[wrong]))

    # Jump test: is the wrong accepted fit reachable from the previous estimate?
    print("\n  " + _describe("step from previous SAM-H estimate, wrong fits (px)", jump_a[wrong][np.isfinite(jump_a[wrong])]))
    print("  " + _describe("step from previous SAM-H estimate, good fits  (px)", jump_a[right][np.isfinite(jump_a[right])]))

    # Evidence test: does the acceptance test itself separate them at all?
    print("\n  " + _describe("boundary residual that accepted a wrong fit (px)", res_a[wrong]))
    print("  " + _describe("boundary residual that accepted a good fit  (px)", res_a[right]))
    print("  " + _describe("n_ok backing a wrong fit", nok_a[wrong]))
    print("  " + _describe("n_ok backing a good fit ", nok_a[right]))

    if args.reference_config is not None:
        _report_reference(
            np.array(ref_err), wrong, np.array(rej_err), np.array(rej_ref_err),
            args.reference_config.stem,
        )
    return 0


def _report_reference(
    ref_a: np.ndarray,
    wrong: np.ndarray,
    rej_a: np.ndarray,
    rej_ref_a: np.ndarray,
    ref_name: str,
) -> None:
    """Is the aligner's failure the mask's fault, or the aligner's?

    On the same frame and (near-)identical SAM3 mask, a stateless estimator
    that is right where the incremental aligner is wrong means the evidence was
    there and the aligner failed to use it — recoverable by re-initialization.
    Both being wrong means the mask itself is bad, which no aligner change fixes.

    Args:
        ref_a: Reference-run SAM-H error on the aligner's accepted GT frames.
        wrong: Boolean mask selecting the accepted-but-wrong frames.
        rej_a: Aligner error on GT frames it rejected (coasting).
        rej_ref_a: Reference-run error on those same frames.
        ref_name: Reference config stem, for the printout.
    """
    print(f"\n  reference run ({ref_name}) on the same frames:")
    both = np.isfinite(ref_a) & wrong
    if both.any():
        print(
            f"    accepted-but-wrong frames: reference right (<= {_WRONG_PX:.0f}px) on "
            f"{float(np.mean(ref_a[both] <= _WRONG_PX)):.1%} of {int(both.sum())}"
        )
    ok = np.isfinite(rej_ref_a) & np.isfinite(rej_a)
    if ok.any():
        lost = ok & (rej_a > _WRONG_PX)
        print(
            f"    rejected (coasted) frames: aligner wrong on "
            f"{float(np.mean(rej_a[ok] > _WRONG_PX)):.1%} of {int(ok.sum())}; "
            f"of those, reference right on {float(np.mean(rej_ref_a[lost] <= _WRONG_PX)):.1%}"
        )
        print("    " + _describe("reference error, aligner-lost frames (px)", rej_ref_a[lost]))


if __name__ == "__main__":
    sys.exit(main())
