#!/usr/bin/env python3
"""Offline calibration check for corner_fusion's per-branch covariances.

For any existing ``corner_fusion`` run -- calibrated or not -- joins each
frame's stored ``loma_H``/``samh_H`` candidate with GT corners and computes
the squared Mahalanobis error under that branch's own covariance.
``fusion_{loma,samh}_cov`` in the debug npz holds each branch's *raw* fit
covariance: ``FusionTelemetry`` is built from the unscaled arrays, the
``*_cov_scale`` factors are applied only to the local copies used for the
decision (``flatsam2/tracking/corner_fusion.py``). So a calibrated run is a
valid input and no separate uncalibrated grid is needed. Under a
correctly calibrated Gaussian model this statistic is chi-square(2): median
1.386, 5% of (frame, corner) pairs should exceed the 95% quantile 5.99.

Reports median / tail-exceedance per branch and the median-matching scale
correction that would recalibrate each one -- the same diagnostic that
produced ``corner_fusion_alignh_calibrated``'s ``loma_cov_scale=32.0`` /
``samh_cov_scale=70.0`` (n=20 PTi sample, see docs/2026-08-07.md). Run this
at larger n to check those constants still hold.

No mask IO, no dataset images -- just debug npz + GT annotation files. Still
imports cv2 (via flatsam2.datasets), so submit via slurm/submit_cmd.py rather
than running bare on the login node (see CLAUDE.md "Which machine am I on?").

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/fusion_calibration_check.py \\
        pti --config configs/tracker/corner_fusion_wholeframe.py --split test
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir
from flatsam2.corner_covariance import mahalanobis_sq
from flatsam2.datasets import pot_load_gt, pti_load_gt, warp_corners
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

# chi-square(2) reference quantiles: median (50%) and 95% tail.
_CHI2_2_MEDIAN = 1.386
_CHI2_2_P95 = 5.99


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(
        f"fusion_calibration_check only supports pot/pti GT loaders, got {loader!r}"
    )


def _branch_chi2_stats(
    H_stack: np.ndarray,
    cov_stack: np.ndarray,
    frame_idx: np.ndarray,
    init_corners: np.ndarray,
    gt_corners: np.ndarray,
    valid: np.ndarray,
) -> list[float]:
    """Squared Mahalanobis error (pred vs GT) per (frame, corner) with usable data.

    Args:
        H_stack: (T, 3, 3) branch homography per tracked frame, NaN when unavailable.
        cov_stack: (T, 4, 3) per-corner covariance packed (xx, xy, yy), NaN when unavailable.
        frame_idx: (T,) 0-based frame index into ``gt_corners``/``valid``.
        init_corners: (2, 4) template corners.
        gt_corners: (N, 2, 4) GT corners for the whole sequence.
        valid: (N,) GT-availability flags.

    Returns:
        List of chi-square(2) statistics, one per (frame, corner) with a
        finite H and finite covariance for that corner.
    """
    stats: list[float] = []
    for t, fidx in enumerate(frame_idx):
        if fidx >= len(valid) or not valid[fidx]:
            continue
        H = H_stack[t]
        if not np.all(np.isfinite(H)):
            continue
        pred = warp_corners(H, init_corners)  # (2, 4)
        gt = gt_corners[fidx]  # (2, 4)
        for k in range(4):
            xx, xy, yy = cov_stack[t, k]
            if not np.isfinite(xx) or not np.isfinite(xy) or not np.isfinite(yy):
                continue
            cov = np.array([[xx, xy], [xy, yy]])
            delta = pred[:, k] - gt[:, k]
            stats.append(mahalanobis_sq(delta, cov))
    return stats


def _report(branch: str, stats: list[float]) -> None:
    if not stats:
        print(f"{branch}: no (frame, corner) pairs with usable H + covariance")
        return
    arr = np.array(stats)
    median = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    tail_frac = float(np.mean(arr > _CHI2_2_P95))
    scale = median / _CHI2_2_MEDIAN  # multiplicative cov-scale correction
    print(
        f"{branch}: n={len(arr)}  median_chi2={median:.2f} (nominal {_CHI2_2_MEDIAN})  "
        f"p90={p90:.2f}  p99={p99:.2f}  frac>chi2_95%={tail_frac:.1%} (nominal 5%)  "
        f"implied cov_scale={scale:.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)

    sequences = get_sequences(args.dataset, base_dir, subset=args.subset, split=args.split)
    load_gt_fn = _load_gt_fn(args.dataset)

    print(f"Calibration check: config={args.config.stem}  dataset={ds_info.label}  "
          f"n_sequences={len(sequences)}")

    loma_stats: list[float] = []
    samh_stats: list[float] = []
    n_seqs_with_telemetry = 0
    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            print(f"  (skip, no debug npz) {seq_name}")
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "fusion_loma_cov" not in data:
                print(f"  (skip, not a corner_fusion run) {seq_name}")
                continue
            n_seqs_with_telemetry += 1
            frame_idx = data["frame_idx"]
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]

            loma_stats += _branch_chi2_stats(
                data["loma_H"], data["fusion_loma_cov"], frame_idx, init_corners, gt_corners, valid
            )
            samh_stats += _branch_chi2_stats(
                data["samh_H"], data["fusion_samh_cov"], frame_idx, init_corners, gt_corners, valid
            )

    print(f"\nSequences with corner_fusion telemetry: {n_seqs_with_telemetry}/{len(sequences)}")
    _report("LoMa", loma_stats)
    _report("SAM-H", samh_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
