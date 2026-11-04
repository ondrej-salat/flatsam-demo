#!/usr/bin/env python3
"""Is the Hough branch's line-intersection corner covariance correctly scaled?

``flatsam2.corner_covariance.line_intersection_cov`` turns two fitted boundary
lines into a corner position and a 2x2 covariance. It is implemented, unit
tested, and **not called by any tracker** — it was written for the quad SAM-H
branch and never wired in. Before wiring it in, its *scale* has to be trusted,
and the module docstring is explicit that this is where a factor-of-5 bug
survived once already, precisely because the Monte-Carlo cross-check "was never
run" for this estimator. This script is that cross-check.

Two levels, cheapest first:

**A. Monte-Carlo self-consistency (no data).** Put two lines through a known
corner at a known angle, sample points on them, perturb the points with known
perpendicular noise, and intersect. Repeat. The scatter of the intersections is
the truth the analytic covariance is trying to predict. A correct estimator
matches it; a mis-scaled one is off by a constant factor that shows up
identically at every angle and sample count, which is what makes this test able
to name the bug rather than just fail.

Reported as the ratio of predicted to empirical standard deviation along each
principal axis (1.00 is correct) and as the median Mahalanobis distance of the
empirical samples under the predicted covariance, which should be chi-square(2)
with median 1.386.

**B. Calibration on real SAM3 boundaries.** The synthetic test assumes the model
is right and only checks the arithmetic. Real mask boundaries are systematically
wrong, not merely noisy — SAM3 spills onto a laminated sheet, an occluder cuts
the silhouette — and ``fit_line_tls`` estimating noise from its own residual
cannot see that. So the same statistic is computed against ground truth on
stored masks. Under-dispersion here with a clean result in A means the estimator
is arithmetically right and *optimistic*, which is a scale factor to fit, not a
bug to fix (the same situation ``samh_cov_scale`` already handles for the other
branch).

Pure CPU, no GPU and no model: A is numpy only, B reads stored masks.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/line_covariance_validation.py
    python slurm/submit_cmd.py --gpus 0 -- python experiments/line_covariance_validation.py \\
        --real pot --limit 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.corner_covariance import line_intersection_cov, mahalanobis_sq

_CHI2_2_MEDIAN = 1.386
_RNG_SEED = 0
# A predicted corner further than this from every GT corner is not a match,
# and scoring it would measure the correspondence step rather than the
# covariance. Every third annotated frame is enough for a calibration median.
_MATCH_RADIUS_PX = 40.0
_FRAME_STRIDE = 3


def _sample_line(
    corner: np.ndarray,
    angle_rad: float,
    n_pts: int,
    extent_px: float,
    noise_px: float,
    rng: np.random.Generator,
    offset_px: float,
) -> np.ndarray:
    """Points along a line through ``corner``, with perpendicular noise.

    Args:
        corner: (2,) point the line passes through.
        angle_rad: Line direction.
        n_pts: Number of samples.
        extent_px: Points are spread over ``[offset, offset + extent]`` along
            the line, so a boundary segment that does not straddle the corner
            (the realistic case) can be modelled.
        noise_px: Perpendicular Gaussian noise std.
        rng: Random generator.
        offset_px: Distance from the corner at which the segment starts.

    Returns:
        (2, n_pts) points.
    """
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    normal = np.array([-direction[1], direction[0]])
    t = offset_px + np.linspace(0.0, extent_px, n_pts)
    pts = corner[:, None] + direction[:, None] * t[None, :]
    return pts + normal[:, None] * rng.normal(0.0, noise_px, size=n_pts)[None, :]


def monte_carlo_case(
    angle_deg: float,
    n_pts: int,
    noise_px: float,
    extent_px: float,
    offset_px: float,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """One synthetic configuration: predicted vs empirical corner scatter.

    Args:
        angle_deg: Angle between the two lines.
        n_pts: Points sampled per line.
        noise_px: Perpendicular point noise std (px), given to the estimator so
            the test isolates the propagation from the noise estimation.
        extent_px: Length of each sampled boundary segment.
        offset_px: Gap between the corner and the start of each segment.
        n_trials: Monte-Carlo repetitions.
        rng: Random generator.

    Returns:
        Dict of diagnostics; ``std_ratio_*`` are predicted/empirical and should
        be 1.00.
    """
    corner = np.array([100.0, 100.0])
    angle_a = 0.0
    angle_b = np.deg2rad(angle_deg)

    corners = np.zeros((n_trials, 2))
    predicted = np.zeros((n_trials, 2, 2))
    for i in range(n_trials):
        pts_a = _sample_line(corner, angle_a, n_pts, extent_px, noise_px, rng, offset_px)
        pts_b = _sample_line(corner, angle_b, n_pts, extent_px, noise_px, rng, offset_px)
        est_corner, cov = line_intersection_cov(pts_a, pts_b, noise_px=noise_px)
        corners[i] = est_corner
        predicted[i] = cov

    empirical_cov = np.cov(corners.T)
    mean_predicted = predicted.mean(axis=0)

    # Compare along the empirical principal axes, so a covariance that is right
    # in one direction and wrong in the other cannot average out to "correct".
    eigvals, eigvecs = np.linalg.eigh(empirical_cov)
    ratios = []
    for k in range(2):
        axis = eigvecs[:, k]
        emp_var = float(axis @ empirical_cov @ axis)
        pred_var = float(axis @ mean_predicted @ axis)
        ratios.append(np.sqrt(pred_var / emp_var) if emp_var > 0 else np.nan)

    deltas = corners - corners.mean(axis=0)
    d2 = np.array([mahalanobis_sq(deltas[i], predicted[i]) for i in range(n_trials)])

    return {
        "angle_deg": angle_deg,
        "n_pts": n_pts,
        "noise_px": noise_px,
        "std_ratio_minor": ratios[0],
        "std_ratio_major": ratios[1],
        "median_chi2": float(np.median(d2)),
        "empirical_std_px": float(np.sqrt(np.trace(empirical_cov) / 2)),
    }


def run_monte_carlo(n_trials: int) -> int:
    """Sweep angle, sample count and noise; report scale fidelity.

    Args:
        n_trials: Repetitions per configuration.

    Returns:
        Process exit status: 0 when every configuration is within tolerance.
    """
    rng = np.random.default_rng(_RNG_SEED)
    print(f"A. Monte-Carlo self-consistency ({n_trials} trials per row)")
    print("   std ratios are predicted/empirical; 1.00 is correct.")
    print("   median chi2 should be 1.386 for a correctly scaled 2x2 covariance.\n")
    header = (
        f"  {'angle':>6} {'n_pts':>6} {'noise':>6} {'emp std':>8} "
        f"{'ratio minor':>12} {'ratio major':>12} {'med chi2':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for angle_deg in (90.0, 60.0, 30.0, 15.0):
        for n_pts in (20, 100, 500):
            for noise_px in (0.5, 2.0):
                row = monte_carlo_case(
                    angle_deg, n_pts, noise_px,
                    extent_px=120.0, offset_px=10.0,
                    n_trials=n_trials, rng=rng,
                )
                rows.append(row)
                print(
                    f"  {row['angle_deg']:6.0f} {row['n_pts']:6d} {row['noise_px']:6.1f} "
                    f"{row['empirical_std_px']:8.3f} {row['std_ratio_minor']:12.3f} "
                    f"{row['std_ratio_major']:12.3f} {row['median_chi2']:9.3f}"
                )

    ratios = np.array(
        [r["std_ratio_minor"] for r in rows] + [r["std_ratio_major"] for r in rows]
    )
    chi2 = np.array([r["median_chi2"] for r in rows])
    print(
        f"\n  across all {len(rows)} configurations: std ratio "
        f"{np.nanmin(ratios):.3f} - {np.nanmax(ratios):.3f} (median {np.nanmedian(ratios):.3f}), "
        f"median chi2 {np.nanmin(chi2):.2f} - {np.nanmax(chi2):.2f}"
    )
    # A constant offset across every configuration is the signature of a scale
    # bug; a drift with angle or n_pts is a propagation bug. Both are worth
    # naming explicitly rather than leaving to the reader.
    within = float(np.nanmedian(np.abs(ratios - 1.0)))
    if within < 0.10:
        print("  VERDICT: correctly scaled (median deviation from 1.00 is "
              f"{100 * within:.1f}%).")
        return 0
    print(f"  VERDICT: MIS-SCALED — median deviation {100 * within:.1f}%. "
          f"Implied correction factor {np.nanmedian(ratios):.3f}.")
    return 1


def line_intersection_cov_capped(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    noise_px: float,
    n_eff_cap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """``line_intersection_cov`` with the effective sample size capped.

    The deployed estimator assumes independent per-point perpendicular noise, so
    its uncertainty falls as 1/sqrt(n) in the number of contour pixels. Part B
    shows that is the wrong model for a segmentation boundary: the true corner
    error is flat in n while the prediction keeps shrinking, because a mask
    boundary is *systematically* displaced, not noisy — 2000 pixels along a
    spilled edge are one wrong observation, not 2000 independent ones.

    Capping the effective count at ``n_eff_cap`` is the minimal expression of
    that: below the cap the fit still benefits from more evidence, above it the
    extra pixels are treated as redundant, which is what they are.

    Args:
        pts_a: (2, N) boundary points of the first line.
        pts_b: (2, M) boundary points of the second line.
        noise_px: Perpendicular boundary error (px). Fixed rather than estimated
            from the residual, which measures how straight the contour is, not
            how right it is.
        n_eff_cap: Maximum effective sample size per line.

    Returns:
        ``(corner (2,), covariance (2, 2))``.
    """
    from flatsam2.corner_covariance import (
        _intersect,
        _intersection_jacobian,
        _line_param_cov,
        _line_params,
    )

    params = []
    for pts in (pts_a, pts_b):
        n_pts = pts.shape[1]
        centroid = pts.mean(axis=1)
        centered = pts - centroid[:, None]
        scatter = (centered @ centered.T) / n_pts
        eigvals, eigvecs = np.linalg.eigh(scatter)
        normal = eigvecs[:, 0]
        spread_along_line = float(eigvals[1])
        n_eff = min(float(n_pts), n_eff_cap)
        var = noise_px**2
        sigma_phi = float(np.sqrt(var / (n_eff * max(spread_along_line, 1e-12))))
        sigma_rho = float(np.sqrt(var / n_eff))
        phi, rho = _line_params(centroid, normal)
        params.append((centroid, phi, rho, sigma_phi, sigma_rho))

    (cen_a, phi_a, rho_a, sphi_a, srho_a) = params[0]
    (cen_b, phi_b, rho_b, sphi_b, srho_b) = params[1]
    corner = _intersect(phi_a, rho_a, phi_b, rho_b)
    param_cov = np.zeros((4, 4))
    param_cov[:2, :2] = _line_param_cov(phi_a, cen_a, sphi_a, srho_a)
    param_cov[2:, 2:] = _line_param_cov(phi_b, cen_b, sphi_b, srho_b)
    jac = _intersection_jacobian(phi_a, rho_a, phi_b, rho_b)
    return corner, jac @ param_cov @ jac.T


def _available_frames(mask_dir: Path) -> set:
    """0-based frame indices with a selected-mask PNG, from a single readdir."""
    import os

    frames = set()
    if not mask_dir.is_dir():
        return frames
    with os.scandir(mask_dir) as it:
        for entry in it:
            name = entry.name
            if len(name) == 9 and name.endswith(".png") and name[:5].isdigit():
                frames.add(int(name[:5]) - 1)
    return frames


def _corner_covs_from_mask(
    mask: np.ndarray,
    topk_lines: int,
    noise_px: Optional[float],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Corners and covariances from one mask, via the deployed Hough pipeline.

    Re-runs the same stages ``mask_to_homography`` runs, but keeps the per-line
    point sets that it discards, which is exactly what the covariance needs.

    Args:
        mask: (H, W) binary mask.
        topk_lines: Number of Hough lines, as in the tracker config.
        noise_px: Perpendicular noise; None estimates it from the fit residual.

    Returns:
        List of ``(corner, cov, n_boundary_points, pts_a, pts_b)``.
    """
    import cv2

    from flatsam2.hough_homography import (
        _approx_touching_edges,
        _precompute_contour_samples,
        contour_hough,
        lines_to_points,
        p2e,
        refine_lines,
    )

    contours_cv, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        c[:, 0, :].T.astype(np.float64) for c in contours_cv if c.shape[0] > 20
    ]
    if not contours:
        return []
    # Points touching the image border are *excluded* from the fit, so this flag
    # has to be built exactly as mask_to_homography builds it -- marking them all
    # True excludes every point and yields zero lines.
    on_boundaries = [
        np.logical_or.reduce((
            _approx_touching_edges(c + np.array([[0], [1]]), mask.shape, margin=1),
            _approx_touching_edges(c + np.array([[0], [-1]]), mask.shape, margin=1),
            _approx_touching_edges(c + np.array([[1], [0]]), mask.shape, margin=1),
            _approx_touching_edges(c + np.array([[-1], [0]]), mask.shape, margin=1),
        ))
        for c in contours
    ]
    samples = _precompute_contour_samples(contours, on_boundaries)
    lines = contour_hough(
        mask, contours, on_boundaries, topk=topk_lines, _samples=samples
    )
    if lines is None or lines.shape[1] < 2:
        return []
    refined, line_pts = refine_lines(lines, contours, on_boundaries, _samples=samples)
    if refined.shape[1] < 2:
        return []
    intersections_h, parents = lines_to_points(refined)
    intersections = p2e(intersections_h)

    out = []
    for k, (line_a, line_b) in enumerate(parents):
        pts_a, pts_b = line_pts[line_a], line_pts[line_b]
        if pts_a.shape[1] < 2 or pts_b.shape[1] < 2:
            continue
        _, cov = line_intersection_cov(pts_a, pts_b, noise_px=noise_px)
        out.append((intersections[:, k], cov, pts_a.shape[1] + pts_b.shape[1], pts_a, pts_b))
    return out


def run_real(dataset: str, limit: int, noise_px: Optional[float], topk_lines: int) -> int:
    """Calibration of the same estimator on stored SAM3 masks against GT corners.

    Each predicted corner is matched to its nearest GT corner and the squared
    Mahalanobis distance of that offset is accumulated. Matching by proximity
    rather than by label is deliberate: the Hough pipeline's intersections carry
    no naming, and this measures the *covariance*, not the correspondence step
    that sits on top of it. Pairs further than ``_MATCH_RADIUS_PX`` are dropped,
    because a corner matched to the wrong GT corner measures nothing.

    Args:
        dataset: ``pot`` or ``pti``.
        limit: Maximum sequences to read.
        noise_px: Perpendicular noise; None estimates it from the fit residual,
            which is the deployed default.
        topk_lines: Hough lines per frame.

    Returns:
        Process exit status (always 0; this is a measurement, not a gate).
    """
    import cv2  # noqa: F401  (imported for its side effect on _load_mask's path)

    from flatsam2.datasets import pot_load_gt, pti_load_gt
    from flatsam2.path_config import paths
    from flatsam2.registry import DATASETS
    from flatsam2.sequences import get_sequences

    ds_info = DATASETS[dataset]
    base_dir = getattr(paths, ds_info.base_dir_attr)
    mask_root = paths.OUTPUT_DIRECTORY / "sam3_candidates" / ds_info.label / "masks"
    if dataset == "pot":
        load_gt = pot_load_gt
    else:
        def load_gt(b, s):
            return pti_load_gt(b, s, paths.PT_INITFIX_DIR)

    sequences = [
        s for s in get_sequences(dataset, base_dir) if (mask_root / s).is_dir()
    ][:limit]
    print(f"\nB. Calibration on stored SAM3 masks: {dataset}, {len(sequences)} sequences")
    print(f"   noise_px={'from residual' if noise_px is None else noise_px}, "
          f"topk_lines={topk_lines}, match radius {_MATCH_RADIUS_PX:.0f} px")

    d2_all: List[float] = []
    offsets: List[float] = []
    pred_std: List[float] = []
    n_points: List[float] = []
    variants: List[tuple] = []
    n_frames = n_corners = 0
    for seq_name in sequences:
        try:
            gt_corners, valid = load_gt(base_dir, seq_name)
        except Exception:
            continue
        mask_dir = mask_root / seq_name
        for frame_idx in sorted(_available_frames(mask_dir))[::_FRAME_STRIDE]:
            if frame_idx >= len(valid) or not valid[frame_idx]:
                continue
            img = cv2.imread(str(mask_dir / f"{frame_idx + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            found = _corner_covs_from_mask(img > 127, topk_lines, noise_px)
            if not found:
                continue
            n_frames += 1
            gt = gt_corners[frame_idx]                      # (2, 4)
            for corner, cov, _n, _pa, _pb in found:
                dists = np.linalg.norm(gt - corner[:, None], axis=0)
                k = int(np.argmin(dists))
                if dists[k] > _MATCH_RADIUS_PX:
                    continue
                n_corners += 1
                offsets.append(float(dists[k]))
                d2_all.append(mahalanobis_sq(corner - gt[:, k], cov))
                # Predicted 1-D scale of the ellipse, to compare directly with
                # the offset it is supposed to explain, and the sample count the
                # 1/sqrt(n) shrinkage is driven by.
                pred_std.append(float(np.sqrt(np.trace(cov) / 2)))
                n_points.append(float(_n))
                variants.append((corner - gt[:, k], _pa, _pb, float(_n)))

    if not d2_all:
        print("   no matched corners — nothing to report")
        return 0
    d2 = np.array(d2_all)
    offs = np.array(offsets)
    median = float(np.median(d2))
    print(f"\n   frames {n_frames}, matched corners {n_corners}")
    print(f"   offset from GT: median {np.median(offs):.2f} px, p90 {np.percentile(offs, 90):.2f} px")
    print(f"   median chi2 {median:.2f} (nominal {_CHI2_2_MEDIAN})  "
          f"frac>5.99 {100 * float(np.mean(d2 > 5.99)):.1f}% (nominal 5%)")
    print(f"   implied variance scale {median / _CHI2_2_MEDIAN:.1f}x "
          f"(std scale {np.sqrt(median / _CHI2_2_MEDIAN):.1f}x)")

    ps = np.array(pred_std)
    npts = np.array(n_points)
    print(f"\n   predicted std: median {np.median(ps):.3f} px, "
          f"against a median GT offset of {np.median(offs):.2f} px")
    # If a single constant can fix this, the implied scale must not move with the
    # number of boundary points; the 1/sqrt(n) shrinkage says it will.
    print(f"   {'n boundary pts':>16} {'corners':>8} {'pred std':>9} "
          f"{'offset':>8} {'med chi2':>9}")
    edges = np.percentile(npts, [0, 25, 50, 75, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (npts >= lo) & (npts <= hi)
        if sel.sum() < 10:
            continue
        print(f"   {int(lo):7d}-{int(hi):<8d} {int(sel.sum()):8d} "
              f"{np.median(ps[sel]):9.3f} {np.median(offs[sel]):8.2f} "
              f"{np.median(d2[sel]):9.1f}")
    _sweep_capped(variants)
    print("\n   Read with part A: A says the propagation is right under its own model,")
    print("   so an inflation here is the model's blind spot — the boundary being")
    print("   systematically wrong rather than noisy — and is a scale factor to fit,")
    print("   exactly as samh_cov_scale already does for the other branch.")
    return 0


def _sweep_capped(variants: List[tuple]) -> None:
    """Grid the capped estimator: which (noise, n_eff cap) is calibrated *and* flat?

    Two properties are needed, and the median alone shows only one. A cap that
    lands the median chi2 on 1.386 but still drifts across the n quartiles has
    merely re-centred a wrong model; the flat column is what says the 1/sqrt(n)
    dependence is gone.

    Args:
        variants: ``(delta (2,), pts_a, pts_b, n_points)`` per matched corner.
    """
    if not variants:
        return
    npts = np.array([v[3] for v in variants])
    quartiles = np.percentile(npts, [0, 25, 50, 75, 100])
    print(f"\n   capped-estimator sweep over {len(variants)} corners")
    print(f"   {'noise':>6} {'cap':>6} {'med chi2':>9} {'frac>5.99':>10} "
          f"{'chi2 by n quartile (drift)':>34}")
    for noise_px in (1.0, 2.0, 3.0):
        for cap in (1.0, 2.0, 5.0, 10.0, 25.0):
            d2 = np.empty(len(variants))
            for i, (delta, pts_a, pts_b, _n) in enumerate(variants):
                _, cov = line_intersection_cov_capped(pts_a, pts_b, noise_px, cap)
                d2[i] = mahalanobis_sq(delta, cov)
            by_q = []
            for lo, hi in zip(quartiles[:-1], quartiles[1:]):
                sel = (npts >= lo) & (npts <= hi)
                by_q.append(float(np.median(d2[sel])) if sel.sum() >= 10 else np.nan)
            drift = np.nanmax(by_q) / max(np.nanmin(by_q), 1e-9)
            cells = " ".join(f"{v:6.2f}" for v in by_q)
            print(f"   {noise_px:6.1f} {cap:6.0f} {np.median(d2):9.2f} "
                  f"{100 * float(np.mean(d2 > 5.99)):9.1f}% {cells}  x{drift:.1f}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument(
        "--real", default=None, choices=["pot", "pti"],
        help="Also run part B against stored masks of this dataset.",
    )
    parser.add_argument("--limit", type=int, default=40, help="Sequences for part B.")
    parser.add_argument("--noise-px", type=float, default=None)
    parser.add_argument("--topk-lines", type=int, default=4)
    args = parser.parse_args(argv)

    status = run_monte_carlo(args.trials)
    if args.real is None:
        print("\n(part B skipped; pass --real pot to run it against stored masks)")
        return status
    run_real(args.real, args.limit, args.noise_px, args.topk_lines)
    return status


if __name__ == "__main__":
    sys.exit(main())
