"""Naming a planar pose from keypoint correspondences.

A silhouette determines a planar pose only up to which corner is called first:
the four cyclic relabelings of a quadrilateral describe the same outline, score
the same raster overlap, and are indistinguishable to anything that looks only
at shape. Measured on PlanarTrack, 44.6% of the mask registrar's wrong
hypotheses are exactly that permutation, and a wrong one scored IoU 0.94 while
being 219 px out.

Texture breaks the tie, and the keypoint matcher has already computed the
texture evidence: a set of correspondences between the template image and the
current frame. This module turns them into a scalar that says *how much of that
evidence a hypothesis explains*, which is the same question for a pose and for
a naming:

    U(H) = sum_i max(0, 1 - d_i^2 / tau^2),   d_i = || H p_i - q_i ||

``U`` counts, in units of *effective matches*, how many correspondences the
hypothesis puts where they belong. Three properties matter:

* **Bounded.** A correspondence that is simply wrong (a mismatch) contributes
  zero under every hypothesis, so it cancels exactly when hypotheses are
  compared. An unbounded loss (L1, Huber) lets one gross outlier move the
  comparison, and mismatch rates are highest on the blurred, low-texture frames
  where naming is hardest — precisely where the score must not wobble.
* **Continuous.** ``min(d^2, tau^2)`` approaches its cap smoothly, so unlike a
  hard inlier count there is no cliff at ``tau`` and the result is insensitive
  to its exact value.
* **Comparable across hypotheses and scale-free after normalization.** ``U/N``
  is the fraction of available evidence explained, in [0, 1], which is directly
  multiplicable with a raster IoU.

The correspondences are used **before** RANSAC, deliberately. RANSAC inliers
were selected under a homography that already carries one of the four namings,
so scoring namings against them is circular; raw matches are naming-independent
evidence.

Used in two places, with the same scalar: the quad-free registrar multiplies
``U/N`` into its verification score (so the alias with the best correspondence
support wins, IoU being tied across aliases by construction), and the
quad-based tracker uses :func:`name_by_match_support` as a standalone namer in
place of the DINO appearance vote.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# Bin edges (px) for the per-hypothesis residual histogram written to a run's
# debug telemetry. Chosen so that every threshold we might want to re-test
# offline is an edge: with the per-bin sum of squared residuals stored beside
# the counts, `sum min(d^2, tau^2)` is then exact for tau in {10, 15, 20, 30}
# rather than approximated at bin resolution.
RESIDUAL_BIN_EDGES: Tuple[float, ...] = (
    0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, np.inf
)
N_RESIDUAL_BINS = len(RESIDUAL_BIN_EDGES) - 1

# A homography has 8 degrees of freedom; below that many correspondences the
# notion "this naming explains the evidence" has no content, so the incumbent
# naming stands. Not a tuned threshold — a structural precondition.
MIN_MATCHES = 8


def homography_from_corners(
    init_corners: np.ndarray, corners: np.ndarray
) -> Optional[np.ndarray]:
    """Exact 4-point homography, (2, 4) -> (2, 4), or None when degenerate.

    Args:
        init_corners: (2, 4) template corners, ``[[x...], [y...]]``.
        corners: (2, 4) corresponding corners in the current frame.

    Returns:
        (3, 3) homography, or None.
    """
    try:
        H = cv2.getPerspectiveTransform(
            init_corners.T.astype(np.float32), corners.T.astype(np.float32)
        )
    except cv2.error:
        return None
    return H.astype(np.float64) if np.isfinite(H).all() else None


def match_residuals(
    H: np.ndarray, pts_src: np.ndarray, pts_dst: np.ndarray
) -> Optional[np.ndarray]:
    """Reprojection distance of every correspondence under ``H``.

    Args:
        H: (3, 3) homography mapping template-frame points to current-frame ones.
        pts_src: (N, 2) template-frame match points.
        pts_dst: (N, 2) current-frame match points, row-aligned with ``pts_src``.

    Returns:
        (N,) distances in pixels, or None when the warp is degenerate.
    """
    if len(pts_src) == 0 or len(pts_src) != len(pts_dst):
        return None
    proj = H @ np.hstack([pts_src, np.ones((len(pts_src), 1))]).T
    w = proj[2]
    if not np.all(np.isfinite(w)) or np.any(np.abs(w) < 1e-9):
        return None
    warped = (proj[:2] / w).T
    return np.linalg.norm(warped - pts_dst, axis=1)


def support_from_residuals(residuals: np.ndarray, tau_px: float) -> float:
    """Effective number of correspondences explained, ``sum max(0, 1-d^2/tau^2)``."""
    if tau_px <= 0.0:
        return 0.0
    return float(np.maximum(0.0, 1.0 - (residuals / tau_px) ** 2).sum())


def match_support(
    H: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    tau_px: float,
) -> Optional[float]:
    """Effective matches explained by ``H``; None when it cannot be measured."""
    residuals = match_residuals(H, pts_src, pts_dst)
    if residuals is None:
        return None
    return support_from_residuals(residuals, tau_px)


def residual_histogram(residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Counts and per-bin sums of squared residuals over :data:`RESIDUAL_BIN_EDGES`.

    Storing the squared sums beside the counts is what makes an offline re-test
    of ``tau`` exact rather than bin-resolution: the truncated-quadratic cost is
    recoverable in closed form for any ``tau`` that is a bin edge.

    Args:
        residuals: (N,) reprojection distances (px).

    Returns:
        Tuple ``(counts (B,) int32, sum_sq (B,) float32)``.
    """
    edges = np.asarray(RESIDUAL_BIN_EDGES)
    idx = np.clip(np.searchsorted(edges, residuals, side="right") - 1, 0, N_RESIDUAL_BINS - 1)
    counts = np.bincount(idx, minlength=N_RESIDUAL_BINS).astype(np.int32)
    sum_sq = np.bincount(
        idx, weights=residuals.astype(np.float64) ** 2, minlength=N_RESIDUAL_BINS
    ).astype(np.float32)
    return counts, sum_sq


def alias_supports(
    corners: np.ndarray,
    init_corners: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    tau_px: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Correspondence support of all four cyclic namings of one pose.

    Args:
        corners: (2, 4) pose whose namings are in question.
        init_corners: (2, 4) template corners.
        pts_src: (N, 2) template-frame match points.
        pts_dst: (N, 2) current-frame match points.
        tau_px: Truncation distance of the support (px).

    Returns:
        Tuple ``(support (4,), counts (4, B) or None, sum_sq (4, B) or None)``.
        Support is NaN for a naming whose homography is degenerate. The
        histograms are telemetry only and are None when nothing was measurable.
    """
    support = np.full(4, np.nan)
    counts = np.zeros((4, N_RESIDUAL_BINS), dtype=np.int32)
    sum_sq = np.zeros((4, N_RESIDUAL_BINS), dtype=np.float32)
    measured = False
    for r in range(4):
        H = homography_from_corners(init_corners, np.roll(corners, r, axis=1))
        if H is None:
            continue
        residuals = match_residuals(H, pts_src, pts_dst)
        if residuals is None:
            continue
        support[r] = support_from_residuals(residuals, tau_px)
        counts[r], sum_sq[r] = residual_histogram(residuals)
        measured = True
    if not measured:
        return support, None, None
    return support, counts, sum_sq


def name_by_match_support(
    corners: np.ndarray,
    init_corners: np.ndarray,
    pts_src: Optional[np.ndarray],
    pts_dst: Optional[np.ndarray],
    tau_px: float,
    min_matches: int = MIN_MATCHES,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Re-name a pose's corners to the naming its correspondences support.

    The whole rule: with enough correspondences to mean anything, take the
    cyclic naming that explains most of them. No margin over the incumbent and
    no abstain threshold — weak evidence makes the four supports similar *and*
    small rather than misleading, and whether a guard would have helped is
    answerable offline from the histograms this module also produces.

    Args:
        corners: (2, 4) pose to name.
        init_corners: (2, 4) template corners.
        pts_src: (N, 2) template-frame match points, or None.
        pts_dst: (N, 2) current-frame match points, or None.
        tau_px: Truncation distance of the support (px).
        min_matches: Structural minimum; below it the incumbent naming stands.

    Returns:
        Tuple ``(corners (2, 4), shift, support (4,))``. ``shift`` is the applied
        cyclic shift: ``-1`` when there was no evidence to look at, ``0`` when
        the four namings were scored and the incumbent kept, ``1``-``3`` when it
        was overturned.

        The ``-1`` matters. Returning ``0`` for "no matches" made abstention
        indistinguishable from a considered decision to keep the naming, so a
        run could look as though the namer had examined every frame when it had
        in fact skipped many of them (review, 2026-08-19).
    """
    support = np.full(4, np.nan)
    if pts_src is None or pts_dst is None or len(pts_src) < min_matches:
        return corners, -1, support
    support, _, _ = alias_supports(corners, init_corners, pts_src, pts_dst, tau_px)
    if not np.any(np.isfinite(support)):
        return corners, -1, support
    best = int(np.nanargmax(support))
    if best == 0:
        return corners, 0, support
    return np.roll(corners, best, axis=1), best, support
