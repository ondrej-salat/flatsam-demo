"""Signed mask-discrepancy metrics: area depth and outline coverage.

General ``reference`` vs ``other`` mask comparison — nothing here is SAM/GT/LoMa
specific. Both metrics are measured against the *reference* boundary:

- :func:`signed_area_discrepancy` — fraction of the reference *area* where
  ``other`` sticks out past the reference (> T outside) or leaves it uncovered
  (< −T inside), via the reference signed distance field.
- :func:`outline_normal_coverage` — fraction of the reference *outline* where
  ``other`` overshoots outward / falls short inward by > T, probed along the
  outline normal (directional, so a wide overshoot or a spill through a narrow
  neck is read by its outward extent, not capped by a nearby lateral boundary).

All thresholds are caller-supplied. Each metric optionally returns a debug record
(``debug=True``) exposing exactly which pixels / probe points were counted, for
visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


# Default window (in contour points) for the smoothed tangent used for normals.
DEFAULT_SMOOTH_WIN = 3


def signed_distance(region: np.ndarray, precise: bool = True) -> np.ndarray:
    """Signed Euclidean distance field: + outside the region, − inside, 0 on it.

    Both halves matter to :func:`signed_area_discrepancy`: the outside half for
    the other mask's excess past the region, the inside half for the region's
    interior left uncovered. True Euclidean (``DIST_MASK_PRECISE``) by default —
    that exactness is what :func:`signed_area_discrepancy` needs.

    Args:
        region: Bool region mask (H, W).
        precise: Exact Euclidean transform. False uses the 3x3 chamfer
            approximation, which is several times cheaper and accurate to a few
            percent — enough for a registration objective, not for the area
            discrepancy above.

    Returns:
        Float32 signed distance map, same shape.
    """
    mask_size = cv2.DIST_MASK_PRECISE if precise else cv2.DIST_MASK_3
    outside = cv2.distanceTransform(
        np.where(region, 0, 255).astype(np.uint8), cv2.DIST_L2, mask_size
    )
    inside = cv2.distanceTransform(
        np.where(region, 255, 0).astype(np.uint8), cv2.DIST_L2, mask_size
    )
    return outside - inside


@dataclass
class AreaDiscrepancyDebug:
    """Counted pixels behind :func:`signed_area_discrepancy`.

    Attributes:
        reference_signed_distance: The reference signed field (+ out / − in).
        other_mask: The other mask compared against the reference.
        outside_counted: ``T -> bool mask`` of ``other & (sdf > T)`` (excess).
        inside_counted: ``T -> bool mask`` of ``~other & (sdf < −T)`` (deficit).
    """

    reference_signed_distance: np.ndarray
    other_mask: np.ndarray
    outside_counted: dict[int, np.ndarray]
    inside_counted: dict[int, np.ndarray]


def signed_area_discrepancy(
    reference_signed_distance: np.ndarray,
    other_mask: np.ndarray,
    reference_area: int,
    thresholds: Sequence[int],
    *,
    debug: bool = False,
):
    """Signed area discrepancy of ``other`` vs a reference region.

    Measured against the reference boundary (its signed field), as a fraction of
    the reference area:

    - **outside** (excess): ``other`` pixels with ``sdf_ref > T`` — ``other`` area
      more than T px *past* the reference boundary.
    - **inside** (deficit): non-``other`` pixels with ``sdf_ref < −T`` — reference
      interior left uncovered more than T px *inside* the boundary. Background
      pixels (``sdf_ref > 0``) are never counted.

    Outside may exceed 1.0 when ``other`` is much larger than the reference;
    inside is at most 1.0.

    Args:
        reference_signed_distance: Reference signed field from
            :func:`signed_distance` (+ outside / − inside).
        other_mask: Bool mask compared against the reference (same shape).
        reference_area: Reference area in pixels (the denominator).
        thresholds: Pixel thresholds T to evaluate.
        debug: When True, also return an :class:`AreaDiscrepancyDebug`.

    Returns:
        ``(outside, inside)`` dicts keyed by T, or ``(outside, inside, debug)``
        when ``debug`` (debug is ``None`` if ``reference_area`` is 0). Values are
        NaN when ``reference_area`` is 0.
    """
    if reference_area <= 0:
        nan = {int(t): float("nan") for t in thresholds}
        return (dict(nan), dict(nan), None) if debug else (dict(nan), dict(nan))

    not_other = ~other_mask
    outside: dict[int, float] = {}
    inside: dict[int, float] = {}
    outside_counted: dict[int, np.ndarray] = {}
    inside_counted: dict[int, np.ndarray] = {}
    for t in thresholds:
        t = int(t)
        excess = other_mask & (reference_signed_distance > float(t))
        deficit = not_other & (reference_signed_distance < -float(t))
        outside[t] = float(excess.sum()) / reference_area
        inside[t] = float(deficit.sum()) / reference_area
        if debug:
            outside_counted[t] = excess
            inside_counted[t] = deficit
    if debug:
        record = AreaDiscrepancyDebug(
            reference_signed_distance, other_mask, outside_counted, inside_counted
        )
        return outside, inside, record
    return outside, inside


def cropped_signed_area_discrepancy(
    reference: np.ndarray,
    other: np.ndarray,
    thresholds: Sequence[int],
):
    """:func:`signed_area_discrepancy` with the distance transform cropped.

    The metric only ever counts pixels within ``max(thresholds)`` px of the
    reference boundary, so the full-frame signed distance is wasted work. This
    crops both masks to the union bounding box padded by ``max(thresholds) + 2``
    before computing the signed field — an object-sized window instead of the
    whole frame. The pad guarantees every pixel that could be counted at any
    threshold survives the crop, and the reference area (the denominator) is
    taken from the *full* reference, so the result is bit-identical to running
    on the full frame. The crop is an invisible optimization.

    Args:
        reference: Bool reference region mask (H, W).
        other: Bool mask compared against the reference (same shape).
        thresholds: Pixel thresholds T to evaluate.

    Returns:
        ``(outside, inside)`` dicts keyed by int T, identical to
        ``signed_area_discrepancy(signed_distance(reference), other,
        int(reference.sum()), thresholds)``. NaN dicts when the reference is empty.
    """
    reference = reference > 0
    other = other > 0
    reference_area = int(reference.sum())
    if reference_area == 0:
        nan = {int(t): float("nan") for t in thresholds}
        return dict(nan), dict(nan)
    union = reference | other
    ys = np.where(union.any(axis=1))[0]
    xs = np.where(union.any(axis=0))[0]
    pad = int(max(thresholds)) + 2
    h, w = reference.shape
    y0, y1 = max(0, ys[0] - pad), min(h, ys[-1] + pad + 1)
    x0, x1 = max(0, xs[0] - pad), min(w, xs[-1] + pad + 1)
    sdf = signed_distance(reference[y0:y1, x0:x1])
    return signed_area_discrepancy(sdf, other[y0:y1, x0:x1], reference_area, thresholds)


def outline_points_and_normals(
    region: np.ndarray, smooth_win: int = DEFAULT_SMOOTH_WIN
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Outline points and smoothed outward unit normals of a region.

    Uses ``findContours`` over *all* external contours, so a multi-part region
    keeps every piece's outline. Per contour the normal is the tangent over a
    ``±smooth_win`` window rotated 90° (the windowed tangent smooths pixel
    staircasing into continuous, low-jitter normals), oriented outward by a single
    per-contour majority vote — a contour's orientation is constant, so if most
    candidate normals step inside the region the whole contour is flipped (robust,
    and free of OpenCV outer-vs-hole / y-down sign conventions).

    Args:
        region: Bool region mask (H, W).
        smooth_win: Half-window (in contour points) for the tangent.

    Returns:
        ``(points, normals)`` float arrays of shape ``(P, 2)`` in ``(x, y)``
        order, concatenated across contours, or ``None`` for a degenerate region.
    """
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    h, w = region.shape
    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)  # (P, 2) x,y, closed
        if len(points) < 2 * smooth_win + 2:
            continue
        tangent = np.roll(points, -smooth_win, axis=0) - np.roll(points, smooth_win, axis=0)
        length = np.hypot(tangent[:, 0], tangent[:, 1])
        keep = length > 1e-6
        points, tangent, length = points[keep], tangent[keep], length[keep]
        if len(points) == 0:
            continue
        normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / length[:, None]
        # Per-contour outward orientation by majority vote.
        probe_x = np.clip(np.rint(points[:, 0] + 2 * normals[:, 0]).astype(int), 0, w - 1)
        probe_y = np.clip(np.rint(points[:, 1] + 2 * normals[:, 1]).astype(int), 0, h - 1)
        if region[probe_y, probe_x].mean() > 0.5:
            normals = -normals
        all_points.append(points)
        all_normals.append(normals)
    if not all_points:
        return None
    return np.concatenate(all_points, axis=0), np.concatenate(all_normals, axis=0)


@dataclass
class CoverageDebug:
    """Probe points behind :func:`outline_normal_coverage`.

    Attributes:
        points: ``(P, 2)`` reference-outline points in ``(x, y)``.
        normals: ``(P, 2)`` outward unit normals at those points.
        overshoot_probe: ``T -> (P, 2)`` int outward probe coords ``P + T·n``.
        overshoot_hit: ``T -> (P,)`` bool, ``other`` present at the outward probe.
        shortfall_probe: ``T -> (P, 2)`` int inward probe coords ``P − T·n``.
        shortfall_hit: ``T -> (P,)`` bool, ``other`` *absent* at the inward probe.
    """

    points: np.ndarray
    normals: np.ndarray
    overshoot_probe: dict[int, np.ndarray]
    overshoot_hit: dict[int, np.ndarray]
    shortfall_probe: dict[int, np.ndarray]
    shortfall_hit: dict[int, np.ndarray]


def outline_normal_coverage(
    reference_mask: np.ndarray,
    other_mask: np.ndarray,
    thresholds: Sequence[int],
    *,
    smooth_win: int = DEFAULT_SMOOTH_WIN,
    debug: bool = False,
):
    """Signed coverage of a reference outline by ``other``, along its normal.

    For each reference-outline point ``P`` with outward unit normal ``n``:

    - **overshoot** (excess): ``other[P + T·n]`` — ``other`` present T px outward.
    - **shortfall** (deficit): ``not other[P − T·n]`` — ``other`` absent T px inward.

    Probing along the normal reads the outward/inward extent directly (a wide
    overshoot or a narrow-neck spill is not capped by a nearby lateral boundary),
    and counting reference-outline points keeps it robust to thin lateral spills.
    Denominator is the number of reference-outline points.

    Args:
        reference_mask: Bool reference region whose outline is walked.
        other_mask: Bool mask probed along the outline normals (same shape).
        thresholds: Pixel thresholds T to evaluate.
        smooth_win: Half-window (in contour points) for the tangent.
        debug: When True, also return a :class:`CoverageDebug`.

    Returns:
        ``(overshoot, shortfall)`` dicts keyed by T, or
        ``(overshoot, shortfall, debug)`` when ``debug`` (debug is ``None`` if the
        reference has no usable contour). Values are NaN when no usable contour.
    """
    pts_normals = outline_points_and_normals(reference_mask, smooth_win)
    if pts_normals is None:
        nan = {int(t): float("nan") for t in thresholds}
        return (dict(nan), dict(nan), None) if debug else (dict(nan), dict(nan))

    points, normals = pts_normals
    xs, ys = points[:, 0], points[:, 1]
    nx, ny = normals[:, 0], normals[:, 1]
    h, w = other_mask.shape
    overshoot: dict[int, float] = {}
    shortfall: dict[int, float] = {}
    overshoot_probe: dict[int, np.ndarray] = {}
    overshoot_hit: dict[int, np.ndarray] = {}
    shortfall_probe: dict[int, np.ndarray] = {}
    shortfall_hit: dict[int, np.ndarray] = {}
    for t in thresholds:
        t = int(t)
        ox = np.clip(np.rint(xs + t * nx).astype(int), 0, w - 1)
        oy = np.clip(np.rint(ys + t * ny).astype(int), 0, h - 1)
        hit_out = other_mask[oy, ox]
        overshoot[t] = float(hit_out.mean())
        ix = np.clip(np.rint(xs - t * nx).astype(int), 0, w - 1)
        iy = np.clip(np.rint(ys - t * ny).astype(int), 0, h - 1)
        miss_in = ~other_mask[iy, ix]
        shortfall[t] = float(miss_in.mean())
        if debug:
            overshoot_probe[t] = np.stack([ox, oy], axis=1)
            overshoot_hit[t] = hit_out
            shortfall_probe[t] = np.stack([ix, iy], axis=1)
            shortfall_hit[t] = miss_in
    if debug:
        record = CoverageDebug(
            points, normals, overshoot_probe, overshoot_hit, shortfall_probe, shortfall_hit
        )
        return overshoot, shortfall, record
    return overshoot, shortfall
