#!/usr/bin/env python3
"""Estimate the mask-branch pose from lines, and show what changes.

Runs on the frozen cases of ``experiments/maskh_cases.py``: no tracking, no
segmentation, no GPU. For each case it replays the Hough stage exactly as the
tracker runs it, then compares two ways of turning the result into a pose:

* **corners** -- the deployed path. Intersect consecutive line pairs, keep the
  intersections near the contour, and match them to template corners; four or
  more give a homography, two or three a similarity on the incumbent, one a
  translation. Truncation removes intersections, so this is where the pose is
  lost.
* **lines** -- :mod:`flatsam2.line_homography`. Assign detected lines to template
  sides, fit the pose from the lines and any surviving corner correspondences,
  and take the directions the data does not constrain from the incumbent.

Both are scored against ground truth, and against the mask the pose should
explain. ``--figures`` draws every intermediate quantity: raw Hough lines with
their supported segments, intersections before and after the near-contour
filter, which lines were matched to which template side, and the two poses.

Imports cv2 -- submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/maskh_line_fit.py
    python slurm/submit_cmd.py --gpus 0 -- python experiments/maskh_line_fit.py \\
        --figures --cases V22_5_f00171 V29_3_f00251
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from einops import rearrange

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2 import hough_homography as hh
from flatsam2.datasets import warp_corners
from flatsam2.line_homography import fit_homography, line_through
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from maskh_cases import cases, load

# Deployed Hough settings (configs/tracker/loma_samh_matchnaming.py).
_GRADIENT_STEP = 4
_TOPK_LINES = 4
_CORNER_MAX_DISTANCE_PX = 80.0
# A pose that does not contain the mask it was fitted to is wrong whatever its
# residual says; truncation makes the converse (mask contains pose) common.
_MIN_MASK_COVERAGE = 0.90
# ... but "contains the mask" is also satisfied by an arbitrarily large quad, so
# bound the other side. Under truncation the true pose is legitimately bigger
# than the visible silhouette; four times its area is well past that.
_MAX_AREA_RATIO = 4.0
# How strongly a corner with no hard evidence is pulled toward the incumbent's
# own prediction for it. Low: the incumbent is data about a *different* frame,
# not a measurement of this one, and the surviving lines should dominate.
_SOFT_PRIOR_WEIGHT = 0.08
# A thin, extreme quad can still be convex and cover most of the mask (a wedge
# narrow far away, wide where the mask is) -- observed on real cases here, not
# hypothetical. Reject any corner this far outside the frame in image-diagonal
# units; a real, even occluded, corner is not usually many frames away.
_MAX_CORNER_MARGIN_DIAGONALS = 0.5


@dataclass
class Stage:
    """What the Hough stage produced on one mask."""

    contours: List[np.ndarray]
    lines: np.ndarray                      # (3, K) refined, filtered lines
    segments: List[np.ndarray]             # (2, 2) endpoints of each line's support
    intersections: np.ndarray              # (2, M) all consecutive-pair intersections
    kept: np.ndarray                       # (M,) bool, survived the contour filter
    parents: List[Tuple[int, int]]         # line indices behind each intersection


def run_hough(mask: np.ndarray) -> Stage:
    """Replay the deployed Hough stage on one mask.

    Args:
        mask: (H, W) bool SAM3 mask.

    Returns:
        The stage's intermediate quantities.
    """
    contours_cv, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        rearrange(c, "N 1 xy -> xy N", xy=2).astype(float)
        for c in contours_cv if c.shape[0] > 20
    ]
    empty = Stage([], np.zeros((3, 0)), [], np.zeros((2, 0)), np.zeros(0, bool), [])
    if not contours:
        return empty
    on_boundaries = [
        np.logical_or.reduce(tuple(
            hh._approx_touching_edges(c + shift, mask.shape, margin=1)
            for shift in (np.array([[0], [1]]), np.array([[0], [-1]]),
                          np.array([[1], [0]]), np.array([[-1], [0]]))
        ))
        for c in contours
    ]
    raw = hh.contour_hough(
        mask.astype(np.uint8), contours, on_boundaries,
        gradient_step=_GRADIENT_STEP, topk=_TOPK_LINES,
    )
    refined, line_points = hh.refine_lines(
        raw, contours, on_boundaries, gradient_step=_GRADIENT_STEP,
        use_orthogonal_fit=False,
    )
    supported = np.array([pts.shape[1] > 0 for pts in line_points])
    refined = refined[:, supported]
    line_points = [pts for pts, ok in zip(line_points, supported) if ok]
    segments = [hh._line_to_segment(refined[:, i], line_points[i])
                for i in range(refined.shape[1])]
    long_enough = np.array([hh._segment_length(s) >= 5 for s in segments], dtype=bool)
    refined = refined[:, long_enough]
    segments = [s for s, ok in zip(segments, long_enough) if ok]
    if refined.shape[1] < 2:
        return Stage(contours, refined, segments, np.zeros((2, 0)), np.zeros(0, bool), [])
    intersections_h, parents = hh.lines_to_points(refined)
    intersections = hh.p2e(intersections_h)
    _, kept = hh.keep_only_points_close_to_contours(
        intersections, contours, max_distance_px=_CORNER_MAX_DISTANCE_PX,
        parents=parents, lines=refined, min_angle_deg=0.0,
    )
    return Stage(contours, refined, segments, intersections, kept, parents)


def deployed_pose(
    mask: np.ndarray,
    init_corners: np.ndarray,
    incumbent_H: np.ndarray,
) -> Tuple[np.ndarray, int, str]:
    """The pose the shipped mask branch produces, and how it got there.

    Args:
        mask: (H, W) bool mask.
        init_corners: (2, 4) template corners.
        incumbent_H: (3, 3) branch's previous homography.

    Returns:
        ``(homography, corners_used, fit)``.
    """
    stage = run_hough(mask)
    if stage.lines.shape[1] < 2:
        return incumbent_H.copy(), 0, "incumbent"
    valid = stage.intersections[:, stage.kept]
    valid_parents = [p for p, ok in zip(stage.parents, stage.kept) if ok]
    matches = hh.discard_diagonal_matches(
        hh.bruteforce_pointset_matching(init_corners, valid),
        valid_parents, init_corners.shape[1],
    )
    homography, _, best_ids = hh.find_closest_H(init_corners, valid, incumbent_H, matches)
    n_used = len(best_ids[0])
    fit = ("DLT" if n_used >= 4 else "similarity" if n_used > 1
           else "translation" if n_used == 1 else "incumbent")
    return homography, n_used, fit


def template_sides(init_corners: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The four template side lines and their endpoint segments.

    Args:
        init_corners: (2, 4) template corners, in order around the quad.

    Returns:
        ``(lines, segments)``; side ``j`` joins corner ``j`` to ``j + 1``.
    """
    lines, segments = [], []
    for j in range(4):
        a, b = init_corners[:, j], init_corners[:, (j + 1) % 4]
        lines.append(line_through(a, b))
        segments.append(np.stack([a, b], axis=1))
    return lines, segments


def candidate_assignments(n_lines: int) -> List[Dict[int, int]]:
    """Every injective map from detected line index to template side index.

    Deliberately not seeded from the incumbent. On exactly the frames this is
    meant to fix, the incumbent is already hundreds of pixels wrong, so a gate
    built from it rejects the correct assignment; the mask verifies each fit
    afterwards and does not need the prior to be any good.

    Two lines are not enough: they pin at most one corner, leaving three to a
    weak soft pull with nothing else to hold them, which is exactly the
    configuration that let the geometric refinement run away to a degenerate
    quad in testing. Three lines are the case this module exists for -- two
    real corners and a soft pull along the two known lines for the rest -- so
    that is the floor.

    Args:
        n_lines: Number of detected lines.

    Returns:
        Assignments of at least three lines, largest first.
    """
    assignments: List[Dict[int, int]] = []
    for size in range(min(n_lines, 4), 2, -1):
        for chosen in itertools.combinations(range(n_lines), size):
            for sides in itertools.permutations(range(4), size):
                assignments.append(dict(zip(chosen, sides)))
    return assignments


def corner_correspondences(
    stage: Stage,
    assignment: Dict[int, int],
    init_corners: np.ndarray,
    max_distance_px: float = _CORNER_MAX_DISTANCE_PX,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Point correspondences an assignment implies, with no prior involved.

    Template side ``j`` runs from corner ``j`` to corner ``j + 1``, so whenever
    an assignment places lines on both sides meeting at a corner, those two
    lines' intersection names that corner -- computed directly from the
    assigned pair, not from :func:`hough_homography.lines_to_points`, which
    only intersects lines that are cyclically adjacent in *detection* order and
    therefore misses this pair whenever the correct assignment reorders them
    (exactly the case this module exists for). The near-contour filter is
    reapplied here for the same reason: a spurious intersection far from the
    mask is not evidence for a corner.

    Args:
        stage: Replayed Hough stage.
        assignment: Detected line index to template side index.
        init_corners: (2, 4) template corners.
        max_distance_px: Reject an intersection farther than this from every
            contour point.

    Returns:
        ``(template_corner (2,), detected_intersection (2,))`` pairs.
    """
    side_to_line = {side: line for line, side in assignment.items()}
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for corner in range(4):
        first, second = side_to_line.get((corner - 1) % 4), side_to_line.get(corner)
        if first is None or second is None or first == second:
            continue
        point_h = hh.line_line_intersection(stage.lines[:, first], stage.lines[:, second])
        point = hh.p2e(point_h)
        if not np.all(np.isfinite(point)):
            continue
        _, near = hh.keep_only_points_close_to_contours(
            point, stage.contours, max_distance_px=max_distance_px
        )
        if near[0]:
            pairs.append((init_corners[:, corner], point[:, 0]))
    return pairs


def incumbent_soft_points(
    init_corners: np.ndarray,
    incumbent_H: np.ndarray,
    hard_pairs: List[Tuple[np.ndarray, np.ndarray]],
    weight: float = _SOFT_PRIOR_WEIGHT,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Weak pulls toward the incumbent, one per corner with no hard evidence.

    Only the corners :func:`corner_correspondences` did not pin need this: each
    such corner still sits exactly on whichever one of its two sides *is*
    assigned (the line constraint already forces that), so the soft point only
    has to settle the position along that line, and a small weight is enough
    for it to do that without overriding a real line.

    Args:
        init_corners: (2, 4) template corners.
        incumbent_H: (3, 3) branch's previous homography.
        hard_pairs: Output of :func:`corner_correspondences`, used only to see
            which template corners it already covered.
        weight: Soft-point weight, forwarded to :func:`fit_homography`.

    Returns:
        ``(template_corner, incumbent_prediction, weight)`` triples.
    """
    hard_corner_ids = {
        corner for corner in range(4)
        for template_point, _ in hard_pairs
        if np.array_equal(template_point, init_corners[:, corner])
    }
    predicted = warp_corners(incumbent_H, init_corners)
    return [
        (init_corners[:, corner], predicted[:, corner], weight)
        for corner in range(4) if corner not in hard_corner_ids
    ]


def line_pose(
    case: Dict,
    stage: Stage,
    top_k: int = 4,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Fit the pose from assigned lines plus the corners those lines imply.

    Every injective assignment is fitted cheaply without refinement and scored
    against the mask; only the best few are refined, since the geometric
    refinement costs far more than the linear solve and the ranking is already
    stable without it.

    Verification is prior-free: the pose must contain the silhouette it was
    fitted to, and is ranked by how much of the visible boundary its outline
    explains. Ground truth is never consulted.

    Args:
        case: Frozen case dict.
        stage: Replayed Hough stage.
        top_k: How many candidates to refine geometrically.

    Returns:
        ``(homography, info)``; homography is None when nothing verified.
    """
    init_corners = case["init_corners"]
    incumbent_H = case["incumbent_H"]
    mask = case["mask"]
    sides, segments = template_sides(init_corners)
    info: Dict = {"assignments": 0, "rejected": 0}
    scored: List[Tuple[Tuple[float, float], Dict]] = []
    for assignment in candidate_assignments(stage.lines.shape[1]):
        point_pairs = corner_correspondences(stage, assignment, init_corners)
        candidate = {
            "assignment": dict(assignment),
            "line_pairs": [(sides[j], stage.lines[:, i]) for i, j in assignment.items()],
            "segments": [segments[j] for _, j in assignment.items()],
            "point_pairs": point_pairs,
            "soft_points": incumbent_soft_points(init_corners, incumbent_H, point_pairs),
        }
        linear_H, linear_report = fit_homography(
            candidate["line_pairs"], candidate["point_pairs"], candidate["soft_points"],
            template_segments=candidate["segments"], refine=False,
        )
        info["assignments"] += 1
        verdict = _verify(linear_H, init_corners, mask)
        if verdict is None:
            info["rejected"] += 1
            continue
        explained, _, _ = verdict
        candidate["linear_H"] = linear_H
        candidate["linear_report"] = linear_report
        scored.append((_rank_key(explained, linear_H, case), candidate))
    if not scored:
        info["reason"] = "no assignment verified"
        return None, info

    scored.sort(key=lambda item: item[0])
    best: Optional[Tuple[Tuple[float, float], np.ndarray, Dict]] = None
    for _, candidate in scored[:top_k]:
        # The geometric refinement is a numerical Gauss-Newton step and can, on
        # a near-degenerate configuration, land on a self-intersecting or
        # flipped quad even though it started from a verified one -- observed
        # on real cases here, not hypothetical. Fall back to the linear result
        # whenever that happens rather than losing a candidate that already
        # verified; never let refinement make a candidate worse than not
        # refining it at all.
        refined_H, refined_report = fit_homography(
            candidate["line_pairs"], candidate["point_pairs"], candidate["soft_points"],
            template_segments=candidate["segments"], refine=True,
        )
        refined_verdict = _verify(refined_H, init_corners, mask)
        linear_verdict = _verify(candidate["linear_H"], init_corners, mask)
        refined_key = (
            _rank_key(refined_verdict[0], refined_H, case)
            if refined_verdict is not None else None
        )
        linear_key = (
            _rank_key(linear_verdict[0], candidate["linear_H"], case)
            if linear_verdict is not None else None
        )
        if refined_key is not None and (linear_key is None or refined_key <= linear_key):
            homography, report, verdict, key = refined_H, refined_report, refined_verdict, refined_key
        elif linear_key is not None:
            homography, report = candidate["linear_H"], candidate["linear_report"]
            report.refined = False
            verdict, key = linear_verdict, linear_key
        else:
            continue
        explained, median_px, coverage = verdict
        if best is None or key < best[0]:
            best = (key, homography, {
                "assignment": candidate["assignment"],
                "rank": report.rank,
                "prior_dims": report.prior_dims,
                "residual_px": report.residual_px,
                "refined": report.refined,
                "coverage": coverage,
                "explained": explained,
                "median_px": median_px,
                "n_lines": report.n_lines,
                "n_points": report.n_points,
            })
    if best is None:
        info["reason"] = "refinement broke every candidate"
        return None, info
    info.update(best[2])
    return best[1], info


def _rank_key(
    explained: float,
    homography: np.ndarray,
    case: Dict,
) -> Tuple[float, float]:
    """Rank candidates: silhouette agreement first, then the incumbent.

    The mask cannot separate the four cyclic namings of a quadrilateral -- they
    explain the same boundary exactly -- so agreement is rounded to the level at
    which it is meaningful and the incumbent breaks the remaining tie, which is
    what the deployed corner path does with ``find_closest_H``. Corner naming is
    settled downstream by correspondence support, not here.

    Args:
        explained: Fraction of visible boundary the pose explains.
        homography: (3, 3) candidate pose.
        case: Frozen case dict.

    Returns:
        Sort key, smaller is better.
    """
    corners = warp_corners(homography, case["init_corners"])
    predicted = warp_corners(case["incumbent_H"], case["init_corners"])
    displacement = float(np.mean(np.linalg.norm(corners - predicted, axis=0)))
    return (-round(explained, 2), displacement)


def _verify(
    homography: Optional[np.ndarray],
    init_corners: np.ndarray,
    mask: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Mask-only acceptance test for one candidate pose.

    Args:
        homography: (3, 3) candidate, or None.
        init_corners: (2, 4) template corners.
        mask: (H, W) bool mask.

    Returns:
        ``(explained, median_px, coverage)``, or None when the pose is rejected.
    """
    if homography is None or not hh.good_homography(homography, init_corners):
        return None
    corners = warp_corners(homography, init_corners)
    height, width = mask.shape
    diagonal = float(np.hypot(width, height))
    margin = _MAX_CORNER_MARGIN_DIAGONALS * diagonal
    if (corners[0].min() < -margin or corners[0].max() > width + margin
            or corners[1].min() < -margin or corners[1].max() > height + margin):
        return None
    coverage = mask_coverage(corners, mask)
    if coverage < _MIN_MASK_COVERAGE:
        return None
    area = 0.5 * abs(float(np.dot(corners[0], np.roll(corners[1], -1))
                           - np.dot(corners[1], np.roll(corners[0], -1))))
    if area > _MAX_AREA_RATIO * max(float(mask.sum()), 1.0):
        return None
    explained, median_px = boundary_fit(corners, mask)
    return explained, median_px, coverage


def boundary_fit(
    corners: np.ndarray,
    mask: np.ndarray,
    tolerance_px: float = 3.0,
) -> Tuple[float, float]:
    """How well a pose's outline explains the mask's own boundary.

    Ground truth cannot score a frame whose target is partly unobservable: when
    a whole side is behind an occluder or outside the image, no estimator can
    place it and the ground-truth error is dominated by a corner nobody can see.
    What is still checkable is the visible silhouette -- every mask-boundary
    point that is a real object edge should lie on the predicted quadrilateral.

    Contour points touching the image border are excluded: they are where the
    frame cut the object, not where the object ends. Points on an occluder's
    edge cannot be identified and stay in, which is why the fraction within a
    tolerance is reported next to the median: a pose that explains the visible
    sides scores high on the fraction even when the occluded stretch drags the
    median up.

    Args:
        corners: (2, 4) predicted quadrilateral.
        mask: (H, W) bool mask.
        tolerance_px: Distance counted as explained.

    Returns:
        ``(fraction_within_tolerance, median_distance_px)``; ``(nan, nan)`` when
        the mask has no interior boundary.
    """
    contours_cv, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    points = [rearrange(c, "N 1 xy -> xy N", xy=2).astype(float)
              for c in contours_cv if c.shape[0] > 20]
    if not points:
        return float("nan"), float("nan")
    boundary = np.concatenate(points, axis=1)
    height, width = mask.shape
    interior = (
        (boundary[0] > 1) & (boundary[0] < width - 2)
        & (boundary[1] > 1) & (boundary[1] < height - 2)
    )
    boundary = boundary[:, interior]
    if boundary.shape[1] == 0:
        return float("nan"), float("nan")
    # Distance to the quadrilateral's outline, as a closed polygon of segments.
    distances = np.full(boundary.shape[1], np.inf)
    for j in range(4):
        start, end = corners[:, j], corners[:, (j + 1) % 4]
        edge = end - start
        length_sq = float(edge @ edge)
        if length_sq < 1e-12:
            continue
        t = np.clip(((boundary - start[:, None]).T @ edge) / length_sq, 0.0, 1.0)
        closest = start[:, None] + edge[:, None] * t[None, :]
        distances = np.minimum(distances, np.linalg.norm(boundary - closest, axis=0))
    return (float(np.mean(distances <= tolerance_px)), float(np.median(distances)))


def mask_coverage(corners: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the mask that falls inside a quadrilateral.

    Args:
        corners: (2, 4) quadrilateral corners.
        mask: (H, W) bool mask.

    Returns:
        ``|quad and mask| / |mask|``, or 0 when the mask is empty.
    """
    filled = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(filled, [np.round(corners.T).astype(np.int32)], 1)
    total = int(mask.sum())
    return float(np.logical_and(filled.astype(bool), mask).sum()) / total if total else 0.0


def _draw_line(canvas: np.ndarray, line: np.ndarray, color, thickness: int = 1) -> None:
    """Draw an infinite line across the canvas."""
    height, width = canvas.shape[:2]
    a, b, c = line
    if abs(b) > abs(a):
        p0, p1 = (0, int(round(-c / b))), (width, int(round(-(a * width + c) / b)))
    elif abs(a) > 1e-9:
        p0, p1 = (int(round(-c / a)), 0), (int(round(-(b * height + c) / a)), height)
    else:
        return
    cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)


def _label(canvas: np.ndarray, text: str) -> np.ndarray:
    """Put a caption band under one panel."""
    band = np.full((44, canvas.shape[1], 3), 22, np.uint8)
    cv2.putText(band, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([canvas, band])


def figure(
    case: Dict,
    stage: Stage,
    deployed: Tuple[np.ndarray, int, str],
    line_H: Optional[np.ndarray],
    info: Dict,
    out_path: Path,
) -> Path:
    """Six panels: the evidence, what each estimator used, and both results.

    Args:
        case: Frozen case dict.
        stage: Replayed Hough stage.
        deployed: ``(homography, corners_used, fit)`` of the deployed path.
        line_H: Homography from the line fit, or None.
        info: Report of the line fit.
        out_path: PNG destination.

    Returns:
        The written path.
    """
    image, mask = case["image"], case["mask"]
    init_corners, gt = case["init_corners"], case["gt_corners"]
    tinted = image.copy()
    tinted[mask] = (0.65 * tinted[mask]
                    + 0.35 * np.array([0, 100, 255], np.float32)).astype(np.uint8)
    side_colors = [(60, 220, 255), (255, 200, 40), (255, 90, 220), (120, 255, 120)]

    panels: List[np.ndarray] = []

    first = tinted.copy()
    cv2.polylines(first, [np.round(gt.T).astype(np.int32)], True, (90, 200, 0), 3, cv2.LINE_AA)
    incumbent = warp_corners(case["incumbent_H"], init_corners)
    cv2.polylines(first, [np.round(incumbent.T).astype(np.int32)], True,
                  (200, 200, 200), 2, cv2.LINE_AA)
    panels.append(_label(first, "mask (orange), GT (green), incumbent pose (grey)"))

    second = tinted.copy()
    for i in range(stage.lines.shape[1]):
        _draw_line(second, stage.lines[:, i], (180, 180, 180), 1)
        segment = stage.segments[i]
        cv2.line(second, tuple(np.round(segment[:, 0]).astype(int)),
                 tuple(np.round(segment[:, 1]).astype(int)),
                 side_colors[i % 4], 5, cv2.LINE_AA)
    panels.append(_label(second, f"{stage.lines.shape[1]} refined lines, supported segments"))

    third = tinted.copy()
    for i in range(stage.intersections.shape[1]):
        point = tuple(np.round(stage.intersections[:, i]).astype(int))
        if stage.kept[i]:
            cv2.circle(third, point, 11, (60, 255, 120), -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(third, point, (60, 60, 255), cv2.MARKER_TILTED_CROSS, 26, 4)
    panels.append(_label(
        third,
        f"intersections: {int(stage.kept.sum())} kept (dots), "
        f"{int((~stage.kept).sum())} dropped (crosses)",
    ))

    fourth = tinted.copy()
    assignment = info.get("assignment", {})
    for i in range(stage.lines.shape[1]):
        if i in assignment:
            colour = side_colors[assignment[i] % 4]
            _draw_line(fourth, stage.lines[:, i], colour, 3)
        else:
            _draw_line(fourth, stage.lines[:, i], (90, 90, 90), 1)
    if assignment:
        incumbent_quad = warp_corners(case["incumbent_H"], init_corners)
        for j in range(4):
            cv2.line(fourth, tuple(np.round(incumbent_quad[:, j]).astype(int)),
                     tuple(np.round(incumbent_quad[:, (j + 1) % 4]).astype(int)),
                     side_colors[j % 4], 1, cv2.LINE_AA)
    panels.append(_label(
        fourth,
        f"lines matched to template sides: {len(assignment)} "
        f"(thin: incumbent's sides)",
    ))

    fifth = tinted.copy()
    cv2.polylines(fifth, [np.round(gt.T).astype(np.int32)], True, (90, 200, 0), 3, cv2.LINE_AA)
    deployed_corners = warp_corners(deployed[0], init_corners)
    cv2.polylines(fifth, [np.round(deployed_corners.T).astype(np.int32)], True,
                  (230, 60, 130), 3, cv2.LINE_AA)
    deployed_error = float(alignment_error(gt[None], deployed_corners[None])[0])
    panels.append(_label(
        fifth,
        f"deployed: {deployed[1]} corners -> {deployed[2]}, {deployed_error:.1f} px",
    ))

    sixth = tinted.copy()
    cv2.polylines(sixth, [np.round(gt.T).astype(np.int32)], True, (90, 200, 0), 3, cv2.LINE_AA)
    if line_H is not None:
        line_corners = warp_corners(line_H, init_corners)
        cv2.polylines(sixth, [np.round(line_corners.T).astype(np.int32)], True,
                      (230, 60, 130), 3, cv2.LINE_AA)
        line_error = float(alignment_error(gt[None], line_corners[None])[0])
        caption = (f"lines: rank {info.get('rank', 0)}, "
                   f"{info.get('prior_dims', 0)} prior dims, "
                   f"residual {info.get('residual_px', float('nan')):.2f} px, "
                   f"{line_error:.1f} px")
    else:
        caption = f"lines: no fit ({info.get('reason', 'unknown')})"
    panels.append(_label(sixth, caption))

    rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Case stems to run (default: all).")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    out_dir = args.out or (results_dir / "experiments" / "maskh_line_fit")
    selected = [p for p in cases(results_dir)
                if args.cases is None or p.stem in args.cases]
    if not selected:
        raise ValueError("no cases found; run experiments/maskh_cases.py first")

    print("  GT error, then how much of the visible silhouette each pose explains")
    print(f"  {'case':<22} {'depl px':>8} {'line px':>8} {'fit':>11} "
          f"{'depl expl':>10} {'line expl':>10} {'depl med':>9} {'line med':>9} {'rank':>5}")
    rows: List[Tuple[str, float, float, float, float, float]] = []
    for path in selected:
        case = load(path)
        stage = run_hough(case["mask"])
        deployed = deployed_pose(case["mask"], case["init_corners"], case["incumbent_H"])
        line_H, info = line_pose(case, stage)
        gt = case["gt_corners"]
        deployed_error = float(alignment_error(
            gt[None], warp_corners(deployed[0], case["init_corners"])[None])[0])
        if line_H is not None:
            line_error = float(alignment_error(
                gt[None], warp_corners(line_H, case["init_corners"])[None])[0])
        else:
            line_error = float("nan")
        deployed_explained, deployed_median = boundary_fit(
            warp_corners(deployed[0], case["init_corners"]), case["mask"])
        if line_H is not None:
            line_explained, line_median = boundary_fit(
                warp_corners(line_H, case["init_corners"]), case["mask"])
        else:
            line_explained, line_median = float("nan"), float("nan")
        print(f"  {path.stem:<22} {deployed_error:8.1f} {line_error:8.1f} "
              f"{deployed[2]:>11} {deployed_explained:10.2f} {line_explained:10.2f} "
              f"{deployed_median:9.1f} {line_median:9.1f} {info.get('rank', 0):5d}")
        rows.append((path.stem, case["stored_error_px"], deployed_error, line_error,
                     deployed_explained, line_explained))
        if args.figures:
            figure(case, stage, deployed, line_H, info, out_dir / f"{path.stem}.png")

    failing = [r for r in rows if r[1] > 15.0]
    healthy = [r for r in rows if r[1] <= 15.0]
    for name, group in (("failing", failing), ("healthy", healthy)):
        if not group:
            continue
        deployed_mean = float(np.mean([r[2] for r in group]))
        line_values = [r[3] for r in group if np.isfinite(r[3])]
        line_mean = float(np.mean(line_values)) if line_values else float("nan")
        fixed = sum(1 for r in group if np.isfinite(r[3]) and r[3] <= 15.0 < r[2])
        broken = sum(1 for r in group if r[2] <= 15.0 and not (np.isfinite(r[3]) and r[3] <= 15.0))
        deployed_expl = float(np.nanmean([r[4] for r in group]))
        line_expl = float(np.nanmean([r[5] for r in group]))
        print(f"\n  {name} cases: {len(group)}\n"
              f"    GT error   deployed {deployed_mean:7.1f} px   lines {line_mean:7.1f} px"
              f"   fixed {fixed}, broken {broken}\n"
              f"    silhouette deployed {deployed_expl:7.2f}      lines {line_expl:7.2f}"
              f"   (fraction of visible boundary within 3 px)")
    if args.figures:
        print(f"\n  figures in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
