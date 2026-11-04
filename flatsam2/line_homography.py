"""Homography from line correspondences, points, and an incumbent pose.

The Hough mask branch turns boundary lines into corner intersections and needs
four of them; when an occluder or the image border removes one, it discards
correctly fitted lines and falls back to a similarity on its previous pose. Three
lines still carry six of the eight degrees of freedom, so the pose is mostly
determined and only the missing directions need the incumbent.

This module solves that directly:

1. **Normalize.** Points and lines are conditioned by Hartley similarities, which
   is what a raw algebraic solve on pixel coordinates gets wrong -- entries of
   the design matrix otherwise differ by six orders of magnitude and the small
   singular values that carry the answer are lost in rounding.
2. **Solve where the data determines the pose, regularize where it does not.**
   Directions the hard data leaves free are filled by low-weight *soft points*
   -- typically an incumbent pose's own prediction for a corner no line pins --
   folded into the same normalized least-squares system rather than projected
   onto a null space afterward. An earlier version of this module did the
   latter: it solved for the pose satisfying the lines exactly and closest to
   the incumbent in raw matrix (Frobenius) coordinates, which is not the same
   as closest in image (pixel) coordinates, and the two can disagree by
   hundreds of pixels on a real corner even though both look correct in the
   normalized system. Weighting the incumbent's own prediction as data, in the
   same units as everything else, does not have that failure mode.
3. **Refine geometrically.** The linear step minimizes an algebraic residual. A
   short Gauss-Newton pass on true point-to-line distances follows, which is the
   same algebraic-versus-geometric correction that was needed for the
   point-based fit, applied to the same weighted objective.

A line is a homogeneous ``(3,)`` vector ``(a, b, c)`` with ``ax + by + c = 0``.
Template lines live in initial-frame coordinates, detected lines in the current
frame, and the homography maps initial to current, so ``l ~ H^T m``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from einops import rearrange

# Below this ratio to the largest singular value, a direction counts as
# unconstrained by the data and is taken from the incumbent instead.
_RANK_TOLERANCE = 0.02
# Gauss-Newton on the geometric residual; the linear solution starts inside a
# pixel or two, so this converges in a handful of steps or not at all.
_REFINE_ITERS = 20
_REFINE_STEP_TOL = 1e-7
_SAMPLES_PER_LINE = 16


@dataclass
class LineFitReport:
    """What the fit had, what it used, and how well it ended up agreeing.

    Attributes:
        n_lines: Line correspondences supplied.
        n_points: Point correspondences supplied.
        rank: Effective rank of the design matrix (degrees of freedom the data
            fixed, of the eight a homography has).
        singular_values: Singular values of the normalized design matrix.
        prior_dims: Dimensions taken from the incumbent (``9 - rank``).
        residual_px: RMS geometric residual after refinement, in pixels.
        refined: Whether Gauss-Newton improved on the linear solution.
        reason: Empty when the fit succeeded, else why it did not.
    """

    n_lines: int = 0
    n_points: int = 0
    rank: int = 0
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prior_dims: int = 0
    residual_px: float = float("inf")
    refined: bool = False
    reason: str = ""


def normalizing_similarity(points: np.ndarray) -> np.ndarray:
    """Hartley conditioning transform for a point set.

    Args:
        points: (2, N) coordinates.

    Returns:
        (3, 3) similarity mapping the points to zero mean and mean radius
        ``sqrt(2)``.
    """
    centroid = points.mean(axis=1)
    centered = points - centroid[:, None]
    mean_radius = float(np.mean(np.linalg.norm(centered, axis=0)))
    scale = np.sqrt(2.0) / mean_radius if mean_radius > 1e-12 else 1.0
    return np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]]
    )


def transform_line(similarity: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Map a line through a point transform, normalized to unit normal.

    Args:
        similarity: (3, 3) transform applied to points.
        line: (3,) homogeneous line in the original coordinates.

    Returns:
        (3,) line in the transformed coordinates.
    """
    mapped = np.linalg.inv(similarity).T @ line
    norm = np.linalg.norm(mapped[:2])
    return mapped / norm if norm > 1e-12 else mapped


def line_through(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Homogeneous line through two points, with a unit normal.

    Args:
        point_a: (2,) point.
        point_b: (2,) point.

    Returns:
        (3,) line ``(a, b, c)`` with ``a^2 + b^2 = 1``.
    """
    line = np.cross(np.append(point_a, 1.0), np.append(point_b, 1.0))
    norm = np.linalg.norm(line[:2])
    return line / norm if norm > 1e-12 else line


def point_line_distance(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perpendicular distances from points to a line.

    Args:
        line: (3,) homogeneous line.
        points: (2, N) coordinates.

    Returns:
        (N,) distances.
    """
    norm = np.linalg.norm(line[:2])
    norm = norm if norm > 1e-12 else 1.0
    return np.abs(line[0] * points[0] + line[1] * points[1] + line[2]) / norm


def _line_rows(template_line: np.ndarray, current_line: np.ndarray) -> np.ndarray:
    """Two rows of the design matrix for one line correspondence.

    The constraint is ``l x (H^T m) = 0`` for template line ``l`` and current
    line ``m``; two of its three components are independent.

    Args:
        template_line: (3,) line in template coordinates.
        current_line: (3,) line in current-frame coordinates.

    Returns:
        (2, 9) rows acting on ``vec(H)`` in row-major order.
    """
    l1, l2, l3 = template_line
    rows = np.zeros((2, 9))
    for i in range(3):
        # u_k = sum_i H_ik m_i ; H_ik is entry 3 * i + k of vec(H).
        rows[0, 3 * i + 2] += l2 * current_line[i]
        rows[0, 3 * i + 1] -= l3 * current_line[i]
        rows[1, 3 * i + 0] += l3 * current_line[i]
        rows[1, 3 * i + 2] -= l1 * current_line[i]
    return rows


def _point_rows(template_point: np.ndarray, current_point: np.ndarray) -> np.ndarray:
    """Two DLT rows for one point correspondence.

    Args:
        template_point: (2,) point in template coordinates.
        current_point: (2,) point in current-frame coordinates.

    Returns:
        (2, 9) rows acting on ``vec(H)`` in row-major order.
    """
    x, y = template_point
    u, v = current_point
    return np.array([
        [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u],
        [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v],
    ])


def _geometric_residuals(
    homography: np.ndarray,
    line_samples: Sequence[Tuple[np.ndarray, np.ndarray]],
    point_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Point-to-line and point-to-point distances under one homography.

    Args:
        homography: (3, 3) homography, template to current.
        line_samples: ``(samples (2, K), current_line (3,))`` per line
            correspondence; samples lie on the template line.
        point_pairs: ``(template_point (2,), current_point (2,))`` pairs.

    Returns:
        (M,) residuals in the current frame's units.
    """
    residuals: List[np.ndarray] = []
    for samples, current_line in line_samples:
        warped = _warp(homography, samples)
        residuals.append(point_line_distance(current_line, warped))
    for template_point, current_point in point_pairs:
        warped = _warp(homography, template_point[:, None])
        residuals.append(np.linalg.norm(warped[:, 0] - current_point)[None])
    return np.concatenate(residuals) if residuals else np.zeros(0)


def _warp(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to (2, N) points."""
    homogeneous = np.vstack([points, np.ones((1, points.shape[1]))])
    warped = homography @ homogeneous
    denominator = np.where(np.abs(warped[2:3]) < 1e-12, 1e-12, warped[2:3])
    return warped[:2] / denominator


def _weighted_residuals(
    homography: np.ndarray,
    line_samples: Sequence[Tuple[np.ndarray, np.ndarray]],
    point_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    soft_points: Sequence[Tuple[Tuple[np.ndarray, np.ndarray], float]],
) -> np.ndarray:
    """Geometric residuals with each soft point pre-scaled by ``sqrt(weight)``.

    Args:
        homography: (3, 3) homography.
        line_samples: As in :func:`_geometric_residuals`.
        point_pairs: As in :func:`_geometric_residuals`.
        soft_points: ``((template_point, current_point), weight)`` pairs.

    Returns:
        Concatenated residuals; hard evidence unscaled, soft points scaled so
        the same sum-of-squares objective trades them off against the hard data.
    """
    hard = _geometric_residuals(homography, line_samples, point_pairs)
    if not soft_points:
        return hard
    soft = _geometric_residuals(
        homography, [], [pair for pair, _ in soft_points]
    )
    weights = np.array([np.sqrt(max(w, 0.0)) for _, w in soft_points])
    return np.concatenate([hard, soft * weights])


def _refine(
    homography: np.ndarray,
    line_samples: Sequence[Tuple[np.ndarray, np.ndarray]],
    point_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    soft_points: Sequence[Tuple[Tuple[np.ndarray, np.ndarray], float]] = (),
) -> Tuple[np.ndarray, bool]:
    """Gauss-Newton on the geometric residual, in normalized coordinates.

    The linear solve minimizes an algebraic quantity that weights correspondences
    by how far they sit from the origin; this pass minimizes the distances that
    actually matter, including the same weighting the linear solve used for the
    soft points. Eight parameters, with the ninth fixed by renormalizing.

    Args:
        homography: (3, 3) starting homography.
        line_samples: As in :func:`_geometric_residuals`.
        point_pairs: As in :func:`_geometric_residuals`.
        soft_points: As in :func:`_weighted_residuals`.

    Returns:
        ``(homography, improved)``.
    """
    current = homography / np.linalg.norm(homography)
    residual_fn = lambda h: _weighted_residuals(h, line_samples, point_pairs, soft_points)
    best_cost = float(np.sum(residual_fn(current) ** 2))
    improved = False
    for _ in range(_REFINE_ITERS):
        residuals = residual_fn(current)
        jacobian = np.zeros((residuals.size, 9))
        step = 1e-6
        for k in range(9):
            perturbed = current.copy()
            perturbed[k // 3, k % 3] += step
            perturbed /= np.linalg.norm(perturbed)
            jacobian[:, k] = (residual_fn(perturbed) - residuals) / step
        # The scale direction is a gauge freedom; damping keeps it harmless.
        delta, *_ = np.linalg.lstsq(
            jacobian.T @ jacobian + 1e-9 * np.eye(9), -jacobian.T @ residuals, rcond=None
        )
        candidate = current + rearrange(delta, "(r c) -> r c", r=3, c=3)
        norm = np.linalg.norm(candidate)
        if norm < 1e-12:
            break
        candidate /= norm
        cost = float(np.sum(residual_fn(candidate) ** 2))
        if not np.isfinite(cost) or cost >= best_cost:
            break
        current, best_cost, improved = candidate, cost, True
        if np.linalg.norm(delta) < _REFINE_STEP_TOL:
            break
    return current, improved


def fit_homography(
    line_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    point_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    soft_points: Sequence[Tuple[np.ndarray, np.ndarray, float]] = (),
    template_segments: Optional[Sequence[np.ndarray]] = None,
    rank_tolerance: float = _RANK_TOLERANCE,
    refine: bool = True,
) -> Tuple[Optional[np.ndarray], LineFitReport]:
    """Homography from lines and points, softly regularized where they run out.

    Args:
        line_pairs: ``(template_line (3,), current_line (3,))`` correspondences,
            in unnormalized image coordinates. Treated as exact.
        point_pairs: ``(template_point (2,), current_point (2,))`` correspondences,
            also treated as exact.
        soft_points: ``(template_point (2,), current_point (2,), weight)`` triples
            for directions the hard data leaves unconstrained -- typically an
            incumbent pose's own prediction for a corner no line or point pins.
            ``weight`` in ``(0, 1]`` scales how much the point pulls the fit; it
            is a regularizer, not a measurement, and is folded into the same
            normalized least-squares solve as the hard data rather than applied
            as a separate matrix-space projection, which is what let a
            geometrically distant answer look "closest to the incumbent" in the
            first version of this function.
        template_segments: Optional ``(2, 2)`` endpoint pairs, one per line
            correspondence, giving the extent of each template side. The
            geometric refinement samples these; without them it samples a unit
            span around each template line's foot point, which is scale-blind.
        rank_tolerance: Singular values below this fraction of the largest count
            as unconstrained by the hard data; reported for diagnostics only,
            since ``soft_points`` (not this) decides what fills the gap.
        refine: Run the geometric refinement after the linear solve. Its
            numerical Gauss-Newton step can, on a near-degenerate configuration,
            land on a self-intersecting or flipped quad even when the linear
            solve was fine -- observed on real cases, not hypothetical. A caller
            with its own domain check (e.g. convexity against known reference
            points) should re-verify the refined result and fall back to the
            unrefined one if it fails, rather than trust ``refine=True``
            unconditionally.

    Returns:
        ``(homography, report)``; the homography is None when the fit failed, and
        the report says why.

    Raises:
        ValueError: When ``template_segments`` is given with a different length
            than ``line_pairs``.
    """
    report = LineFitReport(n_lines=len(line_pairs), n_points=len(point_pairs))
    if template_segments is not None and len(template_segments) != len(line_pairs):
        raise ValueError(
            f"{len(template_segments)} template segments for {len(line_pairs)} lines"
        )
    if 2 * len(line_pairs) + 2 * len(point_pairs) + 2 * len(soft_points) < 4:
        report.reason = "fewer than two correspondences"
        return None, report

    # Conditioning: build the similarities from whatever geometry is at hand,
    # hard evidence first so the frame is not skewed by a possibly-stale prior.
    template_anchors = [p for p, _ in point_pairs] or [p for p, _, _ in soft_points]
    current_anchors = [q for _, q in point_pairs] or [q for _, q, _ in soft_points]
    if template_segments is not None:
        for segment in template_segments:
            template_anchors.extend([segment[:, 0], segment[:, 1]])
    if not template_anchors or not current_anchors:
        report.reason = "no geometry to condition on"
        return None, report
    template_T = normalizing_similarity(np.stack(template_anchors, axis=1))
    current_T = normalizing_similarity(np.stack(current_anchors, axis=1))

    hard_rows: List[np.ndarray] = []
    for template_line, current_line in line_pairs:
        hard_rows.append(_line_rows(
            transform_line(template_T, template_line),
            transform_line(current_T, current_line),
        ))
    for template_point, current_point in point_pairs:
        hard_rows.append(_point_rows(
            _warp(template_T, template_point[:, None])[:, 0],
            _warp(current_T, current_point[:, None])[:, 0],
        ))
    hard_design = (
        np.concatenate(hard_rows, axis=0) if hard_rows else np.zeros((0, 9))
    )
    row_norms = np.linalg.norm(hard_design, axis=1, keepdims=True)
    hard_design = hard_design / np.where(row_norms < 1e-12, 1.0, row_norms)

    if hard_design.shape[0]:
        singular_values = np.linalg.svd(hard_design, compute_uv=False)
        report.singular_values = singular_values
        largest = singular_values[0]
        report.rank = (
            int(np.sum(singular_values > rank_tolerance * largest)) if largest > 0 else 0
        )
        report.prior_dims = 9 - report.rank
    elif not soft_points:
        report.reason = "no hard data and no prior"
        return None, report

    soft_rows: List[np.ndarray] = []
    for template_point, current_point, weight in soft_points:
        rows = _point_rows(
            _warp(template_T, template_point[:, None])[:, 0],
            _warp(current_T, current_point[:, None])[:, 0],
        )
        row_scale = np.linalg.norm(rows, axis=1, keepdims=True)
        soft_rows.append(np.sqrt(max(weight, 0.0)) * rows
                         / np.where(row_scale < 1e-12, 1.0, row_scale))
    design = np.concatenate([hard_design] + soft_rows, axis=0)
    if design.shape[0] < 2:
        report.reason = "fewer than two normalized rows"
        return None, report

    _, _, vt = np.linalg.svd(design)
    solution = vt[-1]
    norm = np.linalg.norm(solution)
    if norm < 1e-12:
        report.reason = "degenerate design matrix"
        return None, report
    normalized_H = rearrange(solution / norm, "(r c) -> r c", r=3, c=3)

    if refine:
        samples = _line_samples(line_pairs, template_segments, template_T, current_T)
        normalized_points = [
            (_warp(template_T, p[:, None])[:, 0], _warp(current_T, q[:, None])[:, 0])
            for p, q in point_pairs
        ]
        normalized_soft = [
            ((_warp(template_T, p[:, None])[:, 0], _warp(current_T, q[:, None])[:, 0]), w)
            for p, q, w in soft_points
        ]
        normalized_H, report.refined = _refine(
            normalized_H, samples, normalized_points, normalized_soft
        )

    homography = np.linalg.inv(current_T) @ normalized_H @ template_T
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    if not np.all(np.isfinite(homography)):
        report.reason = "non-finite homography"
        return None, report

    report.residual_px = float(np.sqrt(np.mean(_geometric_residuals(
        homography,
        _line_samples(line_pairs, template_segments, None, None),
        list(point_pairs),
    ) ** 2))) if line_pairs or point_pairs else 0.0
    return homography, report


def _line_samples(
    line_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    template_segments: Optional[Sequence[np.ndarray]],
    template_T: Optional[np.ndarray],
    current_T: Optional[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample points along each template line, paired with its current line.

    Args:
        line_pairs: Line correspondences.
        template_segments: Optional (2, 2) endpoint pairs per correspondence.
        template_T: Optional conditioning transform for the samples.
        current_T: Optional conditioning transform for the current lines.

    Returns:
        ``(samples (2, K), current_line (3,))`` per correspondence.
    """
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, (template_line, current_line) in enumerate(line_pairs):
        if template_segments is not None:
            start, end = template_segments[i][:, 0], template_segments[i][:, 1]
        else:
            # Foot of the origin on the line, plus a unit step along it.
            normal = template_line[:2] / max(np.linalg.norm(template_line[:2]), 1e-12)
            foot = -template_line[2] * normal
            direction = np.array([-normal[1], normal[0]])
            start, end = foot - direction, foot + direction
        t = np.linspace(0.0, 1.0, _SAMPLES_PER_LINE)
        samples = start[:, None] + t[None, :] * (end - start)[:, None]
        if template_T is not None:
            samples = _warp(template_T, samples)
        line = transform_line(current_T, current_line) if current_T is not None else current_line
        out.append((samples, line))
    return out
