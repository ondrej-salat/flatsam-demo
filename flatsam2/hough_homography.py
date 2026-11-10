"""Mask -> Hough lines -> homography estimation pipeline.

Estimates a homography from a segmentation mask by:
1. Extracting mask contours
2. Running a gradient-based Hough transform to detect dominant lines
3. Refining line positions via SVD fitting to contour points
4. Intersecting line pairs to obtain candidate corner points
5. Matching candidate corners to template corners
6. Estimating a homography from the best match

Also provides appearance-based symmetry resolution (cyclic corner shift
disambiguation) using DINOv2 features, since the geometric pipeline and
NN quality estimation cannot distinguish between cyclically equivalent
homographies.

This module is self-contained and does not depend on the main FlatSAM tracker
pipeline.  It provides an alternative homography estimation path that works
purely from segmentation masks, without optical flow.

Ported from the original flatsam repository (src/flatsam/flatsam.py).
"""
import itertools
import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, pack
from skimage.feature import peak_local_max
from torchvision.transforms.functional import normalize, to_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projective geometry helpers
# ---------------------------------------------------------------------------

def e2p(pts: np.ndarray) -> np.ndarray:
    """Convert (D, N) euclidean coordinates to (D+1, N) projective."""
    return np.vstack((pts, np.ones(pts.shape[1])))


def p2e(pts: np.ndarray) -> np.ndarray:
    """Convert (D+1, N) projective coordinates to (D, N) euclidean."""
    if pts.size == 0:
        return pts
    return pts[:-1, :] / pts[-1, np.newaxis, :]


def H_warp_pts(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography to (xy, N) euclidean points.

    Args:
        H: (3, 3) homography matrix.
        pts: (2, N) or (2,) point coordinates.

    Returns:
        Transformed points, same shape as input.
    """
    flat_input = pts.ndim == 1
    if flat_input:
        pts = pts[:, np.newaxis]
    result = p2e(H @ e2p(pts))
    if flat_input:
        result = result.flatten()
    return result


def H_warp_lines(H: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Transform lines by a homography: lines' = H^{-T} * lines.

    Args:
        H: (3, 3) homography matrix that maps points from source to
            destination space.  Lines are transformed from the source
            coordinate system into the destination coordinate system.
        lines: (3, N) or (3,) homogeneous line coordinates (a, b, c).

    Returns:
        Transformed lines, same shape as input.
    """
    flat_input = lines.ndim == 1
    if flat_input:
        lines = lines[:, np.newaxis]
    result = np.linalg.inv(H).T @ lines
    if flat_input:
        result = result.flatten()
    return result


def line_line_intersection(
    line_1: np.ndarray,
    line_2: np.ndarray,
) -> np.ndarray:
    """Compute intersection of two lines in projective coordinates.

    Args:
        line_1: (3,) line (a, b, c) from ax + by + c = 0.
        line_2: (3,) line (a, b, c).

    Returns:
        (3, 1) point in homogeneous coordinates.
    """
    a1, b1, c1 = line_1.flatten()
    a2, b2, c2 = line_2.flatten()
    return np.array([[b1 * c2 - b2 * c1],
                     [a2 * c1 - a1 * c2],
                     [a1 * b2 - a2 * b1]])


def _signed_area(pts: np.ndarray) -> float:
    """Signed area of polygon with vertices (xy, N)."""
    x, y = pts
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _convex_quad(pts: np.ndarray) -> bool:
    """Check if 4 points form a convex quadrilateral."""
    signs = []
    for i in range(pts.shape[1]):
        a = pts[:, i] - pts[:, (i - 1) % 4]
        b = pts[:, (i + 1) % 4] - pts[:, i]
        signs.append(a[0] * b[1] - a[1] * b[0] > 0)
    return all(signs) or not any(signs)


def good_homography(H: Optional[np.ndarray], ref_pts: np.ndarray) -> bool:
    """Validate a homography: non-singular, non-twisting, non-flipping.

    Args:
        H: (3, 3) candidate homography (or None).
        ref_pts: (2, N) reference points to check geometric validity.

    Returns:
        True if the homography passes all validity checks.
    """
    if H is None:
        return False
    if np.linalg.matrix_rank(H) != 3:
        return False
    warped = H_warp_pts(H, ref_pts)
    if not _convex_quad(warped):
        return False
    if np.sign(_signed_area(ref_pts)) != np.sign(_signed_area(warped)):
        return False
    return True


# ---------------------------------------------------------------------------
# Contour utilities
# ---------------------------------------------------------------------------

def _approx_touching_edges(
    pts: np.ndarray,
    shape: tuple,
    margin: int = 0,
) -> np.ndarray:
    """Detect contour points near image boundaries.

    Args:
        pts: (xy, N) contour points.
        shape: (H, W) image shape.
        margin: pixels from edge to count as touching.

    Returns:
        (N,) bool array, True for boundary-adjacent points.
    """
    return np.any(
        (pts <= margin)
        | (pts >= [[shape[1] - 1 - margin], [shape[0] - 1 - margin]]),
        axis=0,
    )


def _remap(
    x: float,
    src_low: float,
    src_high: float,
    dst_low: float,
    dst_high: float,
) -> float:
    """Linear remapping from [src_low, src_high] to [dst_low, dst_high]."""
    return ((x - src_low) / (src_high - src_low)) * (dst_high - dst_low) + dst_low


# ---------------------------------------------------------------------------
# Shared contour pre-computation (avoids redundant traversal in Hough + refine)
# ---------------------------------------------------------------------------

@dataclass
class _ContourSamples:
    """Pre-computed tangent information for sampled contour points.

    Attributes:
        points: (2, M) sampled point coordinates (original dtype).
        tangents: (2, M) finite-difference tangent vectors (unnormalised).
    """

    points: np.ndarray
    tangents: np.ndarray


def _precompute_contour_samples(
    contours: List[np.ndarray],
    on_boundaries: List[np.ndarray],
    *,
    gradient_step: int = 4,
    max_samples: int = 1_000_000,
) -> List[_ContourSamples]:
    """Pre-compute sample points and tangent vectors for each contour.

    Filters out boundary-adjacent points and zero-tangent points.

    Args:
        contours: List of (xy, N_i) contour arrays.
        on_boundaries: List of (N_i,) bool arrays.
        gradient_step: Contour offset for tangent estimation.
        max_samples: Maximum contour points per contour.

    Returns:
        List of :class:`_ContourSamples`, one per contour (may be empty).
    """
    result: List[_ContourSamples] = []
    for pts, on_boundary in zip(contours, on_boundaries, strict=True):
        N = pts.shape[1]
        step_outer = max(N // max_samples, 1)

        sample_idx = np.arange(0, N, step_outer)
        sample_idx = sample_idx[~on_boundary[sample_idx]]
        if len(sample_idx) == 0:
            result.append(_ContourSamples(
                points=np.zeros((2, 0), dtype=pts.dtype),
                tangents=np.zeros((2, 0), dtype=pts.dtype),
            ))
            continue

        a = pts[:, sample_idx]                                  # (2, M)
        diff = (pts[:, (sample_idx + gradient_step) % N]
                - pts[:, (sample_idx - gradient_step + N) % N])  # (2, M)

        nonzero = (diff[0] != 0) | (diff[1] != 0)
        result.append(_ContourSamples(
            points=a[:, nonzero],
            tangents=diff[:, nonzero],
        ))

    return result


# ---------------------------------------------------------------------------
# Hough transform on contours
# ---------------------------------------------------------------------------

def contour_hough(
    mask: np.ndarray,
    contours: List[np.ndarray],
    on_boundaries: List[np.ndarray],
    *,
    gradient_step: int = 4,
    max_samples: int = 1_000_000,
    topk: int = 4,
    _samples: Optional[List[_ContourSamples]] = None,
) -> np.ndarray:
    """Gradient-based Hough transform on mask contours.

    Accumulates votes in (distance, angle) space from contour tangent
    directions, then extracts the top-k peak lines.

    Args:
        mask: (H, W) binary mask (unused directly, kept for API compat).
        contours: List of (xy, N_i) contour arrays in image coordinates.
        on_boundaries: List of (N_i,) bool arrays marking boundary-adjacent
            points.
        gradient_step: Contour offset for finite-difference tangent estimation.
        max_samples: Max contour points to process per contour.
        topk: Number of lines to return.
        _samples: Pre-computed contour samples (internal optimisation).
            When provided, *contours* / *on_boundaries* / *gradient_step* /
            *max_samples* are ignored for tangent computation.

    Returns:
        (3, K) array of lines in homogeneous form ``ax + by + c = 0``,
        with K <= topk.  Lines are in image coordinates.
    """
    if len(contours) == 0:
        return np.zeros((3, 0))

    if _samples is None:
        _samples = _precompute_contour_samples(
            contours, on_boundaries,
            gradient_step=gradient_step, max_samples=max_samples,
        )

    # Center contours for numerical stability of the Hough accumulator.
    all_cnt = np.concatenate(contours, axis=1)
    contour_center = np.mean(all_cnt, axis=1, keepdims=True)  # (2, 1)
    point_dists = [
        np.sqrt(np.sum((cnt - contour_center) ** 2, axis=0))
        for cnt in contours
    ]
    max_dist = max(np.amax(d) for d in point_dists)
    min_dist = 0.0

    N_angles, N_dists = 359, 100
    accumulator = np.zeros((N_dists, N_angles))
    degree_range = 10  # +/- degrees of angular spread per vote
    rad_range = np.radians(degree_range)
    delta_angles = np.linspace(-rad_range, rad_range, 11)  # (D,)
    vote_weights = _remap(np.abs(delta_angles), 0, rad_range, 1, 0.5)  # (D,)

    for sample in _samples:
        if sample.points.shape[1] == 0:
            continue

        a = sample.points - contour_center  # (2, M')  centred
        diff = sample.tangents              # (2, M')

        # Tangent angle and signed distance to origin.
        angle = np.arctan2(diff[1], diff[0])                  # (M',)
        dist = np.cos(angle) * a[1] - np.sin(angle) * a[0]   # (M',)

        # Normal direction conventions (vectorised conditional).
        norm_angle = angle + np.pi / 2                        # (M',)
        norm_to_direction = np.full_like(dist, -np.pi / 2)
        neg = dist < 0
        norm_angle[neg] -= np.pi
        norm_to_direction[neg] = np.pi / 2

        # Broadcast over angular spread:  (M', 1) + (1, D) -> (M', D).
        res_angle = (
            (norm_angle[:, np.newaxis] + delta_angles[np.newaxis, :]
             + 2 * np.pi) % (2 * np.pi)
        )
        dir_angle = res_angle + norm_to_direction[:, np.newaxis]

        d = np.abs(
            np.cos(dir_angle) * a[1, :, np.newaxis]
            - np.sin(dir_angle) * a[0, :, np.newaxis]
        )  # (M', D)

        # Map to accumulator bins.
        row = np.clip(
            np.round(
                _remap(d, min_dist, max_dist, 0, N_dists - 1)
            ).astype(np.intp),
            0, N_dists - 1,
        )
        col = np.clip(
            np.round(
                _remap(res_angle, 0, 2 * np.pi, 0, N_angles - 1)
            ).astype(np.intp),
            0, N_angles - 1,
        )

        # Accumulate votes (unbuffered add handles duplicate indices).
        np.add.at(
            accumulator,
            (row.ravel(), col.ravel()),
            np.broadcast_to(vote_weights, d.shape).ravel(),
        )

    # Peak detection with cyclic wrap-around in the angle dimension.
    cyclic = np.tile(accumulator, (1, 3))
    cyclic = scipy.ndimage.gaussian_filter(cyclic, sigma=4)

    peaks = peak_local_max(
        cyclic, min_distance=10, threshold_rel=None, exclude_border=False,
    )
    peaks = peaks[
        (peaks[:, 1] >= N_angles) & (peaks[:, 1] < 2 * N_angles), :
    ]
    peaks[:, 1] -= N_angles

    if len(peaks) == 0:
        return np.zeros((3, 0))

    blurred = cyclic[:, N_angles : 2 * N_angles]
    peak_vals = blurred[peaks[:, 0], peaks[:, 1]]
    real_topk = min(topk, len(peak_vals))
    top_ids = np.argpartition(peak_vals, -real_topk)[-real_topk:]
    peaks = peaks[top_ids]
    peaks = peaks[np.argsort(peaks[:, 1])]  # sort by angle

    # Convert peaks to line equations in centered coordinates.
    lines = []
    for peak in peaks:
        distance = _remap(peak[0], 0, N_dists - 1, min_dist, max_dist)
        angle = _remap(peak[1], 0, N_angles - 1, 0, 2 * np.pi)
        lines.append(np.array([np.cos(angle), np.sin(angle), -distance]))

    lines = rearrange(np.array(lines), 'N abc -> abc N', abc=3)

    # Transform lines from centered coordinates back to image coordinates.
    # H_uncenter maps points from centered space to image space:
    #   x_image = x_centered + center
    H_uncenter = np.eye(3)
    H_uncenter[0, 2] = contour_center[0, 0]
    H_uncenter[1, 2] = contour_center[1, 0]
    lines = H_warp_lines(H_uncenter, lines)

    return lines


# ---------------------------------------------------------------------------
# Line refinement
# ---------------------------------------------------------------------------

def refine_lines(
    lines: np.ndarray,
    contours: List[np.ndarray],
    on_boundaries: List[np.ndarray],
    *,
    gradient_step: int = 4,
    max_samples: int = 100_000,
    use_orthogonal_fit: bool = False,
    _samples: Optional[List[_ContourSamples]] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Refine Hough lines by fitting to nearby contour points.

    For each contour point, assigns it to the closest line (by angle and
    distance), then re-fits each line via SVD.

    Args:
        lines: (3, N) lines from :func:`contour_hough` in image coordinates.
        contours: List of (xy, N_i) contour arrays in image coordinates.
        on_boundaries: List of (N_i,) bool arrays.
        gradient_step: Contour offset for tangent estimation.
        max_samples: Max points to consider per contour.
        use_orthogonal_fit: Use centered total least squares. When False, preserves
            legacy algebraic SVD fit on ``[x, y, 1]``.
        _samples: Pre-computed contour samples (internal optimisation).
            When provided, *contours* / *on_boundaries* / *gradient_step* /
            *max_samples* are ignored for tangent computation.

    Returns:
        Tuple of ``(refined_lines, point_lists)`` where ``refined_lines``
        is (3, N) and ``point_lists[i]`` is (xy, M_i) array of contour
        points assigned to line *i* (may be empty ``(2, 0)``).
    """
    N_lines = lines.shape[1]
    if N_lines == 0:
        return lines, []

    if _samples is None:
        _samples = _precompute_contour_samples(
            contours, on_boundaries,
            gradient_step=gradient_step, max_samples=max_samples,
        )

    normalsT = lines[:2, :].T  # (N_lines, 2)
    distances = lines[2, :]     # (N_lines,)
    # Collect (2, K_j) point arrays per line (one chunk per contour).
    line_pt_chunks: List[List[np.ndarray]] = [[] for _ in range(N_lines)]

    for sample in _samples:
        if sample.points.shape[1] == 0:
            continue

        a_pts = sample.points.astype(np.float64)                   # (2, M')
        line_vecs = sample.tangents.astype(np.float64)             # (2, M')
        norms = np.hypot(line_vecs[0], line_vecs[1])              # (M',)
        line_vecs = line_vecs / norms                              # (2, M')
        M = a_pts.shape[1]

        # Signed distance from each point to each line:  (N_lines, M').
        dist_diffs = normalsT @ a_pts + distances[:, np.newaxis]

        # Angular agreement:  (N_lines, M').
        direction_dots = np.abs(normalsT @ line_vecs)
        angle_diffs = np.abs(
            np.arccos(np.clip(direction_dots, 0, 1)) - np.pi / 2
        )

        valid_angle = np.degrees(angle_diffs) <= 15
        valid_dist = np.abs(dist_diffs) <= 15
        valid = valid_angle & valid_dist                            # (N_lines, M')

        # Closest line for each point.
        closest_line = np.argmin(np.abs(dist_diffs), axis=0)       # (M',)
        assigned = valid[closest_line, np.arange(M)]               # (M',)

        # Group assigned points by line.
        for line_i in range(N_lines):
            mask = (closest_line == line_i) & assigned
            if np.any(mask):
                line_pt_chunks[line_i].append(sample.points[:, mask])

    # Merge chunks and re-fit each line via SVD.
    new_lines = []
    pts_out: List[np.ndarray] = []
    for line_i in range(N_lines):
        chunks = line_pt_chunks[line_i]
        if not chunks:
            new_lines.append(lines[:, line_i])
            pts_out.append(np.zeros((2, 0)))
            continue

        combined = np.concatenate(chunks, axis=1)  # (2, K)
        pts_out.append(combined)

        if use_orthogonal_fit:
            if combined.shape[1] < 2:
                new_lines.append(lines[:, line_i])
                continue
            centroid = np.mean(combined.astype(np.float64), axis=1)
            centered = combined.T.astype(np.float64) - centroid[np.newaxis, :]
            _, _, vt = np.linalg.svd(centered)
            a, b = vt[-1, :]
            n = math.hypot(a, b)
            if n <= 0:
                new_lines.append(lines[:, line_i])
                continue
            a, b = a / n, b / n
            c = -(a * centroid[0] + b * centroid[1])
        else:
            A = np.column_stack([combined.T, np.ones(combined.shape[1])])  # (K, 3)
            _, _, vt = np.linalg.svd(A)
            a, b, c = vt[-1, :]
            n = math.hypot(a, b)
            if n > 0:
                a, b, c = a / n, b / n, c / n
        # Convention: c <= 0
        if c > 0:
            a, b, c = -a, -b, -c
        new_lines.append(np.array([a, b, c]))

    refined = rearrange(np.array(new_lines), 'N abc -> abc N', abc=3)
    return refined, pts_out


# ---------------------------------------------------------------------------
# Line segment geometry
# ---------------------------------------------------------------------------

def _line_to_segment(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project contour points onto a line to get segment endpoints.

    Args:
        line: (3,) line coefficients (a, b, c).
        pts: (xy, M) points assigned to this line.

    Returns:
        (xy, 2) projected endpoint coordinates.
    """
    a, b, c = line
    norm = a ** 2 + b ** 2
    if norm == 0 or pts.shape[1] == 0:
        return np.zeros((2, 2))

    # Project points onto line direction vector to find extremes.
    line_dir = np.array([[-b, a]]) / norm
    proj = (line_dir @ pts).flatten()
    min_id, max_id = int(np.argmin(proj)), int(np.argmax(proj))

    xs = pts[0, [min_id, max_id]]
    ys = pts[1, [min_id, max_id]]

    # Orthogonal projection of the extreme points onto the line.
    xs_proj = (b * (b * xs - a * ys) - a * c) / norm
    ys_proj = (a * (-b * xs + a * ys) - b * c) / norm
    return np.stack([xs_proj, ys_proj], axis=0)


def _segment_length(endpoints: np.ndarray) -> float:
    """Length of a line segment from (xy, 2) endpoint array."""
    return float(np.linalg.norm(endpoints[:, 0] - endpoints[:, 1]))


# ---------------------------------------------------------------------------
# Line intersections
# ---------------------------------------------------------------------------

def lines_to_points(
    lines: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Intersect consecutive line pairs to get candidate corner points.

    Lines are treated as forming a cyclic chain: line 0 ∩ line N-1 gives
    the first intersection, line 0 ∩ line 1 the second, etc.

    Args:
        lines: (3, N) homogeneous lines.

    Returns:
        Tuple of ``(points_h, parents)`` where ``points_h`` is (3, N) in
        homogeneous coordinates and ``parents[i]`` is
        ``(line_idx_1, line_idx_2)`` for each intersection.
    """
    N = lines.shape[1]
    if N < 2:
        return np.zeros((3, 0)), []

    pts = []
    parents = []
    for i in range(N):
        prev = lines[:, (i - 1) % N]
        cur = lines[:, i]
        pts.append(line_line_intersection(prev, cur))
        parents.append(((i - 1) % N, i))

    return np.concatenate(pts, axis=1), parents


# ---------------------------------------------------------------------------
# Point matching and homography estimation
# ---------------------------------------------------------------------------

def keep_only_points_close_to_contours(
    points: np.ndarray,
    contours: List[np.ndarray],
    max_distance_px: float = 80.0,
    *,
    parents: Optional[List[Tuple[int, int]]] = None,
    lines: Optional[np.ndarray] = None,
    min_angle_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points to those near mask contours.

    Args:
        points: (xy, N) candidate points.
        contours: List of (xy, M_i) contour arrays.
        max_distance_px: Maximum distance threshold (pixels).
        parents: Optional line-parent indices for each point.
        lines: Optional (3, K) homogeneous lines used for angle filtering.
        min_angle_deg: Minimum angle between parent lines in degrees. Zero disables
            angle filtering.

    Returns:
        Tuple of ``(filtered_points (xy, K), good_mask (N,) bool)``.
    """
    if points.size == 0:
        return points, np.zeros(0, dtype=bool)

    flat_contours, _ = pack(contours, 'xy *')
    # (xy, N_pts, 1) - (xy, 1, N_contours)
    diff = (
        points[:, :, np.newaxis] - flat_contours[:, np.newaxis, :]
    )
    dists_sq = np.sum(diff ** 2, axis=0)  # (N_pts, N_contours)
    min_dist_sq = np.min(dists_sq, axis=1)  # (N_pts,)
    good = min_dist_sq < max_distance_px ** 2
    if min_angle_deg > 0.0:
        if parents is None or lines is None:
            raise ValueError("parents and lines are required when min_angle_deg > 0")
        if len(parents) != points.shape[1]:
            raise ValueError(
                f"Expected {points.shape[1]} parent records, got {len(parents)}"
            )
        min_sin = math.sin(math.radians(min_angle_deg))
        angle_ok = np.zeros(points.shape[1], dtype=bool)
        for point_i, (line_i, line_j) in enumerate(parents):
            normal_i = lines[:2, line_i]
            normal_j = lines[:2, line_j]
            denom = float(np.linalg.norm(normal_i) * np.linalg.norm(normal_j))
            if denom <= 0.0:
                continue
            sin_angle = abs(float(normal_i[0] * normal_j[1] - normal_i[1] * normal_j[0]))
            angle_ok[point_i] = sin_angle / denom >= min_sin
        good = good & angle_ok
    return points[:, good], good


def bruteforce_pointset_matching(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> List[Tuple[tuple, tuple]]:
    """Enumerate all possible point matchings between two point sets.

    For small point sets (N, M <= ~5), this is tractable.  The number of
    matchings grows quickly, but in practice N=4 (template corners) and
    M<=4 (line intersections), giving at most a few hundred candidates.

    Args:
        pts_a: (xy, N) template points.
        pts_b: (xy, M) candidate points.

    Returns:
        List of ``(a_indices, b_indices)`` tuples.
    """
    if pts_b.size == 0:
        return []
    N_a, N_b = pts_a.shape[1], pts_b.shape[1]
    matchings: List[Tuple[tuple, tuple]] = []
    for n in range(1, min(N_a, N_b) + 1):
        for subset_a in itertools.combinations(range(N_a), n):
            for perm_b in itertools.permutations(range(N_b), n):
                matchings.append((subset_a, perm_b))
    return matchings


def discard_diagonal_matches(
    matches: List[Tuple[tuple, tuple]],
    parents: List[Tuple[int, int]],
    n_template_points: int,
) -> List[Tuple[tuple, tuple]]:
    """Remove geometrically inconsistent matchings.

    Discards matches where two consecutive template corners map to
    intersection points from non-adjacent lines (and vice versa).

    Args:
        matches: List of ``(a_ids, b_ids)`` from brute-force matching.
        parents: Line parent indices from :func:`lines_to_points`.
        n_template_points: Number of template corners (typically 4).

    Returns:
        Filtered list of matchings.
    """
    result = []
    for a_ids, b_ids in matches:
        if len(a_ids) == 2:
            a_consec = (
                abs(a_ids[0] - a_ids[1]) == 1
            ) or (
                min(a_ids) == 0 and max(a_ids) == n_template_points - 1
            )
            b_consec = (
                len(set.union(*(set(parents[i]) for i in b_ids))) == 3
            )
            if a_consec != b_consec:
                continue
        result.append((a_ids, b_ids))
    return result


# ---------------------------------------------------------------------------
# Transformation estimators  (TRS / translation / full-H fallback chain)
# ---------------------------------------------------------------------------

def _find_TRS(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate similarity transform (translation + rotation + uniform scale).

    Solves the system ``dst = [[A, B], [-B, A]] @ src + [[C], [D]]``
    via least squares.

    Args:
        src: (xy, N) source points.
        dst: (xy, N) destination points.

    Returns:
        (3, 3) similarity-transform homography.
    """
    X, Y = src[0, :], src[1, :]
    Xp, Yp = dst[0, :], dst[1, :]
    ones = np.ones_like(X)
    zeros = np.zeros_like(X)

    ax = np.stack([X, Y, ones, zeros], axis=1)   # (N, 4)
    ay = np.stack([Y, -X, zeros, ones], axis=1)   # (N, 4)

    A = np.concatenate([ax, ay], axis=1)           # (N, 8)
    b = np.stack([Xp, Yp], axis=1)                 # (N, 2)

    A = rearrange(A, 'N (two four) -> (N two) four', two=2, four=4)
    b = rearrange(b, 'N (two one) -> (N two) one', two=2, one=1)

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a_val, b_val, c_val, d_val = x.flatten()

    return np.array([
        [a_val,  b_val, c_val],
        [-b_val, a_val, d_val],
        [0,      0,     1],
    ])


def _find_T(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate pure translation from point correspondences.

    Args:
        src: (xy, N) source points.
        dst: (xy, N) destination points.

    Returns:
        (3, 3) translation homography.
    """
    shift = reduce(dst - src, 'xy N -> xy', reduction='mean', xy=2)
    H = np.eye(3)
    H[:2, 2] = shift
    return H


def _find_transformation(
    init_pts: np.ndarray,
    current_pts: np.ndarray,
    last_H_init2current: np.ndarray,
) -> np.ndarray:
    """Estimate transformation from point correspondences with fallbacks.

    Handles different numbers of correspondences:
    - >= 4: full homography via DLT
    - 2-3: similarity (TRS) relative to last frame
    - 1:   translation relative to last frame
    - 0:   keep last homography (no update)

    Args:
        init_pts: (xy, N) template point coordinates.
        current_pts: (xy, N) current frame point coordinates.
        last_H_init2current: (3, 3) previous homography (init -> current).

    Returns:
        (3, 3) estimated homography (init -> current).
    """
    N = init_pts.shape[1]
    if N >= 4:
        H, _ = cv2.findHomography(
            rearrange(init_pts, 'xy N -> N 1 xy', xy=2),
            rearrange(current_pts, 'xy N -> N 1 xy', xy=2),
            method=0,
        )
        if H is not None:
            return H
        return last_H_init2current

    if N > 1:
        last_pts = H_warp_pts(last_H_init2current, init_pts)
        trs = _find_TRS(last_pts, current_pts)
        return trs @ last_H_init2current

    if N == 1:
        last_pts = H_warp_pts(last_H_init2current, init_pts)
        t = _find_T(last_pts, current_pts)
        return t @ last_H_init2current

    return last_H_init2current


def find_closest_H(
    init_coords: np.ndarray,
    current_coords: np.ndarray,
    last_H_init2current: np.ndarray,
    point_matches: List[Tuple[tuple, tuple]],
) -> Tuple[np.ndarray, float, Tuple]:
    """Find the homography most consistent with previous tracking state.

    Tries all candidate matchings and picks the one whose warped template
    corners are closest to the last-frame prediction.

    Args:
        init_coords: (xy, 4) template corner coordinates.
        current_coords: (xy, M) detected intersection points.
        last_H_init2current: (3, 3) previous homography (init -> current).
        point_matches: List of ``(a_ids, b_ids)`` candidate matchings.

    Returns:
        Tuple of ``(best_H, best_diff, best_ids)``.
    """
    best_diff = np.inf
    best_H = last_H_init2current.copy()
    best_ids: Tuple = ((), ())
    most_matches = max((len(a) for a, _ in point_matches), default=0)

    for a_ids, b_ids in point_matches:
        if len(a_ids) < most_matches:
            continue
        a_pts = init_coords[:, list(a_ids)]
        b_pts = current_coords[:, list(b_ids)]
        H = _find_transformation(a_pts, b_pts, last_H_init2current)

        if not good_homography(H, init_coords):
            continue
        H = H / H[2, 2]

        warped = H_warp_pts(H, init_coords)
        last_pred = H_warp_pts(last_H_init2current, init_coords)
        diff = float(np.mean(np.linalg.norm(warped - last_pred, axis=0)))

        if diff < best_diff:
            best_diff = diff
            best_H = H
            best_ids = (a_ids, b_ids)

    return best_H, best_diff, best_ids


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def mask_to_homography(
    mask: np.ndarray,
    init_corners: np.ndarray,
    last_H_init2current: np.ndarray,
    *,
    gradient_step: int = 4,
    topk_lines: int = 4,
    corner_max_distance_px: float = 80.0,
    corner_max_distance_image_fraction: Optional[float] = None,
    min_intersection_angle_deg: float = 0.0,
    line_refinement_method: str = "legacy_algebraic",
) -> Tuple[np.ndarray, int]:
    """Estimate homography from a segmentation mask using Hough line detection.

    Top-level function implementing the full pipeline:
    mask -> contours -> Hough lines -> refined lines -> intersections
    -> point matching -> homography.

    Args:
        mask: (H, W) binary segmentation mask (bool or uint8).
        init_corners: (xy, 4) template corner coordinates
            ``[[x0,x1,x2,x3], [y0,y1,y2,y3]]``.
        last_H_init2current: (3, 3) previous-frame homography estimate
            (init -> current).
        gradient_step: Contour tangent estimation step.
        topk_lines: Number of Hough lines to extract.
        corner_max_distance_px: Maximum intersection distance from mask contour.
        corner_max_distance_image_fraction: Optional image-relative distance threshold.
        min_intersection_angle_deg: Minimum angle between adjacent lines for keeping
            an intersection. Zero preserves legacy behavior.
        line_refinement_method: ``legacy_algebraic`` preserves previous line fit;
            ``orthogonal_tls`` uses centered total least squares.

    Returns:
        Tuple of ``(H_init2current (3, 3), n_correspondences)``.  When
        the pipeline fails at any stage, returns
        ``(last_H_init2current.copy(), 0)``.
    """
    # 1. Extract contours (skip tiny fragments).
    contours_cv, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    contours_cv = [c for c in contours_cv if c.shape[0] > 20]
    contours = [
        rearrange(c, 'N 1 xy -> xy N', xy=2) for c in contours_cv
    ]

    if not contours:
        return last_H_init2current.copy(), 0

    # 2. Mark boundary-adjacent contour points (shifted ±1 in each direction
    #    to catch points right at the edge).
    on_boundaries = [
        np.logical_or.reduce((
            _approx_touching_edges(c + np.array([[0], [1]]), mask.shape, margin=1),
            _approx_touching_edges(c + np.array([[0], [-1]]), mask.shape, margin=1),
            _approx_touching_edges(c + np.array([[1], [0]]), mask.shape, margin=1),
            _approx_touching_edges(c + np.array([[-1], [0]]), mask.shape, margin=1),
        ))
        for c in contours
    ]

    # 3. Hough transform -> top-k dominant lines.
    hough_lines = contour_hough(
        mask, contours, on_boundaries,
        gradient_step=gradient_step, topk=topk_lines,
    )

    # 4. Refine line positions via SVD fitting.
    refined_lines, line_pts = refine_lines(
        hough_lines, contours, on_boundaries,
        gradient_step=gradient_step,
        use_orthogonal_fit=line_refinement_method == "orthogonal_tls",
    )

    # 5. Filter out lines with zero support.
    has_support = np.array([pts.shape[1] > 0 for pts in line_pts])
    if not np.any(has_support):
        return last_H_init2current.copy(), 0
    refined_lines = refined_lines[:, has_support]
    line_pts = [p for p, s in zip(line_pts, has_support) if s]

    # 6. Filter out short line segments.
    endpoints = [
        _line_to_segment(refined_lines[:, i], line_pts[i])
        for i in range(refined_lines.shape[1])
    ]
    short = np.array([_segment_length(ep) < 5 for ep in endpoints])
    refined_lines = refined_lines[:, ~short]

    if refined_lines.shape[1] < 2:
        return last_H_init2current.copy(), 0

    # 7. Intersect consecutive line pairs -> candidate corner points.
    intersections_h, parents = lines_to_points(refined_lines)
    intersections = p2e(intersections_h)

    # 8. Keep only plausible intersections close to the mask contour.
    max_distance_px = float(corner_max_distance_px)
    if corner_max_distance_image_fraction is not None:
        max_distance_px = max(
            max_distance_px,
            float(corner_max_distance_image_fraction) * float(max(mask.shape)),
        )
    valid_pts, valid_mask = keep_only_points_close_to_contours(
        intersections, contours,
        max_distance_px=max_distance_px,
        parents=parents,
        lines=refined_lines,
        min_angle_deg=min_intersection_angle_deg,
    )
    valid_parents = [p for p, v in zip(parents, valid_mask) if v]

    if valid_pts.shape[1] == 0:
        return last_H_init2current.copy(), 0

    # 9. Enumerate all possible point correspondences.
    matches = bruteforce_pointset_matching(init_corners, valid_pts)
    matches = discard_diagonal_matches(
        matches, valid_parents, init_corners.shape[1],
    )

    if not matches:
        return last_H_init2current.copy(), 0

    # 10. Pick the homography closest to the last tracking state.
    best_H, best_diff, best_ids = find_closest_H(
        init_corners, valid_pts, last_H_init2current, matches,
    )
    n_correspondences = len(best_ids[0])

    return best_H, n_correspondences


# ---------------------------------------------------------------------------
# Symmetry resolution  (cyclic corner shift disambiguation via appearance)
# ---------------------------------------------------------------------------

class DINOFeatureExtractor:
    """Extract DINOv2 patch features for appearance comparison.

    Uses the DINOv2-small model with registers.  Input images must be
    exactly (224, 224, 3) BGR uint8.  Output is an L2-normalised feature
    map of shape (16, 16, 384).
    """

    def __init__(self, model_name: str = "dinov2_vits14_reg"):
        # Pin ref to skip torch.hub's default-branch detection, which does a
        # live GitHub urlopen even when the repo + weights are already cached
        # (offline compute nodes return HTTP 504 there).
        model = torch.hub.load(
            "facebookresearch/dinov2:main",
            model_name,
            force_reload=False,
            trust_repo=True,
            skip_validation=True,
        )
        model.eval()
        model.cuda()
        model.requires_grad_(False)
        self.model = model

    def extract(self, img_bgr: np.ndarray, layer: int = 11) -> torch.Tensor:
        """Extract normalised patch features.

        Args:
            img_bgr: (224, 224, 3) BGR uint8 image.
            layer: DINOv2 transformer layer to read (0-11).

        Returns:
            (16, 16, 384) L2-normalised feature tensor (CPU, float32).
        """
        assert img_bgr.shape == (224, 224, 3), (
            f"Expected (224, 224, 3), got {img_bgr.shape}"
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t = normalize(
            to_tensor(img_rgb).unsqueeze(0),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        feat_maps = self.model.get_intermediate_layers(img_t.cuda(), n=layer)
        feat = rearrange(
            feat_maps[0], '1 (H W) D -> H W D', H=16, W=16, D=384,
        )
        return F.normalize(feat.cpu(), p=2, dim=2)


def _corners_to_mask(corners: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Convert (xy, 4) corners to (H, W) bool mask."""
    poly = rearrange(corners, 'xy N -> N 1 xy', xy=2).astype(np.int32)
    mask = np.zeros(hw, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask > 0


def get_shifted_homographies(
    src_corners: np.ndarray,
    dst_corners: np.ndarray,
) -> List[Optional[np.ndarray]]:
    """Generate homographies for all cyclic shifts of destination corners.

    For a quadrilateral (N=4), this produces 4 homographies corresponding
    to the 4 possible corner label assignments.  Shift 0 is the identity
    assignment; shift *k* maps ``src_corner[i]`` to ``dst_corner[(i+k) % N]``.

    Args:
        src_corners: (xy, N) template corners.
        dst_corners: (xy, N) detected corners in the current frame.

    Returns:
        List of N homographies (or None where estimation failed).
    """
    N = dst_corners.shape[1]
    result: List[Optional[np.ndarray]] = [None] * N
    for shift in range(N):
        shifted = np.roll(dst_corners, shift, axis=1)
        H, _ = cv2.findHomography(
            rearrange(src_corners, 'xy N -> N 1 xy', xy=2),
            rearrange(shifted, 'xy N -> N 1 xy', xy=2),
        )
        if H is not None and np.linalg.matrix_rank(H) == 3:
            result[shift] = H
    return result


def score_shifted_homographies(
    shifted_Hs: List[Optional[np.ndarray]],
    init_corners: np.ndarray,
    init_frame: np.ndarray,
    frame: np.ndarray,
    extractor: DINOFeatureExtractor,
    *,
    resolution_hw: Tuple[int, int] = (224, 224),
    dino_layer: int = 11,
) -> np.ndarray:
    """Score each shifted homography by DINO feature similarity.

    Warps both the current frame and the template into a canonical
    (224 x 224) view, extracts DINOv2 patch features, and computes
    cosine similarity.

    Args:
        shifted_Hs: Homographies from :func:`get_shifted_homographies`.
        init_corners: (xy, 4) template corners.
        init_frame: Template image (H, W, 3) BGR uint8.
        frame: Current image (H, W, 3) BGR uint8.
        extractor: :class:`DINOFeatureExtractor` instance.
        resolution_hw: Canonical view size (must be (224, 224) for DINO).
        dino_layer: DINOv2 layer index.

    Returns:
        (N,) score array (higher = better match, -inf for invalid shifts).
    """
    scores = np.full(len(shifted_Hs), -np.inf)
    tH, tW = resolution_hw
    target_corners = np.array(
        [[0, tW - 1, tW - 1, 0],
         [0, 0, tH - 1, tH - 1]],
        dtype=np.float32,
    )

    # Find the first valid shifted H to compute H_fit (current → canonical).
    H_fit = None
    current_mask = None
    for shifted_H in shifted_Hs:
        if shifted_H is None:
            continue
        current_corners = H_warp_pts(shifted_H, init_corners)
        H_fit, _ = cv2.findHomography(
            rearrange(current_corners, 'xy N -> N 1 xy', xy=2),
            rearrange(target_corners, 'xy N -> N 1 xy', xy=2),
        )
        if H_fit is not None:
            canonical_corners = H_warp_pts(
                H_fit @ shifted_H, init_corners,
            )
            current_mask = _corners_to_mask(canonical_corners, resolution_hw)
            break

    if H_fit is None:
        return scores

    # Warp current frame to canonical view and extract features.
    fitted_frame = cv2.warpPerspective(
        frame, H_fit, (tW, tH),
    )
    fitted_frame[~current_mask] = 0
    with torch.no_grad():
        current_features = extractor.extract(fitted_frame, dino_layer)

    # Score each shift by warping the template and comparing.
    for i, shifted_H in enumerate(shifted_Hs):
        if shifted_H is None:
            continue
        H_init_to_canonical = H_fit @ shifted_H
        warped_init = cv2.warpPerspective(
            init_frame, H_init_to_canonical, (tW, tH),
        )
        warped_init[~current_mask] = 0

        with torch.no_grad():
            init_features = extractor.extract(warped_init, dino_layer)

        similarity = reduce(
            current_features * init_features,
            'H W D -> H W', reduction='sum', D=384,
        )
        scores[i] = float(similarity.mean())

    return scores


@dataclass
class SymmetryState:
    """Persistent state for temporal consistency of symmetry resolution."""

    best_shift: int = 0
    run_count: int = 0
    previously_not_enough: bool = False


def resolve_symmetry(
    H_init2current: np.ndarray,
    init_corners: np.ndarray,
    init_frame: np.ndarray,
    frame: np.ndarray,
    n_correspondences: int,
    state: SymmetryState,
    extractor: DINOFeatureExtractor,
    *,
    only_when_lost: bool = True,
    n_correspondences_enough: int = 4,
    resolution_hw: Tuple[int, int] = (224, 224),
    dino_layer: int = 11,
    abs_thr: float = 0.9,
    fts_thr: float = 0.2,
    run_length: int = 5,
) -> Tuple[np.ndarray, bool]:
    """Resolve cyclic corner ambiguity using appearance.

    Uses DINOv2 features to determine which cyclic shift of the corner
    correspondences best matches the template.  Includes temporal
    consistency: a shift must be confirmed by either (a) high absolute
    score *and* margin over the runner-up, or (b) consistent agreement
    over several consecutive frames.

    Updates *state* in-place.

    Args:
        H_init2current: (3, 3) candidate homography (init -> current).
        init_corners: (xy, 4) template corner coordinates.
        init_frame: Template image (H, W, 3) BGR uint8.
        frame: Current image (H, W, 3) BGR uint8.
        n_correspondences: Number of point matches from the Hough pipeline.
        state: Mutable :class:`SymmetryState` (persists across frames).
        extractor: :class:`DINOFeatureExtractor` instance.
        only_when_lost: Only run when recovering from lost state.
        n_correspondences_enough: Correspondence count considered "enough".
        resolution_hw: Canonical view size for DINO.
        dino_layer: DINOv2 transformer layer.
        abs_thr: Minimum absolute score for immediate confirmation.
        fts_thr: Minimum margin over second-best for immediate confirmation.
        run_length: Consecutive frames needed for run-based confirmation.

    Returns:
        Tuple of ``(resolved_H, shift_confirmed)``.
    """
    if only_when_lost:
        if n_correspondences < n_correspondences_enough:
            state.previously_not_enough = True
            state.best_shift = 0
            state.run_count = 0
            return H_init2current, False

        if not state.previously_not_enough:
            return H_init2current, False

        # Recovering: enough correspondences after being lost.

    # Generate all cyclic shifts and score them.
    current_corners = H_warp_pts(H_init2current, init_corners)
    shifted_Hs = get_shifted_homographies(init_corners, current_corners)

    scores = score_shifted_homographies(
        shifted_Hs, init_corners, init_frame, frame, extractor,
        resolution_hw=resolution_hw, dino_layer=dino_layer,
    )

    ranked = np.argsort(scores)[::-1]
    best_i = int(ranked[0])
    second_best_i = int(ranked[1])
    best_score = scores[best_i]
    margin = scores[best_i] - scores[second_best_i]

    # Temporal consistency: track run of the same best shift.
    if best_i != state.best_shift:
        state.best_shift = best_i
        state.run_count = 1
    else:
        state.run_count += 1

    # Confirmation criteria.
    shift_confirmed = False
    if best_score > abs_thr and margin > fts_thr:
        shift_confirmed = True
    if state.run_count > run_length and best_score > 0:
        shift_confirmed = True

    if shift_confirmed:
        if only_when_lost:
            state.previously_not_enough = False
        if shifted_Hs[best_i] is not None:
            return shifted_Hs[best_i], True

    return H_init2current, False
