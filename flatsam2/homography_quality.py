"""Homography quality estimation via mask-boundary consistency.

Scores and explains a candidate homography H by comparing:
  - ∂A' : boundary of the initial-frame reference mask warped to the current
          frame by H.  The warp is applied *directly to contour points*, not
          by rasterizing the warped mask.
  - ∂B  : boundary of the segmentation mask in the current frame.

Per-point classification (ALIGNED / OCCLUDED / INTERIOR for ∂A';
ALIGNED / EXCESS / OCCL_EDGE for ∂B) is computed from the signed distance
field of each mask.  Curvature-weighted aggregation yields a rich
explanation struct.  See ``homography_mask_explanation_score_plan.txt`` for
the full algorithm description.

Typical usage::

    ref = ReferenceContour.build(init_mask)
    result = assess_homography(ref, seg_mask, homography, (h, w))

Phase 0 (``ReferenceContour.build``) is done once at initialisation.
Phase 1–6 run per frame inside ``assess_homography``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    profile
except NameError:
    profile = lambda f: f


# ---------------------------------------------------------------------------
# Constants / tuneable defaults
# ---------------------------------------------------------------------------

#: Curvature weight scale α in w[i] = 1 + α · |κ[i]|
_ALPHA: float = 10.0

#: σ_rel: σ = σ_rel · sqrt(area_A)   (alignment tolerance)
_SIGMA_REL: float = 0.03

#: τ = σ / 3 — sharpness of inside/outside sigmoid transition
_TAU_FRAC: float = 1.0 / 3.0

#: Curvature evidence threshold for "sufficient evidence"
_EVIDENCE_THRESHOLD: float = 0.15

#: Contradictory penalty weight γ
_GAMMA: float = 0.5

#: Smoothing sigma (in points) for curvature estimation
_CURV_SMOOTH_SIGMA: float = 2.0

#: Target spacing between sampled contour points (pixels)
_CONTOUR_SAMPLE_SPACING: float = 2.0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PointClassA(Enum):
    """Classification of a ∂A' boundary point."""
    ALIGNED = auto()
    OCCLUDED = auto()     # outside B — occluded / out-of-reach
    INTERIOR = auto()     # deep inside B — contradictory (B larger than A')
    OUT_OF_FRAME = auto()


class PointClassB(Enum):
    """Classification of a ∂B boundary point."""
    ALIGNED = auto()
    EXCESS = auto()       # outside A' — segmenter excess / error
    OCCL_EDGE = auto()    # deep inside A' — occlusion boundary
    FRAME_EDGE = auto()   # touching image border — out-of-frame clip


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A contiguous run of contour points sharing a dominant classification.

    Attributes:
        label: Dominant class of this segment.
        point_indices: Indices into the parent contour point array.
        arc_length: Sum of inter-point distances within the segment (px).
        total_weight: Σ w[i] over all points in the segment.
        mean_distance: Mean unsigned distance to the other boundary for
            ALIGNED segments; ``nan`` otherwise.
        representative: (x, y) midpoint of the segment (for visualisation).
        points: (N, 2) float32 array of (x, y) positions (current frame).
    """
    label: PointClassA | PointClassB
    point_indices: List[int]
    arc_length: float
    total_weight: float
    mean_distance: float
    representative: Tuple[float, float]
    points: np.ndarray  # (N, 2)


@dataclass
class HomographyExplanation:
    """Full, explainable result of a homography quality assessment.

    All spatial outputs use the current-frame coordinate system.

    Attributes:
        confidence: Overall confidence that H is correct — 0..1.  This is
            the primary score for hypothesis comparison.
        precision: Mean boundary alignment distance (px) on the aligned
            portion of ∂A'.  Lower = tighter alignment.
        frac_aligned_A: Curvature-weighted fraction of ∂A' classified ALIGNED.
        frac_occluded_A: Curvature-weighted fraction classified OCCLUDED.
        frac_interior_A: Curvature-weighted fraction classified INTERIOR
            (contradictory — B is larger than A' here).
        frac_out_of_frame_A: Fraction of ∂A' points outside the image frame.
        frac_aligned_B: Fraction of ∂B points classified ALIGNED.
        frac_excess_B: Fraction classified EXCESS (segmenter overshoot).
        frac_occl_edge_B: Fraction classified OCCL_EDGE (occlusion boundary).
        frac_frame_edge_B: Fraction classified FRAME_EDGE.
        segments_A: Ordered spatial segments of ∂A' with dominant class.
        segments_B: Ordered spatial segments of ∂B with dominant class.
        occlusion_mask: Rasterized A' AND NOT B (occluded region), uint8 0/255.
        excess_B_mask: Rasterized B AND NOT A' (segmenter excess), uint8 0/255.
        alignment_loss: ``-log(max(confidence, 1e-6))`` — suitable for
            gradient-based homography refinement.
        explanation: Human-readable summary string.
    """
    confidence: float = 0.0
    precision: float = float("inf")
    frac_aligned_A: float = 0.0
    frac_occluded_A: float = 0.0
    frac_interior_A: float = 0.0
    frac_out_of_frame_A: float = 0.0
    frac_aligned_B: float = 0.0
    frac_excess_B: float = 0.0
    frac_occl_edge_B: float = 0.0
    frac_frame_edge_B: float = 0.0
    segments_A: List[Segment] = field(default_factory=list)
    segments_B: List[Segment] = field(default_factory=list)
    occlusion_mask: Optional[np.ndarray] = None
    excess_B_mask: Optional[np.ndarray] = None
    alignment_loss: float = float("inf")
    explanation: str = ""


# ---------------------------------------------------------------------------
# Phase 0 — reference contour (computed once at initialisation)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceContour:
    """Pre-computed contour of the initial-frame reference mask.

    Created once via :meth:`build`.  Passed to :func:`assess_homography`
    every frame — the points, curvatures and weights never change.

    Attributes:
        points: (N, 2) float32 array of (x, y) contour points sampled at
            roughly ``sample_spacing``-px intervals along the boundary.
        curvature: (N,) float32 unsigned curvature at each point.
        weights: (N,) float32 distinctiveness weight w[i] = 1 + α·|κ[i]|.
        area: Pixel area of the reference mask (used to set σ adaptively).
        sample_spacing: Approximate arc-length spacing between points (px).
        mask: (H, W) uint8 original binary mask (stored for NN preprocessing).
        tangent_sin: (N,) float32 sin(θ_i) at each contour point.
        tangent_cos: (N,) float32 cos(θ_i) at each contour point.
    """
    points: np.ndarray      # (N, 2) float32
    curvature: np.ndarray   # (N,) float32
    weights: np.ndarray     # (N,) float32
    area: float
    sample_spacing: float = _CONTOUR_SAMPLE_SPACING
    mask: Optional[np.ndarray] = None           # (H, W) uint8
    tangent_sin: Optional[np.ndarray] = None    # (N,) float32
    tangent_cos: Optional[np.ndarray] = None    # (N,) float32
    boundary_pts: Optional[np.ndarray] = None   # (M, 2) float32 — all raw boundary pixels

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        mask: np.ndarray,
        sample_spacing: float = _CONTOUR_SAMPLE_SPACING,
        alpha: float = _ALPHA,
        curv_smooth_sigma: float = _CURV_SMOOTH_SIGMA,
    ) -> "ReferenceContour":
        """Build a ReferenceContour from a binary mask.

        Args:
            mask: uint8 binary mask (0 / non-zero), shape (H, W).
            sample_spacing: Target arc-length spacing between sampled points
                (pixels).
            alpha: Curvature weight scale (w = 1 + alpha * |κ|).
            curv_smooth_sigma: Gaussian smoothing sigma applied to the raw
                curvature estimate (in points) before weighting.

        Returns:
            A :class:`ReferenceContour` ready for use with
            :func:`assess_homography`.

        Raises:
            ValueError: If the mask is empty.
        """
        binary = (mask > 0).astype(np.uint8)
        area = float(np.count_nonzero(binary))
        if area == 0:
            raise ValueError("Reference mask is empty.")

        raw_pts = _extract_and_sample_contour(binary, sample_spacing)
        if len(raw_pts) < 5:
            raise ValueError(
                f"Reference mask boundary too short ({len(raw_pts)} points)."
            )

        curv = _compute_curvature(raw_pts, smooth_sigma=curv_smooth_sigma)
        weights = 1.0 + alpha * np.abs(curv)
        weights = weights.astype(np.float32)

        tan_sin, tan_cos = _compute_tangent_directions(raw_pts)
        boundary_pts = _extract_raw_boundary(binary)

        return cls(
            points=raw_pts,
            curvature=curv,
            weights=weights,
            area=area,
            sample_spacing=sample_spacing,
            mask=binary * 255,
            tangent_sin=tan_sin,
            tangent_cos=tan_cos,
            boundary_pts=boundary_pts,
        )


    def get_nn_points(
        self, n_points: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample contour to exactly n_points with tangent directions.

        Args:
            n_points: Number of output points.

        Returns:
            Tuple of (points, tangent_sin, tangent_cos), each arrays of
            shape (n_points, 2), (n_points,), (n_points,) respectively.
        """
        spacing = _contour_perimeter(self.points) / n_points
        pts = _resample_at_spacing(self.points, spacing)
        # _resample_at_spacing may return slightly different count; force exact
        if len(pts) != n_points:
            pts = _resample_to_n_points(self.points, n_points)
        tan_sin, tan_cos = _compute_tangent_directions(pts)
        return pts, tan_sin, tan_cos


# ---------------------------------------------------------------------------
# Contour extraction and curvature helpers
# ---------------------------------------------------------------------------

def _extract_raw_boundary(binary: np.ndarray) -> np.ndarray:
    """Return all contour pixels of a binary mask as (M, 2) float32 (x, y).

    Uses CHAIN_APPROX_NONE so every boundary pixel is included — no
    sub-sampling.  All connected-component boundaries are concatenated.

    Args:
        binary: uint8 binary mask (0/1 or 0/255).

    Returns:
        (M, 2) float32 array, or shape (0, 2) for an empty mask.
    """
    mask_u8 = (binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(
        [c.squeeze(axis=1) for c in contours if len(c) >= 1],
        axis=0,
    ).astype(np.float32)


def _extract_and_sample_contour(
    binary: np.ndarray,
    spacing: float = _CONTOUR_SAMPLE_SPACING,
) -> np.ndarray:
    """Extract contour points from a binary mask at uniform arc-length spacing.

    All connected components are included.  Points are returned as a single
    (N, 2) float32 array of (x, y) coordinates.

    Args:
        binary: uint8 binary mask (0/1 or 0/255).
        spacing: Target arc-length spacing in pixels.

    Returns:
        (N, 2) float32 array.  Shape is (0, 2) for an empty mask.
    """
    mask_u8 = (binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    all_pts: List[np.ndarray] = []
    for cnt in contours:
        pts = cnt.squeeze(axis=1).astype(np.float32)  # (M, 2)
        if len(pts) < 3:
            continue
        resampled = _resample_at_spacing(pts, spacing)
        all_pts.append(resampled)

    if not all_pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(all_pts, axis=0)


def _resample_at_spacing(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed contour at uniform arc-length spacing.

    Args:
        pts: (M, 2) float32 contour points.
        spacing: Desired spacing in pixels.

    Returns:
        (N, 2) float32 resampled points.
    """
    # Segment lengths including the closing edge back to pts[0].
    # diffs has shape (M, 2) and seg_lens has shape (M,).
    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg_lens = np.linalg.norm(diffs, axis=1)

    # cum_len[i] = arc length from pts[0] to pts[i]; cum_len has M+1 entries
    # (the last entry is the full closed-contour perimeter).
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < spacing:
        return pts  # Too small to resample meaningfully

    n_out = max(3, int(round(total_len / spacing)))
    target = np.linspace(0.0, total_len, n_out, endpoint=False)

    # Extend pts by its first point so that xp (cum_len, M+1 entries) and
    # fp (pts_closed, M+1 entries) have matching lengths, and np.interp can
    # correctly interpolate the closing wrap-around segment.
    pts_closed = np.concatenate([pts, pts[:1]], axis=0)  # (M+1, 2)
    xs = np.interp(target, cum_len, pts_closed[:, 0])
    ys = np.interp(target, cum_len, pts_closed[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _contour_perimeter(pts: np.ndarray) -> float:
    """Compute the perimeter of a closed contour.

    Args:
        pts: (N, 2) float32 contour points.

    Returns:
        Total arc length (closed) in pixels.
    """
    diffs = np.diff(pts, axis=0, append=pts[:1])
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _resample_to_n_points(pts: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed contour to exactly n_points using arc-length interp.

    Args:
        pts: (M, 2) float32 contour points.
        n_points: Desired number of output points.

    Returns:
        (n_points, 2) float32 resampled points.
    """
    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.tile(pts[0], (n_points, 1)).astype(np.float32)

    target = np.linspace(0.0, total_len, n_points, endpoint=False)
    pts_closed = np.concatenate([pts, pts[:1]], axis=0)
    xs = np.interp(target, cum_len, pts_closed[:, 0])
    ys = np.interp(target, cum_len, pts_closed[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _compute_tangent_directions(
    pts: np.ndarray,
    smooth_sigma: float = _CURV_SMOOTH_SIGMA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute tangent directions at each point of a closed contour.

    Applies Gaussian smoothing before central differences to suppress
    digitisation noise from pixel-grid contours.

    Args:
        pts: (N, 2) float32 contour points (closed, ordered).
        smooth_sigma: Gaussian smoothing sigma in points.  Set to 0 to
            disable smoothing.

    Returns:
        Tuple of (tangent_sin, tangent_cos), each (N,) float32.
    """
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)

    if smooth_sigma > 0 and n >= 5:
        xs = gaussian_filter1d(pts[:, 0], sigma=smooth_sigma, mode="wrap")
        ys = gaussian_filter1d(pts[:, 1], sigma=smooth_sigma, mode="wrap")
    else:
        xs, ys = pts[:, 0], pts[:, 1]

    idx_next = np.arange(1, n + 1) % n
    idx_prev = np.arange(-1, n - 1) % n
    dx = xs[idx_next] - xs[idx_prev]
    dy = ys[idx_next] - ys[idx_prev]
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
    tangent_cos = (dx / norm).astype(np.float32)
    tangent_sin = (dy / norm).astype(np.float32)
    return tangent_sin, tangent_cos


def _compute_curvature(
    pts: np.ndarray,
    smooth_sigma: float = _CURV_SMOOTH_SIGMA,
) -> np.ndarray:
    """Estimate unsigned curvature κ at each point of a closed contour.

    Uses the tangent-angle change rate (dθ/ds) with Gaussian pre-smoothing.

    Args:
        pts: (N, 2) float32 contour points (closed, ordered).
        smooth_sigma: Gaussian smoothing sigma (in points) applied before
            finite differences.

    Returns:
        (N,) float32 unsigned curvature values.
    """
    n = len(pts)
    if n < 5:
        return np.zeros(n, dtype=np.float32)

    # Smooth the contour coordinates to suppress digitization noise
    xs = gaussian_filter1d(pts[:, 0], sigma=smooth_sigma, mode="wrap")
    ys = gaussian_filter1d(pts[:, 1], sigma=smooth_sigma, mode="wrap")
    smoothed = np.stack([xs, ys], axis=1)

    # Tangent via central differences
    p_next = smoothed[np.arange(1, n + 1) % n]
    p_prev = smoothed[np.arange(-1, n - 1) % n]
    tangents = p_next - p_prev  # (N, 2)

    # Tangent angle
    theta = np.arctan2(tangents[:, 1], tangents[:, 0])

    # Arc-length element (from original points, not smoothed)
    seg_lens = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
    seg_lens = np.maximum(seg_lens, 1e-6)

    # dθ/ds — wrap angle differences to [-π, π]
    dtheta = np.angle(np.exp(1j * np.diff(theta, append=theta[:1])))
    curv = np.abs(dtheta) / seg_lens
    return curv.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 helpers — SDF and UDT computation
# ---------------------------------------------------------------------------

@profile
def _compute_sdf(mask_u8: np.ndarray) -> np.ndarray:
    """Compute the signed distance field of a binary mask.

    Sign convention (matching the plan):
      - **positive** outside the mask (far from boundary)
      - **negative** inside the mask

    Args:
        mask_u8: uint8 binary mask (0 / 255), shape (H, W).

    Returns:
        float32 SDF array of the same shape.
    """
    if mask_u8.max() == 0:
        # Everything is "outside" an empty mask
        return np.full(mask_u8.shape, 1e6, dtype=np.float32)

    fg = (mask_u8 > 127).astype(np.uint8)
    bg = 1 - fg

    # Distance to boundary from outside (positive side)
    dt_outside = cv2.distanceTransform(
        bg * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    ).astype(np.float32)
    # Distance to boundary from inside (will be negated)
    dt_inside = cv2.distanceTransform(
        fg * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
    ).astype(np.float32)

    sdf = np.where(fg, -dt_inside, dt_outside)
    return sdf


def _compute_udt_from_sdf(sdf: np.ndarray) -> np.ndarray:
    """Return unsigned distance to boundary from a signed distance field.

    Args:
        sdf: float32 SDF (positive outside, negative inside).

    Returns:
        float32 unsigned boundary distance (always ≥ 0).
    """
    return np.abs(sdf)


def _bilinear_sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2-D field at sub-pixel (x, y) locations.

    Args:
        field: (H, W) float32 array.
        pts: (N, 2) float32 array of (x, y) query points.

    Returns:
        (N,) float32 sampled values.  Out-of-bounds points receive 1e6.
    """
    h, w = field.shape[:2]
    n = len(pts)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]

    oob = (x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)

    x = np.clip(x, 0, w - 1 - 1e-6)
    y = np.clip(y, 0, h - 1 - 1e-6)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x1 = np.minimum(x1, w - 1)
    y1 = np.minimum(y1, h - 1)

    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)

    v00 = field[y0, x0]
    v10 = field[y0, x1]
    v01 = field[y1, x0]
    v11 = field[y1, x1]

    val = (
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )
    val[oob] = 1e6
    return val.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 — per-frame setup
# ---------------------------------------------------------------------------

def _warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of (x, y) points.

    Args:
        pts: (N, 2) float32 source points.
        H: (3, 3) float64 homography matrix.

    Returns:
        (N, 2) float32 warped points.
    """
    if len(pts) == 0:
        return pts
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.concatenate([pts.astype(np.float64), ones], axis=1)  # (N, 3)
    warped_hom = (H @ hom.T).T  # (N, 3)
    warped = warped_hom[:, :2] / warped_hom[:, 2:3]
    return warped.astype(np.float32)


def _rasterize_warped_mask(
    ref: ReferenceContour,
    H: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterize the warped reference mask for SDF_A' computation (∂B side).

    This is the *only* place we rasterize A' — exclusively to compute SDF_A'
    for scoring ∂B points.  The ∂A' analysis uses the warped point cloud
    directly.

    Args:
        ref: Precomputed reference contour.
        H: (3, 3) homography matrix.
        image_shape: (height, width) of the output image.

    Returns:
        uint8 mask of shape ``image_shape``, 0/255.
    """
    h, w = image_shape
    # Use the stored mask if available (better quality — preserves fine detail)
    if ref.mask is not None:
        return cv2.warpPerspective(
            ref.mask, H, (w, h),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
    # Fallback: warp contour points and fill polygon
    pts_warped = _warp_points(ref.points, H)
    canvas = np.zeros((h, w), dtype=np.uint8)
    if len(pts_warped) >= 3:
        poly = np.round(pts_warped).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly], 255)
    return canvas


# ---------------------------------------------------------------------------
# Phase 2 — per-point soft classification
# ---------------------------------------------------------------------------

def _classify_dA_prime(
    sdf_B_at_pts: np.ndarray,
    udt_B_at_pts: np.ndarray,
    sigma: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-classify ∂A' points as ALIGNED / OCCLUDED / INTERIOR.

    Args:
        sdf_B_at_pts: (N,) SDF of mask B sampled at warped ∂A' points
            (positive outside B, negative inside).
        udt_B_at_pts: (N,) unsigned boundary distance of mask B.
        sigma: Alignment tolerance (px).
        tau: Inside/outside sigmoid sharpness (px).

    Returns:
        Tuple (s_aligned, s_occluded, s_interior) each shape (N,), summing to 1.
    """
    s_aligned = np.exp(-(udt_B_at_pts ** 2) / (sigma ** 2)).astype(np.float32)
    residual = 1.0 - s_aligned

    def _sigmoid(x: np.ndarray, t: float) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-x / t))).astype(np.float32)

    s_occluded = residual * _sigmoid(sdf_B_at_pts, tau)    # positive SDF = outside B
    s_interior = residual * _sigmoid(-sdf_B_at_pts, tau)   # negative SDF = inside B
    return s_aligned, s_occluded, s_interior


def _classify_dB(
    sdf_Ap_at_pts: np.ndarray,
    udt_Ap_at_pts: np.ndarray,
    sigma: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-classify ∂B points as ALIGNED / EXCESS / OCCL_EDGE.

    Args:
        sdf_Ap_at_pts: (N,) SDF of warped mask A' sampled at ∂B points.
        udt_Ap_at_pts: (N,) unsigned boundary distance of A'.
        sigma: Alignment tolerance.
        tau: Sigmoid sharpness.

    Returns:
        Tuple (s_aligned, s_excess, s_occl_edge) each shape (N,), summing to 1.
    """
    s_aligned = np.exp(-(udt_Ap_at_pts ** 2) / (sigma ** 2)).astype(np.float32)
    residual = 1.0 - s_aligned

    def _sigmoid(x: np.ndarray, t: float) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-x / t))).astype(np.float32)

    s_excess = residual * _sigmoid(sdf_Ap_at_pts, tau)      # outside A'
    s_occl_edge = residual * _sigmoid(-sdf_Ap_at_pts, tau)  # inside A'
    return s_aligned, s_excess, s_occl_edge


# ---------------------------------------------------------------------------
# Phase 4 — spatial segmentation
# ---------------------------------------------------------------------------

def _build_segments(
    pts: np.ndarray,
    dominant_class: np.ndarray,
    weights: np.ndarray,
    udt_at_pts: np.ndarray,
    class_enum,
    label_map: dict,
) -> List[Segment]:
    """Group consecutive contour points by dominant classification into Segments.

    Args:
        pts: (N, 2) float32 point coordinates.
        dominant_class: (N,) int array of dominant class index.
        weights: (N,) float32 per-point weights.
        udt_at_pts: (N,) unsigned boundary distance for each point.
        class_enum: The enum class (PointClassA or PointClassB).
        label_map: Mapping from dominant_class int index to class_enum value.

    Returns:
        List of :class:`Segment` objects.
    """
    n = len(pts)
    if n == 0:
        return []

    segments: List[Segment] = []
    start = 0

    while start < n:
        cls_idx = int(dominant_class[start])
        end = start + 1
        while end < n and int(dominant_class[end]) == cls_idx:
            end += 1

        seg_pts = pts[start:end]
        seg_w = weights[start:end]
        seg_udt = udt_at_pts[start:end]

        # Arc length
        if len(seg_pts) > 1:
            arc = float(np.sum(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1)))
        else:
            arc = 0.0

        total_w = float(np.sum(seg_w))
        lbl = label_map[cls_idx]

        is_aligned = lbl in (PointClassA.ALIGNED, PointClassB.ALIGNED)
        mean_dist = float(np.mean(seg_udt)) if is_aligned and len(seg_udt) > 0 else float("nan")

        mid_idx = (start + end) // 2
        rep = (float(pts[mid_idx, 0]), float(pts[mid_idx, 1]))

        segments.append(Segment(
            label=lbl,
            point_indices=list(range(start, end)),
            arc_length=arc,
            total_weight=total_w,
            mean_distance=mean_dist,
            representative=rep,
            points=seg_pts,
        ))
        start = end

    return segments


# ---------------------------------------------------------------------------
# Phase 5 — confidence score
# ---------------------------------------------------------------------------

def _compute_confidence(
    s_aligned_A: np.ndarray,
    s_occluded_A: np.ndarray,
    s_interior_A: np.ndarray,
    weights_A: np.ndarray,
    oof_mask: np.ndarray,
    udt_B_at_dA: np.ndarray,
    sigma: float,
    evidence_threshold: float = _EVIDENCE_THRESHOLD,
    gamma: float = _GAMMA,
) -> Tuple[float, float, float]:
    """Compute confidence, precision and curvature_evidence.

    Implements the revised Phase 5 formula from the plan.

    Args:
        s_aligned_A: (N,) soft ALIGNED scores for ∂A' points.
        s_occluded_A: (N,) soft OCCLUDED scores.
        s_interior_A: (N,) soft INTERIOR scores.
        weights_A: (N,) curvature weights w[i].
        oof_mask: (N,) bool — True where point is out of frame.
        udt_B_at_dA: (N,) unsigned boundary distance to ∂B.
        sigma: Alignment tolerance (px).
        evidence_threshold: Curvature evidence threshold for full confidence.
        gamma: Contradictory penalty weight.

    Returns:
        Tuple (confidence, precision_px, curvature_evidence).
    """
    in_frame = ~oof_mask  # (N,)

    # Visibility mask: in-frame AND not predominantly occluded
    visible_mask = in_frame & (s_occluded_A < 0.5)

    W_vis = float(np.sum(weights_A[visible_mask])) + 1e-12

    # Quality: fraction of visible boundary that aligns (curvature-weighted)
    vis_aligned = float(
        np.sum(weights_A[visible_mask] * s_aligned_A[visible_mask])
    ) / W_vis

    # Precision: mean alignment distance on aligned, visible points
    aligned_visible_w = weights_A[visible_mask] * s_aligned_A[visible_mask]
    total_aligned_w = float(np.sum(aligned_visible_w)) + 1e-12
    mean_align_dist = float(
        np.sum(aligned_visible_w * udt_B_at_dA[visible_mask])
    ) / total_aligned_w
    precision = float(np.exp(-(mean_align_dist ** 2) / (sigma ** 2)))

    quality = vis_aligned * precision

    # Curvature evidence: how much distinctive boundary is visible and aligned
    W_total_all = float(np.sum(weights_A)) + 1e-12
    W_vis_aligned = float(np.sum(weights_A[visible_mask] * s_aligned_A[visible_mask]))
    curvature_evidence = W_vis_aligned / W_total_all

    # Sufficiency
    sufficiency = float(np.clip(curvature_evidence / evidence_threshold, 0.0, 1.0))

    # Contradictory penalty (visible points classified INTERIOR)
    vis_contradictory = float(
        np.sum(weights_A[visible_mask] * s_interior_A[visible_mask])
    ) / W_vis
    penalty = gamma * vis_contradictory

    confidence = float(np.clip(quality - penalty, 0.0, 1.0)) * sufficiency

    return confidence, mean_align_dist, curvature_evidence


# ---------------------------------------------------------------------------
# Phase 3 — curvature-weighted aggregation
# ---------------------------------------------------------------------------

def _aggregate_dA(
    s_aligned: np.ndarray,
    s_occluded: np.ndarray,
    s_interior: np.ndarray,
    weights: np.ndarray,
    oof_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute curvature-weighted fractions for ∂A'.

    Args:
        s_aligned: (N,) soft aligned scores.
        s_occluded: (N,) soft occluded scores.
        s_interior: (N,) soft interior scores.
        weights: (N,) curvature weights.
        oof_mask: (N,) bool out-of-frame mask.

    Returns:
        (frac_aligned, frac_occluded, frac_interior, frac_oof) all in [0,1].
    """
    in_frame = ~oof_mask
    W_all = float(np.sum(weights)) + 1e-12
    W_inf = float(np.sum(weights[in_frame])) + 1e-12

    frac_oof = float(np.sum(weights[oof_mask])) / W_all

    frac_aligned = float(np.sum(weights[in_frame] * s_aligned[in_frame])) / W_inf * (1 - frac_oof)
    frac_occluded = float(np.sum(weights[in_frame] * s_occluded[in_frame])) / W_inf * (1 - frac_oof)
    frac_interior = float(np.sum(weights[in_frame] * s_interior[in_frame])) / W_inf * (1 - frac_oof)

    return frac_aligned, frac_occluded, frac_interior, frac_oof


def _aggregate_dB(
    s_aligned: np.ndarray,
    s_excess: np.ndarray,
    s_occl_edge: np.ndarray,
    frame_edge_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute unweighted fractions for ∂B.

    Args:
        s_aligned: (N,) soft aligned scores for non-FRAME_EDGE ∂B points.
        s_excess: (N,) soft excess scores.
        s_occl_edge: (N,) soft occl_edge scores.
        frame_edge_mask: (N_total,) bool, True for FRAME_EDGE points.

    Returns:
        (frac_aligned, frac_excess, frac_occl_edge, frac_frame_edge).
    """
    N_total = len(frame_edge_mask)
    N_non_fe = len(s_aligned)
    frac_frame_edge = float(np.sum(frame_edge_mask)) / (N_total + 1e-12)

    if N_non_fe == 0:
        return 0.0, 0.0, 0.0, frac_frame_edge

    frac_aligned = float(np.mean(s_aligned)) * (1 - frac_frame_edge)
    frac_excess = float(np.mean(s_excess)) * (1 - frac_frame_edge)
    frac_occl_edge = float(np.mean(s_occl_edge)) * (1 - frac_frame_edge)

    return frac_aligned, frac_excess, frac_occl_edge, frac_frame_edge


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assess_homography(
    ref: ReferenceContour,
    mask_B: np.ndarray,
    H: np.ndarray,
    image_shape: Tuple[int, int],
    sigma_rel: float = _SIGMA_REL,
    evidence_threshold: float = _EVIDENCE_THRESHOLD,
    gamma: float = _GAMMA,
) -> HomographyExplanation:
    """Score and explain a candidate homography using mask-boundary consistency.

    Implements Phases 1–6 of the algorithm described in
    ``homography_mask_explanation_score_plan.txt``.

    Args:
        ref: Pre-computed reference contour from :meth:`ReferenceContour.build`.
        mask_B: uint8 segmentation mask in the current frame, shape (H, W).
        H: (3, 3) float64 homography mapping initial-frame → current-frame.
        image_shape: ``(height, width)`` of the current frame.
        sigma_rel: Relative alignment tolerance (σ = σ_rel · sqrt(area_A)).
        evidence_threshold: Curvature evidence fraction needed for full
            confidence.
        gamma: Contradictory evidence penalty weight.

    Returns:
        :class:`HomographyExplanation` with all explainable fields populated.
    """
    img_h, img_w = image_shape

    # ------------------------------------------------------------------
    # Phase 1a: Warp ∂A points directly (no rasterization of A)
    pts_Ap = _warp_points(ref.points, H)  # (N, 2) in current frame

    # Phase 1b: Out-of-frame mask
    oof_mask = (
        (pts_Ap[:, 0] < 0) | (pts_Ap[:, 0] >= img_w)
        | (pts_Ap[:, 1] < 0) | (pts_Ap[:, 1] >= img_h)
    )  # (N,)

    # Phase 1c: SDF and UDT of mask B
    mask_B_u8 = (mask_B > 127).astype(np.uint8) * 255
    sdf_B = _compute_sdf(mask_B_u8)                 # positive outside B
    udt_B = _compute_udt_from_sdf(sdf_B)            # always >= 0

    # Phase 1d: Rasterize warped A' (only for SDF_A' — ∂B classification)
    mask_Ap_u8 = _rasterize_warped_mask(ref, H, image_shape)
    sdf_Ap = _compute_sdf(mask_Ap_u8)
    udt_Ap = _compute_udt_from_sdf(sdf_Ap)

    # Phase 1e: Extract ∂B contour points
    pts_B = _extract_and_sample_contour(
        (mask_B_u8 > 0).astype(np.uint8),
        ref.sample_spacing,
    )  # (M, 2)

    # Frame-edge mask for ∂B
    if len(pts_B) > 0:
        fe_mask_B = (
            (pts_B[:, 0] <= 0) | (pts_B[:, 0] >= img_w - 1)
            | (pts_B[:, 1] <= 0) | (pts_B[:, 1] >= img_h - 1)
        )
    else:
        fe_mask_B = np.array([], dtype=bool)

    # ------------------------------------------------------------------
    # Phase 2: Per-point soft classification

    # Adaptive sigma
    sigma = sigma_rel * float(np.sqrt(ref.area))
    sigma = max(sigma, 2.0)  # floor at 2px for tiny objects
    tau = sigma * _TAU_FRAC

    # ∂A' classification (in-frame points only; oof points are excluded)
    sdf_B_at_dAp = _bilinear_sample(sdf_B, pts_Ap)
    udt_B_at_dAp = _bilinear_sample(udt_B, pts_Ap)
    # Force oof points to "occluded" (outside B by large margin)
    sdf_B_at_dAp[oof_mask] = 1e6
    udt_B_at_dAp[oof_mask] = 1e6

    s_aligned_A, s_occluded_A, s_interior_A = _classify_dA_prime(
        sdf_B_at_dAp, udt_B_at_dAp, sigma, tau,
    )

    # ∂B classification (non-frame-edge points)
    if len(pts_B) > 0:
        non_fe = ~fe_mask_B
        pts_B_nfe = pts_B[non_fe]
        if len(pts_B_nfe) > 0:
            sdf_Ap_at_dB = _bilinear_sample(sdf_Ap, pts_B_nfe)
            udt_Ap_at_dB = _bilinear_sample(udt_Ap, pts_B_nfe)
            s_aligned_B, s_excess_B, s_occl_edge_B = _classify_dB(
                sdf_Ap_at_dB, udt_Ap_at_dB, sigma, tau,
            )
        else:
            s_aligned_B = s_excess_B = s_occl_edge_B = np.empty(0, dtype=np.float32)
    else:
        non_fe = np.array([], dtype=bool)
        s_aligned_B = s_excess_B = s_occl_edge_B = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Phase 3: Curvature-weighted aggregation

    frac_aligned_A, frac_occluded_A, frac_interior_A, frac_oof_A = _aggregate_dA(
        s_aligned_A, s_occluded_A, s_interior_A, ref.weights, oof_mask,
    )
    frac_aligned_B, frac_excess_B, frac_occl_edge_B, frac_frame_edge_B = _aggregate_dB(
        s_aligned_B, s_excess_B, s_occl_edge_B, fe_mask_B,
    )

    # ------------------------------------------------------------------
    # Phase 5: Confidence score

    confidence, mean_align_dist, curvature_evidence = _compute_confidence(
        s_aligned_A, s_occluded_A, s_interior_A,
        ref.weights, oof_mask, udt_B_at_dAp,
        sigma, evidence_threshold, gamma,
    )
    alignment_loss = float(-np.log(max(confidence, 1e-6)))

    # ------------------------------------------------------------------
    # Phase 4: Spatial segments (after point-level work is done)

    # ∂A' dominant class per point
    stack_A = np.stack([s_aligned_A, s_occluded_A, s_interior_A], axis=1)
    dom_A_idx = np.argmax(stack_A, axis=1)
    # Override oof points
    dom_A_idx[oof_mask] = 3  # index 3 = OUT_OF_FRAME
    label_map_A = {
        0: PointClassA.ALIGNED,
        1: PointClassA.OCCLUDED,
        2: PointClassA.INTERIOR,
        3: PointClassA.OUT_OF_FRAME,
    }
    segments_A = _build_segments(
        pts_Ap, dom_A_idx, ref.weights, udt_B_at_dAp, PointClassA, label_map_A,
    )

    # ∂B dominant class per point (full-length arrays)
    if len(pts_B) > 0:
        s_al_full = np.zeros(len(pts_B), dtype=np.float32)
        s_ex_full = np.zeros(len(pts_B), dtype=np.float32)
        s_oe_full = np.zeros(len(pts_B), dtype=np.float32)
        s_al_full[non_fe] = s_aligned_B
        s_ex_full[non_fe] = s_excess_B
        s_oe_full[non_fe] = s_occl_edge_B

        stack_B = np.stack([s_al_full, s_ex_full, s_oe_full], axis=1)
        dom_B_idx = np.argmax(stack_B, axis=1)
        dom_B_idx[fe_mask_B] = 3  # FRAME_EDGE
        label_map_B = {
            0: PointClassB.ALIGNED,
            1: PointClassB.EXCESS,
            2: PointClassB.OCCL_EDGE,
            3: PointClassB.FRAME_EDGE,
        }
        udt_Ap_at_dB_full = np.full(len(pts_B), float("nan"), dtype=np.float32)
        if len(pts_B_nfe) > 0:
            udt_Ap_at_dB_full[non_fe] = udt_Ap_at_dB
        weights_B_uniform = np.ones(len(pts_B), dtype=np.float32)
        segments_B = _build_segments(
            pts_B, dom_B_idx, weights_B_uniform, udt_Ap_at_dB_full,
            PointClassB, label_map_B,
        )
    else:
        segments_B = []

    # ------------------------------------------------------------------
    # Phase 6: Derived masks and output struct

    a_bool = mask_Ap_u8 > 127
    b_bool = mask_B_u8 > 127
    occlusion_mask = ((a_bool & ~b_bool).astype(np.uint8)) * 255
    excess_B_mask = ((b_bool & ~a_bool).astype(np.uint8)) * 255

    explanation = _build_explanation(
        confidence=confidence,
        mean_align_dist=mean_align_dist,
        curvature_evidence=curvature_evidence,
        frac_aligned_A=frac_aligned_A,
        frac_occluded_A=frac_occluded_A,
        frac_interior_A=frac_interior_A,
        frac_oof_A=frac_oof_A,
        frac_excess_B=frac_excess_B,
        frac_frame_edge_B=frac_frame_edge_B,
        sigma=sigma,
    )

    return HomographyExplanation(
        confidence=confidence,
        precision=mean_align_dist,
        frac_aligned_A=frac_aligned_A,
        frac_occluded_A=frac_occluded_A,
        frac_interior_A=frac_interior_A,
        frac_out_of_frame_A=frac_oof_A,
        frac_aligned_B=frac_aligned_B,
        frac_excess_B=frac_excess_B,
        frac_occl_edge_B=frac_occl_edge_B,
        frac_frame_edge_B=frac_frame_edge_B,
        segments_A=segments_A,
        segments_B=segments_B,
        occlusion_mask=occlusion_mask,
        excess_B_mask=excess_B_mask,
        alignment_loss=alignment_loss,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Explanation string builder
# ---------------------------------------------------------------------------

def _build_explanation(
    confidence: float,
    mean_align_dist: float,
    curvature_evidence: float,
    frac_aligned_A: float,
    frac_occluded_A: float,
    frac_interior_A: float,
    frac_oof_A: float,
    frac_excess_B: float,
    frac_frame_edge_B: float,
    sigma: float,
) -> str:
    """Build a human-readable explanation string.

    Args:
        confidence: Overall confidence score.
        mean_align_dist: Mean alignment distance on aligned visible boundary (px).
        curvature_evidence: Fraction of curvature-weighted boundary that is
            aligned and visible.
        frac_aligned_A: Fraction of ∂A' classified aligned.
        frac_occluded_A: Fraction classified occluded.
        frac_interior_A: Fraction classified interior (contradictory).
        frac_oof_A: Fraction out of frame.
        frac_excess_B: Fraction of ∂B classified excess.
        frac_frame_edge_B: Fraction of ∂B on the frame edge.
        sigma: Alignment tolerance used (px).

    Returns:
        Multi-sentence explanation string.
    """
    parts = []

    if confidence > 0.85:
        verdict = "Strong alignment — homography very likely correct."
    elif confidence > 0.6:
        verdict = "Good alignment — homography probably correct."
    elif confidence > 0.35:
        verdict = "Moderate alignment — homography may be imprecise."
    elif confidence > 0.1:
        verdict = "Weak alignment — homography suspect."
    else:
        verdict = "No meaningful alignment — homography likely wrong."
    parts.append(verdict)

    parts.append(
        f"Confidence={confidence:.3f}, "
        f"mean_align_dist={mean_align_dist:.1f}px (σ={sigma:.1f}px), "
        f"curvature_evidence={curvature_evidence:.3f}."
    )

    # ∂A' breakdown
    parts.append(
        f"∂A': aligned={frac_aligned_A:.2f}, occluded={frac_occluded_A:.2f}, "
        f"interior={frac_interior_A:.2f}, out-of-frame={frac_oof_A:.2f}."
    )

    if frac_occluded_A > 0.4:
        parts.append(
            f"{frac_occluded_A:.0%} of reference boundary is occluded or outside B."
        )
    if frac_interior_A > 0.2:
        parts.append(
            f"{frac_interior_A:.0%} of ∂A' is deep inside B — "
            f"contradictory signal (segmenter excess or H error)."
        )
    if frac_oof_A > 0.1:
        parts.append(
            f"{frac_oof_A:.0%} of ∂A' is outside the image frame."
        )
    if frac_excess_B > 0.25:
        parts.append(
            f"{frac_excess_B:.0%} of ∂B is excess (outside A') — "
            f"possible segmenter overshoot."
        )
    if frac_frame_edge_B > 0.05:
        parts.append(
            f"{frac_frame_edge_B:.0%} of ∂B touches the frame border "
            f"(object may extend out of view)."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Convenience: recover homography from corner correspondences
# ---------------------------------------------------------------------------

def homography_from_corners(
    init_corners: np.ndarray,
    pred_corners: np.ndarray,
) -> Optional[np.ndarray]:
    """Recover a 3×3 homography from init and predicted corner coordinates.

    Args:
        init_corners: (2, 4) array — columns are [x, y] of corners in the
            initial frame.
        pred_corners: (2, 4) array — columns are [x, y] of corners in the
            current frame.

    Returns:
        3×3 float64 homography matrix, or None if the computation fails.
    """
    src = init_corners.T.astype(np.float64)  # (4, 2)
    dst = pred_corners.T.astype(np.float64)  # (4, 2)
    H, _ = cv2.findHomography(src, dst)
    return H
