"""Shared preprocessing pipeline for NN homography quality estimation.

Implements Steps 1-6 from the design plan, producing inputs for both
Option E (boundary transformer) and Option C (SDF-crop CNN).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist

try:
    profile
except NameError:
    profile = lambda f: f

from ..homography_quality import (
    ReferenceContour,
    _bilinear_sample,
    _compute_sdf,
    _compute_tangent_directions,
    _resample_to_n_points,
    _warp_points,
)


# ---------------------------------------------------------------------------
# SDF helpers
# ---------------------------------------------------------------------------

@profile
def _compute_sdf_fast(mask_u8: np.ndarray, downsample: int) -> np.ndarray:
    """Compute SDF at reduced resolution and upsample back.

    Runs distance transforms on a (H/downsample × W/downsample) image then
    bilinearly upsamples and rescales values.  Introduces at most ~downsample
    px error — acceptable for training where thresholds are 15–30 px.

    Args:
        mask_u8: (H, W) uint8 binary mask (0 / 255).
        downsample: Integer downsampling factor (1 = no downsampling).

    Returns:
        float32 SDF array, same shape as mask_u8.
    """
    if downsample <= 1:
        return _compute_sdf(mask_u8)
    h, w = mask_u8.shape
    nh, nw = max(1, round(h / downsample)), max(1, round(w / downsample))
    small = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
    sdf_small = _compute_sdf(small)
    sdf_full = cv2.resize(sdf_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return (sdf_full * downsample).astype(np.float32)


def _sparse_signed_dist(
    query_pts: np.ndarray,
    boundary_pts: np.ndarray,
    is_inside: np.ndarray,
) -> np.ndarray:
    """Signed distance from each query point to its nearest boundary point.

    Avoids a full distance transform — only useful when the number of query
    points is small (e.g. 128) relative to the image size.

    Args:
        query_pts: (N, 2) float32 (x, y) points to evaluate.
        boundary_pts: (M, 2) float32 boundary pixel coordinates.
        is_inside: (N,) bool — True if the query point lies inside the mask.

    Returns:
        (N,) float32 signed distances (negative inside, positive outside).
    """
    if len(boundary_pts) == 0:
        return np.full(len(query_pts), 1e6, dtype=np.float32)
    dists = cdist(query_pts, boundary_pts).min(axis=1)
    signs = np.where(is_inside, -1.0, 1.0)
    return (signs * dists).astype(np.float32)



# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def _extract_contour_data(
    mask_u8: np.ndarray,
    n_points: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract contour from a binary mask in one findContours call.

    Returns both the full raw boundary pixels (for sparse SDF queries) and
    the resampled fixed-N point set (for the NN input).

    Args:
        mask_u8: (H, W) uint8 binary mask (0/255).
        n_points: Number of resampled output points.

    Returns:
        (raw_boundary, resampled_pts) where:
          raw_boundary: (M, 2) float32 — all boundary pixels concatenated,
                        or None if no contour found.
          resampled_pts: (n_points, 2) float32 — proportionally resampled,
                         or None if no contour found.
    """
    binary = (mask_u8 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None, None

    valid = [c.squeeze(axis=1).astype(np.float32)
             for c in contours if len(c) >= 3]
    if not valid:
        return None, None

    raw_boundary = np.concatenate(valid, axis=0)  # (M, 2)

    lengths = np.array([cv2.arcLength(c.astype(np.float32), closed=True)
                        for c in valid], dtype=np.float64)
    total_len = lengths.sum()
    if total_len < 1e-6:
        return None, None

    # Largest-remainder allocation: floor first, then distribute leftover
    # slots to contours with the biggest fractional parts.  Each contour
    # gets at least 1 point; excess is removed from the largest allocations.
    raw = lengths / total_len * n_points
    alloc = np.maximum(1, np.floor(raw).astype(int))
    remaining = n_points - alloc.sum()
    if remaining > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]  # largest remainder first
        for idx in order:
            if remaining == 0:
                break
            alloc[idx] += 1
            remaining -= 1
    elif remaining < 0:
        # np.maximum(1,...) pushed total above n_points; take back from the
        # largest allocations (visit repeatedly until balanced).
        while remaining < 0:
            alloc[np.argmax(alloc)] -= 1
            remaining += 1

    pieces = [_resample_to_n_points(c, int(k)) for c, k in zip(valid, alloc)]
    resampled_pts = np.concatenate(pieces, axis=0)

    return raw_boundary, resampled_pts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@profile
def preprocess(
    ref: ReferenceContour,
    mask_B: np.ndarray,
    H: np.ndarray,
    image_shape: Tuple[int, int],
    n_points: int = 128,
    sdf_downsample: int = 1,
    architecture: str = "both",
) -> Optional[dict]:
    """Run the shared preprocessing pipeline (Steps 1-6).

    For ``architecture="option_e"`` the two full distance transforms and the
    ``warpPerspective`` for mask_Ap are replaced by sparse nearest-boundary
    queries, which is ~10× faster on a 720×1280 canvas.

    Args:
        ref: Pre-computed reference contour with stored mask.
        mask_B: (H, W) uint8 segmentation mask in the current frame.
        H: (3, 3) float64 homography matrix.
        image_shape: (height, width) of the current frame.
        n_points: Number of boundary points to sample.
        sdf_downsample: Downsampling factor for full SDF (option_c / both
            path only; 1 = full resolution).
        architecture: "option_e" uses the fast sparse path and leaves the
            option_c image fields as None.  "option_c" / "both" compute the
            full SDF images needed for the CNN crop.

    Returns:
        Dict with all tensors needed by the requested architecture(s), or
        None if the sample is degenerate.
    """
    img_h, img_w = image_shape

    # Step 1: Warp reference contour points
    pts_A_128, tan_sin_A, tan_cos_A = ref.get_nn_points(n_points)
    pts_Ap = _warp_points(pts_A_128, H)  # (N, 2)

    is_in_view = (
        (pts_Ap[:, 0] >= 0) & (pts_Ap[:, 0] < img_w)
        & (pts_Ap[:, 1] >= 0) & (pts_Ap[:, 1] < img_h)
    ).astype(np.float32)

    # Step 2: Extract ∂B — one findContours call serves both paths
    mask_B_u8 = (mask_B > 0).astype(np.uint8) * 255
    raw_boundary_B, pts_B = _extract_contour_data(mask_B_u8, n_points)
    if pts_B is None:
        return None
    tan_sin_B, tan_cos_B = _compute_tangent_directions(pts_B)

    _bt = 2.0
    is_on_boundary_B = (
        (pts_B[:, 0] < _bt)
        | (pts_B[:, 0] > img_w - 1 - _bt)
        | (pts_B[:, 1] < _bt)
        | (pts_B[:, 1] > img_h - 1 - _bt)
    ).astype(np.float32)

    # Step 3: Bounding box for position encoding
    in_view_mask = is_in_view > 0.5
    all_pts = np.concatenate(
        [pts_Ap[in_view_mask], pts_B], axis=0) if in_view_mask.any() else pts_B

    xy_min = all_pts.min(axis=0)
    xy_max = all_pts.max(axis=0)
    extent = xy_max - xy_min
    xy_min = np.maximum(xy_min - 0.1 * extent, [0.0, 0.0])
    xy_max = np.minimum(xy_max + 0.1 * extent, [img_w - 1.0, img_h - 1.0])
    for d in range(2):
        if xy_max[d] - xy_min[d] < 1.0:
            mid = (xy_max[d] + xy_min[d]) / 2
            xy_min[d] = max(0.0, mid - 0.5)
            xy_max[d] = xy_min[d] + 1.0

    eps = 1e-6
    norm_xy_A = (pts_Ap - xy_min) / (xy_max - xy_min + eps)
    norm_xy_B = (pts_B - xy_min) / (xy_max - xy_min + eps)

    D_large = float(np.sqrt(img_h ** 2 + img_w ** 2))

    # Steps 4-5: SDF values at boundary points — fast sparse, full image, or skipped
    if architecture == "point_baseline":
        # -- Baseline path: coordinates only, no SDF needed ------------------
        # Skip all distance-transform and cdist work.
        sdf_B_at_Ap = None
        sdf_Ap_at_B = None
        mask_Ap = None
        sdf_Ap = None
        sdf_B = None

    elif architecture == "option_e":
        # -- Sparse path: no distance transforms, no full warpPerspective -----
        # sdf_B at A' points: distance to raw B boundary, sign from mask lookup
        yi = np.clip(np.round(pts_Ap[:, 1]).astype(int), 0, img_h - 1)
        xi = np.clip(np.round(pts_Ap[:, 0]).astype(int), 0, img_w - 1)
        inside_B = mask_B_u8[yi, xi] > 0
        sdf_B_at_Ap_raw = _sparse_signed_dist(pts_Ap, raw_boundary_B, inside_B)
        sdf_B_at_Ap = np.where(
            is_in_view > 0.5, sdf_B_at_Ap_raw, D_large).astype(np.float32)

        # sdf_Ap at B points: warp all raw ref boundary pixels by H (dense,
        # no sub-sampling), sign from a small mask_Ap lookup
        boundary_A = ref.boundary_pts if ref.boundary_pts is not None else ref.points
        pts_Ap_dense = _warp_points(boundary_A, H)  # (M_raw, 2)

        _sign_scale = 4
        _sign_h = max(1, img_h // _sign_scale)
        _sign_w = max(1, img_w // _sign_scale)
        H_sign = np.diag([1.0 / _sign_scale, 1.0 / _sign_scale, 1.0]) @ H
        mask_Ap_sign = cv2.warpPerspective(
            ref.mask, H_sign, (_sign_w, _sign_h),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        yi_s = np.clip(np.round(pts_B[:, 1] / _sign_scale).astype(int), 0, _sign_h - 1)
        xi_s = np.clip(np.round(pts_B[:, 0] / _sign_scale).astype(int), 0, _sign_w - 1)
        inside_Ap = mask_Ap_sign[yi_s, xi_s] > 0
        sdf_Ap_at_B = _sparse_signed_dist(pts_B, pts_Ap_dense, inside_Ap)

        mask_Ap = None
        sdf_Ap = None
        sdf_B = None

    else:
        # -- Full path: SDF images needed for option_c crop -------------------
        if ref.mask is None:
            return None
        sdf_B = _compute_sdf_fast(mask_B_u8, sdf_downsample)
        mask_Ap = cv2.warpPerspective(
            ref.mask, H, (img_w, img_h),
            flags=cv2.INTER_NEAREST, borderValue=0,
        )
        sdf_Ap = _compute_sdf_fast(mask_Ap, sdf_downsample)

        sdf_B_at_Ap = _bilinear_sample(sdf_B, pts_Ap)
        sdf_B_at_Ap = np.where(
            is_in_view > 0.5, sdf_B_at_Ap, D_large).astype(np.float32)
        sdf_Ap_at_B = _bilinear_sample(sdf_Ap, pts_B)

    return {
        # For Option E / Baseline (point coordinates + flags)
        "pts_Ap": pts_Ap.astype(np.float32),
        "pts_B": pts_B.astype(np.float32),
        "sdf_B_at_Ap": sdf_B_at_Ap.astype(np.float32) if sdf_B_at_Ap is not None else None,
        "sdf_Ap_at_B": sdf_Ap_at_B.astype(np.float32) if sdf_Ap_at_B is not None else None,
        "tan_sin_A": tan_sin_A.astype(np.float32),
        "tan_cos_A": tan_cos_A.astype(np.float32),
        "tan_sin_B": tan_sin_B.astype(np.float32),
        "tan_cos_B": tan_cos_B.astype(np.float32),
        "is_in_view": is_in_view.astype(np.float32),
        "is_on_boundary_B": is_on_boundary_B,
        "norm_xy_A": norm_xy_A.astype(np.float32),
        "norm_xy_B": norm_xy_B.astype(np.float32),
        "bbox": (float(xy_min[0]), float(xy_min[1]),
                 float(xy_max[0]), float(xy_max[1])),
        # For Option C (None when architecture="option_e")
        "mask_Ap": mask_Ap,
        "sdf_Ap": sdf_Ap,
        "mask_B": mask_B_u8,
        "sdf_B": sdf_B,
    }


def _extract_and_sample_contour_fixed_n(
    mask_u8: np.ndarray,
    n_points: int,
) -> Optional[np.ndarray]:
    """Thin wrapper kept for compatibility — use _extract_contour_data instead."""
    _, resampled = _extract_contour_data(mask_u8, n_points)
    return resampled
