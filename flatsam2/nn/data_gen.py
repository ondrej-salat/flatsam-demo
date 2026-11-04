"""Synthetic data generation for NN homography quality estimation.

Generates training samples on-the-fly with the following pipeline:

1. Generate a random shape (mask_A)
2. Extract contour points from mask_A
3. Generate a GT homography H_GT; warp contour & mask to the "current frame"
4. Generate an error homography H_aug; measure contour-based RMSE as GT score
5. Apply mask augmentations (occlusions, spills, boundary noise)
6. Compute SDFs and cropping bounding box; construct NN inputs

The GT alignment error is computed from *all contour points*, matching how
benchmarks evaluate tracking quality on arbitrary-shaped targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, default_collate

try:
    profile
except NameError:
    profile = lambda f: f

from ..homography_quality import ReferenceContour, _resample_to_n_points, _warp_points
from .models import (
    build_option_c_input,
    build_option_e_features,
    build_point_baseline_features,
)
from .preprocess import preprocess

CANVAS_H, CANVAS_W = 720, 1280

#: Number of contour points used for RMSE measurement.
N_CONTOUR_RMSE = 256


# =========================================================================
# Shape generators
# =========================================================================


def generate_quadrilateral(
    rng: np.random.Generator,
) -> np.ndarray:
    """Random convex quadrilateral mask.

    Generates 4 vertices around a unit circle with random aspect ratio,
    scaled to cover 10-60% of the canvas area.

    Returns:
        (H, W) uint8 binary mask.
    """
    # Enforce a minimum arc gap between consecutive vertices so the shape
    # doesn't degenerate into a near-triangle.  Reserve min_gap per vertex,
    # then distribute the remaining arc with a Dirichlet draw.
    min_gap = 2 * np.pi * 0.15  # 54° minimum between consecutive vertices
    extra = rng.dirichlet([1.0, 1.0, 1.0, 1.0]) * (2 * np.pi - 4 * min_gap)
    gaps = extra + min_gap  # 4 gaps summing to 2π
    start = rng.uniform(0, 2 * np.pi)
    angles = start + np.array(
        [0.0, gaps[0], gaps[0] + gaps[1], gaps[0] + gaps[1] + gaps[2]]
    )

    # Unit circle vertices
    vx = np.cos(angles)
    vy = np.sin(angles)

    # Random aspect ratio
    aspect = rng.uniform(0.3, 3.0)
    rx = np.sqrt(aspect)
    ry = 1.0 / rx
    vx *= rx
    vy *= ry

    # Target area: 10%-60% of canvas
    target_area_frac = rng.uniform(0.10, 0.60)
    target_area = target_area_frac * CANVAS_H * CANVAS_W

    # Current polygon area via shoelace
    vx_s = vx - vx.mean()
    vy_s = vy - vy.mean()
    current_area = 0.5 * abs(
        np.sum(vx_s * np.roll(vy_s, -1) - vy_s * np.roll(vx_s, -1))
    )
    if current_area < 1e-6:
        current_area = 1.0

    scale = np.sqrt(target_area / current_area)
    # Clamp scale so polygon fits within canvas (half-extent < half-dimension)
    max_scale_x = (CANVAS_W / 2 - 2) / max(abs(vx).max(), 1e-6)
    max_scale_y = (CANVAS_H / 2 - 2) / max(abs(vy).max(), 1e-6)
    scale = min(scale, max_scale_x, max_scale_y)
    vx = vx * scale
    vy = vy * scale

    # Center in canvas with random offset
    margin_x = abs(vx).max()
    margin_y = abs(vy).max()
    cx = rng.uniform(margin_x, max(margin_x, CANVAS_W - margin_x))
    cy = rng.uniform(margin_y, max(margin_y, CANVAS_H - margin_y))
    cx = np.clip(cx, abs(vx).max() + 1, CANVAS_W - abs(vx).max() - 1)
    cy = np.clip(cy, abs(vy).max() + 1, CANVAS_H - abs(vy).max() - 1)

    vx += cx
    vy += cy

    pts = np.stack([vx, vy], axis=1).astype(np.int32)
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask


def generate_shape(
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random shape (currently quadrilateral-only).

    Returns:
        (H, W) uint8 binary mask.
    """
    return generate_quadrilateral(rng)


def extract_contour(
    mask: np.ndarray,
    n_points: int = N_CONTOUR_RMSE,
) -> Optional[np.ndarray]:
    """Extract and resample contour points from a binary mask.

    Args:
        mask: (H, W) uint8 binary mask.
        n_points: Number of output contour points.

    Returns:
        (n_points, 2) float32 array of (x, y) points, or None if
        no valid contour is found.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    longest = max(contours, key=lambda c: len(c))
    pts = longest.squeeze(axis=1).astype(np.float32)
    if len(pts) < 3:
        return None
    return _resample_to_n_points(pts, n_points)


# =========================================================================
# Scene transform sampling
# =========================================================================


def sample_scene_transform(
    rng: np.random.Generator,
    image_shape: Tuple[int, int] = (CANVAS_H, CANVAS_W),
) -> np.ndarray:
    """Sample a realistic homography T.

    Returns:
        (3, 3) float64 homography matrix.
    """
    h, w = image_shape
    cx, cy = w / 2, h / 2

    # Rotation
    angle = rng.uniform(-30, 30) * np.pi / 180

    # Scale (log-uniform)
    log_scale = rng.uniform(np.log(0.6), np.log(1.6))
    scale = np.exp(log_scale)

    # Translation
    tx = rng.uniform(-0.15, 0.15) * w
    ty = rng.uniform(-0.15, 0.15) * h

    # Build transform: translate to center, scale+rotate, translate back, translate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array(
        [
            [scale * cos_a, -scale * sin_a, 0],
            [scale * sin_a, scale * cos_a, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    T_to_center = np.array(
        [
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    T_from_center = np.array(
        [
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    T_translate = np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    T = T_from_center @ R @ T_to_center @ T_translate

    # Projective perturbation (perspective terms).
    # The perspective coefficients p1, p2 control how much the
    # homography deviates from an affine transform.  At the image
    # edge (distance ~max(w,h)/2 from center), a coefficient p
    # produces a denominator shift of p * max(w,h)/2.  We sample
    # magnitude log-uniformly so that most samples have moderate
    # perspective while some have strong foreshortening.
    half_diag = np.sqrt(w**2 + h**2) / 2
    # Target denominator shift at image corner: 1% – 30% of 1
    # → p_max = shift / half_diag.   shift ~ U_log[0.01, 0.30]
    log_shift = rng.uniform(np.log(0.01), np.log(0.30))
    shift = np.exp(log_shift)
    p_mag = shift / half_diag
    p_angle = rng.uniform(0, 2 * np.pi)
    T[2, 0] += p_mag * np.cos(p_angle)
    T[2, 1] += p_mag * np.sin(p_angle)
    T /= T[2, 2]

    return T


# =========================================================================
# Mask augmentation
# =========================================================================


def augment_mask_B(
    mask_B_clean: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one augmentation mode to mask_B_clean.

    Returns:
        (H, W) uint8 augmented mask.
    """
    mode = rng.choice(
        ["clean", "occlude", "spill", "noisy", "combo"],
        p=[0.15, 0.35, 0.20, 0.15, 0.15],
    )

    if mode == "clean":
        return mask_B_clean.copy()
    elif mode == "occlude":
        return _apply_occlusion(mask_B_clean, rng)
    elif mode == "spill":
        return _apply_spill(mask_B_clean, rng)
    elif mode == "noisy":
        return _apply_smooth_noise(mask_B_clean, rng)
    else:  # combo
        result = mask_B_clean.copy()
        ops = rng.choice(["occlude", "spill", "noisy"], size=2, replace=False)
        for op in ops:
            if op == "occlude":
                result = _apply_occlusion(result, rng)
            elif op == "spill":
                result = _apply_spill(result, rng)
            else:
                result = _apply_smooth_noise(result, rng)
        return result


def _apply_occlusion(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remove a random rectangular/elliptical occluder from mask."""
    result = mask.copy()
    mask_area = np.count_nonzero(mask)
    if mask_area < 10:
        return result

    occ_area_frac = rng.uniform(0.05, 0.90)

    ys, xs = np.where(mask > 0)
    cx = int(rng.choice(xs))
    cy = int(rng.choice(ys))

    target_occ_area = occ_area_frac * mask_area
    radius = int(np.sqrt(target_occ_area / np.pi))
    radius = max(5, radius)

    occluder = np.zeros_like(mask)
    if rng.random() < 0.5:
        # Rectangle
        hw = int(radius * rng.uniform(0.5, 2.0))
        hh = int(target_occ_area / max(hw * 2, 1))
        x0 = max(0, cx - hw)
        y0 = max(0, cy - hh)
        x1 = min(mask.shape[1], cx + hw)
        y1 = min(mask.shape[0], cy + hh)
        occluder[y0:y1, x0:x1] = 255
    else:
        # Ellipse
        ax1 = int(radius * rng.uniform(0.6, 1.5))
        ax2 = max(5, int(target_occ_area / (np.pi * max(ax1, 1))))
        cv2.ellipse(
            occluder, (cx, cy), (ax1, ax2), rng.uniform(0, 360), 0, 360, 255, -1
        )

    result[occluder > 0] = 0
    return result


def _apply_spill(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a random blob tangent to mask boundary."""
    result = mask.copy()
    mask_area = np.count_nonzero(mask)
    if mask_area < 10:
        return result

    spill_frac = rng.uniform(0.05, 0.40)
    target_area = spill_frac * mask_area
    radius = int(np.sqrt(target_area / np.pi))
    radius = max(5, radius)

    # Find boundary point
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return result

    longest = max(contours, key=lambda c: len(c))
    idx = rng.integers(len(longest))
    bx, by = longest[idx].squeeze()

    blob = np.zeros_like(mask)
    # Offset away from mask center
    ys, xs = np.where(mask > 0)
    mcx, mcy = xs.mean(), ys.mean()
    dx = bx - mcx
    dy = by - mcy
    norm = np.sqrt(dx**2 + dy**2) + 1e-8
    ox = int(bx + dx / norm * radius * 0.5)
    oy = int(by + dy / norm * radius * 0.5)
    ox = np.clip(ox, radius, mask.shape[1] - radius - 1)
    oy = np.clip(oy, radius, mask.shape[0] - radius - 1)

    cv2.circle(blob, (int(ox), int(oy)), radius, 255, -1)
    result[blob > 0] = 255
    return result


def _apply_smooth_noise(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb the mask boundary with smooth radial noise.

    Extracts the contour, displaces each point radially by a spatially
    smoothed Gaussian signal (so some regions grow, others shrink), then
    redraws the filled polygon.  Produces organic-looking boundary variation
    rather than global dilate/erode.
    """
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return mask.copy()
    pts = max(contours, key=lambda c: len(c)).squeeze(axis=1).astype(np.float64)
    if len(pts) < 8:
        return mask.copy()

    # Resample to a fixed number of points for consistent noise frequency
    n = min(512, len(pts))
    indices = np.round(np.linspace(0, len(pts) - 1, n)).astype(int)
    pts = pts[indices]

    # Centroid for radial direction
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    dirs = pts - np.array([cx, cy])  # (n, 2)
    radii = np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-6)
    unit_dirs = dirs / radii  # (n, 2)

    # Smoothed Gaussian noise along the contour
    noise_raw = rng.standard_normal(n)
    # Kernel width controls spatial frequency: wider → smoother bumps
    sigma = rng.uniform(n * 0.04, n * 0.15)
    kw = int(sigma * 4) | 1  # odd kernel width
    k = np.exp(-0.5 * (np.arange(kw) - kw // 2) ** 2 / sigma**2)
    k /= k.sum()
    # Wrap-around convolution for closed contour
    padded = np.concatenate([noise_raw[-(kw // 2) :], noise_raw, noise_raw[: kw // 2]])
    noise_smooth = np.convolve(padded, k, mode="valid")[:n]

    # Scale amplitude to a fraction of the average radius
    avg_radius = radii.mean()
    amplitude = rng.uniform(0.05, 0.5) * 15
    noise_smooth = noise_smooth / (noise_smooth.std() + 1e-8) * amplitude

    # Displace points radially
    pts_new = pts + unit_dirs * noise_smooth[:, None]
    pts_new = pts_new.astype(np.int32)

    result = np.zeros_like(mask)
    cv2.fillPoly(result, [pts_new], 255)
    return result


# =========================================================================
# Error regime and H_candidate construction
# =========================================================================


def contour_rmse(
    H_true: np.ndarray,
    H_candidate: np.ndarray,
    contour_A: np.ndarray,
) -> float:
    """RMSE across all contour points warped by two homographies.

    This measures how far the candidate homography deviates from the
    true homography, using the full contour shape rather than just
    bounding box corners — matching how benchmarks evaluate tracking.

    Args:
        H_true: (3, 3) ground-truth homography.
        H_candidate: (3, 3) candidate homography.
        contour_A: (N, 2) reference contour points in the initial frame.

    Returns:
        Contour alignment RMSE in pixels.
    """
    pts_true = _warp_points(contour_A, H_true)
    pts_cand = _warp_points(contour_A, H_candidate)
    diff = pts_true - pts_cand  # (N, 2)
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def perturb_homography_to_target_rmse(
    T: np.ndarray,
    contour_A: np.ndarray,
    target_rmse: float,
    rng: np.random.Generator,
    max_iter: int = 20,
) -> np.ndarray:
    """Binary-search perturbation magnitude to hit target contour RMSE.

    Args:
        T: (3, 3) ground-truth homography.
        contour_A: (N, 2) reference contour points.
        target_rmse: Desired RMSE in pixels.
        rng: Random number generator.
        max_iter: Maximum binary search iterations.

    Returns:
        (3, 3) perturbed homography.
    """
    h, w = CANVAS_H, CANVAS_W
    cx, cy = w / 2, h / 2

    # Random perturbation direction
    d_angle = rng.standard_normal() * 10 * np.pi / 180
    d_scale = rng.standard_normal() * 0.1
    d_tx = rng.standard_normal() * 20
    d_ty = rng.standard_normal() * 20

    def make_perturb(alpha: float) -> np.ndarray:
        angle = alpha * d_angle
        scale = 1.0 + alpha * d_scale
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        P = np.array(
            [
                [scale * cos_a, -scale * sin_a, alpha * d_tx],
                [scale * sin_a, scale * cos_a, alpha * d_ty],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        # Apply perturbation around center
        T_to = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        T_from = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
        return T @ (T_from @ P @ T_to)

    # Binary search on alpha
    lo, hi = 0.0, 5.0

    # First check if hi is enough
    H_hi = make_perturb(hi)
    rmse_hi = contour_rmse(T, H_hi, contour_A)
    if rmse_hi < target_rmse:
        hi *= 10
        H_hi = make_perturb(hi)

    tolerance = max(target_rmse * 0.1, 0.5)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        H_mid = make_perturb(mid)
        rmse_mid = contour_rmse(T, H_mid, contour_A)

        if abs(rmse_mid - target_rmse) < tolerance:
            return H_mid
        elif rmse_mid < target_rmse:
            lo = mid
        else:
            hi = mid

    return make_perturb((lo + hi) / 2)


def sample_h_candidate(
    T: np.ndarray,
    contour_A: np.ndarray,
    rng: np.random.Generator,
    n_clip: float = 30.0,
) -> Tuple[np.ndarray, float, bool]:
    """Sample H_candidate in one of four error regimes.

    Args:
        T: (3, 3) ground-truth homography.
        contour_A: (N, 2) reference contour points.
        rng: Random number generator.
        n_clip: Censoring threshold in pixels.

    Returns:
        (H_candidate, rmse_actual, censored).
    """
    regime = rng.choice(
        ["perfect", "good", "borderline", "bad"],
        p=[0.10, 0.35, 0.20, 0.35],
    )

    if regime == "perfect":
        H_candidate = T.copy()
        rmse = 0.0

    elif regime == "good":
        target = np.exp(rng.uniform(np.log(0.5), np.log(15)))
        H_candidate = perturb_homography_to_target_rmse(T, contour_A, target, rng)
        rmse = contour_rmse(T, H_candidate, contour_A)

    elif regime == "borderline":
        target = rng.uniform(15, 30)
        H_candidate = perturb_homography_to_target_rmse(T, contour_A, target, rng)
        rmse = contour_rmse(T, H_candidate, contour_A)

    else:  # bad
        # Large random perturbation
        H_candidate = perturb_homography_to_target_rmse(T, contour_A, 100.0, rng)
        rmse = contour_rmse(T, H_candidate, contour_A)

    censored = rmse > n_clip

    return H_candidate, rmse, censored


# =========================================================================
# Full sample generation
# =========================================================================


@profile
def generate_sample(
    rng: np.random.Generator,
    n_points: int = 128,
    canvas_shape: Tuple[int, int] = (CANVAS_H, CANVAS_W),
    keep_intermediates: bool = False,
    architecture: str = "both",
    sdf_downsample: int = 1,
    crop_size: int = 128,
) -> Optional[dict]:
    """Generate one complete training sample.

    Pipeline:
        1. Generate object shape (mask_A, contour_A)
        2. Generate GT motion to the current frame (H_GT, mask_B_clean)
        3. Generate error homography H_aug; measure GT score (contour RMSE)
        4. Apply mask augmentations (occlusions, spills, boundary noise)
        5. Compute SDFs, cropping, and construct NN inputs

    Args:
        rng: Random number generator.
        n_points: Number of boundary points for the NN input.
        canvas_shape: (height, width) of the synthetic canvas.
        keep_intermediates: If True, include all pipeline intermediates
            in the output dict (for visualization / debugging).
        architecture: One of "option_e", "option_c", "both", or
            "point_baseline".  Controls which NN input tensors are built.
        crop_size: Option-C crop size in pixels.

    Returns:
        Dict with keys depending on architecture:
          - option_e / both: features_A (N,20), features_B (N,20)
          - option_c / both: crop_4ch, scale_factor
          - point_baseline:  features_combined (2N, 18)
          - always: gt_rmse, censored
        When keep_intermediates is True, also includes mask_A, mask_B_clean,
        mask_B, mask_Ap, H_GT, H_candidate, contour_A, ref, prep.
        Returns None if degenerate.
    """
    h, w = canvas_shape

    # --- Step 1: Generate object shape ---
    mask_A = generate_shape(rng)
    if np.count_nonzero(mask_A) < 100:
        return None

    contour_A = extract_contour(mask_A)
    if contour_A is None:
        return None

    # --- Step 2: Generate GT motion to the current frame ---
    H_GT = sample_scene_transform(rng, canvas_shape)

    mask_B_clean = cv2.warpPerspective(
        mask_A, H_GT, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    if np.count_nonzero(mask_B_clean) < 100:
        return None

    # --- Step 3: Generate error homography; measure GT score ---
    H_aug, gt_rmse, censored = sample_h_candidate(H_GT, contour_A, rng)

    # --- Step 4: Apply mask augmentations ---
    mask_B = augment_mask_B(mask_B_clean, rng)
    if np.count_nonzero(mask_B) < 100:
        return None

    # --- Step 5: Construct NN inputs ---
    # Random flips for data augmentation (don't affect RMSE — symmetric)
    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.5
    if flip_h or flip_v:
        Fh = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        Fv = np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], dtype=np.float64)

        F = np.eye(3, dtype=np.float64)
        if flip_h:
            F = Fh @ F
        if flip_v:
            F = Fv @ F

        if flip_h:
            mask_A = mask_A[:, ::-1].copy()
            mask_B = mask_B[:, ::-1].copy()
            mask_B_clean = mask_B_clean[:, ::-1].copy()
        if flip_v:
            mask_A = mask_A[::-1, :].copy()
            mask_B = mask_B[::-1, :].copy()
            mask_B_clean = mask_B_clean[::-1, :].copy()

        F_inv = np.linalg.inv(F)
        H_GT = F @ H_GT @ F_inv
        H_aug = F @ H_aug @ F_inv

    try:
        ref = ReferenceContour.build(mask_A)
    except ValueError:
        return None

    prep = preprocess(
        ref,
        mask_B,
        H_aug,
        canvas_shape,
        n_points,
        sdf_downsample=sdf_downsample,
        architecture=architecture,
    )
    if prep is None:
        return None

    feat_A, feat_B = (
        build_option_e_features(prep)
        if architecture in ("option_e", "both")
        else (None, None)
    )
    crop_4ch, s = (
        build_option_c_input(prep, crop_size=crop_size)
        if architecture in ("option_c", "both")
        else (None, 1.0)
    )
    features_combined = (
        build_point_baseline_features(prep)
        if architecture == "point_baseline"
        else None
    )

    result = {
        "features_A": feat_A,
        "features_B": feat_B,
        "crop_4ch": crop_4ch.numpy() if crop_4ch is not None else None,
        "scale_factor": float(s),
        "features_combined": features_combined,
        "gt_rmse": float(gt_rmse),
        "censored": bool(censored),
    }

    if keep_intermediates:
        mask_Ap = cv2.warpPerspective(
            mask_A, H_aug, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        # Re-extract contour from the final (possibly flipped) mask_A
        # so it's consistent with H_GT / H_candidate for visualization.
        contour_A_final = extract_contour(mask_A)
        result.update(
            {
                "mask_A": mask_A,
                "mask_B_clean": mask_B_clean,
                "mask_B": mask_B,
                "mask_Ap": mask_Ap,
                "H_GT": H_GT,
                "H_candidate": H_aug,
                "contour_A": contour_A_final,
                "ref": ref,
                "prep": prep,
                "feat_A": feat_A,
                "feat_B": feat_B,
            }
        )

    return result


# =========================================================================
# Dataset class
# =========================================================================


class SyntheticHomographyDataset(IterableDataset):
    """On-the-fly synthetic data generation.

    Each worker gets its own RNG seeded by (base_seed + worker_id).
    """

    def __init__(
        self,
        architecture: str = "option_e",
        seed: int = 0,
        n_points: int = 128,
        max_samples: Optional[int] = None,
        sdf_downsample: int = 4,
        crop_size: int = 128,
    ):
        self.architecture = architecture
        self.seed = seed
        self.n_points = n_points
        self.max_samples = max_samples
        self.sdf_downsample = sdf_downsample
        self.crop_size = crop_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info else 0)
        rng = np.random.default_rng(worker_seed)

        count = 0
        while self.max_samples is None or count < self.max_samples:
            sample = generate_sample(
                rng,
                self.n_points,
                architecture=self.architecture,
                sdf_downsample=self.sdf_downsample,
                crop_size=self.crop_size,
            )
            if sample is None:
                continue

            count += 1

            if self.architecture == "option_e":
                yield (
                    torch.from_numpy(sample["features_A"]),
                    torch.from_numpy(sample["features_B"]),
                    torch.tensor(sample["gt_rmse"], dtype=torch.float32),
                    torch.tensor(sample["censored"], dtype=torch.bool),
                )
            elif self.architecture == "point_baseline":
                yield (
                    torch.from_numpy(sample["features_combined"]),
                    torch.tensor(sample["gt_rmse"], dtype=torch.float32),
                    torch.tensor(sample["censored"], dtype=torch.bool),
                )
            else:  # option_c
                yield (
                    torch.from_numpy(sample["crop_4ch"]),
                    torch.tensor(sample["scale_factor"], dtype=torch.float32),
                    torch.tensor(sample["gt_rmse"], dtype=torch.float32),
                    torch.tensor(sample["censored"], dtype=torch.bool),
                )


class CachedHomographyDataset(Dataset):
    """Map-style dataset backed by pre-generated .npy files.

    Use ``flatsam2.nn.generate_dataset`` to create the cache directory, then
    pass ``--cache-dir`` to the training script.  Loading is done once at
    construction time; the arrays live in RAM for fast random access.

    For ``option_e``, two cheap augmentations are applied on-the-fly when
    *augment=True*:

    * **Global SDF scale** — multiply both SDF feature columns and gt_rmse by
      the same log-normal factor (σ=0.3), simulating different image scales.
      Censoring is recomputed; because the threshold scales equally it is
      unchanged by the operation.
    * **Tangent flip** — negate all tangent (sin/cos) columns with p=0.5,
      corresponding to a reversed contour orientation.

    For ``point_baseline``, no SDF or tangent features exist so augmentation
    is not applied regardless of the *augment* flag.

    Args:
        cache_dir: Directory containing the pre-generated .npy files.
            option_e requires features_A.npy and features_B.npy.
            point_baseline requires features_combined.npy.
        architecture: "option_e" or "point_baseline"
            (option_c crops are not cached).
        augment: Enable on-the-fly augmentations (default True;
            has no effect for point_baseline).
        n_clip: Censoring threshold in pixels — used to recompute ``censored``
            after SDF scaling.
    """

    # Column indices in the (128, 20) option_e feature arrays
    _SDF_COL = 0  # sdf_B_at_Ap  (feat_A) / sdf_Ap_at_B (feat_B)
    _TAN_COLS = (1, 2)  # tan_sin, tan_cos

    def __init__(
        self,
        cache_dir: str,
        architecture: str = "option_e",
        augment: bool = True,
        n_clip: float = 30.0,
    ):
        if architecture not in ("option_e", "point_baseline"):
            raise ValueError(
                f"CachedHomographyDataset does not support arch={architecture!r}. "
                "Use 'option_e' or 'point_baseline'."
            )
        self.architecture = architecture
        d = Path(cache_dir)

        if architecture == "option_e":
            self.features_A = np.load(d / "features_A.npy")
            self.features_B = np.load(d / "features_B.npy")
            assert len(self.features_A) == len(self.features_B), (
                "features_A and features_B have inconsistent length."
            )
            n = len(self.features_A)
        else:  # point_baseline
            self.features_combined = np.load(d / "features_combined.npy")
            n = len(self.features_combined)

        self.gt_rmse = np.load(d / "gt_rmse.npy")
        self.censored = np.load(d / "censored.npy")
        assert len(self.gt_rmse) == n, "Cache files have inconsistent length."
        self.augment = augment
        self.n_clip = n_clip
        self._rng: Optional[np.random.Generator] = None  # initialised per worker
        print(
            f"Loaded cached dataset: {n:,} samples from {d}"
            f"  arch={architecture}  augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.gt_rmse)

    def __getitem__(self, idx: int):
        gt_rmse = float(self.gt_rmse[idx])
        censored = bool(self.censored[idx])

        if self.architecture == "point_baseline":
            feat = self.features_combined[idx].copy()
            return (
                torch.from_numpy(feat),
                torch.tensor(gt_rmse, dtype=torch.float32),
                torch.tensor(censored, dtype=torch.bool),
            )

        # option_e — .copy() because we may modify in-place below
        feat_A = self.features_A[idx].copy()
        feat_B = self.features_B[idx].copy()

        if self.augment:
            # Lazy RNG init — each DataLoader worker is a separate process,
            # so each gets its own independent generator.
            if self._rng is None:
                self._rng = np.random.default_rng()
            rng = self._rng

            # 1. Global SDF scale (simulate different image resolutions)
            s = float(np.exp(rng.normal(0.0, 0.3)))
            feat_A[:, self._SDF_COL] *= s
            feat_B[:, self._SDF_COL] *= s
            gt_rmse *= s
            censored = (
                gt_rmse > self.n_clip
            )  # unchanged: (rmse*s > n_clip*s) ≡ original

            # 2. Tangent flip (reversed contour orientation, p=0.5)
            if rng.random() < 0.5:
                feat_A[:, self._TAN_COLS[0] : self._TAN_COLS[1] + 1] *= -1
                feat_B[:, self._TAN_COLS[0] : self._TAN_COLS[1] + 1] *= -1

        return (
            torch.from_numpy(feat_A),
            torch.from_numpy(feat_B),
            torch.tensor(gt_rmse, dtype=torch.float32),
            torch.tensor(censored, dtype=torch.bool),
        )


class RealWorldHomographyDataset(Dataset):
    """Map-style dataset backed by collected real-world masks and homographies.

    Args:
        dataset_dir: Directory produced by ``collect_realworld_dataset.py``.
        architecture: One of ``option_e``, ``point_baseline``, or ``option_c``.
        subset_size: Optional subset size selecting ``subsets/split_<N>.json``
            (or the legacy ``subsets/subset_<N>.json``).
            When omitted, all saved samples are used.
        n_points: Number of contour points used by the preprocessing pipeline.
        sdf_downsample: Downsampling factor for option_c SDF computation.
        crop_size: Option-C crop size in pixels.
        augment_masks: If True, apply synthetic mask corruption to the current
            mask before preprocessing. Intended for training only.
    """

    def __init__(
        self,
        dataset_dir: str,
        architecture: str = "option_e",
        subset_size: Optional[int] = None,
        sample_indices: Optional[np.ndarray] = None,
        n_points: int = 128,
        sdf_downsample: int = 4,
        crop_size: int = 128,
        augment_masks: bool = False,
        return_none_on_failure: bool = True,
        return_info: bool = False,
    ):
        if architecture not in ("option_e", "point_baseline", "option_c"):
            raise ValueError(
                f"Unsupported architecture for RealWorldHomographyDataset: {architecture!r}"
            )

        self.dataset_dir = Path(dataset_dir)
        self.architecture = architecture
        self.n_points = n_points
        self.sdf_downsample = sdf_downsample
        self.crop_size = crop_size
        self.augment_masks = augment_masks
        self.return_none_on_failure = return_none_on_failure
        self.return_info = return_info
        self._rng: Optional[np.random.Generator] = None

        samples_path = self.dataset_dir / "samples.npz"
        sequences_path = self.dataset_dir / "sequences.json"
        if not samples_path.exists():
            raise ValueError(f"Samples file not found: {samples_path}")
        if not sequences_path.exists():
            raise ValueError(f"Sequences file not found: {sequences_path}")

        with np.load(samples_path, allow_pickle=False) as sample_data:
            self.seq_ids = sample_data["seq_ids"]
            self.frame_indices = sample_data["frame_indices"]
            self.homographies = sample_data["homographies"]
            self.gt_rmse = sample_data["gt_rmse"]
            self.sources = sample_data["sources"]
            self.config_names = sample_data["config_names"]

        with open(sequences_path, "r") as f:
            self.sequences: List[dict] = json.load(f)

        if subset_size is not None and sample_indices is not None:
            raise ValueError("Provide either subset_size or sample_indices, not both")

        if sample_indices is not None:
            self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        elif subset_size is None:
            self.sample_indices = np.arange(len(self.gt_rmse), dtype=np.int64)
        else:
            # Prefer the newer exclusive split files; fall back to legacy subset files.
            split_path = self.dataset_dir / "subsets" / f"split_{subset_size}.json"
            legacy_path = self.dataset_dir / "subsets" / f"subset_{subset_size}.json"
            if split_path.exists():
                subset_path = split_path
            elif legacy_path.exists():
                subset_path = legacy_path
            else:
                raise ValueError(
                    f"No split file found for size {subset_size} in "
                    f"{self.dataset_dir / 'subsets'} "
                    f"(tried split_{subset_size}.json and subset_{subset_size}.json)"
                )
            with open(subset_path, "r") as f:
                subset_data = json.load(f)
            self.sample_indices = np.asarray(subset_data["indices"], dtype=np.int64)

        self._ref_by_seq_id: Dict[int, ReferenceContour] = {}
        for seq_id in np.unique(self.seq_ids[self.sample_indices]):
            seq_name = str(self.sequences[int(seq_id)]["name"])
            init_mask_path = self.dataset_dir / "masks" / seq_name / "00001.png"
            init_mask = cv2.imread(str(init_mask_path), cv2.IMREAD_GRAYSCALE)
            if init_mask is None:
                raise ValueError(f"Failed to read init mask: {init_mask_path}")
            self._ref_by_seq_id[int(seq_id)] = ReferenceContour.build(init_mask)

        print(
            f"Loaded real-world dataset: {len(self.sample_indices):,} samples from "
            f"{self.dataset_dir}  arch={self.architecture}  augment_masks={self.augment_masks}"
        )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _load_mask(self, sample_idx: int) -> np.ndarray:
        seq_id = int(self.seq_ids[sample_idx])
        seq_name = str(self.sequences[seq_id]["name"])
        frame_idx = int(self.frame_indices[sample_idx])
        mask_path = self.dataset_dir / "masks" / seq_name / f"{frame_idx + 1:05d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read current mask: {mask_path}")
        return mask

    def _build_preprocess(self, sample_idx: int) -> dict:
        seq_id = int(self.seq_ids[sample_idx])
        ref = self._ref_by_seq_id[seq_id]
        H = self.homographies[sample_idx].astype(np.float64)
        base_mask = self._load_mask(sample_idx)

        masks_to_try = [base_mask]
        if self.augment_masks:
            rng = self._get_rng()
            masks_to_try = [augment_mask_B(base_mask, rng) for _ in range(3)] + [
                base_mask
            ]

        for mask_B in masks_to_try:
            prep = preprocess(
                ref,
                mask_B,
                H,
                mask_B.shape[:2],
                n_points=self.n_points,
                sdf_downsample=self.sdf_downsample,
                architecture=self.architecture,
            )
            if prep is not None:
                return prep

        raise ValueError(
            f"Failed to preprocess sample {sample_idx} from {self.dataset_dir}"
        )

    def __getitem__(self, idx: int):
        sample_idx = int(self.sample_indices[idx])
        try:
            prep = self._build_preprocess(sample_idx)
        except ValueError:
            if self.return_none_on_failure:
                return None
            raise
        gt_rmse = float(self.gt_rmse[sample_idx])
        censored = False
        info = {
            "sample_idx": sample_idx,
            "seq_id": int(self.seq_ids[sample_idx]),
            "frame_idx": int(self.frame_indices[sample_idx]),
            "seq_name": str(self.sequences[int(self.seq_ids[sample_idx])]["name"]),
            "source": str(self.sources[sample_idx]),
            "config_name": str(self.config_names[sample_idx]),
        }

        if self.architecture == "option_e":
            feat_A, feat_B = build_option_e_features(prep)
            sample = (
                torch.from_numpy(feat_A),
                torch.from_numpy(feat_B),
                torch.tensor(gt_rmse, dtype=torch.float32),
                torch.tensor(censored, dtype=torch.bool),
            )
            return sample + (info,) if self.return_info else sample

        if self.architecture == "point_baseline":
            features = build_point_baseline_features(prep)
            sample = (
                torch.from_numpy(features),
                torch.tensor(gt_rmse, dtype=torch.float32),
                torch.tensor(censored, dtype=torch.bool),
            )
            return sample + (info,) if self.return_info else sample

        crop_4ch, scale_factor = build_option_c_input(
            prep,
            crop_size=self.crop_size,
        )
        sample = (
            crop_4ch,
            torch.tensor(scale_factor, dtype=torch.float32),
            torch.tensor(gt_rmse, dtype=torch.float32),
            torch.tensor(censored, dtype=torch.bool),
        )
        return sample + (info,) if self.return_info else sample


def collate_skip_none(batch):
    """Collate function that drops invalid samples represented by ``None``."""
    valid_batch = [sample for sample in batch if sample is not None]
    if not valid_batch:
        return None
    return default_collate(valid_batch)
