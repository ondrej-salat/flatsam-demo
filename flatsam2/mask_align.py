"""Direct mask -> homography alignment: the P3' replacement for Hough SAM-H.

Damped Gauss-Newton on the 4-point parameterization (the four image positions
of the template corners; never raw 3x3 coefficients), minimizing a Huber loss
on the target-mask signed distance sampled along the warped template-mask
boundary.

Feasibility measured by ``experiments/mask_alignment_basin.py`` (2026-07-11,
tau=15 px): 92.7% convergence on clean SAM masks from perturbations up to
~20 px translation / 20 deg rotation / 40% scale, median landing error 3.2 px.
Symmetric capture appears only beyond ~45 deg rotations — orientation recovery
stays the keypoint branch's job; this aligner refines from the last-frame
homography, which is well inside the measured basin for tracking-scale motion.

Identifiability caveat: the objective constrains only the mask boundary, so
for template shapes with a homography stabilizer (conics: any ellipse leaves a
3-parameter family; quads leave none) the pose is recovered only up to that
stabilizer. Benchmark quad targets are fully pinned; for near-symmetric
arbitrary shapes the last-frame init keeps the solution on the previous pose
branch, but slow gauge drift along the stabilizer is possible and uncorrected.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from einops import rearrange

from flatsam2.corner_covariance import (
    corner_blocks,
    corners_to_theta,
    invert_normal_matrix,
    theta_to_corners,
)
from flatsam2.datasets import corners_to_mask, warp_corners
from flatsam2.mask_metrics import signed_distance


_MIN_MASK_AREA_PX = 50
# Below this second-moment eccentricity the principal axis is numerical noise
# (a square reads as isotropic just like a circle does), so a moment-based
# re-initialization must try the quarter turns instead of trusting the angle.
_MIN_ECCENTRICITY = 0.05


def template_boundary(template_mask: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    """Subsampled outer-contour points of the template mask.

    The template is constant for a whole sequence, so callers in a tracking loop
    should compute this once and pass it to :func:`align_mask_to_homography` as
    ``boundary`` rather than paying ``findContours`` on every frame.

    Args:
        template_mask: Binary template mask (H, W) in init-frame coordinates.
        n_points: Approximate number of boundary samples.

    Returns:
        Boundary points (N, 2) float64 as (x, y), or None when the mask has no
        usable contour.
    """
    contours, _ = cv2.findContours(
        template_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return None
    pts = rearrange(contour, "n 1 xy -> n xy").astype(np.float64)
    step = max(1, len(pts) // n_points)
    return pts[::step]


def _sample_bilinear(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of a scalar field at (x, y) points, edge-clamped.

    This is the innermost operation of the whole aligner — every Gauss-Newton
    iteration evaluates it nine times, once for the residual and once per
    numeric-Jacobian column — so it is a single OpenCV call rather than four
    fancy-indexed gathers. ``BORDER_REPLICATE`` reproduces the edge clamping.

    Args:
        field: Scalar field (H, W) float.
        pts: Sample points (N, 2) as (x, y).

    Returns:
        Sampled values (N,).
    """
    grid = pts.astype(np.float32, copy=False)
    # remap has no float64 kernel; signed_distance already returns float32, so
    # this only ever converts for an unusual caller.
    sampled = cv2.remap(
        field if field.dtype == np.float32 else field.astype(np.float32),
        grid[None, :, 0],
        grid[None, :, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled[0].astype(np.float64, copy=False)


def _warp_boundary(
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control: np.ndarray,
) -> Optional[np.ndarray]:
    """Warp template-boundary points by the 4-point homography init -> control.

    Args:
        boundary: Template boundary points (N, 2).
        init_corners: Init-frame corner positions (2, 4).
        control: Current control-point positions (2, 4).

    Returns:
        Warped points (N, 2), or None when the 4-point fit is degenerate.
    """
    src = init_corners.T.astype(np.float32)
    dst = control.T.astype(np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    warped = boundary @ H[:, :2].T + H[:, 2]
    z = warped[:, 2:3]
    if not np.all(np.abs(z) > 1e-9):
        return None
    return warped[:, :2] / z


def _residuals(
    sdf: np.ndarray,
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control: np.ndarray,
) -> Optional[np.ndarray]:
    """Signed-distance residuals of the warped template boundary; None if degenerate."""
    warped = _warp_boundary(boundary, init_corners, control)
    if warped is None:
        return None
    return _sample_bilinear(sdf, warped)


def _gn_align(
    sdf: np.ndarray,
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control0: np.ndarray,
    max_iters: int,
    huber_delta: float,
    step_tol: float,
    trim_fraction: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Damped Gauss-Newton over the 8 control-point coordinates.

    Minimizes ``sum huber(sdf_target(warp(boundary, H(control))))`` with a
    numeric forward-difference Jacobian and Levenberg-style damping.

    Args:
        sdf: Signed distance field of the target mask (H, W), positive outside.
        boundary: Template boundary points (N, 2) in init-frame coordinates.
        init_corners: Init-frame corners (2, 4) defining the parameterization.
        control0: Initial control points (2, 4).
        max_iters: Iteration cap.
        huber_delta: Huber transition (px).
        step_tol: Stop when the max control-point step falls below this (px).
        trim_fraction: Share of the worst-fitting boundary samples to drop
            outright each iteration. Huber only *down-weights* a far sample, so
            a hidden stretch of silhouette still pulls the fit onto the
            occluder's edge — measured to drag an exactly-correct pose 86 px off
            a one-third-occluded mask. Discarding the worst tail instead lets
            that much of the outline go missing without biasing the solve.

    Returns:
        Tuple ``(control, n_iters)`` — final control points (2, 4) and the
        number of GN iterations actually run (``max_iters`` means the cap was
        binding, i.e. the frame paid full price without converging).
    """
    control = control0.astype(np.float64).copy()
    lam = 1.0
    n_iters = 0
    r = _residuals(sdf, boundary, init_corners, control)
    if r is None:
        return control, n_iters
    for _ in range(max_iters):
        n_iters += 1
        absr = np.abs(r)
        w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-9))
        if trim_fraction > 0.0:
            n_trim = int(trim_fraction * len(r))
            if 0 < n_trim < len(r):
                w[np.argpartition(absr, -n_trim)[-n_trim:]] = 0.0

        J = np.zeros((len(r), 8))
        flat = corners_to_theta(control)
        delta = 0.5
        degenerate = False
        for p in range(8):
            trial = flat.copy()
            trial[p] += delta
            r_p = _residuals(sdf, boundary, init_corners, theta_to_corners(trial))
            if r_p is None:
                degenerate = True
                break
            J[:, p] = (r_p - r) / delta
        if degenerate:
            break

        JW = J * w[:, None]
        A = J.T @ JW + lam * np.eye(8)
        g = J.T @ (w * r)
        try:
            step = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            break
        trial = theta_to_corners(flat + step)
        r_trial = _residuals(sdf, boundary, init_corners, trial)
        cost = float((w * r * r).sum())
        if r_trial is not None and float((w * r_trial * r_trial).sum()) < cost:
            control, r = trial, r_trial
            lam = max(lam * 0.5, 1e-3)
            if np.abs(step).max() < step_tol:
                break
        else:
            lam = min(lam * 4.0, 1e6)
            if lam >= 1e6:
                break
    return control, n_iters


def _corner_cov_from_boundary_fit(
    sdf: np.ndarray,
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control: np.ndarray,
    r0: np.ndarray,
    huber_delta_px: float,
    step_px: float = 0.5,
) -> Optional[np.ndarray]:
    """Per-corner covariance from the converged GN fit's own normal equations.

    Same idea as :func:`flatsam2.corner_covariance.homography_corner_covs`: the
    residual model IS the fit objective (Huber-weighted boundary signed
    distance), parameterized directly by the four corner positions, so
    ``sigma^2 (J^T W J)^-1`` at the converged point already lives in corner
    space — no separate propagation step, and the same formula the
    correspondence branch uses, just with a different residual function. This
    is what lets a mask aligner and a keypoint matcher report uncertainty in
    one shared currency.

    Args:
        sdf: Signed distance field of the target mask (H, W).
        boundary: Template boundary points (N, 2).
        init_corners: Init-frame corners (2, 4).
        control: Converged control points (2, 4).
        r0: Residuals at ``control`` (from :func:`_residuals`).
        huber_delta_px: Huber transition (px), matching the fit.
        step_px: Finite-difference step for the Jacobian (px).

    Returns:
        (4, 2, 2) per-corner covariances in pixels, or None when the fit is
        degenerate (too few boundary points, singular normal matrix).
    """
    n_r = len(r0)
    if n_r <= 8:
        return None
    absr = np.abs(r0)
    w = np.where(absr <= huber_delta_px, 1.0, huber_delta_px / np.maximum(absr, 1e-9))

    theta0 = corners_to_theta(control)
    jac = np.zeros((n_r, 8))
    for p in range(8):
        plus, minus = theta0.copy(), theta0.copy()
        plus[p] += step_px
        minus[p] -= step_px
        r_plus = _residuals(sdf, boundary, init_corners, theta_to_corners(plus))
        r_minus = _residuals(sdf, boundary, init_corners, theta_to_corners(minus))
        if r_plus is None or r_minus is None:
            return None
        jac[:, p] = (r_plus - r_minus) / (2.0 * step_px)

    dof = n_r - 8
    var = float((w * r0 * r0).sum()) / dof
    theta_cov = invert_normal_matrix(jac.T @ (w[:, None] * jac), var)
    if theta_cov is None:
        return None
    return corner_blocks(theta_cov)


def _mask_pose(
    mask_bool: np.ndarray,
) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """Centroid, orientation, area and eccentricity of a mask from its moments.

    Args:
        mask_bool: (H, W) boolean mask.

    Returns:
        ``(centroid (2,), orientation_rad, area_px, eccentricity)``, or None
        when the mask is too small or its second moments are degenerate.
        Eccentricity is ``(l1 - l2) / (l1 + l2)`` of the second-moment matrix:
        it says how *trustworthy* the orientation is, and goes to 0 for shapes
        whose second moments are isotropic (a circle — but also a square, whose
        pose still matters), where the principal axis carries no information.
    """
    m = cv2.moments(mask_bool.astype(np.uint8), binaryImage=True)
    if m["m00"] < _MIN_MASK_AREA_PX:
        return None
    centroid = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    mu20, mu11, mu02 = m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"]
    if not np.isfinite([mu20, mu11, mu02]).all() or (mu20 + mu02) <= 0:
        return None
    theta = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
    spread = float(np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11**2))
    eccentricity = spread / (mu20 + mu02)
    return centroid, theta, float(m["m00"]), eccentricity


def moment_pose_inits(
    template_mask: np.ndarray,
    mask_bool: np.ndarray,
    init_corners: np.ndarray,
) -> list:
    """Stateless re-initializations: template moments mapped onto the mask's.

    The last-frame homography is a *tracking* initializer — it cannot recover
    once the estimate has drifted off the object, which is where the aligner
    branch loses most of its accuracy (PTi, 2026-08-11: on frames the aligner
    rejects, the stateless Hough-quad branch is still right 76% of the time, so
    the evidence is in the mask and only the initialization is missing). Second
    moments give a similarity transform (translation, rotation, isotropic scale)
    from the template silhouette to the current one, with no quadrilateral
    assumption — moments are defined for any shape.

    The principal axis is a line, not a direction, so its orientation is only
    defined modulo pi: both branches are returned and the caller picks by
    residual. When either silhouette has near-isotropic second moments (a
    square as much as a circle) the axis itself is meaningless, so the quarter
    turns are added too rather than trusting an angle read off numerical noise.

    Args:
        template_mask: (H, W) init-frame object mask.
        mask_bool: (H, W) current segmentation mask.
        init_corners: (2, 4) template corner coordinates.

    Returns:
        List of (2, 4) control-point initializations (possibly empty).
    """
    template_pose = _mask_pose(template_mask > 0)
    current_pose = _mask_pose(mask_bool)
    if template_pose is None or current_pose is None:
        return []
    c_tpl, theta_tpl, area_tpl, ecc_tpl = template_pose
    c_cur, theta_cur, area_cur, ecc_cur = current_pose

    scale = float(np.sqrt(area_cur / area_tpl))
    if not np.isfinite(scale) or scale <= 0:
        return []

    isotropic = min(ecc_tpl, ecc_cur) < _MIN_ECCENTRICITY
    # With no usable axis, the measured angles are noise, not evidence: keep the
    # template's own orientation and let the quarter turns cover the rest.
    base_angle = 0.0 if isotropic else theta_cur - theta_tpl
    turns = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi) if isotropic else (0.0, np.pi)

    inits = []
    for half_turn in turns:
        angle = base_angle + half_turn
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        control = c_cur[:, None] + scale * (rot @ (init_corners - c_tpl[:, None]))
        if np.isfinite(control).all():
            inits.append(control)
    return inits


def align_mask_to_homography(
    mask: np.ndarray,
    init_corners: np.ndarray,
    last_H_init2current: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
    *,
    boundary: Optional[np.ndarray] = None,
    n_boundary_points: int = 150,
    max_iters: int = 60,
    huber_delta_px: float = 5.0,
    step_tol_px: float = 0.05,
    redetect_above_residual_px: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], int, float, int, Optional[np.ndarray]]:
    """Align the template mask onto the current segmentation mask.

    Drop-in mask->H estimator with the same ``(mask, init_corners, last_H)``
    contract as ``hough_homography.mask_to_homography``, initialized at the
    last-frame homography.

    Args:
        mask: (H, W) current binary segmentation mask (bool or uint8).
        init_corners: (2, 4) template corner coordinates.
        last_H_init2current: (3, 3) previous-frame homography (the GN init).
        template_mask: (H, W) init-frame object mask; rasterized from
            ``init_corners`` when None. Using the real init mask drops the
            quadrilateral assumption.
        boundary: Precomputed template boundary (N, 2) from
            :func:`template_boundary`. The template is constant per sequence, so
            a tracking loop should cache it here instead of paying
            ``findContours`` every frame. Computed on demand when None.
        n_boundary_points: Approximate template-boundary sample count.
        max_iters: GN iteration cap.
        huber_delta_px: Huber transition of the boundary residual (px).
        step_tol_px: GN convergence threshold on the control-point step (px).
        redetect_above_residual_px: When set and the last-H solve lands above
            this residual, re-solve from the mask-moment poses
            (:func:`moment_pose_inits`) and keep the best of all starts. This is
            the aligner's redetection path: it costs extra GN solves only on
            frames the caller was going to reject anyway. None keeps the
            single-start behaviour.

    Returns:
        ``(H_init2current, n_ok_points, median_abs_residual_px, n_iters,
        corner_cov)``. ``H_init2current`` is None on failure; ``n_ok_points``
        counts boundary samples within ``huber_delta_px`` of the target
        boundary at the solution (the ``n_corr`` analogue for downstream
        gates); the residual is ``inf`` on failure; ``n_iters`` is the GN
        iteration count actually spent (0 on early failure), for cost
        accounting; ``corner_cov`` is a (4, 2, 2) per-corner pixel covariance
        from the converged fit's own normal equations (see
        :func:`_corner_cov_from_boundary_fit`), or None when it could not be
        computed — same currency as the correspondence branch's
        ``homography_corner_covs``, so the two are directly comparable.
    """
    failed: Tuple[Optional[np.ndarray], int, float, int, Optional[np.ndarray]] = (
        None, 0, float("inf"), 0, None,
    )
    mask_bool = mask > 0
    if int(mask_bool.sum()) < _MIN_MASK_AREA_PX:
        return failed

    if boundary is None:
        if template_mask is None:
            template_mask = corners_to_mask(init_corners, mask_bool.shape)
        boundary = template_boundary(template_mask > 0, n_boundary_points)
    if boundary is None:
        return failed

    control0 = warp_corners(last_H_init2current, init_corners)
    if not np.isfinite(control0).all():
        return failed

    sdf = signed_distance(mask_bool)

    def solve(start: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """One damped-GN solve from ``start``: (control, residuals, iterations)."""
        solved, iters = _gn_align(
            sdf,
            boundary,
            init_corners,
            start,
            max_iters=max_iters,
            huber_delta=huber_delta_px,
            step_tol=step_tol_px,
        )
        return solved, _residuals(sdf, boundary, init_corners, solved), iters

    control, r, n_iters = solve(control0)

    # Redetection: the last-H start is stuck off the object, so try the poses
    # the current mask implies on its own and keep whichever fits best.
    if redetect_above_residual_px is not None and (
        r is None or float(np.median(np.abs(r))) > redetect_above_residual_px
    ):
        if template_mask is None:
            template_mask = corners_to_mask(init_corners, mask_bool.shape)
        for start in moment_pose_inits(template_mask, mask_bool, init_corners):
            cand_control, cand_r, cand_iters = solve(start)
            n_iters += cand_iters
            if cand_r is None:
                continue
            if r is None or np.median(np.abs(cand_r)) < np.median(np.abs(r)):
                control, r = cand_control, cand_r

    if r is None:
        return failed
    try:
        H = cv2.getPerspectiveTransform(
            init_corners.T.astype(np.float32), control.T.astype(np.float32)
        )
    except cv2.error:
        return failed
    residual = float(np.median(np.abs(r)))
    n_ok = int((np.abs(r) <= huber_delta_px).sum())
    corner_cov = _corner_cov_from_boundary_fit(
        sdf, boundary, init_corners, control, r, huber_delta_px
    )
    return H.astype(np.float64), n_ok, residual, n_iters, corner_cov
