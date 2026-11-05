"""Per-corner position uncertainty, in the same units for both branches.

Each branch predicts four corner positions; this module gives each of them a
2x2 covariance in pixels, so the two branches become comparable measurements
instead of two black boxes with a hand-tuned pixel threshold between them.

Two estimators, one per branch, both first principles:

- Segmentation branch (``line_intersection_cov``): a corner is the intersection
  of two fitted boundary lines.  Fit each line by total least squares, turn the
  fit residual into an angle/offset uncertainty, and push both through the
  intersection Jacobian.  Ported from the old flatsam
  ``experiments/corner_confidences.py``.
- Correspondence branch (``homography_corner_covs``): least squares on the
  reprojection residuals of the RANSAC inliers, in the *four-corner*
  parameterization the mask aligner already uses (:mod:`flatsam2.mask_align`).
  The eight parameters are the four corner positions, so the parameter
  covariance *is* the corner covariance — no second propagation step.

``bootstrap_corner_covs`` re-fits the homography on resampled inliers and takes
the scatter of the warped corners.  It is the offline cross-check on the
analytic route, not a second estimator to deploy: it is ~100x more expensive,
and the role it should have played for the line covariance (where it was never
run, so a factor-of-5 bug survived) is exactly the role it plays here.

Why the Jacobians below point the "wrong" way.  Both estimators differentiate the
*residuals with respect to the corners* (dr/dtheta), while the quantity wanted is
the opposite -- how far a corner moves when the evidence moves (dtheta/dy).  The
inversion of the normal matrix is exactly what converts one into the other, and is
not a bookkeeping step.  theta_hat is defined implicitly, as whatever minimizes the
fit, so it cannot be differentiated directly; instead differentiate the condition
that defines it.  At the optimum J^T W r = 0, and r = f(theta) - y, so perturbing
the evidence gives J^T W J dtheta - J^T W dy = 0, i.e.

    dtheta_hat / dy = (J^T W J)^-1 J^T W

and propagating evidence noise through that sensitivity yields sigma^2 (J^T W J)^-1
whenever Cov(y) = sigma^2 W^-1.  In scalar form with r_i = c*theta - y_i: J^T J =
N c^2 so the variance is sigma^2 / (N c^2), while directly dtheta = dy / c -- the
sensitivity of the estimate is the *reciprocal* of the sensitivity of the residual.
J is the stiffness of the fit against moving a corner; the covariance is the
compliance.  Equivalently J^T W J is the Fisher information and covariance is its
inverse: "the evidence goes badly wrong when this corner moves" is what it means
for that evidence to be informative about the corner.

Caveat this makes visible: the collapse to sigma^2 (J^T W J)^-1 assumes the weights
are inverse variances.  ``mask_align``'s W holds Huber weights, chosen for
robustness rather than as a noise model, so the sandwich does not strictly collapse
there -- one of the reasons the deployed covariances need empirical scale
correction (see docs/tracker-cards.html section 3).

What the covariance does and does not model: it is a *precision* model.  It says
how much the corner would move under resampling of the evidence the branch used.
It says nothing about the evidence being the wrong object — a distractor gives a
tight, confident, wrong ellipse.  Gross-error rejection is a separate mechanism
(see docs/correction_research.md section 9).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from einops import rearrange


# Finite-difference step for the corner Jacobian (px). Half a pixel matches the
# mask aligner's; smaller steps lose precision to cancellation, not gain it.
_JACOBIAN_STEP_PX = 0.5
# Bootstrap resamples — offline cross-check only, never on the tracker's path.
_N_BOOTSTRAP = 100
# A homography needs 4 correspondences; below this a fit is degenerate.
_MIN_CORRESPONDENCES = 8
# A fit claiming sub-this-std confidence is not credible: pixel quantization and
# model-form error put a floor under any real measurement. Without it, a
# near-singular normal matrix (near-collinear evidence, or a boundary arc too
# short to pin an axis) gives a covariance with one eigenvalue near zero, which
# blows up any Mahalanobis distance computed against it to numerical nonsense
# (observed in the field: disagreement scores up to ~1e8 on some sequences).
_MIN_STD_PX = 0.25
# An eigenvalue this far below zero (relative to the largest) is a failed fit
# rather than round-off, and must not be floored into false confidence.
_NEGATIVE_EIGENVALUE_TOL = 1e-8
# The normal matrix is PSD by construction; a condition number worse than this
# means some direction is unconstrained by the evidence.
_MIN_EIGENVALUE_RATIO = 1e-12


def _floor_cov_eigenvalues(cov: np.ndarray, min_std_px: float = _MIN_STD_PX) -> np.ndarray:
    """Clamp a 2x2 covariance's eigenvalues from below at ``min_std_px**2``.

    Same eigenvectors (same *shape* of uncertainty), floored magnitude. Applied
    to every per-corner covariance this module and :mod:`flatsam2.mask_align`
    return, so no near-singular fit can masquerade as near-certain.

    Args:
        cov: (2, 2) covariance.
        min_std_px: Minimum credible per-axis standard deviation (px).

    Returns:
        (2, 2) covariance, symmetric, eigenvalues >= ``min_std_px**2``.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Flooring is for a fit that is legitimately precise, not for one that came
    # back indefinite. A materially negative eigenvalue means the normal matrix
    # was not usable, and clamping it to the floor would report the *most
    # confident possible* uncertainty in exactly the direction the fit failed —
    # which then wins the arbitration (review, 2026-08-19). Round-off around
    # zero is fine to clamp; a real negative is not.
    scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    if eigvals.min() < -_NEGATIVE_EIGENVALUE_TOL * max(scale, 1.0):
        return None
    eigvals = np.maximum(eigvals, min_std_px**2)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def invert_normal_matrix(normal: np.ndarray, var: float) -> Optional[np.ndarray]:
    """``var * normal^-1``, or None when the normal matrix cannot support it.

    ``J^T W J`` is symmetric positive semi-definite by construction, so a
    non-positive or numerically hopeless spectrum means the evidence does not
    determine all eight parameters — a boundary arc too short to pin an axis, a
    correspondence set collapsed onto a line. Inverting anyway produces a
    covariance whose "confidence" is numerical noise.

    Args:
        normal: (n, n) normal matrix ``J^T W J``.
        var: Residual variance estimate multiplying the inverse.

    Returns:
        (n, n) parameter covariance, or None when the fit is degenerate.
    """
    eigvals = np.linalg.eigvalsh(normal)
    largest = float(eigvals[-1])
    if not np.isfinite(largest) or largest <= 0.0:
        return None
    if float(eigvals[0]) <= largest * _MIN_EIGENVALUE_RATIO:
        return None
    try:
        cov = var * np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    return cov if np.isfinite(cov).all() else None


def corners_to_theta(corners: np.ndarray) -> np.ndarray:
    """(2, 4) corners -> (8,) parameter vector, corner-major.

    The one place the parameter layout is defined: ``(x1, y1, x2, y2, x3, y3,
    x4, y4)``. Both estimators build their Jacobians against this vector and
    both read per-corner blocks back out of it with :func:`corner_blocks`, so
    the layout cannot drift apart between them.

    Args:
        corners: (2, 4) corner coordinates, ``[[x...], [y...]]``.

    Returns:
        (8,) parameter vector.
    """
    return rearrange(corners, "xy corners -> (corners xy)")


def theta_to_corners(theta: np.ndarray) -> np.ndarray:
    """(8,) corner-major parameter vector -> (2, 4) corners. Inverse of the above."""
    return rearrange(theta, "(corners xy) -> xy corners", xy=2)


def corner_blocks(theta_cov: np.ndarray) -> np.ndarray:
    """(8, 8) parameter covariance -> (4, 2, 2) per-corner covariances.

    Corner k's coordinates sit at rows/cols ``2k, 2k+1`` of a vector built by
    :func:`corners_to_theta`; taking that diagonal block is only a per-corner
    covariance under that layout. Eigenvalues are floored, so no near-singular
    fit is reported as near-certain.

    Args:
        theta_cov: (8, 8) covariance of a corner-major parameter vector.

    Returns:
        (4, 2, 2) per-corner covariances in pixels, or None when any block came
        back indefinite — a fit that cannot support a covariance at all.
    """
    blocks = [
        _floor_cov_eigenvalues(theta_cov[2 * k : 2 * k + 2, 2 * k : 2 * k + 2])
        for k in range(4)
    ]
    if any(b is None for b in blocks):
        # One degenerate corner condemns the fit: the branch reports no
        # covariance, which the fuser already reads as "no opinion".
        return None
    return np.stack(blocks)


def fit_line_tls(
    pts: np.ndarray,
    noise_px: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Total-least-squares line fit with angle and offset uncertainties.

    Args:
        pts: (2, N) points on the line, ``[[x...], [y...]]``.
        noise_px: Assumed perpendicular point noise (px).  ``None`` estimates it
            from the fit residual itself, which is the principled default but
            under-reports whenever the boundary is systematically wrong (mask
            spill, occlusion) rather than noisy.

    Returns:
        ``(centroid (2,), unit normal (2,), sigma_phi, sigma_rho)`` where
        ``sigma_phi`` is the line-angle std (rad) and ``sigma_rho`` the
        perpendicular-offset std (px).

    Raises:
        ValueError: When fewer than two points are given.
    """
    n_pts = pts.shape[1]
    if n_pts < 2:
        raise ValueError(f"need >= 2 points for a line fit, got {n_pts}")

    centroid = pts.mean(axis=1)
    centered = pts - centroid[:, None]
    scatter = (centered @ centered.T) / n_pts

    eigvals, eigvecs = np.linalg.eigh(scatter)  # ascending
    normal = eigvecs[:, 0]
    spread_along_line = float(eigvals[1])

    var = noise_px**2 if noise_px is not None else max(float(eigvals[0]), 1e-12)
    sigma_phi = float(np.sqrt(var / (n_pts * max(spread_along_line, 1e-12))))
    sigma_rho = float(np.sqrt(var / n_pts))
    return centroid, normal, sigma_phi, sigma_rho


def _line_params(centroid: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Centroid + normal -> ``(phi, rho)`` normal-form line parameters."""
    return float(np.arctan2(normal[1], normal[0])), float(-normal @ centroid)


def _intersect(phi_a: float, rho_a: float, phi_b: float, rho_b: float) -> np.ndarray:
    """Intersection (2,) of two lines given in ``(phi, rho)`` normal form."""
    A = np.array([[np.cos(phi_a), np.sin(phi_a)], [np.cos(phi_b), np.sin(phi_b)]])
    return np.linalg.solve(A, -np.array([rho_a, rho_b]))


def _intersection_jacobian(
    phi_a: float, rho_a: float, phi_b: float, rho_b: float
) -> np.ndarray:
    """(2, 4) Jacobian of the intersection wrt ``(phi_a, rho_a, phi_b, rho_b)``."""
    A = np.array([[np.cos(phi_a), np.sin(phi_a)], [np.cos(phi_b), np.sin(phi_b)]])
    A_inv = np.linalg.inv(A)
    point = _intersect(phi_a, rho_a, phi_b, rho_b)

    jac = np.zeros((2, 4))
    dA_phi_a = np.array([[-np.sin(phi_a), np.cos(phi_a)], [0.0, 0.0]])
    jac[:, 0] = -A_inv @ dA_phi_a @ point
    jac[:, 1] = -A_inv[:, 0]
    dA_phi_b = np.array([[0.0, 0.0], [-np.sin(phi_b), np.cos(phi_b)]])
    jac[:, 2] = -A_inv @ dA_phi_b @ point
    jac[:, 3] = -A_inv[:, 1]
    return jac


def _line_param_cov(
    phi: float,
    centroid: np.ndarray,
    sigma_phi: float,
    sigma_rho: float,
) -> np.ndarray:
    """(2, 2) covariance of ``(phi, rho)`` for one TLS-fitted line.

    The fit's two uncertainties are independent in the line's *own* frame: the
    angle pivots about the centroid, the offset shifts perpendicular to it.  The
    ``rho`` of the normal form is measured from the image origin instead, so
    ``rho = -n(phi) . centroid - s`` picks up a dependence on ``phi`` with a lever
    arm equal to the centroid's distance from the origin.  Treating the pair as
    independent (a diagonal matrix in ``(phi, rho)``) therefore double-counts the
    pivot and inflates the corner covariance by a factor that grows with the
    distance from the origin — several-fold at typical image coordinates.
    """
    dn_dphi = np.array([-np.sin(phi), np.cos(phi)])
    jac = np.array([[1.0, 0.0], [-float(dn_dphi @ centroid), -1.0]])
    return jac @ np.diag([sigma_phi**2, sigma_rho**2]) @ jac.T


def line_intersection_cov(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    noise_px: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner position and 2x2 covariance from the two boundary lines meeting there.

    Args:
        pts_a: (2, N) boundary points assigned to the first line.
        pts_b: (2, M) boundary points assigned to the second line.
        noise_px: Perpendicular point noise; see :func:`fit_line_tls`.

    Returns:
        ``(corner (2,), covariance (2, 2))`` in pixels.  Nearly parallel lines
        blow the covariance up along the shared direction, which is the correct
        answer, not a failure.
    """
    centroid_a, normal_a, sigma_phi_a, sigma_rho_a = fit_line_tls(pts_a, noise_px)
    centroid_b, normal_b, sigma_phi_b, sigma_rho_b = fit_line_tls(pts_b, noise_px)
    phi_a, rho_a = _line_params(centroid_a, normal_a)
    phi_b, rho_b = _line_params(centroid_b, normal_b)

    corner = _intersect(phi_a, rho_a, phi_b, rho_b)
    param_cov = np.zeros((4, 4))
    param_cov[:2, :2] = _line_param_cov(phi_a, centroid_a, sigma_phi_a, sigma_rho_a)
    param_cov[2:, 2:] = _line_param_cov(phi_b, centroid_b, sigma_phi_b, sigma_rho_b)
    jac = _intersection_jacobian(phi_a, rho_a, phi_b, rho_b)
    return corner, jac @ param_cov @ jac.T


def _warp(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    proj = H @ np.hstack([pts, np.ones((len(pts), 1))]).T
    return (proj[:2] / proj[2]).T


def _h_from_corners(init_corners: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    """Exact 4-point homography (2, 4) -> (2, 4), or None when degenerate."""
    try:
        H = cv2.getPerspectiveTransform(
            init_corners.T.astype(np.float32), corners.T.astype(np.float32)
        )
    except cv2.error:
        return None
    return H.astype(np.float64) if np.isfinite(H).all() else None


def homography_corner_covs(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    init_corners: np.ndarray,
    corners: np.ndarray,
    step_px: float = _JACOBIAN_STEP_PX,
) -> Optional[np.ndarray]:
    """Per-corner 2x2 covariance of a correspondence-based homography.

    Linear least squares on the reprojection residuals, parameterized by the four
    corner positions themselves (the ``mask_align`` parameterization).  With that
    choice the parameter covariance ``sigma^2 (J^T J)^-1`` already lives in corner
    space, and the four 2x2 diagonal blocks are what we want.  The residual scale
    is estimated from the fit: ``sigma^2 = sum |r|^2 / (2N - 8)``.

    This is why the occlusion case comes out right without a coverage term.  What
    enters ``J`` is how much each corner has to move to change the prediction at
    the correspondences: inliers packed against one corner barely constrain the
    far ones, so those get a large covariance, while a dense two-dimensional
    patch pins the geometry down even when it covers a quarter of the target.

    Args:
        pts_src: (N, 2) template-frame inlier points.
        pts_dst: (N, 2) current-frame inlier points, row-aligned with ``pts_src``.
        init_corners: (2, 4) template corners, ``[[x...], [y...]]``.
        corners: (2, 4) the fit's predicted current-frame corners — the point the
            residuals are linearized about.
        step_px: Finite-difference step for the Jacobian.  Central differences on
            an exactly-linear-in-nothing but very smooth map; sub-pixel steps are
            numerically worse, not better.

    Returns:
        (4, 2, 2) per-corner covariances in pixels, or ``None`` when there are too
        few correspondences or the configuration is degenerate.
    """
    n_pts = len(pts_src)
    if n_pts < _MIN_CORRESPONDENCES or len(pts_dst) != n_pts:
        return None

    def residual(theta: np.ndarray) -> Optional[np.ndarray]:
        H = _h_from_corners(init_corners, theta_to_corners(theta))
        if H is None:
            return None
        return (_warp(H, pts_src) - pts_dst).ravel()

    theta0 = corners_to_theta(corners)
    r0 = residual(theta0)
    if r0 is None:
        return None

    jac = np.zeros((2 * n_pts, 8))
    for p in range(8):
        plus, minus = theta0.copy(), theta0.copy()
        plus[p] += step_px
        minus[p] -= step_px
        r_plus, r_minus = residual(plus), residual(minus)
        if r_plus is None or r_minus is None:
            return None
        jac[:, p] = (r_plus - r_minus) / (2.0 * step_px)

    dof = 2 * n_pts - 8
    var = float(r0 @ r0) / dof
    theta_cov = invert_normal_matrix(jac.T @ jac, var)
    if theta_cov is None:
        return None
    return corner_blocks(theta_cov)


def bootstrap_corner_covs(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    init_corners: np.ndarray,
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Resampling cross-check on :func:`homography_corner_covs` — offline only.

    Resamples the correspondences with replacement, re-fits the homography on
    each resample, warps the template corners through it, and takes the
    covariance of the resulting corner scatter.

    Do not put this on the tracker's path: it is ~100 homography fits per frame
    against one linear solve, and it answers the same question to first order.
    Its value is that it makes a *different* modelling assumption — it resamples
    the correspondence set, where the analytic route conditions on the inliers
    RANSAC chose and propagates only their residuals.  Agreement between the two
    is evidence the analytic covariance is right; a systematic gap localizes what
    the analytic model leaves out.

    Args:
        pts_src: (N, 2) template-frame inlier points.
        pts_dst: (N, 2) current-frame inlier points, row-aligned with ``pts_src``.
        init_corners: (2, 4) template corners, ``[[x...], [y...]]``.
        n_bootstrap: Number of resamples.
        seed: RNG seed (fixed by default — the tracker is deterministic).

    Returns:
        (4, 2, 2) per-corner covariances in pixels, or ``None`` when there are
        too few correspondences or too few resamples produced a usable fit.
    """
    n_pts = len(pts_src)
    if n_pts < _MIN_CORRESPONDENCES or len(pts_dst) != n_pts:
        return None

    rng = np.random.default_rng(seed)
    warped: list[np.ndarray] = []
    corners_h = np.vstack([init_corners, np.ones((1, 4))])
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_pts, size=n_pts)
        H, _ = cv2.findHomography(pts_src[idx], pts_dst[idx], 0)
        if H is None or not np.isfinite(H).all():
            continue
        proj = H @ corners_h
        if np.any(np.abs(proj[2]) < 1e-9):
            continue
        warped.append(proj[:2] / proj[2])

    if len(warped) < n_bootstrap // 2:
        return None
    samples = np.stack(warped)  # (B, 2, 4)
    return np.stack([np.cov(samples[:, :, k], rowvar=False) for k in range(4)])
