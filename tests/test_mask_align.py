"""Tests for the P3' direct mask -> homography aligner."""

import cv2
import numpy as np

from flatsam2.mask_align import (
    align_mask_to_homography,
    moment_pose_inits,
    template_boundary,
)
from flatsam2.datasets import corners_to_mask, warp_corners
from flatsam2.metrics import alignment_error, best_rotation_error


_HW = (240, 320)
_INIT_CORNERS = np.array(
    [[100.0, 220.0, 220.0, 100.0], [60.0, 60.0, 180.0, 180.0]]
)  # (2, 4) axis-aligned rectangle


def _homography_from_corners(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact 4-point homography (2, 4) -> (2, 4)."""
    return cv2.getPerspectiveTransform(
        src.T.astype(np.float32), dst.T.astype(np.float32)
    ).astype(np.float64)


def _shifted_gt() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ground-truth warp (translation + mild scale), its mask, and corners."""
    gt_corners = _INIT_CORNERS * 0.9 + np.array([[25.0], [12.0]])
    H_gt = _homography_from_corners(_INIT_CORNERS, gt_corners)
    mask = corners_to_mask(gt_corners, _HW)
    return H_gt, gt_corners, mask


def test_align_recovers_translation_offset():
    """A 10 px translated init converges back onto the target mask."""
    _H_gt, gt_corners, mask = _shifted_gt()
    H_init = _homography_from_corners(
        _INIT_CORNERS, gt_corners + np.array([[10.0], [-8.0]])
    )
    H, n_ok, residual, _iters, _cov = align_mask_to_homography(mask, _INIT_CORNERS, H_init)
    assert H is not None
    err = alignment_error(gt_corners, warp_corners(H, _INIT_CORNERS))
    assert err < 2.0, f"corner error {err:.2f} px"
    assert residual < 2.0
    assert n_ok > 100


def test_align_recovers_rotation():
    """A 15 deg rotated init converges back (inside the measured basin)."""
    _H_gt, gt_corners, mask = _shifted_gt()
    centroid = gt_corners.mean(axis=1, keepdims=True)
    theta = np.deg2rad(15.0)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated = R @ (gt_corners - centroid) + centroid
    H_init = _homography_from_corners(_INIT_CORNERS, rotated)
    H, _n_ok, residual, _iters, _cov = align_mask_to_homography(mask, _INIT_CORNERS, H_init)
    assert H is not None
    err = alignment_error(gt_corners, warp_corners(H, _INIT_CORNERS))
    assert err < 2.0, f"corner error {err:.2f} px"
    assert residual < 2.0


def test_align_rejects_empty_mask():
    """An (almost) empty target mask fails cleanly instead of diverging."""
    mask = np.zeros(_HW, dtype=np.uint8)
    H_init = np.eye(3)
    H, n_ok, residual, _iters, _cov = align_mask_to_homography(mask, _INIT_CORNERS, H_init)
    assert H is None
    assert n_ok == 0
    assert not np.isfinite(residual)


def test_cached_boundary_matches_on_the_fly_contour():
    """Passing a precomputed boundary is a pure speed optimization, not a change.

    The tracker contours the template once per sequence instead of once per
    frame; that must not move the solution at all.
    """
    _H_gt, gt_corners, mask = _shifted_gt()
    H_init = _homography_from_corners(
        _INIT_CORNERS, gt_corners + np.array([[10.0], [-8.0]])
    )
    template = corners_to_mask(_INIT_CORNERS, _HW)
    H_ref, n_ref, res_ref, iters_ref, _cov_ref = align_mask_to_homography(
        mask, _INIT_CORNERS, H_init, template_mask=template
    )
    cached = template_boundary(template > 0, 150)
    H_cached, n_cached, res_cached, iters_cached, _cov_cached = align_mask_to_homography(
        mask, _INIT_CORNERS, H_init, template_mask=template, boundary=cached
    )
    assert H_ref is not None and H_cached is not None
    np.testing.assert_allclose(H_cached, H_ref, rtol=0, atol=0)
    assert (n_cached, res_cached, iters_cached) == (n_ref, res_ref, iters_ref)


def test_align_reports_iteration_count():
    """Iteration count is reported and bounded by the configured cap."""
    _H_gt, gt_corners, mask = _shifted_gt()
    H_init = _homography_from_corners(
        _INIT_CORNERS, gt_corners + np.array([[10.0], [-8.0]])
    )
    _H, _n_ok, _residual, iters, _cov = align_mask_to_homography(
        mask, _INIT_CORNERS, H_init, max_iters=7
    )
    assert 1 <= iters <= 7
    # A failure before the GN loop reports zero cost, not a bogus count.
    _H2, _n2, _r2, iters_empty, _cov2 = align_mask_to_homography(
        np.zeros(_HW, dtype=np.uint8), _INIT_CORNERS, np.eye(3)
    )
    assert iters_empty == 0


def test_align_uses_arbitrary_template_mask():
    """A non-quad (elliptical) template aligns onto its warped rasterization.

    An ellipse is a conic, stabilized by a 3-parameter homography subgroup, so
    the GT corner pose is NOT identifiable from mask alignment alone — assert
    mask agreement (what the aligner guarantees), not corner recovery.
    """
    template = np.zeros(_HW, dtype=np.uint8)
    cv2.ellipse(template, (160, 120), (70, 45), 0, 0, 360, 255, -1)
    gt_corners = _INIT_CORNERS + np.array([[14.0], [-6.0]])
    H_gt = _homography_from_corners(_INIT_CORNERS, gt_corners)
    target = cv2.warpPerspective(template, H_gt.astype(np.float32), _HW[::-1])
    H, _n_ok, residual, _iters, _cov = align_mask_to_homography(
        target, _INIT_CORNERS, np.eye(3), template_mask=template
    )
    assert H is not None
    assert residual < 2.0
    # Binarize at mid-level: warpPerspective's bilinear fringe dilates a >0
    # threshold by ~1 px on each mask. IoU on this shape loses ~4% per px of
    # boundary offset, so 0.93 tolerates the ~1 px rasterization + convergence
    # slack while still failing on any real misalignment (5 px -> IoU ~0.83).
    warped = cv2.warpPerspective(template, H.astype(np.float32), _HW[::-1]) > 127
    tgt = target > 127
    iou = float((warped & tgt).sum()) / float((warped | tgt).sum())
    assert iou > 0.93, f"mask IoU {iou:.3f}"


# An elongated target: its principal axis carries real orientation evidence,
# which a square (isotropic second moments) does not.
_TRAPEZOID_CORNERS = np.array(
    [[60.0, 220.0, 195.0, 85.0], [95.0, 95.0, 145.0, 145.0]]
)


def _rotate_about(corners: np.ndarray, degrees: float, center: np.ndarray) -> np.ndarray:
    """Rotate (2, 4) corners about a point."""
    angle = np.deg2rad(degrees)
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return center[:, None] + rot @ (corners - center[:, None])


def test_moment_inits_land_near_the_target():
    """Mask moments alone put the template close enough for GN to finish."""
    _H_gt, gt_corners, mask = _shifted_gt()
    template = corners_to_mask(_INIT_CORNERS, _HW)

    inits = moment_pose_inits(template, mask > 0, _INIT_CORNERS)

    # The template here is a square: isotropic second moments, so the axis
    # angle is meaningless and all four quarter turns are offered.
    assert len(inits) == 4
    best = min(float(alignment_error(gt_corners, c)) for c in inits)
    assert best < 10.0, f"best moment init off by {best:.1f} px"


def test_moment_inits_use_the_axis_when_the_shape_has_one():
    """An elongated silhouette gets the two axis branches, not four turns."""
    template = np.zeros(_HW, dtype=np.uint8)
    cv2.ellipse(template, (160, 120), (80, 30), 0, 0, 360, 255, -1)
    target = np.zeros(_HW, dtype=np.uint8)
    cv2.ellipse(target, (150, 130), (80, 30), 25, 0, 360, 255, -1)

    inits = moment_pose_inits(template, target > 0, _INIT_CORNERS)

    assert len(inits) == 2


def test_redetection_recovers_from_a_lost_init():
    """Past the convergence basin the last-H start cannot recover; moments can.

    The object has turned 80 degrees since the frame the tracker last got
    right — far outside the aligner's measured ~20 degree basin. Refining from
    the last homography lands on a wrong pose (a homography has enough freedom
    to sit the template on this silhouette with the corners permuted), while
    the current mask's own moments point at the right one.
    """
    gt_corners = _rotate_about(_TRAPEZOID_CORNERS, 80.0, np.array([160.0, 120.0]))
    mask = corners_to_mask(gt_corners, _HW)
    lost_H = np.eye(3)  # still on the template pose; the object has turned away

    stuck, _n_ok, stuck_residual, _iters, _cov = align_mask_to_homography(
        mask, _TRAPEZOID_CORNERS, lost_H
    )
    recovered, _n_ok2, residual, _iters2, _cov2 = align_mask_to_homography(
        mask, _TRAPEZOID_CORNERS, lost_H, redetect_above_residual_px=0.05
    )

    assert alignment_error(gt_corners, warp_corners(stuck, _TRAPEZOID_CORNERS)) > 15.0
    assert residual <= stuck_residual
    # Pose recovered up to the corner relabeling the boundary objective cannot
    # see (module docstring, "identifiability caveat"): the silhouette is back
    # on the object, which is what redetection is for. Which of the labelings
    # is the right one is not information this branch has.
    recovered_err, _shift = best_rotation_error(
        gt_corners, warp_corners(recovered, _TRAPEZOID_CORNERS)
    )
    assert recovered_err < 5.0


def test_redetection_leaves_a_good_fit_alone():
    """A fit already under the threshold is returned untouched (no extra cost)."""
    _H_gt, gt_corners, mask = _shifted_gt()
    H_init = _homography_from_corners(
        _INIT_CORNERS, gt_corners + np.array([[6.0], [-4.0]])
    )

    plain = align_mask_to_homography(mask, _INIT_CORNERS, H_init)
    with_redetect = align_mask_to_homography(
        mask, _INIT_CORNERS, H_init, redetect_above_residual_px=6.0
    )

    np.testing.assert_allclose(with_redetect[0], plain[0])
    assert with_redetect[3] == plain[3]  # same iteration count: no extra solves


def _imperfect_target(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A target mask the template cannot fit exactly, and the pose to fit it from.

    A mask rasterized straight from the template's own warp leaves ~0 residual, so
    ``sigma^2`` collapses and every corner covariance sits on the eigenvalue floor
    — no information, nothing to test. Jittering the target's corners by ~1.5 px
    gives the fit a realistic residual scale.
    """
    rng = np.random.default_rng(seed)
    gt_corners = _INIT_CORNERS * 0.9 + np.array([[25.0], [12.0]])
    mask = corners_to_mask(gt_corners + rng.normal(0.0, 1.5, gt_corners.shape), _HW)
    return mask, _homography_from_corners(_INIT_CORNERS, gt_corners)


def _template_with_edge_removed(edge: int, n_keep: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Template mask boundary with one of the four edges nearly deleted.

    Args:
        edge: Edge index; edge ``e`` runs from corner ``e`` to corner ``(e+1) % 4``.
        n_keep: Boundary samples left on that edge, so the fit stays non-singular.

    Returns:
        ``(template_mask, boundary)`` — the full template mask and its boundary
        samples minus the removed stretch.
    """
    # 60 samples, not the tracker's 150: a densely sampled outline determines the
    # pose so well that every eigenvalue lands on the 0.25 px floor and the four
    # corners become indistinguishable, which would make this test vacuous.
    template_mask = corners_to_mask(_INIT_CORNERS, _HW)
    boundary = template_boundary(template_mask > 0, 60)
    start, end = _INIT_CORNERS[:, edge], _INIT_CORNERS[:, (edge + 1) % 4]
    direction = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-direction[1], direction[0]])
    on_edge = np.abs((boundary - start) @ normal) < 2.0
    keep = ~on_edge
    keep[np.flatnonzero(on_edge)[:: max(1, on_edge.sum() // n_keep)]] = True
    return template_mask, boundary[keep]


def test_boundary_cov_grows_for_the_corners_of_a_missing_edge():
    """Delete one edge of the outline; its two corners get the loosest ellipses.

    Runs over all four edges, so a per-corner block that was really a mixture of
    two corners' coordinates cannot pass by accident.
    """
    mask, H_gt = _imperfect_target()
    for edge in range(4):
        template_mask, boundary = _template_with_edge_removed(edge)
        _H, _n_ok, _res, _iters, cov = align_mask_to_homography(
            mask, _INIT_CORNERS, H_gt, template_mask=template_mask, boundary=boundary
        )
        assert cov is not None, f"edge {edge}: no covariance"
        traces = [float(np.trace(c)) for c in cov]
        loose = set(np.argsort(traces)[-2:])
        assert loose == {edge, (edge + 1) % 4}, f"edge {edge}: traces {traces}"


def test_boundary_cov_is_elongated_across_the_missing_edge():
    """The corners of a missing horizontal edge lose their *y*, not their *x*.

    Checks the shape of the ellipse, not just its size: the deleted edge is the
    top one (corners 0 and 1, constant y), so those two corners must be free in y
    and still pinned in x by the side edges.
    """
    mask, H_gt = _imperfect_target()
    template_mask, boundary = _template_with_edge_removed(0)
    _H, _n_ok, _res, _iters, cov = align_mask_to_homography(
        mask, _INIT_CORNERS, H_gt, template_mask=template_mask, boundary=boundary
    )
    assert cov is not None
    for k in (0, 1):
        assert cov[k][1, 1] > 1.8 * cov[k][0, 0], f"corner {k} should be loose in y: {cov[k]}"
    for k in (2, 3):
        assert cov[k][1, 1] < 1.2 * cov[k][0, 0], f"corner {k} should stay round: {cov[k]}"
