"""Tests for per-corner covariance fusion of LoMa + SAM-H (``corner_fusion``)."""

import numpy as np

from flatsam2.datasets import warp_corners
from flatsam2.tracking.corner_fusion import fuse_corners

_INIT_CORNERS = np.array(
    [[0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 10.0, 10.0]]
)  # (2, 4) axis-aligned square


def _translation_H(dx: float, dy: float) -> np.ndarray:
    """A trivial 3x3 homography that just translates by (dx, dy)."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _iso_cov(std_px: float) -> np.ndarray:
    """(4, 2, 2) isotropic per-corner covariance, same for all 4 corners."""
    single = np.eye(2) * std_px**2
    return np.stack([single] * 4)


def test_agreement_pulls_toward_the_confident_branch():
    """When branches agree, fuse by precision -- closer to the tighter one."""
    loma_H = _translation_H(5.0, 5.0)
    samh_H = _translation_H(5.5, 5.0)  # 0.5 px away: well within any sane gate
    loma_cov = _iso_cov(1.0)  # confident
    samh_cov = _iso_cov(9.0)  # much less confident
    last_corners = _INIT_CORNERS.copy()

    fused, telemetry = fuse_corners(
        _INIT_CORNERS, loma_H, loma_cov, samh_H, samh_cov, last_corners,
        disagreement_gate=20.0,
    )

    assert telemetry.modes == ("fused", "fused", "fused", "fused")
    loma_pts = warp_corners(loma_H, _INIT_CORNERS)
    samh_pts = warp_corners(samh_H, _INIT_CORNERS)
    for k in range(4):
        d_loma = np.linalg.norm(fused[:, k] - loma_pts[:, k])
        d_samh = np.linalg.norm(fused[:, k] - samh_pts[:, k])
        assert d_loma < d_samh, f"corner {k}: fused should sit closer to the confident branch"


def test_disagreement_picks_one_branch_outright_not_a_blend():
    """Beyond the gate, trust the more confident branch exactly -- no averaging."""
    loma_H = _translation_H(5.0, 5.0)
    samh_H = _translation_H(50.0, 5.0)  # far away: no sane gate should fuse this
    loma_cov = _iso_cov(1.0)  # tighter trace -> wins the tie-break
    samh_cov = _iso_cov(3.0)
    last_corners = _INIT_CORNERS.copy()

    fused, telemetry = fuse_corners(
        _INIT_CORNERS, loma_H, loma_cov, samh_H, samh_cov, last_corners,
        disagreement_gate=5.99,
    )

    assert telemetry.modes == ("gated_confident",) * 4
    loma_pts = warp_corners(loma_H, _INIT_CORNERS)
    np.testing.assert_allclose(fused, loma_pts)


def test_missing_branch_defers_fully_to_the_other():
    """A branch that produced nothing (H=None) contributes nothing."""
    loma_H = _translation_H(3.0, -2.0)
    loma_cov = _iso_cov(1.0)
    last_corners = _INIT_CORNERS.copy()

    fused, telemetry = fuse_corners(
        _INIT_CORNERS, loma_H, loma_cov, None, None, last_corners,
        disagreement_gate=5.99,
    )

    assert telemetry.modes == ("loma_only_cov",) * 4
    np.testing.assert_allclose(fused, warp_corners(loma_H, _INIT_CORNERS))


def test_both_missing_coasts_on_last_corners():
    """Neither branch has anything to say -- hold the previous fused position."""
    last_corners = _INIT_CORNERS + np.array([[2.0], [-1.0]])

    fused, telemetry = fuse_corners(
        _INIT_CORNERS, None, None, None, None, last_corners,
        disagreement_gate=5.99,
    )

    assert telemetry.modes == ("coast",) * 4
    np.testing.assert_allclose(fused, last_corners)


def test_homography_without_covariance_is_unmeasured_not_free():
    """A branch with a point but no covariance must not win by default.

    If it were treated as free/infinite precision instead of zero, it would
    dominate the fused result; instead the other (measured) branch must be
    used exclusively.
    """
    loma_H = _translation_H(5.0, 5.0)  # has a homography...
    samh_H = _translation_H(5.5, 5.0)
    samh_cov = _iso_cov(2.0)
    last_corners = _INIT_CORNERS.copy()

    fused, telemetry = fuse_corners(
        _INIT_CORNERS, loma_H, None, samh_H, samh_cov, last_corners,
        disagreement_gate=5.99,
    )

    assert telemetry.modes == ("samh_only_cov",) * 4
    np.testing.assert_allclose(fused, warp_corners(samh_H, _INIT_CORNERS))


def _fake_tracker(occlusion_max_recall: float, template: np.ndarray, last_H: np.ndarray):
    """Minimal stand-in for FlatSAMTracker: the two attributes the screen reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        cfg=SimpleNamespace(
            fusion=SimpleNamespace(
                occlusion_max_recall=occlusion_max_recall,
                occlusion_min_precision=0.90,
            )
        ),
        state=SimpleNamespace(init_mask=template, last_homography=last_H),
    )


def _template_mask(size: int = 60) -> np.ndarray:
    """(size, size) uint8 mask: a 40x40 square, the object's whole silhouette."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    return mask


def test_occlusion_screen_fires_on_a_clean_partial_view():
    """Half the object visible, in the predicted place -- the mask branch abstains."""
    from flatsam2.tracking.corner_fusion import _mask_is_partial_view

    template = _template_mask()
    half = template.copy()
    half[30:, :] = 0  # bottom half hidden by an occluder

    trk = _fake_tracker(0.80, template, np.eye(3))

    assert _mask_is_partial_view(trk, half, np.eye(3))


def test_occlusion_screen_ignores_a_complete_view_and_a_displaced_one():
    """Neither a full mask nor a mask somewhere else is a partial view.

    A full mask has nothing hidden (recall ~1), and a mask that has moved off
    the prediction loses *precision*, which is drift or a wrong object rather
    than occlusion -- both cases the screen is not entitled to judge.
    """
    from flatsam2.tracking.corner_fusion import _mask_is_partial_view

    template = _template_mask()
    trk = _fake_tracker(0.80, template, np.eye(3))

    assert not _mask_is_partial_view(trk, template.copy(), np.eye(3))

    displaced = np.zeros_like(template)
    displaced[10:50, 45:] = 255  # mostly outside the predicted silhouette
    assert not _mask_is_partial_view(trk, displaced, np.eye(3))


def test_occlusion_screen_predicts_from_the_keypoint_pose_not_the_previous_frame():
    """Fast motion must not read as occlusion: the prediction follows LoMa.

    The object has receded since the previous frame, so it now covers a quarter
    of the area last frame's pose predicts. Judged against that stale pose the
    complete mask looks like a clean partial view -- high precision, recall
    0.25 -- and the mask branch would abstain for no reason. Judged against the
    keypoint branch's current-frame pose it is exactly what is expected.
    """
    from flatsam2.tracking.corner_fusion import _mask_is_partial_view

    template = np.zeros((160, 160), dtype=np.uint8)
    template[20:100, 20:100] = 255  # 80x80 silhouette in the init frame

    shrunk = np.zeros((160, 160), dtype=np.uint8)
    shrunk[20:60, 20:60] = 255  # half the size, still inside the stale prediction

    half_scale = np.array([[0.5, 0.0, 10.0], [0.0, 0.5, 10.0], [0.0, 0.0, 1.0]])
    trk = _fake_tracker(0.80, template, np.eye(3))  # last frame: full size

    assert not _mask_is_partial_view(trk, shrunk, half_scale)
    assert _mask_is_partial_view(trk, shrunk, None)  # the stale-pose failure mode
