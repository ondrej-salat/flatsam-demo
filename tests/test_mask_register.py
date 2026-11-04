"""Tests for the quad-free propose/refine/verify mask registration."""

import cv2
import numpy as np

from flatsam2.datasets import corners_to_mask, warp_corners
from flatsam2.mask_register import (
    RegisterConfig,
    RegistrationTemplate,
    moment_affine_inits,
    register_mask,
    second_moments,
    verification_score,
)
from flatsam2.metrics import alignment_error


_HW = (240, 320)
# An L-shaped template: no quadrilateral, and asymmetric enough that its
# silhouette pins the pose on its own (unlike a rectangle).
_L_POLY = np.array(
    [[80, 60], [220, 60], [220, 110], [140, 110], [140, 190], [80, 190]], dtype=np.int32
)
_INIT_CORNERS = np.array([[80.0, 220.0, 220.0, 80.0], [60.0, 60.0, 190.0, 190.0]])


def _l_mask(H: np.ndarray = None) -> np.ndarray:
    """Rasterize the L-shaped template, optionally warped by ``H``."""
    poly = _L_POLY.astype(np.float64)
    if H is not None:
        pts = np.concatenate([poly, np.ones((len(poly), 1))], axis=1) @ H.T
        poly = pts[:, :2] / pts[:, 2:3]
    mask = np.zeros(_HW, dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def _H(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact 4-point homography (2, 4) -> (2, 4)."""
    return cv2.getPerspectiveTransform(
        src.T.astype(np.float32), dst.T.astype(np.float32)
    ).astype(np.float64)


def _template(cfg: RegisterConfig = RegisterConfig()) -> RegistrationTemplate:
    return RegistrationTemplate.build(_INIT_CORNERS, _l_mask(), _HW, cfg)


def test_second_moments_reject_a_degenerate_mask():
    """A mask with no area has no usable moments."""
    assert second_moments(np.zeros(_HW, dtype=bool)) is None


def test_moment_affine_inits_recover_a_known_affine():
    """One swept rotation reproduces the affine that actually warped the mask."""
    gt_corners = _INIT_CORNERS * 0.7 + np.array([[40.0], [30.0]])
    H_gt = _H(_INIT_CORNERS, gt_corners)
    mask = _l_mask(H_gt) > 0

    inits = moment_affine_inits(
        second_moments(_l_mask() > 0), second_moments(mask), _INIT_CORNERS, n_angles=24
    )
    assert len(inits) == 24
    errs = [alignment_error(gt_corners, c) for c in inits]
    assert min(errs) < 15.0, f"best moment proposal {min(errs):.1f} px off"


def test_register_recovers_from_a_cold_start():
    """With no usable initialization the search still finds the object."""
    gt_corners = _INIT_CORNERS * 0.75 + np.array([[60.0], [25.0]])
    mask = _l_mask(_H(_INIT_CORNERS, gt_corners))

    result = register_mask(_template(), mask, last_H=np.eye(3))

    assert result.H is not None
    err = alignment_error(gt_corners, warp_corners(result.H, _INIT_CORNERS))
    assert err < 15.0, f"cold-start corner error {err:.1f} px"


def test_register_keeps_a_good_incumbent_without_searching():
    """A healthy last-frame pose is accepted outright: one solve, no search."""
    gt_corners = _INIT_CORNERS + np.array([[6.0], [-4.0]])
    mask = _l_mask(_H(_INIT_CORNERS, gt_corners))
    H_init = _H(_INIT_CORNERS, gt_corners + 3.0)

    result = register_mask(_template(), mask, last_H=H_init)

    assert result.source == "last"
    assert result.n_solves == 1, f"fast path paid {result.n_solves} solves"


def _occluded(gt_corners: np.ndarray) -> np.ndarray:
    """The warped template with its right-hand third cut away."""
    mask = _l_mask(_H(_INIT_CORNERS, gt_corners))
    xs = np.nonzero(mask)[1]
    mask[:, int(np.percentile(xs, 66)):] = 0
    return mask


def _textured_frames(H_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An init frame of smooth random texture and its warp by ``H_gt``."""
    rng = np.random.default_rng(0)
    init = cv2.GaussianBlur(
        rng.integers(0, 255, _HW, dtype=np.uint8).astype(np.uint8), (0, 0), 3.0
    )
    init = cv2.normalize(init, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return init, cv2.warpPerspective(init, H_gt, (_HW[1], _HW[0]))


def test_trimming_protects_a_healthy_pose_under_occlusion():
    """``trim_fraction`` is what stops an occluder's edge capturing the fit.

    A hidden third of the outline leaves residuals the Huber loss only
    *down-weights*, so the solve still gets dragged onto the cut edge. Dropping
    the worst tail outright removes that pull. The knob is off by default —
    measured on PTi/POT dev it costs more elsewhere than it buys here (see
    :class:`RegisterConfig`) — so this test pins the behaviour it provides, not
    the default configuration.
    """
    gt_corners = _INIT_CORNERS + np.array([[5.0], [-3.0]])
    H_gt = _H(_INIT_CORNERS, gt_corners)
    init_frame, frame = _textured_frames(H_gt)
    mask = _occluded(gt_corners)

    def err_with(trim: float) -> float:
        cfg = RegisterConfig(trim_fraction=trim)
        template = RegistrationTemplate.build(
            _INIT_CORNERS, _l_mask(), _HW, cfg, init_frame=init_frame
        )
        result = register_mask(template, mask, last_H=H_gt, cfg=cfg, frame=frame)
        return alignment_error(gt_corners, warp_corners(result.H, _INIT_CORNERS))

    trimmed = err_with(0.25)
    assert trimmed < 25.0, f"trimmed fit still {trimmed:.1f} px off"
    assert trimmed < err_with(0.0)


def test_appearance_breaks_a_silhouette_alias():
    """Texture recovers a pose the silhouette cannot distinguish.

    A rectangle's outline is invariant under its own half turn, so a mask-only
    verifier scores the 180-degree alias exactly as well as the truth — the
    single largest error mode measured on real data (45% of wrong hypotheses).
    """
    corners = np.array([[90.0, 230.0, 230.0, 90.0], [70.0, 70.0, 170.0, 170.0]])
    template_mask = corners_to_mask(corners, _HW)
    gt_corners = corners + np.array([[8.0], [5.0]])
    H_gt = _H(corners, gt_corners)
    init_frame, frame = _textured_frames(H_gt)
    mask = corners_to_mask(gt_corners, _HW)

    cfg = RegisterConfig(accept_score=1.01)  # force the search on every frame
    template = RegistrationTemplate.build(corners, template_mask, _HW, cfg, init_frame)
    result = register_mask(
        template, mask, last_H=np.eye(3), cfg=cfg, frame=frame, keep_hypotheses=True
    )

    scored = [h for h in result.hypotheses if h.source != "propose" and "ncc" in h.features]
    right = [h for h in scored if alignment_error(gt_corners, h.control) < 15.0]
    aliases = [
        h for h in scored
        if alignment_error(gt_corners, h.control) > 100.0 and h.features["iou"] > 0.9
    ]
    assert right and aliases, "test needs both a correct pose and a high-IoU alias"

    # The silhouette cannot tell them apart; the texture can, and does.
    assert min(h.features["iou"] for h in right) - max(
        h.features["iou"] for h in aliases
    ) < 0.05
    assert min(h.features["ncc"] for h in right) > max(h.features["ncc"] for h in aliases) + 0.5
    assert min(h.score for h in right) > max(h.score for h in aliases)
    assert alignment_error(gt_corners, warp_corners(result.H, corners)) < 15.0


def test_occlusion_lowers_the_reported_score():
    """A partly hidden object yields a *less confident* answer, not a false one.

    Measured on real data (2026-08-11): when the SAM3 mask covers under 85% of
    the object, no mask-based estimator recovers the pose — the quad pipeline
    included, and the search's own oracle sits at 3%. What registration can
    still do there is report that it does not know, so the fusion layer can
    down-weight it instead of trusting a confident wrong pose.
    """
    gt_corners = _INIT_CORNERS * 0.8 + np.array([[30.0], [20.0]])
    H_gt = _H(_INIT_CORNERS, gt_corners)
    template = _template()

    clean = register_mask(template, _l_mask(H_gt), last_H=H_gt)
    occluded = register_mask(template, _occluded(gt_corners), last_H=H_gt)

    assert occluded.score < clean.score, (
        f"occluded score {occluded.score:.3f} not below clean {clean.score:.3f}"
    )


def test_register_rejects_an_empty_mask():
    """Nothing to register against is a failure, not a guess."""
    result = register_mask(_template(), np.zeros(_HW, dtype=np.uint8), last_H=np.eye(3))
    assert result.H is None


def test_verification_score_prefers_matching_texture():
    """With equal silhouette overlap, the better-correlating pose wins."""
    cfg = RegisterConfig()
    good = {"iou": 0.9, "ncc": 0.9}
    alias = {"iou": 0.9, "ncc": -0.2}
    assert verification_score(good, cfg) > verification_score(alias, cfg)


def test_verification_score_ignores_a_missing_correlation():
    """No texture measurement is no evidence, not evidence against."""
    cfg = RegisterConfig()
    assert verification_score({"iou": 0.8}, cfg) == 0.8**cfg.overlap_power
