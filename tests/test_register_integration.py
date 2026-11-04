"""The tracker's ``hough.method='register'`` wiring, without SAM3 or a GPU.

Exercises :meth:`FlatSAMTracker._compute_mask_h` on a constructor-free tracker
(the pattern in ``test_loma_samh_logic``) so the plumbing — config knobs to
``RegisterConfig``, the cached template context, acceptance gating and
telemetry — is checked on CPU rather than by burning a tracking grid.
"""

import cv2
import numpy as np

from flatsam2.config import HoughConfig, TrackerConfig
from flatsam2.datasets import corners_to_mask, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.tracker import FlatSAMTracker, TrackerState


_HW = (240, 320)
_CORNERS = np.array([[80.0, 220.0, 220.0, 80.0], [60.0, 60.0, 190.0, 190.0]])


def _H(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        src.T.astype(np.float32), dst.T.astype(np.float32)
    ).astype(np.float64)


def _frames(H_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A textured init frame and its warp, as BGR images."""
    rng = np.random.default_rng(1)
    gray = cv2.GaussianBlur(rng.integers(0, 255, _HW, dtype=np.uint8), (0, 0), 3.0)
    init = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return init, cv2.warpPerspective(init, H_gt, (_HW[1], _HW[0]))


def _tracker(last_H: np.ndarray, init_frame: np.ndarray, **hough) -> FlatSAMTracker:
    """A tracker with only what the mask-H verb touches."""
    tracker = object.__new__(FlatSAMTracker)
    tracker.cfg = TrackerConfig(
        hough=HoughConfig(enabled=True, method="register", align_fallback_hough=False, **hough)
    )
    tracker.state = TrackerState()
    tracker.state.init_corners = _CORNERS
    tracker.state.init_mask = corners_to_mask(_CORNERS, _HW)
    tracker.state.init_frame = init_frame
    tracker.state.last_homography = last_H
    tracker._register_template = None
    tracker._register_extra_H = ()
    # Correspondences the registrar's support term would score; None means
    # "no keypoint evidence this frame", which is the shape-only path.
    tracker._register_matches = None
    return tracker


def test_register_branch_returns_a_homography_and_telemetry():
    """The wired-up branch estimates the pose and records its own decision."""
    gt_corners = _CORNERS + np.array([[7.0], [-5.0]])
    H_gt = _H(_CORNERS, gt_corners)
    init_frame, frame = _frames(H_gt)
    tracker = _tracker(H_gt, init_frame)

    mask_H, n_ok = tracker._compute_mask_h(corners_to_mask(gt_corners, _HW), True, frame)

    assert mask_H is not None
    assert alignment_error(gt_corners, warp_corners(mask_H, _CORNERS)) < 15.0
    assert n_ok > 0
    tel = tracker._maskh_telemetry
    assert tel.source == "register"
    assert tel.align_accepted and tel.align_attempted
    assert 0.0 <= tel.register_score <= 1.0
    assert tel.register_source in {"last", "moment"}
    assert tracker._maskh_last_corner_cov is not None


def test_register_branch_rejects_below_min_score():
    """A gate above any achievable score suppresses the branch, not crashes it."""
    gt_corners = _CORNERS + np.array([[7.0], [-5.0]])
    H_gt = _H(_CORNERS, gt_corners)
    init_frame, frame = _frames(H_gt)
    tracker = _tracker(H_gt, init_frame, register_min_score=1.01)

    mask_H, n_ok = tracker._compute_mask_h(corners_to_mask(gt_corners, _HW), True, frame)

    assert mask_H is None and n_ok == 0
    assert tracker._maskh_telemetry.align_attempted
    assert not tracker._maskh_telemetry.align_accepted
    assert tracker._maskh_last_corner_cov is None


def test_template_context_is_built_once_per_sequence():
    """The per-sequence context is cached, not rebuilt on every frame."""
    gt_corners = _CORNERS + np.array([[7.0], [-5.0]])
    H_gt = _H(_CORNERS, gt_corners)
    init_frame, frame = _frames(H_gt)
    tracker = _tracker(H_gt, init_frame)
    mask = corners_to_mask(gt_corners, _HW)

    tracker._compute_mask_h(mask, True, frame)
    first = tracker._register_template
    tracker._compute_mask_h(mask, True, frame)

    assert first is not None
    assert tracker._register_template is first


def test_register_recovers_when_the_last_homography_is_lost():
    """The branch's reason for existing: a lost incumbent is not fatal."""
    gt_corners = _CORNERS * 0.75 + np.array([[55.0], [30.0]])
    H_gt = _H(_CORNERS, gt_corners)
    init_frame, frame = _frames(H_gt)
    # The incumbent sits small and far away in a corner: no local refinement
    # reaches the object from there, so only the stateless search can.
    lost = _H(_CORNERS, _CORNERS * 0.3 + np.array([[5.0], [5.0]]))
    tracker = _tracker(lost, init_frame)

    mask_H, _ = tracker._compute_mask_h(corners_to_mask(gt_corners, _HW), True, frame)

    assert mask_H is not None
    err = alignment_error(gt_corners, warp_corners(mask_H, _CORNERS))
    assert err < 15.0, f"failed to recover from a lost pose ({err:.1f} px)"
    assert tracker._maskh_telemetry.register_source == "moment"


def test_keypoint_proposal_is_used_then_cleared():
    """An offered keypoint pose competes as a hypothesis and does not persist.

    The mask cannot say which corner is which; the keypoint branch can, so its
    homography is offered as an extra proposal. It must be *verified* like any
    other hypothesis rather than trusted — and it must not leak into the next
    frame, where it would be a stale pose.
    """
    gt_corners = _CORNERS * 0.75 + np.array([[55.0], [30.0]])
    H_gt = _H(_CORNERS, gt_corners)
    init_frame, frame = _frames(H_gt)
    lost = _H(_CORNERS, _CORNERS * 0.3 + np.array([[5.0], [5.0]]))
    tracker = _tracker(lost, init_frame)
    tracker._register_extra_H = (H_gt,)  # what the keypoint branch would supply

    mask_H, _ = tracker._compute_mask_h(corners_to_mask(gt_corners, _HW), True, frame)

    assert mask_H is not None
    assert alignment_error(gt_corners, warp_corners(mask_H, _CORNERS)) < 15.0
    assert tracker._maskh_telemetry.register_source == "extra0"
    assert tracker._register_extra_H == (), "proposal must not survive the frame"


def test_a_wrong_keypoint_proposal_is_outscored():
    """A bad keypoint pose is rejected on the evidence, not followed."""
    gt_corners = _CORNERS + np.array([[6.0], [-4.0]])
    H_gt = _H(_CORNERS, gt_corners)
    init_frame, frame = _frames(H_gt)
    tracker = _tracker(H_gt, init_frame)
    # A keypoint estimate that has drifted far off the object.
    tracker._register_extra_H = (_H(_CORNERS, _CORNERS * 0.5 + np.array([[150.0], [90.0]])),)

    mask_H, _ = tracker._compute_mask_h(corners_to_mask(gt_corners, _HW), True, frame)

    assert mask_H is not None
    err = alignment_error(gt_corners, warp_corners(mask_H, _CORNERS))
    assert err < 15.0, f"a wrong keypoint proposal pulled the estimate {err:.1f} px off"
