"""Flow + Hough tracking algorithm (no NN scorer, no LoMa).

Sequential fallback chain for the ``flow_hough`` configs (``fast_kitti``,
``flow_hough*``, ``hough_symmetry_only*``):

1. ``flow_H`` estimated with the last-good-H prewarp (primary path).
2. ``redetection_flow_H`` estimated with the ``mask_H`` prewarp (re-detection
   assisted), when Hough is enabled and produced corner correspondences.
3. ``mask_H`` directly (pure re-detection), when ``fallback_to_mask_H``.

On exhaustion the last good homography is kept. This is the byte-identical
inlining of the old ``_track_impl`` flow path plus ``_track_without_nn``; the
verbs (``_compute_mask_h``, ``_estimate_flow_H``, ``_finalize_*``) stay methods
on the tracker until the later refactor steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from flatsam2.tracking.common import (
    _CANDIDATE_FLOW,
    _CANDIDATE_LAST_H,
    _CANDIDATE_MASK_H,
    _CANDIDATE_REDETECTION_FLOW,
    CandidateScore,
    TrackerDebugInfo,
    TrackResult,
)

if TYPE_CHECKING:
    from flatsam2.tracker import FlatSAMTracker

logger = logging.getLogger(__name__)


def track(
    trk: "FlatSAMTracker",
    frame: np.ndarray,
    seg_mask: np.ndarray,
) -> TrackResult:
    """Run the flow + Hough sequential fallback chain for one frame.

    Args:
        trk: Owning tracker instance (provides verbs and per-frame state).
        frame: (H, W, C) opencv uint8 BGR image.
        seg_mask: segmentation mask (H, W) already obtained for this frame.

    Returns:
        TrackResult with the estimated homography and segmentation mask.
    """
    hough_enabled = trk.cfg.hough.enabled and trk.state.init_corners is not None

    # Mask-H computation (cheap, always ready as a fallback).
    mask_H, n_mask_corr = trk._compute_mask_h(seg_mask, hough_enabled, frame)

    # Step 1: flow with the last-good-H prewarp (primary path).
    last_H = trk.state.last_homography
    flow_H = trk._estimate_flow_H(frame, seg_mask, last_H)

    debug = TrackerDebugInfo(
        flow_H=CandidateScore(H=flow_H),
        mask_H=CandidateScore(H=mask_H),
        mask_h=trk._maskh_telemetry,
    )
    if flow_H is not None:
        debug.selected = _CANDIDATE_FLOW
        return trk._finalize_success(flow_H, frame, seg_mask, n_mask_corr, debug=debug)

    if hough_enabled and mask_H is not None and n_mask_corr > 0:
        redetection_flow_H: Optional[np.ndarray] = trk._estimate_flow_H(
            frame, seg_mask, mask_H
        )
        debug.redetection_flow_H = CandidateScore(H=redetection_flow_H)
        if redetection_flow_H is not None:
            logger.debug(
                "Flow failed with last-H prewarp, succeeded with mask-H prewarp."
            )
            debug.selected = _CANDIDATE_REDETECTION_FLOW
            return trk._finalize_success(
                redetection_flow_H,
                frame,
                seg_mask,
                n_mask_corr,
                debug=debug,
            )

        if trk.cfg.fallback_to_mask_H:
            logger.debug("Flow failed with both prewarps; using mask-H directly.")
            debug.selected = _CANDIDATE_MASK_H
            return trk._finalize_success(
                mask_H, frame, seg_mask, n_mask_corr, debug=debug
            )

    debug.selected = _CANDIDATE_LAST_H
    return trk._finalize_failure(
        frame,
        seg_mask,
        hough_enabled,
        "All homography estimation paths failed; keeping last homography.",
        debug=debug,
    )
