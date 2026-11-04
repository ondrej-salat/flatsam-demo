"""NN min-RMSE tracking algorithm (eager scoring mode, no LoMa).

For ``*_nn`` configs with the NN homography scorer enabled and no
``trust_threshold``: every applicable candidate (flow_H, mask_H, and an optional
mask-H-prewarped redetection flow) is scored eagerly, and the candidate with the
lowest predicted RMSE is selected — unless it exceeds ``rmse_threshold``, in
which case the last good homography is kept.

Byte-identical inlining of the old ``_track_impl`` flow pre-steps plus
``_track_with_nn_min_rmse``; the verbs (``_compute_mask_h``,
``_estimate_flow_H``, ``_score_homography``, ``_select_best_candidate``,
``_finalize_*``) stay methods on the tracker until the later refactor steps.
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
    ScoredCandidate,
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
    """Run NN min-RMSE candidate selection for one frame.

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

    image_shape = frame.shape[:2]
    flow_rmse, flow_sigma = trk._score_homography(seg_mask, flow_H, image_shape)
    mask_rmse, mask_sigma = trk._score_homography(seg_mask, mask_H, image_shape)

    redetection_flow_H: Optional[np.ndarray] = None
    redetection_flow_rmse = float("inf")
    redetection_flow_sigma = float("inf")

    if (
        hough_enabled
        and n_mask_corr > 0
        and mask_H is not None
        and mask_rmse < flow_rmse
    ):
        redetection_flow_H = trk._estimate_flow_H(frame, seg_mask, mask_H)
        redetection_flow_rmse, redetection_flow_sigma = trk._score_homography(
            seg_mask,
            redetection_flow_H,
            image_shape,
        )

    debug = TrackerDebugInfo(
        flow_H=CandidateScore(flow_H, flow_rmse, flow_sigma),
        mask_H=CandidateScore(mask_H, mask_rmse, mask_sigma),
        redetection_flow_H=CandidateScore(
            redetection_flow_H,
            redetection_flow_rmse,
            redetection_flow_sigma,
        ),
        selected=_CANDIDATE_LAST_H,
    )

    candidates: list[ScoredCandidate] = []
    if flow_H is not None:
        candidates.append(ScoredCandidate(_CANDIDATE_FLOW, flow_H, flow_rmse))
    if mask_H is not None and trk.cfg.fallback_to_mask_H:
        candidates.append(ScoredCandidate(_CANDIDATE_MASK_H, mask_H, mask_rmse))
    if redetection_flow_H is not None:
        candidates.append(
            ScoredCandidate(
                _CANDIDATE_REDETECTION_FLOW,
                redetection_flow_H,
                redetection_flow_rmse,
            )
        )

    best_candidate = trk._select_best_candidate(candidates)
    if best_candidate is None:
        return trk._finalize_failure(
            frame,
            seg_mask,
            hough_enabled,
            "All homography estimation paths failed; keeping last homography.",
            debug=debug,
        )

    if best_candidate.rmse > trk.cfg.nn_scorer.rmse_threshold:
        return trk._finalize_failure(
            frame,
            seg_mask,
            hough_enabled,
            "NN scorer rejected all candidates (above threshold); keeping last homography.",
            debug=debug,
        )

    logger.debug(
        "NN scorer selected %r (predicted_rmse=%.2f)",
        best_candidate.label,
        best_candidate.rmse,
    )
    debug.selected = best_candidate.label
    return trk._finalize_success(
        best_candidate.homography,
        frame,
        seg_mask,
        n_mask_corr,
        debug=debug,
    )
