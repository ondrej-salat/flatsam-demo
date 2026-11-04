"""NN trust-cascade tracking algorithm (no LoMa).

For ``*_nn_cascade*`` configs with the NN homography scorer enabled and a
``trust_threshold`` set: candidates are evaluated in preference order
(flow_H -> redetection_flow_H -> mask_H); the first whose predicted RMSE clears
``trust_threshold`` is used immediately, without scoring later options. If none
clear the trust bar, the best remaining candidate is used unless it exceeds
``rmse_threshold``, in which case the last good homography is kept.

Byte-identical inlining of the old ``_track_impl`` flow pre-steps plus
``_track_with_nn_cascade``; the verbs (``_compute_mask_h``, ``_estimate_flow_H``,
``_score_homography``, ``_select_best_candidate``, ``_finalize_*``) stay methods
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
    """Run NN trust-cascade candidate selection for one frame.

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
    trust_thr = trk.cfg.nn_scorer.trust_threshold
    rmse_thr = trk.cfg.nn_scorer.rmse_threshold
    if trust_thr is None:
        raise ValueError("trust_threshold must be set in cascade mode.")

    flow_rmse, flow_sigma = trk._score_homography(seg_mask, flow_H, image_shape)

    redetection_flow_H: Optional[np.ndarray] = None
    redetection_flow_rmse = float("inf")
    redetection_flow_sigma = float("inf")
    mask_rmse = float("inf")
    mask_sigma = float("inf")

    debug = TrackerDebugInfo(
        flow_H=CandidateScore(flow_H, flow_rmse, flow_sigma),
        mask_H=CandidateScore(mask_H, float("inf"), float("inf")),
        redetection_flow_H=CandidateScore(None, float("inf"), float("inf")),
        selected=_CANDIDATE_LAST_H,
    )

    if flow_H is not None and flow_rmse <= trust_thr:
        logger.debug("NN scorer trusted flow_H (rmse=%.2f)", flow_rmse)
        debug.selected = _CANDIDATE_FLOW
        return trk._finalize_success(flow_H, frame, seg_mask, n_mask_corr, debug=debug)

    if hough_enabled and n_mask_corr > 0 and mask_H is not None:
        redetection_flow_H = trk._estimate_flow_H(frame, seg_mask, mask_H)
        redetection_flow_rmse, redetection_flow_sigma = trk._score_homography(
            seg_mask,
            redetection_flow_H,
            image_shape,
        )
        debug.redetection_flow_H = CandidateScore(
            redetection_flow_H,
            redetection_flow_rmse,
            redetection_flow_sigma,
        )
        if redetection_flow_H is not None and redetection_flow_rmse <= trust_thr:
            logger.debug(
                "NN scorer trusted redetection_flow_H (rmse=%.2f)",
                redetection_flow_rmse,
            )
            debug.selected = _CANDIDATE_REDETECTION_FLOW
            return trk._finalize_success(
                redetection_flow_H,
                frame,
                seg_mask,
                n_mask_corr,
                debug=debug,
            )

    if hough_enabled and mask_H is not None and trk.cfg.fallback_to_mask_H:
        mask_rmse, mask_sigma = trk._score_homography(seg_mask, mask_H, image_shape)
        debug.mask_H = CandidateScore(mask_H, mask_rmse, mask_sigma)
        if mask_rmse <= trust_thr:
            logger.debug("NN scorer trusted mask_H (rmse=%.2f)", mask_rmse)
            debug.selected = _CANDIDATE_MASK_H
            return trk._finalize_success(
                mask_H, frame, seg_mask, n_mask_corr, debug=debug
            )

    candidates: list[ScoredCandidate] = []
    if flow_H is not None:
        candidates.append(ScoredCandidate(_CANDIDATE_FLOW, flow_H, flow_rmse))
    if redetection_flow_H is not None:
        candidates.append(
            ScoredCandidate(
                _CANDIDATE_REDETECTION_FLOW,
                redetection_flow_H,
                redetection_flow_rmse,
            )
        )
    if mask_H is not None and trk.cfg.fallback_to_mask_H:
        candidates.append(ScoredCandidate(_CANDIDATE_MASK_H, mask_H, mask_rmse))

    best_candidate = trk._select_best_candidate(candidates)
    if best_candidate is None:
        return trk._finalize_failure(
            frame,
            seg_mask,
            hough_enabled,
            "All homography estimation paths failed; keeping last homography.",
            debug=debug,
        )

    if best_candidate.rmse > rmse_thr:
        return trk._finalize_failure(
            frame,
            seg_mask,
            hough_enabled,
            (
                "NN scorer rejected all candidates (best "
                f"{best_candidate.rmse:.2f} > rmse_threshold {rmse_thr:.2f}); "
                "keeping last homography."
            ),
            debug=debug,
        )

    logger.debug(
        "NN scorer used best remaining %r (rmse=%.2f)",
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
