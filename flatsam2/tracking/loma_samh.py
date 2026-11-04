"""LoMa + SAM-H tracking algorithm (online keypoint homography path).

For ``loma_samh*`` configs. Two sub-variants, selected by
``cfg.loma.decouple_hough_state`` (an inline branch, not a separate algorithm):

- **Decoupled** (``decouple_hough_state=True``): an independent SAM-H sub-tracker
  owns the full mask-H + symmetry chain on its OWN state (a separate
  ``last_homography``/``symmetry``); the LoMa estimate is computed independently
  and the per-frame inlier rule picks LoMa vs the sub-tracker's estimate. Nothing
  feeds back — the SAM-H chain advances every frame regardless of the pick.
- **Coupled** (``decouple_hough_state=False``): mask-H is computed inline on the
  shared state and the inlier rule picks LoMa vs mask-H; symmetry is gated so it
  runs only when mask-H was selected (a keypoint winner is already oriented, N4).

Optional SAM3 self-correction (``cfg.correction.enabled``) fires inline in both
sub-variants. Byte-identical inlining of the old ``_track_impl`` LoMa branch plus
``_track_loma_decoupled``; the verbs (``_compute_mask_h``, ``_estimate_loma_H``,
``_choose_loma_or_mask``, ``_decoupled_select``, ``_maybe_correct_sam3``,
``_finalize_*``) stay methods on the tracker until the later refactor steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from flatsam2.tracking import flow_hough
from flatsam2.tracking.common import (
    _CANDIDATE_MASK_H,
    CandidateScore,
    CorrectionTelemetry,
    TrackerDebugInfo,
    TrackResult,
)

if TYPE_CHECKING:
    from flatsam2.tracker import FlatSAMTracker


def track(
    trk: "FlatSAMTracker",
    frame: np.ndarray,
    seg_mask: np.ndarray,
) -> TrackResult:
    """Run the LoMa + SAM-H step for one frame.

    Args:
        trk: Owning tracker instance (provides verbs and per-frame state).
        frame: (H, W, C) opencv uint8 BGR image.
        seg_mask: segmentation mask (H, W) already obtained for this frame.

    Returns:
        TrackResult with the selected homography and segmentation mask.
    """
    hough_enabled = trk.cfg.hough.enabled and trk.state.init_corners is not None

    loma_active = (
        trk.cfg.loma.enabled
        and trk._loma is not None
        and trk._loma_feats0 is not None
    )

    # Fresh per-frame correction flag + injected mask + observables; overwritten
    # when the correction machinery runs.
    trk._loma_last_corrected = False
    trk._loma_last_corr_mask = None
    trk._corr_telemetry = CorrectionTelemetry()

    # --- Decoupled SAM-H: the sub-tracker owns the entire mask-H + symmetry
    # chain (its own state); we never compute mask-H here, only pick LoMa vs its
    # estimate. The SAM-H chain advances every frame regardless of the pick.
    if loma_active and trk._samh_tracker is not None:
        # When the sub-tracker resolves its cyclic corner ambiguity from the
        # keypoint correspondences, those have to exist before it runs, so LoMa
        # moves ahead of it. Same reordering the keypoint proposal makes in
        # corner_fusion, and like it, only when the option is on.
        name_from_matches = (
            trk._samh_tracker.cfg.symmetry.enabled
            and trk._samh_tracker.cfg.symmetry.method == "matches"
        )
        loma_H, n_inliers = None, 0
        if name_from_matches:
            loma_H, n_inliers = trk._estimate_loma_H(frame, seg_mask)
            trk._samh_tracker._loma_last_matches = trk._loma_last_matches

        samh_result = trk._samh_tracker.step_with_mask(frame, seg_mask)
        samh_H = samh_result.homography

        if not name_from_matches:
            loma_H, n_inliers = trk._estimate_loma_H(frame, seg_mask)
        trk._loma_last_inliers = n_inliers

        # Optional SAM3 self-correction (no-op unless correction is enabled).
        trk._loma_last_corrected = trk._maybe_correct_sam3(seg_mask, loma_H, n_inliers)

        selected_H = trk._decoupled_select(loma_H, n_inliers, samh_H)
        trk.state.last_homography = selected_H
        debug = TrackerDebugInfo(
            loma_H=CandidateScore(H=loma_H),
            samh_H=CandidateScore(H=samh_H),
            selected=trk._loma_last_selected,
            corrected=trk._loma_last_corrected,
            n_inliers=n_inliers,
            correction=trk._corr_telemetry,
            # mask-H is computed inside the sub-tracker, so its estimator
            # decision (align vs Hough quad) lives on that instance.
            mask_h=trk._samh_tracker._maskh_telemetry,
        )
        return TrackResult(
            homography=selected_H,
            mask=seg_mask,
            debug=debug,
            correction_mask=trk._loma_last_corr_mask,
        )

    # --- Mask-H computation (cheap, always ready as a fallback) -------------
    mask_H, n_mask_corr = trk._compute_mask_h(seg_mask, hough_enabled, frame)

    # --- Coupled LoMa: pick LoMa vs mask-H on the shared state -------------
    if loma_active:
        loma_H, n_inliers = trk._estimate_loma_H(frame, seg_mask)
        trk._loma_last_inliers = n_inliers
        selected_H = trk._choose_loma_or_mask(loma_H, n_inliers, mask_H, n_mask_corr)
        trk._loma_last_corrected = trk._maybe_correct_sam3(seg_mask, loma_H, n_inliers)
        debug = TrackerDebugInfo(
            loma_H=CandidateScore(H=loma_H),
            mask_H=CandidateScore(H=mask_H),
            selected=trk._loma_last_selected,
            corrected=trk._loma_last_corrected,
            n_inliers=n_inliers,
            correction=trk._corr_telemetry,
            mask_h=trk._maskh_telemetry,
        )
        if selected_H is None:
            return trk._finalize_failure(
                frame,
                seg_mask,
                hough_enabled,
                "LoMa and mask-H both unavailable; keeping last homography.",
                run_symmetry=False,
                debug=debug,
            )
        # Symmetry resolution applies only to the mask/Hough branch; a
        # keypoint-based estimate is already correctly oriented.
        run_symmetry = trk._loma_last_selected == _CANDIDATE_MASK_H
        return trk._finalize_success(
            selected_H,
            frame,
            seg_mask,
            n_mask_corr,
            run_symmetry=run_symmetry,
            debug=debug,
        )

    # Unreachable for a properly-initialised loma config (loma_active is always
    # True once init() has run). Preserve the pre-refactor fallback — the plain
    # flow + Hough chain — rather than returning nothing.
    return flow_hough.track(trk, frame, seg_mask)
