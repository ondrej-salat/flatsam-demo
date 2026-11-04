"""Corner-level covariance fusion of LoMa + SAM-H (``algorithm='corner_fusion'``).

Replaces the hard 35-inlier switch (``loma_samh``'s ``_decoupled_select``) with
a continuous, per-corner blend. Both branches already report, per corner, a
point estimate and a 2x2 pixel covariance from their own fit's normal
equations — ``homography_corner_covs`` for LoMa (reprojection residuals of the
RANSAC inliers), ``mask_align._corner_cov_from_boundary_fit`` for SAM-H
(boundary signed-distance residuals of the converged GN alignment). Same
formula, different residual, same currency: how much would this corner move
under resampling of the evidence this branch used.

Fusion, per corner, in one sentence: if the two branches agree within their
combined stated uncertainty, average their positions weighted by confidence
(precision-weighted average); if they disagree by more than that, don't
average — averaging a good point with a wrong one still gives a wrong point —
trust whichever branch is more confident about *that* corner instead. Missing
evidence on one side (branch failed, or its covariance could not be measured)
is zero precision, so the formula degrades to "use whichever side has data"
without a special case.

Shares the independent SAM-H sub-tracker with ``loma_samh``'s decoupled path
(``cfg.loma.decouple_hough_state=True``); nothing here is specific to LoMa or
to the mask aligner — any keypoint matcher or segmentation method that can
report a per-corner covariance in this shape plugs in the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from flatsam2 import match_naming
from flatsam2.corner_covariance import mahalanobis_sq
from flatsam2.datasets import warp_corners
from flatsam2.occlusion import is_occluded_view, warp_template_silhouette
from flatsam2.tracking.common import (
    _CANDIDATE_FUSED,
    _CANDIDATE_LAST_H,
    _is_valid_homography,
    CandidateScore,
    CorrectionTelemetry,
    FusionTelemetry,
    TrackerDebugInfo,
    TrackResult,
)

if TYPE_CHECKING:
    from flatsam2.tracker import FlatSAMTracker


_REG_EPS = 1e-6  # covariance regularization before inversion (px^2)


def _safe_precision(cov: Optional[np.ndarray]) -> np.ndarray:
    """Inverse of a 2x2 covariance, or zero precision when unusable.

    Zero precision is the "I have no opinion" element for the weighted
    average below: it drops the term without a branch in the caller.
    """
    if cov is None:
        return np.zeros((2, 2))
    try:
        return np.linalg.inv(cov + np.eye(2) * _REG_EPS)
    except np.linalg.LinAlgError:
        return np.zeros((2, 2))


def _precision_weighted(a_pt: np.ndarray, a_cov: np.ndarray, b_pt: np.ndarray,
                        b_cov: np.ndarray) -> np.ndarray:
    """Best linear unbiased combination of two estimates of the same point.

    Args:
        a_pt: (2,) first estimate; a_cov: (2, 2) its covariance.
        b_pt: (2,) second estimate; b_cov: (2, 2) its covariance.

    Returns:
        (2,) combined position ``(A^-1 + B^-1)^-1 (A^-1 a + B^-1 b)``.
    """
    prec_a, prec_b = _safe_precision(a_cov), _safe_precision(b_cov)
    return np.linalg.solve(prec_a + prec_b, prec_a @ a_pt + prec_b @ b_pt)


def _relabel_to_reference(
    quad: np.ndarray,
    covs: Optional[np.ndarray],
    reference: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Re-name ``quad``'s corners from ``reference``'s naming, if clearly better.

    A silhouette fixes a planar pose only up to which corner is called first:
    the four cyclic relabelings of a quad describe the same outline. Texture
    breaks that tie, so the keypoint branch's naming is borrowed here without
    borrowing its position — the mask branch keeps its own corner *locations*
    and only permutes their order.

    The branch's own naming is the incumbent, as it is inside the registrar's
    alias checks: it carries the temporal evidence of every frame that led to
    it, so a single-frame nearest-neighbour vote may not overturn it on a whim.
    Only a rotation at least ``margin`` times closer than the incumbent is
    taken. Near-square outlines and badly localized reference quads make all
    four rotations similar, which is exactly the case this test rejects.

    Args:
        quad: (2, 4) corner positions to be re-named.
        covs: (4, 2, 2) per-corner covariances of ``quad``, permuted with it,
            or None.
        reference: (2, 4) corner positions whose naming is trusted.
        margin: Required ratio ``d_incumbent / d_best`` before re-naming;
            ``<= 0`` disables the step.

    Returns:
        Tuple ``(quad, covs)``, re-labelled or untouched.
    """
    if margin <= 0.0:
        return quad, covs
    dists = np.array([
        float(np.linalg.norm(np.roll(quad, r, axis=1) - reference)) for r in range(4)
    ])
    best = int(np.argmin(dists))
    # The comparison is a product, not a ratio, so an exact match (distance 0)
    # needs no special case: it beats any margin unless the incumbent also
    # matches exactly, in which case the two namings coincide anyway.
    if best == 0 or dists[0] < margin * dists[best]:
        return quad, covs
    rolled_covs = np.roll(covs, best, axis=0) if covs is not None else None
    return np.roll(quad, best, axis=1), rolled_covs


def fuse_corners(
    init_corners: np.ndarray,
    loma_H: Optional[np.ndarray],
    loma_cov: Optional[np.ndarray],
    samh_H: Optional[np.ndarray],
    samh_cov: Optional[np.ndarray],
    last_corners: np.ndarray,
    disagreement_gate: float,
    loma_cov_scale: float = 1.0,
    samh_cov_scale: float = 1.0,
    decision_scope: str = "per_corner",
    keypoint_naming_margin: float = 0.0,
) -> tuple[np.ndarray, FusionTelemetry]:
    """Fuse the two branches' corner estimates.

    Args:
        init_corners: (2, 4) template corner coordinates.
        loma_H: LoMa homography this frame, or None on failure.
        loma_cov: (4, 2, 2) per-corner covariance of ``loma_H``, or None.
        samh_H: Independent SAM-H sub-tracker homography, or None.
        samh_cov: (4, 2, 2) per-corner covariance of ``samh_H``, or None.
        last_corners: (2, 4) previous frame's fused corners (coast target when
            neither branch has anything to say about a corner).
        disagreement_gate: Squared-Mahalanobis threshold separating "fuse" from
            "trust the more confident one" (``cfg.fusion.disagreement_gate``).
            Applied per corner under ``per_corner``, to the four-corner mean
            under ``whole_frame``.
        loma_cov_scale: Multiplicative calibration correction for
            ``loma_cov`` (``cfg.fusion.loma_cov_scale``; 1.0 = trust the raw
            fit covariance).
        samh_cov_scale: Same, for ``samh_cov``.
        decision_scope: ``'per_corner'`` decides fuse-or-arbitrate separately
            for each corner; ``'whole_frame'`` makes one decision for the whole
            quad (``cfg.fusion.decision_scope``).
        keypoint_naming_margin: When positive, the SAM-H corners are re-named
            from the LoMa corners before the decision, whenever the nearest
            cyclic rotation is that many times closer than SAM-H's own naming
            (``cfg.fusion.keypoint_naming_margin``, see
            ``_relabel_to_reference``).

    Returns:
        Tuple ``(fused_corners (2, 4), telemetry)``.
    """
    c_loma = warp_corners(loma_H, init_corners) if loma_H is not None else None
    c_samh = warp_corners(samh_H, init_corners) if samh_H is not None else None
    a_covs = loma_cov * loma_cov_scale if (loma_cov is not None and c_loma is not None) else None
    b_covs = samh_cov * samh_cov_scale if (samh_cov is not None and c_samh is not None) else None

    # The mask branch's positions with the keypoint branch's naming: the two
    # things the branches are respectively good at, kept separate.
    if c_loma is not None and c_samh is not None:
        c_samh, b_covs = _relabel_to_reference(
            c_samh, b_covs, c_loma, keypoint_naming_margin
        )

    if decision_scope == "whole_frame":
        return _fuse_whole_frame(
            c_loma, a_covs, c_samh, b_covs, last_corners, disagreement_gate, loma_cov, samh_cov
        )

    fused = np.zeros((2, 4))
    modes: list[str] = []
    disagreements: list[float] = []

    for k in range(4):
        a_pt = c_loma[:, k] if c_loma is not None else None
        b_pt = c_samh[:, k] if c_samh is not None else None
        a_cov = a_covs[k] if a_covs is not None else None
        b_cov = b_covs[k] if b_covs is not None else None

        if a_pt is not None and b_pt is not None and a_cov is not None and b_cov is not None:
            d2 = mahalanobis_sq(a_pt - b_pt, a_cov + b_cov)
            disagreements.append(d2)
            if d2 <= disagreement_gate:
                fused[:, k] = _precision_weighted(a_pt, a_cov, b_pt, b_cov)
                modes.append("fused")
            else:
                fused[:, k] = a_pt if np.trace(a_cov) <= np.trace(b_cov) else b_pt
                modes.append("gated_confident")
        elif a_pt is not None and a_cov is not None:
            fused[:, k] = a_pt
            modes.append("loma_only_cov")
        elif b_pt is not None and b_cov is not None:
            fused[:, k] = b_pt
            modes.append("samh_only_cov")
        elif a_pt is not None:
            fused[:, k] = a_pt
            modes.append("loma_only")
        elif b_pt is not None:
            fused[:, k] = b_pt
            modes.append("samh_only")
        else:
            fused[:, k] = last_corners[:, k]
            modes.append("coast")

    telemetry = FusionTelemetry(
        modes=tuple(modes),
        max_disagreement=float(max(disagreements)) if disagreements else float("nan"),
        n_fused=modes.count("fused"),
        n_gated=modes.count("gated_confident"),
        loma_cov=loma_cov,
        samh_cov=samh_cov,
    )
    return fused, telemetry


def _fuse_whole_frame(
    c_loma: Optional[np.ndarray],
    a_covs: Optional[np.ndarray],
    c_samh: Optional[np.ndarray],
    b_covs: Optional[np.ndarray],
    last_corners: np.ndarray,
    disagreement_gate: float,
    loma_cov: Optional[np.ndarray],
    samh_cov: Optional[np.ndarray],
) -> tuple[np.ndarray, FusionTelemetry]:
    """One fuse-or-select decision for the whole quad.

    The per-corner rule can hand two corners to one branch and two to the other,
    which produces a quad neither branch ever proposed. Measurement says there is
    almost nothing to gain from that freedom: frames where the branches are
    genuinely complementary (neither alone within 15 px on all four corners,
    together yes) are 0.66% of PTi, and a per-corner oracle beats a
    whole-branch oracle by 0.48 pp (``experiments/corner_complementarity.py``).
    What the fusion does buy — averaging down the noise when both branches are
    right — needs no per-corner switching. So the decision is taken once:

    * both branches usable and agreeing: every corner is the precision-weighted
      average of the two (the weights are still each corner's own, only the
      *decision* is shared);
    * disagreeing beyond the gate: the whole pose comes from the branch with
      the smaller total corner variance, so the output is always a pose some
      branch actually proposed;
    * one branch only: that branch; neither: coast on ``last_corners``.

    Agreement is tested on the mean per-corner squared Mahalanobis distance, so
    ``disagreement_gate`` keeps its chi-square(2) units (the sum over four
    corners is chi-square(8) under the same model).

    Args:
        c_loma: (2, 4) LoMa corner positions, or None.
        a_covs: (4, 2, 2) scaled LoMa covariances, or None.
        c_samh: (2, 4) SAM-H corner positions, or None.
        b_covs: (4, 2, 2) scaled SAM-H covariances, or None.
        last_corners: (2, 4) previous frame's output.
        disagreement_gate: Gate on the mean per-corner squared Mahalanobis
            distance.
        loma_cov: Unscaled LoMa covariance, passed through to telemetry.
        samh_cov: Same, SAM-H.

    Returns:
        Tuple ``(fused_corners (2, 4), telemetry)``.
    """
    both_usable = c_loma is not None and c_samh is not None and a_covs is not None \
        and b_covs is not None
    disagreements = (
        [mahalanobis_sq(c_loma[:, k] - c_samh[:, k], a_covs[k] + b_covs[k]) for k in range(4)]
        if both_usable
        else []
    )

    if both_usable and float(np.mean(disagreements)) <= disagreement_gate:
        fused = np.column_stack(
            [
                _precision_weighted(c_loma[:, k], a_covs[k], c_samh[:, k], b_covs[k])
                for k in range(4)
            ]
        )
        mode = "fused"
    elif both_usable:
        loma_wins = np.trace(a_covs, axis1=1, axis2=2).sum() <= np.trace(
            b_covs, axis1=1, axis2=2
        ).sum()
        fused = c_loma.copy() if loma_wins else c_samh.copy()
        mode = "gated_loma" if loma_wins else "gated_samh"
    elif c_loma is not None and a_covs is not None:
        fused, mode = c_loma.copy(), "loma_only_cov"
    elif c_samh is not None and b_covs is not None:
        fused, mode = c_samh.copy(), "samh_only_cov"
    elif c_loma is not None:
        fused, mode = c_loma.copy(), "loma_only"
    elif c_samh is not None:
        fused, mode = c_samh.copy(), "samh_only"
    else:
        fused, mode = last_corners.copy(), "coast"

    telemetry = FusionTelemetry(
        modes=(mode,) * 4,
        max_disagreement=float(max(disagreements)) if disagreements else float("nan"),
        n_fused=4 if mode == "fused" else 0,
        n_gated=4 if mode.startswith("gated") else 0,
        loma_cov=loma_cov,
        samh_cov=samh_cov,
    )
    return fused, telemetry


def _record_naming_telemetry(trk: "FlatSAMTracker", samh_H: Optional[np.ndarray]) -> None:
    """Store the four namings' correspondence support on the mask-H telemetry.

    Diagnostic only — it changes nothing the tracker decides. It exists so that
    a finished run can answer "what would a different tau, or a margin over the
    incumbent, have done?" without spending a grid on each question, the same
    role the stored branch estimates play for fusion rules.

    Args:
        trk: Owning tracker instance.
        samh_H: The mask branch's homography this frame, or None.
    """
    telemetry = trk._samh_tracker._maskh_telemetry
    matches = trk._loma_last_matches
    if samh_H is None or matches is None or trk.state.init_corners is None:
        return
    pts_src, pts_dst = matches
    if pts_src is None or len(pts_src) < match_naming.MIN_MATCHES:
        return
    support, counts, sum_sq = match_naming.alias_supports(
        warp_corners(samh_H, trk.state.init_corners),
        trk.state.init_corners,
        pts_src,
        pts_dst,
        trk.cfg.hough.register_support_tau_px,
    )
    telemetry.naming_support = support
    telemetry.naming_n_matches = int(len(pts_src))
    telemetry.naming_hist_counts = counts
    telemetry.naming_hist_sumsq = sum_sq


def _mask_is_partial_view(
    trk: "FlatSAMTracker",
    seg_mask: np.ndarray,
    loma_H: Optional[np.ndarray],
) -> bool:
    """Whether this frame's SAM mask is a clean partial view of the object.

    The covariance the mask branch reports cannot say its boundary belongs to
    the wrong thing, and under occlusion that is exactly what happens: SAM
    returns the visible part, the registrar fits the template to that
    sub-region, and the fit is *precise*, so the branch wins the arbitration
    with a wrong pose (``flatsam2.occlusion``; the three POT occlusion frames
    of docs/journal.md 2026-08-26 all take the mask branch as the more
    confident one). This asks the question the covariance cannot: is the mask a
    subset of the silhouette we expect here?

    The expectation warps the *stored template silhouette* — the tracker is
    quad-free, so a quad through four corners is a different region — from the
    keypoint branch's pose for **this** frame. Predicting from the previous
    frame instead would turn fast motion into false occlusion; predicting from
    the mask branch's own pose would be circular, since that pose is what is
    being judged. With no keypoint pose the previous frame is all there is.

    Args:
        trk: Owning tracker instance.
        seg_mask: (H, W) segmentation mask for this frame.
        loma_H: (3, 3) keypoint-branch homography this frame, or None.

    Returns:
        True when the mask branch should abstain for this frame.
    """
    if trk.state.init_mask is None:
        return False
    predict_H = loma_H if loma_H is not None else trk.state.last_homography
    if predict_H is None:
        return False
    predicted = warp_template_silhouette(
        trk.state.init_mask, predict_H, seg_mask.shape[:2]
    )
    return is_occluded_view(
        seg_mask,
        predicted,
        min_precision=trk.cfg.fusion.occlusion_min_precision,
        max_recall=trk.cfg.fusion.occlusion_max_recall,
    )


def track(
    trk: "FlatSAMTracker",
    frame: np.ndarray,
    seg_mask: np.ndarray,
) -> TrackResult:
    """Run the corner-fusion step for one frame.

    Mirrors ``loma_samh.track``'s decoupled branch (independent SAM-H
    sub-tracker, LoMa estimated separately) but replaces the whole-homography
    inlier-threshold switch with the per-corner covariance fuse above.

    Args:
        trk: Owning tracker instance.
        frame: (H, W, C) opencv uint8 BGR image.
        seg_mask: segmentation mask (H, W) already obtained for this frame.

    Returns:
        TrackResult with the fused homography and segmentation mask.
    """
    trk._loma_last_corrected = False
    trk._loma_last_corr_mask = None
    trk._corr_telemetry = CorrectionTelemetry()

    # A silhouette cannot say which corner is which — measured as the dominant
    # error mode of mask-only registration (docs/2026-08-07.md, 2026-08-11).
    # The keypoint branch can, so it is offered to the registrar as an extra
    # pose proposal. That makes the two branches no longer independent, which
    # costs the exact-replay property `experiments/fusion_rule_replay.py` relies
    # on, so it is opt-in and the estimation order changes only when it is on.
    keypoint_proposal = trk.cfg.hough.register_keypoint_proposal
    if keypoint_proposal:
        loma_H, n_inliers = trk._estimate_loma_H(frame, seg_mask)
        trk._samh_tracker._register_extra_H = () if loma_H is None else (loma_H,)
        # The raw matches go with it. They are what the registrar's
        # correspondence-support term scores, and therefore what decides which
        # of a pose's four cyclic namings wins — the one question its silhouette
        # cannot answer (flatsam2.match_naming).
        trk._samh_tracker._register_matches = trk._loma_last_matches

    samh_result = trk._samh_tracker.step_with_mask(frame, seg_mask)
    samh_H = samh_result.homography
    samh_cov = trk._samh_tracker._maskh_last_corner_cov

    if not keypoint_proposal:
        loma_H, n_inliers = trk._estimate_loma_H(frame, seg_mask)
    trk._loma_last_inliers = n_inliers
    loma_cov = trk._loma_last_corner_cov

    trk._loma_last_corrected = trk._maybe_correct_sam3(seg_mask, loma_H, n_inliers)

    # Naming telemetry: the correspondence support of the returned pose's four
    # cyclic namings, with the residual histogram that lets any alternative
    # naming rule be re-scored offline against this run (match_naming).
    _record_naming_telemetry(trk, samh_H)

    # A mask that is a *precise* fit to part of the object is the one failure
    # its covariance cannot report, so it is screened out before the decision
    # rather than discounted inside it. The branch estimate itself is still
    # recorded below — the screen changes what the fusion sees, not what the
    # run stores, so `experiments/fusion_rule_replay.py` stays exact.
    screened = (
        trk.cfg.fusion.occlusion_max_recall > 0.0
        and samh_H is not None
        and _mask_is_partial_view(trk, seg_mask, loma_H)
    )
    fuse_samh_H = None if screened else samh_H
    fuse_samh_cov = None if screened else samh_cov

    last_corners = warp_corners(trk.state.last_homography, trk.state.init_corners)
    fused_corners, fusion_telemetry = fuse_corners(
        trk.state.init_corners,
        loma_H,
        loma_cov,
        fuse_samh_H,
        fuse_samh_cov,
        last_corners,
        trk.cfg.fusion.disagreement_gate,
        trk.cfg.fusion.loma_cov_scale,
        trk.cfg.fusion.samh_cov_scale,
        trk.cfg.fusion.decision_scope,
        trk.cfg.fusion.keypoint_naming_margin,
    )
    fusion_telemetry.mask_screened_occluded = screened

    # Four fused corner positions -> one homography, the same conversion
    # align_mask_to_homography uses for its own control points. Degenerate
    # only if the four fused points are themselves degenerate (collinear,
    # coincident) or ill-conditioned as a homography -- both already unlikely
    # since "coast" falls back to last_corners, a previously-valid quad.
    try:
        fused_H = cv2.getPerspectiveTransform(
            trk.state.init_corners.T.astype(np.float32),
            fused_corners.T.astype(np.float32),
        ).astype(np.float64)
    except cv2.error:
        fused_H = None
    if fused_H is not None and not _is_valid_homography(fused_H):
        fused_H = None

    selected_H = fused_H if fused_H is not None else trk.state.last_homography
    trk.state.last_homography = selected_H

    debug = TrackerDebugInfo(
        loma_H=CandidateScore(H=loma_H),
        samh_H=CandidateScore(H=samh_H),
        fused_H=CandidateScore(H=fused_H),
        selected=_CANDIDATE_FUSED if fused_H is not None else _CANDIDATE_LAST_H,
        corrected=trk._loma_last_corrected,
        n_inliers=n_inliers,
        correction=trk._corr_telemetry,
        # mask-H is computed inside the sub-tracker, so its estimator
        # decision (align vs Hough quad) lives on that instance.
        mask_h=trk._samh_tracker._maskh_telemetry,
        fusion=fusion_telemetry,
    )
    return TrackResult(
        homography=selected_H,
        mask=seg_mask,
        debug=debug,
        correction_mask=trk._loma_last_corr_mask,
    )
