"""Verbs and types shared across every tracking algorithm.

Pure homography/mask helpers, candidate-source labels, and the per-frame
result/debug dataclasses. Nothing here depends on the
:class:`~flatsam2.tracker.FlatSAMTracker` class, so the per-algorithm modules
can import from here without a circular dependency on ``tracker``.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

from flatsam2.config import CorrectionCfg
from flatsam2.mask_metrics import cropped_signed_area_discrepancy


# Condition-number threshold above which a homography is considered numerically singular.
_HOMOGRAPHY_COND_THRESHOLD = 1e10

_CANDIDATE_FLOW = "flow_H"
_CANDIDATE_MASK_H = "mask_H"
_CANDIDATE_REDETECTION_FLOW = "redetection_flow_H"
_CANDIDATE_LAST_H = "last_H"
_CANDIDATE_LOMA = "loma_H"
_CANDIDATE_SAMH = "samh_H"

# Estimator that produced mask-H on a frame (see MaskHTelemetry.source).
_MASK_H_SOURCE_HOUGH = "hough"
_MASK_H_SOURCE_NONE = "none"

# Ordered superset of candidate homography sources captured in debug output.
# Every tracker path fills the subset it produces; the rest stay empty.
_DEBUG_CANDIDATES = (
    "flow_H",
    "redetection_flow_H",
    "mask_H",
    "loma_H",
    "samh_H",
)


def _is_valid_homography(H: np.ndarray) -> bool:
    """Check whether a homography matrix is numerically valid (non-singular).

    Uses the matrix condition number as a proxy for singularity.  Additional
    geometric validity checks (e.g. vertex-order preservation) can be added
    here in the future.

    Args:
        H: Candidate homography matrix (3, 3).

    Returns:
        True if the homography passes all validity checks, False otherwise.
    """
    if H is None:
        return False
    if H.shape != (3, 3):
        return False
    cond = np.linalg.cond(H)
    return bool(np.isfinite(cond) and cond < _HOMOGRAPHY_COND_THRESHOLD)


def _inflate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate binary mask by a square kernel.

    Args:
        mask: Binary mask `(H, W)`.
        pixels: Dilation radius in pixels; zero disables inflation.

    Returns:
        Inflated binary mask `(H, W)`.
    """
    if pixels <= 0:
        return mask
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def _union_candidate_masks(candidates: dict, fallback: np.ndarray) -> np.ndarray:
    """Union every SAM3 candidate mask into one binary sampling region.

    Args:
        candidates: SAM3 multi-mask output from
            ``SAM3StreamingTracker.track(..., return_all_masks=True)``, with
            ``"masks"`` an `(M, H, W)` bool array (``M`` may be 0 when SAM3
            reports the object absent).
        fallback: Selected mask `(H, W)` used when no candidate carries area
            (e.g. object absent), so densification never collapses to empty.

    Returns:
        Binary union mask `(H, W)` uint8.
    """
    masks = candidates["masks"]
    if masks.shape[0] == 0:
        return (fallback > 0).astype(np.uint8)
    union = np.any(masks, axis=0)
    if not union.any():
        return (fallback > 0).astype(np.uint8)
    return union.astype(np.uint8)


@dataclass
class CorrectionTelemetry:
    """Always-on per-frame correction observables; NaN when a source is unavailable.

    Computed whenever ``cfg.correction.enabled`` (live and dry-run configs
    alike), independent of the inlier gate and of the trigger outcome, so any
    trigger rule can be evaluated offline from a single run.

    Attributes:
        spill_loma: ``(loma_mask & ~sam) / area(loma_mask)`` — the superset-guard
            value measured against the LoMa-induced mask.
        excess_loma: ``dout`` at ``cfg.excess_depth_px`` vs the LoMa-induced mask
            (fraction of SAM area lying more than T px past its boundary).
        area_ratio_loma: ``area(sam) / area(loma_mask)``.
        trigger_fired: Trigger condition held this frame, before the
            content-availability check — ``trigger_fired & ~corrected`` marks an
            abstain (or a dry run).
    """

    spill_loma: float = float("nan")
    excess_loma: float = float("nan")
    area_ratio_loma: float = float("nan")
    trigger_fired: bool = False


def _reference_mask_features(
    sam: np.ndarray,
    reference: np.ndarray,
    excess_depth_px: int,
) -> tuple[float, float, float]:
    """Trigger features of the SAM mask measured against one reference mask.

    Args:
        sam: Current SAM mask `(H, W)` bool.
        reference: Homography-induced reference mask `(H, W)` bool.
        excess_depth_px: Boundary tolerance T (px) for the excess-depth metric.

    Returns:
        Tuple ``(spill, excess, area_ratio)``: superset-guard value
        ``(reference & ~sam) / area(reference)``, ``dout`` at T px, and
        ``area(sam) / area(reference)``; all NaN when the reference is empty.
    """
    a_ref = int(reference.sum())
    if a_ref == 0:
        return float("nan"), float("nan"), float("nan")
    spill = float(int((reference & ~sam).sum()) / a_ref)
    t = int(excess_depth_px)
    dout, _ = cropped_signed_area_discrepancy(reference, sam, [t])
    return spill, float(dout[t]), float(int(sam.sum()) / a_ref)


def _correction_telemetry(
    sam_mask: np.ndarray,
    loma_mask: Optional[np.ndarray],
    cfg: CorrectionCfg,
) -> CorrectionTelemetry:
    """Fill the per-frame correction observables when the LoMa mask exists.

    Args:
        sam_mask: Current SAM mask `(H, W)` (any truthy dtype).
        loma_mask: LoMa-induced mask `(H, W)` bool, or None when loma_H invalid.
        cfg: Correction config (supplies ``excess_depth_px``).

    Returns:
        CorrectionTelemetry with the available fields set (``trigger_fired`` is
        the caller's responsibility).
    """
    telemetry = CorrectionTelemetry()
    if loma_mask is not None:
        telemetry.spill_loma, telemetry.excess_loma, telemetry.area_ratio_loma = (
            _reference_mask_features(sam_mask > 0, loma_mask, cfg.excess_depth_px)
        )
    return telemetry


def _should_correct(
    sam_mask: np.ndarray,
    corr_mask: np.ndarray,
    cfg: CorrectionCfg,
) -> bool:
    """Decide whether SAM3 mask should be corrected.

    Args:
        sam_mask: Current SAM mask `(H, W)`.
        corr_mask: LoMa-implied correction mask `(H, W)`.
        cfg: Correction trigger config.

    Returns:
        True when correction should be applied.

    Raises:
        ValueError: If trigger is unknown.
    """
    sam = sam_mask > 0
    corr = corr_mask > 0
    a_corr = int(corr.sum())
    if a_corr == 0:
        return False
    # Superset guard (shared): abstain when the correction mask spills outside SAM
    # — replacing SAM would then expand it, the wrong direction for an over-seg fix.
    outside = int((corr & ~sam).sum()) / a_corr
    if outside > cfg.superset_tolerance:
        return False
    if cfg.trigger == "area_margin":
        return bool(int(sam.sum()) > (1.0 + cfg.area_margin) * a_corr)
    if cfg.trigger == "sam_excess_depth":
        # dout_corr: SAM area more than T px past the corr boundary, / corr area
        # (reference = corr, other = sam). Cropping is handled inside the util.
        t = int(cfg.excess_depth_px)
        dout, _ = cropped_signed_area_discrepancy(corr, sam, [t])
        return bool(dout[t] >= cfg.excess_margin)
    raise ValueError(f"unknown correction trigger {cfg.trigger!r}")


@dataclass
class CandidateScore:
    """One homography candidate.

    Attributes:
        H: Homography (3, 3), or None if this candidate was not computed.
    """

    H: Optional[np.ndarray] = None


@dataclass
class MaskHTelemetry:
    """Whether mask-H was produced on this frame, and how its corners were named.

    Attributes:
        source: ``'hough'`` when the Hough quad produced mask-H, ``'none'`` when
            mask-H was unavailable this frame.
        naming_support: (4,) correspondence support of the four cyclic namings
            of the returned pose, in effective matches
            (``flatsam2.match_naming``). NaN when it was not measured.
        naming_n_matches: Raw match count backing that support; -1 when N/A.
        naming_shift: Cyclic shift the correspondence namer actually applied --
            0 when it looked and kept the incumbent naming, 1-3 when it
            re-named, -1 when it did not run. Without this a null result is
            ambiguous: "the mechanism did not help" and "the mechanism never
            fired" look identical in the metrics.
        naming_hist_counts: (4, B) residual histogram per naming, over
            ``match_naming.RESIDUAL_BIN_EDGES``.
        naming_hist_sumsq: (4, B) sum of squared residuals per bin, which is
            what makes an offline re-test of ``support_tau_px`` exact rather
            than bin-resolution. Together these let any alternative naming rule
            — a different tau, a margin over the incumbent, an abstain
            threshold — be scored against a finished run without a GPU.
    """

    source: str = _MASK_H_SOURCE_NONE
    naming_support: Optional[np.ndarray] = None
    naming_n_matches: int = -1
    naming_shift: int = -1
    naming_hist_counts: Optional[np.ndarray] = None
    naming_hist_sumsq: Optional[np.ndarray] = None


@dataclass
class TrackerDebugInfo:
    """Universal per-frame tracker telemetry for visualization and analysis.

    Populated by every tracking path, so a single ``{seq}_debug.npz`` captures
    candidate homographies, the selected source, LoMa inlier counts, and SAM3
    correction events. Fields a given path does not produce keep their empty
    defaults (``H=None``).

    Attributes:
        flow_H: Flow-H estimated with last-H prewarp.
        redetection_flow_H: Flow-H estimated with ``mask_H`` prewarp.
        mask_H: ``mask_H`` from the Hough pipeline.
        loma_H: LoMa keypoint homography (online LoMa path).
        samh_H: Independent SAM-H sub-tracker homography (decoupled LoMa path).
        selected: Label of the candidate actually used, e.g. ``'flow_H'``,
            ``'mask_H'``, ``'loma_H'``, ``'samh_H'``, ``'last_H'``.
        corrected: Whether a SAM3 self-correction fired on this frame.
        n_inliers: LoMa RANSAC inlier count this frame; ``-1`` when N/A.
        correction: Always-on correction observables (NaN defaults when the
            correction machinery did not run).
        mask_h: Whether mask-H was produced and how its corners were named. In
            the decoupled LoMa path this mirrors the SAM-H sub-tracker, since
            that is where mask-H is computed.
    """

    flow_H: CandidateScore = field(default_factory=CandidateScore)
    redetection_flow_H: CandidateScore = field(default_factory=CandidateScore)
    mask_H: CandidateScore = field(default_factory=CandidateScore)
    loma_H: CandidateScore = field(default_factory=CandidateScore)
    samh_H: CandidateScore = field(default_factory=CandidateScore)
    selected: str = _CANDIDATE_LAST_H
    corrected: bool = False
    n_inliers: int = -1
    correction: CorrectionTelemetry = field(default_factory=CorrectionTelemetry)
    mask_h: MaskHTelemetry = field(default_factory=MaskHTelemetry)


@dataclass
class TrackResult:
    """Output of a single tracking step.

    Attributes:
        homography: Estimated homography (3, 3) mapping init frame -> current frame.
        mask: Segmentation mask (H, W) uint8 for the tracked object in the current frame.
        debug: Per-frame tracker telemetry (always populated).
        correction_mask: LoMa-implied mask injected into SAM3 memory this frame,
            ``(H, W)`` uint8, or ``None`` when no correction fired. The ``mask``
            field is the original SAM3 output; this is the correction applied.
    """

    homography: np.ndarray
    mask: np.ndarray
    debug: Optional[TrackerDebugInfo] = None
    correction_mask: Optional[np.ndarray] = None


@dataclass
class TrackerState:
    visible: bool = True
    init_frame: Optional[np.ndarray] = None
    init_mask: Optional[np.ndarray] = None
    init_corners: Optional[np.ndarray] = None  # (2, 4) if provided
    last_homography: Optional[np.ndarray] = None  # last_pos = H * initial_pos
    frame_idx: int = 0  # tracked-frame counter (init frame = 0); drives periodic correction

    # Symmetry resolution state (only used when symmetry is enabled).
    symmetry: Optional[object] = None  # SymmetryState from hough_homography
