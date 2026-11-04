"""Quad-free mask -> homography registration: propose, refine, verify.

``mask_align`` answers "where did the object move to *from where it was*": one
damped-GN solve started at the last-frame homography. That is the right tool
while tracking is healthy and useless once it is not — the aligner has no way
back onto an object it has lost, which is where the whole quad-free deficit
lives (docs/2026-08-07.md, 2026-08-11: on frames the aligner rejects it is
wrong 100% of the time, while the stateless Hough-quad branch is right on 76%
of those same frames).

This module keeps the same boundary objective but makes the *initialization*
stateless, without assuming the object is a quadrilateral:

1. **Propose.** Second-moment normalization maps the template silhouette onto
   the mask's up to an unknown rotation (``cov_mask = A cov_tpl A^T`` leaves
   exactly one free rotation), so a 1-D sweep over that angle enumerates every
   affine pose the two silhouettes agree on. Moments exist for any shape; no
   corners, lines or convexity are needed anywhere.
2. **Refine.** Each surviving proposal gets a cheap damped-GN solve on a
   downscaled signed-distance field, the winner one full-resolution solve.
3. **Verify.** Proposals are ranked by raster overlap of the warped template
   against the mask, not by the fit's own residual — the residual was measured
   to separate right from wrong fits only weakly (1.77 px vs 0.95 px), because
   a silhouette can sit convincingly on the wrong part of the image.

The last-frame homography stays a proposal like any other, so nothing is lost
while tracking is healthy: when its verified score is already good the search
is skipped entirely and the cost is one GN solve, the same as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from einops import rearrange

from flatsam2.datasets import corners_to_mask, warp_corners
from flatsam2.mask_align import (
    _corner_cov_from_boundary_fit,
    _gn_align,
    _residuals,
    template_boundary,
)
from flatsam2 import match_naming
from flatsam2.mask_metrics import signed_distance
from flatsam2.profiling import profiler


_MIN_MASK_AREA_PX = 50
_MIN_EIGENVALUE = 1e-6
_COARSE_LONG_SIDE = 256


@dataclass
class RegisterConfig:
    """Knobs of the propose/refine/verify pipeline.

    Attributes:
        n_angles: Rotation samples of the second-moment sweep (the only
            unknown left after affine normalization).
        scale_factors: Extra isotropic scalings of each proposal. A partly
            occluded mask has a smaller second moment than the full silhouette,
            so the moment-implied scale is biased low; these cover that bias.
        n_refine: How many top-scoring proposals get a coarse GN refinement.
        n_full: How many refined proposals get the full-resolution solve.
        accept_score: Verified score above which the last-frame proposal is
            taken immediately and the search is skipped (the fast path).
        incumbent_margin: Fraction by which a searched hypothesis must beat the
            last-frame one to displace it. Without it the search costs accuracy
            while tracking is healthy — a silhouette alias scores within a
            percent of the truth, so ties get broken at random on exactly the
            frames that were already correct. This is a tie-break preference,
            not a dependence on the initialization: when the incumbent is
            genuinely lost its score collapses and any margin is cleared.
        coarse_long_side: Longest side of the downscaled raster used for
            proposal scoring and coarse refinement.
        n_boundary_points: Template-boundary sample count for the GN objective.
        max_iters: GN iteration cap at full resolution.
        coarse_max_iters: GN iteration cap on the downscaled field.
        huber_delta_px: Huber transition of the boundary residual (px).
        step_tol_px: GN convergence threshold on the control-point step (px).
        trim_fraction: Share of worst-fitting boundary samples the GN drops
            each iteration (see :func:`mask_align._gn_align`). Off by default:
            it does fix the one occlusion failure it was built for — a hidden
            third of the outline otherwise drags an exactly-correct pose 86 px
            onto the occluder's edge — but measured on PTi/POT dev it costs
            more than it buys everywhere else, because a fit allowed to ignore
            its worst quarter is also happy to explain only part of the
            silhouette. At 0.10: good +1.7 but drift -3.8, cold -4.9, and P@5
            -10. Kept as a knob for a future two-stage design (converge
            untrimmed, then polish trimmed), not as a default.
        overlap_power: Exponent on the raster IoU between the warped template
            and the mask. Measured on PTi dev against the search's own oracle:
            IoU beats every precision-weighted F-measure variant tried, and
            beats the fit residual by a wide margin.
        residual_tau_px: Scale of the residual sharpness factor in the score.
        residual_power: Exponent on that sharpness factor; 0 ignores the
            residual and ranks on raster overlap alone.
        appearance_power: Exponent on the appearance agreement factor; 0
            disables it. Silhouette overlap cannot separate a pose from its
            aliases (a rectangle's half turn maps its own outline onto itself),
            and measurement says that is *the* failure mode of a mask-only
            verifier: mis-picked hypotheses score IoU 0.94 — the same as the
            correct ones — while being 200+ px wrong. Texture breaks the tie.
        appearance_long_side: Longest side of the raster used for the
            appearance check; the coarse mask raster is too small to keep
            usable texture.
        appearance_min_px: Minimum overlap area (in appearance-raster pixels)
            for the correlation to mean anything; below it the factor is
            neutral rather than guessed.
        sdf_precise: Use the exact Euclidean distance transform for the mask's
            signed distance field. Building that field is the single largest
            fixed cost of the branch (42% of a fast-path frame, measured), and
            the 3x3 approximation is several times cheaper — worth checking
            against accuracy rather than assuming exactness is needed.
        sdf_roi_margin: Build the full-resolution signed-distance field on the
            mask's bounding box grown by this fraction of its longer side,
            instead of on the whole frame. The field is only ever sampled at
            warped template-boundary points, which sit on or near the mask, so
            most of a 1280x720 transform is computed for nothing — and that
            transform is 47% of a fast-path frame. Samples falling outside the
            crop clamp to its border, the same way the full-frame field already
            clamps at the image edge, and the Huber loss saturates beyond a few
            pixels anyway, so a distance that is under-estimated far outside the
            crop changes little. 0 keeps the whole frame.
        alias_shortcut: After refining a proposal, also score the three cyclic
            relabelings of its control points. A cyclic shift keeps the warped
            quadrilateral in place and only renames its corners, so the four
            share a silhouette and are separated by texture alone — which is
            precisely the aliasing the verifier exists to break (45% of wrong
            hypotheses are exactly this permutation). Enumerating them costs
            three verifications and no GN solve, where reaching them through
            the angle sweep costs a coarse solve each and is not even
            guaranteed: aliases tie on the overlap that picks which proposals
            get refined, so which of them survives to refinement is arbitrary.
        sweep_stop_score: Stop refining further proposals once one reaches this
            verified score. 0 disables the early exit and refines ``n_refine``
            of them unconditionally.
        alias_recheck: Score the three cyclic relabelings of the *final* winner
            as well. ``alias_shortcut`` only compares relabelings of coarsely
            refined proposals, before the full-resolution solve moves the pose,
            so a mislabeled pose can survive to the output with the verifier's
            own score preferring one of its aliases. Three verifications, no GN
            solve.
        support_power: Exponent on the *correspondence support* term, the
            fraction of the keypoint matcher's raw matches this hypothesis
            explains (``flatsam2.match_naming``). 0 leaves it out entirely,
            which is the historical behaviour. Set to 1 alongside
            ``appearance_power=0`` it replaces the NCC texture term with the
            correspondences already computed for the keypoint branch, and it is
            what names corners: across the four cyclic relabelings of a pose the
            raster IoU is *identical by construction*, so the alias comparison
            is decided by this term alone.
        support_tau_px: Truncation distance of the support (px). A match beyond
            it contributes nothing to any hypothesis and therefore cancels in
            every comparison.
        support_alias_only: Use the correspondence support *only* to choose
            between the cyclic relabelings of a pose, leaving the verification
            score — and therefore its scale — untouched. This is the correct
            placement, for two reasons found by measurement. The score's
            thresholds are not scale-free: `accept_score` and
            `incumbent_margin` are calibrated against a two-factor product, and
            multiplying a third factor <= 1 into it drops the median score from
            0.766 to 0.669, so 54.6% of frames fail acceptance instead of
            39.8%, the incumbent fast path collapses from 88.6% to 77.4% and
            the stateless search runs 2.4x as often — which cost 3.88 pp of
            br_p@15 on PTi. And the support saturates across nearby poses
            anyway (tau = 30 px), so it is not evidence about *which pose*; it
            is evidence about which naming, where IoU is tied by construction.
            Separating the two decisions is what the mechanism claims to do.
        sweep_turn_fraction: Fraction of the full turn the angle sweep spans.
            Below 1.0 this leans on ``alias_shortcut`` to recover the poses the
            shortened sweep skips — safe for the half turn of a parallelogram,
            not safe in general for a quarter turn, since a 90 degree rotation
            of an elongated shape is a genuinely different pose rather than an
            alias. An ablation axis, not an assumption.
    """

    n_angles: int = 24
    scale_factors: Tuple[float, ...] = (1.0,)
    n_refine: int = 4
    n_full: int = 2
    accept_score: float = 0.70
    incumbent_margin: float = 0.4
    alias_shortcut: bool = False
    alias_recheck: bool = False
    sweep_stop_score: float = 0.0
    sweep_turn_fraction: float = 1.0
    coarse_long_side: int = _COARSE_LONG_SIDE
    n_boundary_points: int = 150
    max_iters: int = 60
    coarse_max_iters: int = 25
    huber_delta_px: float = 5.0
    step_tol_px: float = 0.05
    trim_fraction: float = 0.0
    overlap_power: float = 2.0
    residual_tau_px: float = 5.0
    residual_power: float = 0.0
    appearance_power: float = 0.5
    support_power: float = 0.0
    support_tau_px: float = 30.0
    support_alias_only: bool = False
    appearance_long_side: int = 512
    appearance_min_px: int = 200
    sdf_precise: bool = True
    sdf_roi_margin: float = 0.0


@dataclass
class Hypothesis:
    """One candidate pose with everything the verifier looked at.

    Attributes:
        control: (2, 4) image positions of the template corners.
        source: Where the proposal came from (``last``, ``moment``, ``extra``).
        score: Verified score in [0, 1]; higher is better.
        features: Raw verification quantities (overlap parts, residual, ...).
    """

    control: np.ndarray
    source: str
    score: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegisterResult:
    """Outcome of one registration.

    Attributes:
        H: (3, 3) template -> current homography, or None on failure.
        control: (2, 4) final control points.
        score: Verified score of the winner.
        source: Proposal source of the winner.
        residual_px: Median |boundary residual| of the winner (px).
        n_ok: Boundary samples within ``huber_delta_px`` of the mask boundary.
        corner_cov: (4, 2, 2) per-corner covariance, or None.
        n_solves: GN solves actually run (cost accounting).
        hypotheses: Every scored hypothesis, for offline analysis.
    """

    H: Optional[np.ndarray]
    control: Optional[np.ndarray] = None
    score: float = 0.0
    source: str = "none"
    residual_px: float = float("inf")
    n_ok: int = 0
    corner_cov: Optional[np.ndarray] = None
    n_solves: int = 0
    hypotheses: List[Hypothesis] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Second-moment normalization
# ---------------------------------------------------------------------------


def second_moments(mask_bool: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Centroid, second-moment matrix and area of a mask.

    Args:
        mask_bool: (H, W) boolean mask.

    Returns:
        ``(centroid (2,), cov (2, 2), area_px)``, or None when the mask is too
        small or its second moments are degenerate (a one-pixel-wide sliver).
    """
    m = cv2.moments(mask_bool.astype(np.uint8), binaryImage=True)
    area = m["m00"]
    if area < _MIN_MASK_AREA_PX:
        return None
    centroid = np.array([m["m10"] / area, m["m01"] / area])
    cov = np.array(
        [[m["mu20"] / area, m["mu11"] / area], [m["mu11"] / area, m["mu02"] / area]]
    )
    if not np.isfinite(cov).all():
        return None
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= _MIN_EIGENVALUE:
        return None
    return centroid, cov, float(area)


def _resized(image: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
    """Resize by a scale factor, never smaller than one pixel per side."""
    h, w = image.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=interpolation)


def _inv_sqrt(cov: np.ndarray) -> np.ndarray:
    """Inverse matrix square root of a symmetric positive-definite 2x2."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, _MIN_EIGENVALUE))) @ eigvecs.T


def moment_affine_inits(
    template_moments: Tuple[np.ndarray, np.ndarray, float],
    mask_moments: Tuple[np.ndarray, np.ndarray, float],
    template_corners: np.ndarray,
    n_angles: int,
    scale_factors: Sequence[float] = (1.0,),
    turn_fraction: float = 1.0,
) -> List[np.ndarray]:
    """Every affine pose the two silhouettes' second moments are consistent with.

    Whitening both shapes leaves exactly one degree of freedom — the rotation
    between the two whitened frames — because ``cov_mask = A cov_tpl A^T``
    determines ``A = W_mask^-1 R W_tpl`` only up to ``R``. Sweeping ``R``
    therefore enumerates the affine hypotheses without ever asking what the
    shape *is*: no corners, no lines, no convexity. Reflections are excluded on
    purpose (a planar object seen from the front cannot mirror), which halves
    the sweep compared to a general 2x2 search.

    Args:
        template_moments: :func:`second_moments` of the template silhouette.
        mask_moments: :func:`second_moments` of the current mask.
        template_corners: (2, 4) template corner coordinates.
        n_angles: Rotation samples over the swept arc.
        scale_factors: Extra isotropic scalings applied about the mask centroid.
        turn_fraction: Fraction of the full turn to sweep; below 1.0 the caller
            is expected to recover the skipped poses by relabeling corners.

    Returns:
        List of (2, 4) control-point proposals.
    """
    c_tpl, cov_tpl, _ = template_moments
    c_mask, cov_mask, _ = mask_moments
    w_tpl = _inv_sqrt(cov_tpl)
    w_mask_inv = np.linalg.inv(_inv_sqrt(cov_mask))
    centered = template_corners - c_tpl[:, None]

    inits = []
    for angle in np.linspace(0.0, 2.0 * np.pi * turn_fraction, n_angles, endpoint=False):
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        affine = w_mask_inv @ rot @ w_tpl
        base = affine @ centered
        for scale in scale_factors:
            control = c_mask[:, None] + scale * base
            if np.isfinite(control).all():
                inits.append(control)
    return inits


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _sdf_full(mask_bool: np.ndarray, cfg: RegisterConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Full-resolution signed-distance field, optionally cropped to the mask.

    The field is sampled only at warped template-boundary points, which lie on
    or near the mask, so computing it across the whole frame is mostly wasted —
    and it is the single largest fixed cost of a registration. Cropping to the
    mask's bounding box plus ``cfg.sdf_roi_margin`` of its longer side keeps
    every sample that matters and shrinks the transform by the area ratio.

    Args:
        mask_bool: (H, W) boolean mask.
        cfg: Registration configuration.

    Returns:
        The field, and its ``(x, y)`` origin in image coordinates — ``(0, 0)``
        when the whole frame was used.
    """
    zero = np.zeros(2)
    if cfg.sdf_roi_margin <= 0.0:
        return signed_distance(mask_bool, precise=cfg.sdf_precise), zero

    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return signed_distance(mask_bool, precise=cfg.sdf_precise), zero

    h, w = mask_bool.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = int(round(cfg.sdf_roi_margin * max(x1 - x0, y1 - y0)))
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(w, x1 + margin), min(h, y1 + margin)

    return (
        signed_distance(mask_bool[y0:y1, x0:x1], precise=cfg.sdf_precise),
        np.array([float(x0), float(y0)]),
    )


def _overlap(
    template_small: np.ndarray,
    mask_small: np.ndarray,
    template_corners_small: np.ndarray,
    control_small: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Raster agreement of the warped template silhouette with the mask.

    Region overlap is what tells a pose in the right *place* from a pose that
    merely sits on some boundary; the fit residual does not (2026-08-11 study).
    Both one-sided coverages are returned because they separate the two mask
    failure modes: occlusion removes mask pixels (recall of the warp drops,
    precision holds), over-segmentation adds them (the reverse).

    Args:
        template_small: (h, w) downscaled template silhouette, uint8 0/255.
        mask_small: (h, w) downscaled current mask, bool.
        template_corners_small: (2, 4) template corners in downscaled coords.
        control_small: (2, 4) candidate control points in downscaled coords.

    Returns:
        ``{"precision", "recall", "iou", "area_ratio"}``, or None when the
        4-point transform is degenerate.
    """
    try:
        H = cv2.getPerspectiveTransform(
            template_corners_small.T.astype(np.float32),
            control_small.T.astype(np.float32),
        )
    except cv2.error:
        return None
    if not np.isfinite(H).all():
        return None
    h, w = mask_small.shape
    warped = cv2.warpPerspective(template_small, H, (w, h), flags=cv2.INTER_NEAREST) > 127
    area_warp = float(warped.sum())
    area_mask = float(mask_small.sum())
    if area_warp < 1.0 or area_mask < 1.0:
        return None
    inter = float(np.logical_and(warped, mask_small).sum())
    return {
        "precision": inter / area_mask,
        "recall": inter / area_warp,
        "iou": inter / (area_warp + area_mask - inter),
        "area_ratio": area_warp / area_mask,
    }


def _appearance(
    template_gray: np.ndarray,
    template_mask: np.ndarray,
    frame_gray: np.ndarray,
    mask: np.ndarray,
    template_corners_scaled: np.ndarray,
    control_scaled: np.ndarray,
    min_px: int,
) -> Optional[float]:
    """Normalized cross-correlation of the warped template against the frame.

    The silhouette is invariant under the shape's own symmetries; its texture is
    not. Correlating only inside ``warp AND mask`` keeps the measurement on
    surface the segmenter says is object in both frames, so occluders and
    background never enter the sum — an occluded object simply contributes a
    smaller, still-valid region.

    Args:
        template_gray: (h, w) template frame, grayscale float32, appearance scale.
        template_mask: (h, w) template silhouette at appearance scale, uint8.
        frame_gray: (h, w) current frame, grayscale float32, appearance scale.
        mask: (h, w) current segmentation mask at appearance scale, bool.
        template_corners_scaled: (2, 4) template corners at appearance scale.
        control_scaled: (2, 4) candidate control points at appearance scale.
        min_px: Minimum overlap pixels for the correlation to be reported.

    Returns:
        NCC in [-1, 1], or None when the overlap is too small or degenerate.
    """
    try:
        H = cv2.getPerspectiveTransform(
            template_corners_scaled.T.astype(np.float32),
            control_scaled.T.astype(np.float32),
        )
    except cv2.error:
        return None
    if not np.isfinite(H).all():
        return None
    h, w = frame_gray.shape
    warped_gray = cv2.warpPerspective(template_gray, H, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(template_mask, H, (w, h), flags=cv2.INTER_NEAREST) > 127
    region = np.logical_and(warped_mask, mask)
    n = int(region.sum())
    if n < min_px:
        return None
    a = warped_gray[region]
    b = frame_gray[region]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return None
    return float(np.dot(a, b) / denom)


def verification_score(features: Dict[str, float], cfg: RegisterConfig) -> float:
    """Scalar quality of a hypothesis from its verification features.

    Two independent pieces of evidence multiply: *where the silhouette sits*
    (raster IoU) and *whether the surface underneath is the same surface*
    (texture correlation). The silhouette alone is provably insufficient — a
    shape maps onto itself under its own symmetries, and 45% of the wrong
    hypotheses measured on PTi dev are exactly that: the right region with the
    corner labels permuted, scoring IoU 0.94 while being 200 px wrong.

    Args:
        features: Output of :func:`_overlap`, plus ``residual_px`` and ``ncc``.
        cfg: Registration configuration (exponents and residual scale).

    Returns:
        Score in [0, 1]; 0 when the features are missing or degenerate.
    """
    iou = features.get("iou", 0.0)
    if iou <= 0.0:
        return 0.0
    score = iou**cfg.overlap_power
    if cfg.residual_power != 0.0:
        residual = features.get("residual_px", float("inf"))
        score *= (1.0 / (1.0 + max(residual, 0.0) / cfg.residual_tau_px)) ** cfg.residual_power
    if cfg.appearance_power != 0.0:
        ncc = features.get("ncc", None)
        # A missing correlation is no evidence, not bad evidence: leave the
        # silhouette score alone rather than punishing a small overlap twice.
        if ncc is not None and np.isfinite(ncc):
            score *= (0.5 * (1.0 + float(ncc))) ** cfg.appearance_power
    if cfg.support_power != 0.0 and not cfg.support_alias_only:
        # Same convention as the correlation above: no correspondences is no
        # evidence. Present, it is the fraction of matches explained, already
        # in [0, 1] and so directly multiplicable with the overlap. Excluded
        # under support_alias_only, which keeps the score's scale — and the
        # meaning of accept_score — unchanged.
        support = features.get("support", None)
        if support is not None and np.isfinite(support):
            score *= float(support) ** cfg.support_power
    return float(score)


# ---------------------------------------------------------------------------
# Template context (built once per sequence)
# ---------------------------------------------------------------------------


@dataclass
class RegistrationTemplate:
    """Everything about the template that does not change during a sequence.

    Attributes:
        corners: (2, 4) template corner coordinates.
        mask: (H, W) uint8 template silhouette.
        boundary: (N, 2) subsampled template boundary points.
        moments: :func:`second_moments` of the template silhouette.
        small_mask: Downscaled template silhouette, uint8 0/255.
        scale: Downscale factor applied to build ``small_mask``.
        app_gray: Template frame in grayscale at appearance scale, or None when
            no init image was supplied (the appearance check then stays off).
        app_mask: Template silhouette at appearance scale, uint8 0/255.
        app_scale: Downscale factor of the appearance rasters.
    """

    corners: np.ndarray
    mask: np.ndarray
    boundary: np.ndarray
    moments: Tuple[np.ndarray, np.ndarray, float]
    small_mask: np.ndarray
    scale: float
    app_gray: Optional[np.ndarray] = None
    app_mask: Optional[np.ndarray] = None
    app_scale: float = 1.0

    @classmethod
    def build(
        cls,
        template_corners: np.ndarray,
        template_mask: Optional[np.ndarray],
        image_shape: Tuple[int, int],
        cfg: RegisterConfig = RegisterConfig(),
        init_frame: Optional[np.ndarray] = None,
    ) -> Optional["RegistrationTemplate"]:
        """Precompute the per-sequence template context.

        Args:
            template_corners: (2, 4) template corner coordinates.
            template_mask: (H, W) init-frame silhouette; rasterized from the
                corners when None (which reinstates the quad assumption, so
                callers should pass the real mask).
            image_shape: (H, W) of the frames.
            cfg: Registration configuration.
            init_frame: (H, W, 3) or (H, W) init frame. Supplying it enables the
                appearance tie-break; without it registration is mask-only and
                cannot separate a pose from its silhouette aliases.

        Returns:
            The context, or None when the template is unusable.
        """
        mask = template_mask if template_mask is not None else corners_to_mask(
            template_corners, image_shape
        )
        mask_bool = mask > 0
        moments = second_moments(mask_bool)
        if moments is None:
            return None
        boundary = template_boundary(mask_bool, cfg.n_boundary_points)
        if boundary is None:
            return None
        scale = min(1.0, cfg.coarse_long_side / max(image_shape))
        small = _resized(mask_bool.astype(np.uint8) * 255, scale, cv2.INTER_AREA)
        app_scale = min(1.0, cfg.appearance_long_side / max(image_shape))
        app_gray = None
        app_mask = None
        if init_frame is not None and cfg.appearance_power != 0.0:
            gray = init_frame if init_frame.ndim == 2 else cv2.cvtColor(
                init_frame, cv2.COLOR_BGR2GRAY
            )
            app_gray = _resized(gray.astype(np.float32), app_scale, cv2.INTER_AREA)
            app_mask = (
                _resized(mask_bool.astype(np.uint8) * 255, app_scale, cv2.INTER_AREA) > 127
            ).astype(np.uint8) * 255

        return cls(
            corners=template_corners.astype(np.float64),
            mask=mask_bool.astype(np.uint8) * 255,
            boundary=boundary,
            moments=moments,
            small_mask=(small > 127).astype(np.uint8) * 255,
            scale=scale,
            app_gray=app_gray,
            app_mask=app_mask,
            app_scale=app_scale,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def naming_evidence(hyp: "Hypothesis") -> Optional[float]:
    """Correspondence support of a hypothesis, or None when it was not measured.

    The quantity that arbitrates between a pose's cyclic relabelings. It is kept
    apart from :func:`verification_score` deliberately: the relabelings share a
    silhouette exactly, so the score cannot separate them, while the support
    separates them by orders of magnitude and says nothing useful about which
    *pose* is right.
    """
    value = hyp.features.get("support", None)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _alias_beats(challenger: "Hypothesis", incumbent: "Hypothesis",
                 cfg: RegisterConfig, margin: float = 1.0) -> bool:
    """Whether a relabeling should replace the current one.

    Under ``support_alias_only`` the comparison is on correspondence support and
    the verification score is not consulted at all: between relabelings the
    score is a tie by construction, so consulting it only adds noise. Falls back
    to the score when support is unavailable — a frame with too few matches has
    no naming evidence, and the previous behaviour is the right default there.
    """
    if cfg.support_alias_only:
        a, b = naming_evidence(challenger), naming_evidence(incumbent)
        if a is not None and b is not None:
            return a > b * margin
    return challenger.score > incumbent.score * margin


def _best_naming(
    base: "Hypothesis",
    evaluate_fn,
    residual: float,
    cfg: RegisterConfig,
    margin: float = 1.0,
    source: str = "alias",
) -> "Hypothesis":
    """Best of the four cyclic namings of ``base``, enumerated from ``base`` itself.

    The rolls must all be taken against the *frozen* original control points.
    Rolling a mutable running winner instead silently drops a naming: once shift
    1 is accepted, ``roll(winner, 2)`` is ``roll(original, 3)`` and
    ``roll(winner, 3)`` is the original again, so shift 2 is never scored at all.
    That is invisible in the output — a plausible pose is still returned, just
    not the best one — which is why it survived until an audit.

    Args:
        base: Hypothesis whose naming is in question.
        evaluate_fn: ``(control, source, residual) -> Hypothesis | None``.
        residual: Residual to attribute to the relabelings (they share a fit).
        cfg: Registration configuration.
        margin: Factor a challenger must beat the incumbent by; 1.0 for none.
        source: Source label for the generated hypotheses.

    Returns:
        ``base`` or the naming that beats it by ``margin``.
    """
    winner = base
    for shift in (1, 2, 3):
        alias = evaluate_fn(np.roll(base.control, shift, axis=1), source, residual)
        if alias is None:
            continue
        # Every challenger is compared against the incumbent under the margin;
        # a challenger that has already displaced it competes on level terms,
        # since the temporal standing belonged to the incumbent alone.
        if _alias_beats(alias, winner, cfg, margin if winner is base else 1.0):
            winner = alias
    return winner


def register_mask(
    template: RegistrationTemplate,
    mask: np.ndarray,
    last_H: Optional[np.ndarray] = None,
    extra_H: Sequence[np.ndarray] = (),
    cfg: RegisterConfig = RegisterConfig(),
    *,
    frame: Optional[np.ndarray] = None,
    matches: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    keep_hypotheses: bool = False,
) -> RegisterResult:
    """Estimate the template -> current homography from a segmentation mask.

    Args:
        template: Per-sequence context from :meth:`RegistrationTemplate.build`.
        mask: (H, W) current binary segmentation mask.
        last_H: (3, 3) previous-frame homography, used as one proposal among
            others. None runs the stateless path only.
        extra_H: Further externally supplied proposals — e.g. the keypoint
            branch's homography, which carries evidence the mask does not
            (texture, and therefore the orientation of a symmetric silhouette).
        cfg: Registration configuration.
        frame: (H, W, 3) or (H, W) current frame. Together with the template's
            init image it enables the appearance tie-break, which is what
            separates a pose from its silhouette aliases.
        matches: ``(pts_src (N, 2), pts_dst (N, 2))`` raw keypoint
            correspondences, template frame to current frame, *before* RANSAC.
            Enables the correspondence-support term of the verifier
            (``cfg.support_power``), which is what separates a pose from its
            silhouette aliases when there is no texture term. Pre-RANSAC on
            purpose: inliers were selected under a homography that already
            carries one of the four namings, so scoring namings on them is
            circular.
        keep_hypotheses: Keep every scored hypothesis in the result (offline
            analysis; costs nothing but memory).

    Returns:
        A :class:`RegisterResult`; ``H`` is None when nothing usable was found.
    """
    mask_bool = mask > 0
    if int(mask_bool.sum()) < _MIN_MASK_AREA_PX:
        return RegisterResult(H=None)

    with profiler.section("register.prepare"):
        sdf, sdf_origin = _sdf_full(mask_bool, cfg)
    h, w = mask_bool.shape
    small_shape = (max(1, int(round(w * template.scale))), max(1, int(round(h * template.scale))))
    mask_small = cv2.resize(
        mask_bool.astype(np.uint8) * 255, small_shape, interpolation=cv2.INTER_AREA
    ) > 127
    corners_small = template.corners * template.scale
    boundary_small = template.boundary * template.scale
    with profiler.section("register.prepare_coarse"):
        sdf_small = signed_distance(mask_small, precise=cfg.sdf_precise)

    app_gray = None
    app_mask = None
    app_corners = None
    if frame is not None and template.app_gray is not None and cfg.appearance_power != 0.0:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        app_gray = _resized(gray.astype(np.float32), template.app_scale, cv2.INTER_AREA)
        app_mask = _resized(
            mask_bool.astype(np.uint8) * 255, template.app_scale, cv2.INTER_AREA
        ) > 127
        app_corners = template.corners * template.app_scale

    pts_src = pts_dst = None
    if cfg.support_power != 0.0 and matches is not None:
        pts_src, pts_dst = matches
        if (
            pts_src is None or pts_dst is None
            or len(pts_src) < match_naming.MIN_MATCHES
            or len(pts_src) != len(pts_dst)
        ):
            pts_src = pts_dst = None

    state = {"n_solves": 0}
    trace: List[Hypothesis] = []

    def solve(start: np.ndarray, coarse: bool) -> Tuple[Optional[np.ndarray], float]:
        """One damped-GN solve; returns (control, median |residual|) in full-res px."""
        state["n_solves"] += 1
        if coarse:
            with profiler.section("register.solve_coarse"):
                solved, _ = _gn_align(
                    sdf_small, boundary_small, corners_small, start * template.scale,
                    max_iters=cfg.coarse_max_iters,
                    huber_delta=cfg.huber_delta_px * template.scale,
                    step_tol=cfg.step_tol_px,
                    trim_fraction=cfg.trim_fraction,
                )
                r = _residuals(sdf_small, boundary_small, corners_small, solved)
            if r is None:
                return None, float("inf")
            return solved / template.scale, float(np.median(np.abs(r))) / template.scale
        with profiler.section("register.solve_full"):
            # The field may be cropped to the mask, so control points enter and
            # leave it in crop coordinates while everything else stays in image
            # coordinates.
            solved, _ = _gn_align(
                sdf, template.boundary, template.corners, start - sdf_origin[:, None],
                max_iters=cfg.max_iters,
                huber_delta=cfg.huber_delta_px,
                step_tol=cfg.step_tol_px,
                trim_fraction=cfg.trim_fraction,
            )
            r = _residuals(sdf, template.boundary, template.corners, solved)
        if r is None:
            return None, float("inf")
        return solved + sdf_origin[:, None], float(np.median(np.abs(r)))

    def evaluate(
        control: Optional[np.ndarray],
        source: str,
        residual: float,
        with_appearance: bool = True,
    ) -> Optional[Hypothesis]:
        """Score one control-point set with the raster verifier.

        ``with_appearance`` is off while ranking raw proposals: they have not
        been refined yet, so their texture correlation is uninformative and the
        warp would be the most expensive part of the whole search.
        """
        if control is None or not np.isfinite(control).all():
            return None
        with profiler.section("register.verify_overlap"):
            features = _overlap(
                template.small_mask, mask_small, corners_small, control * template.scale
            )
        if features is None:
            return None
        features["residual_px"] = residual
        if with_appearance and app_gray is not None:
            with profiler.section("register.verify_appearance"):
                ncc = _appearance(
                    template.app_gray, template.app_mask, app_gray, app_mask,
                    app_corners, control * template.app_scale, cfg.appearance_min_px,
                )
            if ncc is not None:
                features["ncc"] = ncc
        if with_appearance and pts_src is not None:
            with profiler.section("register.verify_support"):
                H = match_naming.homography_from_corners(template.corners, control)
                support = (
                    None if H is None
                    else match_naming.match_support(
                        H, pts_src, pts_dst, cfg.support_tau_px
                    )
                )
            if support is not None:
                features["support"] = support / len(pts_src)
        hyp = Hypothesis(control=control, source=source, features=features)
        hyp.score = verification_score(features, cfg)
        if keep_hypotheses:
            trace.append(hyp)
        return hyp

    hypotheses: List[Hypothesis] = []

    # 1. The tracking proposal: last frame's pose, plus anything external.
    starts = []
    if last_H is not None and np.isfinite(last_H).all():
        starts.append((warp_corners(last_H, template.corners), "last"))
    for i, H in enumerate(extra_H):
        if H is not None and np.isfinite(H).all():
            starts.append((warp_corners(H, template.corners), f"extra{i}"))
    for start, source in starts:
        if not np.isfinite(start).all():
            continue
        control, residual = solve(start, coarse=False)
        hyp = evaluate(control, source, residual)
        if hyp is not None:
            hypotheses.append(hyp)

    def selection_score(hyp: Hypothesis) -> float:
        """Verified score with the incumbent's tie-break preference applied."""
        return hyp.score * (1.0 + cfg.incumbent_margin) if hyp.source == "last" else hyp.score

    best = max(hypotheses, key=lambda x: x.score, default=None)

    # 2. Stateless search — only when the tracking proposal is not convincing,
    #    so a healthy frame still costs exactly one GN solve.
    if best is None or best.score < cfg.accept_score:
        mask_moments = second_moments(mask_bool)
        if mask_moments is not None:
            with profiler.section("register.propose"):
                proposals = moment_affine_inits(
                    template.moments, mask_moments, template.corners,
                    cfg.n_angles, cfg.scale_factors, cfg.sweep_turn_fraction,
                )
            scored = []
            for control in proposals:
                hyp = evaluate(control, "propose", float("inf"), with_appearance=False)
                if hyp is not None:
                    # Unrefined proposals have no residual yet: rank on overlap.
                    hyp.score = verification_score(
                        {**hyp.features, "residual_px": 0.0}, cfg
                    )
                    scored.append(hyp)
            scored.sort(key=lambda x: x.score, reverse=True)

            refined = []
            for hyp in scored[: cfg.n_refine]:
                control, residual = solve(hyp.control, coarse=True)
                got = evaluate(control, "coarse", residual)
                if got is None:
                    continue
                # The three cyclic relabelings of a refined pose: same warped
                # quadrilateral, renamed corners. They cost a verification each
                # and no solve, and they are the poses the overlap ranking
                # cannot tell apart.
                if cfg.alias_shortcut:
                    got = _best_naming(got, evaluate, residual, cfg)
                refined.append(got)
                if cfg.sweep_stop_score > 0.0 and got.score >= cfg.sweep_stop_score:
                    break
            refined.sort(key=lambda x: x.score, reverse=True)

            # n_full=0 keeps the coarse solution instead of paying a
            # full-resolution solve — the cheap end of the accuracy/cost dial,
            # not a disabled search.
            if cfg.n_full == 0:
                hypotheses.extend(refined[:1])

            for hyp in refined[: cfg.n_full]:
                control, residual = solve(hyp.control, coarse=False)
                got = evaluate(control, "moment", residual)
                if got is not None:
                    hypotheses.append(got)

    if not hypotheses:
        return RegisterResult(H=None, n_solves=state["n_solves"], hypotheses=trace)

    winner = max(hypotheses, key=selection_score)

    # The alias shortcut above compares relabelings of a *coarsely* refined
    # proposal; the full-resolution solve then moves the pose, and nothing
    # re-examines its naming. Measured on a forced-lost frame (PTi Seq_00006
    # frame 30): the returned pose scored 0.770 at 342 px from GT while its own
    # +1 relabeling scored 0.859 at 20.9 px. So the final pose gets the same
    # three verifications, which cost no GN solve. The incumbent margin is
    # deliberately not applied here: it decided which *pose* wins, and a
    # relabeling is the same pose with different corner names.
    if cfg.alias_recheck:
        # A naming inherited from the previous frame was already verified there,
        # and temporal consistency is evidence the single-frame score does not
        # have. Renaming it therefore has to clear the same margin a challenger
        # pose does; a naming produced by the search this frame has no such
        # standing and is replaced on a plain score comparison. Without this
        # asymmetry the re-check costs 3.0 pp P@15 in the `good` regime while
        # gaining 0.9 in `lost` (maskh_bench, PTi test, 961 cases per regime).
        inherited = winner.source == "last" or winner.source.startswith("extra")
        winner = _best_naming(
            winner,
            evaluate,
            winner.features.get("residual_px", float("inf")),
            cfg,
            margin=(1.0 + cfg.incumbent_margin) if inherited else 1.0,
            source=f"{winner.source}_alias",
        )

    try:
        H = cv2.getPerspectiveTransform(
            template.corners.T.astype(np.float32), winner.control.T.astype(np.float32)
        )
    except cv2.error:
        return RegisterResult(H=None, n_solves=state["n_solves"], hypotheses=trace)

    winner_in_field = winner.control - sdf_origin[:, None]
    r = _residuals(sdf, template.boundary, template.corners, winner_in_field)
    n_ok = 0 if r is None else int((np.abs(r) <= cfg.huber_delta_px).sum())
    corner_cov = None
    if r is not None:
        corner_cov = _corner_cov_from_boundary_fit(
            sdf, template.boundary, template.corners, winner_in_field, r, cfg.huber_delta_px
        )
    return RegisterResult(
        H=H.astype(np.float64),
        control=winner.control,
        score=winner.score,
        source=winner.source,
        residual_px=winner.features.get("residual_px", float("inf")),
        n_ok=n_ok,
        corner_cov=corner_cov,
        n_solves=state["n_solves"],
        hypotheses=trace,
    )


# ---------------------------------------------------------------------------
# Benchmark hooks (consumed by experiments/maskh_bench.py)
# ---------------------------------------------------------------------------


def bench_methods() -> Dict[str, object]:
    """Estimators this module exposes to ``experiments/maskh_bench.py``."""

    def _make(name: str, cfg: RegisterConfig, probe: bool = False, keypoint: bool = False):
        def method(mask, template_mask, template_corners, init_corners, init_H, ctx):
            template = RegistrationTemplate.build(
                template_corners, template_mask, mask.shape, cfg,
                init_frame=ctx.get("init_frame"),
            )
            if template is None:
                return None, {}
            keypoint_H = ctx.get("extra_H") if keypoint else None
            result = register_mask(
                template, mask, last_H=init_H, cfg=cfg,
                extra_H=() if keypoint_H is None else (keypoint_H,),
                frame=ctx.get("frame"), keep_hypotheses=probe,
            )
            extra = {
                "n": float(result.n_ok),
                "residual": result.residual_px,
                "score": result.score,
                "source": result.source,
                "n_solves": float(result.n_solves),
            }
            if probe:
                extra["hypotheses"] = [
                    {"source": h.source, "score": h.score, "control": h.control, **h.features}
                    for h in result.hypotheses
                ]
            return result.H, extra

        method.__name__ = name
        # The benchmark only fills ctx["extra_H"] when it was given a run to
        # source keypoint homographies from. Without it a keypoint method is
        # silently identical to its non-keypoint twin, which reads as "the
        # keypoint proposal does nothing" rather than as a missing input.
        method.needs_extra_h = keypoint
        # The bench loads frames only when some method needs them; a method that
        # uses the appearance term and does not get them is silently its
        # shape-only twin.
        method.needs_frames = cfg.appearance_power != 0.0
        return method

    # Paper ablations. Each line is one axis of the method, held against the
    # deployed default; the benchmark runs them all over one pass of the test
    # set, so a full ablation table costs minutes on CPU rather than a grid.
    ablations = {
        # --- deployed configuration and its keypoint variant ----------------
        "register": RegisterConfig(),
        "register_always": RegisterConfig(accept_score=1.01),

        # --- A. what the verifier is allowed to look at ---------------------
        # The headline mechanism claim: shape alone cannot resolve pose.
        "abl_verify_shape_only": RegisterConfig(appearance_power=0.0),
        "abl_verify_texture_only": RegisterConfig(overlap_power=0.0),
        "abl_verify_residual": RegisterConfig(
            overlap_power=0.0, appearance_power=0.0, residual_power=1.0
        ),
        "abl_verify_shape_texture_residual": RegisterConfig(residual_power=1.0),

        # --- B. where proposals come from -----------------------------------
        # n_angles=0 leaves only the incumbent, i.e. the plain aligner.
        "abl_prop_incumbent_only": RegisterConfig(n_angles=0),
        "abl_prop_angles8": RegisterConfig(n_angles=8),
        "abl_prop_angles12": RegisterConfig(n_angles=12),
        "abl_prop_angles16": RegisterConfig(n_angles=16),
        "abl_prop_angles36": RegisterConfig(n_angles=36),

        # --- C. arbitration between incumbent and challengers ---------------
        "abl_arb_search_always": RegisterConfig(accept_score=1.01),
        "abl_arb_accept55": RegisterConfig(accept_score=0.55),
        "abl_arb_accept85": RegisterConfig(accept_score=0.85),
        "abl_arb_margin0": RegisterConfig(incumbent_margin=0.0),
        "abl_arb_margin20": RegisterConfig(incumbent_margin=0.2),
        "abl_arb_margin80": RegisterConfig(incumbent_margin=0.8),

        # --- D. how much refinement the search buys -------------------------
        "abl_ref_1x1": RegisterConfig(n_refine=1, n_full=1),
        "abl_ref_2x1": RegisterConfig(n_refine=2, n_full=1),
        "abl_ref_6x3": RegisterConfig(n_refine=6, n_full=3),
        "abl_ref_no_full": RegisterConfig(n_full=0),

        # --- E. resolution, i.e. the accuracy/cost dial ---------------------
        "abl_res_coarse128": RegisterConfig(coarse_long_side=128),
        "abl_res_coarse192": RegisterConfig(coarse_long_side=192),
        "abl_res_coarse384": RegisterConfig(coarse_long_side=384),
        "abl_res_app256": RegisterConfig(appearance_long_side=256),
        "abl_res_app384": RegisterConfig(appearance_long_side=384),
        "abl_res_bnd75": RegisterConfig(n_boundary_points=75),
        "abl_res_bnd300": RegisterConfig(n_boundary_points=300),
        # The distance transform is the biggest fixed cost of a frame.
        "abl_res_sdf_fast": RegisterConfig(sdf_precise=False),

        # --- F. occlusion handling (measured, not deployed) -----------------
        "abl_trim10": RegisterConfig(trim_fraction=0.10),
        "abl_trim25": RegisterConfig(trim_fraction=0.25),

        # --- G. making the sweep cheaper ------------------------------------
        # The sweep spends a coarse solve per refined proposal, and its top-k
        # by overlap is frequently four aliases of one pose — the ranking that
        # picks them cannot separate poses that share a silhouette. Relabeling
        # a refined pose reaches the same four for three verifications and no
        # solve, which is what makes an early exit affordable: stop at the
        # first convincing angle and enumerate its aliases explicitly.
        "abl_fast_alias": RegisterConfig(alias_shortcut=True),
        "abl_fast_alias_stop": RegisterConfig(alias_shortcut=True, sweep_stop_score=0.70),
        "abl_fast_alias_stop_1x1": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, n_refine=1, n_full=1
        ),
        "abl_fast_alias_stop_2x1": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, n_refine=2, n_full=1
        ),
        # Half a turn plus relabeling covers the 180 degree alias of any
        # parallelogram; a quarter turn additionally assumes 90 degrees is an
        # alias, which only holds for a near-square outline.
        "abl_fast_half_turn": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, n_angles=12, sweep_turn_fraction=0.5
        ),
        "abl_fast_quarter_turn": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, n_angles=6, sweep_turn_fraction=0.25
        ),
        # --- H. is the verification score's shape justified? ----------------
        # score = IoU^a * ((1+NCC)/2)^b is a weighted geometric mean, i.e. a
        # log-linear pool of two evidence terms. The deployed weights (a=2,
        # b=0.5) were set by hand; these ablations say whether the weighting
        # carries anything over the equal-weight product, which is the form a
        # first-principles argument gives.
        "abl_score_w11": RegisterConfig(
            overlap_power=1.0, appearance_power=1.0,
            alias_shortcut=True, sweep_stop_score=0.70, sdf_roi_margin=1.0,
        ),
        "abl_score_w21": RegisterConfig(
            overlap_power=2.0, appearance_power=1.0,
            alias_shortcut=True, sweep_stop_score=0.70, sdf_roi_margin=1.0,
        ),
        "abl_score_w12": RegisterConfig(
            overlap_power=1.0, appearance_power=2.0,
            alias_shortcut=True, sweep_stop_score=0.70, sdf_roi_margin=1.0,
        ),
        "abl_score_deployed": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, sdf_roi_margin=1.0,
        ),
        # --- I. the alias re-check on the final pose ------------------------
        "abl_alias_recheck": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, sdf_roi_margin=1.0,
            alias_recheck=True,
        ),
        # The two cheap dials that do not touch the search at all.
        "abl_fast_sdf": RegisterConfig(alias_shortcut=True, sweep_stop_score=0.70, sdf_precise=False),
        # The distance transform runs on the whole frame but is only read near
        # the mask. These crop it; margin is a fraction of the mask's longer side.
        "abl_sdf_roi25": RegisterConfig(sdf_roi_margin=0.25),
        "abl_sdf_roi50": RegisterConfig(sdf_roi_margin=0.50),
        "abl_sdf_roi100": RegisterConfig(sdf_roi_margin=1.00),
        "abl_sdf_roi50_fast": RegisterConfig(sdf_roi_margin=0.50, sdf_precise=False),
        "abl_fast_all": RegisterConfig(
            alias_shortcut=True, sweep_stop_score=0.70, n_refine=2, n_full=1,
            n_angles=12, sweep_turn_fraction=0.5, sdf_precise=False,
        ),
    }
    methods = {name: _make(name, cfg) for name, cfg in ablations.items()}

    # The keypoint branch as an extra proposal: it carries the corner identity
    # a silhouette cannot, which is the residual error mode.
    methods["register_kp"] = _make("register_kp", RegisterConfig(), keypoint=True)
    methods["abl_kp_shape_only"] = _make(
        "abl_kp_shape_only", RegisterConfig(appearance_power=0.0), keypoint=True
    )
    # The deployed candidate (configs/tracker/corner_fusion_register_recheck.py)
    # and the same thing with the verification score's two exponents set to one,
    # i.e. score = IoU * (1+NCC)/2 — the plain product of the two evidence terms,
    # with nothing left to tune. `abl_score_w11` measures that weighting without
    # the keypoint proposal or the re-check; this pair measures it where it would
    # actually ship.
    _final_kwargs = dict(
        alias_shortcut=True, sweep_stop_score=0.70, sdf_roi_margin=1.0, alias_recheck=True
    )
    methods["register_final_kp"] = _make(
        "register_final_kp", RegisterConfig(**_final_kwargs), keypoint=True
    )
    methods["register_final_kp_w11"] = _make(
        "register_final_kp_w11",
        RegisterConfig(overlap_power=1.0, appearance_power=1.0, **_final_kwargs),
        keypoint=True,
    )
    # Probe: searches unconditionally and reports every hypothesis it scored, so
    # the benchmark can separate proposal quality from verifier quality.
    methods["register_probe"] = _make(
        "register_probe", RegisterConfig(accept_score=1.01, n_refine=6, n_full=3), probe=True
    )
    return methods
