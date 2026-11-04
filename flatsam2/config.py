"""Configuration system for FlatSAM tracker using Pydantic."""

import importlib.util
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from .registry import DATASETS

Algorithm = Literal["flow_hough", "nn_min_rmse", "nn_cascade", "loma_samh", "corner_fusion"]


class FlowConfig(BaseModel):
    """Optical flow configuration.

    Args:
        model: ptlflow model name (e.g., 'raft', 'gma', 'unimatch')
        weights: model weights variant (e.g., 'sintel', 'kitti', 'things')
    """

    model: str = Field(default="raft", description="ptlflow model name")
    weights: str = Field(default="sintel", description="model weights variant")


class SegmenterConfig(BaseModel):
    """Segmentation model configuration.

    Args:
        model: segmentation model name (currently only 'sam3' supported)
    """

    model: str = Field(default="sam3", description="segmentation model name")


class RansacConfig(BaseModel):
    """RANSAC configuration."""

    inl_threshold: float = Field(
        default=5.0,
        description="RANSAC inlier threshold in pixels",
    )
    ransac_seed: Optional[int] = Field(
        default=0,
        description=(
            "If set, seed OpenCV's global RNG before the flow RANSAC call in "
            "_estimate_flow_H so its result is reproducible. Defaults to 0 "
            "(deterministic, no runtime penalty); set to None to keep OpenCV's "
            "RNG state untouched for legacy/stochastic behaviour. Mirrors "
            "LoMaPathConfig.ransac_seed for the flow path."
        ),
    )


class HoughConfig(BaseModel):
    """Mask -> homography estimator configuration (Hough pipeline or direct alignment)."""

    enabled: bool = Field(
        default=False,
        description="Enable the mask-based homography estimator (SAM-H branch)",
    )
    method: Literal["hough", "align", "register"] = Field(
        default="hough",
        description=(
            "Mask->H estimator: 'hough' = lines->quad pipeline (legacy); 'align' = "
            "direct damped-GN mask alignment on the 4-point parameterization (P3'), "
            "initialized from the last-frame homography; 'register' = quad-free "
            "propose/refine/verify registration (flatsam2/mask_register.py), which "
            "adds a stateless second-moment pose search and a texture-verified "
            "choice among hypotheses, so it can recover from a lost pose"
        ),
    )
    align_n_boundary_points: int = Field(
        default=150,
        description="Template-boundary sample count for the align method",
    )
    align_max_iters: int = Field(
        default=60,
        description="Gauss-Newton iteration cap for the align method",
    )
    align_huber_delta_px: float = Field(
        default=5.0,
        description="Huber transition (px) of the boundary residual",
    )
    align_step_tol_px: float = Field(
        default=0.05,
        description="GN convergence threshold on the control-point step (px)",
    )
    align_residual_ok_px: float = Field(
        default=6.0,
        description=(
            "Accept the aligned H only when the median |boundary residual| is at most "
            "this (px); otherwise fall back per align_fallback_hough"
        ),
    )
    align_fallback_hough: bool = Field(
        default=True,
        description="Run the Hough pipeline when alignment fails or is rejected",
    )
    align_redetect_moments: bool = Field(
        default=False,
        description=(
            "When the last-frame-initialized align solve is about to be rejected "
            "(residual above align_residual_ok_px), re-solve from the poses the "
            "current mask's image moments imply and keep the best fit. Gives the "
            "aligner the stateless redetection the Hough quad pipeline has for "
            "free, without a quadrilateral assumption. Off by default (extra GN "
            "solves, only on frames that would otherwise be rejected)."
        ),
    )
    register_n_angles: int = Field(
        default=24,
        description=(
            "Rotation samples of the register method's second-moment sweep — the "
            "only unknown left once both silhouettes are whitened"
        ),
    )
    register_accept_score: float = Field(
        default=0.70,
        description=(
            "Verified score above which the last-frame pose is taken immediately "
            "and the stateless search is skipped (the fast path). Healthy frames "
            "then cost one GN solve, as with method='align'"
        ),
    )
    register_incumbent_margin: float = Field(
        default=0.4,
        description=(
            "Fraction by which a searched hypothesis must beat the last-frame one "
            "to displace it; without it the search costs accuracy on frames that "
            "were already right, because silhouette aliases score within a percent "
            "of the truth"
        ),
    )
    register_overlap_power: float = Field(
        default=2.0,
        description=(
            "Exponent a on the raster IoU in the verification score "
            "score = IoU^a * ((1+NCC)/2)^b, i.e. a log-linear pool of the shape "
            "and texture evidence. Kept at the historical 2.0 so existing runs "
            "reproduce; a=b=1 (the plain product, no weights to tune) measures "
            "equal or better everywhere except PTi `good` by 0.2 pp — see "
            "configs/tracker/corner_fusion_simple.py"
        ),
    )
    register_appearance_power: float = Field(
        default=0.5,
        description=(
            "Exponent b on the texture-correlation factor of the same score. "
            "0 makes the verifier mask-only, which measurement says gives up ~16 "
            "points of P@15 on a cold start (silhouette aliasing)"
        ),
    )
    register_support_power: float = Field(
        default=0.0,
        description=(
            "Exponent on the correspondence-support term of the verification "
            "score: the fraction of the keypoint matcher's raw (pre-RANSAC) "
            "matches a hypothesis explains, under a truncated-quadratic cost "
            "(flatsam2.match_naming). 0 leaves it out, the historical "
            "behaviour. Set to 1 together with register_appearance_power=0 it "
            "replaces the NCC texture term with the correspondences already "
            "computed for the keypoint branch, giving score = IoU * U/N -- one "
            "shape term and one texture term, no exponents. It is also what "
            "names corners: the four cyclic relabelings of a pose share a "
            "silhouette exactly, so IoU ties between them by construction and "
            "the support term alone decides, which removes the need for any "
            "separate naming step."
        ),
    )
    register_support_tau_px: float = Field(
        default=30.0,
        description=(
            "Truncation distance (px) of the correspondence support. A match "
            "further than this explains nothing under any hypothesis and so "
            "cancels in every comparison, which is why a bounded cost is used "
            "rather than L1 or Huber: a gross mismatch must not be able to move "
            "the comparison between namings. Doubles as the tau of the naming "
            "telemetry written to the debug npz."
        ),
    )
    register_support_alias_only: bool = Field(
        default=False,
        description=(
            "Use the correspondence support only to choose between a pose's "
            "cyclic relabelings, leaving the verification score's scale "
            "untouched. Measured reason: the score's thresholds are not "
            "scale-free. Multiplying U/N into a two-factor product dropped the "
            "median score 0.766 -> 0.669, so 54.6% of frames failed "
            "accept_score instead of 39.8%, the incumbent fast path fell 88.6% "
            "-> 77.4%, the stateless search ran 2.4x as often, and PTi br_p@15 "
            "fell 3.88 pp (corner_fusion_matchncc). The support also saturates "
            "across nearby poses, so it is not evidence about which pose is "
            "right -- only about which naming, where IoU ties by construction."
        ),
    )
    register_keypoint_proposal: bool = Field(
        default=False,
        description=(
            "Offer the keypoint branch's homography to the register method as an "
            "extra pose proposal. A silhouette cannot say which corner is which, "
            "and that corner-labelling alias is the dominant residual error mode; "
            "the keypoint branch carries exactly that information. Costs branch "
            "independence (the SAM-H estimate then depends on LoMa), so offline "
            "fusion-rule replay is no longer exact for such a run"
        ),
    )
    register_alias_shortcut: bool = Field(
        default=False,
        description=(
            "After refining a swept proposal, also score the three cyclic "
            "relabelings of its control points. They share a silhouette with it "
            "and differ only in which corner is which, so they cost three "
            "verifications and no GN solve — while reaching them through the "
            "angle sweep costs a solve each and is not guaranteed, since aliases "
            "tie on the overlap that decides which proposals get refined"
        ),
    )
    register_alias_recheck: bool = Field(
        default=False,
        description=(
            "Also score the three cyclic relabelings of the final winning pose. "
            "register_alias_shortcut compares relabelings of coarsely refined "
            "proposals only, before the full-resolution solve moves the pose, so "
            "a mislabeled pose can reach the output while the verifier's own "
            "score prefers one of its aliases (observed on PTi Seq_00006 frame "
            "30: returned pose 0.770 at 342 px from GT, its +1 relabeling 0.859 "
            "at 20.9 px). Costs three verifications and no GN solve."
        ),
    )
    register_sweep_stop_score: float = Field(
        default=0.0,
        description=(
            "Stop refining further swept proposals once one reaches this verified "
            "score. Safe only together with register_alias_shortcut: stopping at "
            "the first convincing angle is otherwise how a half-turn-wrong pose "
            "gets locked in. 0 disables the early exit"
        ),
    )
    register_sdf_roi_margin: float = Field(
        default=0.0,
        description=(
            "Build the full-resolution signed-distance field on the mask's "
            "bounding box grown by this fraction of its longer side, instead of "
            "on the whole frame. That transform is 47% of a fast-path frame and "
            "is only ever sampled near the mask, so the crop is close to free "
            "accuracy-wise: measured neutral to within 0.2 pp in every regime on "
            "both datasets, at up to 46% less time. 0 keeps the whole frame"
        ),
    )
    register_min_score: float = Field(
        default=0.5,
        description=(
            "Accept the registered H only when its verified score is at least "
            "this. The score is monotone in accuracy (bottom decile 42% P@15, top "
            "two deciles 100%), so it gates far better than the fit residual"
        ),
    )
    topk_lines: int = Field(
        default=4,
        description="Number of Hough lines to detect",
    )
    gradient_step: int = Field(
        default=4,
        description="Contour offset for tangent estimation",
    )
    corner_max_distance_px: float = Field(
        default=80.0,
        description="Maximum corner-intersection distance from mask contour in pixels",
    )
    corner_max_distance_image_fraction: Optional[float] = Field(
        default=None,
        description=(
            "Optional image-relative corner distance threshold. When set, the effective "
            "threshold is max(corner_max_distance_px, fraction * max(height, width))."
        ),
    )
    min_intersection_angle_deg: float = Field(
        default=0.0,
        description=(
            "Minimum angle between adjacent Hough lines for keeping their intersection. "
            "Zero preserves legacy behavior."
        ),
    )
    line_refinement_method: Literal["legacy_algebraic", "orthogonal_tls"] = Field(
        default="legacy_algebraic",
        description=(
            "Line refinement fit. legacy_algebraic preserves previous algebraic SVD fit; "
            "orthogonal_tls uses centered total least squares."
        ),
    )


class NNScorerConfig(BaseModel):
    """Neural network homography quality scorer configuration."""

    enabled: bool = Field(default=False, description="Enable NN-based quality scoring")
    checkpoint_path: str = Field(
        default="", description="Path to trained model checkpoint"
    )
    arch: str = Field(
        default="option_e",
        description="Model architecture (option_e, option_c, or point_baseline)",
    )
    n_points: int = Field(
        default=128, description="Number of boundary points to sample"
    )
    rmse_threshold: float = Field(
        default=float("inf"),
        description="Reject all candidates if best predicted RMSE exceeds this (pixels)",
    )
    trust_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Cascade trust threshold (pixels). When set, enables trust-based cascade "
            "selection: candidates are evaluated in preference order "
            "(flow_H → redetection_flow_H → mask_H) and the first with predicted "
            "RMSE ≤ this value is accepted "
            "immediately without evaluating later options. When None (default), uses "
            "min-RMSE selection over all eagerly evaluated candidates (original behavior)."
        ),
    )


class SymmetryConfig(BaseModel):
    """Cyclic corner shift disambiguation for the quad-based tracker."""

    enabled: bool = Field(
        default=False,
        description="Enable symmetry resolution",
    )
    method: Literal["dino", "matches"] = Field(
        default="dino",
        description=(
            "How the cyclic shift is chosen. 'dino' is the original: warp the "
            "template into a canonical view under each of the four shifts, "
            "score them by DINOv2 feature similarity, and confirm only via a "
            "score/margin pair or a run of consecutive agreeing frames -- "
            "seven parameters and a second network, gated to fire only when "
            "recovering from lost because it is expensive. 'matches' scores the "
            "four shifts by how much of the keypoint matcher's raw "
            "correspondence set each explains (flatsam2.match_naming) and takes "
            "the best. It needs no extra model, is cheap enough to run every "
            "frame -- so only_when_lost, n_correspondences_enough and the four "
            "DINO knobs stop applying -- and is the same measurement the "
            "quad-free tracker's verifier uses for the identical question."
        ),
    )
    match_tau_px: float = Field(
        default=30.0,
        description=(
            "Truncation distance (px) of the correspondence support under "
            "method='matches'. Bounded on purpose: a gross mismatch then "
            "contributes nothing to any of the four shifts and cancels in the "
            "comparison."
        ),
    )
    only_when_lost: bool = Field(
        default=True,
        description="Only resolve symmetry when recovering from lost state",
    )
    n_correspondences_enough: int = Field(
        default=4,
        description="Hough correspondence count considered 'enough'",
    )
    resolution_hw: Tuple[int, int] = Field(
        default=(224, 224),
        description="Canonical view size for DINO feature extraction",
    )
    dino_layer: int = Field(
        default=11,
        description="DINOv2 transformer layer (0-11)",
    )
    abs_thr: float = Field(
        default=0.9,
        description="Minimum absolute DINO score for immediate confirmation",
    )
    fts_thr: float = Field(
        default=0.2,
        description="Minimum margin over second-best for immediate confirmation",
    )
    run_length: int = Field(
        default=5,
        description="Consecutive frames with same best shift for run-based confirmation",
    )


class SamplingCfg(BaseModel):
    """Keypoint sampling configuration for the masked LoMa tracker.

    Args:
        nms_size: NMS window size in pixels; 1 disables NMS.
        subpixel: Apply sub-pixel coordinate refinement.
        subpixel_temp: Temperature for sub-pixel softmax (lower = sharper).
        n_object: Maximum object keypoints selected within mask.
        total_budget: Total keypoint budget (object + background).
            Background fills total_budget - actual_object_count.
    """

    nms_size: int = Field(default=3, description="NMS window size in pixels")
    subpixel: bool = Field(default=True, description="Sub-pixel refinement")
    subpixel_temp: float = Field(
        default=0.5, description="Sub-pixel softmax temperature"
    )
    n_object: int = Field(default=500, description="Max object keypoints within mask")
    total_budget: int = Field(
        default=4096,
        description=(
            "Total keypoint budget (object + background). "
            "Background fills total_budget - actual_object_count."
        ),
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True


class LoMaTrackerConfig(BaseModel):
    """Configuration for the masked LoMa homography tracker (loma_experiment_v2).

    Args:
        model: LoMa model variant name.
        sampling: Keypoint sampling parameters.
        exact_gt_mask_on_init: Use strict GT-mask-only sampling on init frame.
        init_mask_inflate_px: Dilate init-frame mask by this many pixels before
            keypoint sampling.
        current_mask_inflate_px: Dilate current-frame mask by this many pixels
            before keypoint sampling.
        current_mask_sampling: Focus current-frame keypoint sampling on the
            tracked mask. False spends the same budget over the whole frame.
        ransac_method: cv2.findHomography method name.
        ransac_threshold: RANSAC inlier threshold in pixels.
    """

    model: str = Field(
        default="LoMaB",
        description="LoMa model variant (LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR)",
    )
    sampling: SamplingCfg = Field(default_factory=SamplingCfg)
    exact_gt_mask_on_init: bool = Field(
        default=True,
        description=(
            "If True, init-frame keypoint sampling uses exact GT mask only "
            "(forces total_budget=0 for init sampling pass)."
        ),
    )
    init_mask_inflate_px: int = Field(
        default=0,
        ge=0,
        description=(
            "Pixels to dilate init-frame mask before keypoint sampling "
            "(0 disables inflation)."
        ),
    )
    current_mask_inflate_px: int = Field(
        default=0,
        ge=0,
        description=(
            "Pixels to dilate current-frame masks before keypoint sampling "
            "(0 disables inflation)."
        ),
    )
    current_mask_sampling: bool = Field(
        default=True,
        description=(
            "If True, current-frame keypoints are sampled inside the tracked mask "
            "first (background fills the remaining budget). If False, the same "
            "budget is spent over the whole frame -- the no-focusing ablation."
        ),
    )
    ransac_method: str = Field(
        default="RANSAC", description="RANSAC method for cv2.findHomography"
    )
    ransac_threshold: float = Field(
        default=5.0, description="RANSAC inlier threshold in pixels"
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True


class LoMaPathConfig(BaseModel):
    """Online LoMa homography path inside FlatSAMTracker."""

    enabled: bool = Field(
        default=False,
        description="Enable online LoMa homography path",
    )
    model: str = Field(
        default="LoMaR",
        description="LoMa model variant (LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR)",
    )
    sampling: SamplingCfg = Field(default_factory=SamplingCfg)
    exact_gt_mask_on_init: bool = Field(
        default=True,
        description="Use object-only init-frame sampling inside the initial mask",
    )
    init_mask_inflate_px: int = Field(
        default=10,
        ge=0,
        description="Pixels to dilate init-frame mask before LoMa keypoint sampling",
    )
    current_mask_inflate_px: int = Field(
        default=10,
        ge=0,
        description="Pixels to dilate current SAM mask before LoMa keypoint sampling",
    )
    ransac_method: str = Field(
        default="USAC_MAGSAC", description="RANSAC method for cv2.findHomography"
    )
    ransac_threshold: float = Field(
        default=5.0, description="RANSAC inlier threshold in pixels"
    )
    ransac_seed: Optional[int] = Field(
        default=0,
        description=(
            "If set, seed OpenCV's global RNG with this value before every LoMa "
            "RANSAC call so the (otherwise stochastic) USAC/MAGSAC inlier counts "
            "are reproducible across runs. Defaults to 0 (deterministic, no runtime "
            "penalty); set to None to keep OpenCV's default RNG state."
        ),
    )
    inlier_threshold: int = Field(
        default=35, description="Use LoMa when its inlier count reaches this threshold"
    )
    densify_mask: Literal["selected", "candidate_union"] = Field(
        default="selected",
        description=(
            "Mask used to sample LoMa keypoints (densify correspondences). "
            "'selected' uses only the single mask SAM3 picked (default, "
            "byte-identical to prior behavior); 'candidate_union' samples inside "
            "the union of every SAM3 candidate mask, covering more of the object "
            "so more correspondences are produced. Only affects keypoint sampling; "
            "the selected mask still drives mask-H, correction, and the output."
        ),
    )
    decouple_hough_state: bool = Field(
        default=False,
        description=(
            "Run the entire SAM-H branch (mask-H + symmetry resolution) as a "
            "fully independent nested tracker that shares only the SAM3 mask "
            "stream. Its homography/symmetry chain is identical to a standalone "
            "hough_symmetry_only run; the LoMa inlier rule only selects per frame "
            "between the LoMa estimate and that independent SAM-H estimate, never "
            "feeding back. Reproduces the offline ensemble's independent "
            "trajectories."
        ),
    )
    fusion: bool = Field(
        default=False,
        description=(
            "Select algorithm='corner_fusion' instead of 'loma_samh': per-corner "
            "covariance-weighted fusion of the LoMa and SAM-H estimates instead "
            "of the hard inlier-threshold switch. Requires "
            "decouple_hough_state=True (needs the independent SAM-H sub-tracker "
            "for its own corner covariance)."
        ),
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True


class FusionCfg(BaseModel):
    """Corner-level covariance fusion of LoMa + SAM-H (``algorithm='corner_fusion'``).

    Both branches report, per corner, a 2x2 pixel covariance from their own
    fit's normal equations (``flatsam2.corner_covariance``,
    ``flatsam2.mask_align._corner_cov_from_boundary_fit``). Per corner: if the
    two estimates agree within ``disagreement_gate`` (squared Mahalanobis
    distance under the sum of their covariances), fuse by precision-weighted
    average; otherwise trust whichever branch is more confident (smaller
    covariance trace) for that corner alone. See docs/correction_research.md
    section 9-10.
    """

    disagreement_gate: float = Field(
        default=20.0,
        description=(
            "Squared-Mahalanobis gate on the per-corner disagreement. Under a "
            "correctly calibrated model this is chi-square(2) (95% quantile "
            "5.99), but both branches are known to be overconfident in the "
            "field (see docs/correction_research.md section 9), hence the more "
            "conservative default. Needs calibration against real GT before "
            "trusting the exact value."
        ),
    )
    loma_cov_scale: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Multiplicative correction applied to the LoMa branch's per-corner "
            "covariance before fusion (both the disagreement check and the "
            "precision-weighted average / trace tie-break). 1.0 = trust the "
            "raw fit covariance. Both branches are overconfident by different "
            "amounts (calibration check, 20-seq PTi probe, 2026-08-10: LoMa "
            "median chi2_2 stat 44.0 -> implied 31.8x; SAM-H 96.3 -> implied "
            "69.5x — SAM-H is the *more* overconfident of the two, which biases "
            "the uncalibrated trace tie-break toward it regardless of which "
            "branch is actually right). Sample-specific, not a full-dataset "
            "calibration; revisit if the branch or dataset changes."
        ),
    )
    samh_cov_scale: float = Field(
        default=1.0,
        ge=0.0,
        description="Same as loma_cov_scale, for the SAM-H branch.",
    )
    decision_scope: Literal["per_corner", "whole_frame"] = Field(
        default="per_corner",
        description=(
            "Granularity of the fuse-or-arbitrate decision. 'per_corner' takes "
            "it independently for each corner, so the output quad can take two "
            "corners from one branch and two from the other — a pose neither "
            "branch proposed. 'whole_frame' takes it once for all four corners: "
            "fuse every corner (precision-weighted, each corner with its own "
            "weights) or take the whole pose from the branch with the smaller "
            "total corner variance. Measurement behind the simplification: "
            "genuinely complementary frames are 0.66% of PTi and a per-corner "
            "oracle beats a whole-branch oracle by 0.48 pp "
            "(experiments/corner_complementarity.py), while the variance "
            "reduction that fusion actually buys needs no per-corner switching. "
            "Under 'whole_frame' the gate applies to the mean of the four "
            "per-corner squared Mahalanobis distances."
        ),
    )

    keypoint_naming_margin: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "When positive, the SAM-H branch's corners are re-labelled to the "
            "cyclic rotation that sits nearest the keypoint branch's corners, "
            "before the fuse-or-arbitrate decision. A silhouette cannot say "
            "which corner is which, and 89.5% of the shipped tracker's "
            "mis-named PTi frames are frames where the mask branch supplied "
            "the pose (experiments/naming_diagnosis.py); texture can say it, "
            "so the keypoint branch lends its naming without lending its "
            "position. The margin is the incumbent's protection: the "
            "re-labelling is applied only when the nearest rotation is at "
            "least this many times closer to the keypoint quad than the "
            "branch's own naming is, so a near-square outline (all four "
            "rotations similar) or a badly localized keypoint quad leaves the "
            "naming alone. 0.0 disables the step; 1.0 is 'always take the "
            "nearest'."
        ),
    )

    occlusion_max_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "When positive, the mask branch abstains on frames where the SAM "
            "mask is an almost-perfect *subset* of the silhouette the keypoint "
            "branch predicts for this frame: precision >= "
            "occlusion_min_precision and recall <= this value "
            "(flatsam2.occlusion). Under occlusion SAM returns the visible "
            "part only, the registrar fits the template to that sub-region, "
            "and the fit is precise -- small residuals, tight covariance -- so "
            "a confidently *wrong* pose wins the arbitration against an "
            "honestly uncertain keypoint branch (docs/journal.md 2026-08-19, "
            "2026-08-26). That is model-form error, not sampling noise, so no "
            "covariance scale fixes it and the test is a hard switch rather "
            "than a discount. 0.0 disables the screen."
        ),
    )
    occlusion_min_precision: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Containment the mask must reach before the occlusion screen calls "
            "it a partial view. Below it the mask is not a subset of the "
            "expected object at all -- a segmenter that locked onto something "
            "else loses precision, a drifted tracker loses both -- and neither "
            "is a case this screen is entitled to judge. Ignored when "
            "occlusion_max_recall is 0.0."
        ),
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True


class CorrectionCfg(BaseModel):
    """SAM3 self-correction gate for online LoMa tracker.

    Source semantics (oracle-correction experiment grid): the inlier gate
    (``inlier_threshold``) applies to a leg (trigger / content) iff that leg's
    source is ``"loma"``; ``"oracle"`` legs require only a valid GT homography
    on the current frame (plumbed by the runner); ``trigger="periodic"`` needs
    neither. A trigger that fires while the content homography is unavailable
    abstains, recorded as ``trigger_fired=True, corrected=False``.

    Grid cells (all with the mask-divergence trigger unless periodic):

        E1 trigger_source=oracle, content_source=oracle  (no inlier gate)
        E2 trigger_source=oracle, content_source=loma    (gate on content)
        E3 trigger_source=loma,   content_source=oracle  (gate on trigger)
        E5 trigger="periodic",    content_source=oracle  (no gate, no masks)
        E4 loma/loma — the deployed correction, bit-identical to before
    """

    enabled: bool = Field(default=False, description="Enable SAM3 memory correction")
    inlier_threshold: int = Field(
        default=70, description="Correct only when LoMa inliers reach this threshold"
    )
    trigger: Literal["area_margin", "sam_excess_depth", "periodic"] = Field(
        default="area_margin", description="Correction trigger strategy"
    )
    trigger_source: Literal["loma", "oracle"] = Field(
        default="loma",
        description=(
            "Homography inducing the reference mask for the mask-divergence "
            "triggers: warp(init_mask, loma_H) or warp(init_mask, H_gt). Oracle "
            "requires GT plumbed at run time; frames with invalid GT never "
            "trigger. Ignored by trigger='periodic'."
        ),
    )
    content_source: Literal["loma", "oracle"] = Field(
        default="loma",
        description=(
            "Homography whose induced mask is injected into SAM3 memory on fire. "
            "'loma' additionally requires the inlier gate; 'oracle' requires "
            "valid GT this frame (else the fired trigger abstains)."
        ),
    )
    period: int = Field(
        default=30,
        ge=1,
        description=(
            "Fire cadence in frames for trigger='periodic' (absolute grid: fires "
            "when the tracked-frame index is a multiple of this)."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "Evaluate the trigger and record correction telemetry every frame "
            "but never call segmenter.correct — a baseline run with "
            "counterfactual trigger statistics."
        ),
    )
    area_margin: float = Field(
        default=0.25, description="Required SAM mask area excess over correction mask"
    )
    superset_tolerance: float = Field(
        default=0.02,
        description="Max correction-mask fraction allowed outside SAM mask",
    )
    # sam_excess_depth trigger (from sam_overseg_correction study). Fires on the
    # signed-distance depth feature dout_corr: SAM area lying more than
    # excess_depth_px past the correction-mask boundary, as a fraction of the
    # correction-mask area. Robust to few-px boundary jitter, unlike area_margin's
    # raw total-area ratio.
    excess_depth_px: int = Field(
        default=10,
        description="T_feat: boundary tolerance (px) for the SAM-excess depth metric",
    )
    excess_margin: float = Field(
        default=0.10,
        description="Required dout_corr (SAM area > T px outside corr, / corr area)",
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True


class TrackerConfig(BaseModel):
    """Complete tracker configuration.

    Args:
        flow: optical flow configuration
        flow_enabled: enable/disable optical-flow homography paths
        segmenter: segmentation model configuration
        ransac: RANSAC configuration
        filter_by_seg_mask: whether to filter flow correspondences by segmentation mask
        hough: mask-based Hough homography estimator configuration
        symmetry: DINO-based symmetry resolution configuration
        nn_scorer: NN-based homography quality scorer for candidate selection
        fallback_to_mask_H: use ``mask_H`` directly when flow + ``mask_H`` prewarp both fail
        loma: optional online LoMa homography path
        correction: optional SAM3 self-correction config
        candidate_dump: segmentation-only mode that dumps every SAM3 candidate
            mask per frame instead of tracking (no homography)
        algorithm: explicit control-flow discriminator. One of ``flow_hough``,
            ``nn_min_rmse``, ``nn_cascade``, ``loma_samh``, ``corner_fusion``.
            When ``None`` it is derived from the legacy flags (``loma.enabled``,
            ``loma.fusion``, ``nn_scorer.enabled``, ``nn_scorer.trust_threshold``)
            so existing configs keep their behaviour. When set explicitly it must
            agree with that derivation, otherwise a ``ValueError`` is raised —
            the discriminator is a single source of truth, never an override of
            the legacy flags.
    """

    flow: FlowConfig = Field(default_factory=FlowConfig)
    flow_enabled: bool = Field(
        default=True,
        description="Enable optical-flow-based homography estimation paths",
    )
    segmenter: SegmenterConfig = Field(default_factory=SegmenterConfig)
    ransac: RansacConfig = Field(default_factory=RansacConfig)
    filter_by_seg_mask: bool = Field(
        default=True,
        description="Filter flow correspondences by segmentation mask before RANSAC",
    )
    hough: HoughConfig = Field(default_factory=HoughConfig)
    symmetry: SymmetryConfig = Field(default_factory=SymmetryConfig)
    nn_scorer: NNScorerConfig = Field(default_factory=NNScorerConfig)
    fallback_to_mask_H: bool = Field(
        default=True,
        description="Fall back to mask_H directly when flow fails with mask_H prewarp",
    )
    loma: LoMaPathConfig = Field(default_factory=LoMaPathConfig)
    correction: CorrectionCfg = Field(default_factory=CorrectionCfg)
    fusion: FusionCfg = Field(default_factory=FusionCfg)
    candidate_dump: bool = Field(
        default=False,
        description=(
            "Segmentation-only candidate dump mode. When True, run.py bypasses all "
            "homography tracking and instead streams SAM3 over each sequence, saving "
            "every candidate mask per frame (see FlatSAMTracker.track_candidates). "
            "Also drives SAM3's DebugDumpWriter to persist a debug_dump/ directory "
            "for memory_viz (the segmenter runs without SAM3 debug mode). "
            "Pair with flow_enabled=False so no optical-flow model is built."
        ),
    )
    algorithm: Optional[Algorithm] = Field(
        default=None,
        description=(
            "Explicit control-flow discriminator (flow_hough / nn_min_rmse / "
            "nn_cascade / loma_samh). None derives it from the legacy flags; an "
            "explicit value must match that derivation."
        ),
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True

    def _derive_algorithm(self) -> Algorithm:
        """Derive the control-flow discriminator from the legacy flags.

        Mirrors the legacy dispatch precedence the ``tracking/`` algorithm
        modules now implement: LoMa wins first (covers coupled and decoupled
        SAM-H alike), then the NN scorer's cascade vs min-RMSE split, else the
        plain flow/Hough path. The
        ``nn_scorer.enabled`` guard matters: a stray ``trust_threshold`` with the
        scorer disabled still resolves to ``flow_hough``.

        Returns:
            The algorithm name implied by the current flags.
        """
        if self.loma.enabled and self.loma.fusion:
            return "corner_fusion"
        if self.loma.enabled:
            return "loma_samh"
        if self.nn_scorer.enabled and self.nn_scorer.trust_threshold is not None:
            return "nn_cascade"
        if self.nn_scorer.enabled:
            return "nn_min_rmse"
        return "flow_hough"

    @model_validator(mode="after")
    def _resolve_algorithm(self) -> "TrackerConfig":
        """Fill or cross-check ``algorithm`` against the legacy-flag derivation.

        Gotcha for config-file authors: this validator runs on construction
        (``TrackerConfig(...)``) but ``model_copy(update=...)`` — how every
        ``configs/tracker/*.py`` composes a base config — does NOT re-run
        validators. Flipping a flag that changes the derivation (e.g.
        ``loma.fusion``) through ``model_copy`` alone leaves ``algorithm``
        stuck at whatever the base config already resolved to; pass
        ``"algorithm"`` explicitly in the same ``update=`` dict.

        Returns:
            ``self`` with ``algorithm`` resolved to a concrete value.

        Raises:
            ValueError: if ``algorithm`` was set explicitly to a value that
                disagrees with the derivation from the legacy flags.
        """
        derived = self._derive_algorithm()
        if self.algorithm is None:
            self.algorithm = derived
        elif self.algorithm != derived:
            raise ValueError(
                f"algorithm={self.algorithm!r} disagrees with the legacy flags "
                f"(loma.enabled={self.loma.enabled}, loma.fusion={self.loma.fusion}, "
                f"nn_scorer.enabled={self.nn_scorer.enabled}, "
                f"nn_scorer.trust_threshold={self.nn_scorer.trust_threshold}), "
                f"which imply algorithm={derived!r}. Fix the config: the "
                f"discriminator must match the flags."
            )
        if self.algorithm == "corner_fusion" and not self.loma.decouple_hough_state:
            raise ValueError(
                "algorithm='corner_fusion' requires loma.decouple_hough_state=True "
                "(needs the independent SAM-H sub-tracker for its own corner "
                "covariance)."
            )
        return self


def load_config_from_file(
    config_path: Path,
) -> Union["TrackerConfig", "LoMaTrackerConfig"]:
    """Load a TrackerConfig from a Python config file.

    The config file should define a `config` variable containing a TrackerConfig instance,
    or define the config inline and it will be auto-detected.

    Args:
        config_path: path to Python config file

    Returns:
        TrackerConfig instance

    Raises:
        ValueError: if config file doesn't define a valid config
        FileNotFoundError: if config_path doesn't exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the Python module
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _config_types = (TrackerConfig, LoMaTrackerConfig)

    # Try to extract config
    if hasattr(module, "config"):
        config = module.config
        if isinstance(config, _config_types):
            return config
        else:
            raise ValueError(
                f"Config file {config_path} has 'config' variable but it's not a "
                f"TrackerConfig or LoMaTrackerConfig instance"
            )

    # Try to find a TrackerConfig or LoMaTrackerConfig instance in the module
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, _config_types):
            return obj

    raise ValueError(
        f"Config file {config_path} must define a 'config' variable "
        f"or contain a TrackerConfig or LoMaTrackerConfig instance"
    )


def compute_output_dir(
    base_output_dir: Path, dataset: str, config_path: Optional[Path] = None
) -> Path:
    """Compute structured output directory path.

    Creates a directory structure: ``base_output_dir / config_name / dataset_label``
    where ``dataset_label`` comes from :data:`~flatsam2.registry.DATASETS`.

    Args:
        base_output_dir: Top-level output directory.
        dataset: Any accepted dataset CLI key (e.g. ``'pot'``, ``'planartrack'``,
            ``'planartrack-initfix'``, ``'pti'``).
        config_path: Optional path to config file (uses ``'default'`` if ``None``).

    Returns:
        Structured output directory path.

    Raises:
        ValueError: If ``dataset`` is not a recognised key.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Expected one of: {sorted(DATASETS)!r}."
        )
    config_name = Path(config_path).stem if config_path is not None else "default"
    dataset_dir = DATASETS[dataset].label
    return base_output_dir / config_name / dataset_dir
