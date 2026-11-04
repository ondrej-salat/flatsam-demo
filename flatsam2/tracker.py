import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
from einops import rearrange
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from sam3 import SAM3StreamingTracker

from flatsam2 import match_naming
from flatsam2.config import LoMaPathConfig, TrackerConfig
from flatsam2.datasets import warp_corners
from flatsam2.geometry import warp_points
from flatsam2.metrics import alignment_error
from flatsam2.profiling import profiler

# Shared verbs and types live in tracking.common so the per-algorithm modules
# can import them without a circular dependency on this module. Re-exported here
# (constants, dataclasses, pure helpers) to keep `from flatsam2.tracker import
# TrackResult / _DEBUG_CANDIDATES / ...` working for existing callers.
from flatsam2.tracking import corner_fusion, flow_hough, loma_samh, nn_cascade, nn_min_rmse
from flatsam2.tracking.common import (
    _CANDIDATE_FLOW,
    _CANDIDATE_LAST_H,
    _CANDIDATE_LOMA,
    _CANDIDATE_MASK_H,
    _CANDIDATE_REDETECTION_FLOW,
    _CANDIDATE_SAMH,
    _DEBUG_CANDIDATES,  # noqa: F401  re-exported for `from flatsam2.tracker import _DEBUG_CANDIDATES`
    _MASK_H_SOURCE_ALIGN,
    _MASK_H_SOURCE_HOUGH,
    _MASK_H_SOURCE_REGISTER,
    CandidateScore,
    CorrectionTelemetry,
    MaskHTelemetry,
    ScoredCandidate,
    TrackerDebugInfo,
    TrackerState,
    TrackResult,
    _correction_telemetry,
    _inflate_mask,
    _is_valid_homography,
    _should_correct,
    _union_candidate_masks,
)

logger = logging.getLogger(__name__)


# Flat dispatch table: cfg.algorithm -> the linear free function that runs it.
# A greppable literal, not a registry built by decorators — the single permitted
# dispatch point (looked up once per frame in step_with_mask).
_ALGORITHMS = {
    "flow_hough": flow_hough.track,
    "nn_min_rmse": nn_min_rmse.track,
    "nn_cascade": nn_cascade.track,
    "loma_samh": loma_samh.track,
    "corner_fusion": corner_fusion.track,
}


class FlatSAMTracker:
    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        fail_hard: bool = False,
        *,
        segmenter: Optional[object] = None,
        dino: Optional[object] = None,
    ):
        """Create a new FlatSAMTracker.

        Args:
            config: optional TrackerConfig instance. If None, uses defaults.
            fail_hard: if True, any exception inside track() is re-raised
                immediately instead of being suppressed and falling back to the
                last known good homography.  Useful for debugging.
            segmenter: optional pre-built SAM3 segmenter to share instead of
                creating a new one. Used by the decoupled SAM-H sub-tracker so
                the LoMa and SAM-H branches consume one shared SAM3 mask stream.
            dino: optional pre-built DINO feature extractor to share for
                symmetry resolution. Shared with the SAM-H sub-tracker to avoid
                loading the model twice (it is a stateless feature extractor).
        """
        self.cfg = config or TrackerConfig()
        self.fail_hard = fail_hard

        # Initialize SAM3 segmenter (or reuse an injected, shared one).
        if segmenter is not None:
            self.segmenter = segmenter
        else:
            self.segmenter = SAM3StreamingTracker()

        # Initialize ptlflow optical flow model
        self.flower = None
        if self.cfg.flow_enabled:
            self.flower = ptlflow.get_model(self.cfg.flow.model, self.cfg.flow.weights)
            self.flower.eval()
            self.flower.requires_grad_(False)
            self.flower.cuda()

        self.flow_io_adapter = None  # Will be initialized in init()
        self.state = None

        # Optional online LoMa homography path (loaded only when configured).
        self._loma = None
        self._loma_feats0 = None
        self._loma_last_inliers = 0
        self._loma_last_selected = _CANDIDATE_LAST_H
        self._loma_last_corrected = False
        self._loma_last_corr_mask: Optional[np.ndarray] = None
        # Per-corner (4, 2, 2) pixel covariance of the last loma_H / mask_H fit;
        # None when unmeasurable (see flatsam2.corner_covariance).
        self._loma_last_corner_cov: Optional[np.ndarray] = None
        self._maskh_last_corner_cov: Optional[np.ndarray] = None
        # Per-frame GT homography (init -> current) supplied by the runner for
        # oracle correction experiments; None when GT is invalid or not plumbed.
        self._oracle_H: Optional[np.ndarray] = None
        # Per-frame correction observables (survives step_with_mask crashes).
        self._corr_telemetry = CorrectionTelemetry()
        # Per-frame mask-H estimator observables (align vs Hough acceptance).
        self._maskh_telemetry = MaskHTelemetry()
        # Template contour for the align mask-H method; constant per sequence.
        self._align_boundary: Optional[np.ndarray] = None
        self._register_template = None
        # Per-frame extra pose proposals for the register mask-H method, set by
        # the caller (corner_fusion hands over the keypoint branch's H). Cleared
        # after each use so a stale proposal never leaks into the next frame.
        self._register_extra_H: tuple = ()
        self._loma_last_matches: Optional[tuple] = None
        # Raw correspondences handed *in* to this tracker's registrar. On the
        # decoupled path the SAM-H sub-tracker is a separate instance, so the
        # owner sets this on it the same way it sets _register_extra_H.
        self._register_matches: Optional[tuple] = None
        # Per-frame override for LoMa keypoint sampling (candidate-union densify).
        # None keeps the selected SAM3 mask as the sampling region.
        self._loma_densify_mask: Optional[np.ndarray] = None
        if self.cfg.loma.enabled:
            from flatsam2.loma_models import build_loma_model
            from flatsam2.loma_tracker import LoMaTracker

            self._loma = LoMaTracker(build_loma_model(self.cfg.loma.model))

        # DINO extractor for symmetry resolution (loaded lazily, or shared).
        # method='matches' scores the shifts against the keypoint
        # correspondences instead, so the second network is never loaded.
        self._dino = None
        if self.cfg.symmetry.enabled and self.cfg.symmetry.method == "dino":
            if dino is not None:
                self._dino = dino
            else:
                from flatsam2.hough_homography import DINOFeatureExtractor

                self._dino = DINOFeatureExtractor()

        # NN homography quality scorer (loaded lazily).
        self._nn_scorer = None
        if self.cfg.nn_scorer.enabled:
            from flatsam2.nn.inference import NNHomographyScorer

            self._nn_scorer = NNHomographyScorer(
                str(Path(self.cfg.nn_scorer.checkpoint_path).expanduser()),
                arch=self.cfg.nn_scorer.arch,
                n_points=self.cfg.nn_scorer.n_points,
            )

        # Independent SAM-H sub-tracker for the decoupled LoMa path. It runs the
        # full mask-H + symmetry chain exactly like a standalone
        # ``hough_symmetry_only`` tracker, sharing only the SAM3 segmenter (and
        # the stateless DINO extractor). The LoMa inlier rule then selects per
        # frame between the LoMa estimate and this sub-tracker's estimate without
        # ever feeding back, reproducing the offline ensemble.
        self._samh_tracker = None
        if self.cfg.loma.enabled and self.cfg.loma.decouple_hough_state:
            samh_cfg = TrackerConfig(
                flow_enabled=False,
                segmenter=self.cfg.segmenter,
                hough=self.cfg.hough,
                symmetry=self.cfg.symmetry,
                fallback_to_mask_H=self.cfg.fallback_to_mask_H,
                loma=LoMaPathConfig(enabled=False),
            )
            self._samh_tracker = FlatSAMTracker(
                samh_cfg,
                fail_hard=self.fail_hard,
                segmenter=self.segmenter,
                dino=self._dino,
            )

    def init(
        self,
        init_frame: np.ndarray,
        init_mask: Optional[np.ndarray] = None,
        init_corners: Optional[np.ndarray] = None,
        init_segmenter: bool = True,
    ):
        """Initialize tracker with the first frame and an initial region.

        You must provide either init_mask or init_corners (or both):
        - For planar tracking benchmarks: provide init_corners (4 corners of a quadrilateral)
        - For general shape tracking: provide init_mask (arbitrary segmentation mask)
        - Both can be provided: corners will be stored for reference, mask used for tracking

        Args:
            init_frame: first frame (H, W, C) opencv uint8 BGR image
            init_mask: optional binary mask of the object (H, W) uint8 numpy array
            init_corners: optional corner coordinates (2, 4) - [[x1,x2,x3,x4], [y1,y2,y3,y4]]
            init_segmenter: whether to (re)initialise the SAM3 segmenter. Set
                False for a sub-tracker that shares an already-initialised
                segmenter so the SAM3 state is not advanced twice.

        Raises:
            ValueError: if neither init_mask nor init_corners is provided
        """
        if init_mask is None and init_corners is None:
            raise ValueError("Must provide either init_mask or init_corners")

        # If corners provided but no mask, generate mask from corners
        if init_mask is None and init_corners is not None:
            init_mask = self._corners_to_mask(init_corners, init_frame.shape[:2])

        # Initialize state
        self.state = TrackerState()
        self.state.init_frame = init_frame.copy()
        self.state.init_mask = init_mask.copy()
        self.state.init_corners = (
            init_corners.copy() if init_corners is not None else None
        )

        # Initialize with identity homography
        self.state.last_homography = np.eye(3, dtype=np.float64)

        # Fresh per-run correction/oracle state.
        self._oracle_H = None
        self._corr_telemetry = CorrectionTelemetry()
        self._maskh_telemetry = MaskHTelemetry()
        self._align_boundary = None
        self._register_template = None
        self._register_extra_H = ()
        self._loma_last_matches = None
        self._register_matches = None

        # Initialize SAM3 segmenter with the first frame (skipped when sharing an
        # already-initialised segmenter, e.g. the decoupled SAM-H sub-tracker).
        if init_segmenter:
            self.segmenter.init(init_frame, init_mask)

        # Initialize flow IO adapter with frame dimensions
        if self.flower is not None:
            self.flow_io_adapter = IOAdapter(
                self.flower,
                init_frame.shape[:2],
                cuda=True,
            )

        # Initialize symmetry state if enabled.
        if self.cfg.symmetry.enabled:
            from flatsam2.hough_homography import SymmetryState

            self.state.symmetry = SymmetryState()

        # Build reference contour for NN scorer if enabled.
        if self._nn_scorer is not None:
            from flatsam2.homography_quality import ReferenceContour

            self._nn_scorer.init_reference(ReferenceContour.build(init_mask))

        if (
            self.cfg.loma.enabled
            and self._loma is not None
        ):
            sampling_mask = _inflate_mask(init_mask, self.cfg.loma.init_mask_inflate_px)
            if self.cfg.loma.exact_gt_mask_on_init:
                sampling_cfg = self.cfg.loma.sampling.model_copy(
                    update={"total_budget": 0}
                )
            else:
                sampling_cfg = self.cfg.loma.sampling
            self._loma_feats0 = self._loma.get_keypoints(
                init_frame,
                mask=sampling_mask,
                cfg=sampling_cfg,
            )

        # Initialise the decoupled SAM-H sub-tracker on its own state, reusing the
        # parent's already-initialised (shared) segmenter and the resolved init
        # mask so its mask/symmetry chain is identical to a standalone run.
        if self._samh_tracker is not None:
            self._samh_tracker.init(
                init_frame,
                init_mask=init_mask,
                init_corners=init_corners,
                init_segmenter=False,
            )

    @staticmethod
    def _corners_to_mask(corners: np.ndarray, image_shape: tuple) -> np.ndarray:
        """Convert corner coordinates to a binary mask.

        Args:
            corners: Corner coordinates (2, 4) - [[x1,x2,x3,x4], [y1,y2,y3,y4]]
            image_shape: (height, width) of the output mask

        Returns:
            Binary mask (height, width) as uint8
        """
        # Reshape corners to (4, 1, 2) for cv2.fillPoly - (num_points, 1, xy)
        corners_poly = rearrange(corners, "xy corners -> corners 1 xy").astype(np.int32)

        # Create empty mask and fill polygon
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [corners_poly], 255)

        return mask

    def track(
        self,
        frame: np.ndarray,
        oracle_H: Optional[np.ndarray] = None,
    ) -> TrackResult:
        """Update tracker with a new frame and return estimation.

        On any failure (e.g. singular matrix, too few inliers) the method falls
        back to the last known good homography.  Set ``fail_hard=True`` on the
        tracker to re-raise exceptions instead, which is useful for debugging.

        Args:
            frame: (H, W, C) opencv uint8 BGR image
            oracle_H: optional GT homography (3, 3) init frame -> current frame,
                used by oracle correction experiments (``cfg.correction``
                sources). None when GT is invalid/absent for this frame.

        Returns:
            TrackResult with the estimated homography and segmentation mask.
        """
        self._oracle_H = oracle_H
        # Always run the segmenter first so its internal state stays up to date
        # and we have a mask regardless of whether homography estimation succeeds.
        with profiler.section("sam3"):
            if self.cfg.loma.densify_mask == "candidate_union":
                # Pull every candidate mask so LoMa can sample keypoints inside
                # their union (denser correspondences). The selected mask still
                # drives everything else, so the SAM3 stream advances identically.
                seg_mask, candidates = self.segmenter.track(frame, return_all_masks=True)
                self._loma_densify_mask = _union_candidate_masks(candidates, seg_mask)
            else:
                seg_mask = self.segmenter.track(frame)
                self._loma_densify_mask = None
        return self.step_with_mask(frame, seg_mask)

    def track_candidates(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Run SAM3 on *frame* and return every candidate mask (no homography).

        Segmentation-only path used by the candidate-mask dump (``candidate_dump``
        config). It advances the SAM3 stream exactly like :meth:`track` but skips
        all homography estimation, returning the raw multi-candidate output.

        Args:
            frame: (H, W, C) opencv uint8 BGR image.

        Returns:
            Tuple ``(mask, candidates)`` where ``mask`` is the (H, W) bool mask
            SAM3 selected, and ``candidates`` is the dict from
            ``SAM3StreamingTracker.track(..., return_all_masks=True)``:
            ``{"masks": (M, H, W) bool, "logits": (M, H, W) float32,
            "ious": (M,) float32}``. ``M`` is 3 under multimask output, 1
            otherwise, and 0 when SAM3 reports the object absent.
        """
        with profiler.section("sam3"):
            return self.segmenter.track(frame, return_all_masks=True)

    def step_with_mask(self, frame: np.ndarray, seg_mask: np.ndarray) -> TrackResult:
        """Run homography estimation for a frame whose SAM3 mask is already known.

        Identical to :meth:`track` minus the segmenter step, so callers that own
        the SAM3 mask stream (e.g. the decoupled SAM-H sub-tracker, which shares
        the parent's segmenter) can drive estimation without advancing SAM3
        state twice.

        Args:
            frame: (H, W, C) opencv uint8 BGR image.
            seg_mask: segmentation mask (H, W) already obtained for this frame.

        Returns:
            TrackResult with the estimated homography and the given mask.
        """
        self.state.frame_idx += 1
        try:
            return _ALGORITHMS[self.cfg.algorithm](self, frame, seg_mask)
        except Exception:
            if self.fail_hard:
                raise
            logger.warning(
                "track() raised an exception; falling back to last homography.\n%s",
                traceback.format_exc(),
            )
            # Preserve whatever telemetry the frame set before raising — notably a
            # correction that already mutated SAM3 memory — so the event is not
            # silently dropped from the debug sidecar (it would otherwise read as
            # corrected=False while the masks downstream clearly changed).
            return TrackResult(
                homography=self.state.last_homography,
                mask=seg_mask,
                debug=TrackerDebugInfo(
                    selected=self._loma_last_selected,
                    corrected=self._loma_last_corrected,
                    n_inliers=self._loma_last_inliers,
                    correction=self._corr_telemetry,
                ),
                correction_mask=self._loma_last_corr_mask,
            )

    # ------------------------------------------------------------------
    # Flow-based homography estimation (extracted for reuse with different prewarps)
    # ------------------------------------------------------------------

    def _estimate_flow_H(
        self,
        frame: np.ndarray,
        seg_mask: np.ndarray,
        prewarp_H: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate homography via prewarp + optical flow + RANSAC.

        Args:
            frame: Current frame (H, W, C) BGR uint8.
            seg_mask: Segmentation mask (H, W) for filtering correspondences.
            prewarp_H: (3, 3) homography used as prewarp (init -> current).

        Returns:
            Valid homography (3, 3) init -> current, or None on failure.
        """
        if self.flower is None or self.flow_io_adapter is None:
            return None

        H_prewarp = np.linalg.inv(prewarp_H)
        h, w = self.state.init_frame.shape[:2]
        frame_prewarped = cv2.warpPerspective(frame, H_prewarp, (w, h))

        with profiler.section("flow.infer"), torch.no_grad():
            predictions = self.flower(
                self.flow_io_adapter.prepare_inputs(
                    [self.state.init_frame, frame_prewarped]
                )
            )
            flow = predictions["flows"][0, 0]  # (2, H, W)

        flow_np = rearrange(flow.cpu().numpy(), "xy h w -> h w xy")

        # Build correspondences from flow over the init mask.
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        mask_points = self.state.init_mask > 0
        src_pts = np.stack([x_coords[mask_points], y_coords[mask_points]], axis=1)
        dst_pts = src_pts + flow_np[mask_points]

        # Optional: filter by segmentation mask.
        if self.cfg.filter_by_seg_mask:
            dst_pts_current = warp_points(prewarp_H, dst_pts)
            dst_x = np.round(dst_pts_current[:, 0]).astype(int)
            dst_y = np.round(dst_pts_current[:, 1]).astype(int)
            in_bounds = (dst_x >= 0) & (dst_x < w) & (dst_y >= 0) & (dst_y < h)
            in_seg = np.zeros(len(dst_pts), dtype=bool)
            in_seg[in_bounds] = seg_mask[dst_y[in_bounds], dst_x[in_bounds]] > 0
            src_pts = src_pts[in_seg]
            dst_pts = dst_pts[in_seg]

        if len(src_pts) < 4:
            return None

        # Seed OpenCV's global RNG so the flow RANSAC is reproducible (default 0;
        # set ransac_seed=None for legacy unseeded behaviour).
        if self.cfg.ransac.ransac_seed is not None:
            cv2.setRNGSeed(self.cfg.ransac.ransac_seed)

        with profiler.section("flow.ransac"):
            H_delta, _inliers = cv2.findHomography(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.ransac.inl_threshold,
            )

        if H_delta is None:
            return None

        H_total = prewarp_H @ H_delta
        if _is_valid_homography(H_total):
            return H_total

        return None

    def _compute_mask_h(
        self,
        seg_mask: np.ndarray,
        hough_enabled: bool,
        frame: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], int]:
        """Compute mask-H and correspondence count from Hough pipeline.

        Args:
            seg_mask: Segmentation mask (H, W) for current frame.
            hough_enabled: Whether Hough path is enabled for this frame.
            frame: Current frame (H, W, C); used by ``method='register'`` for the
                texture check that separates a pose from its silhouette aliases.

        Returns:
            Tuple ``(mask_H, n_mask_corr)`` where ``mask_H`` is ``None`` when
            unavailable.
        """
        self._maskh_telemetry = MaskHTelemetry()
        self._maskh_last_corner_cov = None

        if not hough_enabled:
            return None, 0

        if self.cfg.hough.method == "register":
            mask_H, n_ok = self._register_mask_h(seg_mask, frame)
            if mask_H is not None:
                return mask_H, n_ok
            if not self.cfg.hough.align_fallback_hough:
                return None, 0

        # P3' direct alignment replaces the Hough pipeline when configured; the
        # last-frame homography init is well inside the measured convergence
        # basin (~20 px / 20 deg) for tracking-scale inter-frame motion.
        if self.cfg.hough.method == "align":
            from flatsam2.mask_align import align_mask_to_homography, template_boundary

            # The template is fixed for the whole sequence: contour it once.
            if self._align_boundary is None and self.state.init_mask is not None:
                self._align_boundary = template_boundary(
                    self.state.init_mask > 0, self.cfg.hough.align_n_boundary_points
                )
            with profiler.section("mask_H.align"):
                align_H, n_ok, residual, n_iters, corner_cov = align_mask_to_homography(
                    seg_mask,
                    self.state.init_corners,
                    self.state.last_homography,
                    template_mask=self.state.init_mask,
                    boundary=self._align_boundary,
                    n_boundary_points=self.cfg.hough.align_n_boundary_points,
                    max_iters=self.cfg.hough.align_max_iters,
                    huber_delta_px=self.cfg.hough.align_huber_delta_px,
                    step_tol_px=self.cfg.hough.align_step_tol_px,
                    redetect_above_residual_px=(
                        self.cfg.hough.align_residual_ok_px
                        if self.cfg.hough.align_redetect_moments
                        else None
                    ),
                )
            accepted = (
                align_H is not None
                and residual <= self.cfg.hough.align_residual_ok_px
                and _is_valid_homography(align_H)
            )
            self._maskh_telemetry.align_attempted = True
            self._maskh_telemetry.align_accepted = bool(accepted)
            self._maskh_telemetry.align_residual_px = (
                float(residual) if align_H is not None else float("nan")
            )
            self._maskh_telemetry.align_n_ok = int(n_ok)
            self._maskh_telemetry.align_iters = int(n_iters)
            if accepted:
                self._maskh_telemetry.source = _MASK_H_SOURCE_ALIGN
                self._maskh_last_corner_cov = corner_cov
                return align_H, n_ok
            if not self.cfg.hough.align_fallback_hough:
                return None, 0

        from flatsam2.hough_homography import mask_to_homography

        # Seed the Hough corner association with the previous prediction.
        with profiler.section("mask_H"):
            mask_H, n_mask_corr = mask_to_homography(
                seg_mask,
                self.state.init_corners,
                self.state.last_homography,
                gradient_step=self.cfg.hough.gradient_step,
                topk_lines=self.cfg.hough.topk_lines,
                corner_max_distance_px=self.cfg.hough.corner_max_distance_px,
                corner_max_distance_image_fraction=(
                    self.cfg.hough.corner_max_distance_image_fraction
                ),
                min_intersection_angle_deg=self.cfg.hough.min_intersection_angle_deg,
                line_refinement_method=self.cfg.hough.line_refinement_method,
            )
        if mask_H is not None:
            self._maskh_telemetry.source = _MASK_H_SOURCE_HOUGH
        return mask_H, n_mask_corr

    def _register_mask_h(
        self,
        seg_mask: np.ndarray,
        frame: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], int]:
        """Quad-free propose/refine/verify registration of the template mask.

        Unlike ``method='align'``, this does not depend on the last homography
        being close: when the incumbent pose fails verification the second-moment
        search re-proposes poses from the current mask alone. Measured on the
        offline benchmark (``experiments/maskh_bench.py``, held-out split), that
        is worth +8 to +15 points of P@15 over the Hough quad once the estimate
        has been lost, at a cost of ~1.5 points while tracking is healthy.

        Args:
            seg_mask: Segmentation mask (H, W) for the current frame.
            frame: Current frame (H, W, C), or None to run mask-only.

        Returns:
            ``(mask_H, n_ok)``; ``mask_H`` is None when registration failed or
            scored below ``hough.register_min_score``.
        """
        from flatsam2.mask_register import (
            RegisterConfig,
            RegistrationTemplate,
            register_mask,
        )

        cfg = RegisterConfig(
            n_angles=self.cfg.hough.register_n_angles,
            accept_score=self.cfg.hough.register_accept_score,
            incumbent_margin=self.cfg.hough.register_incumbent_margin,
            overlap_power=self.cfg.hough.register_overlap_power,
            appearance_power=self.cfg.hough.register_appearance_power,
            support_power=self.cfg.hough.register_support_power,
            support_tau_px=self.cfg.hough.register_support_tau_px,
            support_alias_only=self.cfg.hough.register_support_alias_only,
            alias_shortcut=self.cfg.hough.register_alias_shortcut,
            alias_recheck=self.cfg.hough.register_alias_recheck,
            sweep_stop_score=self.cfg.hough.register_sweep_stop_score,
            sdf_roi_margin=self.cfg.hough.register_sdf_roi_margin,
            n_boundary_points=self.cfg.hough.align_n_boundary_points,
            max_iters=self.cfg.hough.align_max_iters,
            huber_delta_px=self.cfg.hough.align_huber_delta_px,
            step_tol_px=self.cfg.hough.align_step_tol_px,
        )
        # The template is fixed for the whole sequence: build its context once.
        if self._register_template is None:
            self._register_template = RegistrationTemplate.build(
                self.state.init_corners,
                self.state.init_mask,
                seg_mask.shape[:2],
                cfg,
                init_frame=self.state.init_frame,
            )
        if self._register_template is None:
            return None, 0

        extra_H = self._register_extra_H
        self._register_extra_H = ()
        matches = self._register_matches
        self._register_matches = None
        with profiler.section("mask_H.register"):
            result = register_mask(
                self._register_template,
                seg_mask,
                last_H=self.state.last_homography,
                extra_H=extra_H,
                cfg=cfg,
                frame=frame,
                matches=matches,
            )

        accepted = (
            result.H is not None
            and result.score >= self.cfg.hough.register_min_score
            and _is_valid_homography(result.H)
        )
        self._maskh_telemetry.align_attempted = True
        self._maskh_telemetry.align_accepted = bool(accepted)
        self._maskh_telemetry.align_residual_px = float(result.residual_px)
        self._maskh_telemetry.align_n_ok = int(result.n_ok)
        self._maskh_telemetry.align_iters = int(result.n_solves)
        self._maskh_telemetry.register_score = float(result.score)
        self._maskh_telemetry.register_source = result.source
        if not accepted:
            return None, 0
        self._maskh_telemetry.source = _MASK_H_SOURCE_REGISTER
        self._maskh_last_corner_cov = result.corner_cov
        return result.H, result.n_ok

    def _estimate_loma_H(
        self,
        frame: np.ndarray,
        seg_mask: np.ndarray,
    ) -> tuple[Optional[np.ndarray], int]:
        """Estimate current homography with LoMa correspondences.

        Args:
            frame: Current frame `(H, W, C)` BGR image.
            seg_mask: Current SAM mask `(H, W)`.

        Returns:
            Tuple `(H, n_inliers)`, where `H` is None on failure. Also stashes
            the per-corner covariance of the fit on ``self._loma_last_corner_cov``
            (None on failure or when the fit is too degenerate to measure).
        """
        self._loma_last_corner_cov = None
        if self._loma is None or self._loma_feats0 is None:
            return None, 0

        # Densify from the candidate union when configured; otherwise the single
        # selected SAM3 mask. The union covers more of the object, yielding more
        # keypoints (hence correspondences) inside it.
        densify_mask = (
            self._loma_densify_mask
            if self._loma_densify_mask is not None
            else seg_mask
        )
        sampling_mask = _inflate_mask(densify_mask, self.cfg.loma.current_mask_inflate_px)
        with profiler.section("loma.keypoints"):
            feats_i = self._loma.get_keypoints(
                frame,
                mask=sampling_mask,
                cfg=self.cfg.loma.sampling,
            )
        with profiler.section("loma.match"):
            pts_a, pts_b, _idx_a, _ = self._loma.match_with_indices(
                self._loma_feats0,
                feats_i,
            )
        # Kept *before* RANSAC: these are the naming evidence
        # (flatsam2.match_naming). Inliers are selected under a homography that
        # already carries one of the four namings, so they cannot arbitrate
        # between namings without circularity.
        self._loma_last_matches = (pts_a, pts_b)
        if len(pts_a) < 4:
            return None, 0

        # Seed OpenCV's global RNG so the stochastic USAC/MAGSAC inlier counts
        # (which drive the per-frame LoMa-vs-SAM-H selection) are reproducible.
        if self.cfg.loma.ransac_seed is not None:
            cv2.setRNGSeed(self.cfg.loma.ransac_seed)

        with profiler.section("loma.ransac"):
            H, inliers = cv2.findHomography(
                pts_a,
                pts_b,
                getattr(cv2, self.cfg.loma.ransac_method),
                self.cfg.loma.ransac_threshold,
            )
        n_inliers = int(inliers.sum()) if inliers is not None and H is not None else 0
        if not _is_valid_homography(H):
            return None, 0
        if inliers is not None and self.state.init_corners is not None:
            from flatsam2.corner_covariance import homography_corner_covs

            inlier_mask = inliers.ravel().astype(bool)
            corners = warp_corners(H, self.state.init_corners)
            self._loma_last_corner_cov = homography_corner_covs(
                pts_a[inlier_mask], pts_b[inlier_mask], self.state.init_corners, corners
            )
        return H, n_inliers

    def _decoupled_select(
        self,
        loma_H: Optional[np.ndarray],
        n_inliers: int,
        samh_H: np.ndarray,
    ) -> np.ndarray:
        """Pick LoMa vs the independent SAM-H estimate by the inlier rule.

        Mirrors the offline ensemble exactly: use LoMa when its inlier count
        reaches ``inlier_threshold``, otherwise use the SAM-H sub-tracker's
        estimate (which always has a result — it keeps its last homography on
        failure). Neither choice feeds back into the other chain.

        Args:
            loma_H: LoMa homography for this frame, or None on failure.
            n_inliers: LoMa RANSAC inlier count for this frame.
            samh_H: Independent SAM-H sub-tracker homography for this frame.

        Returns:
            The selected homography (3, 3).
        """
        if loma_H is not None and n_inliers >= self.cfg.loma.inlier_threshold:
            self._loma_last_selected = _CANDIDATE_LOMA
            return loma_H
        self._loma_last_selected = _CANDIDATE_SAMH
        return samh_H

    def _choose_loma_or_mask(
        self,
        loma_H: Optional[np.ndarray],
        n_inliers: int,
        mask_H: Optional[np.ndarray],
        n_mask_corr: int,
    ) -> Optional[np.ndarray]:
        """Choose LoMa or mask-H by offline ensemble inlier rule."""
        if loma_H is not None and n_inliers >= self.cfg.loma.inlier_threshold:
            self._loma_last_selected = _CANDIDATE_LOMA
            return loma_H
        if mask_H is not None and n_mask_corr > 0:
            self._loma_last_selected = _CANDIDATE_MASK_H
            return mask_H
        if loma_H is not None:
            self._loma_last_selected = _CANDIDATE_LOMA
            return loma_H
        self._loma_last_selected = _CANDIDATE_LAST_H
        return None

    def _maybe_correct_sam3(
        self,
        seg_mask: np.ndarray,
        loma_H: Optional[np.ndarray],
        n_inliers: int,
    ) -> bool:
        """Apply SAM3 memory correction per ``cfg.correction``; record telemetry.

        Source semantics (see :class:`CorrectionCfg`): the inlier gate applies to
        a leg (trigger / content) iff that leg's source is ``"loma"``; oracle
        legs require only a valid ``self._oracle_H`` this frame; ``"periodic"``
        needs neither. A fired trigger with unavailable content abstains
        (``trigger_fired=True, corrected=False``). ``dry_run`` records everything
        but never touches SAM3 memory.

        Args:
            seg_mask: Current SAM mask `(H, W)`.
            loma_H: LoMa homography for this frame, or None on failure.
            n_inliers: LoMa RANSAC inlier count for this frame.

        Returns:
            True when a correction was injected into SAM3 memory.
        """
        cfg = self.cfg.correction
        if not cfg.enabled:
            return False
        H_img, W_img = seg_mask.shape[:2]

        # One induced mask per available source homography.
        loma_mask = None
        if loma_H is not None:
            loma_mask = (
                cv2.warpPerspective(self.state.init_mask, loma_H, (W_img, H_img)) > 0
            )
        gt_mask = None
        if self._oracle_H is not None:
            gt_mask = (
                cv2.warpPerspective(self.state.init_mask, self._oracle_H, (W_img, H_img))
                > 0
            )

        # Always-on observables: independent of the inlier gate and of the
        # trigger outcome, so any trigger rule can be evaluated offline.
        telemetry = _correction_telemetry(seg_mask, loma_mask, gt_mask, cfg)
        if (
            loma_H is not None
            and self._oracle_H is not None
            and self.state.init_corners is not None
        ):
            telemetry.loma_gt_err = float(
                alignment_error(
                    warp_corners(self._oracle_H, self.state.init_corners),
                    warp_corners(loma_H, self.state.init_corners),
                )
            )
        self._corr_telemetry = telemetry

        # Per-leg source resolution; a "loma" leg additionally requires the
        # inlier gate, an "oracle" leg only a valid GT homography this frame.
        loma_confident = loma_mask is not None and n_inliers >= cfg.inlier_threshold
        confident_loma_mask = loma_mask if loma_confident else None
        trigger_mask = (
            confident_loma_mask if cfg.trigger_source == "loma" else gt_mask
        )
        content_mask = (
            confident_loma_mask if cfg.content_source == "loma" else gt_mask
        )

        if cfg.trigger == "periodic":
            fired = self.state.frame_idx % cfg.period == 0
        else:
            fired = trigger_mask is not None and _should_correct(
                seg_mask, trigger_mask, cfg
            )
        telemetry.trigger_fired = bool(fired)
        if not fired or content_mask is None or cfg.dry_run:
            return False

        corr_mask_u8 = content_mask.astype(np.uint8)
        # Record the correction on self BEFORE the side-effect. segmenter.correct
        # mutates SAM3 memory; if it (or anything after) raises, step_with_mask
        # swallows the exception and the caller's `_loma_last_corrected = ...`
        # assignment never runs — but the memory is already changed, so the event
        # must still reach the debug sidecar. self._loma_last_corr_mask also lets
        # the run save the injected mask next to the original SAM3 mask.
        self._loma_last_corrected = True
        self._loma_last_corr_mask = corr_mask_u8
        self.segmenter.correct(corr_mask_u8)
        return True

    def _score_homography(
        self,
        seg_mask: np.ndarray,
        homography: Optional[np.ndarray],
        image_shape: tuple[int, int],
    ) -> tuple[float, float]:
        """Score candidate homography with NN scorer.

        Args:
            seg_mask: Segmentation mask (H, W) for current frame.
            homography: Candidate homography (3, 3) or ``None``.
            image_shape: Current image shape as ``(height, width)``.

        Returns:
            Tuple ``(rmse, sigma)`` in pixels. Returns ``(inf, inf)`` when the
            candidate is missing.
        """
        if self._nn_scorer is None or homography is None:
            return float("inf"), float("inf")
        return self._nn_scorer.score(seg_mask, homography, image_shape)

    def _select_best_candidate(
        self,
        candidates: list[ScoredCandidate],
    ) -> Optional[ScoredCandidate]:
        """Pick candidate with smallest predicted RMSE.

        Args:
            candidates: Candidate list to compare.

        Returns:
            Best candidate, or ``None`` when list is empty.
        """
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: candidate.rmse)

    def _finalize_success(
        self,
        selected_homography: np.ndarray,
        frame: np.ndarray,
        seg_mask: np.ndarray,
        n_mask_corr: int,
        debug: Optional[TrackerDebugInfo] = None,
        run_symmetry: bool = True,
    ) -> TrackResult:
        """Store selected homography, run symmetry, and build result.

        Args:
            run_symmetry: Resolve the planar flip ambiguity on the selected
                homography. Disable for keypoint-based (LoMa) estimates: their
                corner labelling is fixed by the matches, so symmetry resolution
                — a mask-only disambiguation — can only flip them incorrectly.
        """
        self.state.last_homography = selected_homography
        if run_symmetry:
            self._run_symmetry(selected_homography, frame, n_mask_corr)
        return TrackResult(
            homography=self.state.last_homography,
            mask=seg_mask,
            debug=debug,
            correction_mask=self._loma_last_corr_mask,
        )

    def _finalize_failure(
        self,
        frame: np.ndarray,
        seg_mask: np.ndarray,
        hough_enabled: bool,
        message: str,
        debug: Optional[TrackerDebugInfo] = None,
        run_symmetry: bool = True,
    ) -> TrackResult:
        """Keep last homography and build fallback result.

        Args:
            run_symmetry: Resolve symmetry on the kept homography. Disabled in
                the LoMa path, where the kept estimate may be keypoint-based.
        """
        logger.warning(message)
        if hough_enabled and run_symmetry:
            self._run_symmetry(self.state.last_homography, frame, 0)
        return TrackResult(
            homography=self.state.last_homography,
            mask=seg_mask,
            debug=debug,
            correction_mask=self._loma_last_corr_mask,
        )

    # ------------------------------------------------------------------
    # Symmetry resolution
    # ------------------------------------------------------------------

    def _run_symmetry(
        self,
        H: np.ndarray,
        frame: np.ndarray,
        n_mask_corr: int,
    ) -> None:
        """Run symmetry resolution if enabled, updating state in-place.

        Args:
            H: Current best homography (init -> current).
            frame: Current frame (H, W, C) BGR uint8.
            n_mask_corr: Number of Hough correspondences this frame.
        """
        if not self.cfg.symmetry.enabled or self.state.init_corners is None:
            return

        if self.cfg.symmetry.method == "matches":
            # The four cyclic shifts share a silhouette, so only texture can
            # separate them; the keypoint matcher has already produced the
            # texture evidence, and scoring the shifts against it directly
            # needs no second network, no temporal run-length and no gate.
            matches = self._loma_last_matches
            named, shift, support = match_naming.name_by_match_support(
                warp_corners(H, self.state.init_corners),
                self.state.init_corners,
                *(self._loma_last_matches or (None, None)),
                tau_px=self.cfg.symmetry.match_tau_px,
            )
            # Record what it saw and did, so a null grid result can be read as
            # "fired and did not help" rather than "silently never ran".
            self._maskh_telemetry.naming_shift = int(shift)
            self._maskh_telemetry.naming_support = support
            # Per-naming residual histograms, so `tau` can be re-tested offline
            # without re-running the tracker: the truncated-quadratic support is
            # recoverable in closed form for any tau on a bin edge (30 is one).
            # Only the registrar path recorded these before, which left the
            # deployed `hough` path's one free parameter unablatable.
            if matches is not None and matches[0] is not None:
                _, counts, sum_sq = match_naming.alias_supports(
                    warp_corners(H, self.state.init_corners),
                    self.state.init_corners,
                    matches[0],
                    matches[1],
                    self.cfg.symmetry.match_tau_px,
                )
                self._maskh_telemetry.naming_hist_counts = counts
                self._maskh_telemetry.naming_hist_sumsq = sum_sq
            self._maskh_telemetry.naming_n_matches = (
                -1 if matches is None or matches[0] is None else int(len(matches[0]))
            )
            if shift > 0:
                resolved = match_naming.homography_from_corners(
                    self.state.init_corners, named
                )
                if resolved is not None:
                    self.state.last_homography = resolved
            return

        if self._dino is None:
            return

        from flatsam2.hough_homography import resolve_symmetry

        sym = self.cfg.symmetry
        with profiler.section("symmetry"):
            resolved_H, confirmed = resolve_symmetry(
                H,
                self.state.init_corners,
                self.state.init_frame,
                frame,
                n_mask_corr,
                self.state.symmetry,
                self._dino,
                only_when_lost=sym.only_when_lost,
                n_correspondences_enough=sym.n_correspondences_enough,
                resolution_hw=tuple(sym.resolution_hw),
                dino_layer=sym.dino_layer,
                abs_thr=sym.abs_thr,
                fts_thr=sym.fts_thr,
                run_length=sym.run_length,
            )
        if confirmed:
            logger.info("Symmetry shift confirmed.")
        self.state.last_homography = resolved_H
