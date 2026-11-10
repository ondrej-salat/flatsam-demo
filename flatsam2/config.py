"""Configuration system for FlatSAM tracker using Pydantic."""

import importlib.util
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from .registry import DATASETS

Algorithm = Literal["flow_hough", "loma_samh"]


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
            "RNG state untouched (stochastic). Mirrors "
            "LoMaPathConfig.ransac_seed for the flow path."
        ),
    )


class HoughConfig(BaseModel):
    """Mask -> homography estimator configuration (Hough lines -> quad pipeline)."""

    enabled: bool = Field(
        default=False,
        description="Enable the mask-based homography estimator (SAM-H branch)",
    )
    method: Literal["hough"] = Field(
        default="hough",
        description="Mask->H estimator: 'hough' = lines->quad pipeline",
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
            "DINO knobs stop applying."
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
            "SAM-H (hough + symmetry, LoMa disabled) run; the LoMa inlier rule only "
            "selects per frame between the LoMa estimate and that independent SAM-H "
            "estimate, never feeding back."
        ),
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True


class CorrectionCfg(BaseModel):
    """SAM3 self-correction gate for online LoMa tracker.

    The mask-divergence triggers compare the SAM mask against the mask induced by
    the LoMa homography, and the injected correction is that same mask; both
    require LoMa to reach ``inlier_threshold``. ``trigger="periodic"`` fires on a
    fixed cadence but still needs a confident LoMa mask to inject. A trigger that
    fires without one abstains, recorded as ``trigger_fired=True, corrected=False``.
    """

    enabled: bool = Field(default=False, description="Enable SAM3 memory correction")
    inlier_threshold: int = Field(
        default=70, description="Correct only when LoMa inliers reach this threshold"
    )
    trigger: Literal["area_margin", "sam_excess_depth", "periodic"] = Field(
        default="area_margin", description="Correction trigger strategy"
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
    # sam_excess_depth trigger. Fires on the signed-distance depth feature
    # dout_corr: SAM area lying more than excess_depth_px past the correction-mask
    # boundary, as a fraction of the correction-mask area. Robust to few-px
    # boundary jitter, unlike area_margin's raw total-area ratio.
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
        symmetry: cyclic corner-shift (symmetry) resolution configuration
        fallback_to_mask_H: use ``mask_H`` directly when flow + ``mask_H`` prewarp both fail
        loma: optional online LoMa homography path
        correction: optional SAM3 self-correction config
        algorithm: explicit control-flow discriminator, ``flow_hough`` or
            ``loma_samh``. When ``None`` it is derived from ``loma.enabled``.
            When set explicitly it must agree with that derivation, otherwise a
            ``ValueError`` is raised: the discriminator is a single source of
            truth, never an override of the flag.
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
    fallback_to_mask_H: bool = Field(
        default=True,
        description="Fall back to mask_H directly when flow fails with mask_H prewarp",
    )
    loma: LoMaPathConfig = Field(default_factory=LoMaPathConfig)
    correction: CorrectionCfg = Field(default_factory=CorrectionCfg)
    algorithm: Optional[Algorithm] = Field(
        default=None,
        description=(
            "Explicit control-flow discriminator (flow_hough / loma_samh). None "
            "derives it from loma.enabled; an explicit value must match it."
        ),
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True

    def _derive_algorithm(self) -> Algorithm:
        """Derive the control-flow discriminator from ``loma.enabled``.

        LoMa (coupled or decoupled SAM-H alike) selects ``loma_samh``; otherwise
        the plain flow/Hough path.

        Returns:
            The algorithm name implied by the current flags.
        """
        return "loma_samh" if self.loma.enabled else "flow_hough"

    @model_validator(mode="after")
    def _resolve_algorithm(self) -> "TrackerConfig":
        """Fill or cross-check ``algorithm`` against the flag derivation.

        Gotcha for config-file authors: this validator runs on construction
        (``TrackerConfig(...)``) but ``model_copy(update=...)`` — how every
        ``configs/tracker/*.py`` composes a base config — does NOT re-run
        validators. Flipping a flag that changes the derivation (e.g.
        ``loma.enabled``) through ``model_copy`` alone leaves ``algorithm``
        stuck at whatever the base config already resolved to; pass
        ``"algorithm"`` explicitly in the same ``update=`` dict.

        Returns:
            ``self`` with ``algorithm`` resolved to a concrete value.

        Raises:
            ValueError: if ``algorithm`` was set explicitly to a value that
                disagrees with the derivation from the flags.
        """
        derived = self._derive_algorithm()
        if self.algorithm is None:
            self.algorithm = derived
        elif self.algorithm != derived:
            raise ValueError(
                f"algorithm={self.algorithm!r} disagrees with "
                f"loma.enabled={self.loma.enabled}, which implies "
                f"algorithm={derived!r}. Fix the config: the discriminator must "
                f"match the flags."
            )
        return self


def load_config_from_file(config_path: Path) -> TrackerConfig:
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

    # Try to extract config
    if hasattr(module, "config"):
        config = module.config
        if isinstance(config, TrackerConfig):
            return config
        else:
            raise ValueError(
                f"Config file {config_path} has 'config' variable but it's not a "
                f"TrackerConfig instance"
            )

    # Try to find a TrackerConfig instance in the module
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, TrackerConfig):
            return obj

    raise ValueError(
        f"Config file {config_path} must define a 'config' variable "
        f"or contain a TrackerConfig instance"
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
