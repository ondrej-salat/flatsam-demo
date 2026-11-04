"""Online LoMa + SAM-H inlier-choice tracker."""

from flatsam2.config import (
    HoughConfig,
    LoMaPathConfig,
    SamplingCfg,
    SegmenterConfig,
    SymmetryConfig,
    TrackerConfig,
)


config = TrackerConfig(
    flow_enabled=False,
    segmenter=SegmenterConfig(model="sam3"),
    hough=HoughConfig(enabled=True),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    loma=LoMaPathConfig(
        enabled=True,
        model="LoMaR",
        sampling=SamplingCfg(
            n_object=4096,
            total_budget=4096,
            nms_size=3,
            subpixel=True,
            subpixel_temp=0.5,
        ),
        init_mask_inflate_px=10,
        current_mask_inflate_px=10,
        ransac_method="USAC_MAGSAC",
        ransac_threshold=5.0,
        ransac_seed=0,
        inlier_threshold=35,
    ),
    algorithm="loma_samh",
)
