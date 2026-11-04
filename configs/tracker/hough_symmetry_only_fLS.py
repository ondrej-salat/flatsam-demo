"""Segmentation + Hough homography with fixed orthogonal line refinement."""

from flatsam2.config import HoughConfig, SegmenterConfig, SymmetryConfig, TrackerConfig

config = TrackerConfig(
    flow_enabled=False,
    segmenter=SegmenterConfig(model="sam3"),
    hough=HoughConfig(
        enabled=True,
        line_refinement_method="orthogonal_tls",
    ),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    algorithm="flow_hough",
)
