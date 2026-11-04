"""Segmentation + Hough homography with wider intersection filtering and fixed orthogonal line refinement."""

from flatsam2.config import HoughConfig, SegmenterConfig, SymmetryConfig, TrackerConfig

config = TrackerConfig(
    flow_enabled=False,
    segmenter=SegmenterConfig(model="sam3"),
    hough=HoughConfig(
        enabled=True,
        corner_max_distance_image_fraction=0.5,
        min_intersection_angle_deg=5.0,
        line_refinement_method="orthogonal_tls",
    ),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    algorithm="flow_hough",
)
