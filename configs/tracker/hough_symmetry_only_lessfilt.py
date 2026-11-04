"""Segmentation + Hough homography with wider corner filtering.

Fork of hough_symmetry_only.py that keeps Hough line intersections farther from
the visible mask contour, useful when one quadrilateral corner is out of view.
"""

from flatsam2.config import HoughConfig, SegmenterConfig, SymmetryConfig, TrackerConfig

config = TrackerConfig(
    flow_enabled=False,
    segmenter=SegmenterConfig(model="sam3"),
    hough=HoughConfig(
        enabled=True,
        corner_max_distance_image_fraction=0.5,
        min_intersection_angle_deg=5.0,
    ),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    algorithm="flow_hough",
)
