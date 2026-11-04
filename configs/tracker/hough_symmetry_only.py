"""Segmentation + Hough homography with symmetry disambiguation only.

Primary: mask -> Hough lines -> homography
No optical flow path is used.
Symmetry: DINOv2 appearance-based cyclic corner disambiguation
"""

from flatsam2.config import HoughConfig, SegmenterConfig, SymmetryConfig, TrackerConfig

config = TrackerConfig(
    flow_enabled=False,
    segmenter=SegmenterConfig(model="sam3"),
    hough=HoughConfig(enabled=True),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    algorithm="flow_hough",
)
