"""Flow + Hough mask-H fallback + DINO symmetry resolution.

Primary: prewarp + RAFT optical flow + RANSAC
Fallback: mask -> Hough lines -> homography (when flow fails)
Symmetry: DINOv2 appearance-based cyclic corner disambiguation
          (runs only when recovering from lost state)
"""

from flatsam2.config import (
    FlowConfig,
    HoughConfig,
    RansacConfig,
    SegmenterConfig,
    SymmetryConfig,
    TrackerConfig,
)

config = TrackerConfig(
    flow=FlowConfig(model="raft", weights="sintel"),
    segmenter=SegmenterConfig(model="sam3"),
    ransac=RansacConfig(inl_threshold=5.0),
    filter_by_seg_mask=True,
    hough=HoughConfig(enabled=True),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    algorithm="flow_hough",
)
