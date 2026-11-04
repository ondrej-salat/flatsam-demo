"""FlatSAM tracker configuration from 2025-03-15 (before seg mask filtering)."""
from flatsam2.config import TrackerConfig, FlowConfig, SegmenterConfig

config = TrackerConfig(
    flow=FlowConfig(
        model="raft",
        weights="sintel",
    ),
    segmenter=SegmenterConfig(
        model="sam3"
    ),
    filter_by_seg_mask=False,
    algorithm="flow_hough",
)
