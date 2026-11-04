"""Default FlatSAM tracker configuration."""
from flatsam2.config import TrackerConfig, FlowConfig, SegmenterConfig

config = TrackerConfig(
    flow=FlowConfig(
        model="sea_raft_m",
        weights="sintel",
    ),
    segmenter=SegmenterConfig(
        model="sam3"
    ),
    filter_by_seg_mask=True,
    algorithm="flow_hough",
)
