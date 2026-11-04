"""SAM3 candidate-mask dump — segmentation only, no tracking.

Streams the SAM3 streaming tracker over each sequence and saves *every*
candidate mask per frame (via the streaming tracker's multi-candidate API) for
offline exploration. No optical flow / homography is computed, so no flow model
is built (``flow_enabled=False``). Run with run.py / slurm/submit.py like any
other config; output lands under ``results/sam3_candidates/<dataset>/masks/``.
"""

from flatsam2.config import SegmenterConfig, TrackerConfig

config = TrackerConfig(
    flow_enabled=False,
    segmenter=SegmenterConfig(model="sam3"),
    candidate_dump=True,
)
