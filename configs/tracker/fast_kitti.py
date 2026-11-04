"""Fast tracker configuration with KITTI weights.

Example showing how to compose configs from separate modules.
"""
import sys
from pathlib import Path

# Add configs to path for imports
configs_dir = Path(__file__).parent.parent
sys.path.insert(0, str(configs_dir))

from flatsam2.config import TrackerConfig
from optical_flow.raft_kitti import config as flow_config
from segmenter.sam3_default import config as segmenter_config

config = TrackerConfig(
    flow=flow_config,
    segmenter=segmenter_config,
    algorithm="flow_hough",
)
