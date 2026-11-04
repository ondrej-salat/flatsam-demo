"""SEA-RAFT optical flow with Sintel weights configuration."""
from flatsam2.config import FlowConfig

config = FlowConfig(
    model="sea_raft_m",
    weights="sintel",
)
