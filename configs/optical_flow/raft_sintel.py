"""RAFT optical flow with Sintel weights configuration."""
from flatsam2.config import FlowConfig

config = FlowConfig(
    model="raft",
    weights="sintel",
)
