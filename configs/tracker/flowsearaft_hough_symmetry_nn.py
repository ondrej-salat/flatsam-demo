"""Tracker config: flow + Hough re-detection, NN scorer for candidate selection.

Uses the trained BoundaryTransformer to score each homography candidate
(flow_H, mask_H, redetection_flow_H) and select the best one per frame. Debug info
(all candidates + scores) is saved alongside predictions for offline analysis.

Run:
    flatsam2 run pot --config configs/tracker/flow_hough_nn.py --output-dir ./results
    flatsam2 evaluate pot --results-dir ./results --config configs/tracker/flow_hough_nn.py
"""

from flatsam2.config import (
    FlowConfig,
    HoughConfig,
    SymmetryConfig,
    NNScorerConfig,
    TrackerConfig,
)

config = TrackerConfig(
    flow=FlowConfig(model="sea_raft_m", weights="sintel"),
    hough=HoughConfig(enabled=True),
    symmetry=SymmetryConfig(enabled=True),
    fallback_to_mask_H=True,
    nn_scorer=NNScorerConfig(
        enabled=True,
        checkpoint_path="checkpoints/nn_quality/best.pt",
        arch="option_e",
        n_points=128,
    ),
    algorithm="nn_min_rmse",
)
