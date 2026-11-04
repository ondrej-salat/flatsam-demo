"""Tracker config: sea_raft_m flow + Hough re-detection + symmetry, NN scorer with trust-cascade selection.

Uses the trained BoundaryTransformer to select homography candidates via a
trust-based cascade (flow_H → redetection_flow_H → mask_H): the first candidate whose
predicted RMSE ≤ trust_threshold is used immediately, reflecting the prior
that flow is more precise than mask.  Debug info is saved for offline analysis.

See flowsearaft_hough_symmetry_nn.py for the original min-RMSE selection mode.

Run:
    flatsam2 run pot --config configs/tracker/flowsearaft_hough_symmetry_nn_cascade.py --output-dir ./results
    flatsam2 evaluate pot --results-dir ./results --config configs/tracker/flowsearaft_hough_symmetry_nn_cascade.py
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
        trust_threshold=15.0,
    ),
    algorithm="nn_cascade",
)
