"""Inference wrapper for NN homography quality estimation.

Provides a ready-to-use scorer for tracker integration.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..homography_quality import ReferenceContour
from .models import (
    BoundaryTransformer,
    PointSetBaseline,
    SDFCropCNN,
    build_option_c_input,
    build_option_e_features,
    build_point_baseline_features,
)
from .preprocess import preprocess


class NNHomographyScorer:
    """Ready-to-use inference wrapper for tracker integration.

    Loads a trained model, performs preprocessing, returns
    (predicted_rmse, sigma) in absolute pixel units.
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch: Optional[str] = None,
        device: str = "cuda",
        n_points: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.ref: Optional[ReferenceContour] = None
        self.model, state, checkpoint_arch = load_model_checkpoint(
            checkpoint_path,
            self.device,
            arch=arch,
        )
        self.arch = checkpoint_arch
        checkpoint_config = state.get("config", {})
        self.n_points = int(
            n_points if n_points is not None else checkpoint_config.get("n_points", 128)
        )
        self.crop_size = int(checkpoint_config.get("crop_size", 128))
        self.sdf_downsample = int(checkpoint_config.get("sdf_downsample", 1))
        self.model.to(self.device)
        self.model.eval()

    def init_reference(self, ref: ReferenceContour) -> None:
        """Store reference for repeated scoring. Call once per target."""
        self.ref = ref

    @torch.no_grad()
    def score(
        self,
        mask_B: np.ndarray,
        H: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Score a single homography hypothesis.

        Args:
            mask_B: (H, W) uint8 segmentation mask.
            H: (3, 3) float64 homography matrix.
            image_shape: (height, width) of the current frame.

        Returns:
            (predicted_rmse, sigma) in absolute pixel units.
        """
        if self.ref is None:
            raise RuntimeError("Call init_reference() before scoring.")

        prep = preprocess(
            self.ref,
            mask_B,
            H,
            image_shape,
            self.n_points,
            sdf_downsample=self.sdf_downsample,
            architecture=self.arch,
        )
        if prep is None:
            return float("inf"), float("inf")

        if self.arch == "option_e":
            feat_A, feat_B = build_option_e_features(prep)
            feat_A_t = torch.from_numpy(feat_A).unsqueeze(0).to(self.device)
            feat_B_t = torch.from_numpy(feat_B).unsqueeze(0).to(self.device)
            pred_rmse, log_sigma = self.model(feat_A_t, feat_B_t)
        elif self.arch == "point_baseline":
            features = build_point_baseline_features(prep)
            features_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            pred_rmse, log_sigma = self.model(features_t)
        else:
            crop, s = build_option_c_input(prep, crop_size=self.crop_size)
            crop_t = crop.unsqueeze(0).to(self.device)
            pred_rmse, log_sigma = self.model(crop_t)
            pred_rmse = pred_rmse / s
            log_sigma = log_sigma - math.log(s)

        rmse = pred_rmse.item()
        sigma = math.exp(log_sigma.clamp(-5, 5).item())
        return rmse, sigma

    @torch.no_grad()
    def rank_hypotheses(
        self,
        mask_B: np.ndarray,
        H_candidates: List[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> List[Dict]:
        """Score and rank multiple homography candidates.

        Returns:
            List sorted by predicted_rmse ascending, each entry a dict with
            keys: index, H, predicted_rmse, sigma.
        """
        results = []
        for i, H in enumerate(H_candidates):
            rmse, sigma = self.score(mask_B, H, image_shape)
            results.append(
                {
                    "index": i,
                    "H": H,
                    "predicted_rmse": rmse,
                    "sigma": sigma,
                }
            )
        results.sort(key=lambda r: r["predicted_rmse"])
        return results


def build_model_for_arch(
    arch: str,
    config: Optional[Dict] = None,
) -> torch.nn.Module:
    """Instantiate a model using saved checkpoint hyperparameters when available."""
    config = config or {}
    if arch == "option_e":
        return BoundaryTransformer(
            d_model=int(config.get("d_model", 64)),
            n_heads=int(config.get("n_heads", 4)),
            n_layers=int(config.get("n_layers", 2)),
            n_points=int(config.get("n_points", 128)),
        )
    if arch == "point_baseline":
        return PointSetBaseline(
            d_model=int(config.get("d_model", 64)),
            n_heads=int(config.get("n_heads", 4)),
            n_layers=int(config.get("n_layers", 2)),
        )
    if arch == "option_c":
        return SDFCropCNN(crop_size=int(config.get("crop_size", 128)))
    raise ValueError(f"Unsupported architecture: {arch!r}")


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    arch: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict, str]:
    """Load a checkpoint and rebuild the matching model architecture."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_config = state.get("config", {})
    checkpoint_arch = arch or checkpoint_config.get("arch", "option_e")
    model = build_model_for_arch(checkpoint_arch, checkpoint_config)
    model.load_state_dict(state["model"])
    return model, state, checkpoint_arch
