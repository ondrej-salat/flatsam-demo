"""Censored heteroscedastic regression loss for homography quality estimation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def censored_heteroscedastic_loss(
    pred_rmse: Tensor,
    log_sigma: Tensor,
    gt_rmse: Tensor,
    n_clip: float = 30.0,
    censored: Tensor | None = None,
) -> Tensor:
    """Censored heteroscedastic regression loss.

    For uncensored samples (gt_rmse <= n_clip):
        L = (pred - gt)^2 * exp(-2*log_sigma) + 2*log_sigma

    For censored samples (gt_rmse > n_clip):
        L = relu(n_clip - pred)^2

    Args:
        pred_rmse: (B,) predicted RMSE values.
        log_sigma: (B,) predicted log-sigma values.
        gt_rmse: (B,) ground-truth RMSE values.
        n_clip: Censoring threshold in pixels.
        censored: Optional explicit censoring mask.  When omitted, censoring is
            inferred as ``gt_rmse > n_clip`` for backwards compatibility.

    Returns:
        Scalar mean loss.
    """
    if censored is None:
        censored = gt_rmse > n_clip
    uncensored = ~censored

    total = torch.zeros_like(pred_rmse)

    # Uncensored NLL loss
    if uncensored.any():
        residual = pred_rmse[uncensored] - gt_rmse[uncensored]
        ls = log_sigma[uncensored]
        total[uncensored] = residual**2 * torch.exp(-2 * ls) + 2 * ls

    # Censored hinge loss
    if censored.any():
        shortfall = F.relu(n_clip - pred_rmse[censored])
        total[censored] = shortfall**2

    return total.mean()
