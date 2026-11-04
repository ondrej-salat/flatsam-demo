"""Neural network architectures for homography quality estimation.

Option E: BoundaryTransformer — bipartite attention on boundary point sets.
Option C: SDFCropCNN — depthwise-separable CNN on SDF crops.
Baseline: PointSetBaseline — standard transformer on coordinates only (no SDF).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Feature construction helpers
# =========================================================================

def fourier_pe(xy: torch.Tensor, K: int = 4) -> torch.Tensor:
    """Fourier positional encoding.

    Args:
        xy: (N, 2) or (B, N, 2) normalised coordinates.
        K: Number of frequency bands.

    Returns:
        (N, 4K) or (B, N, 4K) Fourier features.
    """
    result = []
    for k in range(K):
        freq = (2 ** k) * math.pi
        result.append(torch.sin(freq * xy[..., 0]))
        result.append(torch.cos(freq * xy[..., 0]))
        result.append(torch.sin(freq * xy[..., 1]))
        result.append(torch.cos(freq * xy[..., 1]))
    return torch.stack(result, dim=-1)


def build_option_e_features(
    preprocess_dict: dict,
    n_fourier_bands: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert preprocess output to (N, 20) feature arrays.

    Args:
        preprocess_dict: Output from preprocess().
        n_fourier_bands: Number of Fourier PE bands (K).

    Returns:
        (feat_A, feat_B) each (N, 20) float32 numpy arrays.
    """
    # ∂A' features
    raw_A = np.stack([
        preprocess_dict["sdf_B_at_Ap"],
        preprocess_dict["tan_sin_A"],
        preprocess_dict["tan_cos_A"],
        preprocess_dict["is_in_view"],
    ], axis=-1)  # (N, 4)

    pe_A = _fourier_pe_np(preprocess_dict["norm_xy_A"], n_fourier_bands)  # (N, 16)
    feat_A = np.concatenate([raw_A, pe_A], axis=-1)  # (N, 20)

    # ∂B features
    n = len(preprocess_dict["sdf_Ap_at_B"])
    raw_B = np.stack([
        preprocess_dict["sdf_Ap_at_B"],
        preprocess_dict["tan_sin_B"],
        preprocess_dict["tan_cos_B"],
        np.ones(n, dtype=np.float32),
    ], axis=-1)  # (N, 4)

    pe_B = _fourier_pe_np(preprocess_dict["norm_xy_B"], n_fourier_bands)  # (N, 16)
    feat_B = np.concatenate([raw_B, pe_B], axis=-1)  # (N, 20)

    return feat_A.astype(np.float32), feat_B.astype(np.float32)


def _fourier_pe_np(xy: np.ndarray, K: int = 4) -> np.ndarray:
    """Fourier positional encoding (numpy version).

    Args:
        xy: (N, 2) normalised coordinates.
        K: Number of frequency bands.

    Returns:
        (N, 4K) float32 array.
    """
    result = []
    for k in range(K):
        freq = (2 ** k) * math.pi
        result.extend([
            np.sin(freq * xy[:, 0]),
            np.cos(freq * xy[:, 0]),
            np.sin(freq * xy[:, 1]),
            np.cos(freq * xy[:, 1]),
        ])
    return np.stack(result, axis=-1).astype(np.float32)


def build_point_baseline_features(
    preprocess_dict: dict,
    n_fourier_bands: int = 4,
) -> np.ndarray:
    """Build (2N, C) combined feature array for the point-baseline architecture.

    No SDF or tangent features are used.  Each point receives:
      - side indicator  (0.0 = ∂A', 1.0 = ∂B)
      - out-of-view / boundary-of-view flag
          * ∂A' points: 1 when the warped point falls outside the frame
          * ∂B points: 1 when the point lies on the frame boundary
      - Fourier PE of normalised (x, y)  →  4 * n_fourier_bands values

    Args:
        preprocess_dict: Output from preprocess().
        n_fourier_bands: Number of Fourier PE frequency bands (K).

    Returns:
        (2N, 2 + 4*n_fourier_bands) float32 array; ∂A' rows first, then ∂B.
    """
    N = len(preprocess_dict["is_in_view"])

    # ∂A' rows: side=0, flag = out-of-view (1 when outside frame)
    side_A = np.zeros(N, dtype=np.float32)
    oov_A = (1.0 - preprocess_dict["is_in_view"]).astype(np.float32)
    pe_A = _fourier_pe_np(preprocess_dict["norm_xy_A"], n_fourier_bands)  # (N, 4K)
    feat_A = np.concatenate([side_A[:, None], oov_A[:, None], pe_A], axis=-1)

    # ∂B rows: side=1, flag = on frame boundary
    side_B = np.ones(N, dtype=np.float32)
    oov_B = preprocess_dict["is_on_boundary_B"].astype(np.float32)
    pe_B = _fourier_pe_np(preprocess_dict["norm_xy_B"], n_fourier_bands)
    feat_B = np.concatenate([side_B[:, None], oov_B[:, None], pe_B], axis=-1)

    # ∂A' points first, ∂B points second
    return np.concatenate([feat_A, feat_B], axis=0).astype(np.float32)  # (2N, 18)


# =========================================================================
# Option E: Boundary Transformer
# =========================================================================

class TransformerBlock(nn.Module):
    """One block: self-attn A, self-attn B, cross A→B, cross B→A."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_attn_A = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.self_attn_B = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.cross_attn_AB = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.cross_attn_BA = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.norm_sa_A = nn.LayerNorm(d_model)
        self.norm_sa_B = nn.LayerNorm(d_model)
        self.norm_ca_A = nn.LayerNorm(d_model)
        self.norm_ca_B = nn.LayerNorm(d_model)

    def forward(
        self, tok_A: torch.Tensor, tok_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention A
        tok_A = self.norm_sa_A(
            tok_A + self.self_attn_A(tok_A, tok_A, tok_A)[0])
        # Self-attention B
        tok_B = self.norm_sa_B(
            tok_B + self.self_attn_B(tok_B, tok_B, tok_B)[0])
        # Cross-attention A→B (A queries B)
        tok_A = self.norm_ca_A(
            tok_A + self.cross_attn_AB(tok_A, tok_B, tok_B)[0])
        # Cross-attention B→A (B queries A)
        tok_B = self.norm_ca_B(
            tok_B + self.cross_attn_BA(tok_B, tok_A, tok_A)[0])
        return tok_A, tok_B


class BoundaryTransformer(nn.Module):
    """Primary architecture (~115K params).

    Bipartite attention on boundary point sets, inspired by SuperGlue.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_points: int = 128,
        n_fourier_bands: int = 4,
        n_raw_features: int = 4,
    ):
        super().__init__()
        self.input_dim = n_raw_features + 4 * n_fourier_bands  # 4 + 16 = 20

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.type_embed = nn.Embedding(2, d_model)  # 0 = ∂A', 1 = ∂B

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.rmse_head = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initialise sigma head bias to 0 → sigma_init = exp(0) = 1px
        nn.init.zeros_(self.sigma_head[-1].bias)

    def forward(
        self,
        features_A: torch.Tensor,
        features_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features_A: (B, N, 20) raw + Fourier PE for ∂A' points.
            features_B: (B, N, 20) raw + Fourier PE for ∂B points.

        Returns:
            (predicted_rmse, log_sigma) each (B,).
        """
        tok_A = self.input_proj(features_A) + self.type_embed(
            torch.zeros(1, dtype=torch.long, device=features_A.device))
        tok_B = self.input_proj(features_B) + self.type_embed(
            torch.ones(1, dtype=torch.long, device=features_B.device))

        for layer in self.layers:
            tok_A, tok_B = layer(tok_A, tok_B)

        v_A = tok_A.mean(dim=1)  # (B, d)
        v_B = tok_B.mean(dim=1)  # (B, d)
        v = torch.cat([v_A, v_B], dim=-1)  # (B, 2d)

        raw_rmse = self.rmse_head(v)  # (B, 1)
        predicted_rmse = F.softplus(raw_rmse)
        log_sigma = self.sigma_head(v).clamp(-5, 5)  # (B, 1)

        return predicted_rmse.squeeze(-1), log_sigma.squeeze(-1)


# =========================================================================
# Option C: SDF-Crop CNN
# =========================================================================

class DWSBlock(nn.Module):
    """Depthwise-separable conv block: DWConv + PWConv + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1,
            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pw(self.dw(x))))


class SDFCropCNN(nn.Module):
    """Backup architecture (~150K params).

    Depthwise-separable CNN on 4-channel SDF crops.
    """

    def __init__(self, crop_size: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            DWSBlock(4, 32, stride=2),    # → 32 × 64 × 64
            DWSBlock(32, 64, stride=2),   # → 64 × 32 × 32
            DWSBlock(64, 128, stride=2),  # → 128 × 16 × 16
            DWSBlock(128, 128, stride=2), # → 128 × 8 × 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rmse_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sigma_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        nn.init.zeros_(self.sigma_head[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 4, 128, 128) — [binary_A', SDF_A', binary_B, SDF_B].

        Returns:
            (predicted_rmse, log_sigma) each (B,).
            Note: predicted_rmse is in crop-pixel units; caller must
            divide by scale factor s.
        """
        feat = self.pool(self.backbone(x)).flatten(1)  # (B, 128)
        raw_rmse = self.rmse_head(feat)
        predicted_rmse = F.softplus(raw_rmse)
        log_sigma = self.sigma_head(feat).clamp(-5, 5)
        return predicted_rmse.squeeze(-1), log_sigma.squeeze(-1)


# =========================================================================
# Option C input construction
# =========================================================================

def build_option_c_input(
    preprocess_dict: dict,
    crop_size: int = 128,
) -> Tuple[torch.Tensor, float]:
    """Build 4-channel crop from preprocess output.

    Args:
        preprocess_dict: Output from preprocess().
        crop_size: Target crop size (square).

    Returns:
        (tensor_4ch, scale_factor) where tensor_4ch is (4, crop_size, crop_size)
        float32 and scale_factor is the isotropic resize factor.
    """
    mask_Ap = preprocess_dict["mask_Ap"]
    sdf_Ap = preprocess_dict["sdf_Ap"]
    mask_B = preprocess_dict["mask_B"]
    sdf_B = preprocess_dict["sdf_B"]

    # Bounding box of union(A', B), expand 20%
    bbox = _union_bbox(mask_Ap, mask_B)
    if bbox is None:
        # Both masks empty — shouldn't happen but handle gracefully
        tensor = np.zeros((4, crop_size, crop_size), dtype=np.float32)
        return torch.from_numpy(tensor), 1.0

    x0, y0, x1, y1 = bbox
    h_img, w_img = mask_Ap.shape[:2]

    # Expand 20%
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - 0.2 * bw))
    y0 = max(0, int(y0 - 0.2 * bh))
    x1 = min(w_img, int(x1 + 0.2 * bw))
    y1 = min(h_img, int(y1 + 0.2 * bh))

    # Ensure non-zero crop
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1

    # Crop
    crop_Ap = mask_Ap[y0:y1, x0:x1]
    crop_sdf_Ap = sdf_Ap[y0:y1, x0:x1]
    crop_B = mask_B[y0:y1, x0:x1]
    crop_sdf_B = sdf_B[y0:y1, x0:x1]

    # Isotropic resize
    ch, cw = crop_Ap.shape
    s = crop_size / max(ch, cw)
    new_h, new_w = round(ch * s), round(cw * s)
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Resize binary (nearest) and SDF (bilinear)
    bin_Ap = cv2.resize(
        crop_Ap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    sdf_Ap_r = cv2.resize(
        crop_sdf_Ap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bin_B = cv2.resize(
        crop_B, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    sdf_B_r = cv2.resize(
        crop_sdf_B, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to crop_size × crop_size (center-pad with zeros)
    def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
        h, w = arr.shape[:2]
        ph = target - h
        pw = target - w
        top = ph // 2
        left = pw // 2
        return np.pad(arr, ((top, ph - top), (left, pw - left)),
                       mode="constant", constant_values=0)

    bin_Ap = _pad_to(bin_Ap, crop_size)
    sdf_Ap_r = _pad_to(sdf_Ap_r, crop_size)
    bin_B = _pad_to(bin_B, crop_size)
    sdf_B_r = _pad_to(sdf_B_r, crop_size)

    # Stack: rescale SDF by s
    tensor = np.stack([
        (bin_Ap > 127).astype(np.float32),
        sdf_Ap_r.astype(np.float32) * s,
        (bin_B > 127).astype(np.float32),
        sdf_B_r.astype(np.float32) * s,
    ], axis=0)

    return torch.from_numpy(tensor), s


def _union_bbox(
    mask_a: np.ndarray, mask_b: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the union of two masks.

    Returns:
        (x0, y0, x1, y1) or None if both masks are empty.
    """
    union = ((mask_a > 0) | (mask_b > 0)).astype(np.uint8)
    ys, xs = np.where(union)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# =========================================================================
# Baseline: PointSetBaseline
# =========================================================================

class PointSetBaseline(nn.Module):
    """Baseline architecture: standard transformer on the combined point set.

    Uses only point coordinates (via Fourier PE), the side indicator
    (∂A' vs ∂B), and the out-of-view / boundary-of-view flag.
    No SDF or tangent features are used, making this a clean ablation
    of what positional information alone can predict.

    The architecture mirrors BoundaryTransformer in depth and width
    (same d_model, n_heads, n_layers) for a fair comparison, but replaces
    bipartite attention with standard self-attention on the concatenated
    2N-point sequence.

    Input feature layout per point (2 + 4*K = 18 for K=4):
        [side_indicator, oov_or_boundary_flag, fourier_PE(x, y)]
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_fourier_bands: int = 4,
    ):
        super().__init__()
        self.input_dim = 2 + 4 * n_fourier_bands  # side + oov + Fourier PE

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,  # pre-LN, consistent with BoundaryTransformer
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.rmse_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        nn.init.zeros_(self.sigma_head[-1].bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: (B, 2N, C) combined point features from
                build_point_baseline_features().

        Returns:
            (predicted_rmse, log_sigma) each (B,).
        """
        x = self.input_proj(features)  # (B, 2N, d)
        x = self.encoder(x)            # (B, 2N, d)
        v = x.mean(dim=1)              # (B, d)

        raw_rmse = self.rmse_head(v)
        predicted_rmse = F.softplus(raw_rmse)
        log_sigma = self.sigma_head(v).clamp(-5, 5)

        return predicted_rmse.squeeze(-1), log_sigma.squeeze(-1)

