"""Masked LoMa homography tracker.

Wraps the LoMa local feature matcher with mask-constrained keypoint sampling:
  - Frame-0 features cached once; reused for every subsequent frame.
  - Keypoints restricted to the target mask (grid-cell topk inside bbox).
  - Optional global background budget for matcher calibration.

This module is intentionally zero-modification to the LoMa source and imports
it as an installed package (``loma``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from .config import SamplingCfg
from .profiling import profiler


ImageInput = np.ndarray | torch.Tensor


# ---------------------------------------------------------------------------
# Features container
# ---------------------------------------------------------------------------


@dataclass
class Features:
    """Keypoints and descriptors for one frame.

    Attributes:
        kpts: Normalized keypoints (1, N, 2) in [-1, 1].
        desc: Descriptors (1, N, D).
        h_internal: Frame height at DaD-rescaled resolution.
        w_internal: Frame width at DaD-rescaled resolution.
        h_original: Frame height at original resolution.
        w_original: Frame width at original resolution.
        is_object: Optional bool labels (N,) for object-vs-background partition.
    """

    kpts: torch.Tensor
    desc: torch.Tensor
    h_internal: int
    w_internal: int
    h_original: int
    w_original: int
    is_object: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _to_bool_tensor(
    mask: np.ndarray | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert a mask (np.ndarray or tensor) to a bool tensor on *device*.

    Args:
        mask: Binary mask — uint8 (0/255 or 0/1) or bool.
        device: Target device.

    Returns:
        Boolean tensor (H, W).
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask.copy())
    if mask.dtype != torch.bool:
        mask = mask > 0
    return mask.to(device)


def _array_to_rgb_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR/RGB array to RGB PIL image.

    Args:
        image: Image array `(H, W, 3)` or `(H, W, 4)`, usually OpenCV BGR.

    Returns:
        RGB PIL image.

    Raises:
        ValueError: If array shape is unsupported.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected image shape (H, W, 3|4), got {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    rgb = image[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")


def _pil_to_loma_tensor(image: Image.Image, size_wh: tuple[int, int]) -> torch.Tensor:
    """Convert RGB PIL image to LoMa tensor `(1, 3, H, W)` in `[0, 1]`."""
    resized = image.resize(size_wh)
    array = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)
    return rearrange(tensor, "h w c -> 1 c h w").float()


def _downsample_mask(mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour downsample a bool mask to *target_hw*.

    Args:
        mask: Bool tensor (H, W).
        target_hw: Target (height, width).

    Returns:
        Bool tensor at target_hw.
    """
    m = mask.float()[None, None]
    m = F.interpolate(m, size=target_hw, mode="nearest")[0, 0]
    return m > 0.5


def _apply_nms(probs_2d: torch.Tensor, nms_size: int) -> torch.Tensor:
    """Suppress non-maxima in a 2-D probability map.

    Args:
        probs_2d: Dense probability map (H, W).
        nms_size: Window size; 1 disables NMS.

    Returns:
        Suppressed map (H, W) — zeros everywhere except local maxima.
    """
    if nms_size <= 1:
        return probs_2d
    pooled = F.max_pool2d(
        probs_2d[None, None],
        nms_size,
        stride=1,
        padding=nms_size // 2,
    )[0, 0]
    return probs_2d * (probs_2d == pooled)


def _flat_inds_to_normalized(inds: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert flat pixel indices to normalized [-1, 1] coordinates.

    Uses pixel-center convention (matching DaD / grid_sample with align_corners=False).

    Args:
        inds: Flat indices (M,).
        H: Image height.
        W: Image width.

    Returns:
        Normalized (x, y) coordinates (M, 2).
    """
    ys = (inds // W).float()
    xs = (inds % W).float()
    x_n = 2.0 * (xs + 0.5) / W - 1.0
    y_n = 2.0 * (ys + 0.5) / H - 1.0
    return torch.stack([x_n, y_n], dim=-1)


def _topk_global_nms(
    probs_2d: torch.Tensor,
    mask: torch.Tensor,
    n_object: int,
    total_budget: int,
    nms_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select keypoints via a single global NMS pass, partitioned by mask.

    Applies NMS once over the full score map, splits survivors into object
    (within mask) and background (outside mask).  Object gets up to n_object
    highest-scoring survivors; background fills the remainder of total_budget.

    Args:
        probs_2d: Dense probability map (H, W).
        mask: Bool mask (H, W) — True for object pixels.
        n_object: Maximum object keypoints to select.
        total_budget: Total budget; background gets total_budget - actual_object_count.
        nms_size: NMS window size (passed to _apply_nms).

    Returns:
        obj_inds: Flat pixel indices (K,) for object keypoints, K <= n_object.
        bg_inds: Flat pixel indices (M,) for background, M <= total_budget - K.
    """
    with profiler.section("nms"):
        nms_probs = _apply_nms(probs_2d, nms_size)
    flat = nms_probs.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Object: NMS survivors within mask, top n_object by score.
    obj_probs = flat * mask_flat.float()
    obj_survivors = torch.where(obj_probs > 0)[0]
    if obj_survivors.numel() > 0:
        order = torch.argsort(obj_probs[obj_survivors], descending=True)
        obj_inds = obj_survivors[order[:n_object]]
    else:
        obj_inds = torch.empty(0, dtype=torch.long, device=probs_2d.device)

    # Background: NMS survivors outside mask, fills remainder of total_budget.
    bg_budget = total_budget - obj_inds.numel()
    bg_probs = flat * (~mask_flat).float()
    bg_survivors = torch.where(bg_probs > 0)[0]
    if bg_survivors.numel() > 0 and bg_budget > 0:
        order = torch.argsort(bg_probs[bg_survivors], descending=True)
        bg_inds = bg_survivors[order[:bg_budget]]
    else:
        bg_inds = torch.empty(0, dtype=torch.long, device=probs_2d.device)

    return obj_inds, bg_inds


def _apply_subpixel(
    kpts_norm: torch.Tensor,
    flat_inds: torch.Tensor,
    scoremap: torch.Tensor,
    nms_size: int,
    temp: float,
) -> torch.Tensor:
    """Apply sub-pixel refinement to keypoint coordinates.

    Extracts a nms_size×nms_size patch from the raw scoremap around each
    keypoint, applies softmax, and computes a weighted centroid offset in
    normalized coordinate space.

    Args:
        kpts_norm: Base normalized keypoints (M, 2).
        flat_inds: Flat pixel indices (M,) matching kpts_norm.
        scoremap: Raw (pre-softmax) DaD scoremap (H, W).
        nms_size: Patch size (odd integer).
        temp: Softmax temperature.

    Returns:
        Refined normalized keypoints (M, 2).
    """
    if kpts_norm.numel() == 0:
        return kpts_norm

    H, W = scoremap.shape
    radius = nms_size // 2
    M = kpts_norm.shape[0]

    # Pad so every pixel has a full nms_size window.
    padded = F.pad(
        scoremap[None, None], [radius, radius, radius, radius], mode="replicate"
    )[0, 0]
    # Shape: (H + 2*radius, W + 2*radius)
    # Pixel (y, x) in original → slice pad[y : y+nms_size, x : x+nms_size]

    ys = (flat_inds // W).long()
    xs = (flat_inds % W).long()

    # Extract patches (M, nms_size, nms_size). Vectorized: unfold builds a
    # sliding-window view (no copy), then advanced-index the window at each
    # keypoint pixel. Replaces a Python loop over up to ~4096 keypoints — one
    # synced GPU slice per point, run twice per frame — which was the dominant
    # hidden cost inside loma.keypoints. Numerically identical: unfold window
    # (y, x) == padded[y : y+nms_size, x : x+nms_size].
    with profiler.section("subpixel_patches"):
        windows = padded.unfold(0, nms_size, 1).unfold(1, nms_size, 1)
        patches = windows[ys, xs]  # (M, nms_size, nms_size)

    # Softmax weights per patch.
    w = torch.softmax(patches.reshape(M, -1) * temp, dim=1).reshape(
        M, nms_size, nms_size
    )

    # Local offsets from patch center in normalized coord units.
    offsets = (
        torch.arange(nms_size, device=scoremap.device).float() - radius
    )  # e.g. [-1, 0, 1]
    dy = offsets * (2.0 / H)  # (nms_size,)
    dx = offsets * (2.0 / W)  # (nms_size,)

    # Weighted centroid offset (marginal distributions).
    delta_y = (w.sum(dim=2) * dy[None]).sum(dim=1)  # (M,)
    delta_x = (w.sum(dim=1) * dx[None]).sum(dim=1)  # (M,)

    refined = kpts_norm.clone()
    refined[:, 0] = refined[:, 0] + delta_x
    refined[:, 1] = refined[:, 1] + delta_y
    return refined


# ---------------------------------------------------------------------------
# LoMaTracker
# ---------------------------------------------------------------------------


class LoMaTracker:
    """Thin wrapper around a LoMa model that adds masked keypoint sampling.

    Attributes:
        model: The underlying LoMa matcher (owns detector + descriptor).
    """

    def __init__(self, model: object) -> None:
        """Create a tracker.

        Args:
            model: Loaded ``LoMa`` instance (owns ``_detector``, ``_descriptor``).
        """
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        return next(iter(self.model.parameters())).device

    def _prepare_images(
        self,
        img: ImageInput,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Prepare detector and descriptor tensors without disk access for arrays.

        Args:
            img: OpenCV BGR array `(H, W, C)`, or RGB tensor `(1, 3, H, W)`.

        Returns:
            Tuple `(detector_image, descriptor_image, H_original, W_original)`.
        """
        det = self.model._detector
        device = self._device()

        if isinstance(img, np.ndarray):
            pil_image = _array_to_rgb_pil(img)
            W0, H0 = pil_image.size
            if det.keep_aspect_ratio:
                scale = det.resize / max(W0, H0)
                W_det = int((scale * W0) // 8 * 8)
                H_det = int((scale * H0) // 8 * 8)
            else:
                H_det, W_det = det.resize, det.resize
            det_image = _pil_to_loma_tensor(pil_image, (W_det, H_det)).to(device)
            desc_image = _pil_to_loma_tensor(pil_image, (784, 784)).to(device)
            return det_image, desc_image, H0, W0

        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"Expected tensor shape (1, 3, H, W), got {img.shape}")
        img = img.to(device)
        H0, W0 = img.shape[-2:]
        if det.keep_aspect_ratio:
            scale = det.resize / max(W0, H0)
            W_det = int((scale * W0) // 8 * 8)
            H_det = int((scale * H0) // 8 * 8)
        else:
            H_det, W_det = det.resize, det.resize
        det_image = F.interpolate(
            img,
            size=(H_det, W_det),
            mode="bilinear",
            align_corners=False,
        )
        desc_image = F.interpolate(
            img,
            size=(784, 784),
            mode="bilinear",
            align_corners=False,
        )
        return det_image, desc_image, H0, W0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def get_keypoints(
        self,
        img: ImageInput,
        mask: Optional[np.ndarray | torch.Tensor] = None,
        cfg: SamplingCfg = SamplingCfg(),
    ) -> Features:
        """Detect and describe keypoints, optionally restricted to *mask*.

        Mirrors LoMa path mode in memory: detector input uses DaD's aspect-ratio
        resize, descriptor input uses its default 784x784 resolution.

        Args:
            img: OpenCV BGR array `(H, W, C)`, or RGB tensor `(1, 3, H, W)`.
            mask: Binary target mask at original image resolution; or None.
            cfg: Sampling configuration.

        Returns:
            Features for this frame.
        """
        det = self.model._detector
        with profiler.section("prep"):
            det_image, desc_image, H0, W0 = self._prepare_images(img)

        if mask is None:
            kpts = det.detect(det_image, num_keypoints=self.model.cfg.num_keypoints)[
                "keypoints"
            ]
            desc = self.model._descriptor.describe_keypoints(desc_image, kpts)[
                "descriptions"
            ]
            return Features(
                kpts=kpts,
                desc=desc,
                h_internal=det_image.shape[-2],
                w_internal=det_image.shape[-1],
                h_original=H0,
                w_original=W0,
                is_object=None,
            )

        _, _, H, W = det_image.shape

        # --- Masked path ---------------------------------------------------

        # 1. Downsample mask to detector scoremap resolution.
        with profiler.section("mask_ds"):
            mask_bool = _to_bool_tensor(mask, self._device())
            mask_ds = _downsample_mask(mask_bool, (H, W))  # (H, W) bool

        # 2. DaD scoremap (encoder + decoder, no sampling yet).
        with profiler.section("detect"):
            scoremap = det.forward_impl(det_image)  # (1, 1, H, W)
        scoremap_2d = scoremap[0, 0]  # (H, W)

        # 3. Dense probabilities (global softmax, mirrors DaD.forward).
        with profiler.section("softmax"):
            dense_probs = scoremap_2d.reshape(-1).softmax(dim=0).reshape(H, W)

        # 4. Select target keypoints via global NMS.
        target_inds, bg_flat_inds = _topk_global_nms(
            dense_probs,
            mask_ds,
            n_object=cfg.n_object,
            total_budget=cfg.total_budget,
            nms_size=cfg.nms_size,
        )

        # 6. Background budget — topk outside mask.
        bg_kpts: Optional[torch.Tensor] = None
        if bg_flat_inds.numel() > 0:
            bg_norm = _flat_inds_to_normalized(bg_flat_inds, H, W)
            if cfg.subpixel:
                bg_norm = _apply_subpixel(
                    bg_norm, bg_flat_inds, scoremap_2d, cfg.nms_size, cfg.subpixel_temp
                )
            bg_kpts = bg_norm[None]  # (1, n_bg, 2)

        # 7. Target: flat indices → normalized, optional subpixel.
        if target_inds.numel() > 0:
            target_norm = _flat_inds_to_normalized(target_inds, H, W)  # (M, 2)
            if cfg.subpixel:
                target_norm = _apply_subpixel(
                    target_norm,
                    target_inds,
                    scoremap_2d,
                    cfg.nms_size,
                    cfg.subpixel_temp,
                )
            target_kpts = target_norm[None]  # (1, M, 2)
        else:
            target_kpts = torch.zeros(1, 0, 2, device=self._device())

        # 8. Concatenate target + background.
        if bg_kpts is not None and bg_kpts.shape[1] > 0:
            kpts = torch.cat([target_kpts, bg_kpts], dim=1)
            is_object = torch.cat(
                [
                    torch.ones(
                        target_kpts.shape[1], dtype=torch.bool, device=self._device()
                    ),
                    torch.zeros(
                        bg_kpts.shape[1], dtype=torch.bool, device=self._device()
                    ),
                ]
            )
        else:
            kpts = target_kpts
            is_object = torch.ones(
                target_kpts.shape[1], dtype=torch.bool, device=self._device()
            )

        # 9. Describe via LoMa's descriptor at the same 784x784 resolution as path mode.
        with profiler.section("describe"):
            desc = self.model._descriptor.describe_keypoints(desc_image, kpts)[
                "descriptions"
            ]

        return Features(
            kpts=kpts,
            desc=desc,
            h_internal=H,
            w_internal=W,
            h_original=H0,
            w_original=W0,
            is_object=is_object,
        )

    def _match_impl(
        self,
        feats_a: Features,
        feats_b: Features,
        filter_threshold: Optional[float] = None,
        return_resolution: str = "original",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match two feature sets and return points plus matched feature indices.

        Args:
            feats_a: Features from frame A (e.g. template).
            feats_b: Features from frame B (e.g. current frame).
            filter_threshold: Override model's default mutual-NN threshold.
            return_resolution: 'original' (default) or 'internal'.

        Returns:
            pts_a: `(N, 2)` pixel coordinates in frame A.
            pts_b: `(N, 2)` pixel coordinates in frame B.
            idx_a: `(N,)` matched keypoint indices into ``feats_a``.
            idx_b: `(N,)` matched keypoint indices into ``feats_b``.
        """
        from loma.loma import filter_matches, to_pixel_coords

        if filter_threshold is None:
            filter_threshold = self.model.cfg.filter_threshold

        n_a = feats_a.kpts.shape[1]
        n_b = feats_b.kpts.shape[1]
        if n_a == 0 or n_b == 0:
            empty_pts = np.empty((0, 2), dtype=np.float64)
            empty_idx = np.empty(0, dtype=np.int64)
            return empty_pts, empty_pts.copy(), empty_idx, empty_idx.copy()

        with profiler.section("score"):
            scores = self.model(feats_a.kpts, feats_b.kpts, feats_a.desc, feats_b.desc)[
                "scores"
            ]
        m0, *_ = filter_matches(scores, filter_threshold)
        valid = m0[0] > -1
        valid_inds = torch.where(valid)[0]

        idx_a = valid_inds.cpu().numpy().astype(np.int64)
        idx_b = m0[0][valid].cpu().numpy().astype(np.int64)
        kpts_a = feats_a.kpts[0][valid_inds]
        kpts_b = feats_b.kpts[0][m0[0][valid]]

        if return_resolution == "internal":
            ha, wa = feats_a.h_internal, feats_a.w_internal
            hb, wb = feats_b.h_internal, feats_b.w_internal
        else:
            ha, wa = feats_a.h_original, feats_a.w_original
            hb, wb = feats_b.h_original, feats_b.w_original

        pts_a = to_pixel_coords(kpts_a, ha, wa).cpu().numpy().astype(np.float64)
        pts_b = to_pixel_coords(kpts_b, hb, wb).cpu().numpy().astype(np.float64)
        return pts_a, pts_b, idx_a, idx_b

    @torch.inference_mode()
    def match(
        self,
        feats_a: Features,
        feats_b: Features,
        filter_threshold: Optional[float] = None,
        return_resolution: str = "original",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match two feature sets and return pixel-space correspondences.

        Args:
            feats_a: Features from frame A (e.g. template).
            feats_b: Features from frame B (e.g. current frame).
            filter_threshold: Override model's default mutual-NN threshold.
            return_resolution: 'original' (default) or 'internal'.

        Returns:
            pts_a: (N, 2) float64 pixel coordinates in frame A.
            pts_b: (N, 2) float64 pixel coordinates in frame B.
        """
        pts_a, pts_b, _, _ = self._match_impl(
            feats_a,
            feats_b,
            filter_threshold=filter_threshold,
            return_resolution=return_resolution,
        )
        return pts_a, pts_b

    @torch.inference_mode()
    def match_with_indices(
        self,
        feats_a: Features,
        feats_b: Features,
        filter_threshold: Optional[float] = None,
        return_resolution: str = "original",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match two feature sets and also return matched keypoint indices.

        Args:
            feats_a: Features from frame A (e.g. template).
            feats_b: Features from frame B (e.g. current frame).
            filter_threshold: Override model's default mutual-NN threshold.
            return_resolution: 'original' (default) or 'internal'.

        Returns:
            pts_a: `(N, 2)` pixel coordinates in frame A.
            pts_b: `(N, 2)` pixel coordinates in frame B.
            idx_a: `(N,)` matched keypoint indices into ``feats_a``.
            idx_b: `(N,)` matched keypoint indices into ``feats_b``.
        """
        return self._match_impl(
            feats_a,
            feats_b,
            filter_threshold=filter_threshold,
            return_resolution=return_resolution,
        )
