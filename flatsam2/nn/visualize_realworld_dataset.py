"""Visualize sampled real-world dataset entries per RMSE bin.

Usage example:

    python -m flatsam2.nn.visualize_realworld_dataset \
        /scratch/datasets/realworld_val \
        --n-per-bin 2

This reads ``samples.npz`` and ``sequences.json`` from the collected dataset,
selects a small number of examples from each RMSE bin, and writes one image per
sample to ``<dataset_dir>/vis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from ..datasets import warp_corners
from .collect_realworld_dataset import _DEFAULT_BIN_EDGES, _format_bin_label


def _assign_bin_index(rmse: float, bin_edges: np.ndarray) -> int:
    """Map an RMSE value to a valid bin index.

    Args:
        rmse: Scalar RMSE value.
        bin_edges: Sorted RMSE bin edges with shape (num_bins + 1,).

    Returns:
        Bin index in ``[0, num_bins - 1]``.
    """
    n_bins = len(bin_edges) - 1
    bin_idx = int(np.digitize(rmse, bin_edges) - 1)
    return int(np.clip(bin_idx, 0, n_bins - 1))


def _sanitize_bin_label(label: str) -> str:
    """Convert a human-readable bin label into a filename-safe prefix.

    Args:
        label: Bin label such as ``[0, 1)``.

    Returns:
        Filename-safe label.
    """
    return (
        label.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(", ", "_")
        .replace(".", "p")
        .replace("inf", "inf")
    )


def _render_sample(
    mask: np.ndarray, warped_corners: np.ndarray, rmse: float
) -> np.ndarray:
    """Render one visualization image.

    Args:
        mask: Mask image with shape (H, W).
        warped_corners: Warped quadrilateral with shape (2, 4).
        rmse: Sample RMSE value.

    Returns:
        BGR visualization image with overlay.
    """
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    polygon = np.rint(warped_corners.T).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(mask.shape[:2]) / 512.0)
    thickness = max(1, int(round(font_scale * 2)))
    text = f"RMSE: {rmse:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    padding = max(6, int(round(font_scale * 8)))
    cv2.rectangle(
        vis,
        (0, 0),
        (text_width + 2 * padding, text_height + baseline + 2 * padding),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        vis,
        text,
        (padding, text_height + padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize collected real-world dataset samples by RMSE bin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory produced by collect_realworld_dataset.py.",
    )
    parser.add_argument(
        "--n-per-bin",
        type=int,
        default=2,
        help="Number of samples to visualize per RMSE bin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed used when choosing samples within each bin.",
    )
    args = parser.parse_args()

    if args.n_per_bin <= 0:
        raise ValueError(f"n_per_bin must be positive, got {args.n_per_bin}")

    samples_path = args.dataset_dir / "samples.npz"
    sequences_path = args.dataset_dir / "sequences.json"
    meta_path = args.dataset_dir / "meta.json"
    vis_dir = args.dataset_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not samples_path.exists():
        raise ValueError(f"Dataset samples file not found: {samples_path}")
    if not sequences_path.exists():
        raise ValueError(f"Dataset sequence file not found: {sequences_path}")

    bin_edges = _DEFAULT_BIN_EDGES.astype(np.float64)
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "collection_bin_edges" in meta:
            bin_edges = np.asarray(meta["collection_bin_edges"], dtype=np.float64)

    with open(sequences_path, "r") as f:
        sequences: List[dict] = json.load(f)

    with np.load(samples_path, allow_pickle=False) as samples:
        seq_ids = samples["seq_ids"]
        frame_indices = samples["frame_indices"]
        homographies = samples["homographies"]
        gt_rmse = samples["gt_rmse"]

    samples_by_bin: Dict[int, List[int]] = {}
    for sample_idx, rmse in enumerate(gt_rmse):
        bin_idx = _assign_bin_index(float(rmse), bin_edges)
        samples_by_bin.setdefault(bin_idx, []).append(sample_idx)

    rng = np.random.default_rng(args.seed)
    written = 0
    for bin_idx in range(len(bin_edges) - 1):
        sample_indices = samples_by_bin.get(bin_idx, [])
        if not sample_indices:
            continue

        chosen = rng.permutation(sample_indices)[: args.n_per_bin]
        label = _format_bin_label(bin_edges[bin_idx], bin_edges[bin_idx + 1])
        safe_label = _sanitize_bin_label(label)

        for sample_idx in chosen:
            seq_info = sequences[int(seq_ids[sample_idx])]
            seq_name = str(seq_info["name"])
            init_corners = np.asarray(seq_info["init_corners"], dtype=np.float64)
            frame_idx = int(frame_indices[sample_idx])
            mask_path = (
                args.dataset_dir / "masks" / seq_name / f"{frame_idx + 1:05d}.png"
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Skipping sample {sample_idx}: failed to read mask {mask_path}")
                continue

            warped_corners = warp_corners(homographies[sample_idx], init_corners)
            if not np.all(np.isfinite(warped_corners)):
                print(f"Skipping sample {sample_idx}: warped corners are not finite")
                continue

            vis = _render_sample(mask, warped_corners, float(gt_rmse[sample_idx]))
            out_path = vis_dir / (
                f"{safe_label}__idx_{int(sample_idx):06d}__{seq_name}"
                f"__f{frame_idx + 1:05d}.png"
            )
            ok = cv2.imwrite(str(out_path), vis)
            if not ok:
                print(f"Failed to write visualization: {out_path}")
                continue

            written += 1

    print(f"Wrote {written} visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
