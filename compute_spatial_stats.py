#!/usr/bin/env python3
"""Post-process loma_v2 NPZ results to compute per-frame spatial coverage stats.

Reads existing ``*_stats.json`` and ``*.npz`` files from a loma_v2 results
directory, computes spatial coverage metrics for each frame, and writes
``*_spatial_stats.json`` files alongside them.  These files are picked up by
``threshold_ensemble.py`` when spatial metrics are selected.

Metrics written per frame:
- ``hull_coverage``: convex hull area of inlier pts / init-mask hull area
- ``hull_stability``: hull area after removing hull-vertex pts / hull area before
- ``spatial_conditioning``: eigenvalue ratio of inlier covariance (0=collinear, 1=isotropic)

Usage::

    python compute_spatial_stats.py <results_dir> --dataset pot \\
        [--masks-dir <dir>] [--seq SEQ [SEQ ...]] [--force]
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_load_gt,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.path_config import paths
from spatial_coverage import hull_coverage, peeled_hull_coverage, sample_mask_pts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_stats(stats_path: Path) -> Optional[dict]:
    if not stats_path.exists():
        return None
    try:
        return json.loads(stats_path.read_text())
    except Exception:
        return None


def _load_gt_corners(dataset: str, base_dir: Path, seq_name: str, initfix_dir: Optional[Path]) -> Optional[np.ndarray]:
    """Load GT corners for frame 0 only.  Returns ``(2, 4)`` or None."""
    try:
        loader = DATASETS[dataset].gt_loader
        if loader == "pot":
            from flatsam2.datasets import pot_load_gt
            gt_corners, _ = pot_load_gt(base_dir, seq_name, woft_compat=True)
        elif loader == "pt":
            from flatsam2.datasets import pt_load_gt
            gt_corners, _ = pt_load_gt(base_dir, seq_name)
        else:
            if initfix_dir is None:
                return None
            from flatsam2.datasets import pti_load_gt
            gt_corners, _ = pti_load_gt(base_dir, seq_name, initfix_dir)
        return gt_corners[0]  # (2, 4)
    except Exception:
        return None


def _reconstruct_inlier_mask(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    init_corners: np.ndarray,
    pred_corners_i: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Approximate inlier mask from predicted corners + matched pairs.

    Recovers H exactly from 4 init→pred corner correspondences, then
    thresholds reprojection error.

    Args:
        pts_a: Matched init-frame keypoints ``(M, 2)``.
        pts_b: Matched current-frame keypoints ``(M, 2)``.
        init_corners: GT init corners ``(2, 4)``.
        pred_corners_i: Predicted corners frame i ``(2, 4)``.
        threshold_px: RANSAC threshold in pixels.

    Returns:
        Bool inlier mask ``(M,)``.
    """
    if len(pts_a) == 0:
        return np.zeros(0, dtype=bool)

    src = init_corners.T.astype(np.float32)    # (4, 2)
    dst = pred_corners_i.T.astype(np.float32)  # (4, 2)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        return np.zeros(len(pts_a), dtype=bool)

    pts_h = np.hstack([pts_a, np.ones((len(pts_a), 1), dtype=np.float64)])
    proj = (H @ pts_h.T).T
    proj /= proj[:, 2:3]
    err = np.linalg.norm(proj[:, :2] - pts_b, axis=1)
    return err < threshold_px


def _load_pred_corners(txt_path: Path) -> Optional[np.ndarray]:
    """Load predicted corners ``(F, 2, 4)`` from a .txt file."""
    if not txt_path.exists():
        return None
    try:
        flat = np.loadtxt(txt_path)
        if flat.ndim == 1:
            flat = flat[np.newaxis, :]
        from einops import rearrange
        return rearrange(flat, "f (c xy) -> f xy c", c=4, xy=2)
    except Exception:
        return None


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------


def compute_sequence_spatial_stats(
    seq_name: str,
    results_dir: Path,
    masks_dir: Optional[Path],
    dataset: str,
    base_dir: Path,
    initfix_dir: Optional[Path],
) -> dict:
    """Compute spatial stats for one sequence.

    Args:
        seq_name: Sequence name.
        results_dir: Directory containing NPZ / stats / txt for this dataset.
        masks_dir: Optional base masks directory (for init mask).
        dataset: Dataset CLI key.
        base_dir: Dataset root.
        initfix_dir: Initfix dir for PTI datasets.

    Returns:
        Dict with ``"frames"`` list and ``"ref_source"`` string, ready to
        be written as ``*_spatial_stats.json``.
    """
    npz_path = results_dir / f"{seq_name}.npz"
    stats_path = results_dir / f"{seq_name}_stats.json"
    txt_path = results_dir / f"{seq_name}.txt"

    if not npz_path.exists():
        raise ValueError(f"NPZ not found: {npz_path}")

    stats = _load_stats(stats_path)
    ransac_threshold = float((stats or {}).get("ransac_threshold_px", 3.0))

    data = np.load(npz_path)

    # Collect frame indices from NPZ keys.
    frame_indices = sorted(
        int(k.split("_")[-1])
        for k in data.files
        if k.startswith("kptsA_")
    )

    # Init mask → ref_pts (precomputed once per sequence).
    ref_pts: Optional[np.ndarray] = None
    ref_source = "none"
    if masks_dir is not None:
        mask_path = masks_dir / seq_name / "00001.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                ref_pts = sample_mask_pts(mask, stride=8)
                ref_source = "init_mask"
    if ref_pts is None and "kpts0" in data:
        ref_pts = data["kpts0"].astype(np.float64)
        ref_source = "kpts0"

    # For approximate inlier fallback: need init corners + predicted corners.
    init_corners = _load_gt_corners(dataset, base_dir, seq_name, initfix_dir)
    pred_corners = _load_pred_corners(txt_path)

    frames_out = []
    for i in frame_indices:
        frame_key = f"{i:05d}"
        pts_a = data[f"kptsA_{frame_key}"].astype(np.float64)

        inlier_key = f"inlier_mask_{frame_key}"
        if inlier_key in data:
            inlier_mask = data[inlier_key].astype(bool)
            source = "saved"
        elif init_corners is not None and pred_corners is not None and i < len(pred_corners):
            pts_b = data[f"kptsB_{frame_key}"].astype(np.float64)
            inlier_mask = _reconstruct_inlier_mask(
                pts_a, pts_b, init_corners, pred_corners[i], ransac_threshold
            )
            source = "reconstructed"
        else:
            inlier_mask = np.ones(len(pts_a), dtype=bool)
            source = "all_matched"

        inlier_pts = pts_a[inlier_mask]

        frames_out.append({
            "frame_idx": i,
            "inlier_source": source,
            "n_inliers_used": int(inlier_mask.sum()),
            "hull_coverage": hull_coverage(inlier_pts, ref_pts) if ref_pts is not None else None,
            "peeled_hull_coverage": peeled_hull_coverage(inlier_pts, ref_pts) if ref_pts is not None else None,
        })

    return {
        "sequence_name": seq_name,
        "ref_source": ref_source,
        "frames": frames_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-frame spatial coverage stats from loma_v2 NPZ results."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results directory containing <seq>.npz files (dataset-level, e.g. .../loma_r_2000/POT-210/)",
    )
    parser.add_argument(
        "--dataset",
        choices=CLI_CHOICES,
        required=True,
        help="Dataset key (used for GT loading and sequence discovery).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Dataset base directory (uses path_config default if omitted).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Base masks directory containing <seq>/00001.png init masks.",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="Initfix directory for PTI datasets.",
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        help="Specific sequences to process (default: all with NPZ files).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing _spatial_stats.json files.",
    )
    args = parser.parse_args(argv)

    args.results_dir = args.results_dir.expanduser()
    if not args.results_dir.exists():
        print(f"ERROR: results_dir not found: {args.results_dir}")
        return 1

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or Path(getattr(paths, ds_info.base_dir_attr))
    base_dir = base_dir.expanduser()

    initfix_dir = args.initfix_dir
    if ds_info.gt_loader == "pti" and initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR

    # Discover sequences: either from --seq or all NPZ files in results_dir.
    if args.seq:
        sequences = args.seq
    else:
        sequences = sorted(p.stem for p in args.results_dir.glob("*.npz"))
    if not sequences:
        print("No sequences found.")
        return 1

    print(f"Dataset:     {args.dataset} ({ds_info.label})")
    print(f"Results dir: {args.results_dir}")
    print(f"Masks dir:   {args.masks_dir or '(none — will use kpts0 as ref)'}")
    print(f"Sequences:   {len(sequences)}")

    n_ok = n_skip = n_fail = 0
    for seq_name in sequences:
        out_path = args.results_dir / f"{seq_name}_spatial_stats.json"
        if out_path.exists() and not args.force:
            n_skip += 1
            continue
        try:
            result = compute_sequence_spatial_stats(
                seq_name=seq_name,
                results_dir=args.results_dir,
                masks_dir=args.masks_dir,
                dataset=args.dataset,
                base_dir=base_dir,
                initfix_dir=initfix_dir,
            )
            _save_json(out_path, result)
            n_ok += 1
            ref = result["ref_source"]
            n_frames = len(result["frames"])
            print(f"  {seq_name}: {n_frames} frames, ref={ref}")
        except Exception as exc:
            n_fail += 1
            print(f"  ERROR {seq_name}: {exc}")
            print(traceback.format_exc())

    print(f"\nDone: {n_ok} written, {n_skip} skipped, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
