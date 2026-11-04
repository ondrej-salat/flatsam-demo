#!/usr/bin/env python3
"""Masked LoMa feature tracking experiment (v2).

Improved over loma_experiment.py:
  - Keypoints restricted to target mask (grid-cell topk, one per bbox cell).
  - Frame-0 features cached; not re-computed each frame.
  - Pre-computed SAM masks loaded from disk when available.
  - Saves both NPZ correspondences (debug) and TXT predicted corners (eval).
  - Restart-safe: skips sequences where output files are already complete.

Pre-computed masks are expected at:
    <masks-dir>/<seq>/{frame_idx + 1:05d}.png
(i.e. the ``masks/`` sub-directory produced by ``run.py --save-masks``).

Output layout:
    <output-dir>/<config-stem>/<dataset-label>/<seq>.txt   — predicted corners
    <output-dir>/<config-stem>/<dataset-label>/<seq>.npz   — per-frame correspondences
    <output-dir>/<config-stem>/<dataset-label>/<seq>_stats.json
        — per-frame keypoint/match/inlier statistics
    <output-dir>/<config-stem>/<dataset-label>/keypoint_stats_report.html
        — aggregate sequence-level histogram report

Usage examples:
    # POT-210, LoMaB, with precomputed SAM masks from a prior run:
    python experiments/loma_experiment_v2.py pot \\
        --config configs/loma/loma_b_default.py \\
        --precomputed-masks-dir /mnt/.../results/fast_kitti/POT-210/masks \\
        --output-dir /mnt/.../loma_v2/

    # Dry-run on a single sequence:
    python experiments/loma_experiment_v2.py pot \\
        --config configs/loma/loma_b_default.py \\
        --precomputed-masks-dir /mnt/.../masks \\
        --output-dir /mnt/.../loma_v2/ \\
        --sequences seq_name

    # SLURM subset (passed automatically by submit_loma_v2.py):
    python experiments/loma_experiment_v2.py pot \\
        --config configs/loma/loma_b_default.py \\
        --precomputed-masks-dir /mnt/.../masks \\
        --output-dir /mnt/.../loma_v2/ \\
        --subset-file slurm/tmp/part1.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import tqdm
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.config import LoMaTrackerConfig, load_config_from_file
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    corners_to_mask,
    get_sequences,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
    warp_corners,
)
from flatsam2.path_config import paths

from flatsam2.loma_models import LOMA_AVAILABLE as _LOMA_AVAILABLE
from flatsam2.loma_models import LOMA_MODEL_FACTORIES as _LOMA_MODEL_FACTORIES

try:
    from flatsam2.loma_tracker import LoMaTracker
except ImportError:
    _LOMA_AVAILABLE = False

_DEFAULT_OUTPUT_DIR = Path("/mnt/personal/serycjon/flatsam2/loma_v2/")

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _save_corners(corners: np.ndarray, output_path: Path) -> None:
    """Save (num_frames, 2, 4) corner array in standard 8-floats-per-line format.

    Args:
        corners: Predicted corners (num_frames, 2, 4).
        output_path: Destination .txt file.
    """
    corners_flat = rearrange(corners, "f xy c -> f (c xy)", c=4, xy=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, corners_flat, fmt="%.6f")


def _save_stats(stats: dict, output_path: Path) -> None:
    """Save per-sequence keypoint statistics as JSON.

    Args:
        stats: JSON-serializable per-sequence stats payload.
        output_path: Destination ``*_stats.json`` file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))


def _is_complete(
    npz_path: Path, txt_path: Path, stats_path: Path, num_frames: int
) -> bool:
    """Check whether all output files exist and are complete for *num_frames*.

    Args:
        npz_path: Path to the NPZ correspondences file.
        txt_path: Path to the corners .txt file.
        stats_path: Path to the keypoint-stats JSON file.
        num_frames: Total frame count including init frame.

    Returns:
        True if all outputs exist and contain the expected data.
    """
    if not txt_path.exists() or not npz_path.exists() or not stats_path.exists():
        return False
    # Check txt row count.
    try:
        data = np.loadtxt(txt_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if len(data) != num_frames:
            return False
    except Exception:
        return False
    # Check npz has entries for every non-init frame.
    try:
        npz = np.load(npz_path)
        for i in range(1, num_frames):
            if f"kptsA_{i:05d}" not in npz or f"kptsB_{i:05d}" not in npz:
                return False
    except Exception:
        return False
    try:
        stats = json.loads(stats_path.read_text())
        if int(stats.get("num_frames", -1)) != num_frames:
            return False
    except Exception:
        return False
    return True


def _count_object_background(
    is_object: Optional[torch.Tensor],
) -> Tuple[Optional[int], Optional[int]]:
    """Count object/background labels.

    Args:
        is_object: Optional bool labels `(N,)`, True for object keypoints.

    Returns:
        Tuple ``(n_object, n_background)`` or ``(None, None)`` when labels are
        unavailable.
    """
    if is_object is None:
        return None, None
    n_object = int(is_object.sum().item())
    n_total = int(is_object.numel())
    return n_object, n_total - n_object


def _count_selected_object_background(
    is_object: Optional[torch.Tensor],
    indices: np.ndarray,
) -> Tuple[Optional[int], Optional[int]]:
    """Count object/background labels for selected keypoint indices.

    Args:
        is_object: Optional bool labels `(N,)`, True for object keypoints.
        indices: Selected keypoint indices `(M,)`.

    Returns:
        Tuple ``(n_object, n_background)`` for selected labels, or
        ``(None, None)`` when labels are unavailable.
    """
    if is_object is None:
        return None, None
    if len(indices) == 0:
        return 0, 0
    selected = is_object[
        torch.as_tensor(indices, device=is_object.device, dtype=torch.long)
    ]
    return _count_object_background(selected)


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    """Mean over non-null values, or None when no values are available."""
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _summarize_sequence_stats(
    seq_name: str,
    num_frames: int,
    cfg: "LoMaTrackerConfig",
    init_detected_object: Optional[int],
    init_detected_background: Optional[int],
    frames: List[Dict[str, Optional[float]]],
) -> dict:
    """Aggregate per-frame keypoint stats into one per-sequence payload.

    Args:
        seq_name: Sequence name.
        num_frames: Total sequence length including init frame.
        cfg: Tracker config.
        init_detected_object: Init-frame detected object-keypoint count.
        init_detected_background: Init-frame detected background-keypoint count.
        frames: Per-frame stats for frames `1..T-1`.

    Returns:
        JSON-serializable per-sequence stats dict.
    """
    return {
        "sequence_name": seq_name,
        "num_frames": int(num_frames),
        "num_tracked_frames": max(int(num_frames) - 1, 0),
        "ransac_method": cfg.ransac_method,
        "ransac_threshold_px": float(cfg.ransac_threshold),
        "init_detected_object": init_detected_object,
        "init_detected_background": init_detected_background,
        "aggregate": {
            "init_detected_object": init_detected_object,
            "init_detected_background": init_detected_background,
            "num_masked_frames": int(
                sum(bool(f.get("mask_available")) for f in frames)
            ),
            "detected_current_object_mean": _mean_or_none(
                [f.get("detected_current_object") for f in frames]
            ),
            "detected_current_background_mean": _mean_or_none(
                [f.get("detected_current_background") for f in frames]
            ),
            "matched_total_mean": _mean_or_none(
                [f.get("matched_total") for f in frames]
            ),
            "matched_object_mean": _mean_or_none(
                [f.get("matched_object") for f in frames]
            ),
            "matched_background_mean": _mean_or_none(
                [f.get("matched_background") for f in frames]
            ),
            "inliers_total_mean": _mean_or_none(
                [f.get("inliers_total") for f in frames]
            ),
            "inliers_object_mean": _mean_or_none(
                [f.get("inliers_object") for f in frames]
            ),
            "inliers_background_mean": _mean_or_none(
                [f.get("inliers_background") for f in frames]
            ),
            "inlier_ratio_mean": _mean_or_none([f.get("inlier_ratio") for f in frames]),
        },
        "frames": frames,
    }


def _generate_stats_report(
    output_dir: Path,
    config_stem: str,
    dataset_label: str,
    dataset: str,
    base_dir: Path,
) -> None:
    """Generate aggregate histogram report from saved ``*_stats.json`` files."""
    experiments_dir = str(Path(__file__).resolve().parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)

    try:
        from loma_v2_stats_report import _build_html, load_stats_dir  # type: ignore[import]
    except ImportError as exc:
        print(
            f"WARNING: could not import loma_v2_stats_report — skipping report ({exc})",
            file=sys.stderr,
        )
        return

    try:
        report_data = load_stats_dir(output_dir, dataset=dataset, base_dir=base_dir)
    except ValueError as exc:
        print(f"WARNING: keypoint stats report skipped — {exc}", file=sys.stderr)
        return

    title = f"LoMa v2 keypoint stats — {config_stem} / {dataset_label}"
    report_path = output_dir / "keypoint_stats_report.html"
    report_path.write_text(_build_html(report_data, title), encoding="utf-8")
    print(f"Keypoint stats report → {report_path}")


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------


def _load_frame_mask(
    masks_dir: Optional[Path],
    seq_name: str,
    frame_idx: int,
) -> Optional[np.ndarray]:
    """Load a pre-computed mask PNG for *frame_idx* (0-based).

    Expects ``<masks-dir>/<seq>/{frame_idx + 1:05d}.png``.

    Args:
        masks_dir: Base directory containing per-sequence mask sub-dirs.
        seq_name: Sequence name.
        frame_idx: 0-based frame index.

    Returns:
        Grayscale uint8 mask (H, W), or None if not available.
    """
    if masks_dir is None:
        return None
    mask_path = masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return mask  # uint8, 0/255


def _inflate_mask(mask: Optional[np.ndarray], inflate_px: int) -> Optional[np.ndarray]:
    """Dilate binary mask by *inflate_px* pixels.

    Args:
        mask: Input uint8 mask (H, W), or None.
        inflate_px: Dilation radius in pixels (0 keeps mask unchanged).

    Returns:
        Inflated uint8 mask, or None when mask is None.
    """
    if mask is None:
        return None
    if inflate_px <= 0:
        return mask
    kernel_size = 2 * inflate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _make_gt_loader(
    dataset: str,
    initfix_dir: Optional[Path],
) -> Callable[[Path, str], Tuple[np.ndarray, np.ndarray]]:
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir=initfix_dir)


def _make_image_paths_fn(dataset: str) -> Callable[[Path, str], List[Path]]:
    if DATASETS[dataset].gt_loader == "pot":
        return pot_image_paths
    return pt_image_paths


# ---------------------------------------------------------------------------
# Core tracking
# ---------------------------------------------------------------------------


def _run_sequence(
    tracker: "LoMaTracker",
    cfg: "LoMaTrackerConfig",
    image_paths: List[Path],
    init_corners: np.ndarray,
    masks_dir: Optional[Path],
    seq_name: str,
    ransac_method: int,
) -> Tuple[np.ndarray, dict, dict]:
    """Run masked LoMa tracking on one sequence.

    When *masks_dir* is None, a SAM3StreamingTracker is created and run on
    every frame to produce segmentation masks on the fly.  When *masks_dir* is
    given, masks are loaded from disk; any frame whose mask file is missing
    falls back to unmasked (full-frame) keypoint sampling.

    Args:
        tracker: Initialised LoMaTracker.
        cfg: LoMaTrackerConfig (sampling + RANSAC params).
        image_paths: Ordered image paths; index 0 is the init frame.
        init_corners: GT corners for frame 0 (2, 4).
        masks_dir: Directory with per-frame masks, or None to run SAM3.
        seq_name: Sequence name (used for mask path lookup).
        ransac_method: cv2 method constant.

    Returns:
        corners: Predicted corners (num_frames, 2, 4).
        npz_data: Dict with kptsA/kptsB for each non-init frame,
                  plus kpts0 (frame-0 keypoints in original image pixel coords).
        stats: Per-sequence keypoint statistics payload.
    """
    num_frames = len(image_paths)
    sampling_cfg = cfg.sampling

    # --- Init frame --------------------------------------------------------
    init_frame_bgr = cv2.imread(str(image_paths[0]))
    if init_frame_bgr is None:
        raise ValueError(f"Failed to load image: {image_paths[0]}")
    H_img, W_img = init_frame_bgr.shape[:2]
    init_mask = corners_to_mask(init_corners, (H_img, W_img))  # (H, W) uint8
    init_sampling_mask = _inflate_mask(init_mask, cfg.init_mask_inflate_px)

    # --- SAM3 segmenter (only when no precomputed masks) -------------------
    segmenter = None
    if masks_dir is None:
        from sam3 import SAM3StreamingTracker

        segmenter = SAM3StreamingTracker()
        segmenter.init(init_frame_bgr, init_mask)

    # Optional exact GT mask on frame 0 — object only, zero background.
    # total_budget=0 ensures bg_budget = 0 - actual_obj_count < 0 → no bg selected.
    if cfg.exact_gt_mask_on_init:
        init_sampling_cfg = sampling_cfg.model_copy(update={"total_budget": 0})
    else:
        init_sampling_cfg = sampling_cfg
    feats_0 = tracker.get_keypoints(
        init_frame_bgr, mask=init_sampling_mask, cfg=init_sampling_cfg
    )

    all_corners = np.zeros((num_frames, 2, 4), dtype=np.float64)
    all_corners[0] = init_corners
    last_H = np.eye(3, dtype=np.float64)

    from loma.loma import to_pixel_coords as _to_px

    kpts0_px = _to_px(feats_0.kpts[0], feats_0.h_original, feats_0.w_original)
    npz_data: dict = {
        "kpts0": kpts0_px.cpu()
        .numpy()
        .astype(np.float32),  # (N, 2) original image pixel coords
    }
    init_detected_object, init_detected_background = _count_object_background(
        feats_0.is_object
    )
    frame_stats: List[Dict[str, Optional[float]]] = []

    for i in tqdm.trange(1, num_frames, desc="  frames", leave=False, unit="frame"):
        frame_bgr = cv2.imread(str(image_paths[i]))
        if frame_bgr is None:
            raise ValueError(f"Failed to load image: {image_paths[i]}")
        if segmenter is not None:
            frame_mask = segmenter.track(frame_bgr)
        else:
            frame_mask = _load_frame_mask(masks_dir, seq_name, i)
        if cfg.current_mask_sampling:
            frame_sampling_mask = _inflate_mask(frame_mask, cfg.current_mask_inflate_px)
        else:
            # No-focusing ablation: same keypoint budget, whole frame. An all-ones
            # mask keeps the budgeted sampling path (the mask=None path would fall
            # back to the detector's own keypoint count instead).
            frame_sampling_mask = np.full(frame_bgr.shape[:2], 255, dtype=np.uint8)

        feats_i = tracker.get_keypoints(
            frame_bgr, mask=frame_sampling_mask, cfg=sampling_cfg
        )
        detected_current_object, detected_current_background = _count_object_background(
            feats_i.is_object
        )

        pts_a, pts_b, idx_a, _ = tracker.match_with_indices(feats_0, feats_i)
        npz_data[f"kptsA_{i:05d}"] = pts_a.astype(np.float32)
        npz_data[f"kptsB_{i:05d}"] = pts_b.astype(np.float32)
        matched_object, matched_background = _count_selected_object_background(
            feats_0.is_object, idx_a
        )

        H: Optional[np.ndarray] = None
        inlier_mask = np.zeros(len(pts_a), dtype=bool)
        if len(pts_a) >= 4:
            H_est, ransac_inliers = cv2.findHomography(
                pts_a, pts_b, ransac_method, cfg.ransac_threshold
            )
            if H_est is not None:
                H = H_est
                last_H = H
                if ransac_inliers is not None:
                    inlier_mask = ransac_inliers.reshape(-1).astype(bool)

        if H is None:
            H = last_H

        npz_data[f"inlier_mask_{i:05d}"] = inlier_mask.astype(np.uint8)

        inlier_object, inlier_background = _count_selected_object_background(
            feats_0.is_object,
            idx_a[inlier_mask],
        )
        n_inliers = int(inlier_mask.sum())
        frame_stats.append(
            {
                "frame_idx": i,
                "mask_available": frame_mask is not None,
                "detected_current_object": detected_current_object,
                "detected_current_background": detected_current_background,
                "matched_total": int(len(pts_a)),
                "matched_object": matched_object,
                "matched_background": matched_background,
                "inliers_total": n_inliers,
                "inliers_object": inlier_object,
                "inliers_background": inlier_background,
                "inlier_ratio": float(n_inliers / len(pts_a))
                if len(pts_a) > 0
                else None,
            }
        )

        all_corners[i] = warp_corners(H, init_corners)

    return (
        all_corners,
        npz_data,
        _summarize_sequence_stats(
            seq_name,
            num_frames,
            cfg,
            init_detected_object,
            init_detected_background,
            frame_stats,
        ),
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> None:
    if not _LOMA_AVAILABLE:
        print("ERROR: loma package not installed.", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(args.config)
    raw_cfg = load_config_from_file(cfg_path)
    if not isinstance(raw_cfg, LoMaTrackerConfig):
        print(
            f"ERROR: {cfg_path} defines a {type(raw_cfg).__name__}, expected LoMaTrackerConfig.",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg: LoMaTrackerConfig = raw_cfg

    ds_info = DATASETS[args.dataset]
    if ds_info.uses_split and args.split is None:
        args.split = "test"

    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)

    config_stem = cfg_path.stem
    output_dir = args.output_dir / config_stem / ds_info.label
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir: Optional[Path] = args.precomputed_masks_dir

    sequences = get_sequences(
        args.dataset,
        base_dir,
        sequences=args.sequences,
        subset=getattr(args, "subset", None),
        subset_file=getattr(args, "subset_file", None),
        split=args.split,
    )
    if not sequences:
        print("No sequences found.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset  : {ds_info.label}")
    print(f"Config   : {config_stem}  ({cfg_path})")
    print(f"Model    : {cfg.model}")
    print(f"RANSAC   : {cfg.ransac_method}  threshold={cfg.ransac_threshold}px")
    print(
        f"Sampling : n_object={cfg.sampling.n_object}  total_budget={cfg.sampling.total_budget}  nms={cfg.sampling.nms_size}"
    )
    print(f"Init mask: exact_gt_mask_on_init={cfg.exact_gt_mask_on_init}")
    print(f"Output   : {output_dir}")
    print(f"Masks    : {masks_dir or '(none — running SAM3 segmenter per frame)'}")
    print(f"Sequences: {len(sequences)}")

    # Resolve RANSAC method constant.
    ransac_method_val = getattr(cv2, cfg.ransac_method, None)
    if ransac_method_val is None:
        print(f"ERROR: unknown RANSAC method {cfg.ransac_method!r}", file=sys.stderr)
        sys.exit(1)
    ransac_method_val = int(ransac_method_val)

    gt_loader = _make_gt_loader(args.dataset, getattr(args, "initfix_dir", None))
    image_paths_fn = _make_image_paths_fn(args.dataset)

    # Pre-scan for skip.
    needs_processing: List[Tuple[str, List[Path]]] = []
    n_skip = 0
    for seq_name in sequences:
        image_paths = image_paths_fn(base_dir, seq_name)
        txt_path = output_dir / f"{seq_name}.txt"
        npz_path = output_dir / f"{seq_name}.npz"
        stats_path = output_dir / f"{seq_name}_stats.json"
        if not args.force and _is_complete(
            npz_path, txt_path, stats_path, len(image_paths)
        ):
            n_skip += 1
        else:
            needs_processing.append((seq_name, image_paths))

    if n_skip:
        print(f"Skipping {n_skip} already-complete sequence(s).")

    # Load model once.
    print(f"Loading LoMa model {cfg.model!r}...")
    if cfg.model not in _LOMA_MODEL_FACTORIES:
        print(f"ERROR: unknown model {cfg.model!r}", file=sys.stderr)
        sys.exit(1)
    model = _LOMA_MODEL_FACTORIES[cfg.model]()
    tracker = LoMaTracker(model)

    n_missing_masks = 0
    for seq_name, image_paths in tqdm.tqdm(
        needs_processing, desc="sequences", unit="seq"
    ):
        # Warn once per sequence if masks are absent.
        if masks_dir is not None:
            seq_mask_dir = masks_dir / seq_name
            if not seq_mask_dir.is_dir():
                print(f"  WARNING: no mask dir for {seq_name}: {seq_mask_dir}")
                n_missing_masks += 1

        gt_corners, _ = gt_loader(base_dir, seq_name)
        init_corners = gt_corners[0]

        pred_corners, npz_data, stats_data = _run_sequence(
            tracker,
            cfg,
            image_paths,
            init_corners,
            masks_dir,
            seq_name,
            ransac_method_val,
        )

        txt_path = output_dir / f"{seq_name}.txt"
        npz_path = output_dir / f"{seq_name}.npz"
        stats_path = output_dir / f"{seq_name}_stats.json"
        _save_corners(pred_corners, txt_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **npz_data)
        _save_stats(stats_data, stats_path)

    if n_missing_masks:
        print(
            f"WARNING: {n_missing_masks} sequence(s) had no precomputed mask dir — "
            f"subsequent frames used full-frame (unmasked) keypoints."
        )

    _generate_stats_report(
        output_dir, config_stem, ds_info.label, args.dataset, base_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Masked LoMa feature tracking experiment (v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to process.")
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a LoMaTrackerConfig Python file (configs/loma/*.py).",
    )
    parser.add_argument(
        "--precomputed-masks-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing pre-computed SAM masks "
            "(<DIR>/<seq>/{frame+1:05d}.png, produced by run.py --save-masks). "
            "If omitted, a SAM3StreamingTracker is run on every frame to produce "
            "masks on the fly (requires sam3 to be installed)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base output directory. Results go to <output-dir>/<config-stem>/<dataset-label>/.",
    )
    parser.add_argument(
        "--base-dir", type=Path, help="Dataset root (defaults to path_config)."
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        help="Initfix dir for PTI (defaults to path_config).",
    )
    parser.add_argument(
        "--sequences", nargs="+", metavar="SEQ", help="Explicit sequence names."
    )
    parser.add_argument(
        "--subset",
        metavar="NAME",
        help="Subset file name (configs/subsets/<name>.txt).",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        metavar="PATH",
        help="Direct path to subset file (for SLURM).",
    )
    parser.add_argument(
        "--split", choices=["train", "test"], help="Train/test split (PT/PTI only)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
