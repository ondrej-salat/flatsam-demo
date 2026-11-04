#!/usr/bin/env python3
"""Precompute PTi SAM3 mask quality metrics for later plotting.

Evaluates precomputed SAM3 masks against GT quadrilateral masks on valid PTi
frames only (excluding frame 0 by default).

Metrics include region overlap and boundary alignment:
    - IoU, Dice
    - Boundary hit-rate at pixel tolerances (pred -> GT)
    - Boundary F1 at same tolerances
    - Directed/symmetric Chamfer distance stats on mask boundaries
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from flatsam2.datasets import corners_to_mask, get_sequences, pt_image_paths, pti_load_gt
from flatsam2.path_config import paths


_SEQ_CSV_HEADER = [
    "frame_idx",
    "gt_valid",
    "evaluated",
    "mask_exists",
    "iou",
    "dice",
    "boundary_hit_t5",
    "boundary_hit_t10",
    "boundary_hit_t15",
    "boundary_hit_t20",
    "boundary_hit_t25",
    "boundary_hit_t30",
    "boundary_f1_t5",
    "boundary_f1_t10",
    "boundary_f1_t15",
    "boundary_f1_t20",
    "boundary_f1_t25",
    "boundary_f1_t30",
    "chamfer_mean_px",
    "chamfer_p95_px",
    "chamfer_max_px",
    "area_gt_px",
    "area_pred_px",
]


@dataclass
class FrameMetrics:
    """Per-frame metric record.

    Args:
        frame_idx: Frame index.
        gt_valid: Whether GT marks frame valid.
        evaluated: Whether frame is part of metric evaluation.
        mask_exists: Whether SAM mask exists and could be loaded.
        iou: Intersection-over-union.
        dice: Dice score.
        boundary_hit_t5: Fraction of predicted boundary points within 5 px of GT boundary.
        boundary_hit_t10: Fraction of predicted boundary points within 10 px of GT boundary.
        boundary_hit_t15: Fraction of predicted boundary points within 15 px of GT boundary.
        boundary_hit_t20: Fraction of predicted boundary points within 20 px of GT boundary.
        boundary_hit_t25: Fraction of predicted boundary points within 25 px of GT boundary.
        boundary_hit_t30: Fraction of predicted boundary points within 30 px of GT boundary.
        boundary_f1_t5: Boundary F1 at 5 px tolerance.
        boundary_f1_t10: Boundary F1 at 10 px tolerance.
        boundary_f1_t15: Boundary F1 at 15 px tolerance.
        boundary_f1_t20: Boundary F1 at 20 px tolerance.
        boundary_f1_t25: Boundary F1 at 25 px tolerance.
        boundary_f1_t30: Boundary F1 at 30 px tolerance.
        chamfer_mean_px: Symmetric mean Chamfer distance in px.
        chamfer_p95_px: Symmetric p95 Chamfer distance in px.
        chamfer_max_px: Symmetric max boundary distance in px.
        area_gt_px: GT foreground area in pixels.
        area_pred_px: Predicted foreground area in pixels.
    """

    frame_idx: int
    gt_valid: bool
    evaluated: bool
    mask_exists: bool
    iou: float
    dice: float
    boundary_hit_t5: float
    boundary_hit_t10: float
    boundary_hit_t15: float
    boundary_hit_t20: float
    boundary_hit_t25: float
    boundary_hit_t30: float
    boundary_f1_t5: float
    boundary_f1_t10: float
    boundary_f1_t15: float
    boundary_f1_t20: float
    boundary_f1_t25: float
    boundary_f1_t30: float
    chamfer_mean_px: float
    chamfer_p95_px: float
    chamfer_max_px: float
    area_gt_px: int
    area_pred_px: int


def _load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """Load binary mask.

    Args:
        mask_path: PNG path.

    Returns:
        Binary uint8 mask with values in {0, 255}, or None if missing/unreadable.
    """
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return ((mask > 0).astype(np.uint8) * 255)


def _compute_iou_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float]:
    """Compute region-overlap metrics.

    Args:
        gt_mask: GT binary mask (H, W).
        pred_mask: Pred binary mask (H, W).

    Returns:
        Tuple (iou, dice).
    """
    gt_bin = gt_mask > 0
    pred_bin = pred_mask > 0
    intersection = int(np.logical_and(gt_bin, pred_bin).sum())
    union = int(np.logical_or(gt_bin, pred_bin).sum())
    area_sum = int(gt_bin.sum() + pred_bin.sum())

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float((2.0 * intersection) / area_sum) if area_sum > 0 else 0.0
    return iou, dice


def _boundary_map(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel-ish boundary map.

    Args:
        mask: Binary uint8 mask (H, W).

    Returns:
        Binary boundary map (H, W) as bool.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = (mask > 0) & (eroded == 0)
    return boundary


def _distance_to_boundary(boundary: np.ndarray) -> np.ndarray:
    """Distance map to nearest boundary pixel.

    Args:
        boundary: Boundary bool map (H, W), True at boundary pixels.

    Returns:
        Float32 distance map (H, W), L2 distance in pixels.
    """
    inv = np.where(boundary, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)


def _boundary_hit_rates(
    gt_boundary: np.ndarray,
    pred_boundary: np.ndarray,
    tolerances_px: tuple[float, ...],
) -> dict[float, float]:
    """Compute one-sided boundary hit-rate (pred -> GT) at tolerances.

    For each tolerance ``t``:
        hit_rate_t = fraction of predicted boundary pixels whose nearest GT
                     boundary pixel lies within ``t`` pixels.

    Args:
        gt_boundary: GT boundary bool map (H, W).
        pred_boundary: Pred boundary bool map (H, W).
        tolerances_px: Pixel tolerances.

    Returns:
        Mapping ``tolerance -> hit_rate``.
    """
    n_pred = int(pred_boundary.sum())
    if n_pred == 0:
        return {float(t): 0.0 for t in tolerances_px}

    n_gt = int(gt_boundary.sum())
    if n_gt == 0:
        return {float(t): 0.0 for t in tolerances_px}

    d_to_gt = _distance_to_boundary(gt_boundary)
    pred_dist = d_to_gt[pred_boundary]
    return {float(t): float(np.mean(pred_dist <= float(t))) for t in tolerances_px}


def _boundary_f1_scores(
    gt_boundary: np.ndarray,
    pred_boundary: np.ndarray,
    tolerances_px: tuple[float, ...],
) -> dict[float, float]:
    """Compute boundary F1 scores at tolerances.

    Precision is pred->GT hit-rate; recall is GT->pred hit-rate.

    Args:
        gt_boundary: GT boundary bool map (H, W).
        pred_boundary: Pred boundary bool map (H, W).
        tolerances_px: Pixel tolerances.

    Returns:
        Mapping ``tolerance -> f1``.
    """
    n_gt = int(gt_boundary.sum())
    n_pred = int(pred_boundary.sum())

    if n_gt == 0 and n_pred == 0:
        return {float(t): 1.0 for t in tolerances_px}
    if n_gt == 0 or n_pred == 0:
        return {float(t): 0.0 for t in tolerances_px}

    d_to_gt = _distance_to_boundary(gt_boundary)
    d_to_pred = _distance_to_boundary(pred_boundary)
    pred_dist = d_to_gt[pred_boundary]
    gt_dist = d_to_pred[gt_boundary]

    out: dict[float, float] = {}
    for tol in tolerances_px:
        tol = float(tol)
        precision = float(np.mean(pred_dist <= tol))
        recall = float(np.mean(gt_dist <= tol))
        if precision + recall == 0.0:
            out[tol] = 0.0
        else:
            out[tol] = float((2.0 * precision * recall) / (precision + recall))
    return out


def _chamfer_stats(gt_boundary: np.ndarray, pred_boundary: np.ndarray) -> tuple[float, float, float]:
    """Symmetric boundary Chamfer stats.

    Args:
        gt_boundary: GT boundary bool map (H, W).
        pred_boundary: Pred boundary bool map (H, W).

    Returns:
        Tuple (mean, p95, max) in pixels.
    """
    n_gt = int(gt_boundary.sum())
    n_pred = int(pred_boundary.sum())
    if n_gt == 0 and n_pred == 0:
        return 0.0, 0.0, 0.0
    if n_gt == 0 or n_pred == 0:
        return float("inf"), float("inf"), float("inf")

    d_to_gt = _distance_to_boundary(gt_boundary)
    d_to_pred = _distance_to_boundary(pred_boundary)

    pred_to_gt = d_to_gt[pred_boundary].astype(np.float64)
    gt_to_pred = d_to_pred[gt_boundary].astype(np.float64)
    both = np.concatenate([pred_to_gt, gt_to_pred])

    return float(np.mean(both)), float(np.percentile(both, 95.0)), float(np.max(both))


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    """Compute finite-value summary stats.

    Args:
        values: Numeric values.

    Returns:
        Dict with count/mean/median/p05/p95/min/max.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _frame_metrics_to_csv_row(r: FrameMetrics) -> list[str]:
    """Serialize frame metrics to one CSV row."""
    return [
        str(r.frame_idx),
        str(int(r.gt_valid)),
        str(int(r.evaluated)),
        str(int(r.mask_exists)),
        f"{r.iou:.8f}",
        f"{r.dice:.8f}",
        f"{r.boundary_hit_t5:.8f}",
        f"{r.boundary_hit_t10:.8f}",
        f"{r.boundary_hit_t15:.8f}",
        f"{r.boundary_hit_t20:.8f}",
        f"{r.boundary_hit_t25:.8f}",
        f"{r.boundary_hit_t30:.8f}",
        f"{r.boundary_f1_t5:.8f}",
        f"{r.boundary_f1_t10:.8f}",
        f"{r.boundary_f1_t15:.8f}",
        f"{r.boundary_f1_t20:.8f}",
        f"{r.boundary_f1_t25:.8f}",
        f"{r.boundary_f1_t30:.8f}",
        f"{r.chamfer_mean_px:.8f}",
        f"{r.chamfer_p95_px:.8f}",
        f"{r.chamfer_max_px:.8f}",
        str(r.area_gt_px),
        str(r.area_pred_px),
    ]


def _write_sequence_csv(seq_rows: list[FrameMetrics], seq_csv_path: Path) -> None:
    """Write per-sequence CSV atomically.

    Args:
        seq_rows: Full frame rows for sequence.
        seq_csv_path: Destination CSV path.
    """
    seq_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = seq_csv_path.with_suffix(seq_csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SEQ_CSV_HEADER)
        for r in seq_rows:
            writer.writerow(_frame_metrics_to_csv_row(r))
    tmp_path.replace(seq_csv_path)


def _load_sequence_csv(seq_csv_path: Path) -> Optional[list[FrameMetrics]]:
    """Load per-sequence CSV if present and parseable.

    Args:
        seq_csv_path: Per-sequence CSV path.

    Returns:
        Parsed rows or None if file missing/invalid.
    """
    if not seq_csv_path.exists():
        return None
    try:
        with seq_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None
            missing = [k for k in _SEQ_CSV_HEADER if k not in reader.fieldnames]
            if missing:
                return None

            rows: list[FrameMetrics] = []
            for d in reader:
                rows.append(
                    FrameMetrics(
                        frame_idx=int(d["frame_idx"]),
                        gt_valid=bool(int(d["gt_valid"])),
                        evaluated=bool(int(d["evaluated"])),
                        mask_exists=bool(int(d["mask_exists"])),
                        iou=float(d["iou"]),
                        dice=float(d["dice"]),
                        boundary_hit_t5=float(d["boundary_hit_t5"]),
                        boundary_hit_t10=float(d["boundary_hit_t10"]),
                        boundary_hit_t15=float(d["boundary_hit_t15"]),
                        boundary_hit_t20=float(d["boundary_hit_t20"]),
                        boundary_hit_t25=float(d["boundary_hit_t25"]),
                        boundary_hit_t30=float(d["boundary_hit_t30"]),
                        boundary_f1_t5=float(d["boundary_f1_t5"]),
                        boundary_f1_t10=float(d["boundary_f1_t10"]),
                        boundary_f1_t15=float(d["boundary_f1_t15"]),
                        boundary_f1_t20=float(d["boundary_f1_t20"]),
                        boundary_f1_t25=float(d["boundary_f1_t25"]),
                        boundary_f1_t30=float(d["boundary_f1_t30"]),
                        chamfer_mean_px=float(d["chamfer_mean_px"]),
                        chamfer_p95_px=float(d["chamfer_p95_px"]),
                        chamfer_max_px=float(d["chamfer_max_px"]),
                        area_gt_px=int(d["area_gt_px"]),
                        area_pred_px=int(d["area_pred_px"]),
                    )
                )
    except Exception:
        return None
    return rows


def _build_summary(
    dataset: str,
    split: str,
    sequences: list[str],
    failed_sequences: list[str],
    num_valid_frames_total: int,
    records: list[FrameMetrics],
    sequence_csv_dir: Path,
    sequence_csv_files: dict[str, str],
    processed_sequences: Optional[list[str]] = None,
    elapsed_sec: Optional[float] = None,
) -> dict:
    """Build summary payload from current state.

    Args:
        dataset: Dataset key.
        split: Dataset split.
        sequences: Requested sequence list.
        failed_sequences: Sequences that failed processing.
        num_valid_frames_total: Count of valid GT frames (evaluation mask sum).
        records: Evaluated frame metrics.
        sequence_csv_dir: Directory with per-sequence CSV files.
        sequence_csv_files: Mapping sequence -> CSV path.
        processed_sequences: Optional list of already processed sequences.
        elapsed_sec: Optional elapsed wall-clock time.

    Returns:
        JSON-serializable summary dictionary.
    """
    summary = {
        "dataset": dataset,
        "split": split,
        "num_sequences_requested": int(len(sequences)),
        "num_sequences_failed": int(len(failed_sequences)),
        "failed_sequences": failed_sequences,
        "num_valid_frames_total": int(num_valid_frames_total),
        "num_frames_recorded": int(len(records)),
        "num_frames_with_mask": int(sum(int(r.mask_exists) for r in records)),
        "num_frames_missing_mask": int(sum(int(not r.mask_exists) for r in records)),
        "per_sequence_csv_dir": str(sequence_csv_dir),
        "per_sequence_csv_files": sequence_csv_files,
        "metrics": {
            "iou": _finite_stats(np.array([r.iou for r in records], dtype=np.float64)),
            "dice": _finite_stats(np.array([r.dice for r in records], dtype=np.float64)),
            "boundary_hit_t5": _finite_stats(
                np.array([r.boundary_hit_t5 for r in records], dtype=np.float64)
            ),
            "boundary_hit_t10": _finite_stats(
                np.array([r.boundary_hit_t10 for r in records], dtype=np.float64)
            ),
            "boundary_hit_t15": _finite_stats(
                np.array([r.boundary_hit_t15 for r in records], dtype=np.float64)
            ),
            "boundary_hit_t20": _finite_stats(
                np.array([r.boundary_hit_t20 for r in records], dtype=np.float64)
            ),
            "boundary_hit_t25": _finite_stats(
                np.array([r.boundary_hit_t25 for r in records], dtype=np.float64)
            ),
            "boundary_hit_t30": _finite_stats(
                np.array([r.boundary_hit_t30 for r in records], dtype=np.float64)
            ),
            "boundary_f1_t5": _finite_stats(
                np.array([r.boundary_f1_t5 for r in records], dtype=np.float64)
            ),
            "boundary_f1_t10": _finite_stats(
                np.array([r.boundary_f1_t10 for r in records], dtype=np.float64)
            ),
            "boundary_f1_t15": _finite_stats(
                np.array([r.boundary_f1_t15 for r in records], dtype=np.float64)
            ),
            "boundary_f1_t20": _finite_stats(
                np.array([r.boundary_f1_t20 for r in records], dtype=np.float64)
            ),
            "boundary_f1_t25": _finite_stats(
                np.array([r.boundary_f1_t25 for r in records], dtype=np.float64)
            ),
            "boundary_f1_t30": _finite_stats(
                np.array([r.boundary_f1_t30 for r in records], dtype=np.float64)
            ),
            "chamfer_mean_px": _finite_stats(
                np.array([r.chamfer_mean_px for r in records], dtype=np.float64)
            ),
            "chamfer_p95_px": _finite_stats(
                np.array([r.chamfer_p95_px for r in records], dtype=np.float64)
            ),
            "chamfer_max_px": _finite_stats(
                np.array([r.chamfer_max_px for r in records], dtype=np.float64)
            ),
        },
    }
    if processed_sequences is not None:
        summary["num_sequences_processed"] = int(len(processed_sequences))
        summary["processed_sequences"] = processed_sequences
    if elapsed_sec is not None:
        summary["elapsed_sec"] = float(elapsed_sec)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Precompute PTi SAM3 mask metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["pti", "planartrack-initfix"],
        default="pti",
        help="Dataset key. Only PTi accepted.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="PTi split for sequence discovery.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=paths.PT_BASE,
        help="PlanarTrack base directory.",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=paths.PT_INITFIX_DIR,
        help="Initfix directory for PTi GT.",
    )
    parser.add_argument(
        "--precomputed-masks-dir",
        type=Path,
        default=paths.SAM3_PRECOMPUTED_MASKS_PTI,
        help="Directory with SAM3 masks: <dir>/<seq>/{frame+1:05d}.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.OUTPUT_DIRECTORY / "experiments" / "pti_sam3_mask_metrics",
        help="Output directory for metric tables.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQ",
        help="Optional explicit sequence list.",
    )
    parser.add_argument(
        "--subset",
        metavar="NAME",
        help="Subset name from configs/subsets/<name>.txt.",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        metavar="PATH",
        help="Direct subset file path.",
    )
    parser.add_argument(
        "--include-init-frame",
        action="store_true",
        help="Include frame 0 in evaluation. Default excludes init frame.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N processed sequences.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write partial summary every N processed sequences.",
    )
    return parser


def main() -> None:
    """Run PTi mask metric precomputation."""
    args = build_parser().parse_args()
    dataset = "pti" if args.dataset == "pti" else "planartrack-initfix"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = get_sequences(
        dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split,
    )

    print(f"Dataset     : {dataset}")
    print(f"Split       : {args.split}")
    print(f"Sequences   : {len(sequences)}")
    print(f"Masks dir   : {args.precomputed_masks_dir}")
    print(f"Output dir  : {output_dir}")

    tolerances_px = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

    records: list[FrameMetrics] = []
    failed_sequences: list[str] = []
    num_valid_frames_total = 0
    sequence_csv_dir = output_dir / "per_sequence"
    sequence_csv_dir.mkdir(parents=True, exist_ok=True)
    sequence_csv_files: dict[str, str] = {}
    processed_sequences: list[str] = []
    processed_eval_frames = 0
    resumed_sequences = 0
    scratch_sequences = 0
    scratch_elapsed_sec = 0.0
    start_time = time.perf_counter()
    partial_summary_path = output_dir / "summary.partial.json"

    try:
        for seq_name in sequences:
            seq_start_time = time.perf_counter()
            sequence_resumed = False
            try:
                gt_corners, valid = pti_load_gt(args.base_dir, seq_name, args.initfix_dir)
                image_paths = pt_image_paths(args.base_dir, seq_name)
                T = min(len(gt_corners), len(valid), len(image_paths))

                gt_corners = gt_corners[:T]
                valid = valid[:T]
                image_paths = image_paths[:T]

                eval_mask = valid.copy()
                if not args.include_init_frame and len(eval_mask) > 0:
                    eval_mask[0] = False
                n_eval_seq = int(eval_mask.sum())

                seq_csv_path = sequence_csv_dir / f"{seq_name}.csv"
                loaded_rows = _load_sequence_csv(seq_csv_path)
                seq_rows: list[FrameMetrics]
                can_resume = False
                if loaded_rows is not None and len(loaded_rows) == T:
                    has_all_idx = all(loaded_rows[i].frame_idx == i for i in range(T))
                    gt_match = all(loaded_rows[i].gt_valid == bool(valid[i]) for i in range(T))
                    eval_match = all(
                        loaded_rows[i].evaluated == bool(eval_mask[i]) for i in range(T)
                    )
                    can_resume = has_all_idx and gt_match and eval_match

                if can_resume:
                    seq_rows = loaded_rows
                    resumed_sequences += 1
                    sequence_resumed = True
                else:
                    seq_rows = []
                    for frame_idx in range(T):
                        seq_rows.append(
                            FrameMetrics(
                                frame_idx=frame_idx,
                                gt_valid=bool(valid[frame_idx]),
                                evaluated=bool(eval_mask[frame_idx]),
                                mask_exists=False,
                                iou=float("nan"),
                                dice=float("nan"),
                                boundary_hit_t5=float("nan"),
                                boundary_hit_t10=float("nan"),
                                boundary_hit_t15=float("nan"),
                                boundary_hit_t20=float("nan"),
                                boundary_hit_t25=float("nan"),
                                boundary_hit_t30=float("nan"),
                                boundary_f1_t5=float("nan"),
                                boundary_f1_t10=float("nan"),
                                boundary_f1_t15=float("nan"),
                                boundary_f1_t20=float("nan"),
                                boundary_f1_t25=float("nan"),
                                boundary_f1_t30=float("nan"),
                                chamfer_mean_px=float("nan"),
                                chamfer_p95_px=float("nan"),
                                chamfer_max_px=float("nan"),
                                area_gt_px=0,
                                area_pred_px=0,
                            )
                        )

                    for frame_idx in np.where(eval_mask)[0]:
                        frame_idx = int(frame_idx)
                        mask_path = args.precomputed_masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
                        pred_mask = _load_mask(mask_path)
                        if pred_mask is None:
                            frame = cv2.imread(str(image_paths[frame_idx]))
                            if frame is None:
                                raise ValueError(f"Failed to load image: {image_paths[frame_idx]}")
                            h_img, w_img = frame.shape[:2]
                            gt_mask = corners_to_mask(gt_corners[frame_idx], (h_img, w_img))
                            seq_rows[frame_idx].area_gt_px = int((gt_mask > 0).sum())
                            continue

                        h_img, w_img = pred_mask.shape[:2]
                        gt_mask = corners_to_mask(gt_corners[frame_idx], (h_img, w_img))

                        iou, dice = _compute_iou_dice(gt_mask, pred_mask)

                        gt_boundary = _boundary_map(gt_mask)
                        pred_boundary = _boundary_map(pred_mask)
                        hit_rates = _boundary_hit_rates(
                            gt_boundary,
                            pred_boundary,
                            tolerances_px=tolerances_px,
                        )
                        f1_scores = _boundary_f1_scores(
                            gt_boundary,
                            pred_boundary,
                            tolerances_px=tolerances_px,
                        )
                        chamfer_mean, chamfer_p95, chamfer_max = _chamfer_stats(
                            gt_boundary,
                            pred_boundary,
                        )

                        row = seq_rows[frame_idx]
                        row.mask_exists = True
                        row.iou = iou
                        row.dice = dice
                        row.boundary_hit_t5 = hit_rates[5.0]
                        row.boundary_hit_t10 = hit_rates[10.0]
                        row.boundary_hit_t15 = hit_rates[15.0]
                        row.boundary_hit_t20 = hit_rates[20.0]
                        row.boundary_hit_t25 = hit_rates[25.0]
                        row.boundary_hit_t30 = hit_rates[30.0]
                        row.boundary_f1_t5 = f1_scores[5.0]
                        row.boundary_f1_t10 = f1_scores[10.0]
                        row.boundary_f1_t15 = f1_scores[15.0]
                        row.boundary_f1_t20 = f1_scores[20.0]
                        row.boundary_f1_t25 = f1_scores[25.0]
                        row.boundary_f1_t30 = f1_scores[30.0]
                        row.chamfer_mean_px = chamfer_mean
                        row.chamfer_p95_px = chamfer_p95
                        row.chamfer_max_px = chamfer_max
                        row.area_gt_px = int((gt_mask > 0).sum())
                        row.area_pred_px = int((pred_mask > 0).sum())

                    _write_sequence_csv(seq_rows, seq_csv_path)

                records.extend([r for r in seq_rows if r.evaluated])
                num_valid_frames_total += n_eval_seq
                processed_eval_frames += n_eval_seq
                sequence_csv_files[seq_name] = str(seq_csv_path)
            except Exception:
                failed_sequences.append(seq_name)

            seq_elapsed = time.perf_counter() - seq_start_time
            if not sequence_resumed:
                scratch_sequences += 1
                scratch_elapsed_sec += seq_elapsed

            processed_sequences.append(seq_name)
            elapsed_sec = time.perf_counter() - start_time
            done = len(processed_sequences)
            remaining = max(len(sequences) - done, 0)
            if scratch_sequences > 0:
                avg_sec_per_scratch_seq = scratch_elapsed_sec / scratch_sequences
                eta_sec = remaining * avg_sec_per_scratch_seq
                eta_str = f"{eta_sec / 60.0:.1f} min"
            else:
                eta_str = "n/a"

            if args.progress_every > 0 and done % args.progress_every == 0:
                print(
                    f"Progress: {done}/{len(sequences)} seq | "
                    f"resumed: {resumed_sequences} | "
                    f"scratch: {scratch_sequences} | "
                    f"frames(eval): {processed_eval_frames} | "
                    f"elapsed: {elapsed_sec / 60.0:.1f} min | "
                    f"ETA: {eta_str}"
                )

            if args.checkpoint_every > 0 and done % args.checkpoint_every == 0:
                partial_summary = _build_summary(
                    dataset=dataset,
                    split=args.split,
                    sequences=sequences,
                    failed_sequences=failed_sequences,
                    num_valid_frames_total=num_valid_frames_total,
                    records=records,
                    sequence_csv_dir=sequence_csv_dir,
                    sequence_csv_files=sequence_csv_files,
                    processed_sequences=processed_sequences,
                    elapsed_sec=elapsed_sec,
                )
                partial_summary["num_sequences_resumed"] = int(resumed_sequences)
                partial_summary_path.write_text(
                    json.dumps(partial_summary, indent=2),
                    encoding="utf-8",
                )
                print(f"Checkpoint saved: {partial_summary_path}")
    except KeyboardInterrupt:
        elapsed_sec = time.perf_counter() - start_time
        partial_summary = _build_summary(
            dataset=dataset,
            split=args.split,
            sequences=sequences,
            failed_sequences=failed_sequences,
            num_valid_frames_total=num_valid_frames_total,
            records=records,
            sequence_csv_dir=sequence_csv_dir,
            sequence_csv_files=sequence_csv_files,
            processed_sequences=processed_sequences,
            elapsed_sec=elapsed_sec,
        )
        partial_summary["num_sequences_resumed"] = int(resumed_sequences)
        partial_summary_path.write_text(
            json.dumps(partial_summary, indent=2),
            encoding="utf-8",
        )
        print(f"\nInterrupted. Checkpoint saved: {partial_summary_path}")
        raise SystemExit(130)

    if not records:
        raise ValueError("No frames evaluated. Check masks path / split / subset.")

    elapsed_sec = time.perf_counter() - start_time
    summary = _build_summary(
        dataset=dataset,
        split=args.split,
        sequences=sequences,
        failed_sequences=failed_sequences,
        num_valid_frames_total=num_valid_frames_total,
        records=records,
        sequence_csv_dir=sequence_csv_dir,
        sequence_csv_files=sequence_csv_files,
        processed_sequences=processed_sequences,
        elapsed_sec=elapsed_sec,
    )
    summary["num_sequences_resumed"] = int(resumed_sequences)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== PTi SAM3 mask metrics ===")
    print(f"Valid frames     : {summary['num_valid_frames_total']}")
    print(f"Frames with mask : {summary['num_frames_with_mask']}")
    print(f"Masks missing    : {summary['num_frames_missing_mask']}")
    print(f"Sequences resumed: {summary['num_sequences_resumed']}")
    print(f"IoU mean         : {summary['metrics']['iou']['mean']:.4f}")
    print(f"Boundary hit@10  : {summary['metrics']['boundary_hit_t10']['mean']:.4f}")
    print(f"Chamfer mean [px]: {summary['metrics']['chamfer_mean_px']['mean']:.4f}")
    print(f"Elapsed [min]    : {summary['elapsed_sec'] / 60.0:.2f}")
    print(f"Per-sequence CSV : {sequence_csv_dir}")
    print(f"Summary JSON     : {summary_path}")


if __name__ == "__main__":
    main()
