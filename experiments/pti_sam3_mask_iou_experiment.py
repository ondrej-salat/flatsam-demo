#!/usr/bin/env python3
"""Evaluate SAM3 precomputed masks vs PTi GT masks and tracker quality.

Computes mask IoU against GT quadrilateral masks on valid PTi frames and
stratifies tracker quality by segmentation success/failure.

Failure definition:
    IoU < --iou-fail-threshold

GT validity:
    Only frames marked valid by PTi annotation loader.
"""

import argparse
import csv
import json
import traceback
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from einops import rearrange

from flatsam2.datasets import corners_to_mask, get_sequences, pt_image_paths, pti_load_gt
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths


DEFAULT_TRACKER_NAME = "LMr_allobj_infl10_X_SAM3H_thr35"


@dataclass
class FrameRecord:
    """Per-frame evaluation payload.

    Args:
        sequence: Sequence name.
        frame_idx: Frame index.
        image_path: Path to frame image.
        valid_gt: Whether GT says frame valid.
        mask_path: Precomputed mask path for frame.
        mask_exists: Whether mask file exists and loads.
        iou: IoU value if mask exists, else None.
        segmentation_failed: True when mask exists and iou < threshold.
        tracker_error_px: Alignment error for tracker prediction, inf when invalid.
        tracker_p15: Whether tracker error is <= 15 px.
    """

    sequence: str
    frame_idx: int
    image_path: Path
    valid_gt: bool
    mask_path: Path
    mask_exists: bool
    iou: Optional[float]
    segmentation_failed: bool
    tracker_error_px: float
    tracker_p15: bool


def _load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    """Load tracker predictions from txt file.

    Args:
        results_dir: Tracker output directory.
        seq_name: Sequence name.

    Returns:
        Predicted corners with shape (frames, 2, 4).
    """
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise ValueError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(
        pred_flat,
        "frames (corners xy) -> frames xy corners",
        corners=4,
        xy=2,
    )


def _load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """Load precomputed mask as binary uint8.

    Args:
        mask_path: PNG mask path.

    Returns:
        Binary mask (H, W) in {0, 255}, or None when file missing/unreadable.
    """
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return ((mask > 0).astype(np.uint8) * 255)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU for two binary masks.

    Args:
        mask_a: Binary mask (H, W).
        mask_b: Binary mask (H, W).

    Returns:
        IoU in [0, 1].
    """
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def _safe_tracker_error(gt_corners: np.ndarray, pred_corners: np.ndarray) -> float:
    """Compute tracker corner alignment error or inf when invalid.

    Args:
        gt_corners: GT corners (2, 4).
        pred_corners: Pred corners (2, 4).

    Returns:
        Alignment error in px, inf when non-finite input.
    """
    if not np.all(np.isfinite(gt_corners)) or not np.all(np.isfinite(pred_corners)):
        return float("inf")
    err = alignment_error(gt_corners, pred_corners)
    return float(err)


def _mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _p15(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(100.0 * np.mean(values <= 15.0))


def _draw_overlay(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    sam_mask: np.ndarray,
    gt_corners: np.ndarray,
    pred_corners: np.ndarray,
    iou: float,
    tracker_error_px: float,
    seq_name: str,
    frame_idx: int,
) -> np.ndarray:
    """Draw visual debugging overlay for one failed frame.

    Args:
        image_bgr: Input frame image.
        gt_mask: GT mask (H, W).
        sam_mask: SAM mask (H, W).
        gt_corners: GT corners (2, 4).
        pred_corners: Pred corners (2, 4).
        iou: Mask IoU.
        tracker_error_px: Tracker error for this frame.
        seq_name: Sequence name.
        frame_idx: Frame index.

    Returns:
        BGR image with overlays.
    """
    vis = image_bgr.copy()
    gt_bin = gt_mask > 0
    sam_bin = sam_mask > 0

    gt_only = gt_bin & (~sam_bin)
    sam_only = sam_bin & (~gt_bin)
    overlap = gt_bin & sam_bin

    tint = np.zeros_like(vis)
    tint[gt_only] = (255, 80, 80)
    tint[sam_only] = (60, 60, 255)
    tint[overlap] = (80, 220, 80)
    vis = cv2.addWeighted(vis, 0.75, tint, 0.25, 0)

    gt_poly = rearrange(gt_corners, "xy c -> c 1 xy").astype(np.int32)
    pred_poly = rearrange(pred_corners, "xy c -> c 1 xy").astype(np.int32)
    cv2.polylines(vis, [gt_poly], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.polylines(vis, [pred_poly], isClosed=True, color=(0, 255, 255), thickness=2)

    lines = [
        f"{seq_name} frame={frame_idx}",
        f"IoU={iou:.3f}",
        f"tracker_err={tracker_error_px:.2f}px",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return vis


def _save_failure_report(
    records: list[FrameRecord],
    output_dir: Path,
    vis_dir: Path,
    fail_threshold: float,
) -> None:
    """Save compact HTML report for failed segmentation frames.

    Args:
        records: Failed-frame records.
        output_dir: Experiment output directory.
        vis_dir: Directory containing per-frame visualization PNGs.
        fail_threshold: IoU threshold used for failure.
    """
    rows = []
    for rec in records:
        rel = (vis_dir / rec.sequence / f"{rec.frame_idx:05d}.jpg").relative_to(output_dir)
        rows.append(
            "<tr>"
            f"<td>{escape(rec.sequence)}</td>"
            f"<td>{rec.frame_idx}</td>"
            f"<td>{rec.iou:.3f}</td>"
            f"<td>{rec.tracker_error_px:.2f}</td>"
            f"<td>{'yes' if rec.tracker_p15 else 'no'}</td>"
            f"<td><a href='{escape(str(rel))}'>open</a></td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>PTi SAM3 failures</title>
  <style>
    body {{ font-family: monospace; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h2>PTi SAM3 mask failures</h2>
  <p>failure if IoU &lt; {fail_threshold:.3f}; sorted by IoU ascending</p>
  <table>
    <thead>
      <tr><th>sequence</th><th>frame</th><th>IoU</th><th>tracker err [px]</th><th>tracker p@15</th><th>image</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "failure_report.html").write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="PTi SAM3 mask IoU experiment with tracker stratification.",
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
        "--tracker-name",
        default=DEFAULT_TRACKER_NAME,
        help="Tracker result directory name under OUTPUT_DIRECTORY.",
    )
    parser.add_argument(
        "--tracker-results-dir",
        type=Path,
        default=None,
        help="Optional explicit tracker directory (<dir>/<seq>.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.OUTPUT_DIRECTORY / "experiments" / "pti_sam3_iou_eval",
        help="Output directory for summary, CSV, and failure visuals.",
    )
    parser.add_argument(
        "--iou-fail-threshold",
        type=float,
        default=0.5,
        help="Segmentation failure threshold (IoU < threshold).",
    )
    parser.add_argument(
        "--max-failure-visualizations",
        type=int,
        default=500,
        help="Cap number of failure frames rendered to images.",
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
    return parser


def main() -> None:
    """Run PTi SAM3 IoU experiment."""
    args = build_parser().parse_args()
    dataset = "pti" if args.dataset == "pti" else "planartrack-initfix"

    ds_label = "PlanarTrackInitfix"
    tracker_dir = (
        args.tracker_results_dir
        if args.tracker_results_dir is not None
        else paths.OUTPUT_DIRECTORY / args.tracker_name / ds_label
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "failure_frames"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sequences = get_sequences(
        dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split,
    )

    print(f"Dataset             : {dataset} ({ds_label})")
    print(f"Split               : {args.split}")
    print(f"Sequences           : {len(sequences)}")
    print(f"Masks dir           : {args.precomputed_masks_dir}")
    print(f"Tracker results dir : {tracker_dir}")
    print(f"Fail threshold      : IoU < {args.iou_fail_threshold:.3f}")
    print(f"Output dir          : {output_dir}")

    all_records: list[FrameRecord] = []
    failed_sequences: list[str] = []
    error_log: list[tuple[str, str, str]] = []

    for seq_name in sequences:
        try:
            gt_corners, valid = pti_load_gt(args.base_dir, seq_name, args.initfix_dir)
            image_paths = pt_image_paths(args.base_dir, seq_name)
            pred_corners = _load_predictions(tracker_dir, seq_name)

            T = min(len(gt_corners), len(valid), len(image_paths), len(pred_corners))
            gt_corners = gt_corners[:T]
            valid = valid[:T]
            image_paths = image_paths[:T]
            pred_corners = pred_corners[:T]

            eval_mask = valid.copy()
            if len(eval_mask) > 0:
                eval_mask[0] = False

            for frame_idx in np.where(eval_mask)[0]:
                frame_path = image_paths[int(frame_idx)]
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    raise ValueError(f"Failed to load image {frame_path}")
                h_img, w_img = frame_bgr.shape[:2]

                gt_mask = corners_to_mask(gt_corners[int(frame_idx)], (h_img, w_img))
                mask_path = (
                    args.precomputed_masks_dir / seq_name / f"{int(frame_idx) + 1:05d}.png"
                )
                sam_mask = _load_mask(mask_path)

                tracker_err = _safe_tracker_error(
                    gt_corners[int(frame_idx)], pred_corners[int(frame_idx)]
                )

                if sam_mask is None:
                    all_records.append(
                        FrameRecord(
                            sequence=seq_name,
                            frame_idx=int(frame_idx),
                            image_path=frame_path,
                            valid_gt=True,
                            mask_path=mask_path,
                            mask_exists=False,
                            iou=None,
                            segmentation_failed=False,
                            tracker_error_px=tracker_err,
                            tracker_p15=bool(tracker_err <= 15.0),
                        )
                    )
                    continue

                if sam_mask.shape != gt_mask.shape:
                    sam_mask = cv2.resize(
                        sam_mask,
                        (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST,
                    )

                iou = _mask_iou(gt_mask, sam_mask)
                seg_failed = iou < args.iou_fail_threshold
                all_records.append(
                    FrameRecord(
                        sequence=seq_name,
                        frame_idx=int(frame_idx),
                        image_path=frame_path,
                        valid_gt=True,
                        mask_path=mask_path,
                        mask_exists=True,
                        iou=iou,
                        segmentation_failed=bool(seg_failed),
                        tracker_error_px=tracker_err,
                        tracker_p15=bool(tracker_err <= 15.0),
                    )
                )
        except Exception as exc:
            failed_sequences.append(seq_name)
            error_log.append((seq_name, str(exc), traceback.format_exc()))

    if not all_records:
        raise ValueError("No valid PTi frames evaluated. Check paths/split/subset.")

    mask_available = np.array([r.mask_exists for r in all_records], dtype=bool)
    iou_values = np.array(
        [r.iou for r in all_records if r.iou is not None],
        dtype=np.float64,
    )
    tracker_errors = np.array([r.tracker_error_px for r in all_records], dtype=np.float64)

    seg_fail = np.array([r.segmentation_failed for r in all_records], dtype=bool)
    seg_pass = mask_available & (~seg_fail)
    missing_mask = ~mask_available

    fail_tracker_errors = np.array(
        [r.tracker_error_px for r in all_records if r.segmentation_failed], dtype=np.float64
    )
    pass_tracker_errors = np.array(
        [
            r.tracker_error_px
            for r in all_records
            if r.mask_exists and (not r.segmentation_failed)
        ],
        dtype=np.float64,
    )
    missing_tracker_errors = np.array(
        [r.tracker_error_px for r in all_records if not r.mask_exists], dtype=np.float64
    )

    summary = {
        "dataset": dataset,
        "split": args.split,
        "num_sequences_requested": int(len(sequences)),
        "num_sequences_failed": int(len(failed_sequences)),
        "failed_sequences": failed_sequences,
        "num_valid_eval_frames": int(len(all_records)),
        "num_frames_with_mask": int(mask_available.sum()),
        "num_frames_missing_mask": int(missing_mask.sum()),
        "mean_mask_iou": _mean(iou_values),
        "median_mask_iou": (
            float(np.median(iou_values)) if len(iou_values) else float("nan")
        ),
        "segmentation_fail_threshold": float(args.iou_fail_threshold),
        "num_segmentation_fail_frames": int(seg_fail.sum()),
        "num_segmentation_pass_frames": int(seg_pass.sum()),
        "segmentation_fail_rate_percent": float(
            100.0 * seg_fail.sum() / max(mask_available.sum(), 1)
        ),
        "tracker": {
            "name": args.tracker_name,
            "results_dir": str(tracker_dir),
            "overall": {
                "num_frames": int(len(tracker_errors)),
                "mean_error_px": _mean(tracker_errors),
                "p15": _p15(tracker_errors),
            },
            "on_segmentation_pass_frames": {
                "num_frames": int(len(pass_tracker_errors)),
                "mean_error_px": _mean(pass_tracker_errors),
                "p15": _p15(pass_tracker_errors),
            },
            "on_segmentation_fail_frames": {
                "num_frames": int(len(fail_tracker_errors)),
                "mean_error_px": _mean(fail_tracker_errors),
                "p15": _p15(fail_tracker_errors),
            },
            "on_missing_mask_frames": {
                "num_frames": int(len(missing_tracker_errors)),
                "mean_error_px": _mean(missing_tracker_errors),
                "p15": _p15(missing_tracker_errors),
            },
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_dir / "frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "frame_idx",
                "image_path",
                "mask_path",
                "mask_exists",
                "iou",
                "segmentation_failed",
                "tracker_error_px",
                "tracker_p15",
            ]
        )
        for rec in all_records:
            writer.writerow(
                [
                    rec.sequence,
                    rec.frame_idx,
                    str(rec.image_path),
                    str(rec.mask_path),
                    int(rec.mask_exists),
                    "" if rec.iou is None else f"{rec.iou:.6f}",
                    int(rec.segmentation_failed),
                    f"{rec.tracker_error_px:.6f}",
                    int(rec.tracker_p15),
                ]
            )

    fail_records = sorted(
        [r for r in all_records if r.segmentation_failed and r.iou is not None],
        key=lambda r: (r.iou, r.sequence, r.frame_idx),
    )

    max_vis = max(0, args.max_failure_visualizations)
    rendered = 0
    rendered_records: list[FrameRecord] = []
    sequence_cache: dict[str, tuple[np.ndarray, np.ndarray, list[Path]]] = {}
    for rec in fail_records[:max_vis]:
        if rec.sequence not in sequence_cache:
            gt_corners, _ = pti_load_gt(args.base_dir, rec.sequence, args.initfix_dir)
            pred_corners = _load_predictions(tracker_dir, rec.sequence)
            image_paths = pt_image_paths(args.base_dir, rec.sequence)
            sequence_cache[rec.sequence] = (gt_corners, pred_corners, image_paths)
        gt_corners, pred_corners, image_paths = sequence_cache[rec.sequence]
        T = min(len(gt_corners), len(pred_corners), len(image_paths))
        frame_idx = rec.frame_idx
        if frame_idx >= T:
            continue

        frame_bgr = cv2.imread(str(image_paths[frame_idx]))
        if frame_bgr is None:
            continue
        h_img, w_img = frame_bgr.shape[:2]
        gt_mask = corners_to_mask(gt_corners[frame_idx], (h_img, w_img))
        sam_mask = _load_mask(rec.mask_path)
        if sam_mask is None:
            continue
        if sam_mask.shape != gt_mask.shape:
            sam_mask = cv2.resize(sam_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        vis = _draw_overlay(
            frame_bgr,
            gt_mask,
            sam_mask,
            gt_corners[frame_idx],
            pred_corners[frame_idx],
            float(rec.iou),
            rec.tracker_error_px,
            rec.sequence,
            frame_idx,
        )
        seq_vis_dir = vis_dir / rec.sequence
        seq_vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(seq_vis_dir / f"{frame_idx:05d}.jpg"), vis)
        rendered_records.append(rec)
        rendered += 1

    _save_failure_report(rendered_records, output_dir, vis_dir, args.iou_fail_threshold)

    print("\n=== SAM3 mask IoU experiment (PTi) ===")
    print(f"Valid eval frames          : {summary['num_valid_eval_frames']}")
    print(f"Frames with mask           : {summary['num_frames_with_mask']}")
    print(f"Frames missing mask        : {summary['num_frames_missing_mask']}")
    print(
        "Mean / median IoU          : "
        f"{summary['mean_mask_iou']:.4f} / {summary['median_mask_iou']:.4f}"
    )
    print(f"Segmentation fail frames   : {summary['num_segmentation_fail_frames']}")
    print(f"Segmentation fail rate [%] : {summary['segmentation_fail_rate_percent']:.2f}")
    print(
        "Tracker P@15 (pass/fail): "
        f"{summary['tracker']['on_segmentation_pass_frames']['p15']:.2f} / "
        f"{summary['tracker']['on_segmentation_fail_frames']['p15']:.2f}"
    )
    print(
        "Tracker mean err px (pass/fail): "
        f"{summary['tracker']['on_segmentation_pass_frames']['mean_error_px']:.2f} / "
        f"{summary['tracker']['on_segmentation_fail_frames']['mean_error_px']:.2f}"
    )
    print(f"Rendered failure visuals   : {rendered}")
    print(f"Summary JSON               : {output_dir / 'summary.json'}")
    print(f"Per-frame CSV              : {csv_path}")
    print(f"Failure report             : {output_dir / 'failure_report.html'}")

    if error_log:
        print("\n=== Sequence errors (first 3) ===")
        for seq_name, msg, tb in error_log[:3]:
            print(f"\n{seq_name}: {msg}")
            print(tb, end="")


if __name__ == "__main__":
    main()
