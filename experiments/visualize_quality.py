#!/usr/bin/env python3
"""Visualize homography quality estimation results.

Produces per-frame images or videos showing:
  - Color-coded ∂A' boundary segments by classification
    (ALIGNED=green, OCCLUDED=blue, INTERIOR=red, OUT_OF_FRAME=grey).
  - Color-coded ∂B boundary segments
    (ALIGNED=green, EXCESS=orange, OCCL_EDGE=cyan, FRAME_EDGE=yellow).
  - Occlusion mask (A' \ B) tinted blue, excess mask (B \ A') tinted red.
  - Side panel: per-class fraction bars + explanation text.

Usage:
    # Single frame visualization
    python experiments/visualize_quality.py pot seq_name --frame 42 --show

    # Full sequence video
    python experiments/visualize_quality.py pot seq_name --video

    # Batch: visualize specific frames from a quality eval JSON
    python experiments/visualize_quality.py pot seq_name --from-eval results.json --top-k 10
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.homography_quality import (
    HomographyExplanation,
    PointClassA,
    PointClassB,
    ReferenceContour,
    Segment,
    assess_homography,
    homography_from_corners,
)
from flatsam2.path_config import paths


# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------

# ∂A' class colors
_COLOR_A = {
    PointClassA.ALIGNED:       (50,  220,  50),   # green
    PointClassA.OCCLUDED:      (220, 100,  30),   # blue-orange
    PointClassA.INTERIOR:      (50,   50, 230),   # red
    PointClassA.OUT_OF_FRAME:  (130, 130, 130),   # grey
}

# ∂B class colors
_COLOR_B = {
    PointClassB.ALIGNED:    (50,  220,  50),   # green
    PointClassB.EXCESS:     (30,  140, 255),   # orange
    PointClassB.OCCL_EDGE:  (220, 200,  50),   # cyan
    PointClassB.FRAME_EDGE: (50,  220, 220),   # yellow
}

# Region tint colors (BGR, semi-transparent)
_TINT_OCCLUSION = (180, 80,  20)   # blue
_TINT_EXCESS    = (20,  40, 200)   # red


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_region_tints(
    vis: np.ndarray,
    occlusion_mask: Optional[np.ndarray],
    excess_B_mask: Optional[np.ndarray],
    alpha: float = 0.30,
) -> np.ndarray:
    """Tint occlusion and excess regions on *vis* (in-place copy returned).

    Args:
        vis: BGR uint8 frame.
        occlusion_mask: uint8 mask (0/255), pixels in A' but not B.
        excess_B_mask: uint8 mask (0/255), pixels in B but not A'.
        alpha: Blend weight for the tint overlay.

    Returns:
        Blended copy of *vis*.
    """
    overlay = np.zeros_like(vis)
    if occlusion_mask is not None:
        overlay[occlusion_mask > 0] = _TINT_OCCLUSION
    if excess_B_mask is not None:
        overlay[excess_B_mask > 0] = _TINT_EXCESS
    if overlay.any():
        vis = cv2.addWeighted(vis, 1.0 - alpha, overlay, alpha, 0)
    return vis


def _draw_segments(
    vis: np.ndarray,
    segments: List[Segment],
    color_map: dict,
    thickness_aligned: int = 2,
    thickness_other: int = 2,
) -> None:
    """Draw polylines for each segment, colored by its dominant class.

    Modifies *vis* in place.

    Args:
        vis: BGR uint8 image.
        segments: List of :class:`~flatsam2.homography_quality.Segment`.
        color_map: Mapping from class label enum to BGR color tuple.
        thickness_aligned: Line thickness for ALIGNED segments.
        thickness_other: Line thickness for all other classes.
    """
    for seg in segments:
        if len(seg.points) < 2:
            continue
        color = color_map.get(seg.label, (200, 200, 200))
        is_aligned = seg.label in (PointClassA.ALIGNED, PointClassB.ALIGNED)
        thickness = thickness_aligned if is_aligned else thickness_other
        pts = seg.points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=thickness)


def _draw_text_overlay(
    vis: np.ndarray,
    result: HomographyExplanation,
) -> None:
    """Draw confidence + fraction summary in the top-left corner.

    Modifies *vis* in place.

    Args:
        vis: BGR uint8 image.
        result: :class:`~flatsam2.homography_quality.HomographyExplanation`.
    """
    lines = [
        f"Confidence: {result.confidence:.3f}   Loss: {result.alignment_loss:.2f}",
        f"Precision:  {result.precision:.1f} px",
        f"A: algn={result.frac_aligned_A:.2f}  occ={result.frac_occluded_A:.2f}"
        f"  int={result.frac_interior_A:.2f}  oof={result.frac_out_of_frame_A:.2f}",
        f"B: algn={result.frac_aligned_B:.2f}  exc={result.frac_excess_B:.2f}"
        f"  oce={result.frac_occl_edge_B:.2f}  fe={result.frac_frame_edge_B:.2f}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    y0 = 28
    for i, line in enumerate(lines):
        y = y0 + i * 22
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(vis, (6, y - th - 3), (10 + tw, y + 4), (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, line, (8, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_quality_overlay(
    frame: np.ndarray,
    result: HomographyExplanation,
) -> np.ndarray:
    """Draw the full quality assessment overlay on a frame.

    Shows region tints, class-colored boundary polylines, and text summary.

    Args:
        frame: BGR uint8 image.
        result: :class:`~flatsam2.homography_quality.HomographyExplanation`.

    Returns:
        New BGR uint8 image with overlays.
    """
    vis = frame.copy()
    vis = _draw_region_tints(vis, result.occlusion_mask, result.excess_B_mask)
    _draw_segments(vis, result.segments_A, _COLOR_A, thickness_aligned=2, thickness_other=2)
    _draw_segments(vis, result.segments_B, _COLOR_B, thickness_aligned=1, thickness_other=1)
    _draw_text_overlay(vis, result)
    return vis


# ---------------------------------------------------------------------------
# Side panel: fraction bars + explanation text
# ---------------------------------------------------------------------------

def _draw_fraction_bar(
    panel: np.ndarray,
    y: int,
    label: str,
    fractions: List[Tuple[float, Tuple[int, int, int]]],
    bar_x: int = 130,
    bar_w: int = 220,
    bar_h: int = 12,
) -> int:
    """Draw a stacked horizontal fraction bar on *panel*.

    Args:
        panel: BGR uint8 panel image (modified in place).
        y: Top y-coordinate for this bar row.
        label: Short text label drawn to the left of the bar.
        fractions: List of ``(fraction, bgr_color)`` pairs; must sum ≈ 1.
        bar_x: X offset where the bar starts.
        bar_w: Total pixel width of the bar.
        bar_h: Height of the bar in pixels.

    Returns:
        Updated y position (below this bar row).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, label, (8, y + bar_h - 2), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    x = bar_x
    for frac, color in fractions:
        w = int(round(frac * bar_w))
        if w > 0:
            cv2.rectangle(panel, (x, y), (x + w, y + bar_h), color, cv2.FILLED)
        x += w
    # Border
    cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (100, 100, 100), 1)
    return y + bar_h + 5


def draw_explanation_panel(
    result: HomographyExplanation,
    panel_width: int = 400,
    frame_height: int = 480,
) -> np.ndarray:
    """Build a side panel showing fraction bars and the explanation text.

    Args:
        result: :class:`~flatsam2.homography_quality.HomographyExplanation`.
        panel_width: Width of the panel in pixels.
        frame_height: Height to match the video frame.

    Returns:
        BGR uint8 panel image of shape ``(frame_height, panel_width, 3)``.
    """
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 18
    cv2.putText(panel, "Boundary classification", (8, y), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    y += 10

    # ∂A' bar
    y = _draw_fraction_bar(panel, y, "dA' (ref):", [
        (result.frac_aligned_A,       _COLOR_A[PointClassA.ALIGNED]),
        (result.frac_occluded_A,      _COLOR_A[PointClassA.OCCLUDED]),
        (result.frac_interior_A,      _COLOR_A[PointClassA.INTERIOR]),
        (result.frac_out_of_frame_A,  _COLOR_A[PointClassA.OUT_OF_FRAME]),
    ])

    # ∂B bar
    y = _draw_fraction_bar(panel, y, "dB (seg):", [
        (result.frac_aligned_B,    _COLOR_B[PointClassB.ALIGNED]),
        (result.frac_excess_B,     _COLOR_B[PointClassB.EXCESS]),
        (result.frac_occl_edge_B,  _COLOR_B[PointClassB.OCCL_EDGE]),
        (result.frac_frame_edge_B, _COLOR_B[PointClassB.FRAME_EDGE]),
    ])

    # Legend
    y += 4
    legend_items_A = [
        (PointClassA.ALIGNED,      "aligned"),
        (PointClassA.OCCLUDED,     "occluded"),
        (PointClassA.INTERIOR,     "interior"),
        (PointClassA.OUT_OF_FRAME, "out-of-frame"),
    ]
    legend_items_B = [
        (PointClassB.ALIGNED,    "aligned"),
        (PointClassB.EXCESS,     "excess"),
        (PointClassB.OCCL_EDGE,  "occl-edge"),
        (PointClassB.FRAME_EDGE, "frame-edge"),
    ]
    for cls, txt in legend_items_A:
        color = _COLOR_A[cls]
        cv2.rectangle(panel, (8, y), (22, y + 10), color, cv2.FILLED)
        cv2.putText(panel, f"∂A' {txt}", (28, y + 9), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 14
    y += 4
    for cls, txt in legend_items_B:
        color = _COLOR_B[cls]
        cv2.rectangle(panel, (8, y), (22, y + 10), color, cv2.FILLED)
        cv2.putText(panel, f"∂B  {txt}", (28, y + 9), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 14

    # Scalar summary
    y += 8
    cv2.line(panel, (8, y), (panel_width - 8, y), (80, 80, 80), 1)
    y += 10
    summary_lines = [
        f"Confidence : {result.confidence:.3f}",
        f"Precision  : {result.precision:.1f} px",
        f"Loss       : {result.alignment_loss:.3f}",
    ]
    for line in summary_lines:
        cv2.putText(panel, line, (8, y), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        y += 18

    # Explanation text (word-wrapped)
    y += 10
    cv2.line(panel, (8, y), (panel_width - 8, y), (80, 80, 80), 1)
    y += 14
    max_chars_per_line = (panel_width - 20) // 7  # rough estimate at scale 0.40
    words = result.explanation.split()
    current_line = ""
    for word in words:
        test = (current_line + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, 0.40, 1)
        if tw > panel_width - 18 and current_line:
            cv2.putText(panel, current_line, (8, y), font, 0.40, (190, 190, 190), 1, cv2.LINE_AA)
            y += 16
            current_line = word
        else:
            current_line = test
        if y > frame_height - 20:
            break
    if current_line and y <= frame_height - 20:
        cv2.putText(panel, current_line, (8, y), font, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Single-frame visualization
# ---------------------------------------------------------------------------

def visualize_frame(
    frame: np.ndarray,
    ref: ReferenceContour,
    seg_mask: np.ndarray,
    homography: np.ndarray,
    with_explanation_panel: bool = True,
) -> Tuple[np.ndarray, HomographyExplanation]:
    """Produce a visualization image for one frame.

    Args:
        frame: BGR uint8 image.
        ref: Pre-computed :class:`~flatsam2.homography_quality.ReferenceContour`.
        seg_mask: uint8 segmentation mask in the current frame.
        homography: 3×3 homography matrix (initial → current frame).
        with_explanation_panel: If True, append a side panel.

    Returns:
        ``(vis_image, HomographyExplanation)`` pair.
    """
    h, w = seg_mask.shape[:2]
    result = assess_homography(ref, seg_mask, homography, (h, w))
    vis = draw_quality_overlay(frame, result)

    if with_explanation_panel:
        panel = draw_explanation_panel(result, panel_width=400, frame_height=vis.shape[0])
        vis = np.hstack([vis, panel])

    return vis, result


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def _load_helpers(dataset: str, base_dir: Optional[Path], initfix_dir: Optional[Path]):
    """Return ``(image_paths_fn, load_gt_fn, base_dir)`` for *dataset*.

    Args:
        dataset: Dataset identifier string.
        base_dir: Override base directory, or None to use path config.
        initfix_dir: Required for ``'pti'`` GT loader.

    Returns:
        Tuple of (image_paths_fn, load_gt_fn, resolved_base_dir).
    """
    ds_info = DATASETS[dataset]
    if base_dir is None:
        base_dir = getattr(paths, ds_info.base_dir_attr)

    if ds_info.gt_loader == "pot":
        return pot_image_paths, pot_load_gt, base_dir
    elif ds_info.gt_loader == "pt":
        return pt_image_paths, pt_load_gt, base_dir
    else:
        if initfix_dir is None:
            initfix_dir = paths.PT_INITFIX_DIR
        return pt_image_paths, lambda bd, sn: pti_load_gt(bd, sn, initfix_dir), base_dir


def visualize_sequence_video(
    dataset: str,
    sequence: str,
    base_results_dir: Path,
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
    fps: int = 10,
) -> Path:
    """Produce an annotated quality video for a full sequence.

    Args:
        dataset: Dataset identifier.
        sequence: Sequence name.
        base_results_dir: Root directory of tracker outputs.
        base_dir: Dataset root directory (None = use path config).
        output_dir: Directory to write the video (None = results_dir/quality_vis).
        config_path: Tracker config for output sub-directory resolution.
        initfix_dir: Required for ``'pti'`` GT loader.
        fps: Output video frame rate.

    Returns:
        Path to the written MP4 file.
    """
    img_paths_fn, load_gt_fn, base_dir = _load_helpers(dataset, base_dir, initfix_dir)
    results_dir = compute_output_dir(base_results_dir, dataset, config_path)

    image_paths = img_paths_fn(base_dir, sequence)
    gt_corners, valid = load_gt_fn(base_dir, sequence)
    pred_flat = np.loadtxt(results_dir / f"{sequence}.txt")
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    pred_corners = rearrange(pred_flat, 'f (c xy) -> f xy c', c=4, xy=2)

    masks_dir = results_dir / 'masks' / sequence
    init_mask = cv2.imread(str(masks_dir / f"{1:05d}.png"), cv2.IMREAD_GRAYSCALE)
    if init_mask is None:
        raise FileNotFoundError(f"Init mask not found: {masks_dir / '00001.png'}")

    # Build reference contour once
    ref = ReferenceContour.build(init_mask)

    if output_dir is None:
        output_dir = results_dir / 'quality_vis'
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sequence}_quality.mp4"

    first_img = cv2.imread(str(image_paths[0]))
    h, w = first_img.shape[:2]
    frame_w = w + 400  # overlay + panel
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (frame_w, h))

    init_corners = gt_corners[0]
    n_frames = min(len(image_paths), len(pred_corners))

    for i in range(n_frames):
        frame = cv2.imread(str(image_paths[i]))
        if frame is None:
            continue

        if i == 0:
            # Init frame — show the reference contour points
            vis = frame.copy()
            pts = ref.points.astype(np.int32)
            for pt in pts[::4]:  # subsample for speed
                cv2.circle(vis, tuple(pt), 1, (50, 220, 50), -1)
            panel = np.zeros((h, 400, 3), dtype=np.uint8)
            cv2.putText(panel, "Init frame — reference contour",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(panel, f"Contour points: {len(ref.points)}",
                        (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, f"Object area: {ref.area:.0f} px",
                        (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            vis = np.hstack([vis, panel])
        else:
            seg_mask = cv2.imread(str(masks_dir / f"{i + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if seg_mask is None:
                seg_mask = np.zeros((h, w), dtype=np.uint8)

            H = homography_from_corners(init_corners, pred_corners[i])
            if H is not None:
                vis, _ = visualize_frame(frame, ref, seg_mask, H)
            else:
                vis = frame.copy()
                panel = np.zeros((h, 400, 3), dtype=np.uint8)
                cv2.putText(panel, "Homography recovery failed", (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                vis = np.hstack([vis, panel])

        writer.write(vis)

    writer.release()
    print(f"Wrote quality video: {out_path}")
    return out_path


def visualize_single_frame(
    dataset: str,
    sequence: str,
    frame_idx: int,
    base_results_dir: Path,
    base_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Produce and optionally display a visualization for a single frame.

    Args:
        dataset: Dataset identifier.
        sequence: Sequence name.
        frame_idx: Zero-based frame index to visualize.
        base_results_dir: Root directory of tracker outputs.
        base_dir: Dataset root (None = use path config).
        config_path: Tracker config for output sub-directory resolution.
        initfix_dir: Required for ``'pti'`` GT loader.
        output_path: Optional file path to save the PNG.
        show: If True, open an OpenCV window.

    Returns:
        Path to the saved PNG, or None if not saved.
    """
    img_paths_fn, load_gt_fn, base_dir = _load_helpers(dataset, base_dir, initfix_dir)
    results_dir = compute_output_dir(base_results_dir, dataset, config_path)

    image_paths = img_paths_fn(base_dir, sequence)
    gt_corners, _ = load_gt_fn(base_dir, sequence)
    pred_flat = np.loadtxt(results_dir / f"{sequence}.txt")
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    pred_corners = rearrange(pred_flat, 'f (c xy) -> f xy c', c=4, xy=2)

    masks_dir = results_dir / 'masks' / sequence
    init_mask = cv2.imread(str(masks_dir / f"{1:05d}.png"), cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(str(masks_dir / f"{frame_idx + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(str(image_paths[frame_idx]))

    if init_mask is None or seg_mask is None or frame is None:
        print(f"Could not load data for frame {frame_idx}")
        return None

    H = homography_from_corners(gt_corners[0], pred_corners[frame_idx])
    if H is None:
        print("Homography recovery failed")
        return None

    ref = ReferenceContour.build(init_mask)
    vis, result = visualize_frame(frame, ref, seg_mask, H)
    print(result.explanation)

    if show:
        cv2.imshow("Homography Quality", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)
        print(f"Saved: {output_path}")
        return output_path

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize homography quality estimation"
    )
    parser.add_argument('dataset', choices=CLI_CHOICES)
    parser.add_argument('sequence', help="Sequence name")
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--initfix-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, help="Output directory for videos/images")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--frame', type=int, help="Visualize a single frame (0-based index)")
    mode.add_argument('--video', action='store_true', help="Produce a full sequence video")

    parser.add_argument('--show', action='store_true', help="Display in a window (single frame only)")
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY

    if args.frame is not None:
        out_path = None
        if args.output_dir:
            out_path = Path(args.output_dir) / f"{args.sequence}_frame{args.frame:05d}.png"
        visualize_single_frame(
            args.dataset, args.sequence, args.frame,
            args.results_dir, args.base_dir, args.config, args.initfix_dir,
            output_path=out_path,
            show=args.show,
        )
    else:
        visualize_sequence_video(
            args.dataset, args.sequence,
            args.results_dir, args.base_dir, args.output_dir,
            args.config, args.initfix_dir,
            fps=args.fps,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
