#!/usr/bin/env python3
"""Frames where the segmentation is right and the mask-derived pose is wrong.

The mask branch fails in a way a mask-quality number cannot show: SAM3 returns a
clean, correctly placed mask, and the quadrilateral fitted to it still names a
pose tens of pixels away. Two mechanisms produce it — an occluder cuts the
silhouette, so the visible region is a smaller, valid-looking quadrilateral, and
a mask that settles on a coplanar distractor of the same appearance. Both are
better shown than tabulated, so this script does two things:

1. **Find candidates.** The shipped run stores both branch homographies every
   frame, so frames where the mask branch is far from ground truth while the
   keypoint branch is on it are readable off ``{seq}_debug.npz`` with no
   re-tracking. That pair of conditions is exactly "the object was found, and
   the mask pose is still wrong".
2. **Render them.** With saved SAM3 masks for those sequences, draw one panel
   per failing frame: ground truth, the SAM3 mask, and the mask-branch pose.
   The mask sits on the object and the quadrilateral does not.

Masks are not stored by a normal run. Dump them for the sequences this reports:

    python slurm/submit.py pot --config configs/tracker/loma_samh_matchnaming.py \\
        --save-masks --output-dir $OUTPUT_DIRECTORY/maskpose_maskdump \\
        --sequences V01_5

Imports cv2 — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/mask_pose_failure.py
    python slurm/submit_cmd.py --gpus 0 -- python experiments/mask_pose_failure.py \\
        --panels V22_5:171 V29_3:251 --results-dir $OUTPUT_DIRECTORY/maskpose_maskdump \\
        --masks-dir $OUTPUT_DIRECTORY/maskpose_maskdump/loma_samh_matchnaming/POT-210/masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2 import hough_homography
from flatsam2.datasets import pot_image_paths, pot_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.sequences import pot_sequences
from paper_colors import GT_RGB, MASK_POSE_RGB, SAM3_MASK_RGB, rgb_to_bgr
from ucsb_failures import draw_quad, overlay_mask

# The shipped tracker; its debug sidecar carries loma_H and samh_H per frame.
_CONFIG = "loma_samh_matchnaming"
_COLORS = {
    "GT": rgb_to_bgr(GT_RGB),
    "mask pose": rgb_to_bgr(MASK_POSE_RGB),
    "SAM3 mask": rgb_to_bgr(SAM3_MASK_RGB),
}


def branch_errors(
    base_dir: Path,
    results_dir: Path,
    seq_name: str,
    config: str = _CONFIG,
) -> List[Tuple[int, float, float]]:
    """Per-frame alignment error of both branches on one POT sequence.

    Args:
        base_dir: POT-210 dataset root.
        results_dir: Root of the results tree.
        seq_name: Sequence name.
        config: Result stem whose ``*_debug.npz`` carries ``loma_H``/``samh_H``.

    Returns:
        List of ``(frame_idx, keypoint_error_px, mask_error_px)`` over frames
        with valid ground truth; a branch that did not run contributes ``inf``.
    """
    npz_path = results_dir / config / "POT-210" / f"{seq_name}_debug.npz"
    if not npz_path.exists():
        return []
    gt, valid = pot_load_gt(base_dir, seq_name)
    with np.load(npz_path, allow_pickle=False) as data:
        frame_idx = data["frame_idx"]
        loma_H, samh_H = data["loma_H"], data["samh_H"]
    rows: List[Tuple[int, float, float]] = []
    for row, i in enumerate(frame_idx):
        if i == 0 or i >= len(gt) or not valid[i]:
            continue
        errs = []
        for H in (loma_H[row], samh_H[row]):
            if not np.all(np.isfinite(H)):
                errs.append(float("inf"))
                continue
            corners = warp_corners(H, gt[0])
            errs.append(float(alignment_error(gt[i:i + 1], corners[None])[0]))
        rows.append((int(i), errs[0], errs[1]))
    return rows


def scan(
    base_dir: Path,
    results_dir: Path,
    sequences: List[str],
    min_mask_error: float,
    max_keypoint_error: float,
) -> List[Tuple[float, str, int, float, float]]:
    """Candidate frames: mask pose far off while the keypoint pose is on target.

    Args:
        base_dir: POT-210 dataset root.
        results_dir: Root of the results tree.
        sequences: Sequence names to scan.
        min_mask_error: Keep frames whose mask-branch error is at least this.
        max_keypoint_error: Keep frames whose keypoint-branch error is at most this.

    Returns:
        ``(mask_error, sequence, frame_idx, keypoint_error, mask_error)`` rows,
        worst mask error first.
    """
    found: List[Tuple[float, str, int, float, float]] = []
    for seq_name in sequences:
        for frame_idx, loma_err, samh_err in branch_errors(
            base_dir, results_dir, seq_name
        ):
            if samh_err >= min_mask_error and loma_err <= max_keypoint_error:
                found.append((samh_err, seq_name, frame_idx, loma_err, samh_err))
    return sorted(found, reverse=True)


def failure_panel(
    base_dir: Path,
    seq_name: str,
    frame_idx: int,
    gt: np.ndarray,
    mask_pose: np.ndarray,
    mask: np.ndarray,
    panel_px: int = 1000,
) -> np.ndarray:
    """One square panel: the failing frame with mask, ground truth and mask pose.

    The template is deliberately not shown. The claim is about one frame — the
    segmentation covers the object and the quadrilateral fitted to it does not —
    and both halves of it are visible in that frame alone.

    Args:
        base_dir: POT-210 dataset root.
        seq_name: Sequence to render.
        frame_idx: Zero-based frame index of the failing frame.
        gt: (N, 2, 4) ground-truth corners.
        mask_pose: (2, 4) corners implied by the mask branch on that frame.
        mask: (H, W) bool SAM3 mask of that frame.
        panel_px: Side length of the square panel.

    Returns:
        (panel_px, panel_px, 3) BGR panel.

    Raises:
        ValueError: When the frame cannot be read.
    """
    current = cv2.imread(str(pot_image_paths(base_dir, seq_name)[frame_idx]))
    if current is None:
        raise ValueError(f"cannot read frame {frame_idx} of {seq_name}")

    current, offset = _crop_square(current, [gt[frame_idx], mask_pose, _mask_extent(mask)])
    x0, y0 = int(offset[0]), int(offset[1])
    mask_crop = mask[y0:y0 + current.shape[0], x0:x0 + current.shape[1]]

    panel = cv2.resize(current, (panel_px, panel_px), interpolation=cv2.INTER_CUBIC)
    mask_panel = cv2.resize(
        mask_crop.astype(np.uint8), (panel_px, panel_px), interpolation=cv2.INTER_NEAREST
    )
    overlay_mask(panel, mask_panel > 0, _COLORS["SAM3 mask"])
    # A correct mask hugs the ground-truth quad, so the thin outline drawn by
    # overlay_mask disappears under it. Widen the lower layer instead: the mask
    # boundary then shows on both sides of the ground truth drawn over it.
    contours, _ = cv2.findContours(
        (mask_panel > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(panel, contours, -1, _COLORS["SAM3 mask"], 14, lineType=cv2.LINE_AA)

    scale = panel_px / current.shape[1]
    # Mask pose first: ground truth must stay visible where the two are close.
    draw_quad(panel, (mask_pose - offset[:, None]) * scale, _COLORS["mask pose"], 6)
    draw_quad(panel, (gt[frame_idx] - offset[:, None]) * scale, _COLORS["GT"], 6)
    return panel


def stage_replay(
    mask: np.ndarray,
    init_corners: np.ndarray,
    last_H: np.ndarray,
) -> Tuple[int, int, int, str]:
    """Re-run the Hough pipeline's stages on one mask and report what survived.

    The paper's claim about this failure is that line extraction succeeds and the
    corner-count fallback is what loses the pose, so the stage counts have to be
    reproducible rather than asserted. Mirrors
    ``hough_homography.mask_to_homography`` at its deployed settings.

    Args:
        mask: (H, W) bool SAM3 mask.
        init_corners: (2, 4) template corners.
        last_H: (3, 3) incumbent homography of the mask branch.

    Returns:
        ``(lines_kept, intersections_near_contour, corners_used, fit)`` where
        ``fit`` is ``'DLT'`` (four or more corners), ``'similarity'`` (two or
        three), ``'translation'`` (one) or ``'incumbent'`` (none).
    """
    contours_cv, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        rearrange(c, "N 1 xy -> xy N", xy=2).astype(float)
        for c in contours_cv if c.shape[0] > 20
    ]
    if not contours:
        return 0, 0, 0, "incumbent"
    on_boundaries = [
        np.logical_or.reduce(tuple(
            hough_homography._approx_touching_edges(c + shift, mask.shape, margin=1)
            for shift in (
                np.array([[0], [1]]), np.array([[0], [-1]]),
                np.array([[1], [0]]), np.array([[-1], [0]]),
            )
        ))
        for c in contours
    ]
    lines = hough_homography.contour_hough(
        mask.astype(np.uint8), contours, on_boundaries, gradient_step=4, topk=4
    )
    refined, line_pts = hough_homography.refine_lines(
        lines, contours, on_boundaries, gradient_step=4, use_orthogonal_fit=False
    )
    supported = np.array([pts.shape[1] > 0 for pts in line_pts])
    refined = refined[:, supported]
    line_pts = [pts for pts, ok in zip(line_pts, supported) if ok]
    long_enough = np.array([
        hough_homography._segment_length(
            hough_homography._line_to_segment(refined[:, i], line_pts[i])
        ) >= 5
        for i in range(refined.shape[1])
    ], dtype=bool)
    refined = refined[:, long_enough]
    if refined.shape[1] < 2:
        return refined.shape[1], 0, 0, "incumbent"
    intersections_h, parents = hough_homography.lines_to_points(refined)
    valid, good = hough_homography.keep_only_points_close_to_contours(
        hough_homography.p2e(intersections_h), contours, max_distance_px=80.0,
        parents=parents, lines=refined, min_angle_deg=0.0,
    )
    matches = hough_homography.discard_diagonal_matches(
        hough_homography.bruteforce_pointset_matching(init_corners, valid),
        [p for p, ok in zip(parents, good) if ok],
        init_corners.shape[1],
    )
    _, _, best_ids = hough_homography.find_closest_H(
        init_corners, valid, last_H, matches
    )
    n_used = len(best_ids[0])
    fit = ("DLT" if n_used >= 4 else "similarity" if n_used > 1
           else "translation" if n_used == 1 else "incumbent")
    return refined.shape[1], valid.shape[1], n_used, fit


def _mask_extent(mask: np.ndarray) -> np.ndarray:
    """(2, 4) bounding box of the mask, or a degenerate box when it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((2, 4))
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    return np.array([[x0, x1, x1, x0], [y0, y0, y1, y1]], dtype=float)


def _crop_square(
    img: np.ndarray,
    quads: List[np.ndarray],
    margin: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Square crop holding every quad plus a margin, clipped to the image.

    Args:
        img: (H, W, 3) image.
        quads: (2, 4) corner arrays to keep in view.
        margin: Extra extent as a fraction of the larger geometry dimension.

    Returns:
        ``(square crop, offset (2,))``; subtract the offset before drawing.
    """
    pts = np.concatenate([q for q in quads if np.any(q)], axis=1)
    lo, hi = pts.min(axis=1), pts.max(axis=1)
    center = 0.5 * (lo + hi)
    side = max(float(np.max(hi - lo)) * (1.0 + 2.0 * margin), 40.0)
    h, w = img.shape[:2]
    side_px = int(min(np.ceil(side), h, w))
    x0 = int(np.clip(round(center[0] - side_px / 2), 0, w - side_px))
    y0 = int(np.clip(round(center[1] - side_px / 2), 0, h - side_px))
    return img[y0:y0 + side_px, x0:x0 + side_px].copy(), np.array([x0, y0], float)


def _mask_pose(
    results_dir: Path,
    seq_name: str,
    frame_idx: int,
    gt0: np.ndarray,
    config: str = _CONFIG,
) -> np.ndarray:
    """Corners the mask branch predicted on one frame.

    Args:
        results_dir: Root of the results tree.
        seq_name: Sequence name.
        frame_idx: Zero-based frame index.
        gt0: (2, 4) initial-frame corners the homography is applied to.
        config: Result stem whose debug sidecar holds ``samh_H``.

    Returns:
        (2, 4) mask-branch corners.

    Raises:
        ValueError: When that frame has no finite mask-branch homography.
    """
    with np.load(
        results_dir / config / "POT-210" / f"{seq_name}_debug.npz", allow_pickle=False
    ) as data:
        rows = np.nonzero(data["frame_idx"] == frame_idx)[0]
        if rows.size == 0:
            raise ValueError(f"{seq_name} has no telemetry for frame {frame_idx}")
        H = data["samh_H"][rows[0]]
    if not np.all(np.isfinite(H)):
        raise ValueError(f"{seq_name} frame {frame_idx} has no mask-branch homography")
    return warp_corners(H, gt0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--min-mask-error", type=float, default=30.0)
    parser.add_argument("--max-keypoint-error", type=float, default=5.0)
    parser.add_argument(
        "--condition", default=None,
        help="Restrict the scan to one POT condition split (e.g. occlusion).",
    )
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--stages", nargs="+", default=None,
        help="Report the Hough stages per frame for SEQ:FIRST:LAST entries.",
    )
    parser.add_argument(
        "--panels", nargs="+", default=None,
        help="Render the publication figure from SEQ:FRAME entries, left to right.",
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="Saved SAM3 masks (<dir>/<seq>/00042.png); required by --panels.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output DIRECTORY (default: <results>/experiments/mask_pose_failure/)",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    base_dir = args.base_dir or paths.POT_BASE
    out_dir = args.out or (results_dir / "experiments" / "mask_pose_failure")

    if args.stages:
        if args.masks_dir is None:
            raise ValueError("--stages needs --masks-dir with saved SAM3 masks")
        for entry in args.stages:
            seq_name, first, last = entry.split(":")
            gt, _ = pot_load_gt(base_dir, seq_name)
            with np.load(
                results_dir / _CONFIG / "POT-210" / f"{seq_name}_debug.npz",
                allow_pickle=False,
            ) as data:
                rows = {int(f): i for i, f in enumerate(data["frame_idx"])}
                samh_H = data["samh_H"]
            print(f"\n{seq_name}: Hough stages per frame")
            print(f"  {'frame':>5} {'quad-vs-mask IoU':>17} {'lines':>6} {'corners':>8} "
                  f"{'used':>5}  fit")
            for frame_idx in range(int(first), int(last) + 1):
                mask_path = args.masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_img is None or frame_idx not in rows:
                    continue
                mask = mask_img > 127
                last_H = samh_H[rows.get(frame_idx - 1, rows[frame_idx])]
                n_lines, n_corners, n_used, fit = stage_replay(mask, gt[0], last_H)
                quad = warp_corners(samh_H[rows[frame_idx]], gt[0])
                quad_mask = np.zeros(mask.shape, np.uint8)
                cv2.fillPoly(quad_mask, [np.round(quad.T).astype(np.int32)], 1)
                overlap = np.logical_and(quad_mask.astype(bool), mask).sum()
                union = np.logical_or(quad_mask.astype(bool), mask).sum()
                print(f"  {frame_idx:5d} {overlap / max(union, 1):17.3f} {n_lines:6d} "
                      f"{n_corners:8d} {n_used:5d}  {fit}")
        return 0

    if args.panels:
        if args.masks_dir is None:
            raise ValueError("--panels needs --masks-dir with saved SAM3 masks")
        panels: List[np.ndarray] = []
        stem_parts: List[str] = []
        for entry in args.panels:
            seq_name, _, frame_text = entry.partition(":")
            if not frame_text:
                raise ValueError(f"expected SEQ:FRAME, got {entry!r}")
            frame_idx = int(frame_text)
            mask_path = args.masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError(f"no saved mask at {mask_path}")
            gt, _ = pot_load_gt(base_dir, seq_name)
            pose = _mask_pose(results_dir, seq_name, frame_idx, gt[0])
            panels.append(
                failure_panel(base_dir, seq_name, frame_idx, gt, pose, mask_img > 127)
            )
            stem_parts.append(f"{seq_name}_f{frame_idx:05d}")
            # The mask comes from the run that also stored the pose being drawn,
            # so the caption's numbers must be re-read from that same run.
            for got_idx, loma_err, samh_err in branch_errors(
                base_dir, results_dir, seq_name
            ):
                if got_idx == frame_idx:
                    print(
                        f"{seq_name} frame {frame_idx}: "
                        f"keypoint {loma_err:.2f} px, mask {samh_err:.2f} px"
                    )
        gap = np.full((panels[0].shape[0], 20, 3), 250, np.uint8)
        strip = panels[0]
        for panel in panels[1:]:
            strip = np.hstack([strip, gap, panel])
        out_path = out_dir / ("__".join(stem_parts) + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), strip)
        print(f"wrote {out_path}")
        return 0

    sequences = pot_sequences(base_dir)
    if args.condition is not None:
        split = base_dir / "splits" / f"{args.condition}.txt"
        keep = set(split.read_text().split())
        sequences = [s for s in sequences if s in keep]
    rows = scan(
        base_dir, results_dir, sequences, args.min_mask_error, args.max_keypoint_error
    )
    print(
        f"POT-210 frames with mask-branch error >= {args.min_mask_error:.0f} px and "
        f"keypoint-branch error <= {args.max_keypoint_error:.0f} px "
        f"({len(rows)} of {len(sequences)} sequences scanned)"
    )
    print(f"  {'sequence':<10} {'frame':>6} {'keypoint px':>12} {'mask px':>10}")
    for _, seq_name, frame_idx, loma_err, samh_err in rows[: args.top]:
        print(f"  {seq_name:<10} {frame_idx:6d} {loma_err:12.2f} {samh_err:10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
