#!/usr/bin/env python3
"""Where the UCSB regression lives, and what it looks like.

UCSB is the one dataset where the plain legacy flow tracker (WOFTSAM) beats both
current trackers (journal 2026-08-17). Two questions this answers, in that
order:

1. **Is the loss spread over the dataset or concentrated?** Concentrated in a
   handful of sequences means a mechanism to find; spread evenly means the
   method is simply weaker here.
2. **What do the failing frames look like?** For the concentrated case, the
   frames themselves are the evidence — GT, the regressed tracker and the
   baseline drawn on the actual image, on the frames where they diverge.

Reads only what is already on disk: each config's ``UCSB/eval_results.json``
for the tables, and its per-sequence ``{seq}.txt`` corner predictions plus the
dataset frames for the figure. No tracking is re-run.

Imports cv2 — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/ucsb_failures.py
    python slurm/submit_cmd.py --gpus 0 -- python experiments/ucsb_failures.py --paper
    python slurm/submit_cmd.py --gpus 0 -- python experiments/ucsb_failures.py --no-figure
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.datasets import corners_txt_image_paths, corners_txt_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from paper_colors import (
    DINOV2_RGB,
    FLATSAM_RGB,
    GT_RGB,
    SAM3_MASK_RGB,
    WOFTSAM_RGB,
    rgb_to_bgr,
)

# The regressed tracker, the baseline that beats it, and the switch variant
# between them. Keyed by the label used in every table below.
_CONFIGS = {
    "WOFTSAM": "woftsam_samfallback_nl",
    "thr35": "loma_samh_decoupled_rerun",
    "quad-free": "corner_fusion_register_final",
}
_REGRESSED = "quad-free"
_BASELINE = "WOFTSAM"
_P_AT_K_PX = 15.0
_PAPER_MASKS_RELATIVE = Path(
    "ucsb_maskdump/loma_samh_decoupled_rerun/UCSB/masks"
)
# Shared paper colors converted to OpenCV BGR order.
_COLORS = {
    "GT": rgb_to_bgr(GT_RGB),
    "WOFTSAM": rgb_to_bgr(WOFTSAM_RGB),
    "thr35": rgb_to_bgr(DINOV2_RGB),
    "quad-free": rgb_to_bgr(FLATSAM_RGB),
    "SAM3 mask": rgb_to_bgr(SAM3_MASK_RGB),
}


def load_eval(results_dir: Path, config: str) -> Tuple[Dict[str, dict], dict]:
    """Per-sequence rows and the aggregate block of one config's UCSB eval.

    Args:
        results_dir: Root of the results tree.
        config: Config stem (directory name under ``results_dir``).

    Returns:
        Tuple ``({sequence: row}, aggregate)``.

    Raises:
        ValueError: When the eval JSON is missing.
    """
    path = results_dir / config / "UCSB" / "eval_results.json"
    if not path.exists():
        raise ValueError(f"No UCSB eval for {config}: {path}")
    with open(path) as fh:
        data = json.load(fh)
    return {row["sequence"]: row for row in data["sequences"]}, data["aggregate"]


def split_name(seq_name: str) -> Tuple[str, str]:
    """UCSB sequence name -> ``(texture, category)``.

    Names are ``<texture>_<category><index>`` (``bricks_motion8``) or
    ``<texture>_<two_word_category>`` (``bricks_dynamic_lighting``).

    Args:
        seq_name: Sequence name.

    Returns:
        Tuple ``(texture, category)``; the category keeps no trailing index, so
        the 9 ``motion`` levels of one texture group together.
    """
    texture, rest = seq_name.split("_", 1)
    return texture, re.sub(r"\d+$", "", rest)


def concentration(deltas: List[Tuple[float, str]]) -> List[Tuple[int, float, float]]:
    """How much of the mean gap the worst k sequences account for.

    Args:
        deltas: ``(delta_pp, sequence)`` pairs, any order.

    Returns:
        List of ``(k, cumulative_pp, share_of_total_percent)``.
    """
    ordered = sorted(d for d, _ in deltas)
    total = sum(ordered) / len(ordered)
    cumulative = np.cumsum(ordered) / len(ordered)
    return [
        (k, float(cumulative[k - 1]), 100.0 * float(cumulative[k - 1]) / total)
        for k in (1, 2, 3, 5, 10, 20)
        if k <= len(ordered)
    ]


def quad_area(corners: np.ndarray) -> float:
    """Shoelace area of one quadrilateral.

    Args:
        corners: (2, 4) corner coordinates in order around the quad.

    Returns:
        Area in square pixels.
    """
    x, y = corners[0], corners[1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def scale_ratios(
    base_dir: Path,
    results_dir: Path,
    sequences: List[str],
) -> Dict[str, float]:
    """Median linear scale of each config's quad against GT.

    The failure frames all show the same thing: the prediction is the right
    shape in the right place but too *large*, because SAM3 segments the sheet
    the target is printed on rather than the annotated target. That is a
    measurable claim — a systematic scale ratio above 1 — and this is the
    measurement. sqrt(area) puts it in linear (per-side) units.

    Args:
        base_dir: UCSB dataset root.
        results_dir: Root of the results tree.
        sequences: Sequence names to pool over.

    Returns:
        ``{config label: median sqrt(area_pred / area_gt)}``.
    """
    seq_info = DATASETS["ucsb"]
    ratios: Dict[str, List[float]] = {label: [] for label in _CONFIGS}
    shifts: Dict[str, List[float]] = {label: [] for label in _CONFIGS}
    for seq_name in sequences:
        gt, valid = corners_txt_load_gt(
            base_dir, seq_name, seq_info.seq_subdir, seq_info.gt_subdir
        )
        for label, cfg in _CONFIGS.items():
            pred_path = results_dir / cfg / "UCSB" / f"{seq_name}.txt"
            if not pred_path.exists():
                continue
            pred = np.loadtxt(pred_path).reshape(-1, 4, 2).transpose(0, 2, 1)
            n = min(len(gt), len(pred))
            err = alignment_error(gt[:n], pred[:n])
            for i in range(n):
                # Conditioned on the frame actually failing: a median over all
                # frames is dominated by the frames that worked and says nothing
                # about the failure mode.
                if not valid[i] or err[i] <= _P_AT_K_PX:
                    continue
                area_gt = quad_area(gt[i])
                if area_gt <= 0:
                    continue
                ratios[label].append(float(np.sqrt(quad_area(pred[i]) / area_gt)))
                shifts[label].append(
                    float(
                        np.linalg.norm(pred[i].mean(axis=1) - gt[i].mean(axis=1))
                        / np.sqrt(area_gt)
                    )
                )
    out: Dict[str, float] = {}
    for label in _CONFIGS:
        out[f"{label} scale"] = float(np.median(ratios[label])) if ratios[label] else float("nan")
        out[f"{label} shift"] = float(np.median(shifts[label])) if shifts[label] else float("nan")
        out[f"{label} n"] = float(len(ratios[label]))
    return out


def branch_attribution(
    base_dir: Path,
    results_dir: Path,
    sequences: List[str],
    config: str = "loma_samh_decoupled_rerun",
) -> None:
    """Is UCSB lost by the branches, or by the choice between them?

    The scale measurement says the failing frames are over-large quads, which
    points at the mask branch. That is an argument, not a measurement: the run
    stores *both* branch homographies every frame, so the question can be
    answered directly. If the keypoint branch is inside 15 px on frames the
    tracker got wrong, the evidence was present and the selection rule discarded
    it — a rule problem. If neither branch is inside 15 px, the information was
    never there — a segmentation-scope problem, and nothing about arbitration
    can recover it.

    Args:
        base_dir: UCSB dataset root.
        results_dir: Root of the results tree.
        sequences: Sequence names to pool over.
        config: Result stem whose ``*_debug.npz`` carries ``loma_H``/``samh_H``.
    """
    seq_info = DATASETS["ucsb"]
    # Pooled over frames, because the per-sequence mean weights a 7-frame
    # sequence like a 200-frame one and UCSB has both.
    n_frames = n_shipped_ok = 0
    branch_ok = {"loma": 0, "samh": 0, "either": 0}
    fail_branch_ok = {"loma": 0, "samh": 0, "either": 0, "neither": 0}
    selected_on_fail: Dict[str, int] = defaultdict(int)
    for seq_name in sequences:
        npz_path = results_dir / config / "UCSB" / f"{seq_name}_debug.npz"
        pred_path = results_dir / config / "UCSB" / f"{seq_name}.txt"
        if not npz_path.exists() or not pred_path.exists():
            continue
        gt, valid = corners_txt_load_gt(
            base_dir, seq_name, seq_info.seq_subdir, seq_info.gt_subdir
        )
        pred = np.loadtxt(pred_path).reshape(-1, 4, 2).transpose(0, 2, 1)
        with np.load(npz_path, allow_pickle=False) as data:
            frame_idx = data["frame_idx"]
            branch_H = {"loma": data["loma_H"], "samh": data["samh_H"]}
            selected = data["selected"]
        n = min(len(gt), len(pred))
        shipped_err = alignment_error(gt[:n], pred[:n])
        for row, i in enumerate(frame_idx):
            if i >= n or not valid[i] or i == 0:
                continue
            errs = {}
            for name, H in branch_H.items():
                if not np.all(np.isfinite(H[row])):
                    continue
                corners = warp_corners(H[row], gt[0])
                errs[name] = float(alignment_error(gt[i:i + 1], corners[None])[0])
            n_frames += 1
            ok = {name: e <= _P_AT_K_PX for name, e in errs.items()}
            for name in ("loma", "samh"):
                branch_ok[name] += int(ok.get(name, False))
            branch_ok["either"] += int(any(ok.values()))
            if shipped_err[i] <= _P_AT_K_PX:
                n_shipped_ok += 1
                continue
            selected_on_fail[str(selected[row])] += 1
            for name in ("loma", "samh"):
                fail_branch_ok[name] += int(ok.get(name, False))
            fail_branch_ok["either"] += int(any(ok.values()))
            fail_branch_ok["neither"] += int(not any(ok.values()))

    if not n_frames:
        print("\nbranch attribution: no debug telemetry on disk")
        return
    n_fail = n_frames - n_shipped_ok
    print(f"\nbranch attribution ({config}), pooled over {n_frames} annotated frames")
    print(f"  shipped within {_P_AT_K_PX:.0f} px  {100 * n_shipped_ok / n_frames:6.2f}%")
    for name in ("loma", "samh", "either"):
        print(f"  {name + ' alone':<18} {100 * branch_ok[name] / n_frames:6.2f}%")
    print(f"\n  on the {n_fail} frames the tracker got wrong:")
    if n_fail:
        for name in ("loma", "samh", "either", "neither"):
            print(f"    {name:<10} within 15 px  {100 * fail_branch_ok[name] / n_fail:6.2f}%")
        chosen = ", ".join(f"{k}={v}" for k, v in sorted(selected_on_fail.items()))
        print(f"    branch actually selected: {chosen}")
    # The headline: an oracle that picks the better branch every frame bounds
    # what any selection rule, however clever, could reach on this dataset.
    print(f"\n  oracle branch selection would reach {100 * branch_ok['either'] / n_frames:.2f}% "
          f"vs WOFTSAM's pooled 96.30%")


def worst_frame(
    gt: np.ndarray,
    valid: np.ndarray,
    pred_bad: np.ndarray,
    pred_good: np.ndarray,
) -> Optional[int]:
    """Frame index where the regressed tracker fails hardest while the baseline holds.

    Args:
        gt: (N, 2, 4) ground-truth corners.
        valid: (N,) GT-availability flags.
        pred_bad: (M, 2, 4) regressed tracker's corners.
        pred_good: (M, 2, 4) baseline's corners.

    Returns:
        Frame index, or None when no frame separates the two.
    """
    n = min(len(gt), len(pred_bad), len(pred_good))
    err_bad = alignment_error(gt[:n], pred_bad[:n])
    err_good = alignment_error(gt[:n], pred_good[:n])
    separating = valid[:n] & (err_good <= _P_AT_K_PX) & (err_bad > _P_AT_K_PX)
    if not separating.any():
        return None
    return int(np.argmax(np.where(separating, err_bad, -np.inf)))


def draw_quad(canvas: np.ndarray, corners: np.ndarray, color, thickness: int) -> None:
    """Draw one closed quadrilateral, unlabelled.

    Labels are deliberately not drawn on the image: the legend lives in the
    caption band so nothing is written over the object under inspection.

    Args:
        canvas: (H, W, 3) BGR image, modified in place.
        corners: (2, 4) corner coordinates.
        color: BGR tuple.
        thickness: Stroke width in pixels.
    """
    pts = np.round(corners.T).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)
    # A filled dot on corner 0 shows the labelling, which is what a cyclic
    # mis-naming changes and an outline alone cannot show.
    cv2.circle(canvas, tuple(np.round(corners[:, 0]).astype(int)), thickness * 3,
               color, -1, lineType=cv2.LINE_AA)


def overlay_mask(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    """Tint the segmentation mask and outline it, leaving the image readable."""
    if mask is None or mask.shape[:2] != canvas.shape[:2]:
        return
    sel = mask.astype(bool)
    canvas[sel] = (0.65 * canvas[sel] + 0.35 * np.array(color, np.float32)).astype(np.uint8)
    contours, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 2, lineType=cv2.LINE_AA)


def _caption(width: int, lines: List[Tuple[str, tuple]], height: int) -> np.ndarray:
    """Caption band: coloured legend keys and text, off the imagery."""
    band = np.full((height, width, 3), 22, np.uint8)
    x, y = 24, 40
    for text, color in lines:
        if color is not None:
            cv2.rectangle(band, (x, y - 20), (x + 46, y + 4), color, -1)
            cv2.putText(band, text, (x + 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (245, 245, 245), 2, cv2.LINE_AA)
            x += 60 + 20 * len(text)
        else:
            cv2.putText(band, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2, cv2.LINE_AA)
            y += 46
            x = 24
        if x > width - 400:
            x, y = 24, y + 46
    return band


def _crop_around(
    img: np.ndarray,
    quads: List[np.ndarray],
    margin: float = 1.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop to the object plus context, so the quads are big enough to judge.

    A UCSB target often covers a small part of the frame; at full frame the
    difference between a right and a wrong pose is a few pixels on screen. The
    crop keeps ``margin`` times the quad extent around it, which leaves enough
    surroundings to see what the tracker latched onto.

    Args:
        img: (H, W, 3) image.
        quads: (2, 4) corner arrays to keep in view.
        margin: Extra extent as a fraction of the quad's size, per side.

    Returns:
        ``(cropped image, offset (2,))``; subtract the offset from any point
        before drawing it on the crop.
    """
    pts = np.concatenate(quads, axis=1)
    lo = pts.min(axis=1)
    hi = pts.max(axis=1)
    size = np.maximum(hi - lo, 40.0)
    lo = np.floor(lo - margin * size)
    hi = np.ceil(hi + margin * size)
    h, w = img.shape[:2]
    x0, y0 = int(max(0, lo[0])), int(max(0, lo[1]))
    x1, y1 = int(min(w, hi[0])), int(min(h, hi[1]))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return img, np.zeros(2)
    return img[y0:y1, x0:x1].copy(), np.array([x0, y0], float)


def _crop_square_around(
    img: np.ndarray,
    quads: List[np.ndarray],
    margin: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a square around geometry while preserving nearby context.

    Args:
        img: (H, W, 3) image.
        quads: (2, 4) corner arrays to keep in view.
        margin: Extra extent as a fraction of the larger geometry dimension.

    Returns:
        ``(square crop, offset (2,))``.
    """
    pts = np.concatenate(quads, axis=1)
    lo = pts.min(axis=1)
    hi = pts.max(axis=1)
    center = 0.5 * (lo + hi)
    side = max(float(np.max(hi - lo)) * (1.0 + 2.0 * margin), 40.0)
    h, w = img.shape[:2]
    side_px = int(min(np.ceil(side), h, w))
    x0 = int(np.clip(round(center[0] - side_px / 2), 0, w - side_px))
    y0 = int(np.clip(round(center[1] - side_px / 2), 0, h - side_px))
    return img[y0:y0 + side_px, x0:x0 + side_px].copy(), np.array([x0, y0], float)


def _mask_bounds(mask: np.ndarray) -> Optional[np.ndarray]:
    """Bounding quadrilateral of the largest connected mask component."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return np.array(
        [[x, x + width, x + width, x], [y, y, y + height, y + height]],
        dtype=float,
    )


def failure_figure(
    base_dir: Path,
    results_dir: Path,
    seq_name: str,
    frame_idx: int,
    gt: np.ndarray,
    preds: Dict[str, np.ndarray],
    out_path: Path,
    masks_dir: Optional[Path],
    paper: bool = False,
    min_panel_px: int = 1000,
) -> Optional[Path]:
    """One PNG for one (sequence, frame): template beside the failing frame.

    Both panels carry the ground truth, so a wrong pose is read against the
    shape the tracker was initialised on. Nothing is annotated over the imagery
    — the legend is a band underneath — because the point of the figure is to
    let the eye judge the geometry.

    Args:
        base_dir: UCSB dataset root.
        results_dir: Root of the results tree.
        seq_name: Sequence to render.
        frame_idx: Frame to render beside the template.
        gt: (N, 2, 4) ground-truth corners.
        preds: ``{label: (M, 2, 4)}`` predictions to draw, by config label.
        out_path: PNG destination.
        masks_dir: Directory of saved SAM3 masks for this sequence, or None.
        paper: Draw only GT in both panels and the SAM3 mask in the current panel.
        min_panel_px: Each panel is upscaled to at least this width.

    Returns:
        The written path, or None when the frames could not be read.

    Raises:
        ValueError: When paper mode has no saved mask for the selected frame.
    """
    seq_info = DATASETS["ucsb"]
    paths = corners_txt_image_paths(base_dir, seq_name, seq_info.seq_subdir)
    if frame_idx >= len(paths):
        return None
    template = cv2.imread(str(paths[0]))
    current = cv2.imread(str(paths[frame_idx]))
    if template is None or current is None:
        return None

    mask_img = None
    if masks_dir is not None:
        mask_path = masks_dir / f"{frame_idx + 1:05d}.png"
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if paper and mask_img is None:
        raise ValueError(f"paper figure requires SAM3 mask for frame {frame_idx}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if paper:
        mask = mask_img > 127
        current_quads = [gt[frame_idx]]
        mask_quad = _mask_bounds(mask)
        if mask_quad is not None:
            current_quads.append(mask_quad)
        current, offset = _crop_square_around(current, current_quads)
        template, t_offset = _crop_square_around(template, [gt[0]])

        x0, y0 = int(offset[0]), int(offset[1])
        mask_crop = mask[y0:y0 + current.shape[0], x0:x0 + current.shape[1]]
        panel_size = min_panel_px
        current_panel = cv2.resize(
            current, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC
        )
        template_panel = cv2.resize(
            template, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC
        )
        mask_panel = cv2.resize(
            mask_crop.astype(np.uint8),
            (panel_size, panel_size),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay_mask(current_panel, mask_panel > 0, _COLORS["SAM3 mask"])

        template_scale = panel_size / template.shape[1]
        current_scale = panel_size / current.shape[1]
        draw_quad(
            template_panel,
            (gt[0] - t_offset[:, None]) * template_scale,
            _COLORS["GT"],
            6,
        )
        draw_quad(
            current_panel,
            (gt[frame_idx] - offset[:, None]) * current_scale,
            _COLORS["GT"],
            6,
        )
        gap = np.full((panel_size, 20, 3), 250, np.uint8)
        cv2.imwrite(str(out_path), np.hstack([template_panel, gap, current_panel]))
        return out_path

    here = [gt[frame_idx]]
    here += [p[frame_idx] for p in preds.values() if frame_idx < len(p)]
    current, offset = _crop_around(current, here)
    if mask_img is not None:
        mask_crop = mask_img[
            int(offset[1]): int(offset[1]) + current.shape[0],
            int(offset[0]): int(offset[0]) + current.shape[1],
        ]
        overlay_mask(current, mask_crop > 127, _COLORS["SAM3 mask"])
    template, t_offset = _crop_around(template, [gt[0]])

    draw_quad(template, gt[0] - t_offset[:, None], _COLORS["GT"], 4)
    # Predictions first, ground truth last: GT must never be hidden under a
    # prediction that happens to be close to it.
    for label, pred in preds.items():
        if frame_idx < len(pred):
            thickness = 9 if label == _BASELINE else 6
            draw_quad(
                current,
                pred[frame_idx] - offset[:, None],
                _COLORS[label],
                thickness,
            )
    draw_quad(current, gt[frame_idx] - offset[:, None], _COLORS["GT"], 4)

    scale = max(1.0, min_panel_px / max(template.shape[1], 1))
    if scale > 1.0:
        template = cv2.resize(
            template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        current = cv2.resize(
            current, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    panels = []
    target_height = max(template.shape[0], current.shape[0])
    for img in (template, current):
        resize = target_height / max(img.shape[0], 1)
        target_width = max(1, int(round(img.shape[1] * resize)))
        interpolation = cv2.INTER_CUBIC if resize > 1.0 else cv2.INTER_AREA
        panels.append(
            cv2.resize(img, (target_width, target_height), interpolation=interpolation)
        )
    gap = np.full((target_height, 16, 3), 22, np.uint8)
    composition = np.hstack([panels[0], gap, panels[1]])

    errs = {
        label: float(
            alignment_error(
                gt[frame_idx: frame_idx + 1], pred[frame_idx: frame_idx + 1]
            )[0]
        )
        for label, pred in preds.items() if frame_idx < len(pred)
    }
    legend: List[Tuple[str, tuple]] = [("GT", _COLORS["GT"])]
    legend += [
        (f"{label} {errs[label]:.0f}px", _COLORS[label])
        for label in preds if label in errs
    ]
    if masks_dir is not None:
        legend.append(("SAM3 mask", _COLORS["SAM3 mask"]))
    legend.append((f"{seq_name}   template (left)  vs  frame {frame_idx} (right)", None))
    band = _caption(composition.shape[1], legend, height=110)
    cv2.imwrite(str(out_path), np.vstack([composition, band]))
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument(
        "--min-deficit", type=float, default=-3.0,
        help="Render every sequence whose p@15 deficit vs the baseline is at most this.",
    )
    parser.add_argument(
        "--masks-dir", type=Path, default=None,
        help="Directory of saved SAM3 masks (<dir>/<seq>/00042.png) to overlay.",
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Render publication figures without the diagnostic legend band.",
    )
    parser.add_argument("--paper-sequence", default="bricks_perspective")
    parser.add_argument(
        "--paper-frame",
        type=int,
        default=44,
        help="Zero-based frame index for the publication figure.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output DIRECTORY (default: <results>/experiments/ucsb_failures/)",
    )
    args = parser.parse_args(argv)

    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    base_dir = args.base_dir or paths.UCSB_BASE
    if args.paper and args.masks_dir is None:
        args.masks_dir = results_dir / _PAPER_MASKS_RELATIVE
    if args.paper and not args.masks_dir.is_dir():
        raise ValueError(f"paper figure requires saved SAM3 masks: {args.masks_dir}")
    # A config whose results have been deleted (the corner_fusion re-runs of
    # 2026-08-19) drops out of every table rather than killing the script.
    for label, cfg in list(_CONFIGS.items()):
        if not (results_dir / cfg / "UCSB" / "eval_results.json").exists():
            print(f"(skipping {label}: no UCSB eval for {cfg})")
            del _CONFIGS[label]
    if _BASELINE not in _CONFIGS:
        raise ValueError(f"baseline {_BASELINE} has no UCSB eval; nothing to compare against")
    rows = {label: load_eval(results_dir, cfg) for label, cfg in _CONFIGS.items()}
    aggregates = {label: rows[label][1] for label in _CONFIGS}

    # 1. Global first: the whole-dataset numbers the regression was reported as.
    print("UCSB, whole dataset")
    print(f"  {'config':<12} {'pooled p@15':>12} {'mean p@15':>10} {'mean err px':>12}")
    for label in _CONFIGS:
        agg = aggregates[label]
        print(
            f"  {label:<12} {agg['overall_p@15']:12.2f} {agg['mean_p@15']:10.2f} "
            f"{agg['overall_mean_error']:12.2f}"
        )
    global _REGRESSED
    if _REGRESSED not in _CONFIGS:
        # Every downstream table, and the figure's colour key, is written in
        # terms of this label; rebind it once rather than threading a parameter.
        replacement = next(l for l in _CONFIGS if l != _BASELINE)
        _REGRESSED = replacement
    regressed = _REGRESSED
    per_seq = {label: rows[label][0] for label in _CONFIGS}
    common = sorted(set.intersection(*(set(v) for v in per_seq.values())))
    deltas = [
        (per_seq[regressed][s]["p@15"] - per_seq[_BASELINE][s]["p@15"], s) for s in common
    ]
    gap = sum(d for d, _ in deltas) / len(deltas)
    n_worse = sum(1 for d, _ in deltas if d < -0.01)
    print(
        f"\n  {regressed} vs {_BASELINE}: mean gap {gap:+.2f} pp over {len(common)} sequences; "
        f"{n_worse} are worse, {len(common) - n_worse} are equal or better."
    )

    # 1b. Was the evidence there and discarded, or never there at all?
    branch_attribution(base_dir, results_dir, common)

    # 2. Is it concentrated?
    print("\nConcentration (share of the mean gap carried by the worst k sequences)")
    for k, cumulative, share in concentration(deltas):
        print(f"  worst {k:2d}: {cumulative:+6.2f} pp of {gap:+.2f} pp  ({share:5.1f}%)")

    # 3. Which sequences, and do they share a category?
    print("\nWorst sequences")
    print(f"  {'delta':>7} {'sequence':<26} {'nGT':>5} " + " ".join(f"{l:>10}" for l in _CONFIGS))
    for delta, seq in sorted(deltas)[:12]:
        cells = " ".join(f"{per_seq[l][seq]['p@15']:10.2f}" for l in _CONFIGS)
        print(f"  {delta:7.2f} {seq:<26} {per_seq[_BASELINE][seq]['num_valid']:5d} {cells}")

    groups: Dict[str, List[str]] = defaultdict(list)
    textures: Dict[str, List[str]] = defaultdict(list)
    for seq in common:
        texture, category = split_name(seq)
        groups[category].append(seq)
        textures[texture].append(seq)

    for title, grouping in (("category", groups), ("texture", textures)):
        print(f"\nBy {title}")
        print(f"  {title:<18} {'n':>3} {'nGT':>6} " + " ".join(f"{l:>10}" for l in _CONFIGS)
              + f" {'delta':>8}")
        ordered = sorted(
            grouping.items(),
            key=lambda kv: np.mean(
                [per_seq[regressed][s]["p@15"] - per_seq[_BASELINE][s]["p@15"] for s in kv[1]]
            ),
        )
        for name, seqs in ordered:
            means = {
                label: float(np.mean([per_seq[label][s]["p@15"] for s in seqs]))
                for label in _CONFIGS
            }
            n_gt = sum(per_seq[_BASELINE][s]["num_valid"] for s in seqs)
            cells = " ".join(f"{means[l]:10.2f}" for l in _CONFIGS)
            print(
                f"  {name:<18} {len(seqs):3d} {n_gt:6d} {cells} "
                f"{means[_REGRESSED] - means[_BASELINE]:+8.2f}"
            )

    # 4. Is the error a systematic over-scaling? (the failure frames say so)
    worst_seqs = [seq for _, seq in sorted(deltas)[:10]]
    best_seqs = [seq for _, seq in sorted(deltas)[-10:]]
    print("\nOn failing frames only (error > 15 px): is the quad too big, or displaced?")
    print("  scale = median sqrt(area_pred/area_gt); shift = median centroid offset "
          "in units of sqrt(GT area)")
    header = " ".join(f"{l + ' scale':>16} {l + ' shift':>16} {l + ' n':>10}" for l in _CONFIGS)
    print(f"  {'sequences':<12} {header}")
    for title, seqs in (("worst 10", worst_seqs), ("best 10", best_seqs), ("all", common)):
        got = scale_ratios(base_dir, results_dir, seqs)
        cells = " ".join(
            f"{got[l + ' scale']:16.3f} {got[l + ' shift']:16.3f} {got[l + ' n']:10.0f}"
            for l in _CONFIGS
        )
        print(f"  {title:<12} {cells}")

    # 5. The frames themselves: one PNG per problematic sequence, so the cause
    #    can be judged at a glance rather than inferred from a table.
    if args.no_figure:
        return 0
    out_dir = args.out or (results_dir / "experiments" / "ucsb_failures")
    seq_info = DATASETS["ucsb"]
    if args.paper:
        delta_by_sequence = {seq: delta for delta, seq in deltas}
        if args.paper_sequence not in delta_by_sequence:
            raise ValueError(f"paper sequence not available: {args.paper_sequence}")
        problematic = [(delta_by_sequence[args.paper_sequence], args.paper_sequence)]
    else:
        problematic = [(d, seq) for d, seq in sorted(deltas) if d <= args.min_deficit]
    print(f"\nRendering {len(problematic)} sequences with a deficit <= {args.min_deficit} pp"
          f" -> {out_dir}")
    labels = [l for l in (_BASELINE, regressed) if l in _CONFIGS]
    written = 0
    for delta, seq_name in problematic:
        gt, valid = corners_txt_load_gt(
            base_dir, seq_name, seq_info.seq_subdir, seq_info.gt_subdir
        )
        preds = {}
        for label in labels:
            pred_path = results_dir / _CONFIGS[label] / "UCSB" / f"{seq_name}.txt"
            if pred_path.exists():
                preds[label] = np.loadtxt(pred_path).reshape(-1, 4, 2).transpose(0, 2, 1)
        if len(preds) < 2:
            continue
        if args.paper:
            idx = args.paper_frame
            if idx < 0 or idx >= len(gt) or not valid[idx]:
                raise ValueError(f"invalid paper frame {idx} for {seq_name}")
        else:
            idx = worst_frame(gt, valid, preds[regressed], preds[_BASELINE])
        if idx is None:
            print(f"  (no separating frame) {seq_name}")
            continue
        masks_dir = None
        if args.masks_dir is not None:
            candidate = args.masks_dir / seq_name
            masks_dir = candidate if candidate.is_dir() else None
        out_path = out_dir / f"{seq_name}__f{idx:05d}.png"
        got = failure_figure(
            base_dir, results_dir, seq_name, idx, gt, preds, out_path, masks_dir,
            paper=args.paper,
        )
        if got is not None:
            written += 1
            print(f"  {delta:+7.2f} pp  {seq_name:<28} frame {idx:>4}  -> {got.name}")
    print(f"\nWrote {written} figures to {out_dir}")
    if args.masks_dir is None:
        print("  (no --masks-dir given, so the SAM3 mask is not drawn; "
              "UCSB masks are not saved by default)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
