"""Visualization utilities for FlatSAM2.

Produces a video for a given sequence with tracking overlays.
"""
from pathlib import Path
from typing import Optional
import argparse
import numpy as np
import cv2
from einops import rearrange

from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    make_gt_loader,
    make_image_paths,
    warp_corners,
)
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.config import compute_output_dir


# BGR colours; the legend lists only the layers present in a given video
COLOR_GT = (0, 255, 0)  # green, dashed and drawn last so it is never hidden
COLOR_OUTPUT = (255, 0, 0)  # blue
COLOR_LOMA = (255, 255, 0)  # cyan
COLOR_SAMH = (200, 50, 150)  # purple
COLOR_MASK = (0, 165, 255)  # orange
COLOR_COMPARE = (0, 255, 255)  # yellow
OUTPUT_LABEL = "LoMaSAM (output)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_quad(img: np.ndarray, corners: np.ndarray, color: tuple, thickness: int) -> None:
    """Draw a closed quad given (2, 4) corners onto a BGR image in place."""
    pts = rearrange(corners, 'xy corners -> corners 1 xy').round().astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_dashed_quad(img: np.ndarray, corners: np.ndarray, color: tuple, thickness: int,
                     dash: int = 8) -> None:
    """Draw a closed dashed quad from (2, 4) corners; the gaps let layers below show through."""
    pts = rearrange(corners, 'xy corners -> corners xy')
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n = max(int(np.linalg.norm(b - a) // dash), 1)
        ts = np.linspace(0.0, 1.0, n + 1)
        for k in range(0, n, 2):
            p, q = a + (b - a) * ts[k], a + (b - a) * ts[k + 1]
            cv2.line(img, tuple(map(int, p.round())), tuple(map(int, q.round())),
                     color, thickness, cv2.LINE_AA)


def put_text(img: np.ndarray, text: str, org: tuple, color: tuple, scale: float) -> None:
    """Draw text with a black outline so it reads on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 2, cv2.LINE_AA)


def draw_output(img: np.ndarray, corners: np.ndarray) -> None:
    """Draw the tracker output: thick quad, big dot on corner 0 so the naming is visible."""
    draw_quad(img, corners, COLOR_OUTPUT, 2)
    for corner_i, (x, y) in enumerate(corners.T):
        cv2.circle(img, (int(x), int(y)), 8 if corner_i == 0 else 4, COLOR_OUTPUT, -1)


def draw_legend(img: np.ndarray, items: list, y: int = 25) -> None:
    """Draw top-left legend labels, each in its layer's colour, wrapping at the frame edge.

    Args:
        img: BGR image, drawn on in place.
        items: List of (label, BGR colour).
        y: Baseline of the first legend row.
    """
    scale, x = 0.6, 10
    for label, color in items:
        width = cv2.getTextSize(label, FONT, scale, 2)[0][0]
        if x > 10 and x + width > img.shape[1] - 10:
            x, y = 10, y + 24
        put_text(img, label, (x, y), color, scale)
        x += width + 15


def with_error(label: str, gt: Optional[np.ndarray], track: np.ndarray, i: int) -> str:
    """Append frame ``i``'s corner error vs GT (2, 4) to a legend label, when both exist."""
    if gt is None or i >= len(track):
        return label
    return f"{label} {alignment_error(gt, track[i]):.1f}px"


def load_corners(pred_file: Path) -> np.ndarray:
    """Load a ``<seq>.txt`` prediction file (8 floats per line) as (N, 2, 4) corners."""
    if not pred_file.exists():
        raise ValueError(f"Prediction file not found: {pred_file}")
    pred_flat = np.atleast_2d(np.loadtxt(pred_file))
    return rearrange(pred_flat, 'frames (corners xy) -> frames xy corners', corners=4, xy=2)


def visualize_sequence(dataset: str,
                       base_results_dir: Path,
                       sequence: str,
                       base_dir: Optional[Path],
                       base_output_dir: Path,
                       config_path: Optional[Path] = None,
                       fps: int = 10,
                       draw_masks: bool = True,
                       compare_dir: Optional[Path] = None) -> None:
    """Render a video with tracking overlays for one sequence.

    Args:
        dataset: Any accepted dataset CLI key (see flatsam2.registry)
        base_results_dir: Top-level directory containing tracking results
        sequence: Sequence name to visualize
        base_dir: Dataset base directory (optional, uses defaults)
        base_output_dir: Top-level output directory for videos
        config_path: Path to config file (for directory structure)
        fps: Frames per second for the output video
        draw_masks: If True, overlay the run's saved SAM3 masks when available
        compare_dir: Directory with another tracker's ``<seq>.txt``, drawn as an extra quad
    """
    # Compute structured paths
    results_dir = compute_output_dir(base_results_dir, dataset, config_path).expanduser()
    output_dir = compute_output_dir(base_output_dir, dataset, config_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    imgs_base = (
        Path(base_dir) if base_dir is not None
        else getattr(paths, DATASETS[dataset].base_dir_attr)
    )
    image_paths = make_image_paths(dataset)(imgs_base, sequence)

    preds = load_corners(results_dir / f"{sequence}.txt")
    # Another tracker on the same sequence, e.g. a baseline; legend shows its config stem
    compare = load_corners(compare_dir / f"{sequence}.txt") if compare_dir is not None else None

    # Prepare video writer
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = output_dir / f"{sequence}.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    # Ground truth, when the dataset provides it
    try:
        gt_corners, gt_valid = make_gt_loader(dataset)(imgs_base, sequence)  # (N, 2, 4), (N,)
    except (ValueError, FileNotFoundError, OSError):
        gt_corners, gt_valid = None, None

    # Branch quads and per-frame telemetry from the run's debug sidecar (NaN H = no estimate)
    branch_corners = {"loma_H": {}, "samh_H": {}}
    frame_info = {}
    debug_file = results_dir / f"{sequence}_debug.npz"
    if debug_file.exists():
        debug = np.load(debug_file)
        for key, per_frame in branch_corners.items():
            if key in debug:
                for frame_i, H in zip(debug["frame_idx"], debug[key]):
                    if np.isfinite(H).all():
                        per_frame[int(frame_i)] = warp_corners(H, preds[0])
        if "selected" in debug and "n_inliers" in debug:
            rows = zip(debug["frame_idx"], debug["selected"], debug["n_inliers"])
            for frame_i, sel, n_inl in rows:
                frame_info[int(frame_i)] = f"sel={sel}  inliers={n_inl}"

    # SAM3 masks, written by run.py --save-masks
    masks_dir = results_dir / "masks" / sequence
    if draw_masks and not masks_dir.is_dir():
        print(f"No SAM3 masks at {masks_dir}; track with --save-masks to get them")
    draw_masks = draw_masks and masks_dir.is_dir()

    # Static legend part; the tracks' labels carry the per-frame error and are added per frame
    layers = []
    if gt_corners is not None:
        layers.append(("GT", COLOR_GT))
    if draw_masks:
        layers.append(("SAM3 mask", COLOR_MASK))
    if branch_corners["samh_H"]:
        layers.append(("SAM-H", COLOR_SAMH))
    if branch_corners["loma_H"]:
        layers.append(("LoMa", COLOR_LOMA))

    last = len(image_paths) - 1
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: failed to load {img_path}, skipping frame")
            continue

        # Mask first so the quads stay visible on top of it
        if draw_masks:
            mask_file = masks_dir / f"{i+1:05d}.png"
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    m = mask > 0  # tint only mask pixels, 30 % orange
                    img[m] = (0.7 * img[m] + 0.3 * np.array(COLOR_MASK)).astype(np.uint8)
                    # Outline too: the faint tint alone is easy to miss under the quads
                    contours, _ = cv2.findContours(np.uint8(m), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, COLOR_MASK, 1, cv2.LINE_AA)

        if i in branch_corners["samh_H"]:
            draw_quad(img, branch_corners["samh_H"][i], COLOR_SAMH, 1)
        if i in branch_corners["loma_H"]:
            draw_quad(img, branch_corners["loma_H"][i], COLOR_LOMA, 1)
        if compare is not None and i < len(compare):
            draw_quad(img, compare[i], COLOR_COMPARE, 2)
        if i < preds.shape[0]:
            draw_output(img, preds[i])
        has_gt = gt_corners is not None and i < len(gt_corners) and gt_valid[i]
        gt_i = gt_corners[i] if has_gt else None
        if gt_i is not None:
            draw_dashed_quad(img, gt_i, COLOR_GT, 2)  # last: GT is never covered

        put_text(img, f"#{i}/{last}  " + frame_info.get(i, "init" if i == 0 else ""),
                 (10, 28), (255, 255, 255), 0.8)
        legend = list(layers)
        if compare is not None:
            legend.append((with_error(compare_dir.parent.name, gt_i, compare, i), COLOR_COMPARE))
        legend.append((with_error(OUTPUT_LABEL, gt_i, preds, i), COLOR_OUTPUT))
        draw_legend(img, legend, y=56)

        # Write frame
        writer.write(img)

    writer.release()
    print(f"Wrote video: {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='flatsam2-visualize')
    parser.add_argument('dataset', choices=CLI_CHOICES, help='Dataset to visualize')
    parser.add_argument('sequence', help='Sequence name to visualize')
    parser.add_argument('--config', type=Path, help='Path to config file (for directory structure)')
    parser.add_argument('--base-dir', type=Path, help='Dataset base directory (optional, uses default if not specified)')
    parser.add_argument('--src', type=Path, help='Top-level directory containing tracking results (defaults to output dir computed from dataset/config)',
                        default=paths.OUTPUT_DIRECTORY)
    parser.add_argument('--dst', type=Path, default=paths.VIS_DIRECTORY, help='Top-level output directory for videos')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--draw-masks', action=argparse.BooleanOptionalAction, default=True,
                        help="Overlay the run's saved SAM3 masks (--no-draw-masks to hide)")
    parser.add_argument('--compare', type=Path,
                        help="Dir with another tracker's <seq>.txt to overlay, "
                             "e.g. <results>/<other_config>/POT-210")

    args = parser.parse_args(argv)

    visualize_sequence(
        args.dataset,
        args.src,
        args.sequence,
        args.base_dir,
        args.dst,
        args.config,
        fps=args.fps,
        draw_masks=args.draw_masks,
        compare_dir=args.compare,
    )

if __name__ == '__main__':
    main()
