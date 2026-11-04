#!/usr/bin/env python3
"""Visualize synthetic training data for NN homography quality estimation.

Generates samples using the exact training pipeline and renders them in a
clean 2x3 layout:

  Row 0: Reference (A)  |  GT warp (H_GT)  |  Candidate warp (H_aug)
  Row 1: Augmented vs candidate overlap  |  SDF_B + dA' pts  |  SDF_A' + dB pts

Usage:
    python experiments/visualize_nn_training_data.py --n 8 --out grid.png
    python experiments/visualize_nn_training_data.py --n 1 --detailed --out detail.png
    python experiments/visualize_nn_training_data.py --n 4 --show --regime good
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.homography_quality import _warp_points
from flatsam2.nn.data_gen import (
    CANVAS_H,
    CANVAS_W,
    generate_sample,
)

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------

_BG = np.array([30, 30, 35], dtype=np.uint8)          # near-black
_MASK_A = np.array([180, 150, 100], dtype=np.uint8)    # steel blue
_MASK_GT = np.array([130, 180, 100], dtype=np.uint8)   # muted green
_MASK_AUG = np.array([90, 160, 200], dtype=np.uint8)   # warm amber
_CONTOUR_A = np.array([220, 200, 160], dtype=np.uint8) # light blue
_CONTOUR_GT = np.array([160, 220, 140], dtype=np.uint8)  # light green
_CONTOUR_AUG = np.array([120, 190, 240], dtype=np.uint8) # light orange
_TEXT = (210, 210, 210)
_TEXT_DIM = (140, 140, 140)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# SDF colormap
# ---------------------------------------------------------------------------

def _sdf_colormap(sdf: np.ndarray, vmax: float = 40.0) -> np.ndarray:
    """Diverging blue-white-red colormap for signed distance fields.

    Negative (inside) -> blue, zero (boundary) -> white, positive (outside) -> red.
    """
    t = np.clip(sdf / vmax, -1.0, 1.0)
    img = np.zeros((*sdf.shape, 3), dtype=np.uint8)

    # Inside (t < 0): interpolate blue (#3b6cb0) -> white
    neg = t < 0
    a = -t[neg]  # 0..1, 0=boundary 1=deep inside
    img[neg, 0] = (255 - a * (255 - 176)).astype(np.uint8)  # B: 255->176
    img[neg, 1] = (255 - a * (255 - 108)).astype(np.uint8)  # G: 255->108
    img[neg, 2] = (255 - a * (255 - 59)).astype(np.uint8)   # R: 255->59

    # Outside (t >= 0): interpolate white -> red (#c0392b)
    pos = t >= 0
    a = t[pos]  # 0..1
    img[pos, 0] = (255 - a * (255 - 43)).astype(np.uint8)   # B: 255->43
    img[pos, 1] = (255 - a * (255 - 57)).astype(np.uint8)   # G: 255->57
    img[pos, 2] = (255 - a * (255 - 192)).astype(np.uint8)  # R: 255->192

    return img


def _sdf_point_color(sdf_val: float, vmax: float = 30.0):
    """Color a point by its SDF value: green (aligned) -> yellow -> red (far)."""
    t = min(abs(sdf_val) / vmax, 1.0)
    # Green (#27ae60) -> Yellow (#f1c40f) -> Red (#e74c3c)
    if t < 0.5:
        s = t * 2  # 0..1
        r = int(39 + s * (241 - 39))
        g = int(174 + s * (196 - 174))
        b = int(96 + s * (15 - 96))
    else:
        s = (t - 0.5) * 2  # 0..1
        r = int(241 + s * (231 - 241))
        g = int(196 - s * (196 - 76))
        b = int(15 - s * (15 - 60))
    return (b, g, r)


# ---------------------------------------------------------------------------
# Mask rendering helpers
# ---------------------------------------------------------------------------

def _fill_mask(shape, mask, fill_color, bg=None):
    """Render a binary mask as a filled shape on a dark background."""
    if bg is None:
        canvas = np.tile(_BG, (*shape[:2], 1))
    else:
        canvas = bg.copy()
    canvas[mask > 0] = fill_color
    return canvas


def _draw_contour(canvas, mask, color, thickness=1):
    """Draw mask contour on canvas."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, color.tolist(), thickness,
                     cv2.LINE_AA)


def _label(canvas, text, pos=(12, 28), scale=0.7, color=_TEXT, thickness=1):
    """Put text label on canvas."""
    cv2.putText(canvas, text, pos, _FONT, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _draw_nn_input_panel(prep: dict, cell_h: int, cell_w: int) -> np.ndarray:
    """Render the final NN input: both point sets in normalized bbox coordinates.

    - A' points: colored by SDF_B value; out-of-view shown as gray crosses
    - B points: colored by SDF_A' value; drawn as filled circles
    - Tangent arrows drawn for each point
    - Bounding box shown as a rectangle (the unit square in normalized space)
    """
    margin = max(30, min(cell_h, cell_w) // 10)
    canvas = np.tile(_BG, (cell_h, cell_w, 1))

    plot_x0, plot_y0 = margin, margin + 20   # leave space for label
    plot_x1, plot_y1 = cell_w - margin, cell_h - margin

    def _n2px(norm_xy: np.ndarray) -> np.ndarray:
        """Normalized [0,1] -> pixel coords (N, 2)."""
        px = norm_xy[:, 0] * (plot_x1 - plot_x0) + plot_x0
        py = norm_xy[:, 1] * (plot_y1 - plot_y0) + plot_y0
        return np.stack([px, py], axis=1)

    # Draw bounding box as unit rectangle
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1),
                  (90, 90, 90), 1, cv2.LINE_AA)
    # Tick marks at 0.25 / 0.5 / 0.75 along each axis
    for t in (0.25, 0.5, 0.75):
        tx = int(t * (plot_x1 - plot_x0) + plot_x0)
        ty = int(t * (plot_y1 - plot_y0) + plot_y0)
        cv2.line(canvas, (tx, plot_y0 - 3), (tx, plot_y0 + 3), (70, 70, 70), 1)
        cv2.line(canvas, (plot_x0 - 3, ty), (plot_x0 + 3, ty), (70, 70, 70), 1)

    norm_xy_A = prep["norm_xy_A"]      # (N, 2)
    norm_xy_B = prep["norm_xy_B"]      # (N, 2)
    is_in_view = prep["is_in_view"]    # (N,) float32
    sdf_B_at_Ap = prep["sdf_B_at_Ap"] # (N,)
    sdf_Ap_at_B = prep["sdf_Ap_at_B"] # (N,)
    tan_sin_A = prep["tan_sin_A"]      # (N,)
    tan_cos_A = prep["tan_cos_A"]      # (N,)
    tan_sin_B = prep["tan_sin_B"]      # (N,)
    tan_cos_B = prep["tan_cos_B"]      # (N,)

    is_on_boundary_B = prep["is_on_boundary_B"]  # (N,) float32

    px_A = _n2px(norm_xy_A)  # (N, 2)
    px_B = _n2px(norm_xy_B)  # (N, 2)

    arrow_len = max(6, min(cell_h, cell_w) // 50)

    # Draw B points first (so A' is on top)
    for i in range(len(px_B)):
        x, y = int(round(px_B[i, 0])), int(round(px_B[i, 1]))
        on_boundary = bool(is_on_boundary_B[i])
        color = (70, 70, 70) if on_boundary else _sdf_point_color(sdf_Ap_at_B[i])
        ex = int(round(x + tan_cos_B[i] * arrow_len))
        ey = int(round(y + tan_sin_B[i] * arrow_len))
        cv2.line(canvas, (x, y), (ex, ey), color, 1, cv2.LINE_AA)
        if on_boundary:
            cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 3, (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)

    # Draw A' points: filled circles if in-view, cross marker if out-of-view
    for i in range(len(px_A)):
        x, y = int(round(px_A[i, 0])), int(round(px_A[i, 1]))
        # Clamp to canvas so markers don't disappear
        xc = int(np.clip(x, 0, cell_w - 1))
        yc = int(np.clip(y, plot_y0, cell_h - 1))
        in_view = is_in_view[i] > 0.5
        color = _sdf_point_color(sdf_B_at_Ap[i]) if in_view else (70, 70, 70)
        ex = int(round(xc + tan_cos_A[i] * arrow_len))
        ey = int(round(yc + tan_sin_A[i] * arrow_len))
        cv2.line(canvas, (xc, yc), (ex, ey), color, 1, cv2.LINE_AA)
        if in_view:
            # Diamond shape: draw as rotated square
            pts = np.array([[xc, yc - 3], [xc + 3, yc],
                            [xc, yc + 3], [xc - 3, yc]], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], color)
        else:
            cv2.drawMarker(canvas, (xc, yc), color, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)

    # Legend
    _label(canvas, "NN input (normalized bbox)", pos=(12, 18), scale=0.5)
    cv2.circle(canvas, (12, cell_h - 18), 4, (128, 128, 128), -1, cv2.LINE_AA)
    _label(canvas, "dB", pos=(20, cell_h - 14), scale=0.4, color=_TEXT_DIM)
    diam_pts = np.array([[cell_w - 55, cell_h - 21], [cell_w - 51, cell_h - 17],
                         [cell_w - 55, cell_h - 13], [cell_w - 59, cell_h - 17]], dtype=np.int32)
    cv2.fillPoly(canvas, [diam_pts], (180, 180, 180))
    _label(canvas, "dA'", pos=(cell_w - 47, cell_h - 14), scale=0.4, color=_TEXT_DIM)
    cv2.drawMarker(canvas, (cell_w - 95, cell_h - 17), (70, 70, 70),
                   cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
    _label(canvas, "OOV", pos=(cell_w - 87, cell_h - 14), scale=0.4, color=_TEXT_DIM)

    return canvas


def draw_sample(data, cell_h=360, cell_w=480):
    """Render one sample as a 2x3 panel.

    Row 0: Reference (A) | GT warp (H_GT) | Candidate warp (H_aug)
    Row 1: Augmented vs candidate | SDF_B + dA' | NN input (normalized bbox)
    """
    h, w = CANVAS_H, CANVAS_W
    panel_h = cell_h * 2
    panel_w = cell_w * 3
    panel = np.tile(_BG, (panel_h, panel_w, 1))

    def _fit(img, interpolation=cv2.INTER_AREA):
        """Isotropically resize img to fit cell, center on dark bg."""
        src_h, src_w = img.shape[:2]
        s = min(cell_w / src_w, cell_h / src_h)
        nw, nh = max(1, round(src_w * s)), max(1, round(src_h * s))
        resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
        cell = np.tile(_BG, (cell_h, cell_w, 1))
        y0, x0 = (cell_h - nh) // 2, (cell_w - nw) // 2
        cell[y0:y0 + nh, x0:x0 + nw] = resized
        return cell

    def _put(row, col, img, interpolation=cv2.INTER_AREA):
        r0, c0 = row * cell_h, col * cell_w
        panel[r0:r0 + cell_h, c0:c0 + cell_w] = _fit(img, interpolation)

    prep = data["prep"]

    # ── Row 0 ──────────────────────────────────────────────────────────

    # (0,0) Reference mask A
    vis = _fill_mask((h, w, 3), data["mask_A"], _MASK_A)
    _draw_contour(vis, data["mask_A"], _CONTOUR_A)
    _label(vis, "Reference (A)")
    _put(0, 0, vis)

    # (0,1) GT warp: mask_B_clean (= mask_A warped by H_GT)
    vis = _fill_mask((h, w, 3), data["mask_B_clean"], _MASK_GT)
    _draw_contour(vis, data["mask_B_clean"], _CONTOUR_GT)
    _label(vis, "GT warp (H_GT)")
    _put(0, 1, vis)

    # (0,2) Candidate warp: mask_Ap (= mask_A warped by H_aug)
    vis = _fill_mask((h, w, 3), data["mask_Ap"], _MASK_AUG)
    _draw_contour(vis, data["mask_Ap"], _CONTOUR_AUG)
    _label(vis, "Candidate warp (H_aug)")
    _put(0, 2, vis)

    # ── Row 1 ──────────────────────────────────────────────────────────

    # (1,0) Augmented GT-warp (mask_B) vs candidate warp (mask_Ap)
    mask_B = data["mask_B"]
    mask_Ap = data["mask_Ap"]
    vis = np.tile(_BG, (h, w, 1))

    # Show three regions with distinct colors
    only_B = (mask_B > 0) & ~(mask_Ap > 0)
    only_Ap = ~(mask_B > 0) & (mask_Ap > 0)
    both = (mask_B > 0) & (mask_Ap > 0)
    vis[only_B] = _MASK_GT          # green: GT-warped + augmented only
    vis[only_Ap] = _MASK_AUG        # amber: candidate only
    vis[both] = [140, 170, 140]     # muted overlap

    # GT contour (thin) and augmented contour
    _draw_contour(vis, data["mask_B_clean"], _CONTOUR_GT, thickness=1)
    _draw_contour(vis, mask_Ap, _CONTOUR_AUG, thickness=1)

    # Warped contour points for RMSE visualization
    contour_A = data["contour_A"]
    if contour_A is not None:
        pts_gt = _warp_points(contour_A, data["H_GT"])
        pts_aug = _warp_points(contour_A, data["H_candidate"])
        for i in range(0, len(pts_gt), 2):
            gx, gy = int(pts_gt[i, 0]), int(pts_gt[i, 1])
            ax, ay = int(pts_aug[i, 0]), int(pts_aug[i, 1])
            if 0 <= gx < w and 0 <= gy < h:
                cv2.circle(vis, (gx, gy), 2, _CONTOUR_GT.tolist(), -1,
                           cv2.LINE_AA)
            if 0 <= ax < w and 0 <= ay < h:
                cv2.circle(vis, (ax, ay), 2, _CONTOUR_AUG.tolist(), -1,
                           cv2.LINE_AA)

    regime = "censored" if data["censored"] else "uncensored"
    _label(vis, f"mask_B (aug) vs mask_A' | RMSE={data['gt_rmse']:.1f}px "
           f"({regime})")
    _put(1, 0, vis)

    # (1,1) SDF_B + dA' sampled points
    sdf_vis = _sdf_colormap(prep["sdf_B"], vmax=40)
    pts_Ap = prep["pts_Ap"]
    in_view = prep["is_in_view"]
    sdf_at_Ap = prep["sdf_B_at_Ap"]
    for i in range(len(pts_Ap)):
        x, y = int(pts_Ap[i, 0]), int(pts_Ap[i, 1])
        if 0 <= x < w and 0 <= y < h:
            if in_view[i] > 0.5:
                color = _sdf_point_color(sdf_at_Ap[i])
            else:
                color = (80, 80, 80)
            cv2.circle(sdf_vis, (x, y), 3, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(sdf_vis, (x, y), 2, color, -1, cv2.LINE_AA)
    _label(sdf_vis, "SDF(B) + dA' points")
    _put(1, 1, sdf_vis)

    # (1,2) NN input: both point sets in normalized bbox space
    _put(1, 2, _draw_nn_input_panel(prep, cell_h, cell_w), cv2.INTER_NEAREST)

    return panel


# ---------------------------------------------------------------------------
# Sample generation wrapper
# ---------------------------------------------------------------------------

def generate_and_visualize_sample(
    rng: np.random.Generator,
    canvas_shape=(CANVAS_H, CANVAS_W),
    n_points=128,
    target_regime=None,
    max_attempts=100,
):
    """Generate one valid sample with intermediates for visualization.

    Calls :func:`generate_sample` with ``keep_intermediates=True`` and
    optionally filters by error regime.

    Returns dict with all intermediate data, or None after max_attempts.
    """
    for _ in range(max_attempts):
        data = generate_sample(
            rng, n_points, canvas_shape, keep_intermediates=True)
        if data is None:
            continue

        if target_regime is not None:
            rmse = data["gt_rmse"]
            if target_regime == "perfect" and rmse > 0.5:
                continue
            elif target_regime == "good" and not (0.5 < rmse <= 15):
                continue
            elif target_regime == "borderline" and not (15 < rmse <= 30):
                continue
            elif target_regime == "bad" and rmse <= 30:
                continue

        return data

    return None


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def make_grid(samples, n_cols=2, cell_h=360, cell_w=480, gap=3):
    """Arrange sample panels into a grid image."""
    n = len(samples)
    n_rows = (n + n_cols - 1) // n_cols
    panel_h = cell_h * 2
    panel_w = cell_w * 3

    grid_h = n_rows * (panel_h + gap) + gap
    grid_w = n_cols * (panel_w + gap) + gap
    grid = np.tile(_BG, (grid_h, grid_w, 1))

    for idx, data in enumerate(samples):
        row, col = idx // n_cols, idx % n_cols
        p = draw_sample(data, cell_h=cell_h, cell_w=cell_w)
        y0 = gap + row * (panel_h + gap)
        x0 = gap + col * (panel_w + gap)
        grid[y0:y0 + panel_h, x0:x0 + panel_w] = p

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize synthetic NN training data")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--show", action="store_true",
                        help="Display result in a window")
    parser.add_argument("--out", type=str, default=None,
                        help="Save result to file")
    parser.add_argument("--regime", type=str, default=None,
                        choices=["perfect", "good", "borderline", "bad"],
                        help="Filter to a specific error regime")
    parser.add_argument("--detailed", action="store_true",
                        help="Use larger cell size for detailed inspection")
    parser.add_argument("--cols", type=int, default=2,
                        help="Number of sample columns in the grid")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.n} samples (seed={args.seed})...")

    cell_h = 540 if args.detailed else 360
    cell_w = 720 if args.detailed else 480

    samples = []
    attempts = 0
    while len(samples) < args.n and attempts < args.n * 20:
        attempts += 1
        data = generate_and_visualize_sample(
            rng, target_regime=args.regime)
        if data is not None:
            samples.append(data)
            regime = "censored" if data["censored"] else "uncensored"
            print(f"  [{len(samples)}/{args.n}] "
                  f"RMSE={data['gt_rmse']:.2f}px ({regime})")

    if not samples:
        print("No valid samples generated!")
        return

    print(f"Generated {len(samples)} samples in {attempts} attempts")

    if args.n == 1:
        result = draw_sample(samples[0], cell_h=cell_h, cell_w=cell_w)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.out, result)
            print(f"Saved: {args.out}")
    else:
        # result = make_grid(samples, n_cols=args.cols,
        #                    cell_h=cell_h, cell_w=cell_w)
        for sample_i, sample in enumerate(samples):
            result = draw_sample(sample, cell_h=cell_h, cell_w=cell_w)
            if args.out:
                out = Path(args.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out_path = out.parent / f'{out.stem}_{sample_i:03d}{out.suffix}'
                cv2.imwrite(out_path, result)
                print(f"Saved: {out_path}")

    if args.show:
        cv2.imshow("NN Training Data", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.show and not args.out:
        default_out = "nn_training_data_vis.png"
        cv2.imwrite(default_out, result)
        print(f"Saved: {default_out}")


if __name__ == "__main__":
    main()
