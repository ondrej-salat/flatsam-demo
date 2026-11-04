#!/usr/bin/env python3
"""Render fixed positive PlanarTrack examples for the WACV paper.

Examples come from the previous paper's qualitative set rather than a new search over test
results. This script reads completed FlatSAM2 predictions and dataset frames; it does not
run tracking.

Usage:
    python experiments/paper_examples.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

from paper_colors import FLATSAM_RGB, GT_RGB
from flatsam2.path_config import paths


EXAMPLES = (
    ("Seq_00329", 255),  # Large scale and perspective change.
    ("Seq_00341", 40),   # Reflective target.
    ("Seq_00434", 499),  # Textureless target.
    ("Seq_00926", 195),  # Planar face on a non-planar package.
)
IMAGE_SUFFIX = ".jpg"
CELL_SIZE = (480, 320)
GUTTER = 8
CORNER_FONT = Path("/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf")


def load_corners(path: Path) -> np.ndarray:
    """Load corner rows in benchmark x0,y0,... order.

    Args:
        path: Text file containing one or more eight-value corner rows.

    Returns:
        Corner array with shape (frames, 4, 2).

    Raises:
        ValueError: If file does not contain complete quadrilaterals.
    """
    flat = np.loadtxt(path)
    flat = np.atleast_2d(flat)
    if flat.shape[1] != 8:
        raise ValueError(f"expected 8 corner values per row in {path}, got {flat.shape}")
    return rearrange(flat, "frames (corners xy) -> frames corners xy", corners=4, xy=2)


def draw_quad(
    image: Image.Image,
    corners: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw one closed quadrilateral.

    Args:
        image: RGB image modified in place.
        corners: Quadrilateral with shape (4, 2).
        color: RGB line color.
        width: Line width in pixels.
    """
    points = [(float(x), float(y)) for x, y in corners]
    draw = ImageDraw.Draw(image)
    draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def draw_corner_numbers(
    image: Image.Image,
    corners: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 16,
) -> None:
    """Draw visible 1--4 corner identity markers.

    Args:
        image: RGB image modified in place.
        corners: Quadrilateral with shape (4, 2).
        color: RGB marker color.
        radius: Marker radius in output pixels.
    """
    marker_font = (
        ImageFont.truetype(str(CORNER_FONT), size=20)
        if CORNER_FONT.exists()
        else ImageFont.load_default(size=20)
    )
    draw = ImageDraw.Draw(image)
    for number, (x, y) in enumerate(corners, start=1):
        center = (float(x), float(y))
        circle = (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        )
        draw.ellipse(circle, fill=color, outline="white", width=3)
        text = str(number)
        bounds = draw.textbbox((0, 0), text, font=marker_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        draw.text(
            (center[0] - text_width / 2, center[1] - text_height / 2 - bounds[1]),
            text,
            fill="white",
            font=marker_font,
        )


def crop_around(
    image: Image.Image,
    quads: Sequence[np.ndarray],
    margin: float = 0.25,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Crop around quadrilaterals with proportional context.

    Args:
        image: Source RGB image.
        quads: Quadrilaterals with shape (4, 2).
        margin: Context added on each side as fraction of union size.

    Returns:
        Cropped image resized to ``CELL_SIZE`` and its x/y offset and scale.
    """
    points = np.concatenate(quads, axis=0)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_pad = max(24.0, (x_max - x_min) * margin)
    y_pad = max(24.0, (y_max - y_min) * margin)
    crop_width = x_max - x_min + 2 * x_pad
    crop_height = y_max - y_min + 2 * y_pad
    target_ratio = CELL_SIZE[0] / CELL_SIZE[1]
    if crop_width / crop_height < target_ratio:
        crop_width = crop_height * target_ratio
    else:
        crop_height = crop_width / target_ratio
    crop_width = min(crop_width, image.width)
    crop_height = min(crop_height, image.height)
    center_x = float(
        np.clip((x_min + x_max) / 2, crop_width / 2, image.width - crop_width / 2)
    )
    center_y = float(
        np.clip((y_min + y_max) / 2, crop_height / 2, image.height - crop_height / 2)
    )
    box = (
        int(np.floor(center_x - crop_width / 2)),
        int(np.floor(center_y - crop_height / 2)),
        int(np.ceil(center_x + crop_width / 2)),
        int(np.ceil(center_y + crop_height / 2)),
    )
    crop = image.crop(box)
    scale_x = CELL_SIZE[0] / crop.width
    scale_y = CELL_SIZE[1] / crop.height
    resized = crop.resize(CELL_SIZE, resample=Image.Resampling.LANCZOS)
    return resized, (float(box[0]), float(box[1]), scale_x, scale_y)


def transform_corners(
    corners: np.ndarray,
    transform: tuple[float, float, float, float],
) -> np.ndarray:
    """Map source-image corners into a rendered cell.

    Args:
        corners: Quadrilateral with shape (4, 2).
        transform: Crop x/y offset and x/y scale.

    Returns:
        Cell coordinates with shape (4, 2).
    """
    offset_x, offset_y, scale_x, scale_y = transform
    result = corners.copy()
    result[:, 0] = (result[:, 0] - offset_x) * scale_x
    result[:, 1] = (result[:, 1] - offset_y) * scale_y
    return result


def render_example(
    base_dir: Path,
    initfix_dir: Path,
    ours_dir: Path,
    sequence: str,
    frame_idx: int,
) -> tuple[Image.Image, Image.Image, float]:
    """Render initial and current-frame cells for one example.

    Args:
        base_dir: PlanarTrack dataset root.
        initfix_dir: Re-annotated initial-corner directory.
        ours_dir: Completed FlatSAM2 result directory.
        sequence: PlanarTrack sequence name.
        frame_idx: Zero-based frame index.

    Returns:
        Initial cell, current cell, and FlatSAM2 error.
    """
    sequence_dir = base_dir / "sequences" / sequence
    initial_image = Image.open(sequence_dir / f"{1:05d}{IMAGE_SUFFIX}").convert("RGB")
    current_image = Image.open(sequence_dir / f"{frame_idx + 1:05d}{IMAGE_SUFFIX}").convert("RGB")
    gt = load_corners(base_dir / "annos" / f"{sequence}.txt")
    initial_gt = load_corners(initfix_dir / f"{sequence}.txt")[0]
    ours = load_corners(ours_dir / f"{sequence}.txt")[frame_idx]

    draw_quad(initial_image, initial_gt, GT_RGB, width=8)
    draw_quad(current_image, ours, FLATSAM_RGB, width=10)
    draw_quad(current_image, gt[frame_idx], GT_RGB, width=4)

    initial_cell, initial_transform = crop_around(initial_image, [initial_gt])
    current_cell, current_transform = crop_around(current_image, [gt[frame_idx], ours])
    draw_corner_numbers(initial_cell, transform_corners(initial_gt, initial_transform), GT_RGB)
    draw_corner_numbers(current_cell, transform_corners(ours, current_transform), FLATSAM_RGB)
    ours_error = float(np.sqrt(np.sum((gt[frame_idx] - ours) ** 2) / 4.0))
    return initial_cell, current_cell, ours_error


def main() -> None:
    """Render fixed examples to one paper-ready PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=paths.PT_BASE)
    parser.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    parser.add_argument(
        "--ours-dir",
        type=Path,
        default=paths.OUTPUT_DIRECTORY / "loma_samh_matchnaming" / "PlanarTrackInitfix",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.OUTPUT_DIRECTORY / "experiments" / "paper_examples" / "pti_positive.png",
    )
    args = parser.parse_args()

    width = len(EXAMPLES) * CELL_SIZE[0] + (len(EXAMPLES) - 1) * GUTTER
    height = 2 * CELL_SIZE[1] + GUTTER
    figure = Image.new("RGB", (width, height), "white")
    for column, (sequence, frame_idx) in enumerate(EXAMPLES):
        initial, current, ours_error = render_example(
            args.base_dir,
            args.initfix_dir,
            args.ours_dir,
            sequence,
            frame_idx,
        )
        x = column * (CELL_SIZE[0] + GUTTER)
        figure.paste(initial, (x, 0))
        figure.paste(current, (x, CELL_SIZE[1] + GUTTER))
        print(f"{sequence} frame {frame_idx}: FlatSAM2={ours_error:.2f}px")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.out, optimize=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
