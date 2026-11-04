#!/usr/bin/env python3
"""Render the vector FlatSAM2 method overview used by the WACV paper.

Usage:
    python experiments/paper_overview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath

from flatsam2.path_config import paths


BACKGROUND = "#FFFFFF"
INK = "#20242C"
MUTED = "#5E626A"
LANE_FILL = "#F6F6F3"
PROCESS_FILL = "#ECECE8"
PROCESS_BORDER = "#5E626A"
IDENTITY_FILL = "#E8E4DE"
IDENTITY_BORDER = "#75695C"
SELECTOR_FILL = "#30343B"


def draw_box(
    ax: Axes,
    box: tuple[float, float, float, float],
    text: str,
    fill: str = PROCESS_FILL,
    border: str = PROCESS_BORDER,
    text_color: str = INK,
    font_size: float = 9.5,
) -> None:
    """Draw one rounded operation box.

    Args:
        ax: Matplotlib axes.
        box: Left, bottom, width, and height in diagram coordinates.
        text: Centered box label; mathtext is supported.
        fill: Box fill color.
        border: Border color.
        text_color: Label color.
        font_size: Label size in points.
    """
    left, bottom, width, height = box
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=fill,
        edgecolor=border,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(
        left + width / 2,
        bottom + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=font_size,
        linespacing=1.25,
    )


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = PROCESS_BORDER,
) -> None:
    """Draw one straight directed edge.

    Args:
        ax: Matplotlib axes.
        start: Edge start.
        end: Arrow tip.
        color: Edge color.
    """
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_poly_arrow(
    ax: Axes,
    points: list[tuple[float, float]],
    color: str = PROCESS_BORDER,
) -> None:
    """Draw a right-angle directed edge.

    Args:
        ax: Matplotlib axes.
        points: Ordered edge vertices.
        color: Edge color.
    """
    codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1)
    path = PlotPath(points, codes)
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    """Render method flow to a vector PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.OUTPUT_DIRECTORY / "experiments" / "paper_overview" / "overview.pdf",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    figure, ax = plt.subplots(figsize=(14.0, 4.1), facecolor=BACKGROUND)
    ax.set_xlim(0, 24.6)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    input_initial = (0.2, 4.9, 2.6, 1.35)
    input_current = (0.2, 1.3, 2.6, 1.35)
    sam = (3.4, 3.1, 2.65, 1.8)
    loma = (7.1, 5.35, 2.4, 1.3)
    matches = (10.15, 5.35, 2.4, 1.3)
    magsac = (13.2, 5.35, 2.3, 1.3)
    keypoint_pose = (16.1, 5.35, 2.75, 1.3)
    hough = (7.1, 1.35, 2.4, 1.3)
    quad = (10.15, 1.35, 2.4, 1.3)
    naming = (13.2, 1.35, 2.3, 1.3)
    mask_pose = (16.1, 1.35, 2.75, 1.3)
    selector = (20.25, 2.45, 2.9, 3.05)
    output = (23.65, 3.35, 0.75, 1.15)

    ax.add_patch(
        FancyBboxPatch(
            (6.65, 4.8),
            13.25,
            2.45,
            boxstyle="round,pad=0.03,rounding_size=0.25",
            facecolor=LANE_FILL,
            edgecolor="none",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (6.65, 0.75),
            13.25,
            2.45,
            boxstyle="round,pad=0.03,rounding_size=0.25",
            facecolor=LANE_FILL,
            edgecolor="none",
        )
    )
    ax.text(7.0, 7.0, "KEYPOINT BRANCH", fontsize=9.5, weight="bold", color=MUTED)
    ax.text(7.0, 2.95, "MASK BRANCH", fontsize=9.5, weight="bold", color=MUTED)

    draw_box(ax, input_initial, "Initial frame $I_0$\ninitial corners $X_0$", font_size=9.0)
    draw_box(ax, input_current, "Current frame $I_t$", font_size=9.0)
    draw_box(ax, sam, "SAM 3\nshared mask\nstream")
    draw_box(ax, loma, "Masked LoMa\nfeatures")
    draw_box(ax, matches, "Raw matches\n$\\mathcal{C}_t$")
    draw_box(ax, magsac, "MAGSAC\n5 px")
    draw_box(ax, keypoint_pose, "Keypoint pose\n$\\mathbf{H}^{K}_t,\\ n_t$")
    draw_box(ax, hough, "Hough lines")
    draw_box(ax, quad, "Quadrilateral\nfour cyclic\norderings")
    draw_box(
        ax,
        naming,
        "Match-support\ncorner naming",
        fill=IDENTITY_FILL,
        border=IDENTITY_BORDER,
    )
    draw_box(ax, mask_pose, "Mask pose\n$\\mathbf{H}^{M}_t$")
    draw_box(
        ax,
        selector,
        "SELECT\nCOMPLETE POSE\n\n$\\mathbf{H}^{K}_t$ if $n_t \\geq 35$\n"
        "$\\mathbf{H}^{M}_t$ otherwise",
        fill=SELECTOR_FILL,
        border=SELECTOR_FILL,
        text_color="white",
        font_size=9.2,
    )
    draw_box(
        ax,
        output,
        "$\\mathbf{H}_t$",
        fill=SELECTOR_FILL,
        border=SELECTOR_FILL,
        text_color="white",
    )

    draw_arrow(ax, (2.8, 5.55), (3.4, 4.3))
    draw_arrow(ax, (2.8, 1.95), (3.4, 3.7))
    draw_poly_arrow(ax, [(6.05, 4.25), (6.45, 4.25), (6.45, 6.0), (7.1, 6.0)])
    draw_poly_arrow(ax, [(6.05, 3.75), (6.45, 3.75), (6.45, 2.0), (7.1, 2.0)])
    draw_arrow(ax, (9.5, 6.0), (10.15, 6.0))
    draw_arrow(ax, (12.55, 6.0), (13.2, 6.0))
    draw_arrow(ax, (15.5, 6.0), (16.1, 6.0))
    draw_arrow(ax, (9.5, 2.0), (10.15, 2.0))
    draw_arrow(ax, (12.55, 2.0), (13.2, 2.0))
    draw_arrow(ax, (15.5, 2.0), (16.1, 2.0))
    draw_poly_arrow(
        ax,
        [(11.35, 5.35), (11.35, 4.0), (14.35, 4.0), (14.35, 2.65)],
        IDENTITY_BORDER,
    )
    ax.text(12.1, 4.12, "corner identity", fontsize=9.5, color=IDENTITY_BORDER)
    draw_poly_arrow(ax, [(18.85, 6.0), (19.65, 6.0), (19.65, 4.65), (20.25, 4.65)])
    draw_poly_arrow(ax, [(18.85, 2.0), (19.65, 2.0), (19.65, 3.3), (20.25, 3.3)])
    draw_arrow(ax, (23.15, 4.0), (23.65, 4.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
