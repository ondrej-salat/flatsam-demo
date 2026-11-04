#!/usr/bin/env python3
"""Spatial coverage metrics for evaluating keypoint inlier quality.

Measures how well RANSAC inlier keypoints cover the object in the init frame,
complementing scalar inlier counts with geometric quality signals.

Standalone — no flatsam2/loma dependencies.

Metrics
-------
hull_coverage(inlier_pts, ref_pts)
    Convex hull area of inlier_pts / convex hull area of ref_pts.

peeled_hull_coverage(inlier_pts, ref_pts)
    Same but after one layer of convex hull peeling: remove the hull-vertex
    (boundary) inliers first, then measure coverage with the inner hull.
    More conservative — not inflated by a few extreme outlier inliers.

Usage::

    python spatial_coverage.py --benchmark
    python spatial_coverage.py --demo
    python spatial_coverage.py --visualize --npz SEQ.npz --masks-dir DIR --seq NAME --frame 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def hull_area(pts: np.ndarray) -> float:
    """Convex hull area of (N, 2) point array.

    Args:
        pts: Point array, shape ``(N, 2)``.

    Returns:
        Area in squared pixel units, or ``0.0`` if fewer than 3 non-collinear
        points.
    """
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        return float(hull.volume)  # 'volume' is area in 2-D
    except Exception:
        return 0.0


def _peel_hull_vertices(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove convex hull vertices, return (interior_pts, vertex_pts).

    Args:
        pts: ``(N, 2)`` point array.

    Returns:
        ``(interior_pts, vertex_pts)`` where ``interior_pts`` are the non-hull
        points and ``vertex_pts`` are the hull boundary points.  If the hull
        fails (e.g. all points collinear), returns ``(pts, empty)``.
    """
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        vertex_idx = set(hull.vertices.tolist())
        mask = np.array([i not in vertex_idx for i in range(len(pts))])
        return pts[mask], pts[~mask]
    except Exception:
        return pts, np.empty((0, 2), dtype=pts.dtype)


def hull_coverage(inlier_pts: np.ndarray, ref_pts: np.ndarray) -> float:
    """Fraction of reference region spanned by the inlier convex hull.

    Args:
        inlier_pts: RANSAC inlier keypoints in init-frame coords ``(N, 2)``.
        ref_pts: Reference points defining the target region ``(M, 2)``.
            Typically foreground pixels of the init-frame mask.

    Returns:
        ``hull_area(inlier_pts) / hull_area(ref_pts)``, clamped to ``[0, 1]``.
        Returns ``0.0`` if reference hull is degenerate.
    """
    ref_area = hull_area(ref_pts)
    if ref_area <= 0.0:
        return 0.0
    return float(np.clip(hull_area(inlier_pts) / ref_area, 0.0, 1.0))


def peeled_hull_coverage(inlier_pts: np.ndarray, ref_pts: np.ndarray) -> float:
    """Coverage of reference region by the peeled (interior) inlier hull.

    Removes the outermost convex hull layer (hull vertices) from the inlier
    set, then measures coverage using the second hull formed by the remaining
    interior points.  More conservative than :func:`hull_coverage` — not
    inflated by extreme inliers far from the main cluster.

    Args:
        inlier_pts: RANSAC inlier keypoints in init-frame coords ``(N, 2)``.
        ref_pts: Reference points defining the target region ``(M, 2)``.

    Returns:
        ``hull_coverage(interior_pts, ref_pts)``, clamped to ``[0, 1]``.
        Returns ``0.0`` if fewer than 4 inliers or interior hull is degenerate.
    """
    if len(inlier_pts) < 4:
        return 0.0
    interior_pts, _ = _peel_hull_vertices(inlier_pts)
    if len(interior_pts) < 3:
        return 0.0
    return hull_coverage(interior_pts, ref_pts)


# ---------------------------------------------------------------------------
# Mask sampling
# ---------------------------------------------------------------------------


def sample_mask_pts(mask: np.ndarray, stride: int = 8) -> np.ndarray:
    """Foreground pixel coordinates from a uint8 mask, sampled at stride.

    Args:
        mask: Grayscale uint8 mask ``(H, W)``, non-zero = foreground.
        stride: Sampling stride in pixels.

    Returns:
        ``(N, 2)`` float64 array of ``(x, y)`` pixel coordinates.
    """
    ys, xs = np.where(mask[::stride, ::stride] > 0)
    xs = xs * stride
    ys = ys * stride
    return np.stack([xs, ys], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _time_fn(fn, n_reps: int) -> np.ndarray:
    """Return per-call timings in ms for *fn* called *n_reps* times."""
    times = np.empty(n_reps, dtype=np.float64)
    for k in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times[k] = (time.perf_counter() - t0) * 1000.0
    return times


def _stats_line(times: np.ndarray) -> str:
    p = np.percentile(times, [10, 50, 90])
    return (
        f"mean={times.mean():.3f}  "
        f"p10={p[0]:.3f}  p50={p[1]:.3f}  p90={p[2]:.3f}  "
        f"min={times.min():.3f}  max={times.max():.3f}"
    )


def _benchmark(n_reps: int = 500) -> None:
    """Time hull_coverage + peeled_hull_coverage across realistic point counts.

    Reports per-call ms: mean, p10, p50, p90, min, max.
    """
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    rng = np.random.default_rng(42)
    fns = [
        ("hull_coverage",        lambda pts, ref: hull_coverage(pts, ref)),
        ("peeled_hull_coverage", lambda pts, ref: peeled_hull_coverage(pts, ref)),
    ]
    label_w = max(len(name) for name, _ in fns)

    for n in sizes:
        pts = rng.uniform(0, 640, size=(n, 2))
        ref = rng.uniform(0, 640, size=(n * 4, 2))
        print(f"N={n}")
        for name, fn in fns:
            times = _time_fn(lambda fn=fn: fn(pts, ref), n_reps)
            print(f"  {name:<{label_w}}  {_stats_line(times)}")
        print()


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


def _hull_polygon(pts: np.ndarray) -> Optional[np.ndarray]:
    """Return convex hull vertices in cyclic order, or None if degenerate."""
    if len(pts) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        return np.vstack([verts, verts[:1]])  # close the polygon
    except Exception:
        return None


def visualize_coverage(
    inlier_pts: np.ndarray,
    ref_pts: Optional[np.ndarray] = None,
    ax=None,
    title: str = "",
) -> None:
    """Scatter inlier points with hull and peeled-hull overlays.

    Args:
        inlier_pts: Inlier keypoints ``(N, 2)``.
        ref_pts: Reference points for coverage denominator ``(M, 2)`` (optional).
        ax: Matplotlib axes to draw on. Creates a new figure if ``None``.
        title: Plot title prefix.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    if ref_pts is not None and len(ref_pts) > 0:
        ref_poly = _hull_polygon(ref_pts)
        if ref_poly is not None:
            ax.plot(ref_poly[:, 0], ref_poly[:, 1], "b--", lw=1.5, label="mask hull")

    if len(inlier_pts) > 0:
        ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], s=8, c="#ef4444", alpha=0.6, label="inliers")

        poly = _hull_polygon(inlier_pts)
        if poly is not None:
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.10, color="#ef4444")
            ax.plot(poly[:, 0], poly[:, 1], "#ef4444", lw=2, label="hull")

        interior_pts, vertex_pts = _peel_hull_vertices(inlier_pts)
        if len(vertex_pts) > 0:
            ax.scatter(vertex_pts[:, 0], vertex_pts[:, 1], s=30, c="#b91c1c",
                       marker="^", zorder=5, label="hull vertices (peeled)")
        if len(interior_pts) >= 3:
            inner_poly = _hull_polygon(interior_pts)
            if inner_poly is not None:
                ax.fill(inner_poly[:, 0], inner_poly[:, 1], alpha=0.18, color="#15803d")
                ax.plot(inner_poly[:, 0], inner_poly[:, 1], "#15803d", lw=1.5,
                        linestyle="--", label="peeled hull")

    cov = hull_coverage(inlier_pts, ref_pts) if ref_pts is not None else float("nan")
    pcov = peeled_hull_coverage(inlier_pts, ref_pts) if ref_pts is not None else float("nan")
    metrics = f"hull_coverage={cov:.3f}  peeled={pcov:.3f}"
    ax.set_title(f"{title}\n{metrics}" if title else metrics, fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Demo (synthetic sanity-check visualizations)
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Plot synthetic scenarios to sanity-check metrics visually."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    W, H = 640, 480

    def rect_pts(x0, y0, x1, y1, n=20):
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    ref = rect_pts(0, 0, W, H)

    scenarios = [
        (
            "Good: well-spread inliers\n(hull ≈ peeled ≈ high)",
            rect_pts(50, 50, W - 50, H - 50, n=15),
        ),
        (
            "Bad: bunched at center\n(hull ≈ peeled ≈ low)",
            rng.normal([W / 2, H / 2], [20, 20], size=(200, 2)),
        ),
        (
            "Bad: collinear across object\n(hull area = 0)",
            np.column_stack([np.linspace(0, W, 200), np.full(200, H / 2)]),
        ),
        (
            "Tricky: 4 outliers + bunched bulk\n(hull HIGH, peeled LOW)",
            np.vstack([
                rng.normal([W / 2, H / 2], [15, 15], size=(150, 2)),
                np.array([[10, 10], [W - 10, 10], [W - 10, H - 10], [10, H - 10]], dtype=np.float64),
            ]),
        ),
        (
            "OK: quarter coverage\n(hull ≈ peeled ≈ 0.25)",
            rect_pts(0, 0, W / 2, H / 2, n=15),
        ),
        (
            "Sparse but spread\n(hull high, peeled lower — few interior pts)",
            np.array([
                [0, 0], [W, 0], [W, H], [0, H],
                [W / 2, 0], [W, H / 2], [W / 2, H], [0, H / 2],
                [W / 4, H / 4], [3 * W / 4, H / 4],
            ], dtype=np.float64),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Spatial coverage sanity-check  (blue dashed = mask hull, red = inlier hull, green = peeled hull)",
        fontsize=12,
    )
    for ax, (title, inlier_pts) in zip(axes.ravel(), scenarios):
        visualize_coverage(inlier_pts, ref, ax=ax, title=title)
        ax.set_xlim(-20, W + 20)
        ax.set_ylim(H + 20, -20)

    plt.tight_layout()
    out_path = "spatial_coverage_demo.pdf"
    plt.savefig(out_path)
    print(f"Demo saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spatial coverage metrics util")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true", help="Time hull functions across point counts")
    mode.add_argument("--demo", action="store_true", help="Plot synthetic sanity-check scenarios")
    mode.add_argument("--visualize", action="store_true", help="Visualize coverage for one real frame")

    parser.add_argument("--npz", type=Path, help="Path to sequence NPZ file")
    parser.add_argument("--masks-dir", type=Path, help="Base masks directory")
    parser.add_argument("--seq", help="Sequence name (for mask loading)")
    parser.add_argument("--frame", type=int, default=1, help="Frame index to visualize (default: 1)")
    parser.add_argument("--benchmark-reps", type=int, default=500, help="Repetitions per size")

    args = parser.parse_args(argv)

    if args.benchmark:
        _benchmark(n_reps=args.benchmark_reps)
        return 0

    if args.demo:
        _demo()
        return 0

    # Visualize mode
    if not args.npz or not args.npz.exists():
        print("ERROR: --npz required and must exist")
        return 1

    data = np.load(args.npz)
    frame_key = f"{args.frame:05d}"
    pts_a_key = f"kptsA_{frame_key}"
    inlier_key = f"inlier_mask_{frame_key}"

    if pts_a_key not in data:
        print(f"ERROR: {pts_a_key} not in NPZ")
        return 1

    pts_a = data[pts_a_key].astype(np.float64)

    if inlier_key in data:
        inlier_mask = data[inlier_key].astype(bool)
        print(f"Using saved inlier_mask ({inlier_mask.sum()} / {len(inlier_mask)} inliers)")
    else:
        print("WARNING: no saved inlier_mask — showing all matched points")
        inlier_mask = np.ones(len(pts_a), dtype=bool)

    inlier_pts = pts_a[inlier_mask]

    ref_pts = None
    if args.masks_dir and args.seq:
        mask_path = args.masks_dir / args.seq / "00001.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            ref_pts = sample_mask_pts(mask, stride=8)
            print(f"Init mask: {mask.shape}, {len(ref_pts)} sampled ref pts")
        else:
            print(f"WARNING: mask not found at {mask_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    visualize_coverage(inlier_pts, ref_pts, ax=ax, title=f"{args.seq or ''} frame {args.frame}")
    plt.tight_layout()
    out_path = "spatial_coverage_frame.pdf"
    plt.savefig(out_path)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
