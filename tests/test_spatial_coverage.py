"""Unit tests for spatial_coverage.py metrics."""

import numpy as np
import pytest

from spatial_coverage import hull_area, hull_coverage, peeled_hull_coverage, sample_mask_pts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rect(x0, y0, x1, y1, n_per_side=10):
    xs = np.linspace(x0, x1, n_per_side)
    ys = np.linspace(y0, y1, n_per_side)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def _corners(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# hull_area
# ---------------------------------------------------------------------------


def test_hull_area_too_few_points():
    assert hull_area(np.zeros((0, 2))) == 0.0
    assert hull_area(np.zeros((1, 2))) == 0.0
    assert hull_area(np.zeros((2, 2))) == 0.0


def test_hull_area_collinear():
    pts = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    assert hull_area(pts) == pytest.approx(0.0, abs=1e-6)


def test_hull_area_unit_square():
    pts = _rect(0, 0, 1, 1, n_per_side=20)
    assert hull_area(pts) == pytest.approx(1.0, rel=0.01)


def test_hull_area_known_triangle():
    # Right triangle legs 3, 4 → area = 6
    pts = np.array([[0, 0], [3, 0], [0, 4], [1, 1]], dtype=np.float64)
    assert hull_area(pts) == pytest.approx(6.0, rel=0.01)


def test_hull_area_single_cluster():
    rng = np.random.default_rng(0)
    pts = rng.normal(loc=[5.0, 5.0], scale=0.001, size=(50, 2))
    assert hull_area(pts) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# hull_coverage
# ---------------------------------------------------------------------------


def test_hull_coverage_zero_inliers():
    ref = _rect(0, 0, 100, 100)
    assert hull_coverage(np.zeros((0, 2)), ref) == 0.0


def test_hull_coverage_degenerate_ref():
    ref = np.column_stack([np.linspace(0, 10, 20), np.zeros(20)])
    inliers = _rect(0, 0, 5, 5)
    assert hull_coverage(inliers, ref) == 0.0


def test_hull_coverage_full():
    ref = _rect(0, 0, 100, 100)
    inliers = _corners(0, 0, 100, 100)
    assert hull_coverage(inliers, ref) == pytest.approx(1.0, rel=0.01)


def test_hull_coverage_quarter():
    ref = _rect(0, 0, 100, 100)
    inliers = _rect(0, 0, 50, 50)
    assert hull_coverage(inliers, ref) == pytest.approx(0.25, rel=0.05)


def test_hull_coverage_bunched_center():
    ref = _rect(0, 0, 100, 100)
    rng = np.random.default_rng(1)
    inliers = rng.normal(loc=[50, 50], scale=2.0, size=(200, 2))
    assert hull_coverage(inliers, ref) < 0.05


def test_hull_coverage_clamped():
    ref = _rect(10, 10, 20, 20)
    inliers = _rect(0, 0, 100, 100)
    assert hull_coverage(inliers, ref) <= 1.0


# ---------------------------------------------------------------------------
# peeled_hull_coverage
# ---------------------------------------------------------------------------


def test_peeled_hull_coverage_too_few_inliers():
    ref = _rect(0, 0, 100, 100)
    assert peeled_hull_coverage(np.zeros((0, 2)), ref) == 0.0
    assert peeled_hull_coverage(np.zeros((3, 2)), ref) == 0.0


def test_peeled_hull_coverage_uniform_grid():
    # Dense uniform grid: peeling one layer barely changes coverage
    ref = _rect(0, 0, 100, 100)
    inliers = _rect(0, 0, 100, 100, n_per_side=20)
    cov = hull_coverage(inliers, ref)
    pcov = peeled_hull_coverage(inliers, ref)
    # Both should be high and close to each other
    assert pcov > 0.8
    assert pcov <= cov + 1e-6  # peeled ≤ full


def test_peeled_hull_coverage_outlier_driven():
    # Bulk of inliers bunched at center; 4 outliers at corners inflate hull_coverage.
    # peeled_hull_coverage should be much lower (reflects the bulk only).
    rng = np.random.default_rng(2)
    ref = _rect(0, 0, 100, 100)
    bulk = rng.normal(loc=[50, 50], scale=2.0, size=(100, 2))
    outliers = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    inliers = np.vstack([bulk, outliers])

    cov = hull_coverage(inliers, ref)
    pcov = peeled_hull_coverage(inliers, ref)

    assert cov > 0.8          # 4 outliers push hull to full extent
    assert pcov < 0.05        # interior (bulk) is bunched → low peeled coverage


def test_peeled_hull_coverage_leq_hull_coverage():
    # peeled is always ≤ hull_coverage (interior hull ⊆ outer hull)
    rng = np.random.default_rng(3)
    ref = _rect(0, 0, 100, 100)
    for seed in range(10):
        rng2 = np.random.default_rng(seed)
        inliers = rng2.uniform(0, 100, size=(50, 2))
        assert peeled_hull_coverage(inliers, ref) <= hull_coverage(inliers, ref) + 1e-9


def test_peeled_hull_coverage_collinear():
    ref = _rect(0, 0, 100, 100)
    pts = np.column_stack([np.linspace(0, 100, 50), np.full(50, 50.0)])
    # Collinear → no hull area → 0
    assert peeled_hull_coverage(pts, ref) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# sample_mask_pts
# ---------------------------------------------------------------------------


def test_sample_mask_pts_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    pts = sample_mask_pts(mask)
    assert pts.shape == (0, 2)


def test_sample_mask_pts_full():
    mask = np.full((64, 64), 255, dtype=np.uint8)
    pts = sample_mask_pts(mask, stride=8)
    assert pts.shape[1] == 2
    assert len(pts) == (64 // 8) ** 2


def test_sample_mask_pts_bounds():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:50, 20:80] = 255
    pts = sample_mask_pts(mask, stride=4)
    assert pts.shape[1] == 2
    assert pts[:, 0].min() >= 20
    assert pts[:, 0].max() <= 79
    assert pts[:, 1].min() >= 10
    assert pts[:, 1].max() <= 49
