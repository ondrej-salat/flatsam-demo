"""Tests: vectorised Hough transform matches scalar reference.

The scalar reference implementations are the pre-vectorisation code copied
verbatim.  Tests compare the vectorised module functions against them on
several synthetic masks.
"""

import math

import cv2
import numpy as np
import pytest
import scipy.ndimage
from einops import rearrange
from skimage.feature import peak_local_max

from flatsam2.hough_homography import (
    H_warp_lines,
    _approx_touching_edges,
    _remap,
    contour_hough,
    mask_to_homography,
    refine_lines,
)


# ---------------------------------------------------------------------------
# Scalar reference implementations (pre-vectorisation code, verbatim)
# ---------------------------------------------------------------------------


def _contour_hough_scalar(
    mask,
    contours,
    on_boundaries,
    *,
    gradient_step=4,
    max_samples=1_000_000,
    topk=4,
):
    """Scalar reference for contour_hough."""
    if len(contours) == 0:
        return np.zeros((3, 0))

    all_cnt = np.concatenate(contours, axis=1)
    contour_center = np.mean(all_cnt, axis=1, keepdims=True)
    centered = [cnt - contour_center for cnt in contours]

    point_dists = [np.sqrt(np.sum(cnt**2, axis=0)) for cnt in centered]
    max_dist = max(np.amax(d) for d in point_dists)
    min_dist = 0.0

    N_angles, N_dists = 359, 100
    accumulator = np.zeros((N_dists, N_angles))
    degree_range = 10
    rad_range = np.radians(degree_range)
    delta_angles = np.linspace(-rad_range, rad_range, 11).tolist()

    for pts, on_boundary in zip(centered, on_boundaries, strict=True):
        N = pts.shape[1]
        step_outer = max(N // max_samples, 1)
        for i in range(0, N, step_outer):
            if on_boundary[i]:
                continue

            z = pts[:, (i - gradient_step + N) % N]
            a = pts[:, i]
            b = pts[:, (i + gradient_step) % N]

            diff = b - z
            if diff[0] == 0 and diff[1] == 0:
                continue
            angle = np.arctan2(diff[1], diff[0])
            dist = math.cos(angle) * a[1] - math.sin(angle) * a[0]

            norm_angle = angle + np.pi / 2
            norm_to_direction = -np.pi / 2
            if dist < 0:
                norm_angle -= np.pi
                norm_to_direction = np.pi / 2

            for da in delta_angles:
                res_angle = (norm_angle + da + 2 * np.pi) % (2 * np.pi)
                dir_angle = float(res_angle + norm_to_direction)
                d = abs(
                    math.cos(dir_angle) * a[1] - math.sin(dir_angle) * a[0]
                )
                amount = _remap(abs(da), 0, rad_range, 1, 0.5)

                row = int(
                    round(_remap(d, min_dist, max_dist, 0, N_dists - 1))
                )
                row = max(0, min(row, N_dists - 1))
                col = int(
                    round(_remap(res_angle, 0, 2 * np.pi, 0, N_angles - 1))
                )
                col = max(0, min(col, N_angles - 1))
                accumulator[row, col] += amount

    # Peak detection — same as module code.
    cyclic = np.tile(accumulator, (1, 3))
    cyclic = scipy.ndimage.gaussian_filter(cyclic, sigma=4)

    peaks = peak_local_max(
        cyclic, min_distance=10, threshold_rel=None, exclude_border=False,
    )
    peaks = peaks[
        (peaks[:, 1] >= N_angles) & (peaks[:, 1] < 2 * N_angles), :
    ]
    peaks[:, 1] -= N_angles

    if len(peaks) == 0:
        return np.zeros((3, 0))

    blurred = cyclic[:, N_angles : 2 * N_angles]
    peak_vals = blurred[peaks[:, 0], peaks[:, 1]]
    real_topk = min(topk, len(peak_vals))
    top_ids = np.argpartition(peak_vals, -real_topk)[-real_topk:]
    peaks = peaks[top_ids]
    peaks = peaks[np.argsort(peaks[:, 1])]

    lines = []
    for peak in peaks:
        distance = _remap(peak[0], 0, N_dists - 1, min_dist, max_dist)
        angle = _remap(peak[1], 0, N_angles - 1, 0, 2 * np.pi)
        lines.append(np.array([np.cos(angle), np.sin(angle), -distance]))

    lines = rearrange(np.array(lines), "N abc -> abc N", abc=3)

    H_uncenter = np.eye(3)
    H_uncenter[0, 2] = contour_center[0, 0]
    H_uncenter[1, 2] = contour_center[1, 0]
    lines = H_warp_lines(H_uncenter, lines)

    return lines


def _refine_lines_scalar(
    lines,
    contours,
    on_boundaries,
    *,
    gradient_step=4,
    max_samples=100_000,
):
    """Scalar reference for refine_lines."""
    N_lines = lines.shape[1]
    if N_lines == 0:
        return lines, []

    normalsT = lines[:2, :].T
    distances = lines[2, :]
    line_pts = [[] for _ in range(N_lines)]

    for pts, on_boundary in zip(contours, on_boundaries, strict=True):
        N = pts.shape[1]
        step_outer = max(N // max_samples, 1)
        for i in range(0, N, step_outer):
            if on_boundary[i]:
                continue

            z = pts[:, (i - gradient_step + N) % N]
            a = pts[:, i]
            b = pts[:, (i + gradient_step) % N]

            line_vec = (b - z).astype(np.float64)
            norm = np.hypot(line_vec[0], line_vec[1])
            if norm == 0:
                continue
            line_vec /= norm

            dist_diffs = (
                (normalsT @ a[:, np.newaxis]).flatten() + distances
            )

            direction_dots = np.abs(
                (normalsT @ line_vec[:, np.newaxis]).flatten()
            )
            angle_diffs = np.abs(
                np.arccos(np.clip(direction_dots, 0, 1)) - np.pi / 2
            )

            valid_angle = np.degrees(angle_diffs) <= 15
            valid_dist = np.abs(dist_diffs) <= 15
            valid = valid_angle & valid_dist
            min_dist_i = np.argmin(np.abs(dist_diffs))

            if valid[min_dist_i]:
                line_pts[min_dist_i].append(pts[:, i])

    new_lines = []
    for line_i, pts_list in enumerate(line_pts):
        if not pts_list:
            new_lines.append(lines[:, line_i])
            continue

        pts_arr = np.array(pts_list)
        A = np.column_stack([pts_arr, np.ones(len(pts_list))])
        _, _, vt = np.linalg.svd(A)
        a, b, c = vt[-1, :]
        n = math.hypot(a, b)
        if n > 0:
            a, b, c = a / n, b / n, c / n
        if c > 0:
            a, b, c = -a, -b, -c
        new_lines.append(np.array([a, b, c]))

    refined = rearrange(np.array(new_lines), "N abc -> abc N", abc=3)
    pts_out = [
        np.array(p).T if p else np.zeros((2, 0)) for p in line_pts
    ]
    return refined, pts_out


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_quad_mask(h, w, corners):
    """Create a filled quadrilateral mask (H, W) uint8 from (2, 4) corners."""
    poly = rearrange(corners.astype(np.int32), "xy N -> N 1 xy", xy=2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _extract_contours(mask, min_length=20):
    """Extract contours and boundary flags from a mask.

    Returns ``(contours, on_boundaries)`` ready for contour_hough /
    refine_lines.
    """
    contours_cv, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    contours_cv = [c for c in contours_cv if c.shape[0] > min_length]
    contours = [
        rearrange(c, "N 1 xy -> xy N", xy=2) for c in contours_cv
    ]
    on_boundaries = [
        np.logical_or.reduce(
            (
                _approx_touching_edges(
                    c + np.array([[0], [1]]), mask.shape, margin=1
                ),
                _approx_touching_edges(
                    c + np.array([[0], [-1]]), mask.shape, margin=1
                ),
                _approx_touching_edges(
                    c + np.array([[1], [0]]), mask.shape, margin=1
                ),
                _approx_touching_edges(
                    c + np.array([[-1], [0]]), mask.shape, margin=1
                ),
            )
        )
        for c in contours
    ]
    return contours, on_boundaries


def _make_noisy_polygon_mask(h, w, corners, noise_std=8.0, n_extra_vertices=6, seed=0):
    """Create a mask from a noisy polygon derived from *corners*.

    Each edge of the quadrilateral is subdivided into *n_extra_vertices*
    additional points, and all vertices (original + extra) receive
    independent Gaussian noise.  The result is an irregular closed polygon
    whose overall shape is still recognisable as the original quad but whose
    contour is rough and non-straight — a more realistic / stress-test input
    for the Hough pipeline.

    Args:
        h, w: Image dimensions.
        corners: (2, 4) float64 corner coordinates.
        noise_std: Standard deviation of per-vertex Gaussian noise (pixels).
        n_extra_vertices: Number of extra vertices to insert per edge.
        seed: RNG seed for reproducibility.

    Returns:
        (H, W) uint8 mask.
    """
    rng = np.random.default_rng(seed)
    N = corners.shape[1]
    pts = []
    for i in range(N):
        p0 = corners[:, i]
        p1 = corners[:, (i + 1) % N]
        # Original corner + subdivided edge points (excluding the next corner).
        for t in np.linspace(0, 1, n_extra_vertices + 2)[:-1]:
            pts.append(p0 + t * (p1 - p0))
    poly = np.array(pts, dtype=np.float64)          # (M, 2)
    poly += rng.normal(0, noise_std, poly.shape)
    poly = np.clip(poly, 0, [[w - 1, h - 1]])
    poly_cv = poly.astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_cv], 255)
    return mask


# Parametrised shapes: (corners, image_shape)
_SHAPES = [
    pytest.param(
        np.array(
            [[100, 540, 540, 100], [80, 80, 400, 400]], dtype=np.float64
        ),
        (480, 640),
        id="rectangle",
    ),
    pytest.param(
        np.array(
            [[200, 500, 420, 120], [60, 180, 400, 280]], dtype=np.float64
        ),
        (480, 640),
        id="trapezoid",
    ),
    pytest.param(
        np.array(
            [[160, 480, 500, 140], [50, 100, 420, 370]], dtype=np.float64
        ),
        (480, 640),
        id="parallelogram",
    ),
]


# ---------------------------------------------------------------------------
# Tests — contour_hough
# ---------------------------------------------------------------------------


class TestContourHough:
    """Vectorised contour_hough must match the scalar reference."""

    @pytest.mark.parametrize("corners, shape", _SHAPES)
    def test_lines_match(self, corners, shape):
        h, w = shape
        mask = _make_quad_mask(h, w, corners)
        contours, on_boundaries = _extract_contours(mask)

        scalar_lines = _contour_hough_scalar(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        vec_lines = contour_hough(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )

        assert scalar_lines.shape == vec_lines.shape, (
            f"Shape mismatch: {scalar_lines.shape} vs {vec_lines.shape}"
        )
        np.testing.assert_allclose(
            vec_lines,
            scalar_lines,
            atol=1e-10,
            err_msg="Vectorised contour_hough lines differ from scalar",
        )

    def test_empty_contours(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = contour_hough(mask, [], [])
        assert result.shape == (3, 0)

    @pytest.mark.parametrize("corners, shape", _SHAPES[:1])
    def test_topk_respected(self, corners, shape):
        h, w = shape
        mask = _make_quad_mask(h, w, corners)
        contours, on_boundaries = _extract_contours(mask)

        for k in [2, 3, 4]:
            lines = contour_hough(
                mask, contours, on_boundaries, topk=k,
            )
            assert lines.shape[1] <= k


# ---------------------------------------------------------------------------
# Tests — refine_lines
# ---------------------------------------------------------------------------


class TestRefineLines:
    """Vectorised refine_lines must match the scalar reference."""

    @pytest.mark.parametrize("corners, shape", _SHAPES)
    def test_refined_lines_match(self, corners, shape):
        h, w = shape
        mask = _make_quad_mask(h, w, corners)
        contours, on_boundaries = _extract_contours(mask)

        # Use scalar Hough as input (identical for both).
        hough_lines = _contour_hough_scalar(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        if hough_lines.shape[1] == 0:
            pytest.skip("No Hough lines detected")

        scalar_lines, _ = _refine_lines_scalar(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )
        vec_lines, _ = refine_lines(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )

        assert scalar_lines.shape == vec_lines.shape
        np.testing.assert_allclose(
            vec_lines,
            scalar_lines,
            atol=1e-10,
            err_msg="Vectorised refine_lines differ from scalar reference",
        )

    @pytest.mark.parametrize("corners, shape", _SHAPES)
    def test_point_assignments_match(self, corners, shape):
        h, w = shape
        mask = _make_quad_mask(h, w, corners)
        contours, on_boundaries = _extract_contours(mask)

        hough_lines = _contour_hough_scalar(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        if hough_lines.shape[1] == 0:
            pytest.skip("No Hough lines detected")

        _, scalar_pts = _refine_lines_scalar(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )
        _, vec_pts = refine_lines(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )

        assert len(scalar_pts) == len(vec_pts)
        for i, (sp, vp) in enumerate(zip(scalar_pts, vec_pts)):
            assert sp.shape == vp.shape, (
                f"Line {i}: point shape mismatch {sp.shape} vs {vp.shape}"
            )
            if sp.size > 0:
                np.testing.assert_array_equal(
                    vp, sp,
                    err_msg=f"Line {i}: point coordinates differ",
                )

    def test_empty_lines(self):
        lines = np.zeros((3, 0))
        result, pts = refine_lines(lines, [], [])
        assert result.shape == (3, 0)
        assert pts == []


# ---------------------------------------------------------------------------
# Tests — noisy / irregular polygon inputs
# ---------------------------------------------------------------------------

# Each entry: (base_corners, noise_std, n_extra_vertices, seed)
_NOISY_CASES = [
    pytest.param(
        np.array([[100, 540, 540, 100], [80, 80, 400, 400]], dtype=np.float64),
        8.0, 6, 0,
        id="noisy_rectangle",
    ),
    pytest.param(
        np.array([[200, 500, 420, 120], [60, 180, 400, 280]], dtype=np.float64),
        12.0, 8, 1,
        id="noisy_trapezoid",
    ),
    pytest.param(
        np.array([[160, 480, 500, 140], [50, 100, 420, 370]], dtype=np.float64),
        15.0, 10, 42,
        id="noisy_parallelogram_heavy",
    ),
    pytest.param(
        # Near-degenerate: thin elongated quad with lots of noise
        np.array([[80, 560, 550, 90], [200, 210, 270, 260]], dtype=np.float64),
        6.0, 12, 7,
        id="noisy_thin_quad",
    ),
]


class TestNoisyInputs:
    """Scalar vs vectorised on irregular noisy polygon masks.

    These masks have rough, non-straight contours — more representative
    of real segmentation masks than clean geometric shapes.
    """

    @pytest.mark.parametrize("corners, noise_std, n_extra, seed", _NOISY_CASES)
    def test_contour_hough_matches_scalar(self, corners, noise_std, n_extra, seed):
        mask = _make_noisy_polygon_mask(
            480, 640, corners, noise_std=noise_std,
            n_extra_vertices=n_extra, seed=seed,
        )
        contours, on_boundaries = _extract_contours(mask)

        scalar_lines = _contour_hough_scalar(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        vec_lines = contour_hough(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )

        assert scalar_lines.shape == vec_lines.shape, (
            f"Shape mismatch: {scalar_lines.shape} vs {vec_lines.shape}"
        )
        np.testing.assert_allclose(
            vec_lines, scalar_lines, atol=1e-10,
            err_msg="contour_hough mismatch on noisy input",
        )

    @pytest.mark.parametrize("corners, noise_std, n_extra, seed", _NOISY_CASES)
    def test_refine_lines_matches_scalar(self, corners, noise_std, n_extra, seed):
        mask = _make_noisy_polygon_mask(
            480, 640, corners, noise_std=noise_std,
            n_extra_vertices=n_extra, seed=seed,
        )
        contours, on_boundaries = _extract_contours(mask)

        hough_lines = _contour_hough_scalar(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        if hough_lines.shape[1] == 0:
            pytest.skip("No Hough lines detected")

        scalar_lines, scalar_pts = _refine_lines_scalar(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )
        vec_lines, vec_pts = refine_lines(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )

        assert scalar_lines.shape == vec_lines.shape
        np.testing.assert_allclose(
            vec_lines, scalar_lines, atol=1e-10,
            err_msg="refine_lines mismatch on noisy input",
        )

        assert len(scalar_pts) == len(vec_pts)
        for i, (sp, vp) in enumerate(zip(scalar_pts, vec_pts)):
            assert sp.shape == vp.shape, (
                f"Line {i}: point shape mismatch {sp.shape} vs {vp.shape}"
            )
            if sp.size > 0:
                np.testing.assert_array_equal(
                    vp, sp,
                    err_msg=f"Line {i}: point coordinates differ",
                )

    @pytest.mark.parametrize("corners, noise_std, n_extra, seed", _NOISY_CASES[:2])
    def test_different_gradient_steps(self, corners, noise_std, n_extra, seed):
        """Verify match for non-default gradient_step values."""
        mask = _make_noisy_polygon_mask(
            480, 640, corners, noise_std=noise_std,
            n_extra_vertices=n_extra, seed=seed,
        )
        contours, on_boundaries = _extract_contours(mask)

        for step in [2, 6, 10]:
            scalar_lines = _contour_hough_scalar(
                mask, contours, on_boundaries, gradient_step=step, topk=4,
            )
            vec_lines = contour_hough(
                mask, contours, on_boundaries, gradient_step=step, topk=4,
            )
            assert scalar_lines.shape == vec_lines.shape
            np.testing.assert_allclose(
                vec_lines, scalar_lines, atol=1e-10,
                err_msg=f"contour_hough mismatch at gradient_step={step}",
            )


# Masks used by centering and sharing tests: clean quads + noisy variants.
def _all_test_masks():
    """Return list of (mask, contours, on_boundaries) for diverse inputs."""
    cases = []
    h, w = 480, 640
    quad_corners = [
        np.array([[100, 540, 540, 100], [80, 80, 400, 400]], dtype=np.float64),
        np.array([[200, 500, 420, 120], [60, 180, 400, 280]], dtype=np.float64),
        np.array([[160, 480, 500, 140], [50, 100, 420, 370]], dtype=np.float64),
    ]
    for corners in quad_corners:
        mask = _make_quad_mask(h, w, corners)
        contours, on_boundaries = _extract_contours(mask)
        if contours:
            cases.append((mask, contours, on_boundaries))
    # Two noisy variants.
    for corners, noise_std, n_extra, seed in [
        (quad_corners[0], 8.0, 6, 0),
        (quad_corners[1], 12.0, 8, 1),
    ]:
        mask = _make_noisy_polygon_mask(h, w, corners,
                                        noise_std=noise_std,
                                        n_extra_vertices=n_extra, seed=seed)
        contours, on_boundaries = _extract_contours(mask)
        if contours:
            cases.append((mask, contours, on_boundaries))
    return cases


# ---------------------------------------------------------------------------
# Tests — centering correctness
# ---------------------------------------------------------------------------


class TestCentering:
    """Verify that the uncentered-samples + late-centering path is equivalent
    to the original centred-contours approach.

    The refactored code stores points in image coordinates inside
    _ContourSamples, then subtracts contour_center inside contour_hough.
    Tangent vectors (differences) are translation-invariant, so they must
    be identical whether computed before or after centering.  Point positions
    must differ by exactly contour_center.
    """

    @pytest.mark.parametrize(
        "mask, contours, on_boundaries",
        _all_test_masks(),
        ids=[f"case{i}" for i in range(len(_all_test_masks()))],
    )
    def test_tangents_are_translation_invariant(self, mask, contours, on_boundaries):
        """_precompute_contour_samples tangents match those from centred contours.

        Tangent vectors are finite differences so the translation cancels
        exactly.  Point coordinates must differ by the center offset.
        Uses allclose (not array_equal) to tolerate int32→float64 dtype
        promotion when contours are pre-centred.
        """
        from flatsam2.hough_homography import _precompute_contour_samples

        all_cnt = np.concatenate(contours, axis=1)
        center = np.mean(all_cnt, axis=1, keepdims=True)  # (2, 1)

        samples_orig = _precompute_contour_samples(
            contours, on_boundaries, gradient_step=4,
        )
        centred_contours = [c - center for c in contours]
        samples_cent = _precompute_contour_samples(
            centred_contours, on_boundaries, gradient_step=4,
        )

        assert len(samples_orig) == len(samples_cent)
        for i, (s_orig, s_cent) in enumerate(zip(samples_orig, samples_cent)):
            assert s_orig.tangents.shape == s_cent.tangents.shape, (
                f"Contour {i}: tangent shape mismatch"
            )
            # Tangents are translation-invariant: a[b] - a[z] == (a-c)[b] - (a-c)[z].
            np.testing.assert_allclose(
                s_cent.tangents.astype(np.float64),
                s_orig.tangents.astype(np.float64),
                atol=1e-10,
                err_msg=f"Contour {i}: tangents differ after centering",
            )
            # Points must differ by exactly the center offset.
            np.testing.assert_allclose(
                s_cent.points.astype(np.float64),
                s_orig.points.astype(np.float64) - center,
                atol=1e-10,
                err_msg=f"Contour {i}: centred points don't equal orig - center",
            )

    @pytest.mark.parametrize(
        "mask, contours, on_boundaries",
        _all_test_masks(),
        ids=[f"case{i}" for i in range(len(_all_test_masks()))],
    )
    def test_contour_hough_translation_equivariance(self, mask, contours, on_boundaries):
        """Lines from contour_hough are equivariant under rigid translation.

        Shifting all contours by (dx, dy) must shift the output lines by
        the same amount, because lines are always returned in the coordinate
        system of the input contours.  This directly tests that the centering
        step and H_uncenter work together correctly: centering removes the
        mean, the accumulator sees the same centred data, and H_uncenter maps
        lines back to the (shifted) input coordinate system.

        Boundary flags are set to all-False so the test isolates the
        coordinate transformation and is not confused by contour points
        moving in or out of image-boundary zones.
        """
        # All-False boundaries: every contour point is eligible.
        no_boundaries = [np.zeros(c.shape[1], dtype=bool) for c in contours]

        lines_orig = contour_hough(
            mask, contours, no_boundaries, gradient_step=4, topk=4,
        )
        if lines_orig.shape[1] == 0:
            pytest.skip("No Hough lines detected")

        dx, dy = 37.0, -29.0
        shift = np.array([[dx], [dy]])
        shifted_contours = [c.astype(np.float64) + shift for c in contours]

        lines_shifted = contour_hough(
            mask, shifted_contours, no_boundaries, gradient_step=4, topk=4,
        )

        assert lines_orig.shape == lines_shifted.shape, (
            "Number of detected lines changed after translation"
        )
        # Line normals (a, b) are translation-invariant.
        np.testing.assert_allclose(
            lines_shifted[:2, :], lines_orig[:2, :], atol=1e-8,
            err_msg="Line normals changed after translation",
        )
        # Line offset: a(x-dx) + b(y-dy) + c = 0  =>  c' = c - a*dx - b*dy
        expected_c = (
            lines_orig[2, :]
            - lines_orig[0, :] * dx
            - lines_orig[1, :] * dy
        )
        np.testing.assert_allclose(
            lines_shifted[2, :], expected_c, atol=1e-8,
            err_msg="Line offsets wrong after translation",
        )


# ---------------------------------------------------------------------------
# Tests — _samples sharing (intermediate results reuse)
# ---------------------------------------------------------------------------


class TestSamplesSharing:
    """Verify the _samples parameter allows sharing pre-computed tangents
    between contour_hough and refine_lines without changing results.
    """

    @pytest.mark.parametrize(
        "mask, contours, on_boundaries",
        _all_test_masks(),
        ids=[f"case{i}" for i in range(len(_all_test_masks()))],
    )
    def test_shared_samples_contour_hough(self, mask, contours, on_boundaries):
        """contour_hough with explicit _samples matches the default (None) path."""
        from flatsam2.hough_homography import _precompute_contour_samples

        lines_default = contour_hough(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        samples = _precompute_contour_samples(
            contours, on_boundaries, gradient_step=4,
        )
        lines_shared = contour_hough(
            mask, contours, on_boundaries,
            gradient_step=4, topk=4, _samples=samples,
        )

        assert lines_default.shape == lines_shared.shape
        np.testing.assert_allclose(
            lines_shared, lines_default, atol=1e-10,
            err_msg="contour_hough: _samples path differs from default",
        )

    @pytest.mark.parametrize(
        "mask, contours, on_boundaries",
        _all_test_masks(),
        ids=[f"case{i}" for i in range(len(_all_test_masks()))],
    )
    def test_shared_samples_refine_lines(self, mask, contours, on_boundaries):
        """refine_lines with explicit _samples matches the default (None) path."""
        from flatsam2.hough_homography import _precompute_contour_samples

        hough_lines = _contour_hough_scalar(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        if hough_lines.shape[1] == 0:
            pytest.skip("No Hough lines detected")

        lines_default, pts_default = refine_lines(
            hough_lines, contours, on_boundaries, gradient_step=4,
        )
        # Must match max_samples used by refine_lines default path.
        samples = _precompute_contour_samples(
            contours, on_boundaries,
            gradient_step=4, max_samples=100_000,
        )
        lines_shared, pts_shared = refine_lines(
            hough_lines, contours, on_boundaries,
            gradient_step=4, _samples=samples,
        )

        assert lines_default.shape == lines_shared.shape
        np.testing.assert_allclose(
            lines_shared, lines_default, atol=1e-10,
            err_msg="refine_lines: _samples path differs from default",
        )
        for i, (pd, ps) in enumerate(zip(pts_default, pts_shared)):
            assert pd.shape == ps.shape, f"Line {i}: point shape mismatch"
            if pd.size > 0:
                np.testing.assert_array_equal(
                    ps, pd,
                    err_msg=f"Line {i}: point coords differ with shared samples",
                )

    @pytest.mark.parametrize(
        "mask, contours, on_boundaries",
        _all_test_masks()[:2],
        ids=["case0", "case1"],
    )
    def test_same_samples_object_for_both_functions(self, mask, contours, on_boundaries):
        """One _samples object passed to both functions gives the same results
        as independent default calls.

        Uses max_samples=100_000 (the more conservative value) so the shared
        sample set is identical to what refine_lines would compute on its own.
        """
        from flatsam2.hough_homography import _precompute_contour_samples

        # Reference: independent calls, each computing their own samples.
        hough_ref = contour_hough(
            mask, contours, on_boundaries, gradient_step=4, topk=4,
        )
        if hough_ref.shape[1] == 0:
            pytest.skip("No Hough lines detected")
        refined_ref, pts_ref = refine_lines(
            hough_ref, contours, on_boundaries, gradient_step=4,
        )

        # One shared pre-computation for both.
        shared = _precompute_contour_samples(
            contours, on_boundaries,
            gradient_step=4, max_samples=100_000,
        )
        hough_shared = contour_hough(
            mask, contours, on_boundaries,
            gradient_step=4, topk=4, _samples=shared,
        )
        refined_shared, pts_shared = refine_lines(
            hough_ref, contours, on_boundaries,
            gradient_step=4, _samples=shared,
        )

        np.testing.assert_allclose(
            hough_shared, hough_ref, atol=1e-10,
            err_msg="contour_hough differs when using shared samples",
        )
        np.testing.assert_allclose(
            refined_shared, refined_ref, atol=1e-10,
            err_msg="refine_lines differs when using shared samples",
        )
        for i, (pr, ps) in enumerate(zip(pts_ref, pts_shared)):
            assert pr.shape == ps.shape
            if pr.size > 0:
                np.testing.assert_array_equal(ps, pr)


# ---------------------------------------------------------------------------
# Tests — mask_to_homography (end-to-end)
# ---------------------------------------------------------------------------


class TestMaskToHomography:
    """End-to-end smoke test on the full pipeline."""

    def test_rectangle_identity(self):
        corners = np.array(
            [[100, 540, 540, 100], [80, 80, 400, 400]], dtype=np.float64,
        )
        mask = _make_quad_mask(480, 640, corners)
        H, n_corr = mask_to_homography(mask, corners, np.eye(3))
        assert H.shape == (3, 3)
        assert n_corr >= 0
