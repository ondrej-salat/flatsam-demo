"""Tests for the per-corner position uncertainty of both homography branches.

The analytic line-intersection covariance is checked against Monte Carlo over
the same generative model — that is the only honest check, and the one the old
flatsam prototype never got to run (its bootstrap passed points where line
parameters were expected, so it always errored out).
"""

import numpy as np

from flatsam2.corner_covariance import (
    _floor_cov_eigenvalues,
    bootstrap_corner_covs,
    corner_blocks,
    corners_to_theta,
    fit_line_tls,
    homography_corner_covs,
    line_intersection_cov,
    mahalanobis_sq,
    theta_to_corners,
)


_NOISE_PX = 1.0
_INIT_CORNERS = np.array([[100.0, 220.0, 220.0, 100.0], [60.0, 60.0, 180.0, 180.0]])


def _noisy_segment(
    start: np.ndarray, end: np.ndarray, n_pts: int, rng: np.random.Generator
) -> np.ndarray:
    """(2, n) points along a segment with perpendicular Gaussian noise."""
    direction = end - start
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)
    pts = start[:, None] + direction[:, None] * np.linspace(0.0, 1.0, n_pts)
    return pts + normal[:, None] * rng.normal(0.0, _NOISE_PX, n_pts)


def _monte_carlo_cov(
    spec_a: tuple, spec_b: tuple, n_trials: int, rng: np.random.Generator
) -> np.ndarray:
    """Empirical (2, 2) covariance of the intersection over fresh point noise."""
    corners = [
        line_intersection_cov(
            _noisy_segment(*spec_a, rng), _noisy_segment(*spec_b, rng), _NOISE_PX
        )[0]
        for _ in range(n_trials)
    ]
    return np.cov(np.asarray(corners), rowvar=False)


def test_tls_offset_uncertainty_shrinks_with_sample_size():
    """sigma_rho follows the sigma / sqrt(N) it claims to be."""
    rng = np.random.default_rng(0)
    _, _, _, sigma_rho_small = fit_line_tls(
        _noisy_segment(np.array([0.0, 0.0]), np.array([300.0, 0.0]), 50, rng), _NOISE_PX
    )
    _, _, _, sigma_rho_big = fit_line_tls(
        _noisy_segment(np.array([0.0, 0.0]), np.array([300.0, 0.0]), 200, rng), _NOISE_PX
    )
    assert sigma_rho_small == np.sqrt(1.0 / 50)
    assert np.isclose(sigma_rho_small / sigma_rho_big, 2.0)


def test_intersection_cov_matches_monte_carlo():
    """Analytic covariance agrees with resampling, well-conditioned and shallow."""
    rng = np.random.default_rng(1)
    cases = {
        "orthogonal": (
            (np.array([250.0, 400.0]), np.array([550.0, 400.0]), 200),
            (np.array([400.0, 250.0]), np.array([400.0, 550.0]), 200),
        ),
        "shallow_15deg": (
            (np.array([250.0, 400.0]), np.array([550.0, 400.0]), 200),
            (np.array([255.0, 361.0]), np.array([545.0, 439.0]), 200),
        ),
    }
    for name, (spec_a, spec_b) in cases.items():
        _, cov = line_intersection_cov(
            _noisy_segment(*spec_a, rng), _noisy_segment(*spec_b, rng), _NOISE_PX
        )
        empirical = _monte_carlo_cov(spec_a, spec_b, 2000, rng)
        ratio = np.sqrt(np.diag(cov)) / np.sqrt(np.diag(empirical))
        assert np.allclose(ratio, 1.0, atol=0.15), f"{name}: sd ratio {ratio}"


def test_intersection_cov_grows_when_the_corner_is_extrapolated():
    """Short boundary arcs far from their meeting point give a wide ellipse.

    This is the behaviour the whole thing rests on: uncertainty tracks the lever
    arm from the evidence to the corner, not the fraction of the object covered.
    """
    rng = np.random.default_rng(2)
    near_a = _noisy_segment(np.array([250.0, 400.0]), np.array([550.0, 400.0]), 60, rng)
    near_b = _noisy_segment(np.array([400.0, 250.0]), np.array([400.0, 550.0]), 60, rng)
    far_a = _noisy_segment(np.array([100.0, 100.0]), np.array([180.0, 100.0]), 60, rng)
    far_b = _noisy_segment(np.array([600.0, 140.0]), np.array([600.0, 220.0]), 60, rng)

    _, cov_near = line_intersection_cov(near_a, near_b, _NOISE_PX)
    _, cov_far = line_intersection_cov(far_a, far_b, _NOISE_PX)
    assert np.max(np.diag(cov_far)) > 10.0 * np.max(np.diag(cov_near))


def test_intersection_cov_is_calibrated():
    """The squared Mahalanobis error behaves like chi-square with 2 DoF."""
    rng = np.random.default_rng(3)
    truth = np.array([400.0, 400.0])
    spec_a = (np.array([250.0, 400.0]), np.array([550.0, 400.0]), 200)
    spec_b = (np.array([400.0, 250.0]), np.array([400.0, 550.0]), 200)
    d_sq = []
    for _ in range(1000):
        corner, cov = line_intersection_cov(
            _noisy_segment(*spec_a, rng), _noisy_segment(*spec_b, rng), _NOISE_PX
        )
        d_sq.append(mahalanobis_sq(corner - truth, cov))
    d_sq = np.asarray(d_sq)
    assert 1.0 < np.median(d_sq) < 1.9          # chi2_2 median 1.386
    assert 0.02 < np.mean(d_sq > 5.99) < 0.09   # chi2_2 upper 5%


def _translated_correspondences(
    src: np.ndarray, noise_px: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correspondences under a pure translation, plus init + resulting corners."""
    shift = np.array([12.0, -7.0])
    dst = src + shift + rng.normal(0.0, noise_px, src.shape)
    return src, dst, _INIT_CORNERS, _INIT_CORNERS + shift[:, None]


def test_homography_cov_is_tighter_for_well_spread_correspondences():
    """Correspondences spanning the template beat ones bunched along one edge.

    Both sets have the same size and the same residual noise, so only the
    geometry differs — which is the claim the whole feature rests on.
    """
    rng = np.random.default_rng(4)
    spread = rng.uniform([100.0, 60.0], [220.0, 180.0], size=(120, 2))
    edge = np.stack(
        [rng.uniform(100.0, 220.0, 120), rng.normal(60.0, 1.5, 120)], axis=1
    )
    cov_spread = homography_corner_covs(*_translated_correspondences(spread, 0.5, rng))
    cov_edge = homography_corner_covs(*_translated_correspondences(edge, 0.5, rng))
    far_corner = 2  # opposite the edge the bunched points sit on
    assert np.max(np.diag(cov_edge[far_corner])) > 5.0 * np.max(
        np.diag(cov_spread[far_corner])
    )


def test_homography_cov_matches_the_bootstrap():
    """The analytic covariance agrees with resampling the correspondences.

    This is the check the old prototype never ran on the line covariance, where a
    parameterization error survived at a factor of five.  Loose tolerance: the
    two make different modelling assumptions (residual propagation on a fixed
    correspondence set vs resampling the set), so they agree in scale, not to the
    digit.
    """
    rng = np.random.default_rng(5)
    src = rng.uniform([100.0, 60.0], [220.0, 180.0], size=(150, 2))
    args = _translated_correspondences(src, 0.8, rng)
    analytic = homography_corner_covs(*args)
    boot = bootstrap_corner_covs(args[0], args[1], _INIT_CORNERS, n_bootstrap=400, seed=0)
    ratio = np.sqrt(np.diagonal(analytic, axis1=1, axis2=2)) / np.sqrt(
        np.diagonal(boot, axis1=1, axis2=2)
    )
    assert np.allclose(ratio, 1.0, atol=0.35), f"sd ratio {ratio}"


def test_theta_layout_is_corner_major():
    """``corners_to_theta`` interleaves x and y, so corner k sits at 2k, 2k+1.

    Every per-corner block downstream is a 2x2 slice at that offset, in both
    estimators. An xy-major layout (all x then all y) would put two *different
    corners'* x coordinates in the first block, which is exactly the failure this
    pins down.
    """
    corners = np.array([[11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0]])
    theta = corners_to_theta(corners)
    np.testing.assert_allclose(theta, [11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0])
    for k in range(4):
        np.testing.assert_allclose(theta[2 * k : 2 * k + 2], corners[:, k])
    np.testing.assert_allclose(theta_to_corners(theta), corners)


def test_corner_blocks_reads_one_corner_per_block():
    """Block k must hold corner k's (x, y) variances, not two corners' x's."""
    variances = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    blocks = corner_blocks(np.diag(variances))
    for k in range(4):
        np.testing.assert_allclose(np.diag(blocks[k]), variances[2 * k : 2 * k + 2])


def test_homography_cov_localizes_uncertainty_to_the_starved_corner():
    """Drop the correspondences around one corner; that corner's block grows.

    Run for each of the four corners in turn, so a block-ordering error cannot
    pass by being right about one of them.
    """
    rng = np.random.default_rng(11)
    src = rng.uniform([100.0, 60.0], [220.0, 180.0], size=(400, 2))
    for k in range(4):
        keep = np.linalg.norm(src - _INIT_CORNERS[:, k], axis=1) > 90.0
        cov = homography_corner_covs(*_translated_correspondences(src[keep], 0.5, rng))
        traces = [float(np.trace(c)) for c in cov]
        assert np.argmax(traces) == k, f"starved corner {k}, traces {traces}"


def test_homography_cov_matches_the_bootstrap_corner_by_corner():
    """Lopsided evidence: analytic and resampled agree at *each* corner.

    ``bootstrap_corner_covs`` gets its per-corner index by warping the template
    corners through refitted homographies, so its corner identity is independent
    of the parameter layout the analytic route uses. Starving one corner makes
    the four covariances differ by an order of magnitude, so the comparison is
    only passable if the analytic blocks are attached to the right corners.
    """
    rng = np.random.default_rng(13)
    src = rng.uniform([100.0, 60.0], [220.0, 180.0], size=(400, 2))
    src = src[np.linalg.norm(src - _INIT_CORNERS[:, 0], axis=1) > 90.0]
    # 3 px noise, so no corner sits on the 0.25 px eigenvalue floor where the four
    # would be indistinguishable.
    args = _translated_correspondences(src, 3.0, rng)
    analytic = homography_corner_covs(*args)
    boot = bootstrap_corner_covs(args[0], args[1], _INIT_CORNERS, n_bootstrap=600, seed=0)
    traces_analytic = [float(np.trace(c)) for c in analytic]
    traces_boot = [float(np.trace(c)) for c in boot]
    assert min(traces_analytic[1:]) > 4.0 * 0.25**2, "a corner is on the floor"
    assert np.argmax(traces_analytic) == 0 and np.argmax(traces_boot) == 0, (
        f"the starved corner is not the loosest: analytic {traces_analytic}, "
        f"bootstrap {traces_boot}"
    )
    ratio = np.sqrt(np.diagonal(analytic, axis1=1, axis2=2)) / np.sqrt(
        np.diagonal(boot, axis1=1, axis2=2)
    )
    assert np.allclose(ratio, 1.0, atol=0.35), f"per-corner sd ratio {ratio}"


def test_floor_cov_eigenvalues_leaves_wide_covariance_untouched():
    """A covariance already above the floor should be returned unchanged."""
    cov = np.diag([4.0, 9.0])  # std 2px, 3px — well above the 0.25px floor
    np.testing.assert_allclose(_floor_cov_eigenvalues(cov), cov, atol=1e-10)


def test_floor_cov_eigenvalues_clamps_near_singular_covariance():
    """A near-collinear-evidence covariance (one axis ~0) must not blow up
    any downstream Mahalanobis distance to numerical nonsense.
    """
    tiny_axis_cov = np.diag([1e-10, 25.0])  # one axis absurdly confident
    floored = _floor_cov_eigenvalues(tiny_axis_cov, min_std_px=0.25)
    eigvals = np.linalg.eigvalsh(floored)
    assert eigvals.min() >= 0.25**2 - 1e-12
    # The wide axis (25.0, well above the floor) must be unaffected.
    assert np.isclose(eigvals.max(), 25.0)
    # A 1px real disagreement against the *unfloored* cov would have given a
    # squared Mahalanobis distance of ~1e10; floored it stays sane.
    d2 = mahalanobis_sq(np.array([1.0, 0.0]), floored)
    assert d2 < 20.0


def test_homography_cov_declines_too_few_correspondences():
    """Fewer correspondences than a stable fit needs returns None, not a guess."""
    pts = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    assert homography_corner_covs(pts, pts + 1.0, _INIT_CORNERS, _INIT_CORNERS) is None
    assert bootstrap_corner_covs(pts, pts + 1.0, _INIT_CORNERS) is None


# --------------------------------------------------------------------------- #
# Degenerate fits must be rejected, not floored into confidence
# --------------------------------------------------------------------------- #
def test_indefinite_block_is_rejected_rather_than_floored():
    """A negative eigenvalue floored to min_std would be *maximum* confidence."""
    from flatsam2.corner_covariance import corner_blocks

    theta_cov = np.eye(8)
    theta_cov[0, 0] = -5.0          # materially negative, not round-off
    assert corner_blocks(theta_cov) is None


def test_round_off_negatives_are_still_floored():
    from flatsam2.corner_covariance import corner_blocks

    theta_cov = np.eye(8)
    theta_cov[0, 0] = -1e-14
    blocks = corner_blocks(theta_cov)
    assert blocks is not None
    assert np.all(np.linalg.eigvalsh(blocks[0]) > 0)


def test_rank_deficient_normal_matrix_is_refused():
    """An unconstrained direction cannot be given a covariance."""
    from flatsam2.corner_covariance import invert_normal_matrix

    normal = np.eye(8)
    normal[3, 3] = 0.0              # one parameter the evidence says nothing about
    assert invert_normal_matrix(normal, var=1.0) is None


def test_well_conditioned_normal_matrix_inverts():
    from flatsam2.corner_covariance import invert_normal_matrix

    cov = invert_normal_matrix(np.diag(np.arange(1.0, 9.0)), var=2.0)
    assert cov is not None
    assert np.allclose(np.diag(cov), 2.0 / np.arange(1.0, 9.0))
