"""Naming a pose from keypoint correspondences (``flatsam2.match_naming``)."""

import numpy as np

from flatsam2.match_naming import (
    N_RESIDUAL_BINS,
    RESIDUAL_BIN_EDGES,
    alias_supports,
    homography_from_corners,
    match_support,
    name_by_match_support,
    residual_histogram,
    support_from_residuals,
)


_INIT = np.array([[0.0, 200.0, 200.0, 0.0], [0.0, 0.0, 120.0, 120.0]])


def _grid_points(n: int = 6) -> np.ndarray:
    """(N, 2) asymmetric point cloud inside the template quad."""
    xs, ys = np.meshgrid(np.linspace(10, 190, n), np.linspace(10, 110, n))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    # Break the symmetry so the four cyclic namings are genuinely different.
    return pts[pts[:, 0] + 1.7 * pts[:, 1] < 260]


def test_support_counts_effective_matches():
    """Zero residuals give one unit each; residuals at tau give zero."""
    assert support_from_residuals(np.zeros(10), tau_px=30.0) == 10.0
    assert support_from_residuals(np.full(10, 30.0), tau_px=30.0) == 0.0
    assert support_from_residuals(np.full(10, 60.0), tau_px=30.0) == 0.0


def test_support_is_bounded_so_one_gross_outlier_cannot_dominate():
    """The whole reason for a truncated cost rather than L1 or Huber."""
    clean = np.zeros(20)
    with_outlier = np.concatenate([np.zeros(20), [5000.0]])
    assert support_from_residuals(clean, 30.0) == 20.0
    assert support_from_residuals(with_outlier, 30.0) == 20.0


def test_correct_naming_wins_on_a_translated_target():
    pts_src = _grid_points()
    corners = _INIT + np.array([[40.0], [25.0]])
    H = homography_from_corners(_INIT, corners)
    pts_dst = np.stack([
        (H @ np.array([x, y, 1.0]))[:2] / (H @ np.array([x, y, 1.0]))[2]
        for x, y in pts_src
    ])
    support, counts, sum_sq = alias_supports(corners, _INIT, pts_src, pts_dst, tau_px=30.0)
    assert int(np.argmax(support)) == 0
    assert support[0] == len(pts_src)
    assert counts.shape == (4, N_RESIDUAL_BINS)
    assert sum_sq.shape == (4, N_RESIDUAL_BINS)


def test_a_rolled_pose_is_named_back():
    """A pose whose corners are cyclically mislabelled is repaired."""
    pts_src = _grid_points()
    corners = _INIT + np.array([[40.0], [25.0]])
    H = homography_from_corners(_INIT, corners)
    pts_dst = np.stack([
        (H @ np.array([x, y, 1.0]))[:2] / (H @ np.array([x, y, 1.0]))[2]
        for x, y in pts_src
    ])
    mislabelled = np.roll(corners, 2, axis=1)
    named, shift, support = name_by_match_support(
        mislabelled, _INIT, pts_src, pts_dst, tau_px=30.0
    )
    assert shift == 2
    assert np.allclose(named, corners)
    assert np.isfinite(support).all()


def test_too_few_matches_reports_abstention_not_retention():
    """-1 means "no evidence to look at", distinct from 0 = "looked and kept"."""
    corners = np.roll(_INIT, 1, axis=1)
    pts = np.array([[10.0, 10.0], [20.0, 20.0]])
    named, shift, _ = name_by_match_support(corners, _INIT, pts, pts, tau_px=30.0)
    assert shift == -1
    assert np.allclose(named, corners)


def test_missing_matches_report_abstention():
    corners = np.roll(_INIT, 1, axis=1)
    named, shift, _ = name_by_match_support(corners, _INIT, None, None, tau_px=30.0)
    assert shift == -1
    assert np.allclose(named, corners)


def test_keeping_the_incumbent_after_looking_reports_zero():
    """The distinction the telemetry needs: considered, and left alone."""
    pts_src = _grid_points()
    corners = _INIT + np.array([[40.0], [25.0]])
    H = homography_from_corners(_INIT, corners)
    pts_dst = np.stack([
        (H @ np.array([x, y, 1.0]))[:2] / (H @ np.array([x, y, 1.0]))[2]
        for x, y in pts_src
    ])
    _, shift, support = name_by_match_support(corners, _INIT, pts_src, pts_dst, tau_px=30.0)
    assert shift == 0 and np.isfinite(support).all()


def test_histogram_bins_and_squared_sums_are_consistent():
    residuals = np.array([0.5, 1.5, 2.5, 4.0, 12.0, 25.0, 400.0])
    counts, sum_sq = residual_histogram(residuals)
    assert counts.sum() == len(residuals)
    assert np.isclose(sum_sq.sum(), float((residuals**2).sum()))
    # Every residual lands in the bin its magnitude implies.
    edges = np.asarray(RESIDUAL_BIN_EDGES)
    for r in residuals:
        b = int(np.searchsorted(edges, r, side="right") - 1)
        assert counts[b] >= 1


def test_truncated_cost_is_recoverable_from_the_histogram_at_a_bin_edge():
    """Why sum-of-squares is stored beside the counts: exact offline re-test of tau."""
    rng = np.random.default_rng(0)
    residuals = np.abs(rng.normal(0, 12.0, size=500))
    counts, sum_sq = residual_histogram(residuals)
    tau = 15.0  # a bin edge
    direct = float(np.minimum(residuals**2, tau**2).sum())
    edges = np.asarray(RESIDUAL_BIN_EDGES)
    below = edges[1:] <= tau
    from_hist = float(sum_sq[below].sum() + counts[~below].sum() * tau**2)
    assert np.isclose(direct, from_hist)


def test_a_collapsed_pose_explains_almost_nothing():
    """Support is evidence, not a validity check: a bad pose simply scores low."""
    collapsed = np.zeros((2, 4))
    pts = _grid_points()
    support, _, _ = alias_supports(collapsed, _INIT, pts, pts, tau_px=30.0)
    assert np.nanmax(support) < 0.05 * len(pts)


def test_match_support_returns_none_on_an_unusable_warp():
    degenerate = np.zeros((3, 3))
    pts = _grid_points()
    assert match_support(degenerate, pts, pts, tau_px=30.0) is None


def test_support_is_recoverable_from_the_histogram_by_the_rule_tuner():
    """The identity `experiments/naming_rule_replay.py` depends on, checked directly."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from experiments.naming_rule_replay import support_at_tau

    rng = np.random.default_rng(1)
    residuals = np.abs(rng.normal(0.0, 9.0, size=400))
    counts, sum_sq = residual_histogram(residuals)
    edges = np.asarray(RESIDUAL_BIN_EDGES)
    for tau in (10.0, 15.0, 20.0, 30.0):
        direct = support_from_residuals(residuals, tau)
        from_hist = float(support_at_tau(counts, sum_sq, edges, tau))
        assert np.isclose(direct, from_hist), tau


def test_rule_tuner_deployed_pick_respects_raw_match_minimum():
    """Replay must include the shipped eight-match structural guard."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from experiments.naming_rule_replay import _deployed_pick

    edges = np.asarray(RESIDUAL_BIN_EDGES)
    counts = np.zeros((4, len(edges) - 1), dtype=np.int32)
    sum_sq = np.zeros_like(counts, dtype=np.float64)
    counts[1, 0] = 10

    assert _deployed_pick(counts, sum_sq, edges, n_matches=7) == 0
    assert _deployed_pick(counts, sum_sq, edges, n_matches=8) == 1


def test_alias_arbitration_uses_support_not_score():
    """Between relabelings the score is a tie by construction; support decides.

    Regression guard for the placement that `corner_fusion_matchncc` got wrong:
    support must not enter the verification score (whose thresholds are
    calibrated) and must decide the alias comparison.
    """
    from flatsam2.mask_register import Hypothesis, RegisterConfig, _alias_beats, verification_score

    cfg = RegisterConfig(support_power=1.0, support_alias_only=True)
    incumbent = Hypothesis(control=_INIT, source="coarse",
                           features={"iou": 0.9, "ncc": 0.5, "support": 0.10})
    challenger = Hypothesis(control=_INIT, source="alias",
                            features={"iou": 0.9, "ncc": 0.5, "support": 0.90})
    incumbent.score = verification_score(incumbent.features, cfg)
    challenger.score = verification_score(challenger.features, cfg)

    # The support is kept out of the score, so the two tie there ...
    assert incumbent.score == challenger.score
    # ... and the alias comparison is decided by the support alone.
    assert _alias_beats(challenger, incumbent, cfg)
    assert not _alias_beats(incumbent, challenger, cfg)


def test_alias_arbitration_falls_back_to_score_without_support():
    """A frame with too few matches has no naming evidence: keep the old rule."""
    from flatsam2.mask_register import Hypothesis, RegisterConfig, _alias_beats

    cfg = RegisterConfig(support_power=1.0, support_alias_only=True)
    incumbent = Hypothesis(control=_INIT, source="coarse", features={"iou": 0.9})
    challenger = Hypothesis(control=_INIT, source="alias", features={"iou": 0.9})
    incumbent.score, challenger.score = 0.5, 0.8
    assert _alias_beats(challenger, incumbent, cfg)
    challenger.score = 0.4
    assert not _alias_beats(challenger, incumbent, cfg)


def test_alias_arbitration_respects_the_incumbent_margin():
    """An inherited naming must be beaten by the margin, not merely tied."""
    from flatsam2.mask_register import Hypothesis, RegisterConfig, _alias_beats

    cfg = RegisterConfig(support_power=1.0, support_alias_only=True)
    incumbent = Hypothesis(control=_INIT, source="last", features={"support": 0.50})
    challenger = Hypothesis(control=_INIT, source="alias", features={"support": 0.60})
    assert _alias_beats(challenger, incumbent, cfg)                 # no margin
    assert not _alias_beats(challenger, incumbent, cfg, margin=1.4)  # with margin


# --------------------------------------------------------------------------- #
# Occlusion: telling "part of the object is hidden" from "the mask is wrong"
# --------------------------------------------------------------------------- #
def _square_mask(h=200, w=200, y0=40, y1=160, x0=40, x1=160):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


_QUAD = np.array([[40.0, 160.0, 160.0, 40.0], [40.0, 40.0, 160.0, 160.0]])


_PREDICTED = _square_mask() > 0


def test_full_view_is_not_occluded():
    from flatsam2.occlusion import is_occluded_view, subset_evidence

    precision, recall = subset_evidence(_square_mask(), _PREDICTED)
    assert precision > 0.95 and recall > 0.95
    assert not is_occluded_view(_square_mask(), _PREDICTED)


def test_partial_view_is_occluded():
    """High precision, low recall — the signature the veto exists for."""
    from flatsam2.occlusion import is_occluded_view, subset_evidence

    half = _square_mask()
    half[:, 100:] = 0
    precision, recall = subset_evidence(half, _PREDICTED)
    assert precision > 0.95, "everything returned is still part of the object"
    assert recall < 0.6, "but only half of it was returned"
    assert is_occluded_view(half, _PREDICTED)


def test_a_different_object_is_not_read_as_occlusion():
    """A segmenter locked onto something else loses precision, not just recall."""
    from flatsam2.occlusion import is_occluded_view

    assert not is_occluded_view(_square_mask(y0=10, y1=60, x0=10, x1=60), _PREDICTED)


def test_drift_is_not_read_as_occlusion():
    """A drifted prediction loses both precision and recall."""
    from flatsam2.occlusion import is_occluded_view

    assert not is_occluded_view(_square_mask(y0=100, y1=199, x0=100, x1=199), _PREDICTED)


def test_empty_inputs_never_fire():
    from flatsam2.occlusion import is_occluded_view, subset_evidence

    empty = np.zeros((200, 200), np.uint8)
    assert all(np.isnan(v) for v in subset_evidence(empty, _PREDICTED))
    assert not is_occluded_view(empty, _PREDICTED)


def test_the_prediction_warps_the_template_shape_not_a_quad():
    """Quad-free: an L-shaped template must not be predicted as its bounding quad.

    Rasterizing a quad through the corners of a non-convex silhouette invents
    area the object never had, so a mask that is a *complete* view would score
    low recall and be mistaken for an occluded one.
    """
    from flatsam2.occlusion import is_occluded_view, warp_template_silhouette

    template = np.zeros((200, 200), np.uint8)
    template[40:160, 40:160] = 255
    template[100:160, 100:160] = 0          # bite out of the lower-right: an L
    shift = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    predicted = warp_template_silhouette(template, shift, (200, 200))
    observed = np.zeros((200, 200), np.uint8)
    observed[predicted] = 255               # a complete view of the L, shifted

    assert not is_occluded_view(observed, predicted), "a full view must not read as occluded"
    assert predicted.sum() < ((160 - 40) ** 2), "the L is smaller than its bounding quad"


def test_all_four_namings_are_enumerated_from_the_frozen_base():
    """Regression: rolling a mutable winner silently skips a naming.

    With the old loop, accepting shift 1 made the next iterations roll the *new*
    winner, so `roll(winner, 2)` was `roll(base, 3)` and `roll(winner, 3)` was
    the base again — shift 2 was never scored. Here shift 1 is a mild
    improvement and shift 2 is the best; a correct enumeration must return 2.
    """
    from flatsam2.mask_register import Hypothesis, RegisterConfig, _best_naming

    cfg = RegisterConfig(support_power=1.0, support_alias_only=True)
    base = Hypothesis(control=_INIT, source="coarse", features={"support": 0.10})
    # Support keyed by which cyclic shift of the base a control set represents.
    by_shift = {0: 0.10, 1: 0.20, 2: 0.90, 3: 0.15}

    def evaluate_fn(control, source, residual):
        shift = next(
            s for s in range(4) if np.allclose(control, np.roll(_INIT, s, axis=1))
        )
        return Hypothesis(control=control, source=source,
                          features={"support": by_shift[shift]})

    winner = _best_naming(base, evaluate_fn, 0.0, cfg)
    assert np.allclose(winner.control, np.roll(_INIT, 2, axis=1)), "shift 2 must win"


def test_enumeration_keeps_the_base_when_nothing_beats_it():
    from flatsam2.mask_register import Hypothesis, RegisterConfig, _best_naming

    cfg = RegisterConfig(support_power=1.0, support_alias_only=True)
    base = Hypothesis(control=_INIT, source="coarse", features={"support": 0.90})

    def evaluate_fn(control, source, residual):
        return Hypothesis(control=control, source=source, features={"support": 0.10})

    assert _best_naming(base, evaluate_fn, 0.0, cfg) is base


def test_enumeration_applies_the_incumbent_margin_only_to_the_incumbent():
    """A challenger clears the margin; a later one only has to beat that challenger."""
    from flatsam2.mask_register import Hypothesis, RegisterConfig, _best_naming

    cfg = RegisterConfig(support_power=1.0, support_alias_only=True)
    base = Hypothesis(control=_INIT, source="last", features={"support": 0.50})
    by_shift = {0: 0.50, 1: 0.80, 2: 0.85, 3: 0.10}

    def evaluate_fn(control, source, residual):
        shift = next(
            s for s in range(4) if np.allclose(control, np.roll(_INIT, s, axis=1))
        )
        return Hypothesis(control=control, source=source,
                          features={"support": by_shift[shift]})

    # margin 1.4: shift 1 (0.80) clears 0.50*1.4=0.70, then shift 2 (0.85) beats it.
    winner = _best_naming(base, evaluate_fn, 0.0, cfg, margin=1.4)
    assert np.allclose(winner.control, np.roll(_INIT, 2, axis=1))

    # With a margin nothing clears, the inherited naming stands.
    base_high = Hypothesis(control=_INIT, source="last", features={"support": 0.50})
    assert _best_naming(base_high, evaluate_fn, 0.0, cfg, margin=3.0) is base_high
