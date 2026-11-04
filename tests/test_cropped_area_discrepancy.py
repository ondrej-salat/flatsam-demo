"""The cropped signed-area-discrepancy util must be invisible: identical result
to the full-frame computation, and the same code drives both the offline study
and the online tracker trigger."""

import numpy as np
import pytest

from flatsam2.config import CorrectionCfg
from flatsam2.mask_metrics import (
    cropped_signed_area_discrepancy,
    signed_area_discrepancy,
    signed_distance,
)
from flatsam2.tracking.common import _should_correct


T_GRID = (0, 2, 5, 10, 15, 20, 30)


def _full_frame(reference: np.ndarray, other: np.ndarray, thresholds):
    """Reference computation: signed distance over the whole frame."""
    return signed_area_discrepancy(
        signed_distance(reference), other, int(reference.sum()), thresholds
    )


def _disk(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r


# --------------------------------------------------------------------------- #
# crop is invisible
# --------------------------------------------------------------------------- #
def _assert_equal(reference, other, thresholds=T_GRID):
    co, ci = cropped_signed_area_discrepancy(reference, other, thresholds)
    fo, fi = _full_frame(reference, other, thresholds)
    for t in thresholds:
        assert co[t] == pytest.approx(fo[t], abs=1e-9, nan_ok=True), f"outside T{t}"
        assert ci[t] == pytest.approx(fi[t], abs=1e-9, nan_ok=True), f"inside T{t}"


def test_rect_overseg_centered():
    ref = np.zeros((400, 500), bool)
    ref[150:250, 200:300] = True            # 100x100 reference
    other = np.zeros_like(ref)
    other[140:260, 190:360] = True          # larger, deep bulge to the right
    _assert_equal(ref, other)


def test_offset_far_from_origin():
    # Object in a far corner: cropping changes the index frame, must not matter.
    ref = _disk(600, 800, 520, 700, 40)
    other = _disk(600, 800, 520, 712, 55)
    _assert_equal(ref, other)


def test_other_subset_of_reference_under_seg():
    ref = np.zeros((300, 300), bool)
    ref[80:220, 80:220] = True
    other = np.zeros_like(ref)
    other[110:190, 110:190] = True          # SAM strictly inside -> deficit only
    _assert_equal(ref, other)


def test_disjoint_masks():
    ref = _disk(300, 300, 80, 80, 30)
    other = _disk(300, 300, 220, 220, 30)   # no overlap
    _assert_equal(ref, other)


def test_reference_touches_border():
    ref = np.zeros((200, 200), bool)
    ref[0:60, 0:60] = True                  # hugs the top-left corner
    other = np.zeros_like(ref)
    other[0:90, 0:90] = True
    _assert_equal(ref, other)


def test_empty_reference_is_nan():
    ref = np.zeros((100, 100), bool)
    other = _disk(100, 100, 50, 50, 20)
    co, ci = cropped_signed_area_discrepancy(ref, other, T_GRID)
    assert all(np.isnan(co[t]) and np.isnan(ci[t]) for t in T_GRID)


def test_random_masks_match():
    rng = np.random.default_rng(0)
    for _ in range(20):
        h, w = int(rng.integers(120, 400)), int(rng.integers(120, 400))
        ref = _disk(h, w, int(rng.integers(40, h - 40)),
                    int(rng.integers(40, w - 40)), int(rng.integers(10, 35)))
        other = _disk(h, w, int(rng.integers(40, h - 40)),
                      int(rng.integers(40, w - 40)), int(rng.integers(10, 45)))
        _assert_equal(ref, other)


# --------------------------------------------------------------------------- #
# tracker trigger uses the same machinery
# --------------------------------------------------------------------------- #
def _fires_reference(sam, corr, cfg: CorrectionCfg) -> bool:
    """Independent re-implementation of the sam_excess_depth gate, full-frame."""
    a_corr = int(corr.sum())
    if a_corr == 0:
        return False
    if int((corr & ~sam).sum()) / a_corr > cfg.superset_tolerance:
        return False
    dout, _ = _full_frame(corr, sam, [cfg.excess_depth_px])
    return bool(dout[cfg.excess_depth_px] >= cfg.excess_margin)


def test_should_correct_matches_full_frame_formula():
    cfg = CorrectionCfg(
        enabled=True, trigger="sam_excess_depth",
        excess_depth_px=10, excess_margin=0.10, superset_tolerance=0.10,
    )
    rng = np.random.default_rng(1)
    saw_fire = saw_abstain = False
    for _ in range(40):
        h, w = 480, 640
        corr = _disk(h, w, int(rng.integers(80, 400)),
                     int(rng.integers(80, 560)), int(rng.integers(20, 50)))
        # SAM = corr grown by a random margin + occasional spill outside.
        grow = int(rng.integers(0, 30))
        cy, cx = int(rng.integers(80, 400)), int(rng.integers(80, 560))
        sam = _disk(h, w, cy, cx, int(rng.integers(20, 50)) + grow) | corr
        got = _should_correct(sam, corr, cfg)
        want = _fires_reference(sam, corr, cfg)
        assert got == want
        saw_fire |= got
        saw_abstain |= not got
    assert saw_fire and saw_abstain  # both branches exercised


def test_should_correct_superset_guard_abstains():
    cfg = CorrectionCfg(
        enabled=True, trigger="sam_excess_depth",
        excess_depth_px=10, excess_margin=0.10, superset_tolerance=0.05,
    )
    corr = _disk(300, 300, 150, 150, 60)
    sam = _disk(300, 300, 150, 180, 40)     # corr spills well outside sam
    assert _should_correct(sam, corr, cfg) is False
