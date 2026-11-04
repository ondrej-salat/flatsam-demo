import numpy as np

from flatsam2.config import CorrectionCfg, LoMaPathConfig, TrackerConfig
from flatsam2.tracker import (
    _CANDIDATE_LOMA,
    _CANDIDATE_MASK_H,
    _CANDIDATE_SAMH,
    FlatSAMTracker,
    TrackerState,
    _should_correct,
)


def _tracker_with_loma_threshold(inlier_threshold: int = 35) -> FlatSAMTracker:
    tracker = object.__new__(FlatSAMTracker)
    tracker.cfg = TrackerConfig(
        loma=LoMaPathConfig(enabled=True, inlier_threshold=inlier_threshold)
    )
    tracker._loma_last_selected = "last_H"
    # _finalize_success records the correction mask on the result, so a stub
    # built with object.__new__ has to carry the field __init__ would have set.
    tracker._loma_last_corr_mask = None
    return tracker


def test_choose_loma_when_inliers_reach_threshold() -> None:
    tracker = _tracker_with_loma_threshold()
    loma_H = np.eye(3)
    mask_H = np.eye(3) * 2

    selected = tracker._choose_loma_or_mask(loma_H, 35, mask_H, 4)

    assert selected is loma_H


def test_choose_mask_when_loma_below_threshold_and_mask_present() -> None:
    tracker = _tracker_with_loma_threshold()
    loma_H = np.eye(3)
    mask_H = np.eye(3) * 2

    selected = tracker._choose_loma_or_mask(loma_H, 34, mask_H, 4)

    assert selected is mask_H


def test_choose_loma_when_mask_unavailable() -> None:
    tracker = _tracker_with_loma_threshold()
    loma_H = np.eye(3)

    selected = tracker._choose_loma_or_mask(loma_H, 34, None, 0)

    assert selected is loma_H


def test_choose_none_when_loma_and_mask_unavailable() -> None:
    tracker = _tracker_with_loma_threshold()

    selected = tracker._choose_loma_or_mask(None, 0, None, 0)

    assert selected is None


def test_decoupled_selects_loma_when_inliers_reach_threshold() -> None:
    tracker = _tracker_with_loma_threshold()
    loma_H = np.eye(3)
    samh_H = np.eye(3) * 2

    selected = tracker._decoupled_select(loma_H, 35, samh_H)

    assert selected is loma_H
    assert tracker._loma_last_selected == _CANDIDATE_LOMA


def test_decoupled_selects_samh_below_threshold() -> None:
    tracker = _tracker_with_loma_threshold()
    loma_H = np.eye(3)
    samh_H = np.eye(3) * 2

    selected = tracker._decoupled_select(loma_H, 34, samh_H)

    assert selected is samh_H
    assert tracker._loma_last_selected == _CANDIDATE_SAMH


def test_decoupled_selects_samh_when_loma_failed() -> None:
    tracker = _tracker_with_loma_threshold()
    samh_H = np.eye(3) * 2

    # LoMa failure (H None, 0 inliers) still yields the SAM-H estimate, which is
    # always present — mirrors the offline ensemble's other_pred fallback.
    selected = tracker._decoupled_select(None, 0, samh_H)

    assert selected is samh_H
    assert tracker._loma_last_selected == _CANDIDATE_SAMH


def _finalize_tracker() -> tuple[FlatSAMTracker, list]:
    """Tracker stub recording each `_run_symmetry` call's homography."""
    tracker = _tracker_with_loma_threshold()
    tracker.state = TrackerState()
    tracker.state.last_homography = np.eye(3)
    calls: list = []
    tracker._run_symmetry = lambda H, frame, n: calls.append(H)
    return tracker, calls


def test_finalize_runs_symmetry_when_requested() -> None:
    tracker, calls = _finalize_tracker()
    selected = np.eye(3) * 3

    tracker._finalize_success(
        selected, frame=np.zeros((4, 4, 3)), seg_mask=np.zeros((4, 4)),
        n_mask_corr=4, run_symmetry=True,
    )

    assert len(calls) == 1


def test_finalize_skips_symmetry_for_keypoint_homography() -> None:
    tracker, calls = _finalize_tracker()
    selected = np.eye(3) * 3

    tracker._finalize_success(
        selected, frame=np.zeros((4, 4, 3)), seg_mask=np.zeros((4, 4)),
        n_mask_corr=4, run_symmetry=False,
    )

    assert calls == []
    # Selected homography is still stored regardless of symmetry gating.
    assert np.array_equal(tracker.state.last_homography, selected)


def test_symmetry_gate_follows_selected_candidate() -> None:
    tracker = _tracker_with_loma_threshold()

    tracker._loma_last_selected = _CANDIDATE_MASK_H
    assert (tracker._loma_last_selected == _CANDIDATE_MASK_H) is True

    tracker._loma_last_selected = _CANDIDATE_LOMA
    assert (tracker._loma_last_selected == _CANDIDATE_MASK_H) is False


def test_should_correct_when_sam_superset_exceeds_margin() -> None:
    sam = np.zeros((20, 20), dtype=np.uint8)
    corr = np.zeros((20, 20), dtype=np.uint8)
    corr[5:15, 5:15] = 1
    sam[4:17, 4:17] = 1

    assert _should_correct(sam, corr, CorrectionCfg())


def test_should_not_correct_when_same_area() -> None:
    corr = np.zeros((20, 20), dtype=np.uint8)
    corr[5:15, 5:15] = 1


    assert not _should_correct(corr, corr, CorrectionCfg())


def test_should_not_correct_when_corr_leaks_outside_sam() -> None:
    sam = np.zeros((20, 20), dtype=np.uint8)
    corr = np.zeros((20, 20), dtype=np.uint8)
    corr[5:15, 5:15] = 1
    sam[5:15, 5:12] = 1


    assert not _should_correct(sam, corr, CorrectionCfg())


def test_should_not_correct_when_area_margin_too_small() -> None:
    sam = np.zeros((20, 20), dtype=np.uint8)
    corr = np.zeros((20, 20), dtype=np.uint8)
    corr[5:15, 5:15] = 1
    sam[5:16, 5:16] = 1


    assert not _should_correct(sam, corr, CorrectionCfg())


def test_should_not_correct_empty_corr() -> None:
    sam = np.ones((20, 20), dtype=np.uint8)
    corr = np.zeros((20, 20), dtype=np.uint8)

    assert not _should_correct(sam, corr, CorrectionCfg())
