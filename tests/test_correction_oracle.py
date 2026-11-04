"""Constructed-data guards for the oracle-correction grid (E1/E2/E3/E5).

Same stub pattern as ``test_correction_telemetry``: bare tracker, tiny synthetic
masks, no SAM3/flow/GPU. The correction gate (``_maybe_correct_sam3``), the
per-leg source resolution, the periodic trigger, dry-run, and the telemetry
plumbing through ``run._save_debug`` are the real code under test.

Geometry used throughout: 40x40 frame, init mask = block [5:15, 5:15] (area
100), full-frame SAM mask (area 1600), so the ``area_margin=0.0`` trigger fires
whenever a reference mask exists. ``_SHIFT_H`` translates by +10 px in x, so an
oracle-injected mask is distinguishable from a LoMa-injected one (identity).
"""

import types

import numpy as np

from flatsam2.config import CorrectionCfg, LoMaPathConfig, TrackerConfig
from flatsam2.tracker import FlatSAMTracker, TrackerState
from flatsam2.tracking import loma_samh
from flatsam2.tracking.common import CorrectionTelemetry, MaskHTelemetry, TrackerDebugInfo

_SHIFT_H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _bare_tracker(
    correction: CorrectionCfg,
    loma_result: tuple = (np.eye(3), 100),
) -> FlatSAMTracker:
    """Bare decoupled-path tracker with a configurable correction gate."""
    trk = object.__new__(FlatSAMTracker)
    trk.cfg = TrackerConfig(
        loma=LoMaPathConfig(enabled=True, inlier_threshold=1),
        correction=correction,
    )
    trk.state = TrackerState()
    trk.state.last_homography = np.eye(3)
    trk.state.init_corners = None
    init_mask = np.zeros((40, 40), dtype=np.uint8)
    init_mask[5:15, 5:15] = 1
    trk.state.init_mask = init_mask
    trk.segmenter = types.SimpleNamespace(corrections=[])
    trk.segmenter.correct = lambda mask: trk.segmenter.corrections.append(
        np.asarray(mask).copy()
    )
    trk.fail_hard = True

    trk._loma = object()
    trk._loma_feats0 = object()
    trk._loma_last_inliers = 0
    trk._loma_last_selected = "samh_H"
    trk._loma_last_corrected = False
    trk._loma_last_corr_mask = None
    trk._oracle_H = None
    trk._corr_telemetry = CorrectionTelemetry()

    trk._estimate_loma_H = lambda frame, seg_mask: loma_result

    def _decoupled_select(loma_H, n_inliers, samh_H):
        trk._loma_last_selected = "loma_H" if loma_H is not None else "samh_H"
        return loma_H if loma_H is not None else samh_H

    trk._decoupled_select = _decoupled_select
    trk._samh_tracker = types.SimpleNamespace(
        step_with_mask=lambda frame, seg_mask: types.SimpleNamespace(homography=np.eye(3)),
        # The decoupled path reads the sub-tracker's mask-H telemetry into its
        # debug record, and its symmetry settings to decide whether the keypoint
        # correspondences must be computed before it runs, so the stub exposes
        # both like the real sub-tracker does.
        _maskh_telemetry=MaskHTelemetry(),
        cfg=trk.cfg,
    )
    return trk


def _full_seg_mask() -> np.ndarray:
    return np.ones((40, 40), dtype=np.uint8)


def _frame() -> np.ndarray:
    return np.zeros((40, 40, 3), dtype=np.uint8)


def _fire_always_cfg(**overrides) -> CorrectionCfg:
    """area_margin trigger that fires for any non-empty reference mask."""
    base = dict(
        enabled=True,
        inlier_threshold=1,
        trigger="area_margin",
        area_margin=0.0,
        superset_tolerance=1.0,
    )
    base.update(overrides)
    return CorrectionCfg(**base)


def test_oracle_content_injected() -> None:
    """E3: content_source='oracle' injects the GT-warped mask, not the LoMa one."""
    trk = _bare_tracker(_fire_always_cfg(content_source="oracle"))
    trk._oracle_H = _SHIFT_H

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert len(trk.segmenter.corrections) == 1
    injected = trk.segmenter.corrections[0]
    assert injected[5:15, 15:25].all(), "must be the +10px-shifted (oracle) block"
    assert not injected[5:15, 5:14].any(), "must not be the identity (LoMa) block"
    assert result.debug.corrected is True
    assert result.debug.correction.trigger_fired is True


def test_oracle_content_abstains_without_gt() -> None:
    """E3 on an invalid-GT frame: trigger fires, correction abstains, both recorded."""
    trk = _bare_tracker(_fire_always_cfg(content_source="oracle"))
    trk._oracle_H = None

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert trk.segmenter.corrections == []
    assert result.debug.corrected is False
    assert result.debug.correction.trigger_fired is True
    assert result.correction_mask is None


def test_oracle_trigger_ignores_inlier_gate() -> None:
    """E1: pure-oracle correction fires even when LoMa failed entirely."""
    trk = _bare_tracker(
        _fire_always_cfg(
            trigger_source="oracle",
            content_source="oracle",
            inlier_threshold=10_000,
        ),
        loma_result=(None, 0),
    )
    trk._oracle_H = _SHIFT_H

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert len(trk.segmenter.corrections) == 1
    assert result.debug.corrected is True


def test_oracle_trigger_loma_content_needs_inliers() -> None:
    """E2: oracle trigger fires but unconfident LoMa content abstains."""
    trk = _bare_tracker(
        _fire_always_cfg(
            trigger_source="oracle",
            content_source="loma",
            inlier_threshold=70,
        ),
        loma_result=(np.eye(3), 5),
    )
    trk._oracle_H = _SHIFT_H

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert trk.segmenter.corrections == []
    assert result.debug.corrected is False
    assert result.debug.correction.trigger_fired is True


def test_periodic_fires_on_cadence() -> None:
    """E5: periodic trigger fires exactly on frame_idx multiples of period."""
    trk = _bare_tracker(
        CorrectionCfg(
            enabled=True,
            trigger="periodic",
            period=3,
            content_source="oracle",
        )
    )
    trk._oracle_H = _SHIFT_H

    corrected_frames = []
    for _ in range(6):
        result = trk.step_with_mask(_frame(), _full_seg_mask())
        if result.debug.corrected:
            corrected_frames.append(trk.state.frame_idx)

    assert corrected_frames == [3, 6]
    assert len(trk.segmenter.corrections) == 2


def test_frame_counter_increments() -> None:
    trk = _bare_tracker(CorrectionCfg(enabled=False))

    trk.step_with_mask(_frame(), _full_seg_mask())
    trk.step_with_mask(_frame(), _full_seg_mask())

    assert trk.state.frame_idx == 2


def test_telemetry_always_recorded() -> None:
    """Observables land in debug even when no gate passes and nothing fires."""
    trk = _bare_tracker(
        _fire_always_cfg(area_margin=100.0)  # sam(1600) < 101 * ref(100): no fire
    )
    trk._oracle_H = np.eye(3)

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    tel = result.debug.correction
    assert result.debug.corrected is False
    assert tel.trigger_fired is False
    assert tel.spill_loma == 0.0, "identity LoMa block lies fully inside the SAM mask"
    assert tel.area_ratio_loma == 16.0
    assert abs(tel.iou_gt - 100.0 / 1600.0) < 1e-6


def test_dry_run_never_corrects() -> None:
    trk = _bare_tracker(_fire_always_cfg(dry_run=True))

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert trk.segmenter.corrections == []
    assert result.debug.corrected is False
    assert result.debug.correction.trigger_fired is True


def test_new_fields_roundtrip_npz(tmp_path) -> None:
    """corr_* arrays survive run._save_debug with NaN/False defaults and None rows."""
    import run

    records = [
        TrackerDebugInfo(
            correction=CorrectionTelemetry(spill_loma=0.3, iou_gt=0.5, trigger_fired=True)
        ),
        TrackerDebugInfo(),
        None,
    ]
    run._save_debug(records, [1, 2, 3], tmp_path, "seq")

    data = np.load(tmp_path / "seq_debug.npz", allow_pickle=False)
    assert data["corr_spill_loma"].dtype == np.float32
    assert abs(data["corr_spill_loma"][0] - 0.3) < 1e-6
    assert np.isnan(data["corr_spill_loma"][1]) and np.isnan(data["corr_spill_loma"][2])
    assert abs(data["corr_iou_gt"][0] - 0.5) < 1e-6
    assert data["corr_trigger"].dtype == bool
    assert data["corr_trigger"].tolist() == [True, False, False]


def test_gt_homographies() -> None:
    """Validity gating, degenerate corners, and identity on frame 0."""
    from run import _gt_homographies

    square = np.array([[0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 10.0, 10.0]])
    shifted = square + np.array([[5.0], [0.0]])
    collinear = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    gt = np.stack([square, shifted, square, collinear])
    valid = np.array([True, True, False, True])

    hs = _gt_homographies(gt, valid)

    assert np.allclose(hs[0], np.eye(3), atol=1e-6)
    warped = hs[1] @ np.array([0.0, 0.0, 1.0])
    assert abs(warped[0] / warped[2] - 5.0) < 1e-6
    assert hs[2] is None, "invalid GT frame must yield None"
    assert hs[3] is None, "degenerate (collinear) corners must yield None"

    all_none = _gt_homographies(gt, np.array([False, True, True, True]))
    assert all(h is None for h in all_none), "invalid init frame poisons every H"
