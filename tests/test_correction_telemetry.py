"""Constructed-data guards that SAM3 correction telemetry survives to the sidecar.

These exercise the real code path on tiny synthetic masks — no SAM3/flow/GPU —
so a regression that silently drops the ``corrected`` flag is caught here in
seconds instead of after a multi-hour dataset run.

Covered, end to end:

    _maybe_correct_sam3 returns True   (real correction gate)
      -> loma_samh.track debug.corrected = True / correction_mask set
        -> run._save_debug
          -> {seq}_debug.npz 'corrected' == True

The decoupled LoMa verbs we are *not* testing (LoMa estimation, decoupled
selection, the SAM-H sub-tracker, the segmenter) are stubbed; the correction
gate itself is the real method bound on a bare tracker instance.
"""

import types

import numpy as np

from flatsam2.config import CorrectionCfg, LoMaPathConfig, TrackerConfig
from flatsam2.tracker import FlatSAMTracker, TrackerState
from flatsam2.tracking import loma_samh
from flatsam2.tracking.common import CorrectionTelemetry, MaskHTelemetry, TrackerDebugInfo


class _RecordingSegmenter:
    """Stand-in SAM3 segmenter: records correction calls, loads no model."""

    def __init__(self) -> None:
        self.corrections: list = []

    def correct(self, mask: np.ndarray) -> None:
        self.corrections.append(np.asarray(mask).copy())


def _decoupled_tracker(correction_enabled: bool) -> FlatSAMTracker:
    """Bare tracker wired for the decoupled LoMa path with the real correction gate.

    The gate fires because the SAM mask (full frame) is strictly larger than the
    LoMa-implied correction mask (a small block) and fully contains it.
    """
    trk = object.__new__(FlatSAMTracker)
    trk.cfg = TrackerConfig(
        loma=LoMaPathConfig(enabled=True, inlier_threshold=1),
        correction=CorrectionCfg(
            enabled=correction_enabled,
            inlier_threshold=1,
            trigger="area_margin",
            area_margin=0.0,
            superset_tolerance=1.0,
        ),
    )
    trk.state = TrackerState()
    trk.state.last_homography = np.eye(3)
    trk.state.init_corners = None
    init_mask = np.zeros((40, 40), dtype=np.uint8)
    init_mask[5:15, 5:15] = 1  # small block -> corr_mask area = 100
    trk.state.init_mask = init_mask
    trk.segmenter = _RecordingSegmenter()

    # Make the decoupled branch active without real models.
    trk._loma = object()
    trk._loma_feats0 = object()
    trk._loma_last_inliers = 0
    trk._loma_last_selected = "samh_H"
    trk._loma_last_corrected = False
    trk._loma_last_corr_mask = None
    trk._oracle_H = None
    trk._corr_telemetry = CorrectionTelemetry()

    # Stub the verbs that are not under test.
    trk._estimate_loma_H = lambda frame, seg_mask: (np.eye(3), 100)

    def _decoupled_select(loma_H, n_inliers, samh_H):
        trk._loma_last_selected = "loma_H"
        return loma_H

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
    return np.ones((40, 40), dtype=np.uint8)  # SAM mask area 1600 >> 100 -> gate fires


def _frame() -> np.ndarray:
    return np.zeros((40, 40, 3), dtype=np.uint8)


def test_decoupled_correction_sets_debug_flag_and_mask() -> None:
    trk = _decoupled_tracker(correction_enabled=True)

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert len(trk.segmenter.corrections) == 1, "correction must have fired"
    assert result.debug.corrected is True
    assert result.correction_mask is not None
    assert result.correction_mask.shape == (40, 40)


def test_decoupled_no_correction_when_disabled() -> None:
    trk = _decoupled_tracker(correction_enabled=False)

    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    assert trk.segmenter.corrections == []
    assert result.debug.corrected is False
    assert result.correction_mask is None


def test_correction_survives_segmenter_crash() -> None:
    """If segmenter.correct mutates memory then raises, corrected is still recorded.

    This is the real-world failure mode: SAM3's correct() changes memory (so masks
    downstream diverge) and then throws; step_with_mask swallows the exception. The
    flag must survive via self-state, not be lost with the discarded result.
    """

    class _CrashingSegmenter(_RecordingSegmenter):
        def correct(self, mask: np.ndarray) -> None:
            super().correct(mask)  # memory mutation already happened
            raise RuntimeError("sam3 correct boom")

    trk = _decoupled_tracker(correction_enabled=True)
    trk.fail_hard = False
    trk.segmenter = _CrashingSegmenter()

    # Route through step_with_mask so the try/except fallback is exercised.
    result = trk.step_with_mask(_frame(), _full_seg_mask())

    assert len(trk.segmenter.corrections) == 1, "side-effect must have happened"
    assert result.debug is not None, "fallback must still carry telemetry"
    assert result.debug.corrected is True, "corrected flag must survive the crash"
    assert result.correction_mask is not None


def test_corrected_flag_roundtrips_through_npz(tmp_path) -> None:
    """run._save_debug must persist a True corrected flag (the suspected drop point)."""
    import run

    records = [TrackerDebugInfo(corrected=(i == 1)) for i in range(3)]
    run._save_debug(records, [1, 2, 3], tmp_path, "seq")

    data = np.load(tmp_path / "seq_debug.npz", allow_pickle=False)
    assert data["corrected"].dtype == bool
    assert data["corrected"].tolist() == [False, True, False]


def test_track_to_npz_end_to_end(tmp_path) -> None:
    """Full chain: real correction gate -> debug -> npz has corrected == True."""
    import run

    trk = _decoupled_tracker(correction_enabled=True)
    result = loma_samh.track(trk, _frame(), _full_seg_mask())

    run._save_debug([result.debug], [1], tmp_path, "seq")

    data = np.load(tmp_path / "seq_debug.npz", allow_pickle=False)
    assert bool(data["corrected"][0]) is True
