"""E5 oracle-correction cell: periodic trigger + oracle content.

Same decoupled tracker as ``loma_samh_correct_decoupled_excessdepth``, but the
correction fires on an absolute cadence — every 30th tracked frame — injecting
the GT-induced mask, with no mask-divergence comparison and no LoMa inlier
gate. Frames with invalid GT abstain (``corr_trigger=True, corrected=False``).

This is the "continuous coaching" upper bound, deliberately direction-free
(the mask triggers only ever fire on over-segmentation): if E5 clearly beats
E1, episodic divergence-triggered correction is the wrong paradigm and SAM3
needs continuous conditioning instead. Also probes correction persistence: the
per-frame ``corr_iou_gt`` telemetry between cadence points shows how fast SAM3
drifts back after each re-anchor.

For a cadence sweep, copy this file as ``..._periodic{K}_oracleC.py`` with a
different ``period``.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_correct_decoupled_excessdepth.py")
_spec = importlib.util.spec_from_file_location("loma_samh_correct_excessdepth_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "correction": _module.config.correction.model_copy(
            update={
                "trigger": "periodic",
                "period": 30,
                "content_source": "oracle",
            }
        ),
    }
)
