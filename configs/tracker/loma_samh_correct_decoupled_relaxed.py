"""Decoupled LoMa + SAM-H tracker with a relaxed SAM3 self-correction gate.

Same wiring as ``loma_samh_correct_decoupled`` (independent SAM-H sub-tracker +
LoMa self-correction of the shared SAM3 memory). Only the correction *trigger*
differs.

Motivation: the strict gate (``superset_tolerance=0.02``) almost never fires
because the SAM3 mask is rarely a clean superset of the correspondence-based
correction mask — the two masks disagree at the boundary even when LoMa is
right. That made the decoupled-correct run worse, not better, since correction
effectively never triggered on the cases it was meant to fix.

This variant loosens the gate:
- ``superset_tolerance=0.1``  — allow up to 10% of the correction mask outside
  the SAM mask before refusing (was 2%).
- ``area_margin=0.15``       — require the SAM mask to be only 15% larger than
  the correction mask (was 25%).

``inlier_threshold`` stays at 70: confidence in the LoMa estimate is still
gated, only the mask-agreement criterion is relaxed.
"""

import importlib.util
from pathlib import Path

from flatsam2.config import CorrectionCfg


_base_path = Path(__file__).with_name("loma_samh.py")
_spec = importlib.util.spec_from_file_location("loma_samh_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "loma": _module.config.loma.model_copy(
            update={"decouple_hough_state": True}
        ),
        "correction": CorrectionCfg(
            enabled=True,
            inlier_threshold=70,
            trigger="area_margin",
            area_margin=0.15,
            superset_tolerance=0.1,
        ),
    }
)
