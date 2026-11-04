"""Decoupled LoMa + SAM-H tracker with a lower correction inlier threshold.

Same wiring and mask-agreement gate as ``loma_samh_correct_decoupled``
(independent SAM-H sub-tracker + LoMa self-correction; ``area_margin=0.25``,
``superset_tolerance=0.02``). Only ``inlier_threshold`` differs: 35 instead of
70, so correction can fire on lower-confidence LoMa estimates.
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
            inlier_threshold=35,
            trigger="area_margin",
            area_margin=0.25,
            superset_tolerance=0.02,
        ),
    }
)
