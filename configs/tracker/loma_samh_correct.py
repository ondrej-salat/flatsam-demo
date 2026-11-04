"""Online LoMa + SAM-H tracker with SAM3 self-correction."""

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
        "correction": CorrectionCfg(
            enabled=True,
            inlier_threshold=70,
            trigger="area_margin",
            area_margin=0.25,
            superset_tolerance=0.02,
        ),
    }
)
