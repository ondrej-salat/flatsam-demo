"""Decoupled LoMa + SAM-H tracker with the study-optimal over-seg correction.

Same structure as ``loma_samh_correct_decoupled`` (independent SAM-H branch,
LoMa self-correction rewriting the shared SAM3 memory), but the correction
trigger uses the ``sam_excess_depth`` signal from the
``experiments/sam_overseg_correction.py`` feasibility study instead of the raw
total-area ratio.

Trigger values are the grid-search optimum reported by that study (labels:
N=15px align-correct, T_label=10px, over-seg frac=0.10):

    thr=60  tol=0.10  margin=0.10  T_feat=10px   -> benefit=2481 harm=387

mapping to ``CorrectionCfg``:
    inlier_threshold=60, superset_tolerance=0.10,
    trigger="sam_excess_depth", excess_depth_px=10, excess_margin=0.10.
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
            inlier_threshold=60,
            trigger="sam_excess_depth",
            excess_depth_px=10,
            excess_margin=0.10,
            superset_tolerance=0.10,
        ),
    }
)
