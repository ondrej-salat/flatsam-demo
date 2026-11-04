"""Online LoMa + SAM-H tracker: independent SAM-H branch + SAM3 self-correction.

Combines ``loma_samh_decoupled`` and ``loma_samh_correct``:

- The SAM-H branch (mask-H + symmetry) runs as a fully independent nested
  tracker, exactly like ``loma_samh_decoupled`` — no homography/symmetry state
  is shared with the LoMa branch.
- LoMa self-correction is enabled, so a confident LoMa estimate can rewrite the
  shared SAM3 memory.

The *only* coupling between the branches is therefore the intended one: a LoMa
correction changes the shared SAM3 mask stream, which both branches consume on
subsequent frames. There is no SAM-H homography coupling.
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
            area_margin=0.25,
            superset_tolerance=0.02,
        ),
    }
)
