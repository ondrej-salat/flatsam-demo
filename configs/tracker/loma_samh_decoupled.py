"""Online LoMa + SAM-H tracker with a fully independent SAM-H branch.

Identical to ``loma_samh`` except the entire SAM-H branch (mask-H + symmetry
resolution) runs as a nested tracker that shares only the SAM3 mask stream. Its
homography/symmetry chain is byte-identical to a standalone ``hough_symmetry_only``
run; the LoMa inlier rule only picks per frame between the LoMa estimate and that
independent SAM-H estimate, never feeding back. Reproduces the offline ensemble
``LMr_allobj_infl10_X_SAM3H_thr35``.
"""

import importlib.util
from pathlib import Path


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
    }
)
