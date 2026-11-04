"""Decoupled LoMa + SAM-H with the P3' mask aligner replacing the Hough quad.

Identical to ``loma_samh_decoupled_rerun`` except the SAM-H branch (both the
independent sub-tracker and any inline mask-H) estimates the mask homography by
direct damped-GN alignment of the init mask onto the current SAM3 mask
(``hough.method='align'``), initialized from the last-frame homography.
Hough remains as the rejection fallback. Basin study (2026-07-11, tau=15):
92.7% convergence on clean masks, median landing error 3.2 px — targets the
4,784 P@15 misses where the mask is fine but the Hough quad is off.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_decoupled_rerun.py")
_spec = importlib.util.spec_from_file_location("loma_samh_decoupled_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={"method": "align"}
        ),
    }
)
