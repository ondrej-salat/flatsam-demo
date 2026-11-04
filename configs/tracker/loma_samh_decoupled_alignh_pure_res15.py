"""``loma_samh_decoupled_alignh_pure`` with ``align_residual_ok_px`` loosened to 15.

Gate-1 remedy probe (thread retired; see docs/dead-ends.md): the deployed 6px threshold
rejected ~half of all frames on real PTi/POT masks (49.3% / 43.5% ``none``,
see docs/2026-08-07.md), far more than the offline basin study predicted.
Single-sequence probe before committing to a full rerun.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_decoupled_alignh_pure.py")
_spec = importlib.util.spec_from_file_location("loma_samh_decoupled_alignh_pure", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(update={"align_residual_ok_px": 15.0}),
    }
)
