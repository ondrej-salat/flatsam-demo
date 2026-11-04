"""E4 oracle-grid cell: rerun of the deployed excess-depth correction.

Byte-identical settings to ``loma_samh_correct_decoupled_excessdepth`` under a
fresh output directory. Needed because the original run predates the correction
telemetry: its on-disk ``corrected`` flags are stale (all-False, written by an
old binary) and it lacks the ``corr_*`` observables. This rerun provides the
E4 cell of the oracle grid with trustworthy per-frame telemetry.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_correct_decoupled_excessdepth.py")
_spec = importlib.util.spec_from_file_location("loma_samh_correct_excessdepth_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config
