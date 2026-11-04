"""``corner_fusion_alignh_calibrated`` with a tight disagreement gate.

``disagreement_gate=20.0`` (the default in ``FusionCfg``) was chosen before
covariance calibration, as a conservative guess accounting for both branches'
known overconfidence. Now that ``corner_fusion_alignh_calibrated`` applies an
empirical scale correction (loma_cov_scale=32.0, samh_cov_scale=70.0, see
that config's docstring and docs/2026-08-07.md), the covariances are on a
roughly sane scale in aggregate, so this probes the nominal chi-square(2) 95%
quantile (5.99, rounded up to 6) as the gate instead of the pre-calibration
guess. Tighter gate = less averaging, more "trust the confident branch"
arbitration; probes whether the wide 20.0 gate was fusing corners that
should have been arbitrated instead.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_alignh_calibrated.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_alignh_calibrated", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "fusion": _module.config.fusion.model_copy(update={"disagreement_gate": 6.0}),
    }
)
