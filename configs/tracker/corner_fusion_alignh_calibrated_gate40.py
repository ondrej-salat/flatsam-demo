"""``corner_fusion_alignh_calibrated`` with a loose disagreement gate.

Counterpart to ``corner_fusion_alignh_calibrated_gate6``: probes a gate
looser than the pre-calibration default of 20.0. Both branches' calibration
check (docs/2026-08-07.md) showed heavy tails persisting well beyond any
uniform scale correction (99th-percentile chi2 stats in the 1e5-1e8 range) --
that's the gross-error / distractor population, not something a Gaussian
gate should try to average through. A looser gate widens the "fuse" region
into that tail; if it makes things worse, that confirms the gate should stay
near the nominal chi2 quantile rather than being loosened further.
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
        "fusion": _module.config.fusion.model_copy(update={"disagreement_gate": 40.0}),
    }
)
