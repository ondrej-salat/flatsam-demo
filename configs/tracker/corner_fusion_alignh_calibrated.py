"""``corner_fusion_alignh`` with an empirical covariance-scale correction.

Calibration check (20-sequence PTi probe, `configs/subsets/pti_fusion_probe20.txt`,
2026-08-10, see docs/2026-08-07.md): both branches' raw fit covariances are far
too tight (squared-Mahalanobis error vs GT should be chi-square(2), median
1.386; measured median 44.0 for LoMa, 96.3 for SAM-H). SAM-H is the *more*
overconfident of the two — its raw covariance understates its true error more
than LoMa's does — which biases the uncalibrated trace-based tie-break toward
SAM-H regardless of which branch is actually right, and was the identified
cause of `corner_fusion_alignh` regressing some sequences that the SAM-H-only
`alignh_pure` config had already gotten right. This config applies the
median-matching scale correction (rounded) as a first pass; it is a
sample-specific point estimate from n=20 sequences, not a full calibration
campaign, and should be revisited if it doesn't hold up at larger scale.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_alignh.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_alignh", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "fusion": _module.config.fusion.model_copy(
            update={"loma_cov_scale": 32.0, "samh_cov_scale": 70.0}
        ),
    }
)
