"""``corner_fusion_alignh_calibrated`` plus mask-moment redetection in the aligner.

Measured on the full PTi grid (2026-08-11, docs/2026-08-07.md): the align-based
SAM-H branch scores 41.14 p@15 on its own where the Hough-quad branch scores
82.90 — the whole quad-free deficit lives in that branch, not in the fusion rule
(an *oracle* SAM-H acceptance test only reaches 82.77 fused). The reason is
recovery, not accuracy: on the 61.7k GT frames where the aligner rejects its own
fit and coasts, it is wrong on 100%, while the stateless quad branch is right on
76.3% of those same frames (median 4.1 px). The evidence is in the mask; the
aligner just has no way back once its last-frame initialization has drifted off
the object.

This config turns on ``hough.align_redetect_moments``: when the last-H solve
lands above ``align_residual_ok_px``, re-solve from the poses implied by the
current mask's own image moments (centroid, principal axis, area) and keep the
best fit. Moments are defined for any shape, so this restores redetection
without reintroducing the quadrilateral assumption. Extra GN solves happen only
on frames that would otherwise have been rejected.
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
        "hough": _module.config.hough.model_copy(update={"align_redetect_moments": True}),
    }
)
