"""Poses ranked by shape and texture; namings decided by correspondences alone.

The third attempt at the same idea, and the placement the measurements actually support.

* `corner_fusion_matchverify` deleted NCC and let `U/N` carry the texture term. Naming
  improved (gap 1.93 -> 1.72) but `br_p@15` fell 89.81 -> 88.49: at tau = 30 px the support
  saturates across nearby poses, so it cannot rank hypotheses.
* `corner_fusion_matchncc` kept NCC and multiplied `U/N` into the score. Naming improved
  further — **gap 0.93, the best measured anywhere in this thread** — but `br_p@15`
  collapsed to 85.93. The cause is not the evidence, it is arithmetic: every factor is
  <= 1, so a third one shifts the whole score distribution down, and the registrar's
  thresholds are *not* scale-free. Median verified score fell 0.766 -> 0.669, frames
  failing `accept_score` rose 39.8% -> 54.6%, the incumbent fast path fell 88.6% -> 77.4%
  and the stateless search ran 2.4x as often — re-detecting poses that were already right.

Both failures point the same way: the support is evidence about *naming*, not about *pose*,
and it must therefore act only where naming is decided. So here it arbitrates the cyclic
relabelings — inside `alias_shortcut`, and inside `alias_recheck` where it must still clear
the incumbent margin — while the score stays exactly `IoU * (1+NCC)/2` and every threshold
calibrated against it keeps its meaning.

That is also the factorization the mechanism claimed from the start: between the four
relabelings of a pose the raster IoU is identical *by construction*, so the score cannot
separate them and should not be asked to; the support separates them by ~180x. Where the
four supports are unavailable (fewer than 8 matches) the comparison falls back to the
score, which is the old behaviour on exactly the frames that have no naming evidence.

One field against `corner_fusion_simple_norecheck`'s sibling with support enabled; the
score, its exponents (a=b=1) and every threshold are untouched.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_simple_norecheck.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_simple_norecheck", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={
                "register_support_power": 1.0,        # compute the support
                "register_support_tau_px": 30.0,
                "register_support_alias_only": True,  # ... and use it only to name corners
            }
        ),
    }
)
