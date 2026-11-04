"""``corner_fusion_simple`` without the alias re-check.

Two purposes, one grid.

**It removes a knob that measured against itself.** ``register_alias_recheck`` passed
the offline bench (`lost` 79.3 -> 80.1, p90-when-lost 151 -> 129 px) and then came in at
**87.74 vs 87.87 PTi P@15** on the full grid, with the br-gap widening 1.71 -> 1.87 and
POT identical to the decimal. The disagreement is aggregation: ``maskh_bench`` weights
the four initialization regimes equally, while a real sequence takes the fast path on
88.5% of frames, so a `lost`-only gain is diluted about tenfold and any small harm in
`good` is not (``docs/state-of-the-tracker.md`` working rule 9).

**It removes a confound.** ``corner_fusion_simple`` changed three things at once against
``corner_fusion_register_final``: the re-check, the whole-frame fusion decision, and the
verification score. On POT that landed exactly where ``experiments/fusion_rule_replay.py``
predicted (98.80 pooled, predicted 98.80). On PTi it did not — 87.70 against a predicted
88.32 — and with three deltas in flight the shortfall cannot be attributed. Note also
that the replay is exact only for a *fusion-rule* change: the score change moves which
hypothesis the registrar returns, so the branch estimates it re-fuses are no longer the
ones that were replayed, and the two changes are not additive.

This config leaves only the two simplifications that were argued for on first principles
-- one fusion decision per frame, and the verification score as the plain product
IoU * (1+NCC)/2 -- on top of ``corner_fusion_register_final``.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_simple.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_simple", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={"register_alias_recheck": False}
        ),
    }
)
