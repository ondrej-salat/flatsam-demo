"""``corner_fusion_register_final`` plus the whole-frame fusion decision, nothing else.

The isolation grid. ``corner_fusion_simple`` bundled two changes on top of ``_final``
(whole-frame fusion scope, and the verification score reduced to IoU * (1+NCC)/2) and
produced a result that splits by dataset:

* **POT** +1.19 pp pooled P@15 (97.61 -> 98.80), exactly what
  ``experiments/fusion_rule_replay.py`` predicted;
* **PTi** flat P@15 (87.76 -> 87.70) but with *localization* improving --
  br_p@15 89.49 -> 89.80 -- and the whole gain lost to corner naming, the br-gap
  widening 1.73 -> 2.10.

POT has a zero br-gap, i.e. no corner-naming problem, which is why the same localization
gain lands undiluted there. So the two simplifications improve where the object is and
degrade which corner is which, and with both changed at once there is no way to say
which one does the damage.

This config changes **only** ``fusion.decision_scope``, leaving the deployed score
(overlap_power 2.0, appearance_power 0.5) and the alias re-check exactly as ``_final``
has them. It also re-tests the replay's PTi prediction of 88.32 pooled under the one
condition the replay is exact for -- a fusion-rule change with the branch estimates
untouched, which the score change violated.

Reading the outcome, on the br-gap:

* gap stays ~1.7 -> the *score* change degrades naming; keep whole-frame fusion, revisit
  a=b=1 (the texture term's relative weight rises 4x under it, which is the obvious
  suspect for mis-ranking cyclic relabelings).
* gap goes to ~2.1 -> the *fusion scope* degrades naming, most plausibly because handing
  the whole pose to the more confident branch imports that branch's corner naming
  wholesale, and the mask branch cannot name corners.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_register_final.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_register_final", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "fusion": _module.config.fusion.model_copy(
            update={"decision_scope": "whole_frame"}
        ),
    }
)
