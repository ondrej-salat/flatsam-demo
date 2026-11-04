"""``corner_fusion_naming`` plus the occlusion screen on the mask branch.

The quad-free candidate's one regression against the shipped tracker that is a
*known bug* rather than an open question. On POT it scores 97.58 against 97.66,
and the whole difference is the occlusion condition: 97.83 against a perfect
100.00, carried by **three sequences of thirty, one frame each** (V19_5, V01_5,
V30_5; POT annotates those sequences with 5-6 GT frames, so one wrong frame is
16-25 pp of a sequence). All three wrong frames arbitrate as ``gated_samh``,
and on 100% of them the mask branch is the *more confident* of the two
(docs/journal.md 2026-08-26).

That is the failure diagnosed on 2026-08-19: under occlusion SAM returns the
visible part of the object, the registrar fits the template to that sub-region,
and the sub-region has a clean straight edge — so the residuals are small, the
covariance is at the 0.25 px floor, and a confidently *wrong* pose beats a
keypoint branch that is honestly uncertain about the corners it cannot see
(V01_5 frame 105: keypoint 4.7 px, mask 132.4 px, fused takes the mask).

The error is model-form — the branch fitted the wrong boundary — not sampling
noise, so no covariance scale and no calibration constant reaches it. What is
observable is the *signature*: compare the mask against the silhouette the
tracker predicts for this frame and ask two questions instead of one
(``flatsam2/occlusion.py``),

    precision = |mask & predicted| / |mask|
    recall    = |mask & predicted| / |predicted|

High precision with low recall is "an almost perfect subset" — the object is
there and part of it is hidden. That separates occlusion from the two things it
could be confused with, since a segmenter that locked onto something else loses
*precision* and a drifted tracker loses both. On such frames the mask branch
abstains entirely (a hard switch, not a discount: a discount only makes a
confidently-wrong branch lose more slowly, and needs an exponent chosen by
hand).

The prediction warps the stored template *silhouette*, not a quad through four
corners — this tracker is quad-free and the two regions differ for any
non-rectangular object — from the **keypoint branch's current-frame pose**.
Predicting from the previous frame would turn fast motion into false occlusion;
predicting from the mask branch's own pose would be circular.

Everything else is ``corner_fusion_naming`` untouched. The run stores the mask
branch's estimate as before, plus a per-frame ``fusion_occluded`` flag, so
``experiments/fusion_rule_replay.py`` can still re-decide the screen offline.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_naming.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_naming", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "fusion": _module.config.fusion.model_copy(
            update={"occlusion_max_recall": 0.80, "occlusion_min_precision": 0.90},
        ),
    }
)
