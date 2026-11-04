"""``corner_fusion_register_recheck`` with the two hand-tuned parts removed.

Both changes delete parameters rather than add them, and both were measured
offline before costing a GPU hour.

**One fusion decision per frame instead of four.** The deployed rule decides
fuse-or-arbitrate separately for each corner, so its output quad could take two
corners from the keypoint branch and two from the mask branch — a pose neither
branch ever proposed. There is almost nothing to win that way:
``experiments/corner_complementarity.py`` finds genuinely complementary frames
are 0.66% of PTi and a per-corner oracle beats a whole-branch oracle by 0.48 pp.
``fusion.decision_scope='whole_frame'`` takes the decision once — fuse every
corner, or take the whole pose from the branch with the smaller total corner
variance. Replayed offline against the stored branch estimates of
``corner_fusion_register_final`` (``experiments/fusion_rule_replay.py``, exact
for a rule-only change, pooled p@15):

===================  ==========  ==========
rule                 POT (1004)  PTi (133k)
===================  ==========  ==========
per-corner (before)       97.61       87.76
whole-frame               98.80       88.32
whole-branch oracle       99.00       89.14
===================  ==========  ==========

**The verification score is the plain product of its two terms.**
``score = IoU^a * ((1+NCC)/2)^b`` with a=2, b=0.5 was set by hand; a=b=1 is what
a first-principles reading gives (a log-linear pool of two evidence terms with
no reason to weight one above the other) and leaves nothing to ablate. Measured
by ``experiments/maskh_bench.py`` on the test split, mean P@15 over the four
initialization regimes: POT 90.88 -> 91.75, PTi 84.38 -> 84.40, with the worst
regime's p90 error falling 14.5 -> 11.6 px (POT) and 73.6 -> 54.7 px (PTi). The
only loss anywhere is PTi `good`, 0.2 pp.

Everything else — keypoint proposal, cropped SDF, alias shortcut with early
exit, alias re-check — is inherited unchanged from
``corner_fusion_register_recheck``.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_register_recheck.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_register_recheck", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={
                "register_overlap_power": 1.0,
                "register_appearance_power": 1.0,
            }
        ),
        "fusion": _module.config.fusion.model_copy(
            update={"decision_scope": "whole_frame"}
        ),
    }
)
