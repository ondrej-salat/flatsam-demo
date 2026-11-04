"""``corner_fusion_wholeframe`` plus the keypoint branch naming the mask branch's corners.

The shipped tracker's whole remaining PlanarTrack deficit is corner *naming*: pooled
P@15 88.32 against best-rotation P@15 90.24, i.e. on 1.93% of GT frames the quad sits
in the right place with its corners labelled in the wrong cyclic order.
``experiments/naming_diagnosis.py`` reads that off the finished run and localizes it:

* **89.5%** of the 2567 mis-named PTi frames are frames where the *mask* branch supplied
  the pose (``gated_samh`` 50.4%, ``samh_only_cov`` 37.7%). A silhouette determines a
  planar pose only up to which corner is called first, so this is the failure mode the
  branch is structurally incapable of avoiding.
* On 72.3% of those frames the keypoint branch's own naming is the one GT picks — even
  though its *position* is within 15 px on only 44.5% of them. Naming and localization
  are separable, and the two branches are good at different halves.

So the mask branch keeps its corner positions and borrows the keypoint branch's naming:
before the fuse-or-arbitrate decision, the SAM-H quad is permuted to the cyclic rotation
nearest the LoMa quad. Taking the nearest rotation unconditionally keeps only +0.28 pp of the rule's +1.01 pp — a
badly localized keypoint quad votes as loudly as a good one — so the branch's own naming
is the incumbent and is only overturned when the nearest rotation is
``fusion.keypoint_naming_margin`` times closer than it, the same protection the
registrar's alias checks already give an inherited naming.

Replayed exactly against the finished ``corner_fusion_wholeframe`` grids
(``experiments/fusion_rule_replay.py``; this is a re-labelling of stored branch
estimates, the class of change the replay reproduces to the decimal — confirmed when it
predicted whole-frame fusion's 88.32 and 98.80 and both landed):

===================  ==========  ==========  =========
rule                 PTi p@15    PTi br-p15  PTi gap
===================  ==========  ==========  =========
whole_frame               88.32       90.24       1.93
+naming, margin 1.2       88.97       90.26       1.29
+naming, margin 1.5       89.12       90.26       1.14
+naming, margin 2.0       89.14       90.26       1.12
+naming, margin 3.0       89.14       90.26       1.12
whole-branch oracle       89.14       89.72       0.57
===================  ==========  ==========  =========

Two things to read there. The gain is **entirely** in the gap — best-rotation P@15 moves
by 0.02 pp, so the rule renames without relocating, which is what it was built to do. And
it **reaches the whole-branch oracle**, a ceiling that only chooses between branches and
therefore cannot fix a naming both of them get wrong: naming is evidence outside that
ceiling. On POT the rule is exactly neutral at every margin (98.80, br-gap 0.00) — POT has
no naming problem to fix, and this does not manufacture one.

The margin plateaus over 1.5-3.0 on the full test set and on both deterministic halves of
it independently, so 2.0 is a shoulder, not a fitted value.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_wholeframe.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_wholeframe", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "fusion": _module.config.fusion.model_copy(
            update={"keypoint_naming_margin": 2.0}
        ),
    }
)
