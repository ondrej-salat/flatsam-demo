"""``corner_fusion_register_final`` plus the corner-alias re-check on the winner.

``register_alias_shortcut`` compares the three cyclic relabelings of each
*coarsely* refined proposal. The full-resolution solve then runs on the survivor
and moves the pose, and nothing looks at its naming again — so a mislabeled pose
can be returned while the verifier's own score prefers one of its aliases.

Observed while generating the figures for ``docs/tracker-cards.html`` (PTi
Seq_00006 frame 30, incumbent displaced 250 px to force the search): the pose
``register_mask`` returned scored 0.770 and sat 342 px from the ground truth,
while its ``+1`` cyclic relabeling scored 0.859 at 20.9 px. One frame is a lead,
not a rate, which is why this is a separate config to be measured on the grids
rather than a silent default change.

Cost: three raster verifications on the final pose, no extra GN solve. Nothing
changes on frames whose winning naming was already the best of its four.
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
        "hough": _module.config.hough.model_copy(
            update={"register_alias_recheck": True}
        ),
    }
)
