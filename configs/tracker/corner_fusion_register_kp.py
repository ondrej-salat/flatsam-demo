"""``corner_fusion_register`` plus the keypoint branch as an extra pose proposal.

The registrar's remaining error mode is corner *labelling*: a silhouette is
invariant under its own symmetries, so mask evidence alone cannot say which
corner is which. Measured on the full PTi grid, ``corner_fusion_register``
still carries a 2.83 pp best-rotation gap on 11.3% of sequences — right place,
permuted labels. (For scale: mask-moment redetection without a texture check
carried 10.50 pp on 35.0% of sequences, which is what sank that config.)

The keypoint branch has exactly the information the mask lacks, because
correspondences are anchored to texture. This config hands ``loma_H`` to the
registrar as one more pose proposal (``hough.register_keypoint_proposal``); it
is refined and verified like any other hypothesis rather than trusted, so a bad
keypoint estimate is simply outscored.

Offline benchmark (dev half, P@15, good/drift/lost/cold):

| dataset | register | + keypoint proposal |
|---------|----------------------|----------------------|
| PTi     | 88.8 / 85.7 / 80.6 / 80.8 | 88.8 / 86.6 / 83.4 / 83.4 |
| POT     | 87.5 / 84.4 / 81.1 / 81.1 | 87.7 / 86.2 / 86.2 / 85.6 |

Untouched while healthy, +2.6 to +5.1 once the pose is lost, and the p90 corner
error roughly halves (PTi lost 128.9 -> 61.6 px) — the alias tail collapsing,
which is what the proposal was added to do. It is also *faster* in those
regimes (304 -> 222 ms), because a keypoint proposal that already verifies well
lets the moment search be skipped.

Cost: the SAM-H branch now depends on the keypoint branch, so the two are no
longer independent and ``experiments/fusion_rule_replay.py`` is no longer an
exact simulator for runs of this config. That is why the flag is opt-in and
``corner_fusion_register`` is kept as the independent-branch reference.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_register.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_register", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={"register_keypoint_proposal": True}
        ),
    }
)
