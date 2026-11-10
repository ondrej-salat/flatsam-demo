"""The quad-based baseline with DINO replaced by correspondence support.

``loma_samh_decoupled_rerun`` — the 35-inlier switch between the keypoint branch and an
independent SAM-H branch — resolves its cyclic corner ambiguity with an appearance vote:
warp the template into a canonical view under each of the four shifts, score them with
DINOv2 layer 11, and confirm a shift only when it clears an absolute score *and* a margin
over the runner-up, or when it wins ``run_length`` consecutive frames. Because that costs
a second network forward pass it is gated to fire only while recovering from lost.

That is seven parameters (``only_when_lost``, ``n_correspondences_enough``,
``resolution_hw``, ``dino_layer``, ``abs_thr``, ``fts_thr``, ``run_length``) and a second
model, for one 2-bit decision.

This config replaces the whole stage with the measurement the keypoint matcher already
affords: score each of the four shifts by how much of its **raw, pre-RANSAC**
correspondence set the shift explains, under a truncated quadratic, and take the best
(``flatsam2.match_naming``). No second network, no temporal run length, no gate — it is
cheap enough to run every frame, so ``only_when_lost`` and ``n_correspondences_enough``
stop applying too. Six parameters out, one (``tau``) in.

Deliberately unchanged: ``find_closest_H``, the zero-motion assumption that picks the
candidate matching whose warped corners land nearest last frame's prediction. That
selects geometry as well as labelling, not just the naming.

Pre-RANSAC matches on purpose: inliers are selected under a homography that already
carries one of the four namings, so scoring namings against them is circular.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_decoupled_rerun.py")
_spec = importlib.util.spec_from_file_location("loma_samh_decoupled_rerun", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "symmetry": _module.config.symmetry.model_copy(
            update={"method": "matches", "match_tau_px": 30.0}
        ),
    }
)
