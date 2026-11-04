"""The candidate "final" quad-free tracker: every measured improvement, together.

Built on ``corner_fusion_register_kp`` (per-corner covariance fusion of the LoMa
and SAM-H branches, quad-free propose/refine/verify registration, keypoint
homography offered as an extra pose proposal) plus the two search/cost changes
that survived the offline benchmark on the dev half:

* ``register_sdf_roi_margin=1.0`` — the full-resolution signed-distance field is
  built on the mask's bounding box grown by its longer side, not on the whole
  frame. That transform is the single largest fixed cost of the branch (47% of a
  fast-path frame at 1280x720) and is only ever sampled at warped template
  boundary points, which sit on the mask. Measured neutral to within 0.2 pp of
  P@15 in every regime on both datasets, at 40-46% less time in the healthy
  regime that carries almost all frames. The generous margin is deliberate: a
  tighter crop (0.25) is barely faster and did lose 0.4 pp on POT ``lost``.

* ``register_alias_shortcut=True`` + ``register_sweep_stop_score=0.70`` — after
  refining a swept proposal, the three cyclic relabelings of its control points
  are scored too. They share a silhouette with it and differ only in which
  corner is which, so they cost three verifications and no GN solve, where
  reaching them through the angle sweep costs a coarse solve each and is not
  even guaranteed (aliases tie on the overlap that decides which proposals get
  refined). That makes the early exit affordable: stop refining once something
  clears 0.70, having already compared all four namings of it. Offline this is
  +1.1 pp P@15 on PTi ``lost`` and ``cold`` at slightly *less* time; the two
  knobs belong together, since an early exit without the alias check is exactly
  how a half-turn-wrong pose gets locked in.

Deliberately *not* included: ``sdf_precise=False``. The 3x3 chamfer
approximation is another ~20% off the field and measured accuracy-neutral, but
it produced one hard failure (no usable homography) in 513 POT dev cases that
the exact transform did not. For a reference version meant to be checked
thoroughly, robustness wins over the last 20%; it stays an ablation.

Nothing here changes what the tracker *decides* on a healthy frame — the fast
path is unchanged — so a difference against ``corner_fusion_register_kp`` at
grid scale is either the search improvement on lost frames or noise.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_register_kp.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_register_kp", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={
                "register_sdf_roi_margin": 1.0,
                "register_alias_shortcut": True,
                "register_sweep_stop_score": 0.70,
            }
        ),
    }
)
