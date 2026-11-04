"""Decoupled LoMa + SAM-H with *no* quadrilateral machinery anywhere in mask-H.

Stricter than ``loma_samh_decoupled_alignh``, which kept the Hough quad as a
rejection fallback and left DINO symmetry resolution enabled — so its numbers do
not isolate mask alignment. This config removes both:

- ``align_fallback_hough=False`` — a rejected alignment yields *no* mask-H for
  that frame (the LoMa branch or the last homography carries it); the Hough
  lines -> cyclic-quad pipeline never runs.
- ``symmetry.enabled=False`` — DINO cyclic-shift disambiguation exists only
  because Hough re-extracts unordered corners each frame. Alignment initialized
  from the last homography cannot flip, so the subsystem has no role here.

Both settings propagate to the nested SAM-H sub-tracker, which is constructed
from ``cfg.hough`` / ``cfg.symmetry`` (``tracker.py`` sub-tracker setup).

The remaining quad dependency is the benchmark itself: POT/PlanarTrack ground
truth is 4 corners, and ``init_corners`` parameterizes the homography. The mask
*shape* is arbitrary — the template is the SAM3 init mask, not a quad.

Branch usage is recorded per frame in ``{seq}_debug.npz`` as ``maskh_source``;
check it before reading any metric off this run.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_decoupled_rerun.py")
_spec = importlib.util.spec_from_file_location("loma_samh_decoupled_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={
                "method": "align",
                "align_fallback_hough": False,
            }
        ),
        "symmetry": _module.config.symmetry.model_copy(update={"enabled": False}),
    }
)
