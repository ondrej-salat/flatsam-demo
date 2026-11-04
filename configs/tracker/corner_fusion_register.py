"""Corner fusion with the quad-free propose/refine/verify SAM-H branch.

Replaces ``hough.method='align'`` (a single Gauss-Newton solve started at the
last homography) with ``'register'``: the same boundary objective, but the pose
is *searched* rather than assumed. Second-moment normalization enumerates the
affine poses the two silhouettes agree on, each survivor is refined, and the
winner is chosen by raster overlap plus a texture correlation. No quadrilateral
assumption enters anywhere — moments and silhouettes are defined for any shape.

Why, measured offline on ``experiments/maskh_bench.py`` (2778 cases, held-out
half, P@15 against GT), with the initialization as a controlled variable:

| init regime            | align+redetect | Hough quad | register |
|------------------------|----------------|------------|----------|
| good  (GT + 5 px)      | 86.5           | 87.9       | 86.2     |
| drift (GT + 30 px)     | 73.3           | 85.1       | 84.4     |
| lost  (another frame)  | 27.5           | 71.5       | 79.6     |
| cold  (template pose)  | 30.7           | 75.0       | 79.6     |

(PlanarTrackInitfix; on POT-210 the same table reads 86.8 lost / 86.8 cold
against the quad's 71.9 / 76.2.) The deployed aligner is only competitive from
a good initialization, which is exactly the condition that fails on the frames
where the branch is needed. The
search costs ~1.5 points while tracking is healthy and buys 8-15 over the quad
pipeline once the pose has been lost — and it keeps the quad-free property that
the whole thread exists to establish.

Everything else (per-corner covariance fusion, the calibrated cov scales) is
inherited unchanged from ``corner_fusion_alignh_calibrated``, so this is a
controlled swap of the SAM-H estimator alone.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_alignh_calibrated.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_alignh_calibrated", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(update={"method": "register"}),
    }
)
