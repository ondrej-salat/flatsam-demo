"""Shape ranks poses, texture ranks poses, correspondences name corners.

`corner_fusion_matchverify` tested whether the correspondence-support term could replace
NCC outright. It cannot, and the grids say exactly why:

============================  ========  =========  =====  =========
config (PTi 300)              pooled    br-pool    gap    POT pool
============================  ========  =========  =====  =========
whole-frame, a=b=1, NCC          87.55      89.81   2.26      98.80
+ distance-naming (a=2,b=.5)     89.14      90.26   1.12      (n/a)
matchverify (no NCC)             86.77      88.49   1.72      98.21
============================  ========  =========  =====  =========

Naming was never the problem — `matchverify`'s gap (1.72) beats the 1.93 of no naming at
all. Deleting NCC was: `br_p@15` fell 89.81 -> 88.49 on PTi and 98.80 -> 98.41 on POT.

The reason is a difference in *sensitivity curves*. At tau = 30 px a pose 10-20 px off
still explains nearly all of its matches, so `U/N` is flat across nearby poses; it
separates the four cyclic namings by ~180x, because aliases sit hundreds of pixels apart,
and separates competing poses hardly at all. NCC has no such saturation — texture
correlation degrades continuously with pose error, which is precisely what ranks
hypotheses. One term cannot do both jobs.

Coverage says the same thing a second way. On the full `matchverify` grid, 92.5% of
localized frames carry correspondence evidence and are mis-named 0.59% of the time, while
the 7.5% without evidence are mis-named 21.79% and carry **74.9% of all mis-naming**. On a
shape-only verifier those frames tie *exactly* across the four aliases, so naming there is
chosen at random. POT makes it vivid: its gap was 0.00 with NCC and 0.20 without, meaning
NCC had been naming corners there invisibly all along.

So keep both, each doing what it measurably does::

    score = IoU * (1+NCC)/2 * U/N

IoU and NCC rank poses; `U/N` names corners, and dominates the alias comparison where IoU
is tied by construction and NCC is merely weak. No exponents anywhere — a plain product of
three normalized evidence terms.

Expected to beat `corner_fusion_naming`'s 89.14, since it adds a naming signal measured at
99.4% accuracy (where evidence exists) on top of a configuration that already localizes at
br 89.81, and unlike the distance-to-keypoint-quad namer it needs only *matches*, not a
successful keypoint pose — 711 of 2567 mis-named frames in the earlier diagnosis had no
keypoint pose at all.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_simple_norecheck.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_simple_norecheck", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={
                "register_support_power": 1.0,     # correspondences name corners
                "register_support_tau_px": 30.0,
                # register_appearance_power stays at 1.0: NCC ranks poses.
            }
        ),
    }
)
