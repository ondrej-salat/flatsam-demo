"""Every frame's naming is checked against the correspondences, not one frame in ten.

`corner_fusion_aliasname` put the correspondence support in the right place — arbitrating
cyclic relabelings rather than entering the verification score — and the localization
collapse of `corner_fusion_matchncc` disappeared: br_p@15 89.75 against its base's 89.81,
where `matchncc` had fallen to 85.93. Naming improved too, gap 2.26 -> 1.65.

But only to 1.65, while `matchncc` reached 0.93. The reason is coverage, not evidence.
`alias_shortcut` runs **inside the stateless search**, which the incumbent proposal makes
unnecessary on about 88% of frames. On those frames the pose is accepted and its naming is
never re-examined at all, so the best naming evidence available is consulted on roughly one
frame in ten. `matchncc` reached a better gap by accident: putting the support into the
score depressed it below `accept_score`, which forced the search — and therefore the alias
comparison — to run 2.4x as often, at a cost of 3.88 pp of localization.

`alias_recheck` is the honest version of that. It scores the three relabelings of the
*final* winner on every frame, whichever path produced it, for three verifications and no
GN solve. It was previously measured as noise (0.14 pp) — but it was then scored by
`IoU * (1+NCC)/2`, and NCC is a weak naming signal; between relabelings the IoU is
identical by construction, so the comparison rested on the NCC difference alone. The
correspondence support separates the four namings by ~180x, which is what makes the same
check decisive rather than marginal.

Built on `corner_fusion_wholeframe` rather than the plain-product score: the two bases
differ by 0.43 pp of br_p@15 (90.24 against 89.81), and with naming no longer the
bottleneck that localization difference is what decides whether this beats
`corner_fusion_naming`'s 89.14. The a=b=1 variant of this same configuration is the
ablation that prices the exponents once naming is fixed — worth running, but second.

So: `IoU^2 * ((1+NCC)/2)^0.5` ranks poses, exactly as measured best; the correspondence
support names corners on every frame; and the incumbent margin still protects a naming
inherited from the previous frame, which carries temporal evidence a single frame's score
does not.
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
        "hough": _module.config.hough.model_copy(
            update={
                "register_support_power": 1.0,
                "register_support_tau_px": 30.0,
                "register_support_alias_only": True,
                "register_alias_recheck": True,   # ... on every frame, not just searched ones
            }
        ),
    }
)
