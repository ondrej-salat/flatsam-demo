"""Quad-free tracker whose verifier scores shape and correspondences, not shape and NCC.

The mask registrar answers two different questions and they need different evidence:

* **which pose** — candidates from different sweep angles have different silhouettes, so
  the raster IoU can rank them;
* **which naming** — the four cyclic relabelings of one pose share a silhouette
  *exactly*, so IoU ties between them by construction. Not a weak signal: no signal.

Until now a single NCC term did both jobs, which is why deleting it
(``corner_fusion_register_noapp``) barely moves localization and destroys naming: on PTi
br-P@15 falls only 89.49 -> 88.64 while the naming gap explodes 1.73 -> 6.98.

This config replaces that term with the evidence the keypoint branch has already
computed. For each hypothesis, ``U`` is how many of the matcher's **raw, pre-RANSAC**
correspondences it explains under a truncated quadratic (``flatsam2.match_naming``), and
the verifier becomes::

    score = IoU * (U / N)

Two normalized evidence terms, a plain product, nothing to tune but ``tau``. Applied to
every hypothesis *and* to the three cyclic relabelings ``alias_shortcut`` already
enumerates, so **naming stops being a separate step**: across the aliases IoU is
identical, so the correspondence term alone decides which one wins. That deletes the NCC
term, both score exponents, the fusion-time naming step and its margin at once.

Pre-RANSAC matches on purpose: inliers were selected under a homography that already
carries one of the four namings, so scoring namings against them is circular.

The truncated quadratic is the reason a mismatch cannot swing the comparison — beyond
``tau`` it contributes nothing to *any* hypothesis and therefore cancels exactly when
hypotheses are compared, which an unbounded L1 or Huber cost would not do. Mismatch rates
are highest on blurred, low-texture frames, i.e. precisely where naming is hardest.

**What the offline evidence says to expect.** Replaying naming against the finished
shape-only grid (``fusion_rule_replay`` on ``corner_fusion_register_noapp``, using
distance-to-keypoint-quad as a stand-in for correspondence support):

==============================  ==========  ============  =====
rule                            PTi p@15    PTi br-p@15   gap
==============================  ==========  ============  =====
shape-only, whole-frame              82.92         90.83   7.91
shape-only + naming                  88.73         90.88   2.15
a=b=1 with NCC + naming              88.56         89.82   1.27
a=2 b=0.5 with NCC + naming          89.14         90.26   1.12
==============================  ==========  ============  =====

and on POT 98.41 against 98.80 for the NCC configs. So the architecture is roughly a
wash on accuracy — better localization than any NCC variant on PTi (br 90.88), a slightly
worse naming gap, ~0.4 pp behind on POT — bought with a much smaller mechanism. Those
numbers are a *lower bound* on this config: the stand-in needs the keypoint branch to
have produced a pose (711 of 2567 mis-named frames have none, while matches may still
exist), and it acts after the branch rather than inside the verifier, so it cannot fix an
alias before the full-resolution solve or before the accept/reject decision.

Everything else is inherited from ``corner_fusion_simple_norecheck``: whole-frame fusion
scope, cropped SDF, alias shortcut with early exit, no alias re-check.
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
                "register_appearance_power": 0.0,   # NCC out
                "register_support_power": 1.0,      # correspondences in
                "register_support_tau_px": 30.0,
            }
        ),
    }
)
