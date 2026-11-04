"""Corner-level covariance fusion of LoMa + SAM-H, quad-free (``algorithm='corner_fusion'``).

Same substrate as ``loma_samh_decoupled_alignh_pure`` (independent SAM-H
sub-tracker, ``method='align'``, no Hough quad, no DINO symmetry — corner
covariance is only defined for the align method's boundary-fit residual, not
the Hough line-intersection one). The only change from that config: instead
of a hard 35-inlier switch between the two whole homographies
(``loma_samh``'s ``_decoupled_select``), fuse the two branches' four corner
estimates independently, weighted by each branch's own reported uncertainty.
See ``flatsam2/tracking/corner_fusion.py`` and
docs/correction_research.md sections 9-10.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_decoupled_alignh_pure.py")
_spec = importlib.util.spec_from_file_location("loma_samh_decoupled_alignh_pure", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "loma": _module.config.loma.model_copy(update={"fusion": True}),
        # model_copy() does not re-run the algorithm-derivation validator, so the
        # base config's already-resolved "loma_samh" would otherwise stick even
        # though loma.fusion now implies "corner_fusion" — set it explicitly.
        "algorithm": "corner_fusion",
    }
)
