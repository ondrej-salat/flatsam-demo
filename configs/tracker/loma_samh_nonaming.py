"""The quad-based tracker with corner naming switched off entirely.

The naming ablation needs three arms, and two already exist: DINOv2 appearance
voting (`loma_samh_decoupled_rerun`) and correspondence support
(`loma_samh_matchnaming`). This is the third — the tracker with no naming
mechanism at all, so the SAM-H branch keeps whatever cyclic labelling its own
search produced.

Without it the paper can only say "our naming beats their naming"; with it, it
can say what naming is *worth*, which is the claim the br-gap column exists to
support. Expect the headline P@15 to fall a little and the best-rotation gap
(`br_p@15 - p@15`) to open up markedly — the gap is the direct read on corner
labelling, and it is the number to quote for this arm.
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
        "symmetry": _module.config.symmetry.model_copy(update={"enabled": False}),
    }
)
