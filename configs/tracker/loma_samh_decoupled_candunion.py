"""Decoupled LoMa + SAM-H tracker that densifies from the SAM3 candidate union.

Identical to ``loma_samh_decoupled`` except LoMa keypoint sampling draws from the
union of every SAM3 candidate mask (``loma.densify_mask="candidate_union"``)
instead of only the single selected mask. The wider region yields more
correspondences; the selected mask still drives mask-H, correction, and output,
so the SAM-H branch and SAM3 stream are byte-identical to the decoupled run.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_decoupled.py")
_spec = importlib.util.spec_from_file_location(
    "loma_samh_decoupled_config", _base_path
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "loma": _module.config.loma.model_copy(
            update={"densify_mask": "candidate_union"}
        ),
    }
)
