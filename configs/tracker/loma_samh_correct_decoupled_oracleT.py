"""E2 oracle-correction cell: oracle trigger + LoMa content.

Same as ``loma_samh_correct_decoupled_excessdepth`` except the trigger compares
the SAM3 mask against the GT-induced mask (perfect timing), while the injected
content stays the LoMa-induced mask (the deployed content). The LoMa inlier
gate applies to the content leg only: a fired trigger with unconfident LoMa
abstains (``corr_trigger=True, corrected=False`` in the debug sidecar).

Isolates content quality: E1 good + E2 bad means ``warp(init_mask, loma_H)`` is
not good enough as correction content even when the timing is perfect.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("loma_samh_correct_decoupled_excessdepth.py")
_spec = importlib.util.spec_from_file_location("loma_samh_correct_excessdepth_config", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "correction": _module.config.correction.model_copy(
            update={
                "trigger_source": "oracle",
                "content_source": "loma",
            }
        ),
    }
)
