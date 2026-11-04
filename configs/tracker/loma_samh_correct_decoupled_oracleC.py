"""E3 oracle-correction cell: LoMa trigger + oracle content.

Same as ``loma_samh_correct_decoupled_excessdepth`` except the injected content
is the GT-induced mask (perfect shape), while the trigger stays the deployed
one (LoMa-induced reference mask + ``sam_excess_depth``, LoMa inlier gate on
the trigger leg). Frames where the trigger fires but GT is invalid abstain.

Isolates trigger quality: E1 good + E3 bad means the deployed trigger fires at
the wrong times (or not at all), even though perfect content would fix SAM3.
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
                "trigger_source": "loma",
                "content_source": "oracle",
            }
        ),
    }
)
