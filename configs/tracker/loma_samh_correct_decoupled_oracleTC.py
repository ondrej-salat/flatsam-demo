"""E1 oracle-correction cell: oracle trigger + oracle content.

Same decoupled tracker and ``sam_excess_depth`` trigger machinery as
``loma_samh_correct_decoupled_excessdepth`` (the deployed E4 cell), but both
correction legs read the GT homography plumbed by ``run.py``: the trigger
compares the SAM3 mask against ``warp(init_mask, H_gt)`` and the injected
content is that same GT-induced mask. No LoMa inlier gate applies (no "loma"
leg). Frames with invalid GT never trigger.

This is the correction-mechanism upper bound: if perfectly-timed,
perfectly-shaped corrections do not improve the SAM3 stream (and the samh_H
branch), the ``segmenter.correct`` primitive itself is the bottleneck.
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
                "content_source": "oracle",
            }
        ),
    }
)
