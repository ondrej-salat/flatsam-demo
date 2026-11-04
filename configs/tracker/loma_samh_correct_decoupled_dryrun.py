"""E0+ oracle-grid cell: dry-run correction telemetry on the baseline tracker.

Byte-identical *tracking* to ``loma_samh_decoupled_rerun`` (no SAM3 memory is
ever touched), but the correction machinery of
``loma_samh_correct_decoupled_excessdepth`` runs in ``dry_run`` mode every
frame: the deployed trigger is evaluated and the full ``corr_*`` telemetry
(spill/excess/area-ratio vs both the LoMa- and GT-induced masks, IoU vs GT,
LoMa-vs-GT error, would-fire flag) lands in ``{seq}_debug.npz``.

One run therefore yields the counterfactual trigger statistics on unperturbed
masks: offline ROC of any trigger rule against the GT-side observables, without
the feedback effects a live correction introduces.
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
            update={"dry_run": True}
        ),
    }
)
