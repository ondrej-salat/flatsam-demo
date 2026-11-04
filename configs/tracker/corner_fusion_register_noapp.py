"""``corner_fusion_register`` with the verifier's texture evidence switched off.

The mechanism ablation for the paper's central claim: a silhouette is invariant
under its own symmetries, so shape evidence alone cannot say which corner is
which. Offline this costs 15.6 points of P@15 on a cold start (81.5 -> 65.9),
and 44.6% of the wrong hypotheses are pure cyclic corner relabelings.

This config runs the identical pipeline with ``register_appearance_power=0``, so
hypotheses are ranked on raster IoU alone. It exists to show that effect
end-to-end on the tracking benchmark rather than only on the offline test set,
and the number to read is not just mean p@15 but the **best-rotation gap**
(``mean_br_p@15 - mean_p@15``): the prediction is that it opens up toward the
10.50 pp / 35% of sequences that mask-moment redetection showed, against 2.83 pp
/ 11.3% for the full verifier.

Everything else is inherited unchanged, so it is a controlled one-term ablation.
"""

import importlib.util
from pathlib import Path


_base_path = Path(__file__).with_name("corner_fusion_register.py")
_spec = importlib.util.spec_from_file_location("corner_fusion_register", _base_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load base config from {_base_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


config = _module.config.model_copy(
    update={
        "hough": _module.config.hough.model_copy(
            update={"register_appearance_power": 0.0}
        ),
    }
)
