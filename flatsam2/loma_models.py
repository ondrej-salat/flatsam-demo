"""LoMa model factory helpers."""

from typing import Callable


try:
    from loma import LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, LoMaR

    LOMA_AVAILABLE = True
except ImportError:
    LOMA_AVAILABLE = False


if LOMA_AVAILABLE:
    LOMA_MODEL_FACTORIES: dict[str, Callable[[], object]] = {
        "LoMaB": lambda: LoMa(LoMaB()),
        "LoMaB128": lambda: LoMa(LoMaB128()),
        "LoMaL": lambda: LoMa(LoMaL()),
        "LoMaG": lambda: LoMa(LoMaG()),
        "LoMaR": lambda: LoMa(LoMaR()),
    }
else:
    LOMA_MODEL_FACTORIES = {}


def build_loma_model(model: str) -> object:
    """Build a LoMa model by name.

    Args:
        model: LoMa model variant name.

    Returns:
        Loaded LoMa model instance.

    Raises:
        ImportError: If LoMa is unavailable.
        ValueError: If model is unknown.
    """
    if not LOMA_AVAILABLE:
        raise ImportError("LoMa is not installed; cannot enable LoMa tracker path.")
    if model not in LOMA_MODEL_FACTORIES:
        raise ValueError(f"Unknown LoMa model {model!r}")
    return LOMA_MODEL_FACTORIES[model]()
