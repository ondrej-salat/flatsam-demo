"""Determinism controls for reproducible tracking runs.

Two tiers, split by runtime cost:

- :func:`set_reproducible_seeds` only seeds the stack's RNGs (Python ``random``,
  NumPy, OpenCV, PyTorch CPU + CUDA). All O(1), no kernel-selection penalty, so
  ``run.py`` calls it by default — RANSAC and any other CPU/RNG-driven step is
  reproducible across runs out of the box.
- :func:`set_global_determinism` adds the *costly* knobs on top: cuDNN
  deterministic mode (disables the autotuner), deterministic torch algorithms,
  and a fixed cuBLAS workspace. These can slow conv/GEMM-heavy nets, so they stay
  opt-in (``run.py --deterministic``), for comparing outputs across runs with
  minimal run-to-run variance.

Residual non-determinism: SAM3, ptlflow, and DINOv2 are GPU neural nets. Some CUDA
kernels lack deterministic implementations; with ``warn_only=True`` they keep their
non-deterministic path instead of raising. Verification therefore tolerates a small
noise floor (compare two same-seed runs of the *same* code) rather than requiring
bit-exact equality.
"""

import logging
import os

logger = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int = 0) -> None:
    """Seed every RNG in the stack — no runtime penalty.

    Seeds Python ``random``, NumPy, OpenCV, and PyTorch (CPU + CUDA). All of these
    are O(1) and do not affect kernel selection, so this is safe to call on every
    run. Does NOT touch cuDNN/cuBLAS/deterministic-algorithm knobs (see
    :func:`set_global_determinism` for those).

    Args:
        seed: Seed applied to all RNGs.
    """
    import random

    import numpy as np
    import cv2
    import torch

    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Reproducible RNG seeds set (seed=%d).", seed)


def set_global_determinism(seed: int = 0) -> None:
    """Seed every RNG in the stack and request deterministic kernels.

    Seeds all RNGs via :func:`set_reproducible_seeds`, then enables cuDNN
    deterministic mode and requests deterministic torch algorithms
    (``warn_only=True`` so ops without a deterministic kernel degrade gracefully
    instead of raising). The kernel knobs can carry a runtime penalty (autotuner
    off, slower deterministic kernels), so prefer :func:`set_reproducible_seeds`
    unless bit-level determinism is required.

    Args:
        seed: Seed applied to all RNGs.
    """
    import torch

    set_reproducible_seeds(seed)

    # cuBLAS workspace must be fixed for deterministic GEMMs under
    # torch.use_deterministic_algorithms; harmless when CUDA is unused. Hard-set
    # (not setdefault): a preset, differing value would silently break determinism.
    _required_cublas = ":4096:8"
    _existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if _existing is not None and _existing != _required_cublas:
        logger.warning(
            "Overriding CUBLAS_WORKSPACE_CONFIG=%r with %r for determinism.",
            _existing,
            _required_cublas,
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _required_cublas

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("Global determinism set (seed=%d, warn_only deterministic algos).", seed)
