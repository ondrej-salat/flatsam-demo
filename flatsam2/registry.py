"""Dataset registry — single source of truth for all supported dataset variants.

This module is intentionally free of intra-package imports so that both
:mod:`flatsam2.sequences` (stdlib-only) and :mod:`flatsam2.datasets` can
import from it without creating circular dependencies.

Attributes:
    DATASETS: Mapping from every accepted CLI key (canonical + aliases) to
        the corresponding :class:`DatasetInfo`.
    CLI_CHOICES: Sorted list of all accepted CLI strings, suitable for use
        as ``argparse choices=``.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple


@dataclass(frozen=True)
class DatasetInfo:
    """Declarative descriptor for a dataset variant.

    Args:
        key: Canonical CLI key (e.g. ``'planartrack-initfix'``).
        aliases: Additional accepted CLI keys (e.g. ``('pti',)``).
        label: Output subdirectory name and human-readable label
            (e.g. ``'PlanarTrackInitfix'``).
        short_label: Abbreviated name used in table columns (e.g. ``'PTi'``).
        base_dir_attr: Attribute name on
            :class:`~flatsam2.path_config.PathConfig` that holds the dataset
            root directory (e.g. ``'PT_BASE'``).
        gt_loader: Which GT loading strategy to use:
            ``'pot'`` → :func:`~flatsam2.datasets.pot_load_gt`,
            ``'pt'``  → :func:`~flatsam2.datasets.pt_load_gt`,
            ``'pti'`` → :func:`~flatsam2.datasets.pti_load_gt`
            (requires an ``initfix_dir`` at call time),
            ``'corners_txt'`` → :func:`~flatsam2.datasets.corners_txt_load_gt`
            (POIC / TMT / UCSB, which share one annotation format and differ
            only in where the frames and the annotations live — see
            ``seq_subdir`` / ``gt_subdir``),
            ``'mpot'`` → :func:`~flatsam2.datasets.mpot_load_gt` (MPOT-3K,
            whose sequences hold several planar objects at once; one tracking
            task is one ``<sequence>__<instance>`` pair).
        uses_split: Whether this dataset supports train/test splits.
        seq_subdir: Directory holding the per-sequence frame folders, relative
            to the dataset root (``'corners_txt'`` only; ``''`` = the root).
        gt_subdir: Directory holding the ``<sequence>.txt`` annotation files,
            relative to the dataset root (``'corners_txt'`` only).
    """

    key: str
    aliases: Tuple[str, ...]
    label: str
    short_label: str
    base_dir_attr: str
    gt_loader: Literal["pot", "pt", "pti", "corners_txt", "mpot"]
    uses_split: bool
    seq_subdir: str = ""
    gt_subdir: str = ""


# ---------------------------------------------------------------------------
# Registry entries
# ---------------------------------------------------------------------------

_DATASET_DEFS: Tuple[DatasetInfo, ...] = (
    DatasetInfo(
        key="pot",
        aliases=(),
        label="POT-210",
        short_label="POT",
        base_dir_attr="POT_BASE",
        gt_loader="pot",
        uses_split=False,
    ),
    DatasetInfo(
        key="planartrack",
        aliases=(),
        label="PlanarTrack",
        short_label="PT",
        base_dir_attr="PT_BASE",
        gt_loader="pt",
        uses_split=True,
    ),
    DatasetInfo(
        key="planartrack-initfix",
        aliases=("pti",),
        label="PlanarTrackInitfix",
        short_label="PTi",
        base_dir_attr="PT_BASE",
        gt_loader="pti",
        uses_split=True,
    ),
    DatasetInfo(
        key="poic",
        aliases=(),
        label="POIC",
        short_label="POIC",
        base_dir_attr="POIC_BASE",
        gt_loader="corners_txt",
        uses_split=False,
        seq_subdir="sequences",
        gt_subdir="gt",
    ),
    DatasetInfo(
        key="tmt",
        aliases=(),
        label="TMT",
        short_label="TMT",
        base_dir_attr="TMT_BASE",
        gt_loader="corners_txt",
        uses_split=False,
        seq_subdir="data/sequences",
        gt_subdir="data/gt",
    ),
    DatasetInfo(
        key="ucsb",
        aliases=(),
        label="UCSB",
        short_label="UCSB",
        base_dir_attr="UCSB_BASE",
        gt_loader="corners_txt",
        uses_split=False,
        seq_subdir="",
        gt_subdir="",
    ),
    DatasetInfo(
        key="mpot",
        aliases=("mpot-3k",),
        label="MPOT-3K",
        short_label="MPOT",
        base_dir_attr="MPOT_BASE",
        gt_loader="mpot",
        uses_split=True,
    ),
)

# Map every accepted CLI string (canonical key + aliases) → DatasetInfo
DATASETS: Dict[str, DatasetInfo] = {}
for _ds in _DATASET_DEFS:
    DATASETS[_ds.key] = _ds
    for _alias in _ds.aliases:
        DATASETS[_alias] = _ds

# Flat sorted list of all accepted CLI strings, for use as argparse ``choices=``
CLI_CHOICES: List[str] = sorted(DATASETS.keys())
