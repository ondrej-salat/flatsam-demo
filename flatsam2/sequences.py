"""Lightweight sequence-listing utilities for POT-210 and PlanarTrack datasets.

This module intentionally imports only the standard library so lightweight
tooling can resolve sequence lists without importing numpy, cv2, or torch.
"""

from pathlib import Path
from typing import List, Optional

from .path_config import paths
from .registry import DATASETS


def load_sequence_list(file_path: Path) -> List[str]:
    """Load a simple text file listing sequence names (one per line).

    Args:
        file_path: Path to the text file.

    Returns:
        List of sequence names.
    """
    if not file_path.exists():
        raise ValueError(f"Sequence list file not found: {file_path}")
    with file_path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def pot_sequences(base_dir: Path = paths.POT_BASE) -> List[str]:
    """Get list of sequence names in the POT-210 dataset.

    Args:
        base_dir: Base directory of the POT-210 dataset.

    Returns:
        Sorted list of sequence names (directory names).
    """
    images_dir = base_dir / "extracted" / "images"
    if not images_dir.exists():
        raise ValueError(f"POT-210 images directory not found: {images_dir}")
    return sorted([p.stem for p in images_dir.glob("*") if p.is_dir()])


def pt_sequences(base_dir: Path, split: Optional[str] = None) -> List[str]:
    """Get list of sequence names in the PlanarTrack dataset.

    Args:
        base_dir: Base directory of the PlanarTrack dataset.
        split: Optional split name (``'train'``, ``'test'``) or path to a
            custom split file.  If ``None``, returns all sequences.

    Returns:
        List of sequence names.
    """
    if split is None:
        split = 'test'
    if split is not None:
        split_files = {
            "train": "training_set.txt",
            "test": "test_seq.txt",
        }
        if split in split_files:
            split_file = base_dir / "training-test-splits" / split_files[split]
        else:
            split_file = Path(split)

        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")

        with split_file.open("r") as f:
            return [line.strip() for line in f if line.strip()]

    sequences_dir = base_dir / "sequences"
    if not sequences_dir.exists():
        raise ValueError(f"PlanarTrack sequences directory not found: {sequences_dir}")
    return sorted([p.stem for p in sequences_dir.glob("*") if p.is_dir()])


def get_sequences(
    dataset: str,
    base_dir: Path,
    sequences: Optional[List[str]] = None,
    subset_file: Optional[Path] = None,
    split: Optional[str] = None,
) -> List[str]:
    """Resolve sequence selection criteria to a concrete, ordered list of names.

    Priority (highest to lowest):

    1. ``subset_file`` — path to a text file (one sequence per line).
    2. ``sequences`` — explicit list passed by the caller.
    3. Discover all sequences from the dataset.

    ``sequences`` and ``subset_file`` can be combined: the result is their
    intersection (preserving the subset-file order).

    Args:
        dataset: Dataset key — any value from :data:`~flatsam2.registry.CLI_CHOICES`
            (e.g. ``'pot'``, ``'planartrack'``, ``'planartrack-initfix'``, ``'pti'``).
        base_dir: Dataset root directory.
        sequences: Explicit list of sequence names to process.
        subset_file: Path to a subset file (one sequence name per line).
        split: Dataset split for PlanarTrack (``'train'`` or ``'test'``).
            Ignored for datasets where ``uses_split`` is ``False``.

    Returns:
        Ordered list of sequence names satisfying all provided constraints.

    Raises:
        ValueError: If an unknown dataset is given, a required file is
            missing, or the resulting sequence list is empty.
    """
    # 1. Resolve subset source → ordered list from file (if any)
    file_pool: Optional[List[str]] = None
    if subset_file is not None:
        if not subset_file.exists():
            raise ValueError(f"Subset file not found: {subset_file}")
        file_pool = load_sequence_list(subset_file)

    # 2. Determine the candidate pool
    if file_pool is not None and sequences is not None:
        explicit_set = set(sequences)
        pool = [s for s in file_pool if s in explicit_set]
    elif file_pool is not None:
        pool = file_pool
    elif sequences is not None:
        pool = list(sequences)
    else:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset!r}. Expected one of: {sorted(DATASETS)!r}."
            )
        ds_info = DATASETS[dataset]
        if ds_info.gt_loader == "pot":
            pool = pot_sequences(base_dir)
        else:
            pool = pt_sequences(base_dir, split)

    # 3. Apply split filter when pool came from explicit sequences
    if dataset in DATASETS and DATASETS[dataset].uses_split and split is not None and sequences is not None:
        split_pool = set(pt_sequences(base_dir, split))
        pool = [s for s in pool if s in split_pool]

    if not pool:
        raise ValueError(
            f"No sequences remain after applying selection criteria "
            f"(dataset={dataset!r}, sequences={sequences}, "
            f"subset_file={subset_file}, split={split!r})."
        )

    return pool
