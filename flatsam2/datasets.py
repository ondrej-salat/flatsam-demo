"""Dataset loading utilities for POT-210 and PlanarTrack datasets.

This module provides functions to load sequences, ground truth annotations,
and image paths for planar object tracking datasets.

The canonical dataset registry (:data:`DATASETS`, :class:`DatasetInfo`,
:data:`CLI_CHOICES`) lives in :mod:`flatsam2.registry` and is re-exported
here for convenience.  All CLI scripts should import those names from here
or directly from :mod:`flatsam2.registry`.

Lightweight sequence-listing helpers (``get_sequences``, ``pot_sequences``,
``pt_sequences``, ``load_sequence_list``) live in :mod:`flatsam2.sequences`
and are re-exported here.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange

from .registry import CLI_CHOICES, DATASETS, DatasetInfo
from .sequences import get_sequences, load_sequence_list, pot_sequences, pt_sequences

__all__ = [
    # Registry (re-exported from flatsam2.registry)
    "DatasetInfo",
    "DATASETS",
    "CLI_CHOICES",
    # Sequence helpers (re-exported from flatsam2.sequences)
    "get_sequences",
    "load_sequence_list",
    "pot_sequences",
    "pt_sequences",
    # GT loaders
    "pot_load_gt",
    "pt_load_gt",
    "pti_load_gt",
    # Image paths
    "pot_image_paths",
    "pt_image_paths",
    # Per-dataset dispatch
    "make_gt_loader",
    "make_image_paths",
    # Geometry
    "corners_to_mask",
    "warp_corners",
]


def pot_load_gt(base_dir: Path, seq_name: str, woft_compat: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth annotations for a POT-210 sequence.

    Args:
        base_dir: Base directory of POT-210 dataset.
        seq_name: Sequence name.

    Returns:
        annot: Corner annotations with shape (num_frames, 2, 4) - (x,y coords, 4 corners).
        valid: Boolean array (num_frames,) indicating valid frames.
    """
    if woft_compat:
        annot_dir = base_dir / "annotation" / "WOFT_reannotation_coarse_onlychanged"
    else:
        annot_dir = base_dir / "annotation" / "annotation"

    annot_file = annot_dir / f"{seq_name}_gt_points.txt"
    if not annot_file.exists():
        raise ValueError(f"Annotation file not found: {annot_file}")

    annot_flat = np.loadtxt(annot_file)
    if annot_flat.ndim == 1:
        annot_flat = annot_flat[np.newaxis, :]
    annot = rearrange(annot_flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2)

    flag_file = annot_dir / f"{seq_name}_flag.txt"
    if not flag_file.exists():
        raise ValueError(f"Flag file not found: {flag_file}")

    flags = np.loadtxt(flag_file)
    if flags.ndim == 0:
        flags = np.array([flags])

    has_coords = np.any(annot_flat != 0, axis=1)
    valid = has_coords & (flags == 0)

    return annot, valid


def pt_load_gt(base_dir: Path, seq_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth annotations for a PlanarTrack sequence.

    Args:
        base_dir: Base directory of PlanarTrack dataset.
        seq_name: Sequence name.

    Returns:
        annot: Corner annotations with shape (num_frames, 2, 4) - (x,y coords, 4 corners).
        valid: Boolean array (num_frames,) indicating valid frames.
    """
    annot_dir = base_dir / "annos"

    annot_file = annot_dir / f"{seq_name}.txt"
    if not annot_file.exists():
        raise ValueError(f"Annotation file not found: {annot_file}")

    annot_flat = np.loadtxt(annot_file)
    if annot_flat.ndim == 1:
        annot_flat = annot_flat[np.newaxis, :]
    annot = rearrange(annot_flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2)

    flag_file = annot_dir / f"{seq_name}_flag.txt"
    if not flag_file.exists():
        raise ValueError(f"Flag file not found: {flag_file}")

    flags = np.loadtxt(flag_file)
    if flags.ndim == 0:
        flags = np.array([flags])

    has_coords = np.any(annot_flat != 0, axis=1)
    valid = has_coords & (flags == 0)

    return annot, valid


def pti_load_gt(
    base_dir: Path,
    seq_name: str,
    initfix_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth for a PlanarTrack sequence with initfix initial corners.

    Loads the standard PlanarTrack annotation and then patches the first-frame
    corners with the improved initialisation stored in ``initfix_dir``.  The
    initfix file contains a single line of 8 space-separated values
    ``x0 y0 x1 y1 x2 y2 x3 y3`` (one corner per xy pair).

    Args:
        base_dir: Base directory of the PlanarTrack dataset.
        seq_name: Sequence name.
        initfix_dir: Directory containing per-sequence initfix files
            (``<seq_name>.txt``).

    Returns:
        annot: Corner annotations with shape (num_frames, 2, 4) - (x,y coords, 4 corners),
            with frame 0 replaced by the initfix corners.
        valid: Boolean array (num_frames,) indicating valid frames.

    Raises:
        ValueError: If the initfix file is not found or has unexpected shape.
    """
    annot, valid = pt_load_gt(base_dir, seq_name)

    initfix_file = initfix_dir / f"{seq_name}.txt"
    if not initfix_file.exists():
        raise ValueError(f"Initfix file not found: {initfix_file}")

    initfix_flat = np.loadtxt(initfix_file)
    if initfix_flat.shape != (8,):
        raise ValueError(
            f"Initfix file {initfix_file} must contain exactly 8 values "
            f"(x,y per corner × 4), got shape {initfix_flat.shape}"
        )

    annot[0] = rearrange(initfix_flat, "(corners xy) -> xy corners", corners=4, xy=2)

    return annot, valid


def make_gt_loader(
    dataset: str,
    initfix_dir: Optional[Path] = None,
) -> Callable[[Path, str], Tuple[np.ndarray, np.ndarray]]:
    """Return the GT-loading callable for *dataset*.

    Args:
        dataset: Any accepted CLI dataset key.
        initfix_dir: Required when ``DATASETS[dataset].gt_loader == 'pti'``.

    Returns:
        A callable ``(base_dir, seq_name) -> (annot, valid)``.

    Raises:
        ValueError: If the dataset is unknown, or ``initfix_dir`` is missing
            for a dataset that needs it.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected one of: {sorted(DATASETS)!r}.")

    ds_info = DATASETS[dataset]
    if ds_info.gt_loader == "pot":
        return pot_load_gt
    if ds_info.gt_loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir must be provided for the 'pti' GT loader")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def make_image_paths(dataset: str) -> Callable[[Path, str], List[Path]]:
    """Return the image-path callable for *dataset*.

    Args:
        dataset: Any accepted CLI dataset key.

    Returns:
        A callable ``(base_dir, seq_name) -> [frame paths]``.

    Raises:
        ValueError: If the dataset is unknown.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected one of: {sorted(DATASETS)!r}.")

    ds_info = DATASETS[dataset]
    if ds_info.gt_loader == "pot":
        return pot_image_paths
    return pt_image_paths


def pot_image_paths(base_dir: Path, seq_name: str) -> List[Path]:
    """Get sorted list of image paths for a POT-210 sequence.

    Args:
        base_dir: Base directory of POT-210 dataset.
        seq_name: Sequence name.

    Returns:
        List of image file paths.
    """
    seq_dir = base_dir / "extracted" / "images" / seq_name
    if not seq_dir.exists():
        raise ValueError(f"Sequence directory not found: {seq_dir}")

    image_paths = [
        p for p in sorted(seq_dir.glob("*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_paths:
        raise ValueError(f"No images found in: {seq_dir}")

    return image_paths


def pt_image_paths(base_dir: Path, seq_name: str) -> List[Path]:
    """Get sorted list of image paths for a PlanarTrack sequence.

    Args:
        base_dir: Base directory of PlanarTrack dataset.
        seq_name: Sequence name.

    Returns:
        List of image file paths.
    """
    seq_dir = base_dir / "sequences" / seq_name
    if not seq_dir.exists():
        raise ValueError(f"Sequence directory not found: {seq_dir}")

    image_paths = [
        p for p in sorted(seq_dir.glob("*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_paths:
        raise ValueError(f"No images found in: {seq_dir}")

    return image_paths


def corners_to_mask(corners: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert corner coordinates to a binary mask.

    Args:
        corners: Corner coordinates with shape (2, 4) - (x,y coords, 4 corners).
        image_shape: (height, width) of the output mask.

    Returns:
        Binary mask with shape (height, width) as uint8.
    """
    corners_poly = rearrange(corners, "xy corners -> corners 1 xy").astype(np.int32)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [corners_poly], 255)
    return mask


def warp_corners(homography: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply homography transformation to corner coordinates.

    Args:
        homography: (3, 3) homography matrix.
        corners: Corner coordinates with shape (2, 4) - (x,y coords, 4 corners).

    Returns:
        Warped corners with shape (2, 4).
    """
    corners_hom = np.vstack([corners, np.ones((1, 4))])
    warped_hom = homography @ corners_hom
    return warped_hom[:2] / warped_hom[2:3]
