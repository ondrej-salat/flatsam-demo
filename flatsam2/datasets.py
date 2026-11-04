"""Dataset loading utilities for POT-210 and PlanarTrack datasets.

This module provides functions to load sequences, ground truth annotations,
and image paths for planar object tracking datasets.

The canonical dataset registry (:data:`DATASETS`, :class:`DatasetInfo`,
:data:`CLI_CHOICES`) lives in :mod:`flatsam2.registry` and is re-exported
here for convenience.  All CLI scripts should import those names from here
or directly from :mod:`flatsam2.registry`.

Lightweight sequence-listing helpers (``get_sequences``, ``pot_sequences``,
``pt_sequences``, ``load_sequence_list``) live in :mod:`flatsam2.sequences`
and are re-exported here for backwards compatibility.
"""

import re
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
    "corners_txt_load_gt",
    "mpot_load_gt",
    # Image paths
    "pot_image_paths",
    "pt_image_paths",
    "corners_txt_image_paths",
    "mpot_image_paths",
    "parse_mpot_sequence",
    # Per-dataset dispatch
    "make_gt_loader",
    "make_image_paths",
    # Geometry
    "corners_to_mask",
    "warp_corners",
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


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


def _subdir(base_dir: Path, subdir: str) -> Path:
    """Resolve a registry subdirectory against a dataset root (``''`` = the root)."""
    return base_dir / subdir if subdir else base_dir


def _frame_number(file_name: str) -> int:
    """Extract the frame index from a ``frame00123.jpg``-style name.

    Frames are matched on this number rather than on the file name itself
    because TMT's robot sequences zero-pad their images and their annotations to
    different widths (``frame00001.jpg`` against ``frame0001.jpg``).

    Args:
        file_name: Image file name, with or without a directory or suffix.

    Returns:
        The trailing integer in the file stem.

    Raises:
        ValueError: If the stem carries no trailing digits.
    """
    match = re.search(r"(\d+)$", Path(file_name).stem)
    if match is None:
        raise ValueError(f"Cannot read a frame number from: {file_name}")
    return int(match.group(1))


def _read_corners_txt(gt_file: Path) -> dict:
    """Parse a ``frame ulx uly urx ury lrx lry llx lly`` annotation file.

    The format is shared by POIC, TMT and UCSB: one header line, then one line
    per annotated frame naming the image file and its four corners in
    upper-left, upper-right, lower-right, lower-left order.

    Args:
        gt_file: Path to the annotation file.

    Returns:
        Mapping from frame number to a ``(2, 4)`` corner array.

    Raises:
        ValueError: If the file is missing or a row does not hold 8 numbers.
    """
    if not gt_file.exists():
        raise ValueError(f"Annotation file not found: {gt_file}")

    corners_by_frame = {}
    for line_no, line in enumerate(gt_file.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0] == "frame":  # blank line or the header
            continue
        if len(fields) != 9:
            raise ValueError(
                f"{gt_file}:{line_no} must hold a frame name and 8 numbers, got {len(fields)} fields"
            )
        values = np.array([float(v) for v in fields[1:]])
        corners_by_frame[_frame_number(fields[0])] = rearrange(
            values, "(corners xy) -> xy corners", corners=4, xy=2
        )

    if not corners_by_frame:
        raise ValueError(f"No annotations parsed from: {gt_file}")

    return corners_by_frame


def corners_txt_image_paths(base_dir: Path, seq_name: str, seq_subdir: str = "") -> List[Path]:
    """Get the frame paths for a POIC / TMT / UCSB sequence.

    Args:
        base_dir: Dataset root directory.
        seq_name: Sequence name.
        seq_subdir: Registry ``seq_subdir`` for the dataset (``''`` = the root).

    Returns:
        List of image file paths, ordered by frame number.

    Raises:
        ValueError: If the sequence directory is missing or holds no images.
    """
    seq_dir = _subdir(base_dir, seq_subdir) / seq_name
    if not seq_dir.exists():
        raise ValueError(f"Sequence directory not found: {seq_dir}")

    image_paths = [p for p in seq_dir.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES]
    if not image_paths:
        raise ValueError(f"No images found in: {seq_dir}")

    return sorted(image_paths, key=lambda p: _frame_number(p.name))


def corners_txt_load_gt(
    base_dir: Path,
    seq_name: str,
    seq_subdir: str = "",
    gt_subdir: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth for a POIC / TMT / UCSB sequence.

    These datasets annotate one frame per row and normally ship exactly as many
    rows as frames, so rows are matched to frames **by position**.  That is the
    only reading that survives TMT's ``robot_bookI``, whose 588 images are
    numbered from 0 while its 588 annotations are numbered from 1: matching on
    the number instead would drop a frame at each end and shift every
    correspondence by one.

    Only when the counts disagree — two TMT robot sequences ship fewer frames
    than annotations — does matching fall back to the frame number parsed from
    the file name, leaving unmatched frames invalid.

    Args:
        base_dir: Dataset root directory.
        seq_name: Sequence name.
        seq_subdir: Registry ``seq_subdir`` for the dataset.
        gt_subdir: Registry ``gt_subdir`` for the dataset.

    Returns:
        annot: Corner annotations with shape (num_frames, 2, 4) - (x,y coords, 4 corners).
        valid: Boolean array (num_frames,) indicating annotated frames.

    Raises:
        ValueError: If the first frame is unannotated, since it initialises the tracker.
    """
    corners_by_frame = _read_corners_txt(_subdir(base_dir, gt_subdir) / f"{seq_name}.txt")
    image_paths = corners_txt_image_paths(base_dir, seq_name, seq_subdir)

    annot = np.zeros((len(image_paths), 2, 4))
    valid = np.zeros(len(image_paths), dtype=bool)

    if len(corners_by_frame) == len(image_paths):
        for idx, frame_number in enumerate(sorted(corners_by_frame)):
            annot[idx] = corners_by_frame[frame_number]
            valid[idx] = True
    else:
        for idx, image_path in enumerate(image_paths):
            corners = corners_by_frame.get(_frame_number(image_path.name))
            if corners is not None:
                annot[idx] = corners
                valid[idx] = True

    if not valid[0]:
        raise ValueError(
            f"Sequence {seq_name!r} has no annotation for its first frame "
            f"({image_paths[0].name}), which is needed to initialise the tracker"
        )

    return annot, valid


def _mpot_split_dir(base_dir: Path, seq_name: str) -> Path:
    """Find the split directory (``test`` / ``val`` / ``train``) holding *seq_name*."""
    for split in ("test", "val", "train"):
        seq_dir = base_dir / split / seq_name
        if seq_dir.is_dir():
            return seq_dir
    raise ValueError(f"MPOT-3K sequence not found in any split: {seq_name}")


def _mpot_rows(gt_file: Path, instance_id: int) -> List[List[float]]:
    """Read the rows of an MPOT annotation file belonging to one instance.

    Args:
        gt_file: Path to ``gt.txt`` or ``gt_init.txt``.
        instance_id: Instance id to keep (the file's second column).

    Returns:
        Rows ``[frame, id, x1, y1, x2, y2, x3, y3, x4, y4]``, ordered by frame.
    """
    if not gt_file.exists():
        raise ValueError(f"Annotation file not found: {gt_file}")

    rows = []
    for line in gt_file.read_text().splitlines():
        fields = [f for f in line.strip().split(",") if f != ""]
        if len(fields) < 10 or int(float(fields[1])) != instance_id:
            continue
        rows.append([float(f) for f in fields[:10]])

    return sorted(rows, key=lambda row: row[0])


def parse_mpot_sequence(seq_name: str) -> Tuple[str, int]:
    """Split an MPOT task key ``'<sequence>__<instance>'`` into its two parts.

    Args:
        seq_name: Task key, e.g. ``'buildings5-1__3'``.

    Returns:
        The sequence name and the instance id.

    Raises:
        ValueError: If the key does not carry an ``__<instance>`` suffix.
    """
    sequence, separator, instance = seq_name.rpartition("__")
    if not separator or not instance.isdigit():
        raise ValueError(
            f"MPOT sequence key must be '<sequence>__<instance>', got {seq_name!r}"
        )
    return sequence, int(instance)


def mpot_image_paths(base_dir: Path, seq_name: str) -> List[Path]:
    """Get the frame paths for one MPOT-3K tracking task.

    A task is a single planar instance, which need not be present from the
    start of the video, so the list begins at that instance's first annotated
    frame — the "first" initialisation of the official protocol — and runs to
    the end of the sequence.

    Args:
        base_dir: MPOT-3K root directory.
        seq_name: Task key ``'<sequence>__<instance>'``.

    Returns:
        List of image file paths.

    Raises:
        ValueError: If the sequence, the instance, or its frames are missing.
    """
    sequence, instance_id = parse_mpot_sequence(seq_name)
    seq_dir = _mpot_split_dir(base_dir, sequence)

    init_rows = _mpot_rows(seq_dir / "gt" / "gt_init.txt", instance_id)
    if not init_rows:
        raise ValueError(f"No initialisation row for instance {instance_id} in {sequence}")
    init_frame = int(init_rows[0][0])

    image_dir = seq_dir / "seq1"
    image_paths = [
        p
        for p in sorted(image_dir.glob("*"))
        if p.suffix.lower() in _IMAGE_SUFFIXES and int(p.stem) >= init_frame
    ]
    if not image_paths:
        raise ValueError(f"No frames at or after frame {init_frame} in: {image_dir}")

    return image_paths


def mpot_load_gt(base_dir: Path, seq_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth for one MPOT-3K tracking task.

    Args:
        base_dir: MPOT-3K root directory.
        seq_name: Task key ``'<sequence>__<instance>'``.

    Returns:
        annot: Corner annotations with shape (num_frames, 2, 4) - (x,y coords, 4 corners),
            covering the frames returned by :func:`mpot_image_paths`.
        valid: Boolean array (num_frames,) — false where the instance is absent
            (occluded or out of view), which MPOT records by omitting the row.

    Raises:
        ValueError: If the instance has no annotation on its initialisation frame.
    """
    sequence, instance_id = parse_mpot_sequence(seq_name)
    seq_dir = _mpot_split_dir(base_dir, sequence)
    image_paths = mpot_image_paths(base_dir, seq_name)

    corners_by_frame = {
        int(row[0]): rearrange(np.array(row[2:10]), "(corners xy) -> xy corners", corners=4, xy=2)
        for row in _mpot_rows(seq_dir / "gt" / "gt.txt", instance_id)
    }

    annot = np.zeros((len(image_paths), 2, 4))
    valid = np.zeros(len(image_paths), dtype=bool)
    for idx, image_path in enumerate(image_paths):
        corners = corners_by_frame.get(int(image_path.stem))
        if corners is not None:
            annot[idx] = corners
            valid[idx] = True

    if not valid[0]:
        raise ValueError(
            f"Instance {instance_id} of {sequence} has no annotation on its "
            f"initialisation frame ({image_paths[0].stem})"
        )

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
    if ds_info.gt_loader == "corners_txt":
        return lambda bd, sn: corners_txt_load_gt(bd, sn, ds_info.seq_subdir, ds_info.gt_subdir)
    if ds_info.gt_loader == "mpot":
        return mpot_load_gt
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
    if ds_info.gt_loader == "corners_txt":
        return lambda bd, sn: corners_txt_image_paths(bd, sn, ds_info.seq_subdir)
    if ds_info.gt_loader == "mpot":
        return mpot_image_paths
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
