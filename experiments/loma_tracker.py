#!/usr/bin/env python3
"""Estimate homographies from pre-computed LoMa correspondences.

Loads NPZ files produced by loma_experiment.py and for each frame runs
cv2.findHomography (with a configurable RANSAC variant) to estimate the
homography from the init frame.  Predicted corners are saved in the same
format as the FlatSAM2 tracker so that standard evaluation tooling can be
applied directly.

Output structure:
    <output-dir>/<model>__<ransac>[_thr<T>]/<dataset-label>/<seq>.txt

Usage examples:
    # POT-210, LoMaB, default OpenCV RANSAC
    python experiments/loma_tracker.py pot \\
        --model LoMaB --ransac RANSAC \\
        --loma-dir /mnt/.../LoMa/ --output-dir /mnt/.../loma_tracker/

    # PTI, LoMaG, MAGSAC
    python experiments/loma_tracker.py pti \\
        --model LoMaG --ransac USAC_MAGSAC \\
        --loma-dir /mnt/.../LoMa/ --output-dir /mnt/.../loma_tracker/

    # All models × all RANSAC methods on POT
    python experiments/loma_tracker.py pot \\
        --model all --ransac all \\
        --loma-dir /mnt/.../LoMa/ --output-dir /mnt/.../loma_tracker/

    # Only sequences in a subset file (for SLURM)
    python experiments/loma_tracker.py pot \\
        --model LoMaB --ransac RANSAC \\
        --subset-file slurm/tmp/part1.txt \\
        --loma-dir /mnt/.../LoMa/ --output-dir /mnt/.../loma_tracker/
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
    warp_corners,
)
from flatsam2.path_config import paths

_DEFAULT_LOMA_DIR = Path("/mnt/personal/serycjon/flatsam2/LoMa/")
_DEFAULT_OUTPUT_DIR = Path("/mnt/personal/serycjon/flatsam2/loma_tracker/")
_DEFAULT_THRESHOLD = 5.0
_DEFAULT_RANSAC = "RANSAC"

LOMA_MODELS = ["LoMaB", "LoMaB128", "LoMaL", "LoMaG", "LoMaR"]


def _build_ransac_methods() -> dict:
    """Build the available RANSAC method name → cv2 constant mapping.

    USAC variants are only included when the running OpenCV version supports them.
    """
    methods: dict[str, int] = {
        "RANSAC": cv2.RANSAC,
        # "LMEDS": cv2.LMEDS,
        # "RHO": cv2.RHO,
    }
    for name in ("USAC_DEFAULT", "USAC_MAGSAC",
                 # "USAC_FAST", "USAC_ACCURATE", "USAC_PROSAC"
                 ):
        val = getattr(cv2, name, None)
        if val is not None:
            methods[name] = int(val)
    return methods


RANSAC_METHODS: dict[str, int] = _build_ransac_methods()


# ---------------------------------------------------------------------------
# Config name / output path helpers
# ---------------------------------------------------------------------------

def make_config_name(model: str, ransac: str, threshold: float) -> str:
    """Derive a unique output directory stem from tracking settings.

    Args:
        model: LoMa model name (e.g. 'LoMaB').
        ransac: RANSAC method name (e.g. 'USAC_MAGSAC').
        threshold: RANSAC inlier threshold in pixels.

    Returns:
        Config stem string used as the second path component under output-dir.
    """
    thr_tag = f"_thr{threshold:.1f}" if abs(threshold - _DEFAULT_THRESHOLD) > 1e-6 else ""
    return f"{model}__{ransac}{thr_tag}"


# ---------------------------------------------------------------------------
# I/O helpers (mirror tracker format)
# ---------------------------------------------------------------------------

def save_corners(corners: np.ndarray, output_path: Path) -> None:
    """Save (N, 2, 4) corner array in tracker text format (8 floats/line).

    Args:
        corners: Predicted corners (num_frames, 2, 4).
        output_path: Destination .txt file.
    """
    corners_flat = rearrange(
        corners, "frames xy corners -> frames (corners xy)", corners=4, xy=2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, corners_flat, fmt="%.6f")


def is_sequence_complete(output_path: Path, num_frames: int) -> bool:
    """Return True if *output_path* exists and contains exactly *num_frames* rows.

    Args:
        output_path: .txt file to check.
        num_frames: Expected row count.
    """
    if not output_path.exists():
        return False
    try:
        data = np.loadtxt(output_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return len(data) == num_frames
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core tracking
# ---------------------------------------------------------------------------

def run_sequence(
    npz_path: Path,
    init_corners: np.ndarray,
    num_frames: int,
    ransac_method: int,
    ransac_threshold: float,
) -> np.ndarray:
    """Estimate homographies from LoMa correspondences and warp init corners.

    For each frame i ≥ 1 the function:
    1. Loads kptsA / kptsB from the NPZ.
    2. Runs cv2.findHomography with the requested RANSAC variant.
    3. On success: warps init_corners through H.
    4. On failure: falls back to the previous frame's corners (last-good-H strategy).

    Frame 0 is always init_corners.

    Args:
        npz_path: Pre-computed LoMa correspondences (from loma_experiment.py).
        init_corners: Init-frame corners (2, 4).
        num_frames: Total frame count (including init).
        ransac_method: cv2 method constant for cv2.findHomography.
        ransac_threshold: RANSAC inlier pixel threshold.

    Returns:
        Predicted corners (num_frames, 2, 4).
    """
    data = np.load(npz_path)
    all_corners = np.zeros((num_frames, 2, 4), dtype=np.float64)
    all_corners[0] = init_corners

    last_H = np.eye(3, dtype=np.float64)

    for i in range(1, num_frames):
        key_A = f"kptsA_{i:05d}"
        key_B = f"kptsB_{i:05d}"

        H: Optional[np.ndarray] = None
        if key_A in data and key_B in data:
            kptsA = data[key_A].astype(np.float64)  # (N, 2) — init frame points
            kptsB = data[key_B].astype(np.float64)  # (N, 2) — current frame points
            if len(kptsA) >= 4:
                H_est, _ = cv2.findHomography(kptsA, kptsB, ransac_method, ransac_threshold)
                if H_est is not None:
                    H = H_est
                    last_H = H

        if H is None:
            H = last_H

        all_corners[i] = warp_corners(H, init_corners)

    return all_corners


# ---------------------------------------------------------------------------
# Dataset helpers (mirror loma_experiment.py)
# ---------------------------------------------------------------------------

def _make_gt_loader(
    dataset: str,
    initfix_dir: Optional[Path],
) -> Callable[[Path, str], Tuple[np.ndarray, np.ndarray]]:
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir=initfix_dir)


def _make_image_paths_fn(dataset: str) -> Callable[[Path, str], List[Path]]:
    if DATASETS[dataset].gt_loader == "pot":
        return pot_image_paths
    return pt_image_paths


# ---------------------------------------------------------------------------
# Per-model/ransac runner
# ---------------------------------------------------------------------------

def _run_single(args: argparse.Namespace, model: str, ransac: str) -> None:
    """Run one (model, ransac) combination as specified in *args*.

    Args:
        args: Parsed CLI namespace.
        model: Concrete LoMa model name.
        ransac: Concrete RANSAC method name.
    """
    ds_info = DATASETS[args.dataset]

    if args.base_dir is None:
        base_dir = getattr(paths, ds_info.base_dir_attr)
    else:
        base_dir = args.base_dir

    if ds_info.uses_split and args.split is None:
        args.split = "test"

    ransac_method = RANSAC_METHODS[ransac]
    cfg_name = make_config_name(model, ransac, args.ransac_threshold)
    output_dir = args.output_dir / cfg_name / ds_info.label
    loma_dir = args.loma_dir / model / ds_info.label

    sequences = get_sequences(
        args.dataset,
        base_dir,
        sequences=args.sequences,
        subset=getattr(args, "subset", None),
        subset_file=getattr(args, "subset_file", None),
        split=args.split,
    )
    if not sequences:
        print("No sequences found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDataset   : {ds_info.label}")
    print(f"Model     : {model}")
    print(f"RANSAC    : {ransac}  threshold={args.ransac_threshold}px")
    print(f"Config    : {cfg_name}")
    print(f"LoMa dir  : {loma_dir}")
    print(f"Output    : {output_dir}")
    print(f"Sequences : {len(sequences)}")

    gt_loader = _make_gt_loader(args.dataset, getattr(args, "initfix_dir", None))
    image_paths_fn = _make_image_paths_fn(args.dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_skip = 0
    needs_processing: List[Tuple[str, int]] = []
    for seq_name in sequences:
        image_paths = image_paths_fn(base_dir, seq_name)
        num_frames = len(image_paths)
        out_path = output_dir / f"{seq_name}.txt"
        if not args.force and is_sequence_complete(out_path, num_frames):
            n_skip += 1
        else:
            needs_processing.append((seq_name, num_frames))

    if n_skip:
        print(f"Skipping {n_skip} already-complete sequence(s).")

    n_missing_npz = 0
    for seq_name, num_frames in tqdm.tqdm(needs_processing, desc="sequences", unit="seq"):
        npz_path = loma_dir / f"{seq_name}.npz"
        if not npz_path.exists():
            print(f"  WARNING: NPZ not found, skipping {seq_name}: {npz_path}")
            n_missing_npz += 1
            continue

        gt_corners, _ = gt_loader(base_dir, seq_name)
        init_corners = gt_corners[0]

        pred_corners = run_sequence(
            npz_path, init_corners, num_frames, ransac_method, args.ransac_threshold
        )

        out_path = output_dir / f"{seq_name}.txt"
        save_corners(pred_corners, out_path)

    if n_missing_npz:
        print(f"WARNING: {n_missing_npz} sequence(s) had no NPZ (run loma_experiment.py first).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate homographies from LoMa correspondences (RANSAC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to process.")
    parser.add_argument(
        "--model",
        choices=LOMA_MODELS + ["all"],
        required=True,
        help="LoMa model variant to use, or 'all' to run every variant.",
    )
    parser.add_argument(
        "--ransac",
        choices=list(RANSAC_METHODS.keys()) + ["all"],
        default=_DEFAULT_RANSAC,
        help="RANSAC method for cv2.findHomography, or 'all' to try every method.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        metavar="PX",
        help="RANSAC inlier pixel threshold.",
    )
    parser.add_argument(
        "--loma-dir",
        type=Path,
        default=_DEFAULT_LOMA_DIR,
        help="Base directory of pre-computed LoMa NPZs (<loma-dir>/<model>/<dataset-label>/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Base output directory.  Results go into <output-dir>/<config-name>/<dataset-label>/.",
    )
    parser.add_argument("--base-dir", type=Path, help="Dataset root (defaults to path_config).")
    parser.add_argument("--initfix-dir", type=Path, help="Initfix dir for PTI (defaults to path_config).")
    parser.add_argument("--sequences", nargs="+", metavar="SEQ", help="Explicit sequence names.")
    parser.add_argument("--subset", metavar="NAME", help="Subset file name (configs/subsets/<name>.txt).")
    parser.add_argument("--subset-file", type=Path, metavar="PATH", help="Direct path to subset file (for SLURM).")
    parser.add_argument("--split", choices=["train", "test"], help="Train/test split (PT/PTI only).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    models = LOMA_MODELS if args.model == "all" else [args.model]
    ransacs = list(RANSAC_METHODS.keys()) if args.ransac == "all" else [args.ransac]

    combos = [(m, r) for m in models for r in ransacs]
    if len(combos) > 1:
        print(f"Running {len(combos)} combination(s): "
              f"{len(models)} model(s) × {len(ransacs)} RANSAC method(s)")

    for model, ransac in combos:
        if len(combos) > 1:
            print(f"\n{'='*60}\n{model} / {ransac}\n{'='*60}")
        _run_single(args, model, ransac)


if __name__ == "__main__":
    main()
