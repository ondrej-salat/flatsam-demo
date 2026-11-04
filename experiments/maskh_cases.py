#!/usr/bin/env python3
"""A frozen set of mask-branch pose problems, extracted once.

Studying why the mask branch loses the pose under truncation needs one frame's
worth of state, not a tracking run: the SAM3 mask, the template corners, the
branch's incumbent homography, what it actually produced, and the ground truth
to score against. This script writes exactly that into one ``.npz`` per case --
image included, so a case is self-contained -- and nothing downstream re-runs
SAM3, LoMa, or the tracker.

Cases are the annotated frames of the sequences whose masks were dumped for the
paper figure (``$OUTPUT_DIRECTORY/maskpose_maskdump``). Each is labelled by what
the mask branch did on it, so the same file holds the failures to fix and the
healthy frames a fix must not break.

Imports cv2 -- submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/maskh_cases.py
    python slurm/submit_cmd.py --gpus 0 -- python experiments/maskh_cases.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.datasets import pot_image_paths, pot_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths

# The masks dumped for the failure figure; every case comes from this re-run so
# that mask, incumbent and stored pose all belong to one consistent run.
_DUMP = "maskpose_maskdump"
_CONFIG = "loma_samh_matchnaming"
_SEQUENCES = ("V22_5", "V13_5", "V19_5", "V01_5", "V07_6", "V29_3")
_FAIL_PX = 15.0


def case_dir(results_dir: Path) -> Path:
    """Directory holding the frozen cases."""
    return results_dir / "maskh_cases"


def build(results_dir: Path, base_dir: Path, sequences: List[str]) -> List[Path]:
    """Extract every annotated frame of ``sequences`` into a self-contained case.

    Args:
        results_dir: Root of the results tree.
        base_dir: POT-210 dataset root.
        sequences: Sequence names to extract from.

    Returns:
        Paths of the written case files.

    Raises:
        ValueError: When a sequence has no dumped masks or no telemetry.
    """
    out_dir = case_dir(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for seq_name in sequences:
        run_dir = results_dir / _DUMP / _CONFIG / "POT-210"
        masks_dir = run_dir / "masks" / seq_name
        npz_path = run_dir / f"{seq_name}_debug.npz"
        if not masks_dir.is_dir() or not npz_path.exists():
            raise ValueError(f"no dumped run for {seq_name}: {masks_dir}")
        gt, valid = pot_load_gt(base_dir, seq_name)
        image_paths = pot_image_paths(base_dir, seq_name)
        with np.load(npz_path, allow_pickle=False) as data:
            rows = {int(f): i for i, f in enumerate(data["frame_idx"])}
            samh_H = data["samh_H"]
            loma_H = data["loma_H"]
            naming_shift = data["naming_shift"]
        for frame_idx in sorted(rows):
            if frame_idx == 0 or frame_idx >= len(gt) or not valid[frame_idx]:
                continue
            mask_path = masks_dir / f"{frame_idx + 1:05d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(image_paths[frame_idx]))
            if mask is None or image is None:
                continue
            stored_H = samh_H[rows[frame_idx]]
            previous = rows.get(frame_idx - 1)
            if previous is None or not np.all(np.isfinite(samh_H[previous])):
                continue
            if not np.all(np.isfinite(stored_H)):
                continue
            stored_corners = warp_corners(stored_H, gt[0])
            error = float(alignment_error(gt[frame_idx: frame_idx + 1],
                                          stored_corners[None])[0])
            path = out_dir / f"{seq_name}_f{frame_idx:05d}.npz"
            np.savez_compressed(
                path,
                sequence=seq_name,
                frame_idx=frame_idx,
                image=image,
                mask=(mask > 127),
                init_corners=gt[0],
                gt_corners=gt[frame_idx],
                incumbent_H=samh_H[previous],
                stored_H=stored_H,
                keypoint_H=loma_H[rows[frame_idx]],
                naming_shift=int(naming_shift[rows[frame_idx]]),
                stored_error_px=error,
            )
            written.append(path)
    return written


def load(path: Path) -> Dict[str, np.ndarray]:
    """Load one case into a plain dict.

    Args:
        path: Case ``.npz`` written by :func:`build`.

    Returns:
        Dict of the stored arrays; ``sequence`` is a str and ``frame_idx``,
        ``naming_shift`` ints, the rest float or bool arrays.
    """
    with np.load(path, allow_pickle=False) as data:
        case = {k: data[k] for k in data.files}
    case["sequence"] = str(case["sequence"])
    case["frame_idx"] = int(case["frame_idx"])
    case["naming_shift"] = int(case["naming_shift"])
    case["stored_error_px"] = float(case["stored_error_px"])
    return case


def cases(results_dir: Path, only_failing: bool = False) -> List[Path]:
    """Every frozen case, optionally only the ones the mask branch got wrong.

    Args:
        results_dir: Root of the results tree.
        only_failing: Keep only cases whose stored pose is beyond 15 px.

    Returns:
        Sorted case paths.
    """
    paths_found = sorted(case_dir(results_dir).glob("*.npz"))
    if not only_failing:
        return paths_found
    return [p for p in paths_found if load(p)["stored_error_px"] > _FAIL_PX]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--sequences", nargs="+", default=list(_SEQUENCES))
    parser.add_argument("--list", action="store_true", help="List existing cases only.")
    args = parser.parse_args(argv)

    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    base_dir = args.base_dir or paths.POT_BASE

    if not args.list:
        written = build(results_dir, base_dir, args.sequences)
        print(f"wrote {len(written)} cases to {case_dir(results_dir)}")

    print(f"  {'case':<22} {'stored err px':>13} {'shift':>6}  verdict")
    for path in cases(results_dir):
        case = load(path)
        verdict = "FAIL" if case["stored_error_px"] > _FAIL_PX else "ok"
        print(f"  {path.stem:<22} {case['stored_error_px']:13.2f} "
              f"{case['naming_shift']:6d}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
