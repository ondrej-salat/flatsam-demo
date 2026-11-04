#!/usr/bin/env python3
"""P@k of each stored candidate homography in a run (branch-level accuracy).

A run's aggregate score says how well its *selection* did; this says how good
the things it was selecting between actually were. For every candidate stored
in ``{seq}_debug.npz`` (``loma_H``, ``samh_H``, ``mask_H``, ``fused_H``, ...) it
warps the template corners and scores P@5/P@15 against GT, per-sequence mean —
the same aggregation the grid tables quote. A frame with no candidate counts as
a failure, exactly as ``evaluate.py`` scores a frame with no prediction.

Its point of comparison: the quad SAM-H branch vs the mask-align SAM-H branch,
each measured on its own, so branch quality can be separated from the fusion or
switch rule sitting on top of it.

Imports cv2 via flatsam2.datasets — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/branch_p_at_k.py \\
        pti --config configs/tracker/loma_samh_decoupled_rerun.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

_CANDIDATES = ("loma_H", "samh_H", "mask_H", "fused_H", "flow_H")
# The two branches a selection rule actually chooses between, and the label
# for the per-frame oracle over them.
_ORACLE_OVER = ("loma_H", "samh_H")
_ORACLE = "ORACLE(loma,samh)"
_THRESHOLDS = (5.0, 15.0)


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(f"branch_p_at_k supports pot/pti GT loaders, got {loader!r}")


def candidate_p_at_k(
    H_stack: np.ndarray,
    frame_idx: np.ndarray,
    init_corners: np.ndarray,
    gt_corners: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, ...]:
    """P@k of one candidate on one sequence, plus its availability.

    Args:
        H_stack: (T, 3, 3) candidate homography per tracked frame, NaN when the
            candidate was not produced that frame.
        frame_idx: (T,) 0-based frame index into ``gt_corners`` / ``valid``.
        init_corners: (2, 4) template corners.
        gt_corners: (N, 2, 4) GT corners.
        valid: (N,) GT-availability flags.

    Returns:
        ``(p@5, p@15, available_share, n_gt)`` -- percentages, then the number
        of GT frames the percentages are over (its pooling weight); NaN when the
        sequence has no GT frame.
    """
    n_gt = 0
    n_have = 0
    hits = [0 for _ in _THRESHOLDS]
    for t, fidx in enumerate(frame_idx):
        if fidx >= len(valid) or not valid[fidx]:
            continue
        n_gt += 1
        H = H_stack[t]
        if not np.all(np.isfinite(H)):
            continue
        n_have += 1
        err = float(alignment_error(gt_corners[fidx], warp_corners(H, init_corners)))
        for i, k in enumerate(_THRESHOLDS):
            hits[i] += int(err <= k)
    if n_gt == 0:
        return (float("nan"),) * (len(_THRESHOLDS) + 2)
    return (*[100.0 * h / n_gt for h in hits], 100.0 * n_have / n_gt, float(n_gt))


def oracle_p_at_k(
    H_stacks: list[np.ndarray],
    frame_idx: np.ndarray,
    init_corners: np.ndarray,
    gt_corners: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, ...]:
    """P@k of a per-frame oracle that takes whichever branch is closer.

    This is the ceiling on *selection*: no switch, threshold or covariance
    arbitration can beat picking the better branch every frame. Read the gap
    between the shipped run and this row as the headroom a better rule has, and
    the gap between this row and 100 as what no rule can reach because neither
    branch had the answer.

    Args:
        H_stacks: One (T, 3, 3) stack per branch, aligned on ``frame_idx``.
        frame_idx: (T,) 0-based frame index into ``gt_corners`` / ``valid``.
        init_corners: (2, 4) template corners.
        gt_corners: (N, 2, 4) GT corners.
        valid: (N,) GT-availability flags.

    Returns:
        ``(p@5, p@15, available_share, n_gt)`` -- percentages, then the number
        of GT frames they are over; NaN when the sequence has no GT frame.
    """
    n_gt = 0
    n_have = 0
    hits = [0 for _ in _THRESHOLDS]
    for t, fidx in enumerate(frame_idx):
        if fidx >= len(valid) or not valid[fidx]:
            continue
        n_gt += 1
        errs = [
            float(alignment_error(gt_corners[fidx], warp_corners(H[t], init_corners)))
            for H in H_stacks
            if np.all(np.isfinite(H[t]))
        ]
        if not errs:
            continue
        n_have += 1
        best = min(errs)
        for i, k in enumerate(_THRESHOLDS):
            hits[i] += int(best <= k)
    if n_gt == 0:
        return (float("nan"),) * (len(_THRESHOLDS) + 2)
    return (*[100.0 * h / n_gt for h in hits], 100.0 * n_have / n_gt, float(n_gt))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)
    sequences = get_sequences(
        args.dataset, base_dir, sequences=args.sequences, subset=args.subset, split=args.split
    )
    load_gt_fn = _load_gt_fn(args.dataset)

    print(
        f"Branch P@k: config={args.config.stem}  dataset={ds_info.label}  "
        f"n_sequences={len(sequences)}"
    )

    scores: dict[str, list[tuple[float, ...]]] = {c: [] for c in _CANDIDATES}
    scores[_ORACLE] = []
    n_seqs = 0
    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            n_seqs += 1
            frame_idx = data["frame_idx"]
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]
            for cand in _CANDIDATES:
                if cand not in data:
                    continue
                scores[cand].append(
                    candidate_p_at_k(data[cand], frame_idx, init_corners, gt_corners, valid)
                )
            branches = [data[c] for c in _ORACLE_OVER if c in data]
            if len(branches) > 1:
                scores[_ORACLE].append(
                    oracle_p_at_k(branches, frame_idx, init_corners, gt_corners, valid)
                )

    if n_seqs == 0:
        print("No debug npz found — nothing to score.")
        return 1

    print(f"\nScored {n_seqs} sequences.\n")
    print(
        "  candidate        | pooled p@5 | pooled p@15 |  seq p@5 | seq p@15 | available"
    )
    print("  " + "-" * 76)
    for cand in (*_CANDIDATES, _ORACLE):
        rows = np.array([r for r in scores[cand] if np.isfinite(r[0])])
        if rows.size == 0:
            continue
        p5, p15, avail, _ = np.nanmean(rows, axis=0)
        # Pooled over frames -- the aggregation the benchmarks and the paper use.
        w = rows[:, 3]
        pool5 = float(np.sum(rows[:, 0] * w) / np.sum(w))
        pool15 = float(np.sum(rows[:, 1] * w) / np.sum(w))
        print(
            f"  {cand:<16} | {pool5:10.2f} | {pool15:11.2f} | {p5:8.2f} | "
            f"{p15:8.2f} | {avail:8.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
