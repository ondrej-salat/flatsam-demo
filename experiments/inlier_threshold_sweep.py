#!/usr/bin/env python3
"""How arbitrary is the 35-inlier switch? Sweep it offline on stored branch poses.

The quad-based tracker takes the keypoint pose when RANSAC returns at least 35
inliers and the mask pose otherwise. The paper's own motivation calls that a
cliff with an arbitrary number on it, so the number needs defending: if P@15 is
flat across a wide band of thresholds, the choice is not load-bearing and the
criticism is about form rather than substance; if it is a peak, the tracker is
tuned to one dataset and a reviewer is right to say so.

Both branch homographies and the inlier count are already in each run's
``<seq>_debug.npz``, so the sweep costs no GPU. The rule is reconstructed as
``inliers >= T -> loma_H, else samh_H`` and **validated at the deployed T
against the run's own stored ``selected`` array** — the replay reports
MATCH/MISMATCH on that before any swept number, exactly as
``fusion_rule_replay.py`` does. A MISMATCH means the reconstruction is not the
rule and nothing below it means anything.

Split hygiene, since this runs on the test split: 35 is **inherited** from the
predecessor system (`LMr_allobj_infl10_X_SAM3H_thr35`), not fitted here and not
fitted on these sequences. The sweep is therefore a post-hoc sensitivity check
of a value chosen elsewhere, which is a legitimate thing to report on test. It
would *not* be legitimate to read the argmax off this table and adopt it -- that
would be selecting on test. The claim is the flatness, never the peak.

One caveat that cannot be replayed away: selection feeds back into the tracker
(the chosen pose becomes the next frame's prior and drives redetection), so the
stored branch poses are conditioned on the deployed threshold. Off-deployment
columns are therefore a first-order sensitivity, not what a re-run would score.
The flatness of the curve is the claim, not any single swept value.

Imports cv2 via flatsam2.datasets — submit to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/inlier_threshold_sweep.py \\
        pti --config configs/tracker/loma_samh_matchnaming.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

_DEPLOYED_THRESHOLD = 35
_THRESHOLDS = (5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300)
_P_AT_K_PX = 15.0


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(f"inlier_threshold_sweep supports pot/pti, got {loader!r}")


def sequence_errors(
    data,
    gt_corners: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-frame alignment error of each branch, plus the inlier count.

    Args:
        data: Open ``<seq>_debug.npz`` mapping.
        gt_corners: (N, 2, 4) GT corners.
        valid: (N,) GT-availability flags.

    Returns:
        ``{"loma": (M,), "samh": (M,), "inliers": (M,), "selected": (M,)}`` over
        the frames that have GT, with NaN where a branch produced no pose.
    """
    frame_idx = data["frame_idx"]
    init_corners = gt_corners[0]
    keep = [
        t for t, fidx in enumerate(frame_idx)
        if fidx < len(valid) and valid[fidx] and fidx != 0
    ]
    out: Dict[str, np.ndarray] = {}
    for name, key in (("loma", "loma_H"), ("samh", "samh_H")):
        errs = np.full(len(keep), np.nan)
        stack = data[key]
        for j, t in enumerate(keep):
            H = stack[t]
            if not np.all(np.isfinite(H)):
                continue
            fidx = frame_idx[t]
            errs[j] = float(
                alignment_error(gt_corners[fidx], warp_corners(H, init_corners))
            )
        out[name] = errs
    out["inliers"] = data["n_inliers"][keep].astype(np.float64)
    out["selected"] = data["selected"][keep]
    return out


def switch_p_at_15(seq: Dict[str, np.ndarray], threshold: float) -> float:
    """P@15 of the ``inliers >= threshold`` switch on one sequence, in percent.

    A frame whose chosen branch produced no pose counts as a miss, which is how
    ``evaluate.py`` scores a frame with no prediction.
    """
    take_loma = seq["inliers"] >= threshold
    err = np.where(take_loma, seq["loma"], seq["samh"])
    if err.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(np.nan_to_num(err, nan=np.inf) <= _P_AT_K_PX))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)
    sequences = get_sequences(args.dataset, base_dir, subset=args.subset, split=args.split)
    load_gt_fn = _load_gt_fn(args.dataset)

    print(f"Inlier-threshold sweep: config={args.config.stem}  dataset={ds_info.label}")

    per_seq: List[Dict[str, np.ndarray]] = []
    n_rule_frames = n_rule_agree = 0
    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            continue
        gt_corners, valid = load_gt_fn(base_dir, seq_name)
        with np.load(npz_path, allow_pickle=False) as data:
            if "n_inliers" not in data or "selected" not in data:
                continue
            seq = sequence_errors(data, gt_corners, valid)
        per_seq.append(seq)
        reconstructed = np.where(
            seq["inliers"] >= _DEPLOYED_THRESHOLD, "loma_H", "samh_H"
        )
        n_rule_frames += len(reconstructed)
        n_rule_agree += int(np.sum(reconstructed == seq["selected"]))

    if not per_seq:
        print("No debug npz with switch telemetry — nothing to sweep.")
        return 1

    # Self-check first: the reconstruction must be the rule the run applied.
    agree = 100.0 * n_rule_agree / n_rule_frames if n_rule_frames else float("nan")
    verdict = "MATCH" if agree > 99.9 else "MISMATCH"
    print(
        f"\nself-check: reconstructed switch at T={_DEPLOYED_THRESHOLD} reproduces the "
        f"run's own selection on {agree:.2f}% of {n_rule_frames} frames  [{verdict}]"
    )
    if verdict == "MISMATCH":
        print("  -> the reconstruction is not the deployed rule; the sweep below is void.")
        return 1

    print(f"\n  {'T':>5} | {'per-seq mean P@15':>18} | {'pooled P@15':>12} | {'loma share':>10}")
    print("  " + "-" * 56)
    for threshold in _THRESHOLDS:
        means = [switch_p_at_15(s, threshold) for s in per_seq]
        errs = np.concatenate([
            np.where(
                s["inliers"] >= threshold,
                np.nan_to_num(s["loma"], nan=np.inf),
                np.nan_to_num(s["samh"], nan=np.inf),
            )
            for s in per_seq
        ])
        share = 100.0 * float(
            np.mean(np.concatenate([s["inliers"] >= threshold for s in per_seq]))
        )
        pooled = 100.0 * float(np.mean(errs <= _P_AT_K_PX))
        marker = "  <- deployed" if threshold == _DEPLOYED_THRESHOLD else ""
        print(
            f"  {threshold:5d} | {np.nanmean(means):18.2f} | {pooled:12.2f} | "
            f"{share:9.1f}%{marker}"
        )

    # What the band actually costs: the spread over a wide, plausible range is
    # the number that answers "is 35 load-bearing?".
    band = [t for t in _THRESHOLDS if 15 <= t <= 100]
    band_means = [np.nanmean([switch_p_at_15(s, t) for s in per_seq]) for t in band]
    print(
        f"\n  over T in [{band[0]}, {band[-1]}]: best {max(band_means):.2f} at "
        f"T={band[int(np.argmax(band_means))]}, worst {min(band_means):.2f}, "
        f"spread {max(band_means) - min(band_means):.2f} pp"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
