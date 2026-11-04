#!/usr/bin/env python3
"""How fast does a *coasted* SAM-H estimate go stale? (corner_fusion input study)

The SAM-H sub-tracker runs with ``flow_enabled=False``, so on a frame where the
mask aligner rejects its fit the sub-tracker coasts: it returns the previous
homography unchanged, with no fresh covariance. ``fuse_corners`` currently reads
"no fresh covariance" as "branch absent" and discards the coasted estimate
entirely — which is what regresses 11 POT sequences from p@15=100 to 60-80 (see
docs/2026-08-07.md, 2026-08-11 entry).

The fix needs a number: how much uncertainty does one coasted frame add? This
script measures it offline from an existing ``corner_fusion`` run's debug npz,
with no GPU and no dataset images:

    staleness n(t)  — frames since the last accepted align (0 on an accepted
                      frame), from ``maskh_align_accepted``
    error           — per-corner squared distance ``|warp(samh_H) - GT|^2``

Grouped by ``n``, the per-corner error implies an isotropic variance
(``||e||^2 / sigma^2 ~ chi-square(2)``, median 1.386), and a weighted linear fit
of that variance against ``n`` gives the random-walk drift term to add per
coasted frame:

    cov_stale(n) = cov_last * samh_cov_scale + drift_px2_per_frame * n * I

LoMa's error on the *same* frames is reported alongside, since the horizon at
which a coasted SAM-H stops being worth fusing is where it gets worse than the
branch it would be fused with.

No mask IO, no dataset images — debug npz + GT annotations only. Still imports
cv2 (via flatsam2.datasets), so submit it to the cpu partition rather than
running it bare on the login node (CLAUDE.md, "Which machine am I on?").

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/coast_covariance_study.py \\
        pot --config configs/tracker/corner_fusion_alignh_calibrated.py
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
from flatsam2.path_config import paths
from flatsam2.registry import CLI_CHOICES, DATASETS
from flatsam2.sequences import get_sequences

# Median of chi-square(2): ||e||^2 / sigma^2 has this median under an isotropic
# 2D Gaussian, so sigma^2 = median(||e||^2) / 1.386.
_CHI2_2_MEDIAN = 1.386

# Staleness buckets reported individually; everything above goes in the tail bucket.
_MAX_REPORTED_STALENESS = 20


def _load_gt_fn(dataset: str):
    """Return a ``(base_dir, seq_name) -> (gt_corners, valid)`` callable."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pti":
        return lambda base_dir, seq_name: pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)
    raise ValueError(f"coast_covariance_study supports pot/pti GT loaders, got {loader!r}")


def staleness_from_accepted(accepted: np.ndarray) -> np.ndarray:
    """Frames since the last accepted mask-align fit, per tracked frame.

    Args:
        accepted: (T,) bool, ``maskh_align_accepted`` per tracked frame.

    Returns:
        (T,) int array: 0 on an accepted frame, k on the k-th consecutive
        rejected frame after one, and -1 before the first acceptance (the
        sub-tracker is then coasting on the *init* homography, a different
        regime with no measured covariance to inflate).
    """
    staleness = np.full(len(accepted), -1, dtype=np.int64)
    n = -1
    for t, ok in enumerate(accepted):
        if ok:
            n = 0
        elif n >= 0:
            n += 1
        staleness[t] = n
    return staleness


def _corner_sq_errors(
    H_stack: np.ndarray,
    frame_idx: np.ndarray,
    init_corners: np.ndarray,
    gt_corners: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Per-(frame, corner) squared corner error of a branch, NaN where unusable.

    Args:
        H_stack: (T, 3, 3) branch homography per tracked frame, NaN when absent.
        frame_idx: (T,) 0-based frame index into ``gt_corners`` / ``valid``.
        init_corners: (2, 4) template corners.
        gt_corners: (N, 2, 4) GT corners for the sequence.
        valid: (N,) GT-availability flags.

    Returns:
        (T, 4) float array of ``|pred - gt|^2`` in px^2; NaN where the frame has
        no GT or the branch has no homography.
    """
    sq = np.full((len(frame_idx), 4), np.nan)
    for t, fidx in enumerate(frame_idx):
        if fidx >= len(valid) or not valid[fidx]:
            continue
        H = H_stack[t]
        if not np.all(np.isfinite(H)):
            continue
        delta = warp_corners(H, init_corners) - gt_corners[fidx]  # (2, 4)
        sq[t] = np.sum(delta**2, axis=0)
    return sq


def _median_err(sq_errors: np.ndarray) -> float:
    """Median corner error in px from an array of squared errors (NaN if none)."""
    finite = sq_errors[np.isfinite(sq_errors)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(np.sqrt(finite)))


def _p_at_k(sq_errors: np.ndarray, gt_frames: np.ndarray, k_px: float) -> float:
    """P@k of a branch: share of GT frames whose RMS corner error is <= ``k_px``.

    Args:
        sq_errors: (T, 4) per-corner squared error, NaN where the branch had no
            homography (counted as a failure, as ``evaluate.py`` does).
        gt_frames: (T,) bool, frames that have ground truth.
        k_px: Error threshold in pixels.

    Returns:
        Percentage in [0, 100], or NaN when the sequence has no GT frame.
    """
    if not gt_frames.any():
        return float("nan")
    scored = gt_frames & np.isfinite(sq_errors).all(axis=1)
    rms = np.sqrt(np.mean(sq_errors[scored], axis=1))
    return 100.0 * float(np.sum(rms <= k_px)) / float(np.sum(gt_frames))


def _bucket_variance(sq_errors: np.ndarray) -> float:
    """Isotropic variance implied by a set of squared 2D corner errors (px^2)."""
    return float(np.median(sq_errors) / _CHI2_2_MEDIAN)


def _fit_drift(ns: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares fit of ``var(n) = var0 + drift * n``.

    Args:
        ns: (B,) staleness value per bucket.
        variances: (B,) implied isotropic variance per bucket (px^2).
        weights: (B,) sample count per bucket.

    Returns:
        Tuple ``(var0_px2, drift_px2_per_frame)``.
    """
    design = np.stack([np.ones_like(ns, dtype=float), ns.astype(float)], axis=1)
    w = np.sqrt(weights.astype(float))
    coeffs, *_ = np.linalg.lstsq(design * w[:, None], variances * w, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CLI_CHOICES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument(
        "--samh-cov-scale",
        type=float,
        default=70.0,
        help="Calibration scale the run applied to the SAM-H covariance; used to "
        "report the measured variance at n=0 against the covariance the fuser "
        "actually believes (default: the calibrated config's 70.0).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Restrict to these sequence names (intersected with the subset/split).",
    )
    parser.add_argument(
        "--per-sequence",
        action="store_true",
        help="Also print one row per sequence: median accepted vs coasted error. The "
        "pooled table above mixes sequences where SAM-H is momentarily stale with "
        "sequences where it is permanently lost, which are different regimes.",
    )
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    results_dir = args.results_dir or paths.OUTPUT_DIRECTORY
    seq_dir = compute_output_dir(results_dir, args.dataset, args.config)

    sequences = get_sequences(
        args.dataset,
        base_dir,
        sequences=args.sequences,
        subset=args.subset,
        split=args.split,
    )
    load_gt_fn = _load_gt_fn(args.dataset)

    print(
        f"Coast study: config={args.config.stem}  dataset={ds_info.label}  "
        f"n_sequences={len(sequences)}"
    )

    # Per staleness value: SAM-H squared corner errors, LoMa's on the same
    # frames, and the trace of the last measured SAM-H covariance.
    samh_by_n: dict[int, list[float]] = {}
    loma_by_n: dict[int, list[float]] = {}
    cov_trace_by_n: dict[int, list[float]] = {}
    n_frames_total = 0
    n_frames_coasted = 0
    n_seqs = 0
    # Per sequence: (name, median accepted err, median coasted err, median LoMa
    # err on the coasted frames, coasted-sample count). Pooling corner samples
    # across sequences over-weights the long tails of permanently-lost
    # sequences; a sequence-level view separates the two regimes.
    per_sequence: list[tuple[str, float, float, float, int]] = []
    samh_p15: list[float] = []
    loma_p15: list[float] = []
    coast_share: list[float] = []

    for seq_name in sequences:
        npz_path = seq_dir / f"{seq_name}_debug.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            if "fusion_samh_cov" not in data or "maskh_align_accepted" not in data:
                continue
            n_seqs += 1
            frame_idx = data["frame_idx"]
            accepted = data["maskh_align_accepted"]
            samh_cov = data["fusion_samh_cov"]  # (T, 4, 3) packed xx, xy, yy
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            init_corners = gt_corners[0]

            staleness = staleness_from_accepted(accepted)
            samh_sq = _corner_sq_errors(
                data["samh_H"], frame_idx, init_corners, gt_corners, valid
            )
            loma_sq = _corner_sq_errors(
                data["loma_H"], frame_idx, init_corners, gt_corners, valid
            )

            n_frames_total += len(staleness)
            n_frames_coasted += int(np.sum(staleness > 0))

            # Trace of the covariance measured at the last acceptance, carried
            # forward through the coast — what a decay model would inflate.
            last_trace = np.nan
            for t in range(len(staleness)):
                n = int(staleness[t])
                if n < 0:
                    continue
                if n == 0:
                    xx, _, yy = samh_cov[t, :, 0], samh_cov[t, :, 1], samh_cov[t, :, 2]
                    last_trace = float(np.nanmean(xx + yy))
                bucket = min(n, _MAX_REPORTED_STALENESS)
                for k in range(4):
                    if np.isfinite(samh_sq[t, k]):
                        samh_by_n.setdefault(bucket, []).append(float(samh_sq[t, k]))
                    if np.isfinite(loma_sq[t, k]):
                        loma_by_n.setdefault(bucket, []).append(float(loma_sq[t, k]))
                if np.isfinite(last_trace):
                    cov_trace_by_n.setdefault(bucket, []).append(last_trace)

            measured = staleness == 0
            coasting = staleness > 0
            per_sequence.append(
                (
                    seq_name,
                    _median_err(samh_sq[measured]),
                    _median_err(samh_sq[coasting]),
                    _median_err(loma_sq[coasting]),
                    int(np.sum(np.isfinite(samh_sq[coasting]))),
                )
            )
            # Sanity anchor: each branch's own p@15 on this run's telemetry.
            # samh_H's should land near the standalone SAM-H-only config's
            # published-grid number; if it does not, the sub-tracker inside the
            # fusion run is not behaving like the standalone one and the whole
            # staleness picture is measuring something else.
            gt_frames = np.array(
                [fidx < len(valid) and bool(valid[fidx]) for fidx in frame_idx]
            )
            samh_p15.append(_p_at_k(samh_sq, gt_frames, 15.0))
            loma_p15.append(_p_at_k(loma_sq, gt_frames, 15.0))
            coast_share.append(float(np.mean(coasting)))

    if n_seqs == 0:
        print("No corner_fusion debug npz found — nothing to study.")
        return 1

    print(
        f"\nSequences with telemetry: {n_seqs}/{len(sequences)}  "
        f"frames={n_frames_total}  coasted (staleness>0)={n_frames_coasted} "
        f"({n_frames_coasted / max(n_frames_total, 1):.1%})"
    )

    print(
        "\n  n  |     count | SAM-H var (px^2) | SAM-H med err (px) | "
        "LoMa var (px^2) | LoMa med err (px) | last-meas cov trace*scale"
    )
    print("  " + "-" * 108)
    ns: list[int] = []
    variances: list[float] = []
    weights: list[float] = []
    for n in sorted(samh_by_n):
        samh_arr = np.array(samh_by_n[n])
        loma_arr = np.array(loma_by_n.get(n, [np.nan]))
        var_samh = _bucket_variance(samh_arr)
        var_loma = _bucket_variance(loma_arr) if np.isfinite(loma_arr).any() else float("nan")
        traces = np.array(cov_trace_by_n.get(n, [np.nan]))
        believed = float(np.nanmedian(traces)) * args.samh_cov_scale / 2.0
        label = f"{n:>3}" if n < _MAX_REPORTED_STALENESS else f">={n}"
        print(
            f"  {label} | {len(samh_arr):>9} | {var_samh:>16.2f} | "
            f"{float(np.median(np.sqrt(samh_arr))):>18.2f} | {var_loma:>15.2f} | "
            f"{float(np.median(np.sqrt(loma_arr))):>17.2f} | {believed:>25.2f}"
        )
        # Fit only on the individually-resolved buckets; the tail bucket mixes
        # staleness values and would bias the slope.
        if n < _MAX_REPORTED_STALENESS and len(samh_arr) >= 50:
            ns.append(n)
            variances.append(var_samh)
            weights.append(len(samh_arr))

    if len(ns) >= 3:
        var0, drift = _fit_drift(np.array(ns), np.array(variances), np.array(weights))
        print(
            f"\nWeighted linear fit over n=0..{max(ns)}: "
            f"var(n) = {var0:.2f} + {drift:.2f} * n  (px^2)"
        )
        print(
            f"  -> samh_coast_drift_px2_per_frame = {drift:.1f}\n"
            f"  -> measured variance at n=0: {var0:.2f} px^2 "
            f"(vs the covariance the fuser believes, last column)"
        )
        crossings = [
            n
            for n in sorted(loma_by_n)
            if n < _MAX_REPORTED_STALENESS
            and len(loma_by_n[n]) >= 50
            and _bucket_variance(np.array(samh_by_n[n]))
            > _bucket_variance(np.array(loma_by_n[n]))
        ]
        if crossings:
            print(
                f"  -> SAM-H first worse than LoMa at staleness n={crossings[0]} "
                f"(candidate samh_coast_max_frames)"
            )
        else:
            print(
                "  -> SAM-H stays better than LoMa at every resolved staleness "
                "(no crossing; max_frames can be generous)"
            )
    else:
        print("\nToo few populated staleness buckets for a fit.")

    print(
        f"\nBranch p@15 recomputed from this run's telemetry (per-sequence mean): "
        f"samh_H {np.nanmean(samh_p15):.2f}  loma_H {np.nanmean(loma_p15):.2f}  |  "
        f"coasted frame share per sequence: median {np.median(coast_share):.1%}, "
        f"mean {np.mean(coast_share):.1%}"
    )

    _report_per_sequence(per_sequence, show_rows=args.per_sequence)
    return 0


def _report_per_sequence(
    rows: list[tuple[str, float, float, float, int]],
    show_rows: bool,
) -> None:
    """Sequence-level view of coasted SAM-H accuracy.

    The pooled staleness table answers "how wrong is a coasted estimate on a
    random coasted frame", which the permanently-lost sequences dominate. The
    decision the fuser makes is per sequence-regime, so also report how many
    sequences coast *well* — those are the ones a decay model can rescue, and
    the ones the POT regression lives in.

    Args:
        rows: Per-sequence ``(name, median accepted err, median coasted err,
            median LoMa err on coasted frames, coasted sample count)``.
        show_rows: Print every sequence, not just the summary.
    """
    usable = [r for r in rows if r[4] > 0 and np.isfinite(r[2])]
    if not usable:
        print("\nNo sequence has a coasted frame with GT — nothing to stratify.")
        return

    coast_err = np.array([r[2] for r in usable])
    loma_err = np.array([r[3] for r in usable])
    print(f"\nPer-sequence coasted SAM-H error (n={len(usable)} sequences with coasted GT frames):")
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q:<3} {np.percentile(coast_err, q):8.2f} px")
    for thr in (5.0, 15.0):
        share = float(np.mean(coast_err <= thr))
        print(f"  sequences with median coasted error <= {thr:.0f}px: {share:.1%}")
    better = np.isfinite(loma_err) & (coast_err < loma_err)
    print(
        f"  sequences where the coasted SAM-H beats LoMa on those same frames: "
        f"{float(np.mean(better)):.1%}"
    )

    if show_rows:
        print("\n  sequence   | n_coast |  acc err |coast err | LoMa err")
        print("  " + "-" * 54)
        for name, acc, coast, loma, n in sorted(rows, key=lambda r: -(r[2] if np.isfinite(r[2]) else -1)):
            print(f"  {name:<10} | {n:>7} | {acc:>8.2f} | {coast:>8.2f} | {loma:>8.2f}")


if __name__ == "__main__":
    sys.exit(main())
