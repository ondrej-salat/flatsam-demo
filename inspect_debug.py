"""Sanity-check and stats for NN scorer debug npz files.

Accepts a single *_debug.npz file or a directory of them and prints
aggregate diagnostics useful for catching basic issues with the scorer.

Usage
-----
    python inspect_debug.py results/flow_hough_nn/POT-210/
    python inspect_debug.py results/flow_hough_nn/POT-210/some_seq_debug.npz
    python inspect_debug.py results/flow_hough_nn/POT-210/ results/flow_hough_nnpts/POT-210/
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_debug_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*_debug.npz"))
    if not files:
        print(f"WARNING: no *_debug.npz files found in {path}", file=sys.stderr)
    return files


def load_all(files: list[Path]) -> dict:
    """Load and concatenate arrays from multiple npz files."""
    parts: dict[str, list] = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        for k in d.files:
            parts.setdefault(k, []).append(d[k])

    out = {}
    for k, arrays in parts.items():
        if arrays[0].dtype.kind in ("U", "S", "O"):
            out[k] = np.concatenate(arrays)
        else:
            out[k] = np.concatenate(arrays, axis=0)
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

# Superset of candidate sources in {seq}_debug.npz (mirrors tracker._DEBUG_CANDIDATES,
# kept inline so this tool stays import-light — no torch/sam3).
PATHS = ("flow_H", "redetection_flow_H", "mask_H", "loma_H", "samh_H")


def _pct(n: int, total: int) -> str:
    return f"{n:6d}  ({100 * n / total:5.1f}%)" if total else f"{n:6d}"


def _quantile_str(vals: np.ndarray) -> str:
    if len(vals) == 0:
        return "  n/a"
    q = np.quantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])
    return (
        f"p5={q[0]:.1f}  p25={q[1]:.1f}  med={q[2]:.1f}  p75={q[3]:.1f}  p95={q[4]:.1f}"
    )


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def report(data: dict, label: str) -> None:
    T = len(data["selected"])
    print(f"\n{'=' * 70}")
    print(f"  {label}  —  {T} tracked frames")
    print(f"{'=' * 70}")

    # 1. Candidate availability (H matrix is non-NaN)
    print("\n── Candidate availability (H not NaN) ──────────────────────────────")
    for p in PATHS:
        key = p
        if key not in data:
            print(f"  {p:12s}  NOT IN FILE")
            continue
        H = data[key]  # (T, 3, 3)
        available = ~np.isnan(H[:, 0, 0])
        print(f"  {p:12s}  {_pct(available.sum(), T)}")

    # 2. Score finiteness
    print("\n── Score finiteness (rmse is finite) ────────────────────────────────")
    finite_masks = {}
    for p in PATHS:
        key = f"{p}_rmse"
        if key not in data:
            print(f"  {p:12s}  NOT IN FILE")
            continue
        rmse = data[key]
        finite = np.isfinite(rmse)
        finite_masks[p] = finite
        print(f"  {p:12s}  {_pct(finite.sum(), T)}")

    # 3. Score distributions (finite values only)
    print("\n── Predicted RMSE distribution (finite values only) ─────────────────")
    for p in PATHS:
        if p not in finite_masks:
            continue
        rmse = data[f"{p}_rmse"]
        finite_vals = rmse[finite_masks[p]]
        print(f"  {p:12s}  n={len(finite_vals):5d}  {_quantile_str(finite_vals)}")

    # 4. Selection distribution
    print("\n── Selection distribution ───────────────────────────────────────────")
    labels, counts = np.unique(data["selected"], return_counts=True)
    for lbl, cnt in sorted(zip(labels, counts), key=lambda x: -x[1]):
        print(f"  {lbl:14s}  {_pct(cnt, T)}")

    # 4b. LoMa inlier counts + SAM3 corrections (online LoMa paths)
    if "n_inliers" in data:
        valid = data["n_inliers"] >= 0
        if valid.any():
            print("\n── LoMa inlier count (frames with N/A excluded) ─────────────────────")
            print(
                f"  n={int(valid.sum()):5d}  "
                f"{_quantile_str(data['n_inliers'][valid].astype(np.float64))}"
            )
    if "corrected" in data:
        corrected = data["corrected"].astype(bool)
        print("\n── SAM3 corrections fired ───────────────────────────────────────────")
        print(f"  {_pct(int(corrected.sum()), T)}")
        if corrected.any():
            idx = np.where(corrected)[0]
            frames = data["frame_idx"][idx] if "frame_idx" in data else idx
            preview = ", ".join(str(int(f)) for f in frames[:20])
            tail = " …" if len(frames) > 20 else ""
            print(f"  frames: {preview}{tail}")

    # 5. Head-to-head: when both flow_H and mask_H are finite, which scores lower?
    if "flow_H" in finite_masks and "mask_H" in finite_masks:
        both = finite_masks["flow_H"] & finite_masks["mask_H"]
        n_both = both.sum()
        if n_both > 0:
            flow_r = data["flow_H_rmse"][both]
            mask_r = data["mask_H_rmse"][both]
            mask_wins = (mask_r < flow_r).sum()
            flow_wins = (flow_r <= mask_r).sum()
            print(
                f"\n── flow_H vs mask_H (both finite, n={n_both}) ───────────────────────"
            )
            print(f"  flow_H scores lower: {_pct(flow_wins, n_both)}")
            print(f"  mask_H scores lower: {_pct(mask_wins, n_both)}")
            diff = flow_r - mask_r
            print(f"  flow_H_rmse − mask_H_rmse  {_quantile_str(diff)}")

    # 6. Redetection triggered (redetection_flow_H is non-NaN)
    if "redetection_flow_H" in data:
        triggered = ~np.isnan(data["redetection_flow_H"][:, 0, 0])
        print(
            "\n── Re-detection triggered (redetection_flow_H computed) ───────────────"
        )
        print(f"  {_pct(triggered.sum(), T)}")
        if triggered.sum() > 0 and "redetection_flow_H_rmse" in data:
            fm_finite = np.isfinite(data["redetection_flow_H_rmse"][triggered])
            print(
                "  of those, redetection_flow_H scored finite:  "
                f"{_pct(fm_finite.sum(), triggered.sum())}"
            )

    # 7. Suspiciously constant scores (possible model/preprocessing bug)
    print("\n── Score variance check (std of finite rmse values) ─────────────────")
    for p in PATHS:
        if p not in finite_masks:
            continue
        finite_vals = data[f"{p}_rmse"][finite_masks[p]]
        if len(finite_vals) < 2:
            print(f"  {p:12s}  too few finite values to check")
            continue
        std = np.std(finite_vals)
        flag = "  ← nearly constant, check preprocessing!" if std < 0.5 else ""
        print(f"  {p:12s}  std={std:.3f}{flag}")

    # 8. NaN-in-H but finite-score inconsistency
    print("\n── Consistency: finite score but NaN H (should not happen) ──────────")
    for p in PATHS:
        H_key = f"{p}_H"
        r_key = f"{p}_rmse"
        if H_key not in data or r_key not in data:
            continue
        H_nan = np.isnan(data[H_key][:, 0, 0])
        r_finite = np.isfinite(data[r_key])
        bad = H_nan & r_finite
        flag = "  ← BUG" if bad.sum() > 0 else ""
        print(f"  {p:12s}  {_pct(bad.sum(), T)}{flag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check NN scorer debug npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="One or more *_debug.npz files or directories containing them.",
    )
    args = parser.parse_args()

    for path in args.paths:
        files = find_debug_files(path)
        if not files:
            continue
        data = load_all(files)
        label = str(path)
        report(data, label)

    print()


if __name__ == "__main__":
    main()
