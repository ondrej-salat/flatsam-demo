#!/usr/bin/env python3
"""hpc_readout.py — one-off cluster read-out for the alignh / selector / gallery threads.

Run on the HPC from the repo root (or anywhere; paths are resolved relative to
this file). Reuses scripts/hpc_status.py for completeness and reads the
eval_results.json files directly for the P@15 decision numbers. Writes
everything to a single file for hand-off.

Usage:
    python scripts/hpc_readout.py                 # -> /tmp/hpc_readout.txt
    python scripts/hpc_readout.py OUTFILE
"""

import contextlib
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_MANIFEST = _REPO / "experiments" / "pipeline.yml"

sys.path.insert(0, str(_HERE))
import hpc_status  # noqa: E402  (repo-local; same directory)

# Pipelines whose completeness we care about right now.
PIPELINES = ["loma_samh_online", "selector_study", "loma_breakdown_gallery"]

# Configs to compare at the decision metric (alignh prototype vs its baseline).
METRIC_CONFIGS = ["loma_samh_decoupled_rerun", "loma_samh_decoupled_alignh_pure"]
METRIC_LABELS = ["PlanarTrackInitfix", "POT-210"]


# HoughConfig.align_max_iters default; frames hitting it never converged.
ALIGN_MAX_ITERS = 60

# Configs whose mask-H branch usage is worth auditing (align-method runs).
BRANCH_USAGE_CONFIGS = [
    "loma_samh_decoupled_alignh",
    "loma_samh_decoupled_alignh_pure",
]

# Configs using algorithm='corner_fusion' — per-corner fusion mode usage.
FUSION_CONFIGS = [
    "corner_fusion_alignh",
]


def _branch_usage(cfg: str, label: str, base: Path):
    """Count mask-H estimator usage across a config's per-sequence debug npz.

    Args:
        cfg: config stem (results subdirectory name).
        label: dataset label subdirectory.
        base: results root.

    Returns:
        Tuple ``(counts, residuals, iters, n_seqs)`` — frames per
        ``maskh_source`` value, all finite align residuals (px), the GN
        iteration counts (cost proxy), and the number of sequences that carried
        the telemetry. Empty when the run predates it.
    """
    import numpy as np

    counts: dict[str, int] = {}
    residuals: list[float] = []
    iters: list[int] = []
    n_seqs = 0
    seq_dir = base / cfg / label
    if not seq_dir.is_dir():
        return counts, residuals, iters, n_seqs
    for npz_path in sorted(seq_dir.glob("*_debug.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "maskh_source" not in data:
                continue
            n_seqs += 1
            for src in data["maskh_source"]:
                key = str(src)
                counts[key] = counts.get(key, 0) + 1
            if "maskh_align_residual_px" in data:
                res = data["maskh_align_residual_px"]
                residuals.extend(float(v) for v in res[np.isfinite(res)])
            if "maskh_align_iters" in data:
                it = data["maskh_align_iters"]
                iters.extend(int(v) for v in it[it >= 0])
    return counts, residuals, iters, n_seqs


def _fusion_usage(cfg: str, label: str, base: Path):
    """Count per-corner fusion-mode usage across a config's debug npz files.

    Args:
        cfg: config stem (results subdirectory name).
        label: dataset label subdirectory.
        base: results root.

    Returns:
        Tuple ``(mode_counts, disagreements, n_seqs)`` — per-corner mode
        counts pooled over every frame, all finite ``fusion_max_disagreement``
        values, and the number of sequences carrying the telemetry.
    """
    import numpy as np

    mode_counts: dict[str, int] = {}
    disagreements: list[float] = []
    n_seqs = 0
    seq_dir = base / cfg / label
    if not seq_dir.is_dir():
        return mode_counts, disagreements, n_seqs
    for npz_path in sorted(seq_dir.glob("*_debug.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "fusion_modes" not in data:
                continue
            n_seqs += 1
            for row in data["fusion_modes"]:
                for mode in str(row).split(","):
                    if mode:
                        mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if "fusion_max_disagreement" in data:
                d = data["fusion_max_disagreement"]
                disagreements.extend(float(v) for v in d[np.isfinite(d)])
    return mode_counts, disagreements, n_seqs


def _mean(cfg: str, label: str, key: str, base: Path):
    p = base / cfg / label / "eval_results.json"
    if not p.exists():
        return None
    seqs = json.load(open(p))["sequences"]
    vals = [s[key] for s in seqs if key in s]
    return st.mean(vals) if vals else None


def _per_seq(cfg: str, label: str, key: str, base: Path) -> dict:
    """Return {sequence_name: value} for one config/dataset/metric."""
    p = base / cfg / label / "eval_results.json"
    if not p.exists():
        return {}
    seqs = json.load(open(p))["sequences"]
    return {s["sequence"]: s[key] for s in seqs if key in s}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/hpc_readout.txt")
    manifest = hpc_status._load_yaml(_MANIFEST)
    base = Path(manifest["path_variables"]["OUTPUT_DIRECTORY"])

    with open(out, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        print(f"HPC read-out — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Manifest: {_MANIFEST}")
        print(f"OUTPUT_DIRECTORY: {base}")

        # --- Part 1: completeness for the relevant pipelines -------------------
        by_name = {p["name"]: p for p in manifest.get("pipelines", [])}
        for name in PIPELINES:
            pl = by_name.get(name)
            if pl is None:
                print(f"\n(!) pipeline {name!r} not in manifest")
                continue
            hpc_status._check_pipeline(pl, manifest)

        # --- Part 2: P@15 decision numbers (alignh vs baseline) ---------------
        print(f"\n{'=' * 66}")
        print("  DECISION METRICS — per-seq mean (p@5 / p@15 / best-rot p@15)")
        print(f"{'=' * 66}")
        for label in METRIC_LABELS:
            print(f"\n  == {label} ==")
            for cfg in METRIC_CONFIGS:
                p5 = _mean(cfg, label, "p@5", base)
                p15 = _mean(cfg, label, "p@15", base)
                br = _mean(cfg, label, "br_p@15", base)
                if p15 is None:
                    print(f"    {cfg:34s}  (no eval_results.json)")
                    continue
                r = lambda x: "  n/a" if x is None else f"{x:6.2f}"
                print(f"    {cfg:34s}  p@5={r(p5)}  p@15={r(p15)}  br_p@15={r(br)}")

        # --- Part 3: per-sequence alignh - baseline diff (p@15) ---------------
        base_cfg, test_cfg = METRIC_CONFIGS  # rerun (baseline), alignh
        print(f"\n{'=' * 66}")
        print(f"  PER-SEQUENCE p@15 DIFF: {test_cfg}")
        print(f"                    minus  {base_cfg}")
        print(f"{'=' * 66}")
        for label in METRIC_LABELS:
            b = _per_seq(base_cfg, label, "p@15", base)
            t = _per_seq(test_cfg, label, "p@15", base)
            common = sorted(set(b) & set(t))
            if not common:
                print(f"\n  == {label} ==  (no overlapping eval_results.json)")
                continue
            diffs = {s: t[s] - b[s] for s in common}
            deltas = list(diffs.values())
            n = len(deltas)
            worse = [d for d in deltas if d < -0.01]
            better = [d for d in deltas if d > 0.01]
            same = n - len(worse) - len(better)
            net = st.mean(deltas)

            print(f"\n  == {label} ==  ({n} seqs)")
            print(f"    net per-seq mean Δp@15 : {net:+.2f} pp")
            print(f"    regressed : {len(worse):3d}   improved : {len(better):3d}   "
                  f"unchanged : {same:3d}")
            if worse:
                sum_worse = sum(worse)
                print(f"    total negative mass : {sum_worse:+.1f} pp over {len(worse)} seqs "
                      f"(mean {sum_worse/len(worse):+.2f} each)")
                # concentration: share of negative mass in the worst 5 sequences
                top5 = sorted(worse)[:5]
                if len(worse) > 5:
                    print(f"    worst-5 share of neg mass : "
                          f"{100*sum(top5)/sum_worse:.0f}%  → "
                          f"{'CONCENTRATED' if sum(top5) < 0.5*sum_worse else 'BROAD'}")
            ranked = sorted(diffs.items(), key=lambda kv: kv[1])
            print("    biggest regressions:")
            for s, d in ranked[:12]:
                print(f"      {d:+7.2f}   {s}")
            print("    biggest improvements:")
            for s, d in reversed(ranked[-6:]):
                print(f"      {d:+7.2f}   {s}")

        # --- Part 4: mask-H branch usage (align vs Hough quad) ----------------
        # Validity gate for any align-method run: a config that nominally uses
        # mask alignment but falls back to the Hough quad on most frames says
        # nothing about mask alignment. Read this BEFORE reading Part 2.
        print(f"\n{'=' * 66}")
        print("  MASK-H BRANCH USAGE (from {seq}_debug.npz maskh_source)")
        print(f"{'=' * 66}")
        for cfg in BRANCH_USAGE_CONFIGS:
            for label in METRIC_LABELS:
                counts, residuals, iters, n_seqs = _branch_usage(cfg, label, base)
                total = sum(counts.values())
                print(f"\n  == {cfg} / {label} ==")
                if not total:
                    print("    (no debug npz with maskh_source — pre-telemetry run)")
                    continue
                print(f"    {n_seqs} seqs, {total} frames")
                for src in ("align", "hough", "none"):
                    n = counts.get(src, 0)
                    print(f"      {src:6s} : {n:7d}  ({100.0 * n / total:5.1f}%)")
                other = {k: v for k, v in counts.items() if k not in ("align", "hough", "none")}
                for src, n in sorted(other.items()):
                    print(f"      {src:6s} : {n:7d}  ({100.0 * n / total:5.1f}%)")
                if residuals:
                    residuals.sort()
                    mid = residuals[len(residuals) // 2]
                    p90 = residuals[int(0.9 * (len(residuals) - 1))]
                    print(f"    align residual px: median {mid:.2f}  p90 {p90:.2f}  "
                          f"(n={len(residuals)})")
                if iters:
                    iters.sort()
                    at_cap = sum(1 for v in iters if v >= ALIGN_MAX_ITERS)
                    print(f"    align GN iters: median {iters[len(iters) // 2]}  "
                          f"p90 {iters[int(0.9 * (len(iters) - 1))]}  "
                          f"max {iters[-1]}  at-cap {100.0 * at_cap / len(iters):.1f}%")
                hough_share = 100.0 * counts.get("hough", 0) / total
                if hough_share > 5.0:
                    print(f"    (!) Hough quad ran on {hough_share:.1f}% of frames — "
                          f"this run is NOT quad-free")

        # --- Part 5: corner-fusion mode usage (algorithm='corner_fusion') -----
        print(f"\n{'=' * 66}")
        print("  CORNER-FUSION MODE USAGE (from {seq}_debug.npz fusion_modes)")
        print(f"{'=' * 66}")
        for cfg in FUSION_CONFIGS:
            for label in METRIC_LABELS:
                mode_counts, disagreements, n_seqs = _fusion_usage(cfg, label, base)
                total = sum(mode_counts.values())
                print(f"\n  == {cfg} / {label} ==")
                if not total:
                    print("    (no debug npz with fusion_modes — not a corner_fusion run)")
                    continue
                print(f"    {n_seqs} seqs, {total} corner-decisions")
                for mode, n in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
                    print(f"      {mode:16s} : {n:8d}  ({100.0 * n / total:5.1f}%)")
                if disagreements:
                    disagreements.sort()
                    mid = disagreements[len(disagreements) // 2]
                    p90 = disagreements[int(0.9 * (len(disagreements) - 1))]
                    print(f"    max_disagreement (chi2_2): median {mid:.2f}  p90 {p90:.2f}  "
                          f"(n={len(disagreements)})")

        print(f"\n{'=' * 66}")

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
