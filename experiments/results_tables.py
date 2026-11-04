#!/usr/bin/env python3
"""Paper result tables from completed runs: headline, POT conditions, mask quality.

Reads the ``eval_results.json`` a finished grid already wrote and formats the
three tables a paper needs, so no tracking is repeated:

* **cross-dataset** — the same per-sequence mean on all six benchmarks plus the
  worst column, because the claim being made is consistency, not a peak.
* **headline** — per-sequence mean P@5/P@10/P@15 on both benchmarks, plus the
  best-rotation gap. That gap (``br_p@15 - p@15``) is a free corner-labelling
  alias detector and should be quoted next to any redetection method.
* **attributes** — PlanarTrack's own 8 challenging factors, because POT's
  condition table saturates and cannot show robustness where it matters.
* **conditions** — POT-210 is 30 objects x 7 imaging conditions, and the dataset
  ships the membership as split files, so the per-condition breakdown is read
  from ``POT_BASE/splits/*.txt`` rather than assumed from the name suffix.
* **mask-quality** — accuracy against the segmenter's own mask IoU, which bounds
  what any mask-based estimator can do. Needs the ``maskh_bench`` manifest for
  the per-frame covariates, and reports only the sequences both cover.

Pure analysis: no GPU, no dataset frames, no masks. It still imports cv2 through
``flatsam2.datasets``, so send it to the cpu partition.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/results_tables.py \\
        --configs loma_samh_decoupled_rerun corner_fusion_register
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.path_config import paths

_LABELS = {
    "pti": "PlanarTrackInitfix", "pot": "POT-210",
    "poic": "POIC", "tmt": "TMT", "ucsb": "UCSB", "mpot": "MPOT-3K",
}
# The two main benchmarks, then the generalisation set. A method is only as
# strong as its worst column here, so the cross-dataset table quotes both the
# mean and the alias gap on every one of them.
_CROSS_DATASETS = ("poic", "tmt", "ucsb", "mpot")
# POT-210's own split files define the condition membership; the suffix mapping
# below is derived from them, not assumed, and is re-read at runtime.
_POT_CONDITIONS = (
    "scale", "rotation", "perspective", "blur", "occlusion", "out-of-view", "unconstrained",
)
_BENCH_DIR = paths.OUTPUT_DIRECTORY / "experiments" / "maskh_bench"
# PlanarTrack ships one 8-flag line per sequence in challenging_factor/. The
# names and their order are the official toolkit's, copied from
# third_party_examples/evaluation.py rather than assumed.
_PT_ATTRIBUTES = (
    "occlusion", "blur", "rotation", "scale",
    "perspective", "out-of-view", "low-res", "bg-clutter",
)


def _load(config: str, dataset: str) -> Optional[Dict[str, Dict]]:
    """Per-sequence metrics of one run, keyed by sequence name."""
    path = paths.OUTPUT_DIRECTORY / config / _LABELS[dataset] / "eval_results.json"
    if not path.exists():
        return None
    return {r["sequence"]: r for r in json.loads(path.read_text())["sequences"]}


def _mean(rows: List[Dict], key: str) -> float:
    vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")


def table_headline(configs: List[str]) -> None:
    """Per-sequence mean precision on both benchmarks, with the alias gap."""
    print("\n=== headline: per-sequence mean precision, full grids " + "=" * 22)
    header = f"{'config':<38}"
    for ds in ("pti", "pot"):
        header += f"{ds.upper() + ' P@5':>10}{'P@10':>8}{'P@15':>8}{'br-gap':>8}"
    print(header)
    for config in configs:
        line = f"{config:<38}"
        for ds in ("pti", "pot"):
            seqs = _load(config, ds)
            if seqs is None:
                line += f"{'-':>10}{'-':>8}{'-':>8}{'-':>8}"
                continue
            rows = list(seqs.values())
            gap = _mean(rows, "br_p@15") - _mean(rows, "p@15")
            line += (f"{_mean(rows, 'p@5'):10.2f}{_mean(rows, 'p@10'):8.2f}"
                     f"{_mean(rows, 'p@15'):8.2f}{gap:8.2f}")
        print(line)


def table_cross_dataset(configs: List[str], metric: str = "p@15") -> None:
    """Per-sequence mean of one metric on every benchmark, main and generalisation.

    Args:
        configs: Result directory stems to compare, in the order to print them.
        metric: Per-sequence key from ``eval_results.json`` (e.g. ``p@15``).
    """
    datasets = ("pti", "pot") + _CROSS_DATASETS
    print(f"\n=== cross-dataset: per-sequence mean {metric} " + "=" * 30)
    header = f"{'config':<38}"
    for ds in datasets:
        header += f"{_LABELS[ds][:9]:>11}"
    header += f"{'worst':>9}"
    print(header)
    for config in configs:
        line = f"{config:<38}"
        values = []
        for ds in datasets:
            seqs = _load(config, ds)
            if seqs is None:
                line += f"{'-':>11}"
                continue
            value = _mean(list(seqs.values()), metric)
            values.append(value)
            line += f"{value:11.2f}"
        # The weakest benchmark, printed because that is the number a reviewer
        # quotes back; blank when a run is still missing somewhere.
        worst = min(values) if len(values) == len(datasets) else float("nan")
        line += f"{worst:9.2f}" if np.isfinite(worst) else f"{'-':>9}"
        print(line)


def table_cross_dataset_gap(configs: List[str]) -> None:
    """Best-rotation minus plain P@15 on every benchmark: the corner-naming gap."""
    datasets = ("pti", "pot") + _CROSS_DATASETS
    print("\n=== cross-dataset: br-gap (br_p@15 - p@15), lower is better " + "=" * 12)
    header = f"{'config':<38}"
    for ds in datasets:
        header += f"{_LABELS[ds][:9]:>11}"
    print(header)
    for config in configs:
        line = f"{config:<38}"
        for ds in datasets:
            seqs = _load(config, ds)
            if seqs is None:
                line += f"{'-':>11}"
                continue
            rows = list(seqs.values())
            line += f"{_mean(rows, 'br_p@15') - _mean(rows, 'p@15'):11.2f}"
        print(line)


def _pt_attribute_membership(seq_name: str) -> Optional[np.ndarray]:
    """The 8 challenging-factor flags of one PlanarTrack sequence, or None."""
    path = (
        paths.PT_BASE / "challenging_factor" / f"{seq_name}_challenging_factor.txt"
    )
    if not path.exists():
        return None
    return np.loadtxt(path, delimiter=",") > 0


def table_pti_attributes(configs: List[str], metric: str = "p@15") -> None:
    """PTi broken down by the dataset's own challenging factors.

    POT-210's condition table saturates — occlusion is 100.00 for every serious
    tracker — so it cannot show robustness. PlanarTrack is the hard benchmark and
    ships per-sequence attribute flags, which makes the same breakdown
    informative there. Occlusion in particular ranks above aggregate P@15 for
    this project, so it needs its own column rather than being averaged away.

    Args:
        configs: Result directory stems to compare.
        metric: Per-sequence key from ``eval_results.json``.
    """
    print(f"\n=== PlanarTrackInitfix by challenging factor, {metric} " + "=" * 18)
    members: Dict[str, List[str]] = {a: [] for a in _PT_ATTRIBUTES}
    counted = 0
    reference = _load(configs[0], "pti") if configs else None
    if not reference:
        print("  (no PTi results on disk)")
        return
    for seq_name in reference:
        flags = _pt_attribute_membership(seq_name)
        if flags is None:
            continue
        counted += 1
        for attribute, present in zip(_PT_ATTRIBUTES, flags):
            if present:
                members[attribute].append(seq_name)
    if not counted:
        print("  (no challenging_factor annotations found)")
        return

    header = f"{'config':<38}"
    for attribute in _PT_ATTRIBUTES:
        header += f"{attribute[:9]:>11}"
    header += f"{'all':>11}"
    print(header)
    print(f"{'(sequences)':<38}" + "".join(
        f"{len(members[a]):>11}" for a in _PT_ATTRIBUTES
    ) + f"{counted:>11}")
    for config in configs:
        seqs = _load(config, "pti")
        if seqs is None:
            continue
        line = f"{config:<38}"
        for attribute in _PT_ATTRIBUTES:
            rows = [seqs[s] for s in members[attribute] if s in seqs]
            line += f"{_mean(rows, metric):11.2f}" if rows else f"{'-':>11}"
        line += f"{_mean(list(seqs.values()), metric):11.2f}"
        print(line)


def table_pot_conditions(configs: List[str], metric: str) -> None:
    """POT-210 broken down by its seven imaging conditions."""
    splits_dir = paths.POT_BASE / "splits"
    membership = {}
    for condition in _POT_CONDITIONS:
        path = splits_dir / f"{condition}.txt"
        if path.exists():
            membership[condition] = [s.strip() for s in path.read_text().split() if s.strip()]
    if not membership:
        print("\n(POT split files not found — skipping the condition table)")
        return

    print(f"\n=== POT-210 by condition, {metric} " + "=" * 34)
    print(f"{'config':<38}" + "".join(f"{c[:9]:>11}" for c in membership) + f"{'all':>11}")
    for config in configs:
        seqs = _load(config, "pot")
        if seqs is None:
            continue
        line = f"{config:<38}"
        for names in membership.values():
            rows = [seqs[n] for n in names if n in seqs]
            line += f"{_mean(rows, metric):11.2f}" if rows else f"{'-':>11}"
        line += f"{_mean(list(seqs.values()), metric):11.2f}"
        print(line)


def table_mask_quality(configs: List[str], dataset: str, manifest: str) -> None:
    """Accuracy against the segmenter's mask quality — the bound on any mask method.

    Sequence-level: a sequence is placed in a bin by its median mask IoU in the
    benchmark manifest, then the run's own per-sequence P@15 is averaged per bin.
    """
    path = _BENCH_DIR / manifest
    if not path.exists():
        print(f"\n(no benchmark manifest at {path} — skipping the mask-quality table)")
        return
    per_seq: Dict[str, List[float]] = {}
    for case in json.loads(path.read_text())["cases"]:
        if case["dataset"] == dataset:
            per_seq.setdefault(case["seq"], []).append(case["iou"])
    median_iou = {s: float(np.median(v)) for s, v in per_seq.items()}

    edges = [0.0, 0.5, 0.8, 0.9, 0.95, 1.01]
    names = ["<.50", ".50-.80", ".80-.90", ".90-.95", ">.95"]
    print(f"\n=== {_LABELS[dataset]} P@15 by SAM3 mask IoU (per sequence) " + "=" * 12)
    counts = [
        sum(1 for v in median_iou.values() if edges[i] <= v < edges[i + 1])
        for i in range(len(names))
    ]
    print(f"{'config':<38}" + "".join(f"{n:>10}" for n in names))
    print(f"{'(sequences)':<38}" + "".join(f"{c:>10}" for c in counts))
    for config in configs:
        seqs = _load(config, dataset)
        if seqs is None:
            continue
        line = f"{config:<38}"
        for i in range(len(names)):
            rows = [
                seqs[s] for s, v in median_iou.items()
                if s in seqs and edges[i] <= v < edges[i + 1]
            ]
            line += f"{_mean(rows, 'p@15'):10.2f}" if rows else f"{'-':>10}"
        print(line)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="+",
        default=[
            "hough_symmetry_only",
            "loma_r_allobj_inflate10",
            "loma_samh_decoupled_rerun",
            "loma_samh_decoupled_alignh_pure",
            "corner_fusion_alignh_calibrated",
            "corner_fusion_alignh_redetect",
            "corner_fusion_register",
            "corner_fusion_register_kp",
            "corner_fusion_register_noapp",
        ],
    )
    parser.add_argument("--manifest", default="testset_v1.json")
    args = parser.parse_args(argv)

    available = [
        c for c in args.configs
        if any(_load(c, ds) is not None for ds in _LABELS)
    ]
    missing = [c for c in args.configs if c not in available]
    if missing:
        print(f"not on disk yet: {', '.join(missing)}")

    table_headline(available)
    table_cross_dataset(available, 'p@15')
    table_cross_dataset(available, 'p@5')
    table_cross_dataset_gap(available)
    table_pti_attributes(available, 'p@15')
    table_pot_conditions(available, "p@15")
    table_pot_conditions(available, "p@5")
    table_mask_quality(available, "pti", args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
