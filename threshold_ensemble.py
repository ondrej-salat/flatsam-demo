#!/usr/bin/env python3
"""Threshold-based ensemble: keypoint tracker + fallback tracker.

Uses per-frame RANSAC inlier count (or ratio) from a keypoint-based tracker
to decide per-frame which tracker result to use.  Sweeps a range of thresholds,
reports performance at each, and writes predictions for the best (or
user-specified) threshold in standard evaluation format.

The keypoint tracker must have saved ``*_stats.json`` files alongside its
``.txt`` result files (produced by ``experiments/loma_experiment_v2.py``).

Output layout (mirrors oracle_ensemble.py)::

    <output-dir>/<ensemble-name>/<dataset-label>/
        <seq>.txt                    -- ensemble predicted corners
        ensemble_meta/<seq>.json     -- per-sequence selection metadata
        ensemble_manifest.json
        ensemble_report.html

Usage examples::

    # Single dataset:
    python threshold_ensemble.py pot \\
        --kp-tracker loma_r_2000obj_infl10 \\
        --other-tracker hough_symmetry_only \\
        --ensemble-name loma_hough_thr

    # Both PTI (test, primary) and POT-210 — shared threshold grid + cross-dataset plot:
    python threshold_ensemble.py both \\
        --kp-tracker loma_r_2000obj_infl10 \\
        --other-tracker hough_symmetry_only \\
        --ensemble-name loma_hough_thr

    # Specify threshold explicitly:
    python threshold_ensemble.py both \\
        --kp-tracker loma_r_2000obj_infl10 \\
        --other-tracker hough_symmetry_only \\
        --ensemble-name loma_hough_thr50 \\
        --threshold 50
"""

import argparse
import json
import traceback
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

import numpy as np
from einops import rearrange

from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_load_gt,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.metrics import aggregate_results, alignment_error, evaluate_sequence
from flatsam2.path_config import paths
from flatsam2.report_utils import get_plotly_script_tag


THRESHOLD_METRIC_CHOICES = (
    "inliers_total",
    "inliers_object",
    "inlier_ratio",
    "hull_coverage",
    "peeled_hull_coverage",
)

SEGMENTS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("all", 1, None),
    ("1:125", 1, 126),
    ("126:250", 126, 251),
    ("251:375", 251, 376),
    ("376+", 376, None),
)


# ---------------------------------------------------------------------------
# Data loading / saving
# ---------------------------------------------------------------------------


def load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    """Load predicted corners from a ``.txt`` result file.

    Args:
        results_dir: Tracker results directory.
        seq_name: Sequence name.

    Returns:
        Predicted corners ``(frames, 2, 4)``.
    """
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise ValueError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(
        pred_flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2
    )


def save_corners(corners: np.ndarray, output_path: Path) -> None:
    """Save ``(frames, 2, 4)`` corners in standard 8-floats-per-line format.

    Args:
        corners: Predicted corners ``(frames, 2, 4)``.
        output_path: Output ``.txt`` path.
    """
    flat = rearrange(
        corners, "frames xy corners -> frames (corners xy)", corners=4, xy=2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, flat, fmt="%.6f")


def load_kp_frame_stats(stats_path: Path) -> dict[int, dict]:
    """Load per-frame stats from a loma_v2 ``*_stats.json`` file.

    Also merges the sibling ``*_spatial_stats.json`` (produced by
    ``compute_spatial_stats.py``) when present, adding spatial coverage fields
    to each frame record.

    Args:
        stats_path: Path to the ``*_stats.json`` file.

    Returns:
        Mapping ``frame_idx -> stats_record``.  Empty dict if file is missing
        or unreadable.
    """
    if not stats_path.exists():
        return {}
    try:
        data = json.loads(stats_path.read_text())
        frame_map: dict[int, dict] = {int(r["frame_idx"]): dict(r) for r in data.get("frames", [])}
    except Exception:
        return {}

    spatial_path = stats_path.with_name(stats_path.stem + "_spatial_stats.json")
    if spatial_path.exists():
        try:
            spatial_data = json.loads(spatial_path.read_text())
            for r in spatial_data.get("frames", []):
                fidx = int(r["frame_idx"])
                if fidx in frame_map:
                    frame_map[fidx].update(r)
                else:
                    frame_map[fidx] = dict(r)
        except Exception:
            pass

    return frame_map


def build_metric_array(
    frame_stats: dict[int, dict],
    num_frames: int,
    metric: str,
) -> np.ndarray:
    """Build a ``(num_frames,)`` float64 array of per-frame metric values.

    Frame 0 (init) is always 0.  Frames missing from *frame_stats* are 0.

    Args:
        frame_stats: ``frame_idx -> stats_record`` mapping.
        num_frames: Total frame count.
        metric: Key into each stats record (e.g. ``"inliers_total"``).

    Returns:
        Float64 array of shape ``(num_frames,)``.
    """
    vals = np.zeros(num_frames, dtype=np.float64)
    for fidx, record in frame_stats.items():
        if 0 < fidx < num_frames:
            val = record.get(metric)
            if val is not None:
                vals[fidx] = float(val)
    return vals


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def _evaluated_frame_idx(valid: np.ndarray) -> np.ndarray:
    """Return frame indices used for evaluation (valid, excluding frame 0).

    Args:
        valid: Valid-frame mask.

    Returns:
        1-D integer array of frame indices.
    """
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    return np.where(eval_mask)[0]


def _compute_per_frame_errors(
    gt: np.ndarray,
    valid: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-evaluated-frame alignment errors.

    Invalid predictions become ``inf`` so P@k counts them as failures.

    Args:
        gt: GT corners ``(frames, 2, 4)``.
        valid: Valid-frame mask.
        pred: Predicted corners ``(frames, 2, 4)``.

    Returns:
        Tuple ``(eval_idx, errors)``.
    """
    eval_idx = _evaluated_frame_idx(valid)
    if len(eval_idx) == 0:
        return eval_idx, np.array([], dtype=np.float64)

    gt_eval = gt[eval_idx]
    pred_eval = pred[eval_idx]
    errors = np.full(len(eval_idx), np.inf, dtype=np.float64)
    finite = np.all(np.isfinite(gt_eval), axis=(1, 2)) & np.all(
        np.isfinite(pred_eval), axis=(1, 2)
    )
    if finite.any():
        errors[finite] = alignment_error(gt_eval[finite], pred_eval[finite])
    return eval_idx, errors


def _precision_at(errors: np.ndarray, threshold: float) -> float:
    """P@threshold as a percentage.

    Args:
        errors: Error vector (``inf`` counts as failure).
        threshold: Threshold in pixels.

    Returns:
        Precision in percent, or ``nan`` if empty.
    """
    if len(errors) == 0:
        return float("nan")
    return 100.0 * float(np.mean(errors <= threshold))


# ---------------------------------------------------------------------------
# Per-sequence data container
# ---------------------------------------------------------------------------


@dataclass
class SequenceData:
    """Per-sequence data used for threshold sweep.

    Attributes:
        name: Sequence name.
        num_frames: Total frame count.
        eval_idx: ``(N,)`` evaluated frame indices.
        kp_errors: ``(N,)`` per-evaluated-frame kp tracker errors (inf = invalid).
        other_errors: ``(N,)`` per-evaluated-frame other tracker errors.
        metric_eval: ``(N,)`` primary kp metric values at evaluated frames.
        metric_all: ``(num_frames,)`` primary kp metric values at all frames.
        metric2_eval: ``(N,)`` secondary metric values at evaluated frames (2-D sweep).
        metric2_all: ``(num_frames,)`` secondary metric values at all frames.
    """

    name: str
    num_frames: int
    eval_idx: np.ndarray
    kp_errors: np.ndarray
    other_errors: np.ndarray
    metric_eval: np.ndarray
    metric_all: np.ndarray
    metric2_eval: Optional[np.ndarray] = None
    metric2_all: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def _build_threshold_grid(
    sequences: list[SequenceData],
    n_extra_steps: int = 200,
) -> np.ndarray:
    """Build a dense threshold grid from unique metric values.

    Adds a sentinel below min (→ always use kp) and above max (→ always use
    other).  When the metric range is small (integer inlier counts) the unique
    values themselves are used; for larger ranges a uniform grid is also added.

    Args:
        sequences: Sequence data list.
        n_extra_steps: Max extra uniform steps to insert between min and max.

    Returns:
        Sorted, deduplicated threshold array.
    """
    if not sequences:
        return np.array([0.0])

    all_vals = np.concatenate([seq.metric_eval for seq in sequences])
    unique = np.unique(all_vals)

    grid = [unique]
    if len(unique) >= 2:
        lo, hi = float(unique[0]), float(unique[-1])
        step = (hi - lo) / n_extra_steps
        if step > 0:
            extra = np.arange(lo, hi + step * 0.5, step)
            grid.append(extra)

    merged = np.unique(np.concatenate(grid))

    # Sentinels
    sentinel_low = float(merged[0]) - 1.0
    sentinel_high = float(merged[-1]) + 1.0
    return np.concatenate([[sentinel_low], merged, [sentinel_high]])


def sweep_thresholds(
    sequences: list[SequenceData],
    thresholds: np.ndarray,
    segments: tuple = SEGMENTS,
) -> dict:
    """Compute ensemble metrics for every threshold.

    For each threshold *t*: on each evaluated frame, use the kp tracker result
    when ``metric_eval[frame] >= t``, otherwise use the other tracker.

    Args:
        sequences: Sequence data list.
        thresholds: Threshold values to sweep.
        segments: Frame-range segments for segmented precision.

    Returns:
        Dict with keys ``thresholds``, ``p5``, ``p15``, ``kp_frac``,
        and ``seg_p5`` / ``seg_p15`` (per-segment dicts).
    """
    seg_labels = [label for label, _, _ in segments]

    result: dict = {
        "thresholds": [],
        "p5": [],
        "p15": [],
        "kp_frac": [],
        **{f"seg_p5_{lbl}": [] for lbl in seg_labels},
        **{f"seg_p15_{lbl}": [] for lbl in seg_labels},
    }

    for t in thresholds:
        all_errors: list[np.ndarray] = []
        seg_errors: dict[str, list[np.ndarray]] = {lbl: [] for lbl in seg_labels}
        total_eval = 0
        kp_eval = 0

        for seq in sequences:
            use_kp = seq.metric_eval >= t
            errs = np.where(use_kp, seq.kp_errors, seq.other_errors)
            all_errors.append(errs)
            total_eval += len(use_kp)
            kp_eval += int(use_kp.sum())

            for lbl, start, stop in segments:
                mask = seq.eval_idx >= start
                if stop is not None:
                    mask &= seq.eval_idx < stop
                seg_errors[lbl].append(errs[mask])

        merged = np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float64)
        result["thresholds"].append(float(t))
        result["p5"].append(_precision_at(merged, 5.0))
        result["p15"].append(_precision_at(merged, 15.0))
        result["kp_frac"].append(kp_eval / max(total_eval, 1))

        for lbl in seg_labels:
            seg_merged = (
                np.concatenate(seg_errors[lbl])
                if seg_errors[lbl]
                else np.array([], dtype=np.float64)
            )
            result[f"seg_p5_{lbl}"].append(_precision_at(seg_merged, 5.0))
            result[f"seg_p15_{lbl}"].append(_precision_at(seg_merged, 15.0))

    return result


def sweep_thresholds_2d(
    sequences: list[SequenceData],
    thresholds1: np.ndarray,
    thresholds2: np.ndarray,
    segments: tuple = SEGMENTS,
    combine_or: bool = False,
) -> dict:
    """Vectorised 2-D threshold sweep.

    ``combine_or=False`` (default): use_kp = (metric1 >= t1) AND (metric2 >= t2).
    ``combine_or=True``:            use_kp = (metric1 >= t1) OR  (metric2 >= t2).

    Iterates over ``thresholds2`` in the outer loop; for each t2 value, all
    ``thresholds1`` comparisons are handled with numpy broadcasting (no Python
    loop over t1).

    Args:
        sequences: Sequence data list.  ``metric2_eval`` must be set on every
            sequence; sequences without it are treated as always passing the
            metric2 gate.
        thresholds1: Grid of primary-metric thresholds (N1,).
        thresholds2: Grid of secondary-metric thresholds (N2,).
        segments: Frame-range segments for segmented precision.
        combine_or: Combine gates with OR instead of AND.

    Returns:
        Dict with keys ``thresholds1``, ``thresholds2`` (lists) and 2-D arrays
        (N1 × N2) serialised as lists-of-lists: ``p5``, ``p15``, ``kp_frac``,
        and ``seg_p5_*`` / ``seg_p15_*``.
    """
    N1, N2 = len(thresholds1), len(thresholds2)
    seg_labels = [lbl for lbl, _, _ in segments]
    total_eval = sum(len(seq.eval_idx) for seq in sequences)

    p5_arr = np.full((N1, N2), np.nan)
    p15_arr = np.full((N1, N2), np.nan)
    kp_frac_arr = np.zeros((N1, N2))
    seg_p5_arrs = {lbl: np.full((N1, N2), np.nan) for lbl in seg_labels}
    seg_p15_arrs = {lbl: np.full((N1, N2), np.nan) for lbl in seg_labels}

    for i2, t2 in enumerate(thresholds2):
        # Per-t2: accumulate (N1, total_frames) error matrix across sequences.
        all_errs_chunks: list[np.ndarray] = []
        seg_errs_chunks: dict[str, list[np.ndarray]] = {lbl: [] for lbl in seg_labels}
        kp_counts = np.zeros(N1, dtype=np.int64)

        for seq in sequences:
            M = len(seq.eval_idx)
            if M == 0:
                continue

            # Gate on metric2 (row vector, shape (M,))
            if seq.metric2_eval is not None:
                use_m2 = seq.metric2_eval >= t2  # (M,)
            else:
                use_m2 = np.ones(M, dtype=bool)

            # Gate on metric1 for all thresholds at once: (N1, M)
            use_m1 = seq.metric_eval[np.newaxis, :] >= thresholds1[:, np.newaxis]
            use_kp = use_m1 | use_m2[np.newaxis, :] if combine_or else use_m1 & use_m2[np.newaxis, :]  # (N1, M)

            errs = np.where(
                use_kp,
                seq.kp_errors[np.newaxis, :],
                seq.other_errors[np.newaxis, :],
            )  # (N1, M)

            all_errs_chunks.append(errs)
            kp_counts += use_kp.sum(axis=1)

            for lbl, start, stop in segments:
                mask = seq.eval_idx >= start
                if stop is not None:
                    mask &= seq.eval_idx < stop
                seg_errs_chunks[lbl].append(errs[:, mask])

        if not all_errs_chunks:
            continue

        merged = np.concatenate(all_errs_chunks, axis=1)  # (N1, total_frames)

        for i1 in range(N1):
            p5_arr[i1, i2] = _precision_at(merged[i1], 5.0)
            p15_arr[i1, i2] = _precision_at(merged[i1], 15.0)
            kp_frac_arr[i1, i2] = kp_counts[i1] / max(total_eval, 1)
            for lbl in seg_labels:
                chunks = seg_errs_chunks[lbl]
                seg_m = np.concatenate([c[i1] for c in chunks]) if chunks else np.array([], dtype=np.float64)
                seg_p5_arrs[lbl][i1, i2] = _precision_at(seg_m, 5.0)
                seg_p15_arrs[lbl][i1, i2] = _precision_at(seg_m, 15.0)

    # Convert NaN → None for JSON serialisability via _to_jsonable.
    return {
        "thresholds1": thresholds1.tolist(),
        "thresholds2": thresholds2.tolist(),
        "p5": p5_arr.tolist(),
        "p15": p15_arr.tolist(),
        "kp_frac": kp_frac_arr.tolist(),
        **{f"seg_p5_{lbl}": seg_p5_arrs[lbl].tolist() for lbl in seg_labels},
        **{f"seg_p15_{lbl}": seg_p15_arrs[lbl].tolist() for lbl in seg_labels},
    }


def _baseline_metrics(sequences: list[SequenceData], tracker: str) -> dict[str, float]:
    """Compute aggregate and segmented P@5/P@15 for one tracker.

    Args:
        sequences: Sequence data list.
        tracker: ``"kp"`` or ``"other"``.

    Returns:
        Dict with ``p5``, ``p15``, and ``seg_p5_*`` / ``seg_p15_*`` keys.
    """
    all_errors: list[np.ndarray] = []
    seg_errors: dict[str, list[np.ndarray]] = {lbl: [] for lbl, _, _ in SEGMENTS}

    for seq in sequences:
        errs = seq.kp_errors if tracker == "kp" else seq.other_errors
        all_errors.append(errs)
        for lbl, start, stop in SEGMENTS:
            mask = seq.eval_idx >= start
            if stop is not None:
                mask &= seq.eval_idx < stop
            seg_errors[lbl].append(errs[mask])

    merged = np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float64)
    out: dict[str, float] = {
        "p5": _precision_at(merged, 5.0),
        "p15": _precision_at(merged, 15.0),
    }
    for lbl in [l for l, _, _ in SEGMENTS]:
        seg_merged = (
            np.concatenate(seg_errors[lbl])
            if seg_errors[lbl]
            else np.array([], dtype=np.float64)
        )
        out[f"seg_p5_{lbl}"] = _precision_at(seg_merged, 5.0)
        out[f"seg_p15_{lbl}"] = _precision_at(seg_merged, 15.0)
    return out


def _best_threshold(sweep: dict, metric: str = "p15") -> float:
    """Return the threshold achieving the highest value of *metric*.

    Args:
        sweep: Output of :func:`sweep_thresholds`.
        metric: ``"p5"`` or ``"p15"``.

    Returns:
        Best threshold value.
    """
    values = sweep[metric]
    if not values:
        return 0.0
    best_idx = int(np.argmax([v if np.isfinite(v) else -np.inf for v in values]))
    return float(sweep["thresholds"][best_idx])


def build_ensemble_predictions(
    kp_pred: np.ndarray,
    other_pred: np.ndarray,
    metric_all: np.ndarray,
    threshold: float,
    metric2_all: Optional[np.ndarray] = None,
    threshold2: Optional[float] = None,
    combine_or: bool = False,
) -> tuple[np.ndarray, dict]:
    """Build ensemble predictions for one sequence at one threshold (or pair).

    Frame 0 always copies the kp tracker (init frame, identical to GT init).

    Args:
        kp_pred: ``(F, 2, 4)`` kp tracker predictions.
        other_pred: ``(F, 2, 4)`` other tracker predictions.
        metric_all: ``(F,)`` primary kp metric per frame (frame 0 = 0).
        threshold: Use kp when ``metric >= threshold``.
        metric2_all: ``(F,)`` secondary metric (2-D mode).  ``None`` → 1-D.
        threshold2: Secondary threshold.  Ignored when ``metric2_all`` is None.
        combine_or: Combine metric gates with OR instead of AND.

    Returns:
        Tuple ``(ensemble_pred, metadata)``.
    """
    use_kp = metric_all >= threshold
    if metric2_all is not None and threshold2 is not None:
        gate2 = metric2_all >= threshold2
        use_kp = use_kp | gate2 if combine_or else use_kp & gate2
    use_kp[0] = True  # init frame: always kp
    ensemble_pred = np.where(use_kp[:, np.newaxis, np.newaxis], kp_pred, other_pred)

    non_init = use_kp[1:]
    n_kp = int(non_init.sum())
    n_other = int((~non_init).sum())
    meta: dict = {
        "threshold": float(threshold),
        "n_frames_kp": n_kp,
        "n_frames_other": n_other,
        "kp_fraction": n_kp / max(n_kp + n_other, 1),
        "use_kp_by_frame": use_kp.tolist(),
    }
    if threshold2 is not None:
        meta["threshold2"] = float(threshold2)
    return ensemble_pred, meta


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: object) -> object:
    """Recursively convert numpy/Path values to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _save_json(path: Path, payload: dict) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _fmt(v: object) -> str:
    """Format a metric value for display."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{float(v):.1f}"


def _build_summary_table_html(
    methods: list[dict],
) -> str:
    """Render a static HTML table with segmented P@5 and P@15.

    Args:
        methods: List of dicts with ``name``, ``p5_*``, ``p15_*`` keys.

    Returns:
        HTML string.
    """
    seg_labels = [lbl for lbl, _, _ in SEGMENTS]
    head_html = "".join(f"<th>{escape(lbl)}</th>" for lbl in seg_labels)

    def _rows(pk: str, title: str) -> str:
        out = (
            f'<div class="section-label">{escape(title)}</div>'
            '<table class="metrics-table"><thead><tr><th>method</th>'
            f"{head_html}</tr></thead><tbody>"
        )
        for m in methods:
            cells = [f'<td class="method">{escape(m["name"])}</td>']
            for lbl in seg_labels:
                cells.append(f'<td>{_fmt(m.get(f"{pk}_{lbl}"))}</td>')
            out += "<tr>" + "".join(cells) + "</tr>"
        out += "</tbody></table>"
        return out

    return (
        '<div class="tables-grid">'
        f'<div class="table-card">{_rows("p5", "P@5")}</div>'
        f'<div class="table-card">{_rows("p15", "P@15")}</div>'
        "</div>"
    )


def _write_report(
    report_path: Path,
    kp_name: str,
    other_name: str,
    ensemble_name: str,
    metric_name: str,
    best_threshold: float,
    datasets_payload: list[dict],
    cdn: bool = False,
) -> Path:
    """Write self-contained interactive HTML report.

    Args:
        report_path: Output HTML path.
        kp_name: Keypoint tracker name.
        other_name: Other tracker name.
        ensemble_name: Ensemble name.
        metric_name: Threshold metric name.
        best_threshold: Best threshold by combined P@15.
        datasets_payload: One dict per dataset with keys ``label``,
            ``n_sequences``, ``sweep``, ``kp_baseline``, ``other_baseline``,
            ``ensemble_metrics``, ``methods_for_table``.

    Returns:
        Written HTML path.
    """
    seg_labels = [lbl for lbl, _, _ in SEGMENTS]

    # Build per-dataset summary tables (stacked).
    all_tables_html = ""
    for ds in datasets_payload:
        methods = ds["methods_for_table"]
        table_html = _build_summary_table_html(methods)
        all_tables_html += (
            f'<div class="card">'
            f'<div class="section-label">{escape(ds["label"])} — Segmented Precision (ensemble @ best threshold)</div>'
            f"{table_html}</div>"
        )

    thr_label = escape(metric_name)
    meta_parts = [
        f"KP: {escape(kp_name)}",
        f"Other: {escape(other_name)}",
        f"Metric: {thr_label}",
        f"Datasets: {', '.join(escape(ds['label']) for ds in datasets_payload)}",
    ]
    meta_line = " | ".join(meta_parts)

    payload = {
        "kp_name": kp_name,
        "other_name": other_name,
        "ensemble_name": ensemble_name,
        "metric_name": metric_name,
        "best_threshold": best_threshold,
        "datasets": [
            {
                "label": ds["label"],
                "n_sequences": ds["n_sequences"],
                "sweep": ds["sweep"],
                "kp_baseline": ds["kp_baseline"],
                "other_baseline": ds["other_baseline"],
            }
            for ds in datasets_payload
        ],
    }
    payload_json = json.dumps(_to_jsonable(payload), separators=(",", ":"))

    multi_ds_section = ""
    if len(datasets_payload) >= 2:
        multi_ds_section = """
    <div class="card">
      <div class="plot-title">Cross-dataset P@15: {ds0} vs {ds1} (parametric in threshold)</div>
      <div id="plot-cross" class="plot-box"></div>
    </div>""".format(
            ds0=escape(datasets_payload[0]["label"]),
            ds1=escape(datasets_payload[1]["label"]),
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(ensemble_name)} threshold ensemble report</title>
  {get_plotly_script_tag(cdn)}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 18px 20px 28px;
      font-family: system-ui, sans-serif;
      font-size: 13px;
      background: #eef2f7;
      color: #1f2937;
    }}
    h1 {{ font-size: 22px; margin: 0 0 4px; }}
    .meta {{ color: #6b7280; margin-bottom: 14px; }}
    .section-label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: #6b7280;
      margin-bottom: 8px;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,.06);
      padding: 14px 16px;
      margin-bottom: 12px;
    }}
    .tables-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 12px;
    }}
    .table-card {{ overflow-x: auto; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    .metrics-table th, .metrics-table td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: right;
      white-space: nowrap;
    }}
    .metrics-table th:first-child, .metrics-table td.method {{ text-align: left; }}
    .plots-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 12px;
    }}
    .plot-box {{ min-height: 350px; }}
    .plot-title {{ font-weight: 600; margin-bottom: 6px; color: #374151; }}
  </style>
</head>
<body>
  <h1>{escape(ensemble_name)} — threshold ensemble</h1>
  <div class="meta">{meta_line}</div>

  {all_tables_html}

  <div class="plots-grid">
    <div class="card">
      <div class="plot-title">P@15 vs threshold</div>
      <div id="plot-p15" class="plot-box"></div>
    </div>
    <div class="card">
      <div class="plot-title">P@5 vs threshold</div>
      <div id="plot-p5" class="plot-box"></div>
    </div>
    <div class="card">
      <div class="plot-title">KP fraction vs threshold</div>
      <div id="plot-frac" class="plot-box"></div>
    </div>
    <div class="card">
      <div class="plot-title">P@15 vs KP fraction (parametric in threshold)</div>
      <div id="plot-pareto" class="plot-box"></div>
    </div>
    {multi_ds_section}
  </div>

  <script>
    const D = {payload_json};
    const DS_COLORS = ['#059669', '#7c3aed', '#d97706', '#0891b2'];
    const bestT = D.best_threshold;
    const metricName = D.metric_name;

    // All datasets share the same threshold axis (built from combined unique values).
    const thresholds = D.datasets[0].sweep.thresholds;

    function hlineFlat(y, label, color, xArr) {{
      return {{
        x: [xArr[0], xArr[xArr.length - 1]],
        y: [y, y],
        mode: 'lines',
        name: label,
        showlegend: false,
        line: {{ color, width: 1.2, dash: 'dot' }},
        hovertemplate: label + ': ' + (y != null ? y.toFixed(1) : '?') + '%<extra></extra>',
      }};
    }}

    function vline(x, xArr) {{
      return {{
        x: [x, x],
        y: [0, 100],
        mode: 'lines',
        showlegend: false,
        line: {{ color: '#6b7280', width: 1.5, dash: 'dash' }},
        hoverinfo: 'skip',
      }};
    }}

    function renderPrecisionPlot(divId, pk, title) {{
      const traces = [];
      D.datasets.forEach((ds, i) => {{
        const vals = ds.sweep[pk];
        const kpBase = ds.kp_baseline[pk];
        const otherBase = ds.other_baseline[pk];
        const color = DS_COLORS[i % DS_COLORS.length];
        traces.push({{
          x: thresholds,
          y: vals,
          mode: 'lines',
          name: ds.label + ' ensemble',
          line: {{ color, width: 2.5 }},
          hovertemplate: 'thr=%{{x:.1f}}<br>' + ds.label + ' ensemble ' + title + ': %{{y:.1f}}%<extra></extra>',
        }});
        traces.push(hlineFlat(kpBase, ds.label + ' ' + D.kp_name, color, thresholds));
        traces.push(hlineFlat(otherBase, ds.label + ' ' + D.other_name, color, thresholds));
      }});
      traces.push(vline(bestT, thresholds));
      Plotly.react(divId, traces, {{
        margin: {{ l: 52, r: 12, t: 10, b: 90 }},
        xaxis: {{ title: metricName + ' threshold' }},
        yaxis: {{ title: title + ' (%)', range: [0, 100] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
        annotations: [{{
          x: bestT, y: 1.04, xref: 'x', yref: 'paper',
          text: 'best', showarrow: false,
          font: {{ size: 10, color: '#6b7280' }},
        }}],
      }}, {{responsive: true, displayModeBar: false}});
    }}

    function renderFracPlot() {{
      const traces = [];
      D.datasets.forEach((ds, i) => {{
        const color = DS_COLORS[i % DS_COLORS.length];
        traces.push({{
          x: thresholds,
          y: ds.sweep.kp_frac.map(v => v * 100),
          mode: 'lines',
          name: ds.label,
          line: {{ color, width: 2 }},
          hovertemplate: 'thr=%{{x:.1f}}<br>' + ds.label + ' kp frac: %{{y:.1f}}%<extra></extra>',
        }});
      }});
      traces.push(vline(bestT, thresholds));
      Plotly.react('plot-frac', traces, {{
        margin: {{ l: 52, r: 12, t: 10, b: 90 }},
        xaxis: {{ title: metricName + ' threshold' }},
        yaxis: {{ title: '% frames using KP tracker', range: [0, 100] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
    }}

    function renderParetoPlot() {{
      const traces = [];
      D.datasets.forEach((ds, i) => {{
        const color = DS_COLORS[i % DS_COLORS.length];
        const kpFrac = ds.sweep.kp_frac;
        const p15 = ds.sweep.p15;
        const kpBase = ds.kp_baseline.p15;
        const otherBase = ds.other_baseline.p15;
        const bestIdx = thresholds.findIndex(t => Math.abs(t - bestT) < 1e-9);
        traces.push({{
          x: kpFrac.map(v => v * 100),
          y: p15,
          customdata: thresholds,
          mode: 'lines+markers',
          name: ds.label + ' ensemble',
          line: {{ color, width: 1.5 }},
          marker: {{ size: 3, color }},
          hovertemplate: 'thr=%{{customdata:.1f}}<br>' + ds.label + ' kp frac: %{{x:.1f}}%<br>P@15: %{{y:.1f}}%<extra></extra>',
        }});
        traces.push({{
          x: [0], y: [otherBase],
          mode: 'markers',
          name: ds.label + ' ' + D.other_name,
          showlegend: false,
          marker: {{ size: 10, color, symbol: 'x' }},
          hovertemplate: ds.label + ' ' + D.other_name + ' P@15: %{{y:.1f}}%<extra></extra>',
        }});
        traces.push({{
          x: [100], y: [kpBase],
          mode: 'markers',
          name: ds.label + ' ' + D.kp_name,
          showlegend: false,
          marker: {{ size: 10, color, symbol: 'diamond' }},
          hovertemplate: ds.label + ' ' + D.kp_name + ' P@15: %{{y:.1f}}%<extra></extra>',
        }});
        if (bestIdx >= 0) {{
          traces.push({{
            x: [kpFrac[bestIdx] * 100], y: [p15[bestIdx]],
            customdata: [thresholds[bestIdx]],
            mode: 'markers',
            name: ds.label + ' best',
            showlegend: false,
            marker: {{ size: 12, color: '#f59e0b', symbol: 'star' }},
            hovertemplate: 'BEST thr=%{{customdata:.1f}}<br>' + ds.label + ' kp frac: %{{x:.1f}}%<br>P@15: %{{y:.1f}}%<extra></extra>',
          }});
        }}
      }});
      Plotly.react('plot-pareto', traces, {{
        margin: {{ l: 52, r: 12, t: 10, b: 90 }},
        xaxis: {{ title: '% frames using KP tracker', range: [0, 100] }},
        yaxis: {{ title: 'P@15 (%)' }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
    }}

    function renderCrossDatasetPlot() {{
      if (D.datasets.length < 2) return;
      const ds0 = D.datasets[0];
      const ds1 = D.datasets[1];
      const p15_0 = ds0.sweep.p15;
      const p15_1 = ds1.sweep.p15;
      const bestIdx = thresholds.findIndex(t => Math.abs(t - bestT) < 1e-9);

      const traces = [
        {{
          x: p15_0,
          y: p15_1,
          customdata: thresholds,
          mode: 'lines+markers',
          name: 'ensemble (varying threshold)',
          line: {{ color: '#059669', width: 1.5 }},
          marker: {{ size: 3, color: '#059669' }},
          hovertemplate: 'thr=%{{customdata:.1f}}<br>' + ds0.label + ' P@15: %{{x:.1f}}%<br>' + ds1.label + ' P@15: %{{y:.1f}}%<extra></extra>',
        }},
        {{
          x: [ds0.other_baseline.p15], y: [ds1.other_baseline.p15],
          mode: 'markers',
          name: D.other_name,
          marker: {{ size: 10, color: '#dc2626', symbol: 'diamond' }},
          hovertemplate: D.other_name + '<br>' + ds0.label + ': %{{x:.1f}}%<br>' + ds1.label + ': %{{y:.1f}}%<extra></extra>',
        }},
        {{
          x: [ds0.kp_baseline.p15], y: [ds1.kp_baseline.p15],
          mode: 'markers',
          name: D.kp_name,
          marker: {{ size: 10, color: '#2563eb', symbol: 'diamond' }},
          hovertemplate: D.kp_name + '<br>' + ds0.label + ': %{{x:.1f}}%<br>' + ds1.label + ': %{{y:.1f}}%<extra></extra>',
        }},
      ];
      if (bestIdx >= 0) {{
        traces.push({{
          x: [p15_0[bestIdx]], y: [p15_1[bestIdx]],
          customdata: [thresholds[bestIdx]],
          mode: 'markers',
          name: 'best threshold',
          marker: {{ size: 12, color: '#f59e0b', symbol: 'star' }},
          hovertemplate: 'BEST thr=%{{customdata:.1f}}<br>' + ds0.label + ': %{{x:.1f}}%<br>' + ds1.label + ': %{{y:.1f}}%<extra></extra>',
        }});
      }}
      // Diagonal reference line.
      const allVals = [...p15_0, ...p15_1, ds0.kp_baseline.p15, ds1.kp_baseline.p15,
                       ds0.other_baseline.p15, ds1.other_baseline.p15].filter(v => v != null);
      const lo = Math.max(0, Math.min(...allVals) - 2);
      const hi = Math.min(100, Math.max(...allVals) + 2);
      traces.push({{
        x: [lo, hi], y: [lo, hi],
        mode: 'lines',
        name: 'y = x',
        showlegend: false,
        line: {{ color: '#94a3b8', width: 1, dash: 'dot' }},
        hoverinfo: 'skip',
      }});
      Plotly.react('plot-cross', traces, {{
        margin: {{ l: 52, r: 12, t: 10, b: 90 }},
        xaxis: {{ title: ds0.label + ' P@15 (%)', range: [lo, hi] }},
        yaxis: {{ title: ds1.label + ' P@15 (%)', range: [lo, hi] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
    }}

    renderPrecisionPlot('plot-p15', 'p15', 'P@15');
    renderPrecisionPlot('plot-p5', 'p5', 'P@5');
    renderFracPlot();
    renderParetoPlot();
    renderCrossDatasetPlot();
  </script>
</body>
</html>
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# 2-D HTML report
# ---------------------------------------------------------------------------


def _write_report_2d(
    report_path: Path,
    kp_name: str,
    other_name: str,
    ensemble_name: str,
    metric1_name: str,
    metric2_name: Optional[str],
    best_threshold1: float,
    best_threshold2: Optional[float],
    datasets_payload: list[dict],
    cdn: bool = False,
) -> Path:
    """Write interactive HTML report for the 2-D threshold sweep.

    Shows per-dataset P@15 heatmaps (metric1 × metric2) with the best
    threshold pair marked, plus summary tables.

    Args:
        report_path: Output HTML path.
        kp_name: Keypoint tracker name.
        other_name: Other tracker name.
        ensemble_name: Ensemble name.
        metric1_name: Primary threshold metric name.
        metric2_name: Secondary threshold metric name.
        best_threshold1: Best primary threshold.
        best_threshold2: Best secondary threshold.
        datasets_payload: One dict per dataset (from ``build_threshold_ensemble``).

    Returns:
        Written HTML path.
    """
    seg_labels = [lbl for lbl, _, _ in SEGMENTS]

    all_tables_html = ""
    for ds in datasets_payload:
        methods = ds["methods_for_table"]
        table_html = _build_summary_table_html(methods)
        all_tables_html += (
            f'<div class="card">'
            f'<div class="section-label">{escape(ds["label"])} — Segmented Precision (ensemble @ best thresholds)</div>'
            f"{table_html}</div>"
        )

    meta_parts = [
        f"KP: {escape(kp_name)}",
        f"Other: {escape(other_name)}",
        f"Metric1: {escape(metric1_name)}",
        f"Metric2: {escape(metric2_name or '')}",
        f"Datasets: {', '.join(escape(ds['label']) for ds in datasets_payload)}",
    ]
    meta_line = " | ".join(meta_parts)

    payload = {
        "kp_name": kp_name,
        "other_name": other_name,
        "ensemble_name": ensemble_name,
        "metric1_name": metric1_name,
        "metric2_name": metric2_name,
        "best_threshold1": best_threshold1,
        "best_threshold2": best_threshold2,
        "datasets": [
            {
                "label": ds["label"],
                "n_sequences": ds["n_sequences"],
                "sweep": ds["sweep"],
                "kp_baseline": ds["kp_baseline"],
                "other_baseline": ds["other_baseline"],
            }
            for ds in datasets_payload
        ],
    }
    payload_json = json.dumps(_to_jsonable(payload), separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(ensemble_name)} 2-D threshold ensemble report</title>
  {get_plotly_script_tag(cdn)}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 18px 20px 28px;
      font-family: system-ui, sans-serif; font-size: 13px;
      background: #eef2f7; color: #1f2937;
    }}
    h1 {{ font-size: 22px; margin: 0 0 4px; }}
    .meta {{ color: #6b7280; margin-bottom: 14px; }}
    .section-label {{
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .08em; color: #6b7280; margin-bottom: 8px;
    }}
    .card {{
      background: #fff; border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,.06); padding: 14px 16px; margin-bottom: 12px;
    }}
    .tables-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px;
    }}
    .table-card {{ overflow-x: auto; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    .metrics-table th, .metrics-table td {{
      padding: 6px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; white-space: nowrap;
    }}
    .metrics-table th:first-child, .metrics-table td.method {{ text-align: left; }}
    .plots-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 12px;
    }}
    .plot-box {{ min-height: 400px; }}
    .plot-title {{ font-weight: 600; margin-bottom: 6px; color: #374151; }}
  </style>
</head>
<body>
  <h1>{escape(ensemble_name)} — 2-D threshold ensemble</h1>
  <div class="meta">{meta_line}</div>

  {all_tables_html}

  <div class="plots-grid" id="heatmap-grid"></div>

  <script>
    const D = {payload_json};
    const DS_COLORS = ['#059669', '#7c3aed', '#d97706', '#0891b2'];
    const bestT1 = D.best_threshold1;
    const bestT2 = D.best_threshold2;

    function renderHeatmap(divId, pk, title) {{
      const traces = [];
      D.datasets.forEach((ds, i) => {{
        const z = ds.sweep[pk];
        traces.push({{
          z: z,
          x: ds.sweep.thresholds2,
          y: ds.sweep.thresholds1,
          type: 'heatmap',
          colorscale: 'RdYlGn',
          name: ds.label,
          zmin: 0, zmax: 100,
          hovertemplate: (
            D.metric1_name + '=%{{y:.3f}}<br>' +
            D.metric2_name + '=%{{x:.3f}}<br>' +
            ds.label + ' ' + title + ': %{{z:.1f}}%<extra></extra>'
          ),
        }});
      }});
      // Best point marker
      if (bestT1 != null && bestT2 != null) {{
        traces.push({{
          x: [bestT2], y: [bestT1],
          mode: 'markers',
          name: 'best',
          marker: {{ size: 14, color: '#fbbf24', symbol: 'star', line: {{ color: '#000', width: 1 }} }},
          hovertemplate: 'BEST t1=%{{y:.3f}} t2=%{{x:.3f}}<extra></extra>',
        }});
      }}
      Plotly.react(divId, traces, {{
        margin: {{ l: 70, r: 12, t: 30, b: 80 }},
        title: {{ text: title, font: {{ size: 13 }} }},
        xaxis: {{ title: D.metric2_name + ' threshold' }},
        yaxis: {{ title: D.metric1_name + ' threshold' }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.2, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
    }}

    // Build heatmap plot divs dynamically.
    const grid = document.getElementById('heatmap-grid');
    ['p15', 'p5'].forEach(pk => {{
      const label = pk === 'p15' ? 'P@15 (%)' : 'P@5 (%)';
      const card = document.createElement('div');
      card.className = 'card';
      const divId = 'plot-' + pk;
      card.innerHTML = '<div class="plot-title">' + label + ' vs thresholds (heatmap)</div>' +
                       '<div id="' + divId + '" class="plot-box"></div>';
      grid.appendChild(card);
      renderHeatmap(divId, pk, label);
    }});
  </script>
</body>
</html>
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _resolve_results_dir(results_roots: list[Path], tracker_spec: str, dataset: str) -> Path:
    """Resolve tracker spec to its results directory by searching multiple roots.

    Searches each root in order and returns the first match.

    Args:
        results_roots: Top-level results roots to search (in order).
        tracker_spec: Tracker name or config path.
        dataset: Dataset CLI key.

    Returns:
        First matching results directory path.

    Raises:
        ValueError: If no directory is found in any root.
    """
    name = Path(tracker_spec).stem
    dataset_label = DATASETS[dataset].label
    for root in results_roots:
        results_dir = root / name / dataset_label
        if results_dir.exists():
            return results_dir
    searched = ", ".join(str(r / name / dataset_label) for r in results_roots)
    raise ValueError(f"Results directory not found (searched: {searched})")


def _make_load_gt_fn(dataset: str, initfix_dir: Optional[Path], woft_compat: bool = True):
    """Return GT loader for dataset."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return lambda bd, sn: pot_load_gt(bd, sn, woft_compat=woft_compat)
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir required for pti GT loading")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def _load_dataset_sequence_data(
    dataset: str,
    base_dir: Path,
    sequences: list[str],
    initfix_dir: Optional[Path],
    kp_results_dir: Path,
    other_results_dir: Path,
    threshold_metric: str,
    threshold_metric2: Optional[str] = None,
    woft_compat: bool = True,
) -> tuple[list[SequenceData], list[dict]]:
    """Load per-frame data for one dataset.

    Args:
        dataset: Dataset CLI key.
        base_dir: Dataset root directory.
        sequences: Sequence names to process.
        initfix_dir: Initfix directory for PTI datasets.
        kp_results_dir: KP tracker results directory.
        other_results_dir: Other tracker results directory.
        threshold_metric: Primary metric key to extract from stats JSON.
        threshold_metric2: Secondary metric key for 2-D sweep (optional).
        woft_compat: Use WOFT-compatible POT annotations (True) or original (False).

    Returns:
        Tuple ``(sequence_data, failed)`` where ``failed`` is a list of error
        dicts for sequences that could not be loaded.
    """
    load_gt_fn = _make_load_gt_fn(dataset, initfix_dir, woft_compat=woft_compat)
    sequence_data: list[SequenceData] = []
    failed: list[dict] = []

    for seq_name in sequences:
        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            kp_pred = load_predictions(kp_results_dir, seq_name)
            other_pred = load_predictions(other_results_dir, seq_name)

            min_len = min(len(gt_corners), len(kp_pred), len(other_pred))
            gt_corners = gt_corners[:min_len]
            valid = valid[:min_len]
            kp_pred = kp_pred[:min_len]
            other_pred = other_pred[:min_len]

            stats_path = kp_results_dir / f"{seq_name}_stats.json"
            frame_stats = load_kp_frame_stats(stats_path)
            if not frame_stats:
                print(f"  WARNING: no stats file for {seq_name} ({stats_path})")

            metric_all = build_metric_array(frame_stats, min_len, threshold_metric)
            eval_idx, kp_errors = _compute_per_frame_errors(gt_corners, valid, kp_pred)
            _, other_errors = _compute_per_frame_errors(gt_corners, valid, other_pred)

            metric2_all: Optional[np.ndarray] = None
            if threshold_metric2 is not None:
                metric2_all = build_metric_array(frame_stats, min_len, threshold_metric2)

            sequence_data.append(
                SequenceData(
                    name=seq_name,
                    num_frames=min_len,
                    eval_idx=eval_idx,
                    kp_errors=kp_errors,
                    other_errors=other_errors,
                    metric_eval=metric_all[eval_idx],
                    metric_all=metric_all,
                    metric2_eval=metric2_all[eval_idx] if metric2_all is not None else None,
                    metric2_all=metric2_all,
                )
            )
        except Exception as exc:
            failed.append(
                {
                    "sequence": seq_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  ERROR loading {seq_name}: {exc}")

    return sequence_data, failed


def _compute_ensemble_metrics(
    sequence_data: list[SequenceData],
    threshold: float,
    threshold2: Optional[float] = None,
    combine_or: bool = False,
) -> dict[str, float]:
    """Compute aggregate + segmented P@5/P@15 for ensemble at one threshold (or pair).

    Args:
        sequence_data: Sequence data list.
        threshold: Primary threshold value.
        threshold2: Secondary threshold for 2-D mode (optional).
        combine_or: Combine metric gates with OR instead of AND.

    Returns:
        Dict with ``p5``, ``p15``, and ``seg_p5_*`` / ``seg_p15_*`` keys.
    """
    seg_labels = [lbl for lbl, _, _ in SEGMENTS]
    all_errors: list[np.ndarray] = []
    seg_errors: dict[str, list[np.ndarray]] = {lbl: [] for lbl in seg_labels}

    for seq in sequence_data:
        use_kp = seq.metric_eval >= threshold
        if seq.metric2_eval is not None and threshold2 is not None:
            gate2 = seq.metric2_eval >= threshold2
            use_kp = use_kp | gate2 if combine_or else use_kp & gate2
        errs = np.where(use_kp, seq.kp_errors, seq.other_errors)
        all_errors.append(errs)
        for lbl, start, stop in SEGMENTS:
            mask = seq.eval_idx >= start
            if stop is not None:
                mask &= seq.eval_idx < stop
            seg_errors[lbl].append(errs[mask])

    merged = np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float64)
    out: dict[str, float] = {
        "p5": _precision_at(merged, 5.0),
        "p15": _precision_at(merged, 15.0),
    }
    for lbl in seg_labels:
        seg_merged = (
            np.concatenate(seg_errors[lbl])
            if seg_errors[lbl]
            else np.array([], dtype=np.float64)
        )
        out[f"seg_p5_{lbl}"] = _precision_at(seg_merged, 5.0)
        out[f"seg_p15_{lbl}"] = _precision_at(seg_merged, 15.0)
    return out


def _write_dataset_predictions(
    dataset: str,
    base_dir: Path,
    initfix_dir: Optional[Path],
    sequence_data: list[SequenceData],
    kp_results_dir: Path,
    other_results_dir: Path,
    output_dir: Path,
    write_threshold: float,
    kp_name: str,
    other_name: str,
    threshold_metric: str,
    skip_completed: bool,
    write_threshold2: Optional[float] = None,
    combine_or: bool = False,
    woft_compat: bool = True,
) -> tuple[list[str], list[dict], list[dict]]:
    """Write per-sequence prediction files for one dataset at one threshold (or pair).

    Args:
        dataset: Dataset CLI key.
        base_dir: Dataset root directory.
        initfix_dir: Initfix directory for PTI datasets.
        sequence_data: Pre-loaded sequence data.
        kp_results_dir: KP tracker results directory.
        other_results_dir: Other tracker results directory.
        output_dir: Dataset-specific output directory.
        write_threshold: Primary threshold to use for selection.
        kp_name: KP tracker name.
        other_name: Other tracker name.
        threshold_metric: Metric name (for metadata).
        skip_completed: Skip existing complete outputs.
        write_threshold2: Secondary threshold for 2-D mode (optional).
        combine_or: Combine metric gates with OR instead of AND.
        woft_compat: Use WOFT-compatible POT annotations (True) or original (False).

    Returns:
        Tuple ``(completed_sequences, failed, sequence_results)``.
    """
    load_gt_fn = _make_load_gt_fn(dataset, initfix_dir, woft_compat=woft_compat)
    completed: list[str] = []
    failed: list[dict] = []
    seq_results: list[dict] = []

    for seq in sequence_data:
        seq_name = seq.name
        output_file = output_dir / f"{seq_name}.txt"

        if skip_completed and output_file.exists():
            try:
                existing_flat = np.loadtxt(output_file)
                if (existing_flat.ndim == 1 and seq.num_frames == 1) or (
                    existing_flat.ndim == 2 and len(existing_flat) == seq.num_frames
                ):
                    print(f"  Skipping {seq_name} (already complete)")
                    completed.append(seq_name)
                    continue
            except Exception:
                pass

        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            kp_pred = load_predictions(kp_results_dir, seq_name)
            other_pred = load_predictions(other_results_dir, seq_name)
            min_len = min(len(gt_corners), len(kp_pred), len(other_pred))
            gt_corners = gt_corners[:min_len]
            valid = valid[:min_len]
            kp_pred = kp_pred[:min_len]
            other_pred = other_pred[:min_len]

            ensemble_pred, seq_meta = build_ensemble_predictions(
                kp_pred, other_pred, seq.metric_all[:min_len], write_threshold,
                metric2_all=seq.metric2_all[:min_len] if seq.metric2_all is not None else None,
                threshold2=write_threshold2,
                combine_or=combine_or,
            )
            save_corners(ensemble_pred, output_file)

            seq_result = evaluate_sequence(
                gt_corners, ensemble_pred, valid, use_best_rotation=False
            )
            seq_result["sequence"] = seq_name
            seq_results.append(seq_result)

            _save_json(
                output_dir / "ensemble_meta" / f"{seq_name}.json",
                {
                    "sequence": seq_name,
                    "kp_tracker": kp_name,
                    "other_tracker": other_name,
                    "threshold_metric": threshold_metric,
                    "num_frames": min_len,
                    **seq_meta,
                },
            )
            completed.append(seq_name)
        except Exception as exc:
            failed.append(
                {
                    "sequence": seq_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  ERROR writing {seq_name}: {exc}")

    return completed, failed, seq_results


def _threshold_suffix(threshold: float, metric: str) -> str:
    """Format threshold value for use in directory/tracker names.

    Args:
        threshold: Threshold value.
        metric: Threshold metric name.

    Returns:
        String like ``"30"`` for integer metrics or ``"0.75"`` for ratio.
    """
    if metric in ("inlier_ratio", "hull_coverage", "peeled_hull_coverage"):
        return f"{threshold:.2f}"
    return f"{int(round(threshold))}"


def build_threshold_ensemble(
    datasets_info: list[tuple[str, Path, list[str], Optional[Path]]],
    results_roots: list[Path],
    output_root: Path,
    ensemble_name: str,
    kp_tracker_spec: str,
    other_tracker_spec: str,
    threshold_metric: str,
    threshold: Optional[float],
    skip_completed: bool,
    threshold_metric2: Optional[str] = None,
    threshold2: Optional[float] = None,
    combine_or: bool = False,
    cdn: bool = False,
    woft_compat: bool = True,
) -> None:
    """Build threshold ensemble from precomputed tracker results.

    Processes all provided datasets with a shared threshold grid built from
    their combined metric values.  The best threshold is selected by maximising
    P@15 across all datasets combined.  Prediction files are written per
    dataset; a single HTML report covers all datasets.

    When ``threshold_metric2`` is provided, runs a 2-D sweep (use_kp = m1>=t1
    AND m2>=t2) and generates a heatmap report.

    Args:
        datasets_info: List of ``(dataset_key, base_dir, sequences, initfix_dir)``
            tuples.  The first entry is the primary dataset.
        results_roots: Top-level tracker-results roots to search (in order).
        output_root: Top-level ensemble-results root.
        ensemble_name: Output name (used as the results sub-directory name).
        kp_tracker_spec: Keypoint tracker name or config path.
        other_tracker_spec: Other tracker name or config path.
        threshold_metric: Primary per-frame metric key for thresholding.
        threshold: Fixed primary threshold; ``None`` picks best from sweep.
        skip_completed: Skip prediction files that already exist.
        threshold_metric2: Secondary metric for 2-D sweep (optional).
        threshold2: Fixed secondary threshold; ``None`` picks best from sweep.
        combine_or: Combine metric gates with OR instead of AND (2-D mode only).
        woft_compat: Use WOFT-compatible POT annotations (True) or original (False).
    """
    kp_name = Path(kp_tracker_spec).stem
    other_name = Path(other_tracker_spec).stem

    mode_2d = threshold_metric2 is not None
    print(f"Threshold ensemble: {ensemble_name} ({'2-D' if mode_2d else '1-D'} sweep)")
    print(f"KP tracker:    {kp_name}")
    print(f"Other tracker: {other_name}")
    print(f"Metric:        {threshold_metric}" + (f"  {'OR' if combine_or else 'AND'}  {threshold_metric2}" if mode_2d else ""))
    print(f"Datasets:      {len(datasets_info)}")

    # -----------------------------------------------------------------------
    # Phase 1: resolve per-dataset results dirs, load SequenceData
    # -----------------------------------------------------------------------
    per_dataset_data: list[tuple[str, Path, list[str], Optional[Path], Path, Path, list[SequenceData], list[dict]]] = []

    for ds_key, base_dir, sequences, initfix_dir in datasets_info:
        ds_label = DATASETS[ds_key].label
        kp_dir = _resolve_results_dir(results_roots, kp_tracker_spec, ds_key)
        other_dir = _resolve_results_dir(results_roots, other_tracker_spec, ds_key)

        print(f"\n[{ds_label}] Loading {len(sequences)} sequences...")
        print(f"  KP:    {kp_dir}")
        print(f"  Other: {other_dir}")

        seq_data, failed = _load_dataset_sequence_data(
            ds_key, base_dir, sequences, initfix_dir,
            kp_dir, other_dir, threshold_metric,
            threshold_metric2=threshold_metric2,
            woft_compat=woft_compat,
        )
        print(f"  Loaded {len(seq_data)} / {len(sequences)} sequences ({len(failed)} failed)")

        per_dataset_data.append(
            (ds_key, base_dir, sequences, initfix_dir, kp_dir, other_dir, seq_data, failed)
        )

    all_seq_data = [seq for *_, seq_data, _ in per_dataset_data for seq in seq_data]
    if not all_seq_data:
        print("No sequences loaded across any dataset; aborting.")
        return

    # -----------------------------------------------------------------------
    # Phase 2: shared threshold grid + per-dataset sweeps
    # -----------------------------------------------------------------------
    threshold_grid = _build_threshold_grid(all_seq_data)

    # For 2-D mode, build a separate grid from metric2 values.
    if mode_2d:
        # Temporary sequences with metric2 as primary metric for grid building.
        _m2_seqs = [
            SequenceData(
                name=seq.name,
                num_frames=seq.num_frames,
                eval_idx=seq.eval_idx,
                kp_errors=seq.kp_errors,
                other_errors=seq.other_errors,
                metric_eval=seq.metric2_eval if seq.metric2_eval is not None else np.zeros(len(seq.eval_idx)),
                metric_all=seq.metric2_all if seq.metric2_all is not None else np.zeros(seq.num_frames),
            )
            for seq in all_seq_data
        ]
        threshold_grid2 = _build_threshold_grid(_m2_seqs)
        print(
            f"\nSweeping {len(threshold_grid)} × {len(threshold_grid2)} thresholds "
            f"across {len(all_seq_data)} total sequences (2-D)..."
        )
        print(f"  {threshold_metric}: [{threshold_grid[0]:.4g}, {threshold_grid[-1]:.4g}]")
        print(f"  {threshold_metric2}: [{threshold_grid2[0]:.4g}, {threshold_grid2[-1]:.4g}]")
    else:
        threshold_grid2 = np.array([])
        print(f"\nSweeping {len(threshold_grid)} thresholds across {len(all_seq_data)} total sequences...")
        print(f"  {threshold_metric}: [{threshold_grid[0]:.4g}, {threshold_grid[-1]:.4g}]")

    per_dataset_sweep: dict[str, dict] = {}
    for ds_key, *_, seq_data, _ in per_dataset_data:
        if not seq_data:
            continue
        if mode_2d:
            per_dataset_sweep[ds_key] = sweep_thresholds_2d(seq_data, threshold_grid, threshold_grid2, combine_or=combine_or)
        else:
            per_dataset_sweep[ds_key] = sweep_thresholds(seq_data, threshold_grid)

    # Best threshold(s): equal-weight average of per-dataset P@15.
    if mode_2d:
        ds_p15_2d = [np.array(per_dataset_sweep[k]["p15"]) for k in per_dataset_sweep]
        combined_p15_2d = np.nanmean(np.stack(ds_p15_2d, axis=0), axis=0)
        flat_best = int(np.argmax(np.where(np.isfinite(combined_p15_2d), combined_p15_2d, -np.inf).ravel()))
        best_i1, best_i2 = np.unravel_index(flat_best, combined_p15_2d.shape)
        auto_best = float(threshold_grid[best_i1])
        auto_best2: Optional[float] = float(threshold_grid2[best_i2])
        write_threshold = threshold if threshold is not None else auto_best
        write_threshold2: Optional[float] = threshold2 if threshold2 is not None else auto_best2
        print(
            f"Best threshold pair (equal-weight avg P@15 across {len(ds_p15_2d)} dataset(s)): "
            f"{auto_best} / {auto_best2}"
        )
        print(f"Writing predictions at thresholds: {write_threshold} / {write_threshold2}")
    else:
        ds_p15_arrays = [np.array(per_dataset_sweep[k]["p15"]) for k in per_dataset_sweep]
        combined_p15 = np.nanmean(np.stack(ds_p15_arrays, axis=0), axis=0)
        best_idx = int(np.argmax(np.where(np.isfinite(combined_p15), combined_p15, -np.inf)))
        auto_best = float(threshold_grid[best_idx])
        auto_best2 = None
        write_threshold = threshold if threshold is not None else auto_best
        write_threshold2 = None
        print(f"Best threshold (equal-weight avg P@15 across {len(ds_p15_arrays)} dataset(s)): {auto_best}")
        print(f"Writing predictions at threshold: {write_threshold}")

    thr_suffix = _threshold_suffix(write_threshold, threshold_metric)
    if mode_2d and write_threshold2 is not None:
        thr_suffix2 = _threshold_suffix(write_threshold2, threshold_metric2)
        ensemble_name = f"{ensemble_name}_thr{thr_suffix}_thr2{thr_suffix2}"
    else:
        ensemble_name = f"{ensemble_name}_thr{thr_suffix}"
    ensemble_root = output_root / ensemble_name
    ensemble_root.mkdir(parents=True, exist_ok=True)
    print(f"Ensemble name: {ensemble_name}")
    print(f"Output root:   {ensemble_root}")

    # Collision checks + per-dataset output dirs (now that final name is known).
    for ds_key, base_dir, sequences, initfix_dir, kp_dir, other_dir, seq_data, failed in per_dataset_data:
        ds_label = DATASETS[ds_key].label
        out_dir = ensemble_root / ds_label
        if out_dir.resolve() in {kp_dir.resolve(), other_dir.resolve()}:
            raise ValueError(
                f"Ensemble output directory collides with source: {out_dir}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 3: per-dataset baseline metrics + ensemble metrics at best
    # -----------------------------------------------------------------------
    seg_labels = [lbl for lbl, _, _ in SEGMENTS]

    datasets_payload: list[dict] = []
    for ds_key, base_dir, sequences, initfix_dir, kp_dir, other_dir, seq_data, _ in per_dataset_data:
        if not seq_data:
            continue
        ds_label = DATASETS[ds_key].label
        kp_metrics = _baseline_metrics(seq_data, "kp")
        other_metrics = _baseline_metrics(seq_data, "other")
        ens_metrics = _compute_ensemble_metrics(seq_data, auto_best, threshold2=auto_best2, combine_or=combine_or)

        methods_for_table = [
            {"name": other_name, **{f"p5_{lbl}": other_metrics.get(f"seg_p5_{lbl}") for lbl in seg_labels}, **{f"p15_{lbl}": other_metrics.get(f"seg_p15_{lbl}") for lbl in seg_labels}},
            {"name": kp_name, **{f"p5_{lbl}": kp_metrics.get(f"seg_p5_{lbl}") for lbl in seg_labels}, **{f"p15_{lbl}": kp_metrics.get(f"seg_p15_{lbl}") for lbl in seg_labels}},
            {"name": f"{ensemble_name}@{thr_suffix}", **{f"p5_{lbl}": ens_metrics.get(f"seg_p5_{lbl}") for lbl in seg_labels}, **{f"p15_{lbl}": ens_metrics.get(f"seg_p15_{lbl}") for lbl in seg_labels}},
        ]
        datasets_payload.append({
            "label": ds_label,
            "n_sequences": len(seq_data),
            "sweep": per_dataset_sweep[ds_key],
            "kp_baseline": kp_metrics,
            "other_baseline": other_metrics,
            "ensemble_metrics": ens_metrics,
            "methods_for_table": methods_for_table,
        })

        # Print stdout table.
        print(f"\n[{ds_label}] P@5")
        print("| method | " + " | ".join(seg_labels) + " |")
        print("| --- | " + " | ".join(["---"] * len(seg_labels)) + " |")
        for nm, m in [(other_name, other_metrics), (kp_name, kp_metrics), (f"{ensemble_name}@{thr_suffix}", ens_metrics)]:
            vals = " | ".join(_fmt(m.get(f"seg_p5_{lbl}")) for lbl in seg_labels)
            print(f"| {nm} | {vals} |")
        print(f"\n[{ds_label}] P@15")
        print("| method | " + " | ".join(seg_labels) + " |")
        print("| --- | " + " | ".join(["---"] * len(seg_labels)) + " |")
        for nm, m in [(other_name, other_metrics), (kp_name, kp_metrics), (f"{ensemble_name}@{thr_suffix}", ens_metrics)]:
            vals = " | ".join(_fmt(m.get(f"seg_p15_{lbl}")) for lbl in seg_labels)
            print(f"| {nm} | {vals} |")

    # -----------------------------------------------------------------------
    # Phase 4: write per-dataset prediction files
    # -----------------------------------------------------------------------
    all_completed: dict[str, list[str]] = {}
    all_failed: dict[str, list[dict]] = {}

    for ds_key, base_dir, sequences, initfix_dir, kp_dir, other_dir, seq_data, load_failed in per_dataset_data:
        ds_label = DATASETS[ds_key].label
        out_dir = ensemble_root / ds_label
        thr_str = (
            f"threshold={write_threshold} / threshold2={write_threshold2}"
            if mode_2d
            else f"threshold={write_threshold}"
        )
        print(f"\n[{ds_label}] Writing predictions ({thr_str})...")

        completed, write_failed, seq_results = _write_dataset_predictions(
            ds_key, base_dir, initfix_dir, seq_data,
            kp_dir, other_dir, out_dir, write_threshold,
            kp_name, other_name, threshold_metric, skip_completed,
            write_threshold2=write_threshold2,
            combine_or=combine_or,
            woft_compat=woft_compat,
        )
        all_failed_ds = load_failed + write_failed
        all_completed[ds_key] = completed
        all_failed[ds_key] = all_failed_ds

        if seq_results:
            aggregate = aggregate_results(seq_results)
            _save_json(out_dir / "eval_results.json", {
                "aggregate": aggregate,
                "sequences": [
                    {
                        "sequence": r["sequence"],
                        "num_valid": int(r["num_valid"]),
                        "mean_error": float(r["mean_error"]),
                        "median_error": float(r["median_error"]),
                        "p@5": float(r["p@5"]),
                        "p@10": float(r["p@10"]),
                        "p@15": float(r["p@15"]),
                    }
                    for r in seq_results
                ],
                "failed": all_failed_ds,
            })
        print(
            f"  {len(completed)} ok, {len(all_failed_ds)} failed "
            f"(total {len(sequences)})"
        )
        if all_failed_ds:
            for entry in all_failed_ds:
                print(f"  FAILED {entry['sequence']}: {entry['error']}")

    # -----------------------------------------------------------------------
    # Phase 5: write combined manifest + HTML report
    # -----------------------------------------------------------------------
    _save_json(ensemble_root / "ensemble_manifest.json", {
        "ensemble_name": ensemble_name,
        "kp_tracker": kp_name,
        "other_tracker": other_name,
        "threshold_metric": threshold_metric,
        "threshold_used": write_threshold,
        "threshold2_used": write_threshold2,
        "auto_best_threshold": auto_best,
        "auto_best_threshold2": auto_best2,
        "threshold_metric": threshold_metric,
        "threshold_metric2": threshold_metric2,
        "results_roots": results_roots,
        "output_root": ensemble_root,
        "datasets": [
            {
                "dataset": ds_key,
                "dataset_label": DATASETS[ds_key].label,
                "base_dir": base_dir,
                "num_sequences_requested": len(seqs),
                "completed_sequences": all_completed.get(ds_key, []),
                "failed": all_failed.get(ds_key, []),
            }
            for ds_key, base_dir, seqs, *_ in datasets_info
        ],
    })

    if mode_2d:
        report_path = _write_report_2d(
            report_path=ensemble_root / "ensemble_report.html",
            kp_name=kp_name,
            other_name=other_name,
            ensemble_name=ensemble_name,
            metric1_name=threshold_metric,
            metric2_name=threshold_metric2,
            best_threshold1=auto_best,
            best_threshold2=auto_best2,
            datasets_payload=datasets_payload,
            cdn=cdn,
        )
    else:
        report_path = _write_report(
            report_path=ensemble_root / "ensemble_report.html",
            kp_name=kp_name,
            other_name=other_name,
            ensemble_name=ensemble_name,
            metric_name=threshold_metric,
            best_threshold=auto_best,
            datasets_payload=datasets_payload,
            cdn=cdn,
        )
    print(f"\nReport: {report_path}")

    # Summary table: one row per method, one P@5/P@15 pair per dataset.
    if datasets_payload:
        col_headers = []
        for ds in datasets_payload:
            col_headers += [f"{ds['label']} P@5", f"{ds['label']} P@15"]
        col_w = max(len(h) for h in col_headers) + 2
        name_w = max(len(kp_name), len(other_name), len(ensemble_name)) + 2

        header = f"{'method':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in col_headers)
        print(f"\n{'─' * len(header)}")
        print(header)
        print('─' * len(header))
        rows = [
            (other_name,    [ds["other_baseline"]    for ds in datasets_payload]),
            (kp_name,       [ds["kp_baseline"]       for ds in datasets_payload]),
            (ensemble_name, [ds["ensemble_metrics"]  for ds in datasets_payload]),
        ]
        for name, ds_metrics in rows:
            cells = []
            for m in ds_metrics:
                cells += [_fmt(m.get("p5")), _fmt(m.get("p15"))]
            print(f"{name:<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cells))
        print('─' * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold-based ensemble: use keypoint tracker when inlier count "
            "meets threshold, else use fallback tracker."
        )
    )
    parser.add_argument(
        "dataset",
        choices=[*CLI_CHOICES, "both"],
        help=(
            "Dataset to process, or 'both' to run on POT-210 and "
            "PlanarTrackInitfix (test split) together."
        ),
    )
    parser.add_argument(
        "--kp-tracker",
        required=True,
        help=(
            "Keypoint-based tracker name or config path. Must have *_stats.json "
            "files in its results directory."
        ),
    )
    parser.add_argument(
        "--other-tracker",
        required=True,
        help="Fallback tracker name or config path.",
    )
    parser.add_argument(
        "--ensemble-name",
        required=True,
        help="Output name for ensemble results directory.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=THRESHOLD_METRIC_CHOICES,
        default="inliers_total",
        help="Per-frame metric from kp tracker stats used for thresholding.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Threshold value to use for writing predictions. "
            "If not specified, the best threshold (by P@15) from the sweep is used."
        ),
    )
    parser.add_argument(
        "--metric2",
        choices=THRESHOLD_METRIC_CHOICES,
        default=None,
        help=(
            "Optional second per-frame metric for 2D threshold sweep "
            "(e.g. hull_coverage or peeled_hull_coverage). "
            "When set, a 2D grid search over --threshold-metric × --metric2 is run."
        ),
    )
    parser.add_argument(
        "--threshold2",
        type=float,
        default=None,
        help=(
            "Fixed threshold for --metric2. "
            "If not specified, the best (t1, t2) pair by P@15 is selected automatically."
        ),
    )
    parser.add_argument(
        "--combine-or",
        action="store_true",
        help=(
            "Combine metric gates with OR instead of AND (2-D mode only). "
            "Use KP tracker when metric1 >= t1 OR metric2 >= t2; "
            "fall back only when both conditions fail."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Dataset base directory (uses path_config.py default if not specified)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(paths.OUTPUT_DIRECTORY),
        help="Primary top-level directory containing source tracker results",
    )
    parser.add_argument(
        "--extra-results-dirs",
        type=Path,
        nargs="+",
        default=[Path("/mnt/personal/serycjon/flatsam2/loma_v2/")],
        metavar="DIR",
        help=(
            "Additional results roots to search for tracker results "
            "(searched before --results-dir)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(paths.OUTPUT_DIRECTORY),
        help="Top-level directory for ensemble outputs",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to process (default: all)",
    )
    parser.add_argument(
        "--subset",
        help="Subset file name from configs/subsets/<name>.txt",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        help="Direct path to subset file",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Dataset split for datasets that support it",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="Directory with per-sequence initfix files for planartrack-initfix / pti",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Rewrite prediction files even if they already exist",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Load Plotly from CDN instead of inlining it (requires internet).",
    )
    parser.add_argument(
        "--orig-pot-gt",
        action="store_true",
        help="Use original POT annotations instead of WOFT-compatible ones (woft_compat=False).",
    )
    args = parser.parse_args(argv)

    args.results_dir = args.results_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.extra_results_dirs = [d.expanduser() for d in args.extra_results_dirs]

    # Build ordered search list: extra dirs first, then the primary dir.
    results_roots = [*args.extra_results_dirs, args.results_dir]

    missing = [str(d) for d in results_roots if not d.exists()]
    if missing:
        print(f"ERROR: Results director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}")
        return 1

    # Expand "both" → pot + pti(test); otherwise treat as a single dataset.
    if args.dataset == "both":
        raw_datasets = [("pti", None, "test"), ("pot", None, None)]
    else:
        raw_datasets = [(args.dataset, args.base_dir, args.split)]

    datasets_info: list[tuple[str, Path, list[str], Optional[Path]]] = []
    for ds_key, base_dir_override, split_override in raw_datasets:
        ds_info = DATASETS[ds_key]
        base_dir = base_dir_override or getattr(paths, ds_info.base_dir_attr)
        if isinstance(base_dir, str):
            base_dir = Path(base_dir)
        base_dir = base_dir.expanduser()

        initfix_dir = args.initfix_dir
        if ds_info.gt_loader == "pti" and initfix_dir is None:
            initfix_dir = paths.PT_INITFIX_DIR

        # --sequences / --subset / --subset-file apply only to the primary dataset
        # (first in the list). For "both", pot always uses all sequences.
        is_primary = (len(datasets_info) == 0)
        try:
            seqs = get_sequences(
                ds_key,
                base_dir,
                sequences=args.sequences if is_primary else None,
                subset=args.subset if is_primary else None,
                subset_file=args.subset_file if is_primary else None,
                split=split_override,
            )
        except ValueError as exc:
            print(f"ERROR resolving sequences for {ds_key}: {exc}")
            return 1

        if not base_dir.exists():
            print(f"ERROR: Dataset directory not found: {base_dir}")
            return 1

        datasets_info.append((ds_key, base_dir, seqs, initfix_dir))

    try:
        build_threshold_ensemble(
            datasets_info=datasets_info,
            results_roots=results_roots,
            output_root=args.output_dir,
            ensemble_name=args.ensemble_name,
            kp_tracker_spec=args.kp_tracker,
            other_tracker_spec=args.other_tracker,
            threshold_metric=args.threshold_metric,
            threshold=args.threshold,
            threshold_metric2=args.metric2,
            threshold2=args.threshold2,
            combine_or=args.combine_or,
            skip_completed=not args.no_skip_completed,
            cdn=args.cdn,
            woft_compat=not args.orig_pot_gt,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
