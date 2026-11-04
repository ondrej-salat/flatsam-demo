"""Build frame-by-frame oracle tracker from precomputed tracker results.

Loads multiple result folders, compares each tracker prediction against GT on
each evaluated frame, and writes virtual oracle predictions in standard result
format.
"""

import argparse
import base64
import json
import io
import traceback
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from einops import rearrange

from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.metrics import aggregate_results, alignment_error, evaluate_sequence
from flatsam2.path_config import paths
from flatsam2.report_utils import get_plotly_script_tag


SEGMENTS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("all", 1, None),
    ("1:125", 1, 126),
    ("126:250", 126, 251),
    ("251:375", 251, 376),
    ("376+", 376, None),
)
ERROR_PLOT_CLIP_MAX = 40.0


@dataclass(frozen=True)
class TrackerSource:
    """Resolved tracker-result source.

    Args:
        spec: Raw CLI tracker spec.
        name: Resolved tracker name used in directory layout.
        results_dir: Directory containing per-sequence ``.txt`` files.
    """

    spec: str
    name: str
    results_dir: Path


def _make_load_gt_fn(
    dataset: str,
    initfix_dir: Optional[Path],
) -> Callable[[Path, str], Tuple[np.ndarray, np.ndarray]]:
    """Return GT loader for dataset.

    Args:
        dataset: Dataset CLI key.
        initfix_dir: Initfix directory for PTI datasets.

    Returns:
        Callable ``(base_dir, seq_name) -> (annot, valid)``.
    """
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir must be provided for pti GT loading")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def _tracker_name_from_spec(spec: str) -> str:
    """Resolve tracker name from CLI spec.

    Args:
        spec: Tracker name or config path.

    Returns:
        Directory name used under results root.
    """
    return Path(spec).stem


def _resolve_tracker_sources(
    dataset: str,
    results_root: Path,
    tracker_specs: list[str],
) -> list[TrackerSource]:
    """Resolve tracker specs to concrete result directories.

    Args:
        dataset: Dataset CLI key.
        results_root: Top-level results root.
        tracker_specs: Tracker names or config paths.

    Returns:
        Resolved tracker sources.

    Raises:
        ValueError: If names collide or directory is missing.
    """
    dataset_label = DATASETS[dataset].label
    sources = [
        TrackerSource(
            spec=spec,
            name=_tracker_name_from_spec(spec),
            results_dir=results_root / _tracker_name_from_spec(spec) / dataset_label,
        )
        for spec in tracker_specs
    ]

    names = [src.name for src in sources]
    if len(set(names)) != len(names):
        raise ValueError(
            f"Tracker specs resolve to duplicate names: {names}. Use unique tracker names."
        )

    missing = [src.results_dir for src in sources if not src.results_dir.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"Tracker results directory not found: {joined}")

    return sources


def load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    """Load predicted corners for one sequence.

    Args:
        results_dir: Tracker results directory.
        seq_name: Sequence name.

    Returns:
        Predicted corners with shape ``(frames, 2, 4)``.
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
    """Save corners in standard evaluation format.

    Args:
        corners: Predicted corners with shape ``(frames, 2, 4)``.
        output_path: Output ``.txt`` path.
    """
    flat = rearrange(
        corners, "frames xy corners -> frames (corners xy)", corners=4, xy=2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, flat, fmt="%.6f")


def _is_oracle_sequence_complete(
    output_dir: Path,
    seq_name: str,
    num_frames: int,
) -> bool:
    """Check whether oracle output already exists for sequence.

    Args:
        output_dir: Oracle result directory.
        seq_name: Sequence name.
        num_frames: Expected saved frame count.

    Returns:
        True if both result and metadata exist with matching frame count.
    """
    result_file = output_dir / f"{seq_name}.txt"
    meta_file = output_dir / "oracle_meta" / f"{seq_name}.json"
    if not result_file.exists() or not meta_file.exists():
        return False
    try:
        pred_flat = np.loadtxt(result_file)
        if pred_flat.ndim == 1:
            pred_flat = pred_flat[np.newaxis, :]
        return len(pred_flat) == num_frames
    except Exception:
        return False


def _to_jsonable(value: object) -> object:
    """Convert numpy/path values to plain JSON types.

    Args:
        value: Arbitrary nested value.

    Returns:
        JSON-serializable structure.
    """
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
    """Write JSON with stable formatting.

    Args:
        path: Output JSON path.
        payload: Data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
        f.write("\n")


def _load_json(path: Path) -> dict:
    """Load JSON file.

    Args:
        path: Input JSON path.

    Returns:
        Parsed JSON object.
    """
    with path.open() as f:
        return json.load(f)


def _selection_counts_from_meta(meta: dict) -> Counter:
    """Recover tracker-selection counts from per-sequence metadata.

    Args:
        meta: Per-sequence oracle metadata.

    Returns:
        Counter keyed by tracker name.
    """
    if isinstance(meta.get("selected_tracker_counts"), dict):
        return Counter(meta["selected_tracker_counts"])
    selected = meta.get("selected_tracker_by_frame", [])
    return Counter(name for name in selected if name)


def _evaluated_frame_idx(valid: np.ndarray) -> np.ndarray:
    """Return evaluated frame indices for sequence.

    Args:
        valid: Valid-frame mask.

    Returns:
        Frame indices used for evaluation.
    """
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    return np.where(eval_mask)[0]


def _compute_frame_errors(
    gt: np.ndarray,
    valid: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-evaluated-frame alignment errors.

    Invalid predictions become ``inf`` so P@k counts them as failures.

    Args:
        gt: GT corners with shape ``(frames, 2, 4)``.
        valid: Valid-frame mask.
        pred: Predicted corners with shape ``(frames, 2, 4)``.

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


def _segment_errors(eval_idx: np.ndarray, errors: np.ndarray) -> dict[str, np.ndarray]:
    """Split per-frame errors into reporting segments.

    Args:
        eval_idx: Evaluated frame indices.
        errors: Per-evaluated-frame errors.

    Returns:
        Mapping ``segment_label -> errors``.
    """
    out: dict[str, np.ndarray] = {}
    for label, start, stop in SEGMENTS:
        mask = eval_idx >= start
        if stop is not None:
            mask &= eval_idx < stop
        out[label] = errors[mask]
    return out


def _precision_at(errors: np.ndarray, threshold: float) -> float:
    """Compute P@threshold from error vector.

    Args:
        errors: Error vector.
        threshold: Threshold in pixels.

    Returns:
        Precision in percent.
    """
    if len(errors) == 0:
        return float("nan")
    return 100.0 * float(np.mean(errors <= threshold))


def _format_metric(value: float) -> str:
    """Format scalar metric for table output.

    Args:
        value: Metric value.

    Returns:
        Human-readable string.
    """
    if not np.isfinite(value):
        return "-"
    return f"{value:.1f}"


def _print_metric_table(
    title: str,
    method_names: list[str],
    report_store: dict[str, dict[str, list[np.ndarray]]],
    threshold: float,
) -> None:
    """Print markdown table for one threshold family.

    Args:
        title: Section title.
        method_names: Ordered method names.
        report_store: Aggregated segmented errors.
        threshold: Precision threshold.
    """
    segment_labels = [label for label, _, _ in SEGMENTS]
    print(f"\n{title}")
    print("| method | " + " | ".join(segment_labels) + " |")
    print("| --- | " + " | ".join(["---"] * len(segment_labels)) + " |")
    for method_name in method_names:
        values = []
        for label in segment_labels:
            arrays = report_store[method_name][label]
            merged = (
                np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
            )
            values.append(_format_metric(_precision_at(merged, threshold)))
        print("| " + method_name + " | " + " | ".join(values) + " |")


def _print_significant_better_table(
    method_names: list[str],
    better_counts: dict[tuple[str, str], int],
    better_totals: dict[tuple[str, str], int],
    threshold: float,
) -> None:
    """Print pairwise significant-better matrix.

    Cell ``A,B`` means frames where A beats B by at least ``threshold`` pixels.

    Args:
        method_names: Ordered method names.
        better_counts: Ordered-pair significant-win counts.
        better_totals: Ordered-pair comparable-frame counts.
        threshold: Significant-gap threshold.
    """
    print(f"\nFrames Better By >= {threshold:.1f}px")
    print("| method | " + " | ".join(method_names) + " |")
    print("| --- | " + " | ".join(["---"] * len(method_names)) + " |")
    for lhs in method_names:
        row = [lhs]
        for rhs in method_names:
            if lhs == rhs:
                row.append("-")
                continue
            key = (lhs, rhs)
            count = better_counts.get(key, 0)
            total = better_totals.get(key, 0)
            if total == 0:
                row.append("-")
                continue
            pct = 100.0 * count / total
            row.append(f"{count}/{total} ({pct:.1f}%)")
        print("| " + " | ".join(row) + " |")


def _build_segment_metric_rows(
    method_names: list[str],
    report_store: dict[str, dict[str, list[np.ndarray]]],
    threshold: float,
) -> list[dict[str, object]]:
    """Build segmented precision rows for report tables.

    Args:
        method_names: Ordered method names.
        report_store: Aggregated segmented errors.
        threshold: Precision threshold.

    Returns:
        List of row dicts.
    """
    rows: list[dict[str, object]] = []
    for method_name in method_names:
        row = {"method": method_name}
        for label, _, _ in SEGMENTS:
            arrays = report_store[method_name][label]
            merged = (
                np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)
            )
            row[label] = _precision_at(merged, threshold)
        rows.append(row)
    return rows


def _compute_timeline_stats(
    report_sequences: list[dict],
    method_names: list[str],
) -> tuple[int, dict[str, dict[str, list[Optional[float]]]]]:
    """Aggregate per-frame statistics for each method.

    Error-stat curves use finite errors only. Precision curves count ``inf`` as
    failures by using all comparable evaluated frames.

    Args:
        report_sequences: Per-sequence sparse error records.
        method_names: Ordered method names.

    Returns:
        Tuple ``(max_frame_idx, stats_by_method)``.
    """
    max_frame_idx = 0
    for seq in report_sequences:
        if seq["eval_frame_idx"]:
            max_frame_idx = max(max_frame_idx, max(seq["eval_frame_idx"]))

    stats_by_method: dict[str, dict[str, list[Optional[float]]]] = {}
    if max_frame_idx == 0:
        for method_name in method_names:
            stats_by_method[method_name] = {
                "frame_idx": [],
                "coverage_eval": [],
                "coverage_finite": [],
                "mean": [],
                "q10": [],
                "q25": [],
                "q50": [],
                "q75": [],
                "q90": [],
                "p5": [],
                "p15": [],
            }
        return max_frame_idx, stats_by_method

    values = {
        method_name: [[] for _ in range(max_frame_idx + 1)]
        for method_name in method_names
    }

    for seq in report_sequences:
        eval_idx = seq["eval_frame_idx"]
        for method_name in method_names:
            errors = seq["errors"][method_name]
            for frame_idx, err in zip(eval_idx, errors):
                values[method_name][frame_idx].append(float(err))

    for method_name in method_names:
        stats = {
            "frame_idx": list(range(1, max_frame_idx + 1)),
            "coverage_eval": [],
            "coverage_finite": [],
            "mean": [],
            "q10": [],
            "q25": [],
            "q50": [],
            "q75": [],
            "q90": [],
            "p5": [],
            "p15": [],
        }
        for frame_idx in range(1, max_frame_idx + 1):
            arr = np.array(values[method_name][frame_idx], dtype=np.float64)
            stats["coverage_eval"].append(int(len(arr)))
            if len(arr) == 0:
                stats["coverage_finite"].append(0)
                for key in ("mean", "q10", "q25", "q50", "q75", "q90", "p5", "p15"):
                    stats[key].append(None)
                continue

            finite = arr[np.isfinite(arr)]
            stats["coverage_finite"].append(int(len(finite)))
            if len(finite) == 0:
                for key in ("mean", "q10", "q25", "q50", "q75", "q90"):
                    stats[key].append(None)
            else:
                stats["mean"].append(float(np.mean(finite)))
                stats["q10"].append(float(np.quantile(finite, 0.10)))
                stats["q25"].append(float(np.quantile(finite, 0.25)))
                stats["q50"].append(float(np.quantile(finite, 0.50)))
                stats["q75"].append(float(np.quantile(finite, 0.75)))
                stats["q90"].append(float(np.quantile(finite, 0.90)))

            stats["p5"].append(float(np.mean(arr <= 5.0) * 100.0))
            stats["p15"].append(float(np.mean(arr <= 15.0) * 100.0))

        stats_by_method[method_name] = stats

    return max_frame_idx, stats_by_method


def _render_pair_scatter_base64(
    report_sequences: list[dict],
    lhs: str,
    rhs: str,
    clip_max: float = ERROR_PLOT_CLIP_MAX,
) -> str:
    """Render static per-frame scatter plot for ordered method pair.

    Args:
        report_sequences: Per-sequence sparse error records.
        lhs: X-axis method name.
        rhs: Y-axis method name.
        clip_max: Axis clip limit in pixels.

    Returns:
        Base64-encoded PNG image.
    """
    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    clipped = 0

    for seq in report_sequences:
        lhs_err = np.array(seq["errors"][lhs], dtype=np.float64)
        rhs_err = np.array(seq["errors"][rhs], dtype=np.float64)
        finite = np.isfinite(lhs_err) & np.isfinite(rhs_err)
        if not finite.any():
            continue
        x = lhs_err[finite]
        y = rhs_err[finite]
        clipped += int(np.sum((x > clip_max) | (y > clip_max)))
        x = np.clip(x, 0.0, clip_max)
        y = np.clip(y, 0.0, clip_max)
        xs.extend(x.tolist())
        ys.extend(y.tolist())
        colors.extend(
            [
                "#ff7a00"
                if xv + 1e-9 < yv
                else ("#22c55e" if yv + 1e-9 < xv else "#94a3b8")
                for xv, yv in zip(x, y)
            ]
        )

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    if xs:
        ax.scatter(xs, ys, c=colors, s=4, alpha=0.20, linewidths=0, rasterized=True)
    ax.plot([0, clip_max], [0, clip_max], "k--", linewidth=1.0, alpha=0.45)
    ax.set_xlim(0, clip_max)
    ax.set_ylim(0, clip_max)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{lhs} error (px)")
    ax.set_ylabel(f"{rhs} error (px)")
    ax.set_title(f"Per-frame error scatter: {lhs} vs {rhs}")
    subtitle = f"finite pairs={len(xs):,}"
    if clipped > 0:
        subtitle += f", clipped>{clip_max:.0f}px: {clipped:,}"
    ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_pair_scatter_images(
    report_sequences: list[dict],
    pair_method_names: list[str],
) -> dict[str, str]:
    """Build ordered-pair scatter PNGs.

    Args:
        report_sequences: Per-sequence sparse error records.
        pair_method_names: Ordered methods used for pairwise comparison.

    Returns:
        Mapping ``"lhs|||rhs" -> base64_png``.
    """
    images: dict[str, str] = {}
    for lhs in pair_method_names:
        for rhs in pair_method_names:
            if lhs == rhs:
                continue
            images[f"{lhs}|||{rhs}"] = _render_pair_scatter_base64(
                report_sequences,
                lhs,
                rhs,
            )
    return images


def _sequence_image_paths(dataset: str, base_dir: Path, seq_name: str) -> list[Path]:
    """Resolve ordered image paths for sequence.

    Args:
        dataset: Dataset CLI key.
        base_dir: Dataset base directory.
        seq_name: Sequence name.

    Returns:
        Ordered image paths.

    Raises:
        ValueError: If dataset has unsupported GT loader.
    """
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_image_paths(base_dir, seq_name)
    if loader in {"pt", "pti"}:
        return pt_image_paths(base_dir, seq_name)
    raise ValueError(f"Unsupported dataset loader for previews: {loader}")


def _build_sequence_preview_images(
    dataset: str,
    base_dir: Path,
    report_sequences: list[dict],
    max_side: int = 220,
    jpeg_quality: int = 70,
) -> dict[str, str]:
    """Build base64 JPEG first-frame previews keyed by sequence name.

    Args:
        dataset: Dataset CLI key.
        base_dir: Dataset base directory.
        report_sequences: Per-sequence sparse error records.
        max_side: Max preview side length in pixels.
        jpeg_quality: JPEG quality for encoded preview.

    Returns:
        Mapping ``sequence_name -> base64_jpeg``.
    """
    previews: dict[str, str] = {}
    for seq in report_sequences:
        seq_name = str(seq.get("name", ""))
        if not seq_name:
            continue
        try:
            image_paths = _sequence_image_paths(dataset, base_dir, seq_name)
            first_frame = cv2.imread(str(image_paths[0]))
            if first_frame is None:
                continue

            height, width = first_frame.shape[:2]
            max_dim = max(height, width)
            if max_dim > max_side:
                scale = max_side / float(max_dim)
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                first_frame = cv2.resize(
                    first_frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            ok, encoded = cv2.imencode(
                ".jpg",
                first_frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not ok:
                continue
            previews[seq_name] = base64.b64encode(encoded.tobytes()).decode("ascii")
        except Exception:
            continue
    return previews


def _build_metrics_table_html(title: str, rows: list[dict[str, object]]) -> str:
    """Render static HTML metrics table.

    Args:
        title: Table heading.
        rows: Row dicts.

    Returns:
        HTML string.
    """
    headers = [label for label, _, _ in SEGMENTS]
    head_html = "".join(f"<th>{escape(label)}</th>" for label in headers)
    body_rows = []
    for row in rows:
        cells = [f'<td class="method">{escape(str(row["method"]))}</td>']
        for label in headers:
            value = row.get(label)
            txt = (
                "-"
                if value is None or not np.isfinite(value)
                else f"{float(value):.1f}"
            )
            cells.append(f"<td>{txt}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-card">'
        f'<div class="section-label">{escape(title)}</div>'
        '<table class="metrics-table"><thead><tr><th>method</th>'
        f"{head_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _write_interactive_report(
    output_dir: Path,
    dataset: str,
    base_dir: Path,
    oracle_name: str,
    method_names: list[str],
    pair_method_names: list[str],
    table_p5_rows: list[dict[str, object]],
    table_p15_rows: list[dict[str, object]],
    report_sequences: list[dict],
    timeline_stats: dict[str, dict[str, list[Optional[float]]]],
    max_frame_idx: int,
    default_significant_threshold: float,
    cdn: bool = False,
) -> Path:
    """Write self-contained interactive HTML oracle report.

    Args:
        output_dir: Oracle output directory.
        dataset: Dataset key.
        base_dir: Dataset base directory.
        oracle_name: Oracle method name.
        method_names: Ordered method names.
        pair_method_names: Ordered methods used for pairwise comparison.
        table_p5_rows: Segmented P@5 table rows.
        table_p15_rows: Segmented P@15 table rows.
        report_sequences: Per-sequence sparse error records.
        timeline_stats: Precomputed per-frame method statistics.
        max_frame_idx: Largest evaluated frame index.
        default_significant_threshold: Default significant gap threshold.

    Returns:
        Written HTML path.
    """
    pair_scatter_png = _build_pair_scatter_images(report_sequences, pair_method_names)
    sequence_preview_jpg = _build_sequence_preview_images(
        dataset,
        base_dir,
        report_sequences,
    )
    payload = {
        "dataset": dataset,
        "dataset_label": DATASETS[dataset].label,
        "oracle_name": oracle_name,
        "method_names": method_names,
        "pair_method_names": pair_method_names,
        "max_frame_idx": max_frame_idx,
        "default_significant_threshold": default_significant_threshold,
        "error_plot_clip_max": ERROR_PLOT_CLIP_MAX,
        "timeline_stats": timeline_stats,
        "report_sequences": report_sequences,
        "pair_scatter_png": pair_scatter_png,
        "sequence_preview_jpg": sequence_preview_jpg,
    }
    payload_json = json.dumps(_to_jsonable(payload), separators=(",", ":"))
    p5_table_html = _build_metrics_table_html("P@5", table_p5_rows)
    p15_table_html = _build_metrics_table_html("P@15", table_p15_rows)
    report_path = output_dir / "oracle_report.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(oracle_name)} oracle report</title>
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
    .controls-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px 18px;
      align-items: center;
    }}
    .ctrl {{ display: flex; align-items: center; gap: 8px; }}
    .ctrl label {{ font-weight: 600; white-space: nowrap; }}
    .ctrl input[type=range] {{ flex: 1; }}
    .ctrl input[type=number], .ctrl select {{
      min-width: 88px;
      padding: 3px 6px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #f8fafc;
    }}
    .plots-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 12px;
    }}
    .plot-title {{ font-weight: 600; margin-bottom: 6px; color: #374151; }}
    .plot-box {{ min-height: 330px; }}
    .matrix-wrap {{ overflow-x: auto; }}
    .pair-matrix {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    .pair-matrix th, .pair-matrix td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: center; }}
    .pair-matrix th:first-child, .pair-matrix td:first-child {{ text-align: left; font-weight: 600; }}
    .pair-summary {{ color: #4b5563; margin: 6px 0 10px; }}
    #pair-scatter-img {{ display: block; max-width: 100%; height: auto; border-radius: 8px; }}
    .seq-preview-wrap {{
      display: grid;
      grid-template-columns: 200px 1fr;
      gap: 10px;
      margin-top: 8px;
      align-items: start;
    }}
    .seq-preview-media {{
      width: 200px;
      min-height: 120px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #f8fafc;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    #seq-preview-img {{ max-width: 100%; max-height: 160px; display: none; }}
    #seq-preview-empty {{ color: #6b7280; font-size: 12px; padding: 8px; text-align: center; }}
    #seq-preview-name {{ font-weight: 600; color: #0f172a; margin-bottom: 4px; }}
    #seq-preview-metrics {{ color: #475569; font-variant-numeric: tabular-nums; }}
    @media (max-width: 700px) {{
      .seq-preview-wrap {{ grid-template-columns: 1fr; }}
      .seq-preview-media {{ width: 100%; }}
      #seq-preview-img {{ max-height: 190px; }}
    }}
    .note {{ color: #6b7280; font-size: 12px; margin-top: 8px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>{escape(oracle_name)} oracle report</h1>
  <div class="meta">Dataset: {escape(DATASETS[dataset].label)} | methods: {len(method_names)} | sequences: {len(report_sequences)}</div>

  <div class="card">
    <div class="section-label">Segmented Precision</div>
    <div class="tables-grid">
      {p5_table_html}
      {p15_table_html}
    </div>
  </div>

  <div class="card">
    <div class="section-label">Controls</div>
    <div class="controls-grid">
      <div class="ctrl"><label for="pair-a">Method A</label><select id="pair-a"></select></div>
      <div class="ctrl"><label for="pair-b">Method B</label><select id="pair-b"></select></div>
      <div class="ctrl"><label for="sig-thr">Significant gap</label><input id="sig-thr" type="range" min="0" max="50" step="0.5"><input id="sig-thr-num" type="number" min="0" max="200" step="0.5"></div>
      <div class="ctrl"><label for="error-stat">Error stat</label><select id="error-stat">
        <option value="mean">Mean</option>
        <option value="q10">P10</option>
        <option value="q25">P25</option>
        <option value="q50" selected>Median</option>
        <option value="q75">P75</option>
        <option value="q90">P90</option>
      </select></div>
      <div class="ctrl"><label for="precision-kind">Precision curve</label><select id="precision-kind">
        <option value="p5">P@5</option>
        <option value="p15" selected>P@15</option>
      </select></div>
      <div class="ctrl"><label for="scatter-metric">Scatter metric</label><select id="scatter-metric">
        <option value="p5">P@5</option>
        <option value="p15" selected>P@15</option>
      </select></div>
    </div>
    <div class="note">
      Error-stat timelines use finite alignment errors only and display is clipped to {ERROR_PLOT_CLIP_MAX:.0f}px. Precision and significant-better plots count invalid predictions as failures via infinite error. Pairwise comparison widgets use source trackers only, not oracle. Threshold control updates pairwise matrix and pairwise timeline live.
    </div>
  </div>

  <div class="plots-grid">
    <div class="card"><div class="plot-title" id="title-error"></div><div id="plot-error" class="plot-box"></div></div>
    <div class="card"><div class="plot-title" id="title-precision"></div><div id="plot-precision" class="plot-box"></div></div>
    <div class="card"><div class="plot-title" id="title-better"></div><div id="plot-better" class="plot-box"></div><div id="pair-summary" class="pair-summary"></div></div>
    <div class="card">
      <div class="plot-title" id="title-seq-scatter"></div>
      <div id="plot-seq-scatter" class="plot-box"></div>
      <div class="seq-preview-wrap">
        <div class="seq-preview-media">
          <img id="seq-preview-img" alt="sequence preview">
          <div id="seq-preview-empty">Hover or click point</div>
        </div>
        <div>
          <div id="seq-preview-name">No sequence selected</div>
          <div id="seq-preview-metrics">-</div>
        </div>
      </div>
    </div>
    <div class="card"><div class="plot-title">Static scatter</div><img id="pair-scatter-img" alt="pair scatter"></div>
  </div>

  <div class="card">
    <div class="section-label">Pairwise Significant Better Matrix</div>
    <div class="matrix-wrap"><table class="pair-matrix" id="pair-matrix"></table></div>
  </div>

  <script>
    const DATA = {payload_json};
    const COLORS = ['#111827', '#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0891b2', '#be123c'];
    const methodColors = Object.fromEntries(DATA.method_names.map((name, i) => [name, COLORS[i % COLORS.length]]));
    const pairAEl = document.getElementById('pair-a');
    const pairBEl = document.getElementById('pair-b');
    const sigThrEl = document.getElementById('sig-thr');
    const sigThrNumEl = document.getElementById('sig-thr-num');
    const errorStatEl = document.getElementById('error-stat');
    const precisionKindEl = document.getElementById('precision-kind');
    const scatterMetricEl = document.getElementById('scatter-metric');
    const seqPreviewImgEl = document.getElementById('seq-preview-img');
    const seqPreviewEmptyEl = document.getElementById('seq-preview-empty');
    const seqPreviewNameEl = document.getElementById('seq-preview-name');
    const seqPreviewMetricsEl = document.getElementById('seq-preview-metrics');
    let scatterHandlersBound = false;
    const scatterState = {{ lhs: '', rhs: '', metricLabel: 'P@15' }};

    function fillMethodSelects() {{
      for (const el of [pairAEl, pairBEl]) {{
        el.innerHTML = '';
        for (const name of DATA.pair_method_names) {{
          const opt = document.createElement('option');
          opt.value = name;
          opt.textContent = name;
          el.appendChild(opt);
        }}
      }}
      pairAEl.value = DATA.pair_method_names[0] || '';
      pairBEl.value = DATA.pair_method_names[1] || DATA.pair_method_names[0] || '';
      if (pairAEl.value === pairBEl.value && DATA.pair_method_names.length > 1) {{
        pairBEl.value = DATA.pair_method_names[1];
      }}
    }}

    function syncThreshold(fromRange) {{
      const value = fromRange ? sigThrEl.value : sigThrNumEl.value;
      sigThrEl.value = value;
      sigThrNumEl.value = value;
      renderPairwise();
    }}

    function pairKey(lhs, rhs) {{
      return lhs + '|||' + rhs;
    }}

    function computePairTimeline(lhs, rhs, threshold) {{
      const maxF = DATA.max_frame_idx;
      const aBetter = new Array(maxF + 1).fill(0);
      const bBetter = new Array(maxF + 1).fill(0);
      const comparable = new Array(maxF + 1).fill(0);
      let totalABetter = 0;
      let totalBBetter = 0;
      let totalComparable = 0;

      for (const seq of DATA.report_sequences) {{
        const idx = seq.eval_frame_idx;
        const ea = seq.errors[lhs];
        const eb = seq.errors[rhs];
        for (let i = 0; i < idx.length; i++) {{
          const f = idx[i];
          const av = ea[i];
          const bv = eb[i];
          const aFinite = Number.isFinite(av);
          const bFinite = Number.isFinite(bv);
          const comp = aFinite || bFinite;
          if (!comp) continue;
          const aVal = aFinite ? av : Infinity;
          const bVal = bFinite ? bv : Infinity;
          comparable[f] += 1;
          totalComparable += 1;
          if ((bVal - aVal) >= threshold) {{
            aBetter[f] += 1;
            totalABetter += 1;
          }}
          if ((aVal - bVal) >= threshold) {{
            bBetter[f] += 1;
            totalBBetter += 1;
          }}
        }}
      }}

      const frameIdx = [];
      const aPct = [];
      const bPct = [];
      const cov = [];
      for (let f = 1; f <= maxF; f++) {{
        frameIdx.push(f);
        if (comparable[f] > 0) {{
          aPct.push(100 * aBetter[f] / comparable[f]);
          bPct.push(100 * bBetter[f] / comparable[f]);
          cov.push(comparable[f]);
        }} else {{
          aPct.push(null);
          bPct.push(null);
          cov.push(0);
        }}
      }}
      return {{ frameIdx, aPct, bPct, cov, totalABetter, totalBBetter, totalComparable }};
    }}

    function computePairMatrix(threshold) {{
      const counts = new Map();
      const totals = new Map();
      for (const lhs of DATA.pair_method_names) {{
        for (const rhs of DATA.pair_method_names) {{
          if (lhs === rhs) continue;
          counts.set(pairKey(lhs, rhs), 0);
          totals.set(pairKey(lhs, rhs), 0);
        }}
      }}

      for (const seq of DATA.report_sequences) {{
        const idx = seq.eval_frame_idx;
        for (const lhs of DATA.pair_method_names) {{
          const ea = seq.errors[lhs];
          for (const rhs of DATA.pair_method_names) {{
            if (lhs === rhs) continue;
            const eb = seq.errors[rhs];
            const key = pairKey(lhs, rhs);
            for (let i = 0; i < idx.length; i++) {{
              const av = ea[i];
              const bv = eb[i];
              const aFinite = Number.isFinite(av);
              const bFinite = Number.isFinite(bv);
              const comp = aFinite || bFinite;
              if (!comp) continue;
              const aVal = aFinite ? av : Infinity;
              const bVal = bFinite ? bv : Infinity;
              totals.set(key, totals.get(key) + 1);
              if ((bVal - aVal) >= threshold) counts.set(key, counts.get(key) + 1);
            }}
          }}
        }}
      }}
      return {{ counts, totals }};
    }}

    function renderMatrix(threshold) {{
      const matrixEl = document.getElementById('pair-matrix');
      const {{ counts, totals }} = computePairMatrix(threshold);
      const head = '<thead><tr><th>method</th>' + DATA.pair_method_names.map(n => `<th>${{n}}</th>`).join('') + '</tr></thead>';
      const rows = DATA.pair_method_names.map(lhs => {{
        const cells = [`<td>${{lhs}}</td>`];
        for (const rhs of DATA.pair_method_names) {{
          if (lhs === rhs) {{
            cells.push('<td>-</td>');
            continue;
          }}
          const key = pairKey(lhs, rhs);
          const total = totals.get(key) || 0;
          if (!total) {{
            cells.push('<td>-</td>');
            continue;
          }}
          const count = counts.get(key) || 0;
          const pct = 100 * count / total;
          cells.push(`<td>${{count}}/${{total}} (${{pct.toFixed(1)}}%)</td>`);
        }}
        return '<tr>' + cells.join('') + '</tr>';
      }}).join('');
      matrixEl.innerHTML = head + '<tbody>' + rows + '</tbody>';
    }}

    function renderErrorPlot() {{
      const statKey = errorStatEl.value;
      const traces = DATA.method_names.map(name => {{
        const s = DATA.timeline_stats[name];
        return {{
          x: s.frame_idx,
          y: s[statKey].map(v => v == null ? null : Math.min(v, DATA.error_plot_clip_max)),
          mode: 'lines',
          name,
          line: {{ color: methodColors[name], width: name === DATA.oracle_name ? 3 : 2 }},
          customdata: s.coverage_finite,
          hovertemplate: 'frame %{{x}}<br>' + name + ': %{{y:.2f}}px (clipped)<br>finite seqs: %{{customdata}}<extra></extra>',
        }};
      }});
      document.getElementById('title-error').textContent = 'Alignment error over time (' + errorStatEl.options[errorStatEl.selectedIndex].text + ')';
      Plotly.react('plot-error', traces, {{
        margin: {{ l: 48, r: 12, t: 10, b: 80 }},
        xaxis: {{ title: 'Frame index' }},
        yaxis: {{ title: 'Alignment error (px)', range: [0, DATA.error_plot_clip_max] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
    }}

    function renderPrecisionPlot() {{
      const key = precisionKindEl.value;
      const traces = DATA.method_names.map(name => {{
        const s = DATA.timeline_stats[name];
        return {{
          x: s.frame_idx,
          y: s[key],
          mode: 'lines',
          name,
          line: {{ color: methodColors[name], width: name === DATA.oracle_name ? 3 : 2 }},
          customdata: s.coverage_eval,
          hovertemplate: 'frame %{{x}}<br>' + name + ': %{{y:.1f}}%<br>eval seqs: %{{customdata}}<extra></extra>',
        }};
      }});
      document.getElementById('title-precision').textContent = (key === 'p5' ? 'P@5' : 'P@15') + ' over time';
      Plotly.react('plot-precision', traces, {{
        margin: {{ l: 48, r: 12, t: 10, b: 80 }},
        xaxis: {{ title: 'Frame index' }},
        yaxis: {{ title: 'Precision (%)', range: [0, 100] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
    }}

    function ensureDistinctPair(changed) {{
      if (pairAEl.value !== pairBEl.value) return;
      const other = DATA.pair_method_names.find(n => n !== (changed === 'a' ? pairAEl.value : pairBEl.value));
      if (!other) return;
      if (changed === 'a') pairBEl.value = other;
      else pairAEl.value = other;
    }}

    function sequencePrecisionForMethod(seq, methodName, metricKey) {{
      const errors = seq.errors[methodName] || [];
      const threshold = metricKey === 'p5' ? 5.0 : 15.0;
      if (!errors.length) return null;
      let hits = 0;
      for (const val of errors) {{
        if (Number.isFinite(val) && val <= threshold) hits += 1;
      }}
      return 100 * hits / errors.length;
    }}

    function setSequencePreview(seqName, xVal, yVal, lhs, rhs, metricLabel) {{
      seqPreviewNameEl.textContent = seqName;
      seqPreviewMetricsEl.textContent =
        lhs + ' ' + metricLabel + ': ' + xVal.toFixed(1) + '% | ' +
        rhs + ' ' + metricLabel + ': ' + yVal.toFixed(1) + '%';
      const src = DATA.sequence_preview_jpg[seqName];
      if (!src) {{
        seqPreviewImgEl.style.display = 'none';
        seqPreviewImgEl.removeAttribute('src');
        seqPreviewEmptyEl.textContent = 'No preview image';
        seqPreviewEmptyEl.style.display = 'block';
        return;
      }}
      seqPreviewImgEl.src = 'data:image/jpeg;base64,' + src;
      seqPreviewImgEl.style.display = 'block';
      seqPreviewEmptyEl.style.display = 'none';
    }}

    function clearSequencePreview() {{
      seqPreviewImgEl.style.display = 'none';
      seqPreviewImgEl.removeAttribute('src');
      seqPreviewNameEl.textContent = 'No sequence selected';
      seqPreviewMetricsEl.textContent = '-';
      seqPreviewEmptyEl.textContent = 'Hover or click point';
      seqPreviewEmptyEl.style.display = 'block';
    }}

    function bindSequenceScatterEvents() {{
      if (scatterHandlersBound) return;
      const plotEl = document.getElementById('plot-seq-scatter');
      plotEl.on('plotly_hover', (eventData) => {{
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const point = eventData.points[0];
        if (typeof point.text !== 'string') return;
        setSequencePreview(
          point.text,
          Number(point.x),
          Number(point.y),
          scatterState.lhs,
          scatterState.rhs,
          scatterState.metricLabel,
        );
      }});
      plotEl.on('plotly_click', (eventData) => {{
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const point = eventData.points[0];
        if (typeof point.text !== 'string') return;
        setSequencePreview(
          point.text,
          Number(point.x),
          Number(point.y),
          scatterState.lhs,
          scatterState.rhs,
          scatterState.metricLabel,
        );
      }});
      scatterHandlersBound = true;
    }}

    function renderSequenceScatter() {{
      const lhs = pairAEl.value;
      const rhs = pairBEl.value;
      const metricKey = scatterMetricEl.value;
      const metricLabel = metricKey === 'p5' ? 'P@5' : 'P@15';
      scatterState.lhs = lhs;
      scatterState.rhs = rhs;
      scatterState.metricLabel = metricLabel;
      const xs = [];
      const ys = [];
      const labels = [];
      for (const seq of DATA.report_sequences) {{
        const lhsVal = sequencePrecisionForMethod(seq, lhs, metricKey);
        const rhsVal = sequencePrecisionForMethod(seq, rhs, metricKey);
        if (lhsVal == null || rhsVal == null) continue;
        xs.push(lhsVal);
        ys.push(rhsVal);
        labels.push(seq.name);
      }}

      const diag = {{
        x: [0, 100],
        y: [0, 100],
        mode: 'lines',
        name: 'y = x',
        line: {{ color: '#94a3b8', width: 1.5, dash: 'dot' }},
        hoverinfo: 'skip',
      }};
      const points = {{
        x: xs,
        y: ys,
        text: labels,
        mode: 'markers',
        name: 'sequence',
        marker: {{
          size: 8,
          color: '#0f766e',
          line: {{ color: '#134e4a', width: 0.8 }},
          opacity: 0.85,
        }},
        hovertemplate:
          '%{{text}}<br>' +
          lhs + ' ' + metricLabel + ': %{{x:.1f}}%<br>' +
          rhs + ' ' + metricLabel + ': %{{y:.1f}}%<extra></extra>',
      }};

      document.getElementById('title-seq-scatter').textContent = 'Per-sequence scatter (' + metricLabel + ')';
      Plotly.react('plot-seq-scatter', [diag, points], {{
        margin: {{ l: 56, r: 12, t: 10, b: 80 }},
        xaxis: {{ title: lhs + ' ' + metricLabel + ' (%)', range: [0, 100] }},
        yaxis: {{ title: rhs + ' ' + metricLabel + ' (%)', range: [0, 100] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});
      bindSequenceScatterEvents();
      if (labels.length > 0) {{
        setSequencePreview(labels[0], xs[0], ys[0], lhs, rhs, metricLabel);
      }} else {{
        clearSequencePreview();
      }}
    }}

    function renderPairwise() {{
      const lhs = pairAEl.value;
      const rhs = pairBEl.value;
      const thr = parseFloat(sigThrEl.value || '0');
      const pair = computePairTimeline(lhs, rhs, thr);
      const traces = [
        {{
          x: pair.frameIdx,
          y: pair.aPct,
          mode: 'lines',
          name: lhs + ' better',
          line: {{ color: methodColors[lhs], width: 2.5 }},
          customdata: pair.cov,
          hovertemplate: 'frame %{{x}}<br>' + lhs + ' better: %{{y:.1f}}%<br>comparable seqs: %{{customdata}}<extra></extra>',
        }},
        {{
          x: pair.frameIdx,
          y: pair.bPct,
          mode: 'lines',
          name: rhs + ' better',
          line: {{ color: methodColors[rhs], width: 2.5 }},
          customdata: pair.cov,
          hovertemplate: 'frame %{{x}}<br>' + rhs + ' better: %{{y:.1f}}%<br>comparable seqs: %{{customdata}}<extra></extra>',
        }},
      ];
      document.getElementById('title-better').textContent = 'Per-frame significant wins (' + thr.toFixed(1) + ' px gap)';
      Plotly.react('plot-better', traces, {{
        margin: {{ l: 48, r: 12, t: 10, b: 80 }},
        xaxis: {{ title: 'Frame index' }},
        yaxis: {{ title: '% of comparable sequences', range: [0, 100] }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.25, xanchor: 'center', x: 0.5 }},
      }}, {{responsive: true, displayModeBar: false}});

      const summary = document.getElementById('pair-summary');
      const total = pair.totalComparable;
      const aPct = total ? (100 * pair.totalABetter / total) : 0;
      const bPct = total ? (100 * pair.totalBBetter / total) : 0;
      summary.textContent = `${{lhs}} better by >= ${{thr.toFixed(1)}}px: ${{pair.totalABetter}}/${{total}} (${{aPct.toFixed(1)}}%) | ` +
                            `${{rhs}} better by >= ${{thr.toFixed(1)}}px: ${{pair.totalBBetter}}/${{total}} (${{bPct.toFixed(1)}}%)`;

      const img = document.getElementById('pair-scatter-img');
      img.src = 'data:image/png;base64,' + DATA.pair_scatter_png[pairKey(lhs, rhs)];
      img.alt = lhs + ' vs ' + rhs + ' scatter';

      renderSequenceScatter();
      renderMatrix(thr);
    }}

    function boot() {{
      fillMethodSelects();
      sigThrEl.value = String(DATA.default_significant_threshold);
      sigThrNumEl.value = String(DATA.default_significant_threshold);
      pairAEl.addEventListener('change', () => {{ ensureDistinctPair('a'); renderPairwise(); }});
      pairBEl.addEventListener('change', () => {{ ensureDistinctPair('b'); renderPairwise(); }});
      sigThrEl.addEventListener('input', () => syncThreshold(true));
      sigThrNumEl.addEventListener('input', () => syncThreshold(false));
      errorStatEl.addEventListener('change', renderErrorPlot);
      precisionKindEl.addEventListener('change', renderPrecisionPlot);
      scatterMetricEl.addEventListener('change', renderSequenceScatter);
      renderErrorPlot();
      renderPrecisionPlot();
      renderPairwise();
    }}

    boot();
  </script>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path


def _build_oracle_sequence(
    gt: np.ndarray,
    valid: np.ndarray,
    tracker_predictions: list[np.ndarray],
    tracker_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Select best tracker prediction for each evaluated frame.

    Args:
        gt: GT corners with shape ``(frames, 2, 4)``.
        valid: Valid-frame mask with shape ``(frames,)``.
        tracker_predictions: One prediction tensor per tracker.
        tracker_names: Tracker names aligned with ``tracker_predictions``.

    Returns:
        Tuple ``(oracle_predictions, metadata)``.
    """
    if not tracker_predictions:
        raise ValueError("At least one tracker prediction tensor is required")

    num_frames = len(gt)
    oracle_pred = tracker_predictions[0].copy()
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    eval_idx = np.where(eval_mask)[0]

    selected_tracker_by_frame: list[Optional[str]] = [None] * num_frames
    selected_error_by_frame: list[Optional[float]] = [None] * num_frames

    if len(eval_idx) == 0:
        return oracle_pred, {
            "num_frames": num_frames,
            "eval_mask_by_frame": eval_mask.tolist(),
            "selected_tracker_by_frame": selected_tracker_by_frame,
            "selected_error_by_frame": selected_error_by_frame,
            "selected_tracker_counts": {},
            "non_eval_source_tracker": tracker_names[0],
        }

    gt_eval = gt[eval_idx]
    errors = np.full(
        (len(tracker_predictions), len(eval_idx)), np.nan, dtype=np.float64
    )

    for tracker_i, pred in enumerate(tracker_predictions):
        pred_eval = pred[eval_idx]
        finite = np.all(np.isfinite(pred_eval), axis=(1, 2))
        if finite.any():
            errors[tracker_i, finite] = alignment_error(
                gt_eval[finite], pred_eval[finite]
            )

    has_choice = np.any(np.isfinite(errors), axis=0)
    best_tracker_idx = np.zeros(len(eval_idx), dtype=np.int64)
    if has_choice.any():
        best_tracker_idx[has_choice] = np.nanargmin(errors[:, has_choice], axis=0)

    for pos, frame_idx in enumerate(eval_idx):
        if not has_choice[pos]:
            continue
        tracker_i = int(best_tracker_idx[pos])
        oracle_pred[frame_idx] = tracker_predictions[tracker_i][frame_idx]
        selected_tracker_by_frame[frame_idx] = tracker_names[tracker_i]
        selected_error_by_frame[frame_idx] = float(errors[tracker_i, pos])

    counts = Counter(name for name in selected_tracker_by_frame if name is not None)
    return oracle_pred, {
        "num_frames": num_frames,
        "eval_mask_by_frame": eval_mask.tolist(),
        "selected_tracker_by_frame": selected_tracker_by_frame,
        "selected_error_by_frame": selected_error_by_frame,
        "selected_tracker_counts": dict(sorted(counts.items())),
        "non_eval_source_tracker": tracker_names[0],
    }


def _save_eval_results(
    output_dir: Path,
    sequence_results: list[dict],
    failed: list[dict],
) -> None:
    """Write aggregate evaluation JSON in standard shape.

    Args:
        output_dir: Oracle output directory.
        sequence_results: Per-sequence metric dicts.
        failed: Failed-sequence records.
    """
    if not sequence_results:
        return

    aggregate = aggregate_results(sequence_results)
    payload = {
        "aggregate": aggregate,
        "sequences": [
            {
                "sequence": res["sequence"],
                "num_valid": int(res["num_valid"]),
                "mean_error": float(res["mean_error"]),
                "median_error": float(res["median_error"]),
                "p@5": float(res["p@5"]),
                "p@10": float(res["p@10"]),
                "p@15": float(res["p@15"]),
            }
            for res in sequence_results
        ],
        "failed": failed,
    }
    _save_json(output_dir / "eval_results.json", payload)


def _save_manifest(
    output_dir: Path,
    dataset: str,
    base_dir: Path,
    results_root: Path,
    sources: list[TrackerSource],
    sequences: list[str],
    completed: list[str],
    failed: list[dict],
    selection_counts: Counter,
) -> None:
    """Write run-level oracle metadata.

    Args:
        output_dir: Oracle output directory.
        dataset: Dataset CLI key.
        base_dir: Dataset root.
        results_root: Tracker-results root.
        sources: Resolved tracker sources.
        sequences: Requested sequences.
        completed: Completed sequence names.
        failed: Failed-sequence records.
        selection_counts: Global tracker selection counts.
    """
    payload = {
        "dataset": dataset,
        "dataset_label": DATASETS[dataset].label,
        "base_dir": base_dir,
        "results_root": results_root,
        "output_dir": output_dir,
        "trackers": [
            {
                "spec": src.spec,
                "name": src.name,
                "results_dir": src.results_dir,
            }
            for src in sources
        ],
        "num_sequences_requested": len(sequences),
        "completed_sequences": completed,
        "failed": failed,
        "selection_counts": dict(sorted(selection_counts.items())),
        "meta_dir": output_dir / "oracle_meta",
        "eval_results": output_dir / "eval_results.json",
        "report_html": output_dir / "oracle_report.html",
    }
    _save_json(output_dir / "oracle_manifest.json", payload)


def build_oracle_results(
    dataset: str,
    base_dir: Path,
    results_root: Path,
    output_root: Path,
    oracle_name: str,
    tracker_specs: list[str],
    sequences: list[str],
    initfix_dir: Optional[Path],
    skip_completed: bool,
    significant_better_threshold: float,
    cdn: bool = False,
) -> None:
    """Build oracle result set over multiple trackers.

    Args:
        dataset: Dataset CLI key.
        base_dir: Dataset root.
        results_root: Top-level tracker-results root.
        output_root: Top-level oracle-results root.
        oracle_name: Output tracker name.
        tracker_specs: Tracker names or config paths.
        sequences: Sequence names to process.
        initfix_dir: Initfix directory for PTI datasets.
        skip_completed: Skip sequences with existing output.
    """
    sources = _resolve_tracker_sources(dataset, results_root, tracker_specs)
    output_dir = output_root / oracle_name / DATASETS[dataset].label
    output_dir_resolved = output_dir.resolve()
    for src in sources:
        if src.results_dir.resolve() == output_dir_resolved:
            raise ValueError(
                f"Oracle output directory collides with source tracker directory: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    load_gt_fn = _make_load_gt_fn(dataset, initfix_dir)
    tracker_names = [src.name for src in sources]
    if oracle_name in tracker_names:
        raise ValueError(
            f"oracle_name {oracle_name!r} collides with source tracker name"
        )
    report_method_names = [oracle_name, *tracker_names]
    report_store = {
        method_name: {label: [] for label, _, _ in SEGMENTS}
        for method_name in report_method_names
    }
    report_sequences: list[dict] = []
    better_counts: Counter = Counter()
    better_totals: Counter = Counter()

    print(f"Building oracle ensemble: {oracle_name}")
    print(f"Dataset: {DATASETS[dataset].label}")
    print(f"Results root: {results_root}")
    print(f"Output directory: {output_dir}")
    print(f"Trackers: {', '.join(tracker_names)}")

    completed_sequences: list[str] = []
    failed: list[dict] = []
    selection_counts: Counter = Counter()
    sequence_results: list[dict] = []

    _save_manifest(
        output_dir,
        dataset,
        base_dir,
        results_root,
        sources,
        sequences,
        completed_sequences,
        failed,
        selection_counts,
    )

    for seq_name in sequences:
        print(f"\n{'=' * 60}")
        print(f"Sequence: {seq_name}")
        print(f"{'=' * 60}")
        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            tracker_predictions = [
                load_predictions(src.results_dir, seq_name) for src in sources
            ]
            method_predictions = {
                src.name: pred for src, pred in zip(sources, tracker_predictions)
            }

            frame_counts = {
                "gt": len(gt_corners),
                **{
                    src.name: len(pred)
                    for src, pred in zip(sources, tracker_predictions)
                },
            }
            min_len = min(frame_counts.values())
            if len(set(frame_counts.values())) != 1:
                print(
                    f"WARNING: frame count mismatch, truncating to {min_len}: {frame_counts}"
                )

            gt_corners = gt_corners[:min_len]
            valid = valid[:min_len]
            tracker_predictions = [pred[:min_len] for pred in tracker_predictions]
            method_predictions = {
                src.name: pred for src, pred in zip(sources, tracker_predictions)
            }

            if skip_completed and _is_oracle_sequence_complete(
                output_dir, seq_name, min_len
            ):
                print(f"Skipping {seq_name} (already complete)")
                existing_pred = load_predictions(output_dir, seq_name)[:min_len]
                existing_meta = _load_json(
                    output_dir / "oracle_meta" / f"{seq_name}.json"
                )
                seq_result = evaluate_sequence(
                    gt_corners,
                    existing_pred,
                    valid,
                    use_best_rotation=False,
                )
                seq_result["sequence"] = seq_name
                sequence_results.append(seq_result)
                selection_counts.update(_selection_counts_from_meta(existing_meta))
                method_predictions[oracle_name] = existing_pred
                completed_sequences.append(seq_name)
            else:
                oracle_pred, seq_meta = _build_oracle_sequence(
                    gt_corners,
                    valid,
                    tracker_predictions,
                    tracker_names,
                )

                output_file = output_dir / f"{seq_name}.txt"
                save_corners(oracle_pred, output_file)
                print(f"Saved oracle results to: {output_file}")

                seq_result = evaluate_sequence(
                    gt_corners,
                    oracle_pred,
                    valid,
                    use_best_rotation=False,
                )
                seq_result["sequence"] = seq_name
                sequence_results.append(seq_result)

                seq_meta_payload = {
                    "sequence": seq_name,
                    "trackers": tracker_names,
                    "frame_counts": frame_counts,
                    **seq_meta,
                }
                meta_path = output_dir / "oracle_meta" / f"{seq_name}.json"
                _save_json(meta_path, seq_meta_payload)
                print(f"Saved oracle metadata to: {meta_path}")

                selection_counts.update(seq_meta["selected_tracker_counts"])
                method_predictions[oracle_name] = oracle_pred
                completed_sequences.append(seq_name)

            seq_errors: dict[str, np.ndarray] = {}
            ref_eval_idx: Optional[np.ndarray] = None
            for method_name in report_method_names:
                eval_idx, errors = _compute_frame_errors(
                    gt_corners,
                    valid,
                    method_predictions[method_name],
                )
                if ref_eval_idx is None:
                    ref_eval_idx = eval_idx
                elif not np.array_equal(ref_eval_idx, eval_idx):
                    raise ValueError(
                        f"Evaluation-frame mismatch while aggregating reporting stats for {seq_name}"
                    )
                for label, segment_errors in _segment_errors(eval_idx, errors).items():
                    report_store[method_name][label].append(segment_errors)
                seq_errors[method_name] = errors

            report_sequences.append(
                {
                    "name": seq_name,
                    "num_frames": min_len,
                    "eval_frame_idx": ref_eval_idx.tolist()
                    if ref_eval_idx is not None
                    else [],
                    "errors": {
                        method_name: seq_errors[method_name].tolist()
                        for method_name in report_method_names
                    },
                }
            )

            for lhs in report_method_names:
                for rhs in report_method_names:
                    if lhs == rhs:
                        continue
                    comparable = np.isfinite(seq_errors[lhs]) | np.isfinite(
                        seq_errors[rhs]
                    )
                    better_totals[(lhs, rhs)] += int(comparable.sum())
                    better_counts[(lhs, rhs)] += int(
                        np.sum(
                            comparable
                            & (
                                (seq_errors[rhs] - seq_errors[lhs])
                                >= significant_better_threshold
                            )
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
            print(f"ERROR processing {seq_name}: {exc}")
        finally:
            _save_manifest(
                output_dir,
                dataset,
                base_dir,
                results_root,
                sources,
                sequences,
                completed_sequences,
                failed,
                selection_counts,
            )

    _save_eval_results(output_dir, sequence_results, failed)

    table_p5_rows = _build_segment_metric_rows(
        report_method_names,
        report_store,
        threshold=5.0,
    )
    table_p15_rows = _build_segment_metric_rows(
        report_method_names,
        report_store,
        threshold=15.0,
    )
    max_frame_idx, timeline_stats = _compute_timeline_stats(
        report_sequences,
        report_method_names,
    )
    report_path = _write_interactive_report(
        output_dir=output_dir,
        dataset=dataset,
        base_dir=base_dir,
        oracle_name=oracle_name,
        method_names=report_method_names,
        pair_method_names=tracker_names,
        table_p5_rows=table_p5_rows,
        table_p15_rows=table_p15_rows,
        report_sequences=report_sequences,
        timeline_stats=timeline_stats,
        max_frame_idx=max_frame_idx,
        default_significant_threshold=significant_better_threshold,
        cdn=cdn,
    )

    _print_metric_table("P@5", report_method_names, report_store, threshold=5.0)
    _print_metric_table("P@15", report_method_names, report_store, threshold=15.0)
    _print_significant_better_table(
        report_method_names,
        better_counts,
        better_totals,
        threshold=significant_better_threshold,
    )

    total_selected = sum(selection_counts.values())
    if total_selected > 0:
        print("\nOracle selection counts:")
        for name in tracker_names:
            count = selection_counts.get(name, 0)
            pct = 100.0 * count / total_selected
            print(f"  {name}: {count} frames ({pct:.1f}%)")

    print(
        f"\nCompleted: {len(completed_sequences)} ok, "
        f"{len(failed)} failed  (total {len(sequences)})"
    )
    if failed:
        print("Failed sequences:")
        for entry in failed:
            print(f"  {entry['sequence']}: {entry['error']}")
    print(f"Report saved to: {report_path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build frame-by-frame oracle tracker from saved result folders"
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to process")
    parser.add_argument(
        "--trackers",
        nargs="+",
        required=True,
        help=(
            "Tracker result names or config paths. Each resolves to "
            "<results-dir>/<tracker_name>/<dataset>/"
        ),
    )
    parser.add_argument(
        "--oracle-name",
        required=True,
        help="Output tracker name for oracle ensemble directory",
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
        help="Top-level directory containing source tracker results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(paths.OUTPUT_DIRECTORY),
        help="Top-level directory for oracle outputs",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequences to process (if not specified, process all)",
    )
    parser.add_argument(
        "--subset",
        help=(
            "Name of subset file from configs/subsets/<name>.txt. "
            "Intersected with --sequences when both are provided."
        ),
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        help="Direct path to subset file (takes precedence over --subset)",
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
        help=(
            "Directory containing per-sequence initfix files for "
            "planartrack-initfix / pti"
        ),
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Rebuild sequences even if oracle outputs already exist",
    )
    parser.add_argument(
        "--significant-better-threshold",
        type=float,
        default=10.0,
        help="Gap in pixels for pairwise 'significantly better' reporting",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Load Plotly from CDN instead of inlining it (requires internet).",
    )
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR

    args.results_dir = args.results_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()

    if not args.base_dir.exists():
        print(f"ERROR: Dataset directory not found: {args.base_dir}")
        return 1
    if not args.results_dir.exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        return 1

    try:
        sequences = get_sequences(
            args.dataset,
            args.base_dir,
            sequences=args.sequences,
            subset=args.subset,
            subset_file=args.subset_file,
            split=args.split,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        build_oracle_results(
            dataset=args.dataset,
            base_dir=args.base_dir,
            results_root=args.results_dir,
            output_root=args.output_dir,
            oracle_name=args.oracle_name,
            tracker_specs=args.trackers,
            sequences=sequences,
            initfix_dir=args.initfix_dir,
            skip_completed=not args.no_skip_completed,
            significant_better_threshold=args.significant_better_threshold,
            cdn=args.cdn,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
