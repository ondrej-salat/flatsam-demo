#!/usr/bin/env python3
"""Build PTi SAM3 mask metrics vs homography error HTML scatter report.

Joins per-sequence CSV files from ``pti_sam3_mask_metrics.py`` with tracker
alignment errors and plots one dot per evaluated frame.
"""

import argparse
import csv
import json
import math
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from einops import rearrange

from flatsam2.config import TrackerConfig, compute_output_dir, load_config_from_file
from flatsam2.datasets import get_sequences, pt_image_paths, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.report_utils import get_plotly_script_tag


DEFAULT_TRACKER_CONFIG = Path("configs/tracker/hough_symmetry_only.py")
DEFAULT_METRIC = "iou"
DEFAULT_ERROR_CAP_PX = 30.0
PIXEL_ERROR_METRICS = {"chamfer_mean_px", "chamfer_p95_px", "chamfer_max_px"}
CANDIDATE_H_KEYS = ("flow_H", "mask_H", "redetection_flow_H")
CANDIDATE_COLORS = {
    "flow_H": (255, 80, 40),
    "mask_H": (0, 170, 255),
    "redetection_flow_H": (70, 220, 70),
}

MASK_METRICS = [
    "iou",
    "dice",
    "boundary_hit_t5",
    "boundary_hit_t10",
    "boundary_hit_t15",
    "boundary_hit_t20",
    "boundary_hit_t25",
    "boundary_hit_t30",
    "boundary_f1_t5",
    "boundary_f1_t10",
    "boundary_f1_t15",
    "boundary_f1_t20",
    "boundary_f1_t25",
    "boundary_f1_t30",
    "chamfer_mean_px",
    "chamfer_p95_px",
    "chamfer_max_px",
    "area_gt_px",
    "area_pred_px",
]


@dataclass
class FramePoint:
    """Joined frame record for scatter plotting.

    Args:
        sequence: Sequence name.
        frame_idx: Frame index.
        mask_metrics: Mask metric values keyed by metric name.
        alignment_error_px: Raw tracker alignment error in pixels.
    """

    sequence: str
    frame_idx: int
    mask_metrics: dict[str, float]
    alignment_error_px: float


def _load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    """Load tracker predictions.

    Args:
        results_dir: Directory containing ``<seq_name>.txt`` files.
        seq_name: Sequence name.

    Returns:
        Predicted corners with shape (frames, 2, 4).

    Raises:
        ValueError: If prediction file is missing or malformed.
    """
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise ValueError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    if pred_flat.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in prediction file: {pred_file}")
    return rearrange(
        pred_flat,
        "frames (corners xy) -> frames xy corners",
        corners=4,
        xy=2,
    )


def _parse_float(value: str) -> float:
    """Parse CSV float, preserving empty values as NaN."""
    if value == "":
        return float("nan")
    return float(value)


def _load_mask_metric_rows(seq_csv_path: Path) -> dict[int, dict[str, float]]:
    """Load all finite metrics from one mask-metrics per-sequence CSV.

    Args:
        seq_csv_path: Path written by ``pti_sam3_mask_metrics.py``.

    Returns:
        Mapping from evaluated frame index to finite metric values.

    Raises:
        ValueError: If CSV is missing required columns.
    """
    if not seq_csv_path.exists():
        return {}

    with seq_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {seq_csv_path}")
        required = {"frame_idx", "evaluated", *MASK_METRICS}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV {seq_csv_path} missing columns: {missing}")

        rows: dict[int, dict[str, float]] = {}
        for row in reader:
            if bool(int(row["evaluated"])):
                metric_values = {}
                for metric_name in MASK_METRICS:
                    value = _parse_float(row[metric_name])
                    if math.isfinite(value):
                        metric_values[metric_name] = value
                if metric_values:
                    rows[int(row["frame_idx"])] = metric_values
    return rows


def _sequence_errors(
    base_dir: Path,
    initfix_dir: Path,
    results_dir: Path,
    seq_name: str,
) -> np.ndarray:
    """Compute per-frame tracker alignment errors for one sequence.

    Args:
        base_dir: PlanarTrack base directory.
        initfix_dir: Initfix GT directory.
        results_dir: Tracker results directory.
        seq_name: Sequence name.

    Returns:
        Error array of shape (frames,), NaN where not computable.
    """
    gt_corners, valid = pti_load_gt(base_dir, seq_name, initfix_dir)
    pred = _load_predictions(results_dir, seq_name)
    n_frames = min(len(gt_corners), len(valid), len(pred))
    gt_corners = gt_corners[:n_frames]
    valid = valid[:n_frames]
    pred = pred[:n_frames]

    errors = np.full(n_frames, np.nan, dtype=np.float64)
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    eval_idx = np.where(eval_mask)[0]
    gt_eval = gt_corners[eval_idx]
    pred_eval = pred[eval_idx]
    finite = np.all(np.isfinite(gt_eval), axis=(1, 2)) & np.all(
        np.isfinite(pred_eval), axis=(1, 2)
    )
    if finite.any():
        errors[eval_idx[finite]] = alignment_error(gt_eval[finite], pred_eval[finite])
    return errors


def collect_points(
    sequences: list[str],
    base_dir: Path,
    initfix_dir: Path,
    results_dir: Path,
    mask_metrics_dir: Path,
) -> tuple[list[FramePoint], list[dict[str, str]]]:
    """Collect joined scatter points.

    Args:
        sequences: PTi sequence names.
        base_dir: PlanarTrack base directory.
        initfix_dir: Initfix GT directory.
        results_dir: Tracker output directory.
        mask_metrics_dir: Directory containing ``per_sequence`` CSV directory.

    Returns:
        Tuple of points and failed sequence records.
    """
    points: list[FramePoint] = []
    failed: list[dict[str, str]] = []
    sequence_csv_dir = mask_metrics_dir / "per_sequence"

    for seq_name in sequences:
        try:
            metric_by_frame = _load_mask_metric_rows(sequence_csv_dir / f"{seq_name}.csv")
            if not metric_by_frame:
                continue

            errors = _sequence_errors(base_dir, initfix_dir, results_dir, seq_name)
            errors_by_frame = {
                int(frame_idx): float(err)
                for frame_idx, err in enumerate(errors)
                if math.isfinite(float(err))
            }
            for frame_idx, metric_value in metric_by_frame.items():
                if frame_idx not in errors_by_frame:
                    continue
                err = float(errors_by_frame[frame_idx])
                if not math.isfinite(err):
                    continue
                points.append(
                    FramePoint(
                        sequence=seq_name,
                        frame_idx=frame_idx,
                        mask_metrics=metric_value,
                        alignment_error_px=err,
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
    return points, failed


def _stats(values: list[float]) -> dict[str, float]:
    """Compute compact numeric summary."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _build_payload(points: list[FramePoint], default_metric: str, error_cap_px: float) -> dict:
    """Build JSON payload for report."""
    sequences = sorted({p.sequence for p in points})
    seq_to_idx = {seq_name: i for i, seq_name in enumerate(sequences)}
    errors = [p.alignment_error_px for p in points]
    capped_errors = [min(err, error_cap_px) for err in errors]
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(points)).tolist()
    metric_values_by_name = {
        metric_name: [p.mask_metrics.get(metric_name) for p in points]
        for metric_name in MASK_METRICS
    }

    return {
        "default_metric": default_metric,
        "metric_names": MASK_METRICS,
        "pixel_error_metrics": sorted(PIXEL_ERROR_METRICS),
        "error_cap_px": float(error_cap_px),
        "n": int(len(points)),
        "sequence_names": sequences,
        "seq_idx": [seq_to_idx[p.sequence] for p in points],
        "frame_idx": [p.frame_idx for p in points],
        "perm": perm,
        "mask_metrics": metric_values_by_name,
        "alignment_error_px": errors,
        "capped_alignment_error_px": capped_errors,
        "metric_stats": {
            metric_name: _stats([v for v in values if v is not None])
            for metric_name, values in metric_values_by_name.items()
        },
        "error_stats": _stats(errors),
    }


def write_report(
    output_path: Path,
    payload: dict,
    tracker_name: str,
    mask_metrics_dir: Path,
    results_dir: Path,
    cdn: bool,
) -> None:
    """Write interactive HTML scatter report.

    Args:
        output_path: Destination HTML path.
        payload: JSON-serializable data payload.
        tracker_name: Tracker/config label.
        mask_metrics_dir: Source mask metric directory.
        results_dir: Source tracker results directory.
        cdn: Use Plotly CDN instead of inline script.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, separators=(",", ":"))
    plotly_tag = get_plotly_script_tag(cdn)
    default_metric = payload["default_metric"]
    error_cap_px = payload["error_cap_px"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PTi mask metric vs homography error - {tracker_name}</title>
  {plotly_tag}
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 16px; font-family: system-ui, sans-serif;
      background: #f4f6fb; color: #172033; font-size: 13px; }}
    h1 {{ margin: 0 0 10px; font-size: 20px; }}
    .meta {{ background: #fff; border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;
      box-shadow: 0 1px 5px rgba(15, 23, 42, 0.08); line-height: 1.6; }}
    .controls {{ background: #fff; border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;
      box-shadow: 0 1px 5px rgba(15, 23, 42, 0.08); display: flex; gap: 18px;
      flex-wrap: wrap; align-items: end; }}
    .ctrl {{ display: flex; flex-direction: column; gap: 4px; }}
    .label {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .06em; color: #64748b; }}
    input, select {{ padding: 4px 7px; border: 1px solid #cbd5e1; border-radius: 5px; }}
    input[type=range] {{ width: 130px; padding: 0; }}
    button {{ padding: 4px 9px; border: 1px solid #cbd5e1; border-radius: 5px;
      background: #f8fafc; cursor: pointer; }}
    .stats {{ background: #fff; border-radius: 10px; padding: 8px 14px; margin-bottom: 10px;
      box-shadow: 0 1px 5px rgba(15, 23, 42, 0.08); }}
    .quad-table {{ border-collapse: collapse; margin-top: 6px; }}
    .quad-table td, .quad-table th {{ border: 1px solid #cbd5e1; padding: 3px 7px;
      text-align: right; }}
    .quad-table th {{ background: #f8fafc; color: #475569; }}
    #plot {{ height: 720px; background: #fff; border-radius: 10px;
      box-shadow: 0 1px 5px rgba(15, 23, 42, 0.08); }}
    code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PTi SAM3 mask metric vs capped alignment error</h1>
  <div class="meta">
    <b>Tracker:</b> <code>{tracker_name}</code><br>
    <b>Default mask metric:</b> <code>{default_metric}</code><br>
    <b>Frames:</b> <span id="n-frames"></span><br>
    <b>Mask metrics dir:</b> <code>{mask_metrics_dir}</code><br>
    <b>Tracker results dir:</b> <code>{results_dir}</code>
  </div>
  <div class="controls">
    <div class="ctrl">
      <div class="label">Mask metric</div>
      <select id="metric-select" onchange="redraw()"></select>
    </div>
    <div class="ctrl">
      <div class="label">Error cap px</div>
      <input id="error-cap" type="number" min="1" step="1" value="{error_cap_px:.0f}"
        onchange="redraw()">
    </div>
    <div class="ctrl">
      <div class="label">Plot mode</div>
      <select id="plot-mode" onchange="redraw()">
        <option value="scatter">scatter</option>
        <option value="heatmap">heatmap counts</option>
      </select>
    </div>
    <div class="ctrl">
      <div class="label">Heatmap bins</div>
      <input id="heatmap-bins" type="number" min="5" max="200" step="5" value="50"
        onchange="redraw()">
    </div>
    <div class="ctrl">
      <div class="label">Subset <span id="subset-label">100%</span></div>
      <input id="subset-slider" type="range" min="1" max="100" step="1" value="100"
        oninput="document.getElementById('subset-label').textContent=this.value+'%'; redraw()">
    </div>
    <div class="ctrl">
      <div class="label">P threshold px</div>
      <input id="error-threshold" type="number" min="1" step="1" value="15" onchange="redraw()">
    </div>
    <div class="ctrl">
      <div class="label">Metric threshold</div>
      <input id="x-threshold" type="number" step="0.01" value="0.5" onchange="redraw()">
    </div>
    <div class="ctrl">
      <div class="label">AE threshold px</div>
      <input id="y-threshold" type="number" min="0" step="1" value="15" onchange="redraw()">
    </div>
    <div class="ctrl">
      <div class="label">Bin table</div>
      <select id="bin-table-mode" onchange="redraw()">
        <option value="percent">percent</option>
        <option value="count">count</option>
      </select>
    </div>
    <div class="ctrl">
      <div class="label">Marker opacity</div>
      <input id="opacity" type="number" min="0.05" max="1" step="0.05" value="0.45"
        onchange="redraw()">
    </div>
    <div class="ctrl">
      <div class="label">Selection</div>
      <button type="button" onclick="clearSelection()">clear sequence</button>
    </div>
  </div>
  <div class="stats" id="stats-bar">Loading...</div>
  <div id="plot"></div>
  <script>
    const D = {payload_json};
    document.getElementById('n-frames').textContent = D.n.toLocaleString();
    let selectedSeq = null;
    const metricSelect = document.getElementById('metric-select');
    for (const metricName of D.metric_names) {{
      const opt = document.createElement('option');
      opt.value = metricName;
      opt.textContent = metricName;
      opt.selected = metricName === D.default_metric;
      metricSelect.appendChild(opt);
    }}

    function clearSelection() {{
      selectedSeq = null;
      redraw();
    }}

    function redraw() {{
      const metricName = metricSelect.value;
      const cap = parseFloat(document.getElementById('error-cap').value) || D.error_cap_px;
      const threshold = parseFloat(document.getElementById('error-threshold').value) || 15;
      const xThreshold = parseFloat(document.getElementById('x-threshold').value);
      const yThreshold = parseFloat(document.getElementById('y-threshold').value) || 15;
      const binTableMode = document.getElementById('bin-table-mode').value;
      const opacity = parseFloat(document.getElementById('opacity').value) || 0.45;
      const plotMode = document.getElementById('plot-mode').value;
      const heatmapBins = parseInt(document.getElementById('heatmap-bins').value) || 50;
      const pixelErrorMetric = D.pixel_error_metrics.includes(metricName);
      const subsetPct = parseInt(document.getElementById('subset-slider').value) || 100;
      const subsetN = Math.max(1, Math.floor(D.n * subsetPct / 100));
      const subset = D.perm.slice(0, subsetN);
      const x = [];
      const y = [];
      const text = [];
      const custom = [];
      const sx = [];
      const sy = [];
      const stext = [];
      const scustom = [];
      let nP = 0;
      let nShown = 0;
      let nSelected = 0;
      let qLeLe = 0, qGtLe = 0, qLeGt = 0, qGtGt = 0;
      for (const i of subset) {{
        const metricValue = D.mask_metrics[metricName][i];
        if (metricValue === null) continue;
        const err = D.alignment_error_px[i];
        const cappedErr = Math.min(err, cap);
        const plottedMetricValue = pixelErrorMetric ? Math.min(metricValue, cap) : metricValue;
        const seqIdx = D.seq_idx[i];
        const hoverText =
          'seq=' + D.sequence_names[seqIdx] +
          '<br>frame=' + D.frame_idx[i] +
          '<br>' + metricName + '=' + metricValue.toFixed(5) +
          '<br>alignment_error=' + err.toFixed(2) + ' px';
        x.push(plottedMetricValue);
        y.push(cappedErr);
        text.push(hoverText);
        custom.push(seqIdx);
        if (err <= threshold) nP++;
        if (Number.isFinite(xThreshold)) {{
          if (plottedMetricValue <= xThreshold && cappedErr <= yThreshold) qLeLe++;
          if (plottedMetricValue > xThreshold && cappedErr <= yThreshold) qGtLe++;
          if (plottedMetricValue <= xThreshold && cappedErr > yThreshold) qLeGt++;
          if (plottedMetricValue > xThreshold && cappedErr > yThreshold) qGtGt++;
        }}
        nShown++;
        if (seqIdx === selectedSeq) {{
          sx.push(plottedMetricValue);
          sy.push(cappedErr);
          stext.push(hoverText);
          scustom.push(seqIdx);
          nSelected++;
        }}
      }}
      const pRate = nShown > 0 ? (100 * nP / nShown).toFixed(2) : 'nan';
      const selectionText = selectedSeq === null ? 'none' : D.sequence_names[selectedSeq];
      const tableValue = (n) => binTableMode === 'count'
        ? n.toLocaleString()
        : (nShown > 0 ? (100 * n / nShown).toFixed(2) + '%' : 'nan');
      const quadHtml = Number.isFinite(xThreshold) ?
        '<table class="quad-table"><tr><th></th><th>metric &le; ' + xThreshold.toFixed(3) + '</th><th>metric &gt; ' + xThreshold.toFixed(3) + '</th></tr>' +
        '<tr><th>AE &le; ' + yThreshold.toFixed(1) + '</th><td>' + tableValue(qLeLe) + '</td><td>' + tableValue(qGtLe) + '</td></tr>' +
        '<tr><th>AE &gt; ' + yThreshold.toFixed(1) + '</th><td>' + tableValue(qLeGt) + '</td><td>' + tableValue(qGtGt) + '</td></tr></table>' : '';
      document.getElementById('stats-bar').innerHTML =
        '<b>' + nShown.toLocaleString() + '</b> frames shown' +
        ' | P@' + threshold.toFixed(0) + ' = <b>' + pRate + '%</b>' +
        ' | selected sequence: <b>' + selectionText + '</b>' +
        ' (' + nSelected.toLocaleString() + ' frames)' + quadHtml;

      const binSizeY = cap / 40;
      let xMin = Infinity;
      let xMax = -Infinity;
      for (const v of x) {{
        if (!Number.isFinite(v)) continue;
        xMin = Math.min(xMin, v);
        xMax = Math.max(xMax, v);
      }}
      if (xMin === Infinity) {{
        xMin = 0;
        xMax = 1;
      }}
      const xRange = xMax > xMin ? [xMin, xMax] : [xMin - 0.5, xMax + 0.5];
      const binSizeX = (xRange[1] - xRange[0]) / heatmapBins;
      const metricAxisTitle = pixelErrorMetric ? metricName + ' (capped)' : metricName;
      const traces = [];
      if (plotMode === 'heatmap') {{
        traces.push({{
          type: 'histogram2d', x, y, name: 'frame count',
          xbins: {{ start: xRange[0], end: xRange[1], size: binSizeX }},
          ybins: {{ start: 0, end: cap + binSizeY, size: binSizeY }},
          colorscale: 'Viridis', colorbar: {{ title: 'count' }},
          hovertemplate: metricAxisTitle + '=%{{x:.3f}}<br>error=%{{y:.1f}} px' +
            '<br>count=%{{z}}<extra></extra>'
        }});
      }} else {{
        traces.push({{
          type: 'scattergl', mode: 'markers', x, y, text,
          customdata: custom, name: 'frames',
          hovertemplate: '%{{text}}<extra></extra>',
          marker: {{
            size: 4, opacity,
            color: '#2563eb'
          }}
        }});
      }}
      if (plotMode === 'scatter' && selectedSeq !== null) {{
        traces.push({{
          type: 'scattergl', mode: 'markers', x: sx, y: sy, text: stext,
          customdata: scustom, name: 'selected sequence',
          hovertemplate: '%{{text}}<extra></extra>',
          marker: {{ size: 7, opacity: 0.95, color: '#f97316' }}
        }});
      }}
      traces.push({{
        x, type: 'histogram', xaxis: 'x', yaxis: 'y2', showlegend: false,
        opacity: 0.65, marker: {{ color: '#2563eb' }},
        hovertemplate: metricAxisTitle + '=%{{x:.3f}}<br>count=%{{y}}<extra></extra>'
      }});
      traces.push({{
        y, type: 'histogram', xaxis: 'x2', yaxis: 'y', showlegend: false,
        opacity: 0.65, ybins: {{ start: 0, end: cap + binSizeY, size: binSizeY }},
        marker: {{ color: '#2563eb' }},
        hovertemplate: 'error=%{{y:.1f}}<br>count=%{{x}}<extra></extra>'
      }});

      const MAIN_X = [0, 0.82], MAIN_Y = [0, 0.82];
      const MARG_X = [0.84, 1.0], MARG_Y = [0.84, 1.0];
      const layout = {{
        barmode: 'overlay',
        margin: {{ l: 64, r: 18, t: 42, b: 64 }},
        title: 'P@' + threshold.toFixed(0) + ' = ' + pRate + '% (' + nP.toLocaleString() +
          '/' + nShown.toLocaleString() + ')',
        xaxis: {{ title: metricAxisTitle, zeroline: false, domain: MAIN_X, range: xRange }},
        yaxis: {{ title: 'alignment error px (capped)', range: [0, cap], domain: MAIN_Y }},
        yaxis2: {{ domain: MARG_Y, showticklabels: false, fixedrange: true }},
        xaxis2: {{ domain: MARG_X, showticklabels: false, fixedrange: true }},
        legend: {{ orientation: 'h', yanchor: 'top', y: -0.14, xanchor: 'center', x: 0.41 }},
        shapes: [{{
          type: 'line', xref: 'paper', x0: 0, x1: 0.82, yref: 'y', y0: threshold, y1: threshold,
          line: {{ color: '#64748b', width: 1.5, dash: 'dot' }}
        }}]
      }};
      Plotly.react(
        'plot', traces, layout, {{ responsive: true, displayModeBar: true }}
      ).then(() => {{
        const plot = document.getElementById('plot');
        if (plot.removeAllListeners) plot.removeAllListeners('plotly_click');
        plot.on('plotly_click', (evt) => {{
          const p = evt.points && evt.points[0];
          if (!p || p.customdata === undefined || p.customdata === null) return;
          selectedSeq = p.customdata;
          redraw();
        }});
      }});
    }}
    redraw();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


@dataclass
class ViewerState:
    """Runtime state for interactive report server.

    Args:
        payload: Scatter/report payload.
        cache_token: Per-process browser cache-busting token.
        base_dir: PlanarTrack base directory.
        initfix_dir: Initfix GT directory.
        results_dir: Tracker output directory.
        mask_metrics_dir: Directory containing per-sequence metric CSV files.
        masks_dir: Precomputed SAM mask directory.
        hough_cache: Cached Hough debug arrays keyed by (sequence, frame).
    """

    payload: dict
    cache_token: str
    base_dir: Path
    initfix_dir: Path
    results_dir: Path
    mask_metrics_dir: Path
    masks_dir: Optional[Path]
    hough_gradient_step: int
    hough_topk_lines: int
    hough_corner_max_distance_px: float
    hough_corner_max_distance_image_fraction: Optional[float]
    hough_min_intersection_angle_deg: float
    hough_line_refinement_method: str
    sequence_cache: dict[str, dict]
    hough_cache: dict[tuple[str, int], dict]


def _load_debug(results_dir: Path, seq_name: str) -> Optional[dict[str, np.ndarray]]:
    """Load tracker debug NPZ if present."""
    debug_path = results_dir / f"{seq_name}_debug.npz"
    if not debug_path.exists():
        return None
    with np.load(debug_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _draw_poly(
    img: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dashed: bool = False,
) -> None:
    """Draw quadrilateral from corners with shape (2, 4)."""
    import cv2

    if not np.all(np.isfinite(corners)):
        return
    if dashed:
        for i in range(corners.shape[1]):
            p1 = (int(corners[0, i]), int(corners[1, i]))
            p2 = (int(corners[0, (i + 1) % corners.shape[1]]), int(corners[1, (i + 1) % corners.shape[1]]))
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = max(1, int(np.hypot(dx, dy)))
            for start in range(0, dist, 14):
                end = min(dist, start + 8)
                xs = p1[0] + dx * start // dist
                ys = p1[1] + dy * start // dist
                xe = p1[0] + dx * end // dist
                ye = p1[1] + dy * end // dist
                cv2.line(img, (xs, ys), (xe, ye), color, thickness, cv2.LINE_AA)
    else:
        pts = rearrange(corners, "xy c -> c 1 xy").astype(np.int32)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    for corner_i, (x, y) in enumerate(corners.T):
        radius = 7 if corner_i == 0 else 4
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def _blend_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend binary mask into BGR image."""
    import cv2

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored = img.copy()
    colored[mask > 0] = (80, 80, 255)
    return cv2.addWeighted(img, 0.65, colored, 0.35, 0)


def _draw_text_block(img: np.ndarray, lines: list[str], x: int = 8, y: int = 22) -> None:
    """Draw readable multi-line overlay text."""
    import cv2

    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        yy += 20


def _load_sam_mask(masks_dir: Optional[Path], seq_name: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load precomputed SAM mask for frame if available."""
    if masks_dir is None:
        return None
    import cv2

    mask_path = masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def _draw_line_from_abc(
    img: np.ndarray,
    line: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw homogeneous image line `ax + by + c = 0` clipped to image bounds."""
    import cv2

    a, b, c = [float(v) for v in line]
    h, w = img.shape[:2]
    pts: list[tuple[int, int]] = []
    if abs(b) > 1e-8:
        pts.extend(
            [
                (0, int(round(-c / b))),
                (w - 1, int(round(-(a * (w - 1) + c) / b))),
            ]
        )
    if abs(a) > 1e-8:
        pts.extend(
            [
                (int(round(-c / a)), 0),
                (int(round(-(b * (h - 1) + c) / a)), h - 1),
            ]
        )
    unique_pts = list(dict.fromkeys(pts))
    for i, p1 in enumerate(unique_pts):
        for p2 in unique_pts[i + 1 :]:
            ok, cp1, cp2 = cv2.clipLine((0, 0, w, h), p1, p2)
            if ok and cp1 != cp2:
                cv2.line(img, cp1, cp2, color, thickness, cv2.LINE_AA)
                return


def _draw_segment(
    img: np.ndarray,
    endpoints: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    """Draw finite segment endpoints with shape (2, 2)."""
    import cv2

    if endpoints.shape != (2, 2) or not np.all(np.isfinite(endpoints)):
        return
    p1 = (int(round(float(endpoints[0, 0]))), int(round(float(endpoints[1, 0]))))
    p2 = (int(round(float(endpoints[0, 1]))), int(round(float(endpoints[1, 1]))))
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _point_line_distances(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Compute orthogonal point-to-line distances for line `ax + by + c = 0`."""
    denom = float(np.linalg.norm(line[:2]))
    if denom <= 0.0 or pts.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.abs(line[:2] @ pts + line[2]) / denom


def _compute_hough_debug(
    mask: np.ndarray,
    gradient_step: int,
    topk_lines: int,
    corner_max_distance_px: float,
    corner_max_distance_image_fraction: Optional[float],
    min_intersection_angle_deg: float,
    line_refinement_method: str,
) -> dict:
    """Compute Hough debug arrays from stored mask."""
    import cv2
    import scipy.ndimage
    from skimage.feature import peak_local_max

    from flatsam2.hough_homography import (
        _approx_touching_edges,
        _line_to_segment,
        _precompute_contour_samples,
        _remap,
        _segment_length,
        contour_hough,
        keep_only_points_close_to_contours,
        lines_to_points,
        p2e,
        refine_lines,
    )

    contours_cv, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [rearrange(c, "n 1 xy -> xy n", xy=2) for c in contours_cv if c.shape[0] > 20]
    if not contours:
        return {
            "contours": [],
            "hough_lines": np.zeros((3, 0)),
            "refined_lines": np.zeros((3, 0)),
            "line_pts": [],
            "support_counts": np.zeros(0, dtype=np.int64),
            "raw_fit_rms": np.zeros(0),
            "raw_fit_p90": np.zeros(0),
            "fit_rms": np.zeros(0),
            "fit_p90": np.zeros(0),
            "angle_delta_deg": np.zeros(0),
            "support_segments": [],
            "raw_support_segments": [],
            "refined_distances": [],
            "supported_lines": np.zeros((3, 0)),
            "has_support": np.zeros(0, dtype=bool),
            "short_mask": np.zeros(0, dtype=bool),
            "long_lines": np.zeros((3, 0)),
            "intersections": np.zeros((2, 0)),
            "valid_intersections": np.zeros((2, 0)),
            "accumulator": np.zeros((1, 1)),
            "blurred_accumulator": np.zeros((1, 1)),
            "peaks": np.zeros((0, 2), dtype=np.int64),
            "peak_values": np.zeros(0),
        }
    on_boundaries = [
        np.logical_or.reduce(
            (
                _approx_touching_edges(c + np.array([[0], [1]]), mask.shape, margin=1),
                _approx_touching_edges(c + np.array([[0], [-1]]), mask.shape, margin=1),
                _approx_touching_edges(c + np.array([[1], [0]]), mask.shape, margin=1),
                _approx_touching_edges(c + np.array([[-1], [0]]), mask.shape, margin=1),
            )
        )
        for c in contours
    ]
    samples = _precompute_contour_samples(
        contours,
        on_boundaries,
        gradient_step=gradient_step,
    )

    all_cnt = np.concatenate(contours, axis=1)
    contour_center = np.mean(all_cnt, axis=1, keepdims=True)
    point_dists = [np.sqrt(np.sum((cnt - contour_center) ** 2, axis=0)) for cnt in contours]
    max_dist = max(np.amax(d) for d in point_dists)
    n_angles, n_dists = 359, 100
    accumulator = np.zeros((n_dists, n_angles))
    rad_range = np.radians(10)
    delta_angles = np.linspace(-rad_range, rad_range, 11)
    vote_weights = _remap(np.abs(delta_angles), 0, rad_range, 1, 0.5)
    for sample in samples:
        if sample.points.shape[1] == 0:
            continue
        a = sample.points - contour_center
        diff = sample.tangents
        angle = np.arctan2(diff[1], diff[0])
        dist = np.cos(angle) * a[1] - np.sin(angle) * a[0]
        norm_angle = angle + np.pi / 2
        norm_to_direction = np.full_like(dist, -np.pi / 2)
        neg = dist < 0
        norm_angle[neg] -= np.pi
        norm_to_direction[neg] = np.pi / 2
        res_angle = (norm_angle[:, np.newaxis] + delta_angles[np.newaxis, :] + 2 * np.pi) % (2 * np.pi)
        dir_angle = res_angle + norm_to_direction[:, np.newaxis]
        d = np.abs(np.cos(dir_angle) * a[1, :, np.newaxis] - np.sin(dir_angle) * a[0, :, np.newaxis])
        row = np.clip(np.round(_remap(d, 0.0, max_dist, 0, n_dists - 1)).astype(np.intp), 0, n_dists - 1)
        col = np.clip(np.round(_remap(res_angle, 0, 2 * np.pi, 0, n_angles - 1)).astype(np.intp), 0, n_angles - 1)
        np.add.at(accumulator, (row.ravel(), col.ravel()), np.broadcast_to(vote_weights, d.shape).ravel())
    cyclic = np.tile(accumulator, (1, 3))
    cyclic = scipy.ndimage.gaussian_filter(cyclic, sigma=4)
    peaks = peak_local_max(cyclic, min_distance=10, threshold_rel=None, exclude_border=False)
    peaks = peaks[(peaks[:, 1] >= n_angles) & (peaks[:, 1] < 2 * n_angles), :]
    peaks[:, 1] -= n_angles
    blurred = cyclic[:, n_angles : 2 * n_angles]
    peak_values = blurred[peaks[:, 0], peaks[:, 1]] if len(peaks) > 0 else np.zeros(0)

    hough_lines = contour_hough(
        mask,
        contours,
        on_boundaries,
        gradient_step=gradient_step,
        topk=topk_lines,
        _samples=samples,
    )
    refined_lines, line_pts = refine_lines(
        hough_lines,
        contours,
        on_boundaries,
        gradient_step=gradient_step,
        use_orthogonal_fit=line_refinement_method == "orthogonal_tls",
        _samples=samples,
    )
    has_support = np.array([pts.shape[1] > 0 for pts in line_pts], dtype=bool)
    supported_lines = refined_lines[:, has_support]
    supported_pts = [p for p, s in zip(line_pts, has_support) if s]
    support_counts = np.array([pts.shape[1] for pts in line_pts], dtype=np.int64)
    raw_fit_rms = np.full(refined_lines.shape[1], np.nan, dtype=np.float64)
    raw_fit_p90 = np.full(refined_lines.shape[1], np.nan, dtype=np.float64)
    fit_rms = np.full(refined_lines.shape[1], np.nan, dtype=np.float64)
    fit_p90 = np.full(refined_lines.shape[1], np.nan, dtype=np.float64)
    angle_delta_deg = np.full(refined_lines.shape[1], np.nan, dtype=np.float64)
    support_segments = []
    raw_support_segments = []
    refined_distances = []
    for line_i, pts in enumerate(line_pts):
        if pts.shape[1] == 0:
            support_segments.append(np.zeros((2, 2)))
            raw_support_segments.append(np.zeros((2, 2)))
            refined_distances.append(np.zeros(0, dtype=np.float64))
            continue
        raw_dist = _point_line_distances(hough_lines[:, line_i], pts)
        fit_dist = _point_line_distances(refined_lines[:, line_i], pts)
        raw_fit_rms[line_i] = float(np.sqrt(np.mean(raw_dist ** 2)))
        raw_fit_p90[line_i] = float(np.percentile(raw_dist, 90.0))
        fit_rms[line_i] = float(np.sqrt(np.mean(fit_dist ** 2)))
        fit_p90[line_i] = float(np.percentile(fit_dist, 90.0))
        denom = float(
            np.linalg.norm(hough_lines[:2, line_i]) * np.linalg.norm(refined_lines[:2, line_i])
        )
        if denom > 0.0:
            cos_delta = abs(float(hough_lines[:2, line_i] @ refined_lines[:2, line_i]) / denom)
            angle_delta_deg[line_i] = float(np.degrees(np.arccos(np.clip(cos_delta, 0, 1))))
        support_segments.append(_line_to_segment(refined_lines[:, line_i], pts))
        raw_support_segments.append(_line_to_segment(hough_lines[:, line_i], pts))
        refined_distances.append(fit_dist)
    short = np.zeros(supported_lines.shape[1], dtype=bool)
    if supported_lines.shape[1] > 0:
        endpoints = [_line_to_segment(supported_lines[:, i], supported_pts[i]) for i in range(supported_lines.shape[1])]
        short = np.array([_segment_length(ep) < 5 for ep in endpoints], dtype=bool)
    long_lines = supported_lines[:, ~short]
    intersections = np.zeros((2, 0))
    valid_intersections = np.zeros((2, 0))
    if long_lines.shape[1] >= 2:
        intersections_h, parents = lines_to_points(long_lines)
        intersections = p2e(intersections_h)
        max_distance_px = float(corner_max_distance_px)
        if corner_max_distance_image_fraction is not None:
            max_distance_px = max(
                max_distance_px,
                float(corner_max_distance_image_fraction) * float(max(mask.shape)),
            )
        valid_intersections, _ = keep_only_points_close_to_contours(
            intersections,
            contours,
            max_distance_px=max_distance_px,
            parents=parents,
            lines=long_lines,
            min_angle_deg=min_intersection_angle_deg,
        )
    return {
        "contours": contours,
        "hough_lines": hough_lines,
        "refined_lines": refined_lines,
        "line_pts": line_pts,
        "has_support": has_support,
        "support_counts": support_counts,
        "raw_fit_rms": raw_fit_rms,
        "raw_fit_p90": raw_fit_p90,
        "fit_rms": fit_rms,
        "fit_p90": fit_p90,
        "angle_delta_deg": angle_delta_deg,
        "support_segments": support_segments,
        "raw_support_segments": raw_support_segments,
        "refined_distances": refined_distances,
        "supported_lines": supported_lines,
        "short_mask": short,
        "long_lines": long_lines,
        "intersections": intersections,
        "valid_intersections": valid_intersections,
        "accumulator": accumulator,
        "blurred_accumulator": blurred,
        "peaks": peaks,
        "peak_values": peak_values,
    }


def _get_hough_debug(
    state: ViewerState,
    seq_name: str,
    frame_idx: int,
    mask: np.ndarray,
) -> dict:
    """Get cached Hough debug arrays for one frame."""
    cache_key = (seq_name, frame_idx)
    if cache_key not in state.hough_cache:
        state.hough_cache[cache_key] = _compute_hough_debug(
            mask,
            gradient_step=state.hough_gradient_step,
            topk_lines=state.hough_topk_lines,
            corner_max_distance_px=state.hough_corner_max_distance_px,
            corner_max_distance_image_fraction=state.hough_corner_max_distance_image_fraction,
            min_intersection_angle_deg=state.hough_min_intersection_angle_deg,
            line_refinement_method=state.hough_line_refinement_method,
        )
    return state.hough_cache[cache_key]


def _draw_hough_debug(
    img: np.ndarray,
    hough_debug: dict,
    stage: str,
    show_support_points: bool,
    show_support_segments: bool,
    show_raw_support_segments: bool,
    show_bad_support_points: bool,
    show_intersections: bool,
) -> dict[str, int]:
    """Draw cached Hough debug overlay."""
    import cv2

    hough_lines = hough_debug["hough_lines"]
    refined_lines = hough_debug["refined_lines"]
    supported_lines = hough_debug["supported_lines"]
    long_lines = hough_debug["long_lines"]
    line_pts = hough_debug["line_pts"]
    support_segments = hough_debug["support_segments"]
    raw_support_segments = hough_debug["raw_support_segments"]
    refined_distances = hough_debug["refined_distances"]
    support_colors = [
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 255, 0),
        (0, 128, 255),
        (255, 0, 128),
    ]
    if stage in {"raw", "all"}:
        for i in range(hough_lines.shape[1]):
            _draw_line_from_abc(img, hough_lines[:, i], (255, 180, 0), 1)
    if stage in {"refined", "all"}:
        for i in range(refined_lines.shape[1]):
            _draw_line_from_abc(img, refined_lines[:, i], (220, 80, 255), 1)
    if stage in {"supported", "all"}:
        for i in range(supported_lines.shape[1]):
            _draw_line_from_abc(img, supported_lines[:, i], (0, 220, 255), 2)
    if stage in {"long", "all"}:
        for i in range(long_lines.shape[1]):
            _draw_line_from_abc(img, long_lines[:, i], (80, 255, 80), 3)
    for i in range(refined_lines.shape[1]):
        color = support_colors[i % len(support_colors)]
        if show_support_points and i < len(line_pts):
            pts = line_pts[i].T.astype(np.int32)
            for x, y in pts:
                cv2.circle(img, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
        if show_bad_support_points and i < len(line_pts) and i < len(refined_distances):
            pts = line_pts[i].T.astype(np.int32)
            bad = refined_distances[i] > 15.0
            for x, y in pts[bad]:
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if show_raw_support_segments and i < len(raw_support_segments):
            _draw_segment(img, raw_support_segments[i], (255, 180, 0), 2)
        if show_support_segments and i < len(support_segments):
            _draw_segment(img, support_segments[i], color, 4)
    if show_intersections:
        for x, y in hough_debug["intersections"].T:
            cv2.drawMarker(img, (int(x), int(y)), (0, 120, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        for x, y in hough_debug["valid_intersections"].T:
            cv2.drawMarker(img, (int(x), int(y)), (80, 255, 80), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    return {
        "hough_lines": int(hough_lines.shape[1]),
        "refined_lines": int(refined_lines.shape[1]),
        "supported_lines": int(supported_lines.shape[1]),
        "long_lines": int(long_lines.shape[1]),
        "intersections": int(hough_debug["intersections"].shape[1]),
        "valid_intersections": int(hough_debug["valid_intersections"].shape[1]),
    }


def _format_hough_fit_stats(hough_debug: dict) -> list[str]:
    """Format compact per-line Hough refinement diagnostics."""
    lines = []
    counts = hough_debug["support_counts"]
    raw_rms = hough_debug["raw_fit_rms"]
    raw_p90 = hough_debug["raw_fit_p90"]
    rms = hough_debug["fit_rms"]
    p90 = hough_debug["fit_p90"]
    angle_delta = hough_debug["angle_delta_deg"]
    for line_i in range(len(counts)):
        raw_text = (
            "nan"
            if not math.isfinite(float(raw_rms[line_i]))
            else f"{float(raw_rms[line_i]):.1f}/{float(raw_p90[line_i]):.1f}px"
        )
        fit_text = (
            "nan"
            if not math.isfinite(float(rms[line_i]))
            else f"{float(rms[line_i]):.1f}/{float(p90[line_i]):.1f}px"
        )
        angle_text = (
            "nan"
            if not math.isfinite(float(angle_delta[line_i]))
            else f"{float(angle_delta[line_i]):.1f}deg"
        )
        lines.append(
            f"line {line_i}: pts={int(counts[line_i])} raw/ref rms-p90={raw_text}/{fit_text} "
            f"dtheta={angle_text}"
        )
    return lines


def _ensure_view_sequence(state: ViewerState, seq_name: str) -> dict:
    """Load and cache sequence data for interactive viewer."""
    if seq_name in state.sequence_cache:
        return state.sequence_cache[seq_name]

    gt, valid = pti_load_gt(state.base_dir, seq_name, state.initfix_dir)
    pred = _load_predictions(state.results_dir, seq_name)
    image_paths = pt_image_paths(state.base_dir, seq_name)
    debug = _load_debug(state.results_dir, seq_name)
    metric_by_frame = _load_mask_metric_rows(state.mask_metrics_dir / "per_sequence" / f"{seq_name}.csv")
    n_frames = min(len(gt), len(valid), len(pred), len(image_paths))
    gt = gt[:n_frames]
    valid = valid[:n_frames]
    pred = pred[:n_frames]
    image_paths = image_paths[:n_frames]
    errors = np.full(n_frames, np.nan, dtype=np.float64)
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    eval_idx = np.where(eval_mask)[0]
    finite = np.all(np.isfinite(gt[eval_idx]), axis=(1, 2)) & np.all(
        np.isfinite(pred[eval_idx]), axis=(1, 2)
    )
    if finite.any():
        errors[eval_idx[finite]] = alignment_error(gt[eval_idx[finite]], pred[eval_idx[finite]])
    seq = {
        "name": seq_name,
        "gt": gt,
        "valid": valid,
        "pred": pred,
        "image_paths": image_paths,
        "debug": debug,
        "metrics": metric_by_frame,
        "errors": errors,
    }
    state.sequence_cache[seq_name] = seq
    return seq


def _sequence_view_json(state: ViewerState, seq_name: str) -> dict:
    """Build JSON payload for one sequence."""
    seq = _ensure_view_sequence(state, seq_name)
    debug = seq["debug"]
    debug_frames = set()
    selected_by_frame = {}
    corrected_frames: set[int] = set()
    debug_by_frame = {}
    if debug is not None and "frame_idx" in debug:
        debug_frames = {int(v) for v in debug["frame_idx"]}
        if "selected" in debug:
            selected_by_frame = {
                int(frame_idx): str(debug["selected"][row])
                for row, frame_idx in enumerate(debug["frame_idx"])
            }
        if "corrected" in debug:
            corrected_frames = {
                int(frame_idx)
                for row, frame_idx in enumerate(debug["frame_idx"])
                if bool(debug["corrected"][row])
            }
        for row, frame_idx in enumerate(debug["frame_idx"]):
            frame_debug = {}
            for key in CANDIDATE_H_KEYS:
                if key in debug:
                    H = debug[key][row]
                    frame_debug[key] = H.tolist() if np.all(np.isfinite(H)) else None
                rmse_key = f"{key}_rmse"
                if rmse_key in debug:
                    value = float(debug[rmse_key][row])
                    frame_debug[rmse_key] = value if math.isfinite(value) else None
                sigma_key = f"{key}_sigma"
                if sigma_key in debug:
                    value = float(debug[sigma_key][row])
                    frame_debug[sigma_key] = value if math.isfinite(value) else None
            debug_by_frame[int(frame_idx)] = frame_debug
    frame_stats = []
    for frame_idx in range(len(seq["gt"])):
        err = float(seq["errors"][frame_idx]) if np.isfinite(seq["errors"][frame_idx]) else None
        frame_stats.append(
            {
                "frame_idx": frame_idx,
                "valid_gt": bool(seq["valid"][frame_idx]),
                "alignment_error_px": err,
                "metrics": seq["metrics"].get(frame_idx, {}),
                "has_mask": _load_sam_mask(state.masks_dir, seq_name, frame_idx) is not None,
                "has_debug": frame_idx in debug_frames,
                "selected": selected_by_frame.get(frame_idx, ""),
                "corrected": frame_idx in corrected_frames,
                "debug": debug_by_frame.get(frame_idx, {}),
            }
        )
    return {
        "name": seq_name,
        "num_frames": len(seq["gt"]),
        "num_valid_gt": int(seq["valid"].sum()),
        "has_debug": debug is not None,
        "debug_keys": sorted(debug.keys()) if debug is not None else [],
        "corrected_frames": sorted(corrected_frames),
        "frame_stats": frame_stats,
    }


def _hough_text_lines(
    state: "ViewerState",
    seq_name: str,
    frame_idx: int,
    draw_hough_debug: bool,
) -> list[str]:
    """Return hough stats text lines for frame. Uses cached debug data; no drawing."""
    if not draw_hough_debug:
        return []
    seq = _ensure_view_sequence(state, seq_name)
    mask = _load_sam_mask(state.masks_dir, seq_name, frame_idx)
    if mask is None:
        return []
    try:
        hd = _get_hough_debug(state, seq_name, frame_idx, mask)
        lines = [
            f"hough raw={int(hd['hough_lines'].shape[1])} "
            f"refined={int(hd['refined_lines'].shape[1])} "
            f"support={int(hd['supported_lines'].shape[1])} "
            f"long={int(hd['long_lines'].shape[1])}",
            f"intersections={int(hd['intersections'].shape[1])} "
            f"valid={int(hd['valid_intersections'].shape[1])}",
        ]
        lines.extend(_format_hough_fit_stats(hd)[:4])
        return lines
    except Exception:
        return ["hough stats error"]


def _render_view_frame(
    state: ViewerState,
    seq_name: str,
    frame_idx: int,
    show_mask: bool,
    show_gt: bool,
    show_pred: bool,
    show_candidates: bool,
    hough_stage: str,
    show_hough_supports: bool,
    show_hough_support_segments: bool,
    show_hough_raw_segments: bool,
    show_hough_bad_supports: bool,
    show_hough_intersections: bool,
) -> bytes:
    """Render one frame with requested overlays and return PNG bytes."""
    import cv2

    seq = _ensure_view_sequence(state, seq_name)
    frame_idx = int(np.clip(frame_idx, 0, len(seq["gt"]) - 1))
    img = cv2.imread(str(seq["image_paths"][frame_idx]))
    if img is None:
        img = np.full((480, 640, 3), 50, dtype=np.uint8)

    mask = _load_sam_mask(state.masks_dir, seq_name, frame_idx)
    if show_mask and mask is not None:
        img = _blend_mask(img, mask)

    hough_info = {
        "hough_lines": 0,
        "refined_lines": 0,
        "supported_lines": 0,
        "long_lines": 0,
        "intersections": 0,
        "valid_intersections": 0,
    }
    draw_hough_debug = (
        hough_stage != "off"
        or show_hough_supports
        or show_hough_support_segments
        or show_hough_raw_segments
        or show_hough_bad_supports
        or show_hough_intersections
    )
    hough_fit_lines = []

    if show_gt and seq["valid"][frame_idx]:
        _draw_poly(img, seq["gt"][frame_idx], (255, 255, 255), 2, dashed=True)
    if show_candidates and seq["debug"] is not None and "frame_idx" in seq["debug"]:
        row_by_frame = {int(frame): row for row, frame in enumerate(seq["debug"]["frame_idx"])}
        row = row_by_frame.get(frame_idx)
        if row is not None:
            for key in CANDIDATE_H_KEYS:
                if key not in seq["debug"]:
                    continue
                H = seq["debug"][key][row]
                if np.all(np.isfinite(H)):
                    _draw_poly(img, warp_corners(H, seq["gt"][0]), CANDIDATE_COLORS[key], 1)
    if show_pred:
        _draw_poly(img, seq["pred"][frame_idx], (0, 255, 0), 2)

    if draw_hough_debug and mask is not None:
        try:
            hough_debug = _get_hough_debug(state, seq_name, frame_idx, mask)
            hough_info = _draw_hough_debug(
                img,
                hough_debug,
                stage=hough_stage,
                show_support_points=show_hough_supports,
                show_support_segments=show_hough_support_segments,
                show_raw_support_segments=show_hough_raw_segments,
                show_bad_support_points=show_hough_bad_supports,
                show_intersections=show_hough_intersections,
            )
            hough_fit_lines = _format_hough_fit_stats(hough_debug)
        except Exception:
            hough_info = {
                "hough_lines": -1,
                "refined_lines": -1,
                "supported_lines": -1,
                "long_lines": -1,
                "intersections": -1,
                "valid_intersections": -1,
            }

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Failed to encode frame {seq_name}/{frame_idx}")
    return bytes(buf)


def _render_hough_accumulator(
    state: ViewerState,
    seq_name: str,
    frame_idx: int,
    blurred: bool = True,
    show_peaks: bool = True,
    show_constraints: bool = True,
) -> bytes:
    """Render Hough accumulator heatmap for one frame."""
    import cv2

    mask = _load_sam_mask(state.masks_dir, seq_name, frame_idx)
    if mask is None:
        img = np.full((240, 480, 3), 30, dtype=np.uint8)
        _draw_text_block(img, ["no mask"])
    else:
        hough_debug = _get_hough_debug(state, seq_name, frame_idx, mask)
        acc_key = "blurred_accumulator" if blurred else "accumulator"
        acc = hough_debug[acc_key].astype(np.float32)
        if float(np.max(acc)) > 0.0:
            vis = np.clip(acc / float(np.max(acc)) * 255.0, 0, 255).astype(np.uint8)
        else:
            vis = np.zeros(acc.shape, dtype=np.uint8)
        img = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
        img = cv2.resize(img, (718, 240), interpolation=cv2.INTER_NEAREST)
        if show_peaks or show_constraints:
            peaks = hough_debug["peaks"]
            values = hough_debug["peak_values"]
            img_h, img_w = img.shape[:2]
            acc_h, acc_w = acc.shape

            def acc_to_img(row: int, col: int) -> tuple[int, int]:
                return (
                    int(round(col * (img_w - 1) / max(acc_w - 1, 1))),
                    int(round(row * (img_h - 1) / max(acc_h - 1, 1))),
                )

            topk_set: set[int] = set()
            if len(values) > 0:
                topk = min(state.hough_topk_lines, len(values))
                top_ids = np.argpartition(values, -topk)[-topk:]
                topk_set = set(top_ids.tolist())

            if show_constraints:
                # min_distance=10 (accumulator pixels) used by peak_local_max
                rx = int(round(10 * (img_w - 1) / max(acc_w - 1, 1)))
                ry = int(round(10 * (img_h - 1) / max(acc_h - 1, 1)))
                for peak_i, (row, col) in enumerate(peaks):
                    x, y = acc_to_img(row, col)
                    # exclusion ellipse — shows the min_distance suppression zone
                    cv2.ellipse(img, (x, y), (rx, ry), 0, 0, 360, (100, 100, 100), 1, cv2.LINE_AA)
                    # non-selected peaks in cyan
                    if peak_i not in topk_set:
                        cv2.drawMarker(img, (x, y), (0, 200, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

            if show_peaks:
                for peak_i in topk_set:
                    row, col = peaks[peak_i]
                    x, y = acc_to_img(row, col)
                    cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Failed to encode accumulator {seq_name}/{frame_idx}")
    return bytes(buf)


def _interactive_html(
    payload: dict,
    plotly_tag: str,
    tracker_name: str,
    cache_token: str,
) -> str:
    """Build interactive FastAPI viewer HTML."""
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PTi SAM3 detail viewer - {tracker_name}</title>
  {plotly_tag}
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: monospace; background: #1e1e1e; color: #ddd; height: 100vh; overflow: hidden; }}
    #app {{ display: flex; flex-direction: row; height: 100vh; overflow: hidden; }}
    #left {{ width: 46%; min-width: 150px; display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden; }}
    #right {{ flex: 1; min-width: 150px; display: flex; flex-direction: column; overflow: hidden; }}
    .divider-h {{ width: 5px; cursor: ew-resize; background: #2a2a2a; flex-shrink: 0; }}
    .divider-h:hover, .divider-h:active {{ background: #555; }}
    .divider-v {{ height: 5px; cursor: ns-resize; background: #2a2a2a; flex-shrink: 0; }}
    .divider-v:hover, .divider-v:active {{ background: #555; }}
    #toolbar {{ padding: 8px; display: flex; gap: 8px; flex-wrap: wrap; background: #252525; border-bottom: 1px solid #444; flex-shrink: 0; }}
    input, select, button {{ background: #1f1f1f; color: #ddd; border: 1px solid #555; font: inherit; padding: 3px 7px; }}
    #plot {{ flex: 1; min-height: 0; background: #fff; }}
    #header {{ padding: 8px 10px; background: #252525; border-bottom: 1px solid #444; color: #aaa; flex-shrink: 0; }}
    #frame-area {{ flex: 1; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }}
    #frame-img-wrap {{ flex: 1; min-height: 0; background: #111; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
    #frame-img {{ max-width: 100%; max-height: 100%; object-fit: contain; display: none; }}
    #accum-wrap {{ height: 200px; min-height: 60px; background: #0d0d0d; display: none; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }}
    #accum-img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    #empty {{ color: #666; }}
    #controls {{ height: 220px; min-height: 60px; padding: 8px 10px; background: #1a1a1a; display: flex; flex-direction: column; gap: 6px; overflow: auto; flex-shrink: 0; }}
    #nav {{ display: flex; gap: 8px; align-items: center; flex-shrink: 0; }}
    #frame-slider {{ flex: 1; accent-color: #4fc3f7; }}
    #overlays {{ display: flex; gap: 14px; flex-wrap: wrap; flex-shrink: 0; }}
    #info {{ display: grid; grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr); gap: 10px; }}
    #stats, #debug {{ white-space: pre-wrap; color: #bbb; background: #101010; border: 1px solid #333; padding: 8px; min-height: 80px; overflow: auto; }}
    #quad-table {{ border-collapse: collapse; color: #ddd; }}
    #quad-table td, #quad-table th {{ border: 1px solid #444; padding: 3px 7px; text-align: right; }}
    #quad-table th {{ background: #252525; color: #aaa; }}
    .sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; }}
    @media (max-width: 900px) {{ #app {{ flex-direction: column; }} #left {{ width: 100%; height: 45%; }} }}
  </style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="toolbar">
      <label>metric <select id="metric-select" onchange="redraw()"></select></label>
      <label>mode <select id="plot-mode" onchange="redraw()"><option value="scatter">scatter</option><option value="heatmap">heatmap</option></select></label>
      <label>cap <input id="error-cap" type="number" min="1" step="1" value="{payload['error_cap_px']:.0f}" onchange="redraw()"></label>
      <label>bins <input id="bins" type="number" min="5" max="200" step="5" value="50" onchange="redraw()"></label>
      <label>metric thr <input id="x-threshold" type="number" step="0.01" value="0.5" onchange="redraw()"></label>
      <label>AE thr <input id="y-threshold" type="number" step="1" value="15" onchange="redraw()"></label>
      <label>table <select id="bin-table-mode" onchange="redraw()"><option value="percent">percent</option><option value="count">count</option></select></label>
      <span id="plot-stats"></span>
    </div>
    <div style="padding:6px 8px;background:#1a1a1a;border-bottom:1px solid #333;flex-shrink:0"><table id="quad-table"></table></div>
    <div id="plot"></div>
  </div>
  <div id="lr-divider" class="divider-h" title="drag to resize columns"></div>
  <div id="right">
    <div id="header">Click scatter point. Tracker: <b>{tracker_name}</b> <span id="sel"></span></div>
    <div id="frame-area">
      <div id="frame-img-wrap"><span id="empty">No frame selected</span><img id="frame-img" alt="frame"></div>
      <div id="accum-divider" class="divider-v" style="display:none" title="drag to resize accumulator"></div>
      <div id="accum-wrap"><img id="accum-img" alt="hough accumulator"></div>
    </div>
    <div id="fc-divider" class="divider-v" title="drag to resize controls"></div>
    <div id="controls">
      <div id="nav"><button onclick="nav(-1)">&#9664;</button><input id="frame-slider" type="range" min="0" max="0" value="0" oninput="goToFrame(+this.value)"><button onclick="nav(1)">&#9654;</button><span id="frame-num">-</span><button onclick="nextCorrection()" title="Jump to next mask correction">corr &#8594;</button></div>
      <canvas id="corr-timeline" height="14" title="Red ticks = frames where the SAM3 mask was corrected. Click to seek." style="width:100%;height:14px;display:block;cursor:crosshair;margin-top:2px"></canvas>
      <div id="overlays">
        <label><input id="chk-mask" type="checkbox" checked onchange="refreshFrame()"> mask</label>
        <label><input id="chk-gt" type="checkbox" checked onchange="refreshFrame()"> GT</label>
        <label><input id="chk-pred" type="checkbox" checked onchange="refreshFrame()"> pred H</label>
        <label><input id="chk-cand" type="checkbox" checked onchange="refreshFrame()"> candidates</label>
        <label>hough <select id="hough-stage" onchange="refreshFrame()"><option value="off">off</option><option value="raw">raw peaks</option><option value="refined">refined all</option><option value="supported">support-filtered</option><option value="long">length-filtered</option><option value="all">all stages</option></select></label>
        <label><input id="chk-hough-support" type="checkbox" onchange="refreshFrame()"> hough support pts</label>
        <label><input id="chk-hough-segments" type="checkbox" onchange="refreshFrame()"> support segments</label>
        <label><input id="chk-hough-raw-segments" type="checkbox" onchange="refreshFrame()"> raw support segments</label>
        <label><input id="chk-hough-bad-support" type="checkbox" onchange="refreshFrame()"> bad support pts</label>
        <label><input id="chk-hough-intersections" type="checkbox" onchange="refreshFrame()"> intersections</label>
        <label><input id="chk-accumulator" type="checkbox" onchange="refreshFrame()"> accumulator</label>
        <label><input id="chk-accum-blur" type="checkbox" checked onchange="refreshFrame()"> accum blur</label>
        <label><input id="chk-accum-peaks" type="checkbox" checked onchange="refreshFrame()"> accum peaks</label>
        <label><input id="chk-accum-constraints" type="checkbox" checked onchange="refreshFrame()"> accum constraints</label>
      </div>
      <div id="info"><pre id="stats">-</pre><pre id="debug">-</pre></div>
      <div>Legend: GT white dashed, pred green, raw Hough orange, refined magenta, support-filtered cyan, length-filtered green, support segments use per-line colors, invalid/valid intersections orange/green. Red ticks on the frame bar = SAM3 mask corrected this frame.</div>
    </div>
  </div>
</div>
<script>
const D = {payload_json};
const CACHE_TOKEN = {json.dumps(cache_token)};
let seqData = null;
let currentSeq = null;
let currentFrame = 0;
let selectedSeqIdx = null;
let houghLines = [];
const metricSelect = document.getElementById('metric-select');
for (const m of D.metric_names) {{
  const opt = document.createElement('option');
  opt.value = m; opt.textContent = m; opt.selected = m === D.default_metric;
  metricSelect.appendChild(opt);
}}
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.key === 'ArrowLeft') {{ nav(-1); e.preventDefault(); }}
  if (e.key === 'ArrowRight') {{ nav(1); e.preventDefault(); }}
}});

function overlay(id) {{ return document.getElementById(id).checked ? 1 : 0; }}
function currentHoughStage() {{ return document.getElementById('hough-stage').value; }}

function updateUrlState() {{
  const params = new URLSearchParams();
  params.set('metric', metricSelect.value);
  params.set('plot_mode', document.getElementById('plot-mode').value);
  params.set('cap', document.getElementById('error-cap').value);
  params.set('bins', document.getElementById('bins').value);
  params.set('x_thr', document.getElementById('x-threshold').value);
  params.set('y_thr', document.getElementById('y-threshold').value);
  params.set('bin_table', document.getElementById('bin-table-mode').value);
  params.set('mask', overlay('chk-mask'));
  params.set('gt', overlay('chk-gt'));
  params.set('pred', overlay('chk-pred'));
  params.set('cand', overlay('chk-cand'));
  params.set('hough_stage', document.getElementById('hough-stage').value);
  params.set('hough_sup', overlay('chk-hough-support'));
  params.set('hough_seg', overlay('chk-hough-segments'));
  params.set('hough_raw_seg', overlay('chk-hough-raw-segments'));
  params.set('hough_bad', overlay('chk-hough-bad-support'));
  params.set('hough_int', overlay('chk-hough-intersections'));
  params.set('accum', overlay('chk-accumulator'));
  params.set('accum_blur', overlay('chk-accum-blur'));
  params.set('accum_peaks', overlay('chk-accum-peaks'));
  params.set('accum_con', overlay('chk-accum-constraints'));
  if (currentSeq) {{
    params.set('seq', currentSeq);
    params.set('frame', String(currentFrame));
  }}
  window.history.replaceState(null, '', window.location.pathname + '?' + params.toString());
}}

function restoreControlsFromUrl() {{
  const p = new URLSearchParams(window.location.search);
  function setVal(id, key) {{ const v = p.get(key); if (v !== null) document.getElementById(id).value = v; }}
  function setChk(id, key) {{ const v = p.get(key); if (v !== null) document.getElementById(id).checked = v === '1'; }}
  setVal('metric-select', 'metric');
  setVal('plot-mode', 'plot_mode');
  setVal('error-cap', 'cap');
  setVal('bins', 'bins');
  setVal('x-threshold', 'x_thr');
  setVal('y-threshold', 'y_thr');
  setVal('bin-table-mode', 'bin_table');
  setChk('chk-mask', 'mask');
  setChk('chk-gt', 'gt');
  setChk('chk-pred', 'pred');
  setChk('chk-cand', 'cand');
  setVal('hough-stage', 'hough_stage');
  setChk('chk-hough-support', 'hough_sup');
  setChk('chk-hough-segments', 'hough_seg');
  setChk('chk-hough-raw-segments', 'hough_raw_seg');
  setChk('chk-hough-bad-support', 'hough_bad');
  setChk('chk-hough-intersections', 'hough_int');
  setChk('chk-accumulator', 'accum');
  setChk('chk-accum-blur', 'accum_blur');
  setChk('chk-accum-peaks', 'accum_peaks');
  setChk('chk-accum-constraints', 'accum_con');
}}

function redraw() {{
  const metricName = metricSelect.value;
  const cap = parseFloat(document.getElementById('error-cap').value) || D.error_cap_px;
  const bins = parseInt(document.getElementById('bins').value) || 50;
  const xThreshold = parseFloat(document.getElementById('x-threshold').value);
  const yThreshold = parseFloat(document.getElementById('y-threshold').value) || 15;
  const binTableMode = document.getElementById('bin-table-mode').value;
  const plotMode = document.getElementById('plot-mode').value;
  const pixelMetric = D.pixel_error_metrics.includes(metricName);
  const x = [], y = [], text = [], custom = [];
  const sx = [], sy = [], stext = [], scustom = [], sframe = [];
  const cx = [], cy = [], ctext = [], ccustom = [];
  let qLeLe = 0, qGtLe = 0, qLeGt = 0, qGtGt = 0;
  let xMin = Infinity, xMax = -Infinity;
  for (let i = 0; i < D.n; i++) {{
    const rawX = D.mask_metrics[metricName][i];
    if (rawX === null) continue;
    const px = pixelMetric ? Math.min(rawX, cap) : rawX;
    const py = Math.min(D.alignment_error_px[i], cap);
    const seqIdx = D.seq_idx[i];
    x.push(px); y.push(py); custom.push([seqIdx, D.frame_idx[i]]);
    const hoverText = 'seq=' + D.sequence_names[seqIdx] + '<br>frame=' + D.frame_idx[i] + '<br>' + metricName + '=' + rawX.toFixed(5) + '<br>alignment=' + D.alignment_error_px[i].toFixed(2) + 'px';
    text.push(hoverText);
    if (Number.isFinite(xThreshold)) {{
      if (px <= xThreshold && py <= yThreshold) qLeLe++;
      if (px > xThreshold && py <= yThreshold) qGtLe++;
      if (px <= xThreshold && py > yThreshold) qLeGt++;
      if (px > xThreshold && py > yThreshold) qGtGt++;
    }}
    if (seqIdx === selectedSeqIdx) {{
      sx.push(px); sy.push(py); stext.push(hoverText); scustom.push([seqIdx, D.frame_idx[i]]);
      sframe.push(D.frame_idx[i]);
      if (D.frame_idx[i] === currentFrame) {{
        cx.push(px); cy.push(py); ctext.push(hoverText); ccustom.push([seqIdx, D.frame_idx[i]]);
      }}
    }}
    if (Number.isFinite(px)) {{ xMin = Math.min(xMin, px); xMax = Math.max(xMax, px); }}
  }}
  if (xMin === Infinity) {{ xMin = 0; xMax = 1; }}
  const xRange = xMax > xMin ? [xMin, xMax] : [xMin - 0.5, xMax + 0.5];
  const titleX = pixelMetric ? metricName + ' (capped)' : metricName;
  const tableValue = (n) => binTableMode === 'count'
    ? n.toLocaleString()
    : (x.length > 0 ? (100 * n / x.length).toFixed(2) + '%' : 'nan');
  document.getElementById('quad-table').innerHTML = Number.isFinite(xThreshold) ?
    '<tr><th></th><th>metric &le; ' + xThreshold.toFixed(3) + '</th><th>metric &gt; ' + xThreshold.toFixed(3) + '</th></tr>' +
    '<tr><th>AE &le; ' + yThreshold.toFixed(1) + '</th><td>' + tableValue(qLeLe) + '</td><td>' + tableValue(qGtLe) + '</td></tr>' +
    '<tr><th>AE &gt; ' + yThreshold.toFixed(1) + '</th><td>' + tableValue(qLeGt) + '</td><td>' + tableValue(qGtGt) + '</td></tr>' : '';
  const traces = [];
  if (plotMode === 'heatmap') {{
    traces.push({{type: 'histogram2d', x, y, xbins: {{start: xRange[0], end: xRange[1], size: (xRange[1] - xRange[0]) / bins}}, ybins: {{start: 0, end: cap, size: cap / bins}}, colorscale: 'Viridis', colorbar: {{title: 'count'}}}});
  }} else {{
    traces.push({{type: 'scattergl', mode: 'markers', x, y, text, customdata: custom, hovertemplate: '%{{text}}<extra></extra>', marker: {{size: 5, opacity: 0.42, color: '#b91c1c'}}}});
  }}
  if (selectedSeqIdx !== null) {{
    traces.push({{type: 'scattergl', mode: 'markers', x: sx, y: sy, text: stext, customdata: scustom, name: 'selected sequence', hovertemplate: '%{{text}}<extra></extra>', marker: {{size: 8, opacity: 0.95, color: sframe, colorscale: 'Viridis', cmin: 0, cmax: Math.max(1, seqData ? seqData.num_frames - 1 : Math.max(...sframe)), colorbar: {{title: 'frame'}}}}}});
    if (cx.length > 0) {{
      traces.push({{type: 'scattergl', mode: 'markers', x: cx, y: cy, text: ctext, customdata: ccustom, name: 'current frame', hovertemplate: '%{{text}}<extra></extra>', marker: {{size: 16, opacity: 1, color: '#ef4444', symbol: 'circle-open-dot', line: {{color: '#ffffff', width: 2}}}}}});
    }}
  }}
  const layout = {{margin: {{l: 62, r: 12, t: 30, b: 54}}, xaxis: {{title: titleX, range: xRange}}, yaxis: {{title: 'alignment error px (capped)', range: [0, cap]}}, title: x.length.toLocaleString() + ' frames'}};
  document.getElementById('plot-stats').textContent = x.length.toLocaleString() + ' points';
  updateUrlState();
  Plotly.react('plot', traces, layout, {{responsive: true}}).then(() => {{
    const plot = document.getElementById('plot');
    if (plot.removeAllListeners) plot.removeAllListeners('plotly_click');
    plot.on('plotly_click', (evt) => {{
      const p = evt.points && evt.points[0];
      if (!p || !p.customdata) return;
      const seqName = D.sequence_names[p.customdata[0]];
      selectFrame(seqName, p.customdata[1]);
    }});
  }});
}}

async function selectFrame(seqName, frameIdx) {{
  selectedSeqIdx = D.sequence_names.indexOf(seqName);
  if (seqName !== currentSeq) {{
    const res = await fetch('/api/sequence/' + encodeURIComponent(seqName));
    if (!res.ok) return;
    seqData = await res.json();
    currentSeq = seqName;
    document.getElementById('frame-slider').max = Math.max(seqData.num_frames - 1, 0);
  }}
  goToFrame(frameIdx, true);
}}

function nav(delta) {{ if (seqData) goToFrame(currentFrame + delta); }}
function refreshFrame() {{ if (seqData) goToFrame(currentFrame, false); }}

function drawCorrTimeline() {{
  const cv = document.getElementById('corr-timeline');
  if (!cv || !seqData) return;
  const W = cv.clientWidth || 600, H = 14;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#181818'; ctx.fillRect(0, 0, W, H);
  const T = seqData.num_frames;
  if (!T) return;
  ctx.fillStyle = '#ff5577';
  for (const f of (seqData.corrected_frames || [])) {{
    if (f < 0 || f >= T) continue;
    const x = (f + 0.5) / T * W;
    ctx.fillRect(x - 1, 0, 2, H);
  }}
  const cx = T > 1 ? currentFrame / (T - 1) * W : 0;
  ctx.strokeStyle = '#4fc3f7'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
}}

function nextCorrection() {{
  if (!seqData) return;
  const all = seqData.corrected_frames || [];
  if (!all.length) return;
  const ahead = all.filter(f => f > currentFrame);
  goToFrame(ahead.length ? ahead[0] : all[0]);
}}

document.getElementById('corr-timeline').addEventListener('click', (e) => {{
  if (!seqData || !seqData.num_frames) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  goToFrame(Math.round(frac * (seqData.num_frames - 1)));
}});
window.addEventListener('resize', drawCorrTimeline);

async function goToFrame(frameIdx, updatePlot = true) {{
  if (!seqData) return;
  currentFrame = Math.max(0, Math.min(frameIdx, seqData.num_frames - 1));
  document.getElementById('frame-slider').value = currentFrame;
  document.getElementById('frame-num').textContent = currentFrame + ' / ' + (seqData.num_frames - 1);
  document.getElementById('sel').textContent = ' | ' + currentSeq + ' frame ' + currentFrame;
  const houghParams = '&hough_stage=' + encodeURIComponent(currentHoughStage()) +
    '&hough_support=' + overlay('chk-hough-support') +
    '&hough_segments=' + overlay('chk-hough-segments') +
    '&hough_raw_segments=' + overlay('chk-hough-raw-segments') +
    '&hough_bad_support=' + overlay('chk-hough-bad-support') +
    '&hough_intersections=' + overlay('chk-hough-intersections');
  const url = '/api/frame/' + encodeURIComponent(currentSeq) + '/' + currentFrame +
    '?mask=' + overlay('chk-mask') + '&gt=' + overlay('chk-gt') + '&pred=' + overlay('chk-pred') +
    '&candidates=' + overlay('chk-cand') + houghParams +
    '&v=' + encodeURIComponent(CACHE_TOKEN);
  const img = document.getElementById('frame-img');
  img.src = url; img.style.display = 'block'; document.getElementById('empty').style.display = 'none';
  const accum = document.getElementById('accum-img');
  const accumWrapEl2 = document.getElementById('accum-wrap');
  const accumDividerEl = document.getElementById('accum-divider');
  if (overlay('chk-accumulator')) {{
    accum.src = '/api/hough_accumulator/' + encodeURIComponent(currentSeq) + '/' + currentFrame +
      '?blurred=' + overlay('chk-accum-blur') + '&show_peaks=' + overlay('chk-accum-peaks') + '&show_constraints=' + overlay('chk-accum-constraints') + '&v=' + encodeURIComponent(CACHE_TOKEN);
    accumWrapEl2.style.display = 'flex';
    accumDividerEl.style.display = 'block';
  }} else {{
    accumWrapEl2.style.display = 'none';
    accumDividerEl.style.display = 'none';
  }}
  houghLines = [];
  try {{
    const tr = await fetch('/api/frame_text/' + encodeURIComponent(currentSeq) + '/' + currentFrame + '?' + houghParams.slice(1));
    if (tr.ok) {{ const td = await tr.json(); houghLines = td.hough_lines || []; }}
  }} catch(e) {{}}
  updateInfo();
  drawCorrTimeline();
  updateUrlState();
  if (updatePlot) redraw();
}}

function updateInfo() {{
  const s = seqData.frame_stats[currentFrame];
  const metricLines = Object.entries(s.metrics || {{}}).map(([k, v]) => k + ': ' + Number(v).toFixed(5));
  const houghText = houghLines.length > 0 ? '\\n\\nhough debug:\\n' + houghLines.join('\\n') : '';
  const nCorr = (seqData.corrected_frames || []).length;
  document.getElementById('stats').textContent =
    'sequence: ' + currentSeq + '\\nframe: ' + currentFrame + '\\nvalid_gt: ' + s.valid_gt +
    '\\nmask: ' + s.has_mask + '\\nmask corrected: ' + (s.corrected ? 'YES' : 'no') +
    '\\nalignment: ' + (s.alignment_error_px === null ? 'n/a' : s.alignment_error_px.toFixed(2) + ' px') +
    '\\n\\n' + metricLines.join('\\n') + houghText;
  document.getElementById('debug').textContent =
    'sequence debug: ' + seqData.has_debug + '\\nframe debug: ' + s.has_debug + '\\nselected: ' + (s.selected || '-') +
    '\\ncorrected: ' + (s.corrected ? 'YES' : 'no') + '\\nmask corrections in seq: ' + nCorr +
    '\\nkeys:\\n' + (seqData.debug_keys || []).join('\\n') +
    '\\n\\nframe debug:\\n' + JSON.stringify(s.debug || {{}}, null, 2);
}}

// --- resizable pane dividers ---
function makeDraggable(divider, cursor, getStart, applyDelta) {{
  divider.addEventListener('mousedown', (e) => {{
    e.preventDefault();
    const sx = e.clientX, sy = e.clientY;
    const start = getStart();
    document.body.style.cursor = cursor;
    document.body.style.userSelect = 'none';
    const onMove = (e) => applyDelta(e.clientX - sx, e.clientY - sy, start);
    const onUp = () => {{
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }};
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }});
}}
const leftEl = document.getElementById('left');
makeDraggable(document.getElementById('lr-divider'), 'ew-resize',
  () => leftEl.getBoundingClientRect().width,
  (dx, dy, w) => {{ leftEl.style.width = Math.max(150, w + dx) + 'px'; }});
const controlsEl = document.getElementById('controls');
makeDraggable(document.getElementById('fc-divider'), 'ns-resize',
  () => controlsEl.getBoundingClientRect().height,
  (dx, dy, h) => {{ controlsEl.style.height = Math.max(60, h - dy) + 'px'; }});
const accumWrapDrag = document.getElementById('accum-wrap');
makeDraggable(document.getElementById('accum-divider'), 'ns-resize',
  () => accumWrapDrag.getBoundingClientRect().height,
  (dx, dy, h) => {{ accumWrapDrag.style.height = Math.max(60, h - dy) + 'px'; }});
// --- end resizable dividers ---
const initialParams = new URLSearchParams(window.location.search);
restoreControlsFromUrl();
redraw();
const initialSeq = initialParams.get('seq');
const initialFrame = parseInt(initialParams.get('frame') || '0');
if (initialSeq && D.sequence_names.includes(initialSeq)) {{
  selectFrame(initialSeq, Number.isFinite(initialFrame) ? initialFrame : 0);
}}
</script>
</body>
</html>
"""


def create_interactive_app(
    payload: dict,
    tracker_name: str,
    base_dir: Path,
    initfix_dir: Path,
    results_dir: Path,
    mask_metrics_dir: Path,
    masks_dir: Optional[Path],
    hough_gradient_step: int,
    hough_topk_lines: int,
    hough_corner_max_distance_px: float,
    hough_corner_max_distance_image_fraction: Optional[float],
    hough_min_intersection_angle_deg: float,
    hough_line_refinement_method: str,
    cdn: bool,
):
    """Create FastAPI app for interactive report exploration."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    cache_token = str(time.time_ns())
    state = ViewerState(
        payload=payload,
        cache_token=cache_token,
        base_dir=base_dir,
        initfix_dir=initfix_dir,
        results_dir=results_dir,
        mask_metrics_dir=mask_metrics_dir,
        masks_dir=masks_dir,
        hough_gradient_step=hough_gradient_step,
        hough_topk_lines=hough_topk_lines,
        hough_corner_max_distance_px=hough_corner_max_distance_px,
        hough_corner_max_distance_image_fraction=hough_corner_max_distance_image_fraction,
        hough_min_intersection_angle_deg=hough_min_intersection_angle_deg,
        hough_line_refinement_method=hough_line_refinement_method,
        sequence_cache={},
        hough_cache={},
    )
    app = FastAPI(title="PTi SAM3 Mask Homography Report")
    plotly_tag = get_plotly_script_tag(cdn)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse(_interactive_html(payload, plotly_tag, tracker_name, cache_token))

    @app.get("/api/info")
    async def api_info() -> JSONResponse:
        return JSONResponse(
            {
                "tracker": tracker_name,
                "sequences": payload["sequence_names"],
                "results_dir": str(results_dir),
                "mask_metrics_dir": str(mask_metrics_dir),
                "masks_dir": str(masks_dir) if masks_dir is not None else None,
                "hough_gradient_step": hough_gradient_step,
                "hough_topk_lines": hough_topk_lines,
                "hough_corner_max_distance_px": hough_corner_max_distance_px,
                "hough_corner_max_distance_image_fraction": hough_corner_max_distance_image_fraction,
                "hough_min_intersection_angle_deg": hough_min_intersection_angle_deg,
                "hough_line_refinement_method": hough_line_refinement_method,
                "cache_token": cache_token,
            }
        )

    @app.get("/api/sequence/{seq_name}")
    async def api_sequence(seq_name: str) -> JSONResponse:
        if seq_name not in payload["sequence_names"]:
            raise HTTPException(404, f"Unknown sequence: {seq_name}")
        return JSONResponse(_sequence_view_json(state, seq_name))

    @app.get("/api/frame/{seq_name}/{frame_idx}")
    async def api_frame(
        seq_name: str,
        frame_idx: int,
        mask: int = 1,
        gt: int = 1,
        pred: int = 1,
        candidates: int = 1,
        hough_stage: str = "off",
        hough_support: int = 0,
        hough_segments: int = 0,
        hough_raw_segments: int = 0,
        hough_bad_support: int = 0,
        hough_intersections: int = 0,
    ) -> StreamingResponse:
        if seq_name not in payload["sequence_names"]:
            raise HTTPException(404, f"Unknown sequence: {seq_name}")
        if hough_stage not in {"off", "raw", "refined", "supported", "long", "all"}:
            raise HTTPException(400, f"Unknown Hough stage: {hough_stage}")
        png = _render_view_frame(
            state,
            seq_name,
            frame_idx,
            show_mask=bool(mask),
            show_gt=bool(gt),
            show_pred=bool(pred),
            show_candidates=bool(candidates),
            hough_stage=hough_stage,
            show_hough_supports=bool(hough_support),
            show_hough_support_segments=bool(hough_segments),
            show_hough_raw_segments=bool(hough_raw_segments),
            show_hough_bad_supports=bool(hough_bad_support),
            show_hough_intersections=bool(hough_intersections),
        )
        return StreamingResponse(iter([png]), media_type="image/png")

    @app.get("/api/frame_text/{seq_name}/{frame_idx}")
    async def api_frame_text(
        seq_name: str,
        frame_idx: int,
        hough_stage: str = "off",
        hough_support: int = 0,
        hough_segments: int = 0,
        hough_raw_segments: int = 0,
        hough_bad_support: int = 0,
        hough_intersections: int = 0,
    ) -> JSONResponse:
        if seq_name not in payload["sequence_names"]:
            raise HTTPException(404, f"Unknown sequence: {seq_name}")
        draw_hough_debug = (
            hough_stage != "off"
            or bool(hough_support)
            or bool(hough_segments)
            or bool(hough_raw_segments)
            or bool(hough_bad_support)
            or bool(hough_intersections)
        )
        lines = _hough_text_lines(state, seq_name, frame_idx, draw_hough_debug)
        return JSONResponse({"hough_lines": lines})

    @app.get("/api/hough_accumulator/{seq_name}/{frame_idx}")
    async def api_hough_accumulator(
        seq_name: str, frame_idx: int, blurred: int = 1, show_peaks: int = 1, show_constraints: int = 1
    ) -> StreamingResponse:
        if seq_name not in payload["sequence_names"]:
            raise HTTPException(404, f"Unknown sequence: {seq_name}")
        png = _render_hough_accumulator(
            state, seq_name, frame_idx,
            blurred=bool(blurred), show_peaks=bool(show_peaks), show_constraints=bool(show_constraints),
        )
        return StreamingResponse(iter([png]), media_type="image/png")

    return app


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create PTi mask metric vs homography alignment-error HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="PTi split.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=paths.PT_BASE,
        help="PlanarTrack base dir.",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=paths.PT_INITFIX_DIR,
        help="Initfix GT directory.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=paths.OUTPUT_DIRECTORY,
        help="Base tracker output directory.",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=DEFAULT_TRACKER_CONFIG,
        help="Tracker config used to resolve results dir.",
    )
    parser.add_argument(
        "--tracker-results-dir",
        type=Path,
        help=(
            "Direct tracker results dir containing <seq>.txt files. "
            "Overrides results-root/config."
        ),
    )
    parser.add_argument(
        "--mask-metrics-dir",
        type=Path,
        default=paths.OUTPUT_DIRECTORY / "experiments" / "pti_sam3_mask_metrics",
        help="Output dir from pti_sam3_mask_metrics.py.",
    )
    parser.add_argument(
        "--precomputed-masks-dir",
        type=Path,
        default=paths.SAM3_PRECOMPUTED_MASKS_PTI,
        help="Directory with SAM3 masks: <dir>/<seq>/{frame+1:05d}.png.",
    )
    parser.add_argument(
        "--metric",
        choices=MASK_METRICS,
        default=DEFAULT_METRIC,
        help="Mask metric column to plot on x-axis.",
    )
    parser.add_argument(
        "--error-cap-px",
        type=float,
        default=DEFAULT_ERROR_CAP_PX,
        help="Default y-axis cap for alignment error.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQ",
        help="Optional explicit sequences.",
    )
    parser.add_argument(
        "--subset",
        metavar="NAME",
        help="Subset name from configs/subsets/<name>.txt.",
    )
    parser.add_argument(
        "--subset-file",
        type=Path,
        metavar="PATH",
        help="Direct subset file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.OUTPUT_DIRECTORY
        / "experiments"
        / "pti_sam3_mask_homography_report"
        / "mask_metric_vs_alignment_error.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Load Plotly from CDN instead of inlining.",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Only write offline files; do not start FastAPI interactive viewer.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="FastAPI host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8125,
        help="FastAPI port.",
    )
    return parser


def main() -> None:
    """Build report."""
    args = build_parser().parse_args()
    tracker_results_dir = args.tracker_results_dir or compute_output_dir(
        args.results_root,
        "pti",
        args.tracker_config,
    )
    if args.tracker_results_dir is None:
        tracker_name = args.tracker_config.stem
    else:
        tracker_name = tracker_results_dir.name
    hough_gradient_step = 4
    hough_topk_lines = 4
    hough_corner_max_distance_px = 80.0
    hough_corner_max_distance_image_fraction: Optional[float] = None
    hough_min_intersection_angle_deg = 0.0
    hough_line_refinement_method = "legacy_algebraic"
    if args.tracker_config is not None and args.tracker_config.exists():
        tracker_config = load_config_from_file(args.tracker_config)
        if isinstance(tracker_config, TrackerConfig):
            hough_gradient_step = int(tracker_config.hough.gradient_step)
            hough_topk_lines = int(tracker_config.hough.topk_lines)
            hough_corner_max_distance_px = float(tracker_config.hough.corner_max_distance_px)
            hough_corner_max_distance_image_fraction = (
                None
                if tracker_config.hough.corner_max_distance_image_fraction is None
                else float(tracker_config.hough.corner_max_distance_image_fraction)
            )
            hough_min_intersection_angle_deg = float(tracker_config.hough.min_intersection_angle_deg)
            hough_line_refinement_method = tracker_config.hough.line_refinement_method
    sequences = get_sequences(
        "pti",
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split,
    )

    points, failed = collect_points(
        sequences=sequences,
        base_dir=args.base_dir,
        initfix_dir=args.initfix_dir,
        results_dir=tracker_results_dir,
        mask_metrics_dir=args.mask_metrics_dir,
    )
    if not points:
        raise ValueError(
            "No joined frames found. Check mask metrics dir, tracker results dir, split/subset."
        )

    payload = _build_payload(points, args.metric, args.error_cap_px)
    payload["failed_sequences"] = failed
    payload["num_sequences_requested"] = int(len(sequences))
    payload["num_sequences_failed"] = int(len(failed))
    payload["tracker_results_dir"] = str(tracker_results_dir)
    payload["mask_metrics_dir"] = str(args.mask_metrics_dir)

    write_report(
        output_path=args.output,
        payload=payload,
        tracker_name=tracker_name,
        mask_metrics_dir=args.mask_metrics_dir,
        results_dir=tracker_results_dir,
        cdn=args.cdn,
    )

    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== PTi mask metric vs homography error report ===")
    print(f"Tracker results : {tracker_results_dir}")
    print(f"Mask metrics    : {args.mask_metrics_dir}")
    print(f"Metric          : {args.metric}")
    print(f"Frames plotted  : {len(points)}")
    print(f"Sequences failed: {len(failed)}")
    print(f"HTML report     : {args.output}")
    print(f"Payload JSON    : {json_path}")

    # Prefer the masks this run actually saved (output_dir/masks): they reflect the
    # SAM3 stream used during tracking, including corrections. Fall back to the
    # shared precomputed set only when the run did not save its own masks.
    per_config_masks = tracker_results_dir / "masks"
    if per_config_masks.is_dir():
        masks_dir = per_config_masks
        print(f"Masks           : {masks_dir} (per-config; reflects corrections)")
    else:
        masks_dir = args.precomputed_masks_dir
        print(f"Masks           : {masks_dir} (shared precomputed fallback)")

    if not args.no_serve:
        import uvicorn

        app = create_interactive_app(
            payload=payload,
            tracker_name=tracker_name,
            base_dir=args.base_dir,
            initfix_dir=args.initfix_dir,
            results_dir=tracker_results_dir,
            mask_metrics_dir=args.mask_metrics_dir,
            masks_dir=masks_dir,
            hough_gradient_step=hough_gradient_step,
            hough_topk_lines=hough_topk_lines,
            hough_corner_max_distance_px=hough_corner_max_distance_px,
            hough_corner_max_distance_image_fraction=hough_corner_max_distance_image_fraction,
            hough_min_intersection_angle_deg=hough_min_intersection_angle_deg,
            hough_line_refinement_method=hough_line_refinement_method,
            cdn=args.cdn,
        )
        print(f"Serving viewer  : http://{args.host}:{args.port}")
        print(f"SSH tunnel      : ssh -L 8080:{socket.gethostname()}:{args.port} rci")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
