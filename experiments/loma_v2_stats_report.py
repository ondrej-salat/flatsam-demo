#!/usr/bin/env python3
"""Generate aggregate histogram report for loma_v2 keypoint statistics.

Reads ``*_stats.json`` files from one loma_v2 output directory and writes a
self-contained HTML page with one histogram per sequence-level metric.  Each
sequence contributes one aggregated value per metric.

Usage examples:
    python experiments/loma_v2_stats_report.py \
        /mnt/personal/serycjon/flatsam2/loma_v2/loma_r_allobj/POT-210

    python experiments/loma_v2_stats_report.py \
        /mnt/personal/serycjon/flatsam2/loma_v2/loma_r_allobj/PTI \
        --output /tmp/keypoint_stats.html
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.datasets import DATASETS, pot_image_paths, pt_image_paths
from flatsam2.path_config import paths
from flatsam2.report_utils import get_plotly_script_tag


_METRICS = [
    {
        "key": "init_detected_object",
        "label": "Init detected on object",
        "color": "#2E7D32",
    },
    {
        "key": "init_detected_background",
        "label": "Init detected on background",
        "color": "#66BB6A",
    },
    {
        "key": "detected_current_object",
        "label": "Current detected on object",
        "color": "#1565C0",
    },
    {
        "key": "detected_current_background",
        "label": "Current detected on background",
        "color": "#42A5F5",
    },
    {
        "key": "matched_total",
        "label": "Matched total",
        "color": "#6A1B9A",
    },
    {
        "key": "matched_object",
        "label": "Matched on object",
        "color": "#8E24AA",
    },
    {
        "key": "matched_background",
        "label": "Matched on background",
        "color": "#BA68C8",
    },
    {
        "key": "inliers_total",
        "label": "RANSAC inliers total",
        "color": "#EF6C00",
    },
    {
        "key": "inliers_object",
        "label": "RANSAC inliers on object",
        "color": "#FB8C00",
    },
    {
        "key": "inliers_background",
        "label": "RANSAC inliers on background",
        "color": "#FFB74D",
    },
    {
        "key": "inlier_ratio",
        "label": "RANSAC inlier ratio",
        "color": "#C62828",
    },
]


def _sequence_image_paths(dataset: str, base_dir: Path, seq_name: str) -> list[Path]:
    """Return image paths for one sequence."""
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_image_paths(base_dir, seq_name)
    if loader in {"pt", "pti"}:
        return pt_image_paths(base_dir, seq_name)
    raise ValueError(f"Unsupported dataset loader for previews: {loader}")


def _build_sequence_preview_images(
    dataset: str,
    base_dir: Path,
    sequences: list[dict],
    max_side: int = 220,
    jpeg_quality: int = 70,
) -> dict[str, str]:
    """Build base64 JPEG first-frame previews keyed by sequence name."""
    previews: dict[str, str] = {}
    for seq in sequences:
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


def load_stats_dir(
    stats_dir: Path,
    dataset: str | None = None,
    base_dir: Path | None = None,
) -> dict:
    """Load all ``*_stats.json`` files from one loma_v2 output directory.

    Args:
        stats_dir: Directory containing per-sequence stats files.
        dataset: Optional dataset CLI key for preview generation.
        base_dir: Optional dataset root for preview generation.

    Returns:
        JSON-serializable data bundle for report generation.

    Raises:
        ValueError: If no stats files are found.
    """
    json_files = sorted(stats_dir.glob("*_stats.json"))
    if not json_files:
        raise ValueError(f"No *_stats.json files found in {stats_dir}")

    sequences = []
    for json_file in json_files:
        raw = json.loads(json_file.read_text())
        aggregate = dict(raw.get("aggregate", {}))
        frames = list(raw.get("frames", []))

        def _series_from_frames(frame_key: str, aggregate_key: str) -> list:
            if frames:
                return [frame.get(frame_key) for frame in frames]
            value = aggregate.get(aggregate_key)
            return [value] if value is not None else []

        series = {
            "init_detected_object": [raw.get("init_detected_object")],
            "init_detected_background": [raw.get("init_detected_background")],
            "detected_current_object": _series_from_frames(
                "detected_current_object", "detected_current_object_mean"
            ),
            "detected_current_background": _series_from_frames(
                "detected_current_background", "detected_current_background_mean"
            ),
            "matched_total": _series_from_frames("matched_total", "matched_total_mean"),
            "matched_object": _series_from_frames(
                "matched_object", "matched_object_mean"
            ),
            "matched_background": _series_from_frames(
                "matched_background", "matched_background_mean"
            ),
            "inliers_total": _series_from_frames("inliers_total", "inliers_total_mean"),
            "inliers_object": _series_from_frames(
                "inliers_object", "inliers_object_mean"
            ),
            "inliers_background": _series_from_frames(
                "inliers_background", "inliers_background_mean"
            ),
            "inlier_ratio": _series_from_frames("inlier_ratio", "inlier_ratio_mean"),
        }
        sequences.append(
            {
                "name": raw.get("sequence_name")
                or json_file.stem.removesuffix("_stats"),
                "num_frames": int(raw.get("num_frames", 0)),
                "num_tracked_frames": int(raw.get("num_tracked_frames", 0)),
                "num_masked_frames": int(aggregate.get("num_masked_frames", 0)),
                "series": series,
            }
        )

    return {
        "metrics": _METRICS,
        "sequences": sequences,
        "sequence_preview_jpg": (
            _build_sequence_preview_images(dataset, base_dir, sequences)
            if dataset is not None and base_dir is not None
            else {}
        ),
    }


def _build_html(data: dict, title: str, cdn: bool = False) -> str:
    """Build self-contained HTML report."""
    data_json = json.dumps(data, separators=(",", ":"))
    cards_html = "".join(
        (
            f'<div class="card">'
            f'<div class="plot-title">{metric["label"]}</div>'
            f'<div class="plot-subtitle" id="sub-{metric["key"]}"></div>'
            f'<div id="plot-{metric["key"]}" style="height:280px;"></div>'
            f"</div>"
        )
        for metric in data["metrics"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
{get_plotly_script_tag(cdn)}
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 18px 20px 28px;
    font-family: system-ui, sans-serif;
    font-size: 13px;
    background: #f4f4f4;
    color: #222;
  }}
  h1 {{ margin: 0 0 6px; font-size: 1.35rem; }}
  #summary {{ margin: 0 0 14px; color: #666; line-height: 1.45; }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 14px;
    padding: 10px 12px;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    color: #333;
  }}
  .controls label {{ display: flex; align-items: center; gap: 8px; cursor: pointer; }}
  .controls select {{ font: inherit; padding: 3px 6px; }}
  .card {{
    background: #fff;
    border-radius: 10px;
    padding: 12px 14px 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
  }}
  .plot-title {{ font-weight: 700; margin-bottom: 2px; }}
  .plot-subtitle {{ font-size: 12px; color: #666; margin-bottom: 6px; min-height: 18px; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 12px;
  }}
  .note {{
    margin: 0 0 16px;
    padding: 10px 12px;
    border-left: 3px solid #777;
    background: #fff;
    border-radius: 8px;
    color: #444;
    line-height: 1.5;
  }}
  .selection-panel {{
    margin: 0 0 16px;
    padding: 12px;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
  }}
  .selection-title {{ font-weight: 700; margin-bottom: 4px; }}
  .selection-subtitle {{ color: #666; margin-bottom: 8px; }}
  .selection-content {{
    display: grid;
    grid-template-columns: minmax(260px, 1.25fr) minmax(240px, 0.9fr);
    gap: 12px;
    align-items: start;
  }}
  .selection-list {{
    max-height: 240px;
    overflow: auto;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 8px 10px;
    background: #fafafa;
  }}
  .selection-item {{
    display: block;
    width: 100%;
    border: 0;
    border-radius: 4px;
    background: transparent;
    text-align: left;
    padding: 4px 6px;
    cursor: pointer;
    font: inherit;
    font-family: ui-monospace, SFMono-Regular, monospace;
    line-height: 1.45;
  }}
  .selection-item:hover {{ background: #ececec; }}
  .selection-item.active {{ background: #dbeafe; }}
  .preview-card {{
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 10px;
    background: #fafafa;
  }}
  .preview-media {{
    min-height: 140px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #ddd;
    border-radius: 6px;
    background: #fff;
    margin-bottom: 8px;
    overflow: hidden;
  }}
  .preview-media img {{ display: none; max-width: 100%; max-height: 220px; }}
  .preview-empty {{ color: #777; }}
  .preview-name {{ font-weight: 700; margin-bottom: 4px; }}
  .preview-metrics {{ color: #555; line-height: 1.45; }}
  @media (max-width: 900px) {{
    .selection-content {{ grid-template-columns: 1fr; }}
  }}
  code {{ background: #eee; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div id="summary"></div>
<div class="controls">
  <label>sequence statistic
    <select id="sel-agg">
      <option value="mean">mean</option>
      <option value="median">median</option>
      <option value="p10">p10</option>
      <option value="p25">p25</option>
      <option value="p75">p75</option>
      <option value="p90">p90</option>
      <option value="min">min</option>
      <option value="max">max</option>
    </select>
  </label>
  <label><input type="checkbox" id="chk-cumulative"> cumulative histograms</label>
</div>
<div class="note">
  Each histogram uses one aggregated value per sequence. Sequence reduction can switch between
  mean, median, percentiles, min, max. Matched/inlier object-vs-background split uses init-frame
  keypoint labels (<code>kptA</code>). Current detected object-vs-background split uses current-frame
  mask when available.
</div>
<div class="selection-panel">
  <div class="selection-title">Selected Sequences</div>
  <div class="selection-subtitle" id="selection-subtitle">Click histogram bin to list matching sequences.</div>
  <div class="selection-content">
    <div class="selection-list" id="selection-list">No selection.</div>
    <div class="preview-card">
      <div class="preview-media">
        <img id="seq-preview-img" alt="sequence preview">
        <div class="preview-empty" id="seq-preview-empty">Click sequence name to show preview.</div>
      </div>
      <div class="preview-name" id="seq-preview-name">No sequence selected</div>
      <div class="preview-metrics" id="seq-preview-metrics">-</div>
    </div>
  </div>
</div>
<div class="grid">
  {cards_html}
</div>
<script>
const DATA = {data_json};
const sequences = DATA.sequences;
const metrics = DATA.metrics;
const seqPreviewImgEl = document.getElementById('seq-preview-img');
const seqPreviewEmptyEl = document.getElementById('seq-preview-empty');
const seqPreviewNameEl = document.getElementById('seq-preview-name');
const seqPreviewMetricsEl = document.getElementById('seq-preview-metrics');
let currentSelection = null;

function valuesFor(metricKey) {{
  return sequences
    .map(seq => ({{ name: seq.name, value: aggregateSeries(seq.series[metricKey], currentAggMode()) }}))
    .filter(item => item.value !== null && Number.isFinite(item.value));
}}

function valueForSequence(metricKey, seqName) {{
  const seq = sequences.find(item => item.name === seqName);
  if (!seq) return null;
  const value = aggregateSeries(seq.series[metricKey], currentAggMode());
  return Number.isFinite(value) ? value : null;
}}

function fmt(metricKey, value) {{
  if (metricKey === 'inlier_ratio') return (100 * value).toFixed(1) + '%';
  return value.toFixed(1);
}}

function mean(values) {{
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}}

function median(values) {{
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}}

function percentile(values, q) {{
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  if (s.length === 1) return s[0];
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return s[lo];
  const frac = pos - lo;
  return s[lo] * (1 - frac) + s[hi] * frac;
}}

function currentAggMode() {{
  return document.getElementById('sel-agg').value;
}}

function aggregateSeries(values, mode) {{
  const valid = (values || []).filter(v => v !== null && Number.isFinite(v));
  if (!valid.length) return null;
  if (mode === 'mean') return mean(valid);
  if (mode === 'median') return median(valid);
  if (mode === 'p10') return percentile(valid, 0.10);
  if (mode === 'p25') return percentile(valid, 0.25);
  if (mode === 'p75') return percentile(valid, 0.75);
  if (mode === 'p90') return percentile(valid, 0.90);
  if (mode === 'min') return Math.min(...valid);
  if (mode === 'max') return Math.max(...valid);
  return mean(valid);
}}

function isCountMetric(metricKey) {{
  return metricKey !== 'inlier_ratio';
}}

function niceStep(value) {{
  if (value <= 4) return 4;
  if (value <= 8) return 8;
  if (value <= 16) return 16;
  if (value <= 32) return 32;
  const pow10 = Math.pow(10, Math.floor(Math.log10(value)));
  const scaled = value / pow10;
  if (scaled <= 1) return 1 * pow10;
  if (scaled <= 2) return 2 * pow10;
  if (scaled <= 5) return 5 * pow10;
  return 10 * pow10;
}}

function countBinEdges(values) {{
  const fixed = [0, 4, 8, 16, 32];
  const maxVal = Math.max(...values);
  if (maxVal < 32) return fixed;

  const targetStep = Math.max(8, Math.ceil((maxVal - 32) / 10));
  const step = niceStep(targetStep);
  const edges = [...fixed];
  let edge = 32;
  while (edge <= maxVal) {{
    edge += step;
    edges.push(edge);
  }}
  return edges;
}}

function countHistogram(rows) {{
  const values = rows.map(r => r.value);
  const edges = countBinEdges(values);
  const counts = new Array(edges.length - 1).fill(0);
  const bins = counts.map(() => []);
  rows.forEach(row => {{
    const v = row.value;
    for (let i = 0; i < edges.length - 1; i++) {{
      const lo = edges[i];
      const hi = edges[i + 1];
      const isLast = i === edges.length - 2;
      if ((v >= lo && v < hi) || (isLast && v <= hi)) {{
        counts[i] += 1;
        bins[i].push(row);
        break;
      }}
    }}
  }});

  const labels = counts.map((_, i) => {{
    const lo = edges[i];
    const hi = edges[i + 1] - 1;
    return `${{lo}}-${{hi}}`;
  }});
  return {{ labels, counts, bins }};
}}

function ratioBinEdges(values) {{
  const maxVal = Math.max(...values);
  const nbins = Math.max(6, Math.min(24, Math.ceil(Math.sqrt(values.length)) * 2));
  const lo = 0;
  const hi = Math.max(1, maxVal);
  if (hi <= lo) return [0, 1];
  const step = (hi - lo) / nbins;
  const edges = [lo];
  for (let i = 1; i <= nbins; i++) {{
    edges.push(lo + step * i);
  }}
  return edges;
}}

function ratioHistogram(rows) {{
  const values = rows.map(r => r.value);
  const edges = ratioBinEdges(values);
  const counts = new Array(edges.length - 1).fill(0);
  const bins = counts.map(() => []);
  rows.forEach(row => {{
    const v = row.value;
    for (let i = 0; i < edges.length - 1; i++) {{
      const lo = edges[i];
      const hi = edges[i + 1];
      const isLast = i === edges.length - 2;
      if ((v >= lo && v < hi) || (isLast && v <= hi)) {{
        counts[i] += 1;
        bins[i].push(row);
        break;
      }}
    }}
  }});

  const labels = counts.map((_, i) => {{
    const lo = 100 * edges[i];
    const hi = 100 * edges[i + 1];
    return `${{lo.toFixed(1)}}-${{hi.toFixed(1)}}%`;
  }});
  return {{ labels, counts, bins }};
}}

function cumulativeCounts(counts) {{
  let acc = 0;
  return counts.map(v => {{
    acc += v;
    return acc;
  }});
}}

function sequenceTickVals(total) {{
  if (total <= 0) return [0];
  return [0, total / 4, total / 2, (3 * total) / 4, total];
}}

function sequenceAxisConfig(cumulative) {{
  const total = sequences.length;
  return {{
    title: cumulative ? 'cumulative sequences' : 'sequences',
    rangemode: 'tozero',
    range: [0, total],
    tickmode: 'array',
    tickvals: sequenceTickVals(total),
    ticktext: sequenceTickVals(total).map(v => String(Math.round(v))),
    gridcolor: '#d8d8d8',
    zerolinecolor: '#b8b8b8',
  }};
}}

function selectedRowsForBin(bins, binIdx, cumulative) {{
  if (!cumulative) return bins[binIdx] || [];
  const selected = [];
  for (let i = 0; i <= binIdx; i++) {{
    selected.push(...(bins[i] || []));
  }}
  return selected;
}}

function showSelection(metric, label, rows, cumulative) {{
  const sorted = [...rows].sort((a, b) => a.value - b.value || a.name.localeCompare(b.name));
  currentSelection = {{
    metricKey: metric.key,
    metricLabel: metric.label,
    sourceLabel: label,
    sourceCumulative: cumulative,
    seqNames: sorted.map(row => row.name),
    activeSeqName: null,
  }};
  renderSelectionFromState();
}}

function renderSelectionFromState() {{
  const subtitleEl = document.getElementById('selection-subtitle');
  const listEl = document.getElementById('selection-list');
  if (!currentSelection) {{
    subtitleEl.textContent = 'Click histogram bin to list matching sequences.';
    listEl.textContent = 'No selection.';
    clearSequencePreview();
    return;
  }}

  const metric = metrics.find(item => item.key === currentSelection.metricKey);
  const rows = currentSelection.seqNames
    .map(name => ({{ name, value: valueForSequence(currentSelection.metricKey, name) }}))
    .filter(row => row.value !== null);
  rows.sort((a, b) => a.value - b.value || a.name.localeCompare(b.name));

  subtitleEl.textContent =
    `${{currentSelection.metricLabel}} · selected set from ${{currentSelection.sourceCumulative ? 'cumulative through ' : 'bin '}}${{currentSelection.sourceLabel}} · seq ${{currentAggMode()}} values · ${{rows.length}} sequence(s)`;
  listEl.innerHTML = '';
  if (!rows.length) {{
    listEl.textContent = 'No sequence matched selection.';
    clearSequencePreview();
    return;
  }}

  for (const row of rows) {{
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'selection-item';
    btn.textContent = `${{row.name}}  (${{fmt(metric.key, row.value)}})`;
    btn.addEventListener('click', () => {{
      for (const el of listEl.querySelectorAll('.selection-item')) el.classList.remove('active');
      btn.classList.add('active');
      currentSelection.activeSeqName = row.name;
      setSequencePreview(row.name, `${{metric.label}} · seq ${{currentAggMode()}} = ${{fmt(metric.key, row.value)}}`);
    }});
    if (row.name === currentSelection.activeSeqName) {{
      btn.classList.add('active');
    }}
    listEl.appendChild(btn);
  }}

  if (currentSelection.activeSeqName !== null) {{
    const activeValue = valueForSequence(currentSelection.metricKey, currentSelection.activeSeqName);
    if (activeValue !== null) {{
      setSequencePreview(
        currentSelection.activeSeqName,
        `${{currentSelection.metricLabel}} · seq ${{currentAggMode()}} = ${{fmt(currentSelection.metricKey, activeValue)}}`
      );
      return;
    }}
    currentSelection.activeSeqName = null;
  }}
  clearSequencePreview();
}}

function clearSelection() {{
  currentSelection = null;
  renderSelectionFromState();
}}

function setSequencePreview(seqName, metricsText) {{
  seqPreviewNameEl.textContent = seqName;
  seqPreviewMetricsEl.textContent = metricsText;
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
  seqPreviewEmptyEl.textContent = 'Click sequence name to show preview.';
  seqPreviewEmptyEl.style.display = 'block';
}}

function clickBinIndex(ev) {{
  if (!ev.points || !ev.points.length) return null;
  const p = ev.points[0];
  if (p.pointIndex !== undefined) return p.pointIndex;
  if (p.pointNumber !== undefined) return p.pointNumber;
  return null;
}}

function renderAll() {{
  const aggMode = currentAggMode();
  const cumulative = document.getElementById('chk-cumulative').checked;

  metrics.forEach(metric => {{
    const rows = valuesFor(metric.key);
    const vals = rows.map(r => r.value);
    const mu = mean(vals);
    const med = median(vals);
    const subtitle = document.getElementById('sub-' + metric.key);

    if (!vals.length) {{
      subtitle.textContent = 'No data';
      Plotly.newPlot('plot-' + metric.key, [], {{
        margin: {{ l: 36, r: 10, t: 8, b: 36 }},
        paper_bgcolor: '#fff',
        plot_bgcolor: '#fff',
        xaxis: {{ visible: false }},
        yaxis: {{ visible: false }},
        annotations: [{{
          text: 'No data',
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          showarrow: false,
          font: {{ size: 16, color: '#888' }},
        }}],
      }}, {{ responsive: true, displayModeBar: false }});
      return;
    }}

    subtitle.textContent =
      `${{vals.length}} seqs · seq ${{aggMode}} histogram · sample mean ${{fmt(metric.key, mu)}} · sample median ${{fmt(metric.key, med)}}${{cumulative ? ' · cumulative' : ''}}`;

    const plotEl = document.getElementById('plot-' + metric.key);

    if (isCountMetric(metric.key)) {{
      const hist = countHistogram(rows);
      const ys = cumulative ? cumulativeCounts(hist.counts) : hist.counts;
      Plotly.newPlot(
        plotEl,
        [{{
          type: 'bar',
          x: hist.labels,
          y: ys,
          marker: {{ color: metric.color, line: {{ color: '#ffffff', width: 1 }} }},
          hovertemplate: cumulative
            ? '<= bin %{{x}}<extra>%{{y}} sequences</extra>'
            : 'bin %{{x}}<extra>%{{y}} sequences</extra>',
        }}],
        {{
          margin: {{ l: 42, r: 10, t: 8, b: 52 }},
          paper_bgcolor: '#fff',
          plot_bgcolor: '#fff',
          bargap: 0.08,
          xaxis: {{ title: cumulative ? 'count upper bin' : 'count bin' }},
          yaxis: sequenceAxisConfig(cumulative),
        }},
        {{ responsive: true, displayModeBar: false }}
      ).then(() => {{
        plotEl.removeAllListeners?.('plotly_click');
        plotEl.on('plotly_click', ev => {{
          const idx = clickBinIndex(ev);
          if (idx === null) return;
          showSelection(metric, hist.labels[idx], selectedRowsForBin(hist.bins, idx, cumulative), cumulative);
        }});
      }});
      return;
    }}

    const hist = ratioHistogram(rows);
    const ys = cumulative ? cumulativeCounts(hist.counts) : hist.counts;
    Plotly.newPlot(
      plotEl,
      [{{
        type: 'bar',
        x: hist.labels,
        y: ys,
        marker: {{ color: metric.color, line: {{ color: '#ffffff', width: 1 }} }},
        hovertemplate: cumulative
          ? '<= bin %{{x}}<extra>%{{y}} sequences</extra>'
          : 'bin %{{x}}<extra>%{{y}} sequences</extra>',
      }}],
      {{
        margin: {{ l: 42, r: 10, t: 8, b: 38 }},
        paper_bgcolor: '#fff',
        plot_bgcolor: '#fff',
        bargap: 0.04,
        xaxis: {{ title: cumulative ? 'ratio upper bin' : 'ratio bin' }},
        yaxis: sequenceAxisConfig(cumulative),
      }},
      {{ responsive: true, displayModeBar: false }}
    ).then(() => {{
      plotEl.removeAllListeners?.('plotly_click');
      plotEl.on('plotly_click', ev => {{
        const idx = clickBinIndex(ev);
        if (idx === null) return;
        showSelection(metric, hist.labels[idx], selectedRowsForBin(hist.bins, idx, cumulative), cumulative);
      }});
    }});
  }});
  renderSelectionFromState();
}}

document.getElementById('summary').textContent =
  `${{sequences.length}} sequences loaded · one sequence vote per histogram`;
document.getElementById('sel-agg').addEventListener('change', renderAll);
document.getElementById('chk-cumulative').addEventListener('change', renderAll);
renderSelectionFromState();
renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate aggregate histogram report for loma_v2 keypoint stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "stats_dir",
        type=Path,
        help="Directory containing per-sequence *_stats.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <stats_dir>/keypoint_stats_report.html.",
    )
    parser.add_argument(
        "--title",
        help="Report title. Defaults to the stats directory name.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        help="Dataset CLI key used to build sequence previews.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Dataset root used to build sequence previews. Defaults to path_config.",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Load Plotly from CDN instead of inlining it (requires internet).",
    )
    args = parser.parse_args()

    stats_dir = args.stats_dir.resolve()
    if not stats_dir.is_dir():
        print(f"ERROR: {stats_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output = args.output or stats_dir / "keypoint_stats_report.html"
    title = args.title or f"LoMa v2 keypoint stats — {stats_dir.parent.name}"

    base_dir = args.base_dir
    if args.dataset is not None and base_dir is None:
        base_dir = getattr(paths, DATASETS[args.dataset].base_dir_attr)

    data = load_stats_dir(stats_dir, dataset=args.dataset, base_dir=base_dir)
    output.write_text(_build_html(data, title, cdn=args.cdn), encoding="utf-8")
    print(f"Report → {output}")


if __name__ == "__main__":
    main()
