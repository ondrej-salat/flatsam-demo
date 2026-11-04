#!/usr/bin/env python3
"""Generate an interactive HTML report for LoMa analysis results.

Reads *_analysis.json files from a LoMa output directory and generates a
self-contained HTML file with Plotly.js interactive plots showing per-frame
statistics across sequences.  Two slider-controlled plots show:

  1. Fraction of sequences with ≥N inlier count at M px threshold
  2. Fraction of sequences with ≥N% inlier ratio at M px threshold

Both plotted as a function of frame index, with dynamic sliders for N and M.

Usage examples:
    python experiments/loma_report.py \\
        /mnt/personal/serycjon/flatsam2/LoMa/LoMaB/POT-210 \\
        --output loma_report.html

    python experiments/loma_report.py \\
        /mnt/personal/serycjon/flatsam2/LoMa/LoMaG/PTI \\
        --output /tmp/loma_pti.html \\
        --title "LoMaG on PTI"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.report_utils import get_plotly_script_tag


def load_analysis_dir(analysis_dir: Path) -> dict:
    """Load all *_analysis.json files from *analysis_dir*.

    Args:
        analysis_dir: Directory containing per-sequence analysis JSON files.

    Returns:
        Dict with keys:
          - ``thresholds``: list[int] of pixel thresholds.
          - ``sequences``: list of dicts, each with ``name`` and ``frames``.
            Each frame dict has ``idx``, ``n`` (total matches), ``nt`` (target
            matches), ``frac`` (list, one per threshold, or None), ``cnt``
            (list, one per threshold, or None).

    Raises:
        ValueError: If no analysis files are found.
    """
    json_files = sorted(analysis_dir.glob("*_analysis.json"))
    if not json_files:
        raise ValueError(f"No *_analysis.json files found in {analysis_dir}")

    sequences = []
    all_thresholds: Optional[list] = None

    for json_file in json_files:
        seq_name = json_file.stem.removesuffix("_analysis")
        raw = json.loads(json_file.read_text())
        thresholds: list = raw["thresholds_px"]

        if all_thresholds is None:
            all_thresholds = thresholds
        elif all_thresholds != thresholds:
            print(
                f"WARNING: threshold mismatch in {json_file.name} "
                f"({thresholds} vs {all_thresholds}) — skipping mismatched keys",
                file=sys.stderr,
            )

        frames = []
        for f in raw["frames"]:
            if f.get("missing"):
                continue

            frac: Optional[list] = None
            cnt: Optional[list] = None
            if f.get("valid"):
                frac = [f.get(f"target_good_at_{t}px") for t in thresholds]
                cnt = [f.get(f"target_good_count_at_{t}px") for t in thresholds]
                # Treat all-None as no valid data
                if all(v is None for v in frac):
                    frac = None
                    cnt = None

            frames.append({
                "idx": f["frame_idx"],
                "n": f.get("n_matches"),
                "nt": f.get("n_target_kpts"),
                "frac": frac,
                "cnt": cnt,
            })

        if frames:
            sequences.append({"name": seq_name, "frames": frames})

    return {
        "thresholds": all_thresholds or [],
        "sequences": sequences,
    }


_PCT_THRESHOLDS  = [10, 20, 30, 50, 75, 90, 95]
_COUNT_THRESHOLDS = [20, 50, 100]


def _format_summary_from_data(data: dict) -> str:
    """Reproduce the stdout summary from loaded analysis data.

    Mirrors the output of ``_print_summary`` in loma_experiment.py.  Density
    stats (obj/bg area) are omitted — that information is not stored in the
    simplified per-frame dicts produced by :func:`load_analysis_dir`.

    Args:
        data: Dict returned by :func:`load_analysis_dir`.

    Returns:
        Multi-line string matching the printed summary format.
    """
    import numpy as np

    thresholds = data["thresholds"]
    sequences  = data["sequences"]
    n_seq      = len(sequences)

    all_frames   = [f for seq in sequences for f in seq["frames"]]
    valid_frames = [f for f in all_frames if f.get("frac") is not None]
    n_frames     = len(all_frames)
    n_valid      = len(valid_frames)

    lines: list[str] = []
    lines.append(f"\n--- Summary ---")
    lines.append(f"  Sequences         : {n_seq}")
    lines.append(f"  Frames            : {n_frames}  (with GT: {n_valid})")

    n_vals = [f["n"] for f in all_frames if f.get("n") is not None]
    if n_vals:
        lines.append(f"  Tentative corr./frame : {float(np.mean(n_vals)):.1f}")

    obj_fracs = [
        f["nt"] / f["n"]
        for f in all_frames
        if f.get("n") and f.get("nt") is not None and f["n"] > 0
    ]
    if obj_fracs:
        lines.append(f"  Mean kptA obj fraction : {float(np.mean(obj_fracs)):.3f}")

    if not thresholds or not valid_frames:
        return "\n".join(lines)

    thr_cols = "".join(f"  @{t:>3d}px" for t in thresholds)

    lines.append(
        f"\n  Target match accuracy (mean inlier fraction, {n_valid} valid frames):"
    )
    lines.append(f"    {'':10s}{thr_cols}")
    row = "    " + f"{'mean frac':10s}"
    for i in range(len(thresholds)):
        vals = [
            f["frac"][i] if (f["frac"] and f["frac"][i] is not None) else 0.0
            for f in valid_frames
        ]
        row += f"  {float(np.mean(vals)):>6.3f}" if vals else f"  {'n/a':>6s}"
    lines.append(row)

    lines.append(
        f"\n  Fraction of valid frames with ≥N% inliers ({n_valid} frames):"
    )
    lines.append(f"    {'N%':10s}{thr_cols}")
    for pct in _PCT_THRESHOLDS:
        row = f"    {f'≥{pct}%':10s}"
        for i in range(len(thresholds)):
            vals = [
                f["frac"][i] if (f["frac"] and f["frac"][i] is not None) else 0.0
                for f in valid_frames
            ]
            frac = float(np.mean([v >= pct / 100 for v in vals])) if vals else float("nan")
            row += f"  {frac:>6.3f}"
        lines.append(row)

    lines.append(
        f"\n  Fraction of valid frames with ≥N absolute inliers ({n_valid} frames):"
    )
    lines.append(f"    {'N':10s}{thr_cols}")
    for n_min in _COUNT_THRESHOLDS:
        row = f"    {f'≥{n_min}':10s}"
        for i in range(len(thresholds)):
            counts = [
                f["cnt"][i] if (f["cnt"] and f["cnt"][i] is not None) else 0
                for f in valid_frames
            ]
            frac = float(np.mean([c >= n_min for c in counts])) if counts else float("nan")
            row += f"  {frac:>6.3f}"
        lines.append(row)

    return "\n".join(lines)


def _build_html(data: dict, title: str, summary_text: str = "", cdn: bool = False) -> str:  # noqa: C901
    data_json = json.dumps(data, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Python helpers — generate HTML control snippets.
    # These produce plain strings (not f-strings), so no {{ }} escaping.
    # ------------------------------------------------------------------
    def _slider(key: str, label: str, mn: int, mx: int, default: int,
                step: int = 1, disp: str = "") -> str:
        d = disp or str(default)
        return (f'<div class="ctrl"><label>{label}</label>'
                f'<input type="range" data-sync="{key}"'
                f' min="{mn}" max="{mx}" value="{default}" step="{step}">'
                f'<span class="val" data-sync-val="{key}">{d}</span></div>')

    def _sel(key: str, label: str,
             options: list[tuple[str, str]], selected: str = "") -> str:
        opts = "".join(
            f'<option value="{v}"{" selected" if v == selected else ""}>{t}</option>'
            for v, t in options
        )
        return (f'<div class="ctrl"><label>{label}</label>'
                f'<select data-sync="{key}">{opts}</select></div>')

    def _check(key: str, label: str, default: bool = True) -> str:
        ch = " checked" if default else ""
        return (f'<div class="ctrl"><label>{label}</label>'
                f'<input type="checkbox" data-sync="{key}"{ch}></div>')

    def _thr() -> str:
        return ('<div class="ctrl"><label>Px threshold (M):</label>'
                '<select data-sync="thrIdx"></select></div>')

    def _grid(*items: str) -> str:
        return '<div class="controls-grid">' + "".join(items) + "</div>"

    def _ccard(*rows: str) -> str:
        return '<div class="card ctrl-card">' + "".join(rows) + "</div>"

    def _shdr(text: str) -> str:
        return f'<h2 class="section-hdr">{text}</h2>'

    def _pgrid(*cards: str) -> str:
        return '<div class="plots-grid">' + "".join(cards) + "</div>"

    def _pcard(title_id: str, plot_id: str, height: int = 340) -> str:
        return (f'<div class="card">'
                f'<div class="plot-title" id="{title_id}"></div>'
                f'<div id="{plot_id}" style="height:{height}px;"></div></div>')

    # Reusable control snippets
    C_THR    = _thr()
    C_SEQS   = _slider("minSeqs",  "Min seqs at frame:", 1, 200, 30)
    C_SMOOTH = _slider("smoothW",  "Smooth (frames):", 1, 30, 1, disp="off")
    C_XMODE  = _sel("xMode", "X axis:",
                    [("abs", "Absolute frame index"), ("rel", "Relative position (0–1)")],
                    selected="abs")
    C_COUNT  = _slider("minCount", "Min inlier count (N):", 4, 300, 20)
    C_RATIO  = _slider("minRatio", "Min inlier ratio N%:", 0, 100, 30, disp="30%")
    C_STAT   = _sel("statMode", "Statistic:",
                    [("mean","Mean"),("0.1","P10"),("0.25","P25"),
                     ("0.5","P50 (median)"),("0.75","P75"),("0.9","P90")],
                    selected="0.5")
    C_BAND   = _check("showBand", "Show P25–P75 band:", default=True)
    C_W      = _slider("wndW", "Window W (frames):", 1, 50, 5)
    C_WTYPE  = _sel("wndType", "Window type:",
                    [("last","Last W frames (causal)"),("center","Centered ±W/2")],
                    selected="last")
    C_HMMET  = _sel("hmMetric", "Heatmap metric:",
                    [("frac","Inlier ratio (target kpts)"),
                     ("cnt","Inlier count (target kpts)"),
                     ("n","Total n_matches"),
                     ("objfrac","kptA obj fraction")],
                    selected="frac")
    C_HMSORT = _sel("hmSort", "Sort rows by:",
                    [("mean","Mean value"),
                     ("firstfail","First-failure frame"),
                     ("name","Sequence name")],
                    selected="mean")
    C_COVM   = _sel("covMode", "Coverage curve:",
                    [("contain", "% seqs containing frame"),
                     ("dataCov", "% with data (of those containing frame)")],
                    selected="contain")

    # Escape summary text for HTML embedding
    safe_summary = (summary_text.strip()
                    .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    summary_html = (
        f'<div class="card">'
        f'<div class="section-label">Textual summary</div>'
        f'<pre style="margin:0;font-size:12px;line-height:1.5;white-space:pre-wrap;">'
        f'{safe_summary}</pre></div>'
    ) if safe_summary else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
{get_plotly_script_tag(cdn)}
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; font-size: 13px;
         margin: 0; padding: 16px 20px; background: #f0f0f0; color: #222; }}
  h1 {{ font-size: 1.3em; margin: 0 0 4px; }}
  h2.section-hdr {{
    font-size: 0.95em; font-weight: 700; margin: 22px 0 6px;
    padding-left: 9px; border-left: 3px solid #888; color: #333;
  }}
  .card {{
    background: white; border-radius: 8px; padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 10px;
  }}
  .ctrl-card {{ padding: 10px 16px; }}
  .controls-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 6px 20px; align-items: center;
  }}
  .ctrl {{ display: flex; align-items: center; gap: 8px; }}
  .ctrl label {{ font-weight: 600; white-space: nowrap; font-size: 0.88em; }}
  .ctrl input[type=range] {{ flex: 1; min-width: 100px; }}
  .ctrl select {{ padding: 2px 5px; border: 1px solid #bbb;
                  border-radius: 4px; background: #fafafa; font-size: 0.88em; }}
  .ctrl input[type=checkbox] {{ width: 14px; height: 14px; cursor: pointer; }}
  .val {{ font-family: monospace; font-weight: 700; min-width: 44px; font-size: 0.9em; }}
  .plots-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 0; }}
  @media (max-width: 860px) {{ .plots-grid {{ grid-template-columns: 1fr; }} }}
  .plot-title {{ font-weight: 600; margin-bottom: 3px; font-size: 0.85em; color: #444; }}
  #page-stats {{ font-size: 0.82em; color: #777; margin-bottom: 14px; }}
  .section-label {{
    font-size: 0.72em; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; color: #888; margin-bottom: 4px;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<div id="page-stats"></div>

{_shdr("Per-frame fraction above threshold (target kpts)")}
{_ccard(_grid(C_THR, C_SEQS, C_SMOOTH, C_XMODE, C_COUNT, C_RATIO, C_COVM))}
{_pgrid(_pcard("title-count","plot-count"), _pcard("title-ratio","plot-ratio"))}

{_shdr("Windowed fraction — ≥1 success in W frames (target kpts)")}
{_ccard(_grid(C_THR, C_SEQS, C_SMOOTH, C_XMODE, C_COUNT, C_RATIO, C_W, C_WTYPE, C_COVM))}
{_pgrid(_pcard("title-wnd-cnt","plot-wnd-cnt",300), _pcard("title-wnd-rat","plot-wnd-rat",300))}

{_shdr("Quantile of inlier metrics (target kpts, GT frames only)")}
{_ccard(_grid(C_THR, C_SEQS, C_SMOOTH, C_XMODE, C_STAT, C_BAND, C_COVM))}
{_pgrid(_pcard("title-qcnt","plot-qcnt",300), _pcard("title-qrat","plot-qrat",300))}

{_shdr("Match statistics overview (all frames)")}
{_ccard(_grid(C_SEQS, C_SMOOTH, C_XMODE, C_STAT, C_BAND, C_COVM))}
{_pgrid(_pcard("title-matches","plot-matches",280), _pcard("title-objfrac","plot-objfrac",280))}

{_shdr("Survival analysis (target kpts)")}
{_ccard(_grid(C_THR, C_SMOOTH, C_XMODE, C_COUNT, C_RATIO))}
{_pgrid(_pcard("title-surv-cnt","plot-surv-cnt",300), _pcard("title-surv-rat","plot-surv-rat",300))}

{_shdr("Per-sequence heatmap")}
{_ccard(_grid(C_THR, C_XMODE, C_HMMET, C_HMSORT))}
<div class="card">
  <div class="plot-title" id="title-heatmap"></div>
  <div id="plot-heatmap"></div>
</div>

{summary_html}

<script>
const DATA = {data_json};
const thresholds = DATA.thresholds;
const sequences  = DATA.sequences;
const nSeq = sequences.length;

// ---- populate all threshold selectors (data-sync="thrIdx") ----
thresholds.forEach((t, i) => {{
  document.querySelectorAll('[data-sync="thrIdx"]').forEach(sel => {{
    const o = document.createElement('option');
    o.value = i; o.textContent = t + ' px';
    sel.appendChild(o);
  }});
}});

// ---- page stats ----
let maxFrameIdx = 0;
sequences.forEach(s => s.frames.forEach(f => {{ if (f.idx > maxFrameIdx) maxFrameIdx = f.idx; }}));
document.getElementById('page-stats').textContent =
  `${{nSeq}} sequences · max frame index ${{maxFrameIdx}} · ${{thresholds.length}} thresholds: ${{thresholds.join(', ')}} px`;

// ---- central state ----
const S = {{
  thrIdx:   String(Math.max(0, thresholds.findIndex(t => t >= 10))),
  minCount: '20', minRatio: '30', minSeqs: '30', smoothW: '1',
  xMode: 'abs', statMode: '0.5', showBand: true,
  wndW: '5', wndType: 'last', hmMetric: 'frac', hmSort: 'mean',
  covMode: 'contain',
}};

function fmtVal(key, v) {{
  if (key === 'minRatio') return v + '%';
  if (key === 'smoothW')  return parseInt(v) <= 1 ? 'off' : v + ' fr';
  return String(v);
}}

function syncKey(key, v) {{
  document.querySelectorAll(`[data-sync="${{key}}"]`).forEach(el => {{
    if (el.type === 'checkbox') el.checked = v;
    else el.value = v;
  }});
  document.querySelectorAll(`[data-sync-val="${{key}}"]`).forEach(sp => {{
    sp.textContent = fmtVal(key, v);
  }});
}}

// Set initial values on all controls
Object.entries(S).forEach(([k, v]) => syncKey(k, v));
// Set thrIdx selects to correct default
syncKey('thrIdx', S.thrIdx);

// Clamp minSeqs max
document.querySelectorAll('[data-sync="minSeqs"]').forEach(el => {{
  el.max = Math.min(nSeq, 200);
}});

function wireControls() {{
  document.querySelectorAll('[data-sync]').forEach(el => {{
    const key = el.dataset.sync;
    const evtType = el.type === 'range' ? 'input' : 'change';
    el.addEventListener(evtType, () => {{
      const v = el.type === 'checkbox' ? el.checked : el.value;
      S[key] = v;
      syncKey(key, v);
      updatePlots();
    }});
  }});
}}

// ---- precompute per-threshold frame buckets ----
const precomputed = thresholds.map((_, thrIdx) => {{
  function addToBucket(buckets, key, f) {{
    if (!buckets[key]) buckets[key] = {{ cnts: [], fracs: [], n: [], objfrac: [] }};
    const b = buckets[key];
    b.cnts.push(f.cnt ? f.cnt[thrIdx] : null);
    b.fracs.push(f.frac ? f.frac[thrIdx] : null);
    b.n.push(f.n != null ? f.n : null);
    b.objfrac.push((f.n != null && f.nt != null && f.n > 0) ? f.nt / f.n : null);
  }}
  const byFrameAbs = {{}};
  sequences.forEach(seq => seq.frames.forEach(f => addToBucket(byFrameAbs, f.idx, f)));
  const N_BINS = 100;
  const byFrameRel = {{}};
  sequences.forEach(seq => {{
    const nF = seq.frames.length;
    seq.frames.forEach((f, pos) => addToBucket(byFrameRel, Math.floor(pos / nF * N_BINS), f));
  }});
  return {{ byFrameAbs, byFrameRel }};
}});

// ---- stat helpers ----
function quantile(sorted, q) {{
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}}
function aggStat(arr, mode) {{
  const v = arr.filter(x => x != null).sort((a, b) => a - b);
  if (!v.length) return null;
  return mode === 'mean' ? v.reduce((a, b) => a + b, 0) / v.length : quantile(v, parseFloat(mode));
}}
function iqr(arr) {{
  const v = arr.filter(x => x != null).sort((a, b) => a - b);
  return v.length ? {{ lo: quantile(v, 0.25), hi: quantile(v, 0.75) }} : {{ lo: null, hi: null }};
}}
function movAvg(arr, w) {{
  if (w <= 1) return arr;
  return arr.map((_, i) => {{
    let sum = 0, cnt = 0, half = Math.floor(w / 2);
    for (let j = Math.max(0, i-half); j < Math.min(arr.length, i-half+w); j++) {{
      if (arr[j] != null) {{ sum += arr[j]; cnt++; }}
    }}
    return cnt > 0 ? sum / cnt : null;
  }});
}}

// ---- trace builders ----
const plotCfg = {{ responsive: true, displayModeBar: false }};

function mkCoverage(xs, nArr, divArr) {{
  // divArr: per-point divisor array; if omitted uses nSeq for all points.
  return {{ x: xs, y: nArr.map((n, i) => {{ const d = divArr ? divArr[i] : nSeq; return d > 0 ? n / d : 0; }}),
    mode: 'lines', type: 'scatter', fill: 'tozeroy',
    fillcolor: 'rgba(180,180,180,0.18)', line: {{ color: 'rgba(160,160,160,0.35)', width: 1 }},
    hoverinfo: 'skip', showlegend: false, yaxis: 'y2' }};
}}
function mkLine(xs, ys, color, name, hoverPair) {{
  const text = hoverPair
    ? xs.map((x, i) => `${{hoverPair[0]}}: ${{x}}<br>${{hoverPair[1]}}: ${{ys[i] != null ? ys[i].toFixed(3) : 'n/a'}}`)
    : null;
  return {{ x: xs, y: ys, mode: 'lines', type: 'scatter',
    line: {{ color, width: 2 }}, name, connectgaps: true,
    text, hoverinfo: text ? 'text' : 'skip' }};
}}
function mkBand(xs, lo, hi, color) {{
  const xAll = [...xs, ...xs.slice().reverse()];
  const yAll = [...hi, ...lo.slice().reverse()];
  return {{ x: xAll, y: yAll, mode: 'lines', type: 'scatter', fill: 'toself',
    fillcolor: color.replace(')', ', 0.15)').replace('rgb', 'rgba'),
    line: {{ width: 0 }}, hoverinfo: 'skip', showlegend: false }};
}}
function mkLayout(ytitle, yrange, xIsRel, y2title) {{
  const base = {{
    margin: {{ l: 52, r: 52, t: 10, b: 48 }},
    xaxis: {{ title: xIsRel ? 'Relative position' : 'Frame index', gridcolor: '#eee', zeroline: false }},
    yaxis: {{ title: ytitle, gridcolor: '#eee', zeroline: false }},
    yaxis2: {{ overlaying: 'y', side: 'right', range: [0,1], tickformat: '.0%',
               title: y2title || '% seqs with frame', showgrid: false,
               color: '#bbb', tickfont: {{ color: '#bbb' }}, titlefont: {{ color: '#bbb' }} }},
    paper_bgcolor: 'white', plot_bgcolor: '#fafafa', showlegend: false,
  }};
  if (yrange) base.yaxis.range = yrange;
  return base;
}}

let initialized = false;
function renderPlot(id, traces, layout) {{
  initialized ? Plotly.react(id, traces, layout, plotCfg)
              : Plotly.newPlot(id, traces, layout, plotCfg);
}}

// ---- windowed fraction ----
function buildWindowedFrac(thrIdx, minCount, minRatio, W, wndType, minSeqs, xIsRel) {{
  const N_BINS = 100, minRatioFrac = minRatio / 100;
  const byKey = {{}};
  sequences.forEach(seq => {{
    const nF = seq.frames.length;
    seq.frames.forEach((f, pos) => {{
      const hasCnt  = f.cnt  != null;
      const hasFrac = f.frac != null;
      if (!hasCnt && !hasFrac) return;
      const x = xIsRel ? Math.floor(pos / nF * N_BINS) : f.idx;
      let lo, hi;
      if (wndType === 'last') {{ lo = Math.max(0, pos-W+1); hi = pos; }}
      else {{ const half = Math.floor(W/2); lo = Math.max(0, pos-half); hi = Math.min(nF-1, pos+(W-1-half)); }}
      let okCnt = false, okRat = false;
      for (let j = lo; j <= hi; j++) {{
        const fj = seq.frames[j];
        if (!okCnt && fj.cnt) {{ const c = fj.cnt[thrIdx]; if (c != null && c >= minCount) okCnt = true; }}
        if (!okRat && fj.frac) {{ const r = fj.frac[thrIdx]; if (r != null && r >= minRatioFrac) okRat = true; }}
      }}
      if (!byKey[x]) byKey[x] = {{ okCnt: [], okRat: [], nCnt: 0, nRat: 0 }};
      if (hasCnt)  {{ byKey[x].okCnt.push(okCnt);  byKey[x].nCnt++; }}
      if (hasFrac) {{ byKey[x].okRat.push(okRat);  byKey[x].nRat++; }}
    }});
  }});
  const pre = precomputed[thrIdx];
  const byFrame = xIsRel ? pre.byFrameRel : pre.byFrameAbs;
  const xs = [], yC = [], yR = [], nGTs = [], nContainW = [];
  Object.keys(byFrame).map(Number).sort((a,b)=>a-b).forEach(idx => {{
    const d = byKey[idx];
    const nGT = d ? Math.max(d.nCnt, d.nRat) : 0;
    const preD = byFrame[idx];
    const nC = preD ? preD.n.length : 0;   // sequences containing this frame
    if (nC < minSeqs) return;
    xs.push(xIsRel ? idx/N_BINS : idx);
    nGTs.push(nGT);
    nContainW.push(nC);
    yC.push(d && d.okCnt.length ? d.okCnt.filter(v=>v).length/d.okCnt.length : null);
    yR.push(d && d.okRat.length ? d.okRat.filter(v=>v).length/d.okRat.length : null);
  }});
  return {{ xs, yC, yR, nGTs, nContainW }};
}}

// ---- survival ----
function buildSurvivalData(thrIdx, minCount, minRatio, xIsRel) {{
  const N_BINS = 100, minRatioFrac = minRatio / 100;
  const ffCnt = [], ffRat = []; let nWithGT = 0;
  sequences.forEach(seq => {{
    const nF = seq.frames.length; let fcnt = Infinity, frat = Infinity, hasGT = false;
    seq.frames.forEach((f, pos) => {{
      const x = xIsRel ? pos/nF : f.idx;
      if (f.cnt != null) {{ hasGT = true; const c = f.cnt[thrIdx]; if (c != null && c < minCount && fcnt===Infinity) fcnt=x; }}
      if (f.frac != null) {{ hasGT = true; const r = f.frac[thrIdx]; if (r != null && r < minRatioFrac && frat===Infinity) frat=x; }}
    }});
    if (hasGT) {{ ffCnt.push(fcnt); ffRat.push(frat); nWithGT++; }}
  }});
  if (!nWithGT) return null;
  const pre = precomputed[thrIdx];
  const byFrame = xIsRel ? pre.byFrameRel : pre.byFrameAbs;
  const sortedIdxs = Object.keys(byFrame).map(Number).sort((a,b)=>a-b);
  const xs = sortedIdxs.map(i => xIsRel ? i/N_BINS : i);
  const survCnt = xs.map(x => ffCnt.filter(f=>f>x).length/nWithGT);
  const survRat = xs.map(x => ffRat.filter(f=>f>x).length/nWithGT);
  const fracAboveCnt = xs.map((x,i) => {{ const d=byFrame[sortedIdxs[i]]; const v=d.cnts.filter(x=>x!=null); return v.length ? v.filter(c=>c>=minCount).length/v.length : null; }});
  const fracAboveRat = xs.map((x,i) => {{ const d=byFrame[sortedIdxs[i]]; const v=d.fracs.filter(x=>x!=null); return v.length ? v.filter(r=>r>=minRatioFrac).length/v.length : null; }});
  return {{ xs, survCnt, survRat, fracAboveCnt, fracAboveRat, nWithGT }};
}}

// ---- heatmap ----
function getSeqVal(f, thrIdx, metric) {{
  if (metric==='frac') return f.frac ? f.frac[thrIdx] : null;
  if (metric==='cnt')  return f.cnt  ? f.cnt[thrIdx]  : null;
  if (metric==='n')    return f.n != null ? f.n : null;
  if (metric==='objfrac') return (f.n!=null && f.nt!=null && f.n>0) ? f.nt/f.n : null;
  return null;
}}
function buildHeatmapData(thrIdx, metric, sortMode, xIsRel) {{
  const N_BINS = 100;
  const seqData = sequences.map(seq => {{
    const nF = seq.frames.length; const mapping = {{}};
    let firstFail = Infinity;
    seq.frames.forEach((f, pos) => {{
      const x = xIsRel ? Math.floor(pos/nF*N_BINS) : f.idx;
      const val = getSeqVal(f, thrIdx, metric);
      mapping[x] = val;
      if ((metric==='frac'||metric==='cnt') && val !== null && val <= 0 && firstFail===Infinity)
        firstFail = xIsRel ? pos/nF : f.idx;
    }});
    const vals = Object.values(mapping).filter(v=>v!=null);
    return {{ name: seq.name, mapping, meanVal: vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null, firstFail }};
  }});
  if (sortMode==='mean') seqData.sort((a,b)=>(b.meanVal??-Infinity)-(a.meanVal??-Infinity));
  else if (sortMode==='firstfail') seqData.sort((a,b)=>{{ if(a.firstFail===Infinity&&b.firstFail===Infinity)return 0; if(a.firstFail===Infinity)return -1; if(b.firstFail===Infinity)return 1; return b.firstFail-a.firstFail; }});
  else seqData.sort((a,b)=>a.name.localeCompare(b.name));
  const xSet = new Set(); seqData.forEach(sd=>Object.keys(sd.mapping).forEach(k=>xSet.add(+k)));
  const allX = [...xSet].sort((a,b)=>a-b);
  const xLabels = xIsRel ? allX.map(x=>(x/N_BINS).toFixed(2)) : allX;
  return {{ x: xLabels, y: seqData.map(s=>s.name), z: seqData.map(sd=>allX.map(x=>{{ const v=sd.mapping[x]; return (v!=null&&v!==undefined)?v:null; }})), metric }};
}}

// ---- main render ----
function updatePlots() {{
  const thrIdx   = parseInt(S.thrIdx);
  const minCount = parseInt(S.minCount);
  const minRatio = parseInt(S.minRatio);
  const smoothW  = parseInt(S.smoothW);
  const minSeqs  = parseInt(S.minSeqs);
  const xIsRel   = S.xMode === 'rel';
  const statMode = S.statMode;
  const showBand = S.showBand;
  const W        = parseInt(S.wndW);
  const wndType  = S.wndType;
  const hmMetric = S.hmMetric;
  const hmSort   = S.hmSort;
  const thrPx    = thresholds[thrIdx];
  const N_BINS   = 100;
  const statLabel = statMode === 'mean' ? 'mean' : statMode==='0.5' ? 'median' : 'P'+Math.round(parseFloat(statMode)*100);

  const covMode = S.covMode;
  const pre = precomputed[thrIdx];
  const byFrame = xIsRel ? pre.byFrameRel : pre.byFrameAbs;
  const sortedIdxs = Object.keys(byFrame).map(Number).sort((a,b)=>a-b);

  const xs=[],yCnt=[],yRat=[],yQcnt=[],yQrat=[],yN=[],yOF=[];
  const loQcnt=[],hiQcnt=[],loQrat=[],hiQrat=[],loN=[],hiN=[],loOF=[],hiOF=[];
  const nContain=[],nDataAll=[],nDataGT=[];

  sortedIdxs.forEach(idx => {{
    const d = byFrame[idx];
    const nC  = d.n.length;                          // sequences containing this frame
    const nAll = d.n.filter(v=>v!=null).length;      // sequences with match data
    const nGT  = d.cnts.filter(v=>v!=null).length;   // sequences with GT+target data
    if (nC < minSeqs) return;
    xs.push(xIsRel ? idx/N_BINS : idx);
    nContain.push(nC); nDataAll.push(nAll); nDataGT.push(nGT);
    yCnt.push(nGT>0 ? d.cnts.filter(v=>v!=null&&v>=minCount).length/nGT : null);
    yRat.push(nGT>0 ? d.fracs.filter(v=>v!=null&&v>=minRatio/100).length/nGT : null);
    yQcnt.push(aggStat(d.cnts,statMode)); const iC=iqr(d.cnts); loQcnt.push(iC.lo); hiQcnt.push(iC.hi);
    yQrat.push(aggStat(d.fracs,statMode)); const iR=iqr(d.fracs); loQrat.push(iR.lo); hiQrat.push(iR.hi);
    yN.push(aggStat(d.n,statMode));    const iN=iqr(d.n);    loN.push(iN.lo);  hiN.push(iN.hi);
    yOF.push(aggStat(d.objfrac,statMode)); const iO=iqr(d.objfrac); loOF.push(iO.lo); hiOF.push(iO.hi);
  }});

  const sm = arr => movAvg(arr, smoothW);
  const [yCntS,yRatS,yQcntS,yQratS,yNS,yOFS] = [yCnt,yRat,yQcnt,yQrat,yN,yOF].map(sm);
  const [loQcntS,hiQcntS,loQratS,hiQratS,loNS,hiNS,loOFS,hiOFS] =
    [loQcnt,hiQcnt,loQrat,hiQrat,loN,hiN,loOF,hiOF].map(sm);

  const y2lbl = covMode==='contain' ? '% seqs with frame' : '% with data';
  const covAll = () => covMode==='contain'
    ? mkCoverage(xs, nContain)
    : mkCoverage(xs, nDataAll, nContain);
  const covGT  = () => covMode==='contain'
    ? mkCoverage(xs, nContain)
    : mkCoverage(xs, nDataGT, nContain);

  // Row 1 — per-frame fraction above threshold
  document.getElementById('title-count').textContent =
    `Fraction of seqs with ≥${{minCount}} target-kpt inliers @ ${{thrPx}} px (kptA on target in init frame)`;
  renderPlot('plot-count', [covGT(), mkLine(xs,yCntS,'#2196F3',null,['frame','frac'])], mkLayout('Fraction of seqs',[0,1],xIsRel,y2lbl));

  document.getElementById('title-ratio').textContent =
    `Fraction of seqs with ≥${{minRatio}}% target-kpt inlier ratio @ ${{thrPx}} px (kptA on target in init frame)`;
  renderPlot('plot-ratio', [covGT(), mkLine(xs,yRatS,'#4CAF50',null,['frame','frac'])], mkLayout('Fraction of seqs',[0,1],xIsRel,y2lbl));

  // Row 2 — windowed fraction
  const wndLabel = wndType==='last' ? `last ${{W}} frames` : `±${{Math.floor(W/2)}} frames`;
  const wndData  = buildWindowedFrac(thrIdx, minCount, minRatio, W, wndType, minSeqs, xIsRel);
  {{
    const {{xs:wx,yC,yR,nGTs,nContainW}} = wndData;
    const wCov = () => covMode==='contain'
      ? mkCoverage(wx, nContainW)
      : mkCoverage(wx, nGTs, nContainW);
    const L = mkLayout('Fraction of seqs',[0,1],xIsRel,y2lbl);
    document.getElementById('title-wnd-cnt').textContent =
      `Windowed: ≥${{minCount}} target-kpt inliers in ${{wndLabel}} @ ${{thrPx}} px`;
    renderPlot('plot-wnd-cnt', [wCov(), mkLine(wx,sm(yC),'#1565C0',null,['frame','frac'])], L);
    document.getElementById('title-wnd-rat').textContent =
      `Windowed: ≥${{minRatio}}% target-kpt inlier ratio in ${{wndLabel}} @ ${{thrPx}} px`;
    renderPlot('plot-wnd-rat', [wCov(), mkLine(wx,sm(yR),'#2E7D32',null,['frame','frac'])], L);
  }}

  // Row 3 — quantile of inlier metrics
  document.getElementById('title-qcnt').textContent =
    `${{statLabel}} target-kpt inlier count @ ${{thrPx}} px (kptA on target in init frame)`;
  {{
    const tr=[covGT()]; if(showBand) tr.push(mkBand(xs,loQcntS,hiQcntS,'rgb(33,150,243)'));
    tr.push(mkLine(xs,yQcntS,'#2196F3',statLabel,['frame','count']));
    renderPlot('plot-qcnt', tr, mkLayout('Inlier count',null,xIsRel,y2lbl));
  }}
  document.getElementById('title-qrat').textContent =
    `${{statLabel}} target-kpt inlier ratio @ ${{thrPx}} px (kptA on target in init frame)`;
  {{
    const tr=[covGT()]; if(showBand) tr.push(mkBand(xs,loQratS,hiQratS,'rgb(76,175,80)'));
    tr.push(mkLine(xs,yQratS,'#4CAF50',statLabel,['frame','ratio']));
    renderPlot('plot-qrat', tr, mkLayout('Inlier ratio',[0,1],xIsRel,y2lbl));
  }}

  // Row 4 — overview match stats
  document.getElementById('title-matches').textContent =
    `${{statLabel}} total LoMa matches per frame (all seqs)`;
  {{
    const tr=[covAll()]; if(showBand) tr.push(mkBand(xs,loNS,hiNS,'rgb(255,152,0)'));
    tr.push(mkLine(xs,yNS,'#FF9800',statLabel,['frame','n_matches']));
    renderPlot('plot-matches', tr, mkLayout('n_matches',null,xIsRel,y2lbl));
  }}
  document.getElementById('title-objfrac').textContent =
    `${{statLabel}} fraction of matches sourced from object (kptA on target in init frame)`;
  {{
    const tr=[covAll()]; if(showBand) tr.push(mkBand(xs,loOFS,hiOFS,'rgb(156,39,176)'));
    tr.push(mkLine(xs,yOFS,'#9C27B0',statLabel,['frame','obj_frac']));
    renderPlot('plot-objfrac', tr, mkLayout('kptA-on-obj fraction',[0,1],xIsRel,y2lbl));
  }}

  // Row 5 — survival
  const survData = buildSurvivalData(thrIdx, minCount, minRatio, xIsRel);
  if (survData) {{
    const {{xs:sx,survCnt,survRat,fracAboveCnt,fracAboveRat,nWithGT}} = survData;
    const sxS = xIsRel ? 'Relative position' : 'Frame index';
    const sLay = {{ margin:{{l:52,r:16,t:10,b:48}}, xaxis:{{title:sxS,gridcolor:'#eee',zeroline:false}},
      yaxis:{{title:'Fraction of seqs',range:[0,1],gridcolor:'#eee',zeroline:false}},
      paper_bgcolor:'white', plot_bgcolor:'#fafafa', showlegend:true,
      legend:{{x:1,xanchor:'right',y:1,bgcolor:'rgba(255,255,255,0.8)',font:{{size:11}}}} }};
    const ss = arr => movAvg(arr, smoothW);
    document.getElementById('title-surv-cnt').textContent =
      `Survival: never below ${{minCount}} target-kpt inliers @ ${{thrPx}} px (${{nWithGT}} seqs with GT)`;
    renderPlot('plot-surv-cnt', [
      {{x:sx,y:ss(survCnt),    mode:'lines',type:'scatter',name:'survival (never failed)',    line:{{color:'#2196F3',width:2}},hoverinfo:'skip'}},
      {{x:sx,y:ss(fracAboveCnt),mode:'lines',type:'scatter',name:'currently above N (snapshot)',line:{{color:'#2196F3',width:1.5,dash:'dot'}},hoverinfo:'skip'}},
    ], sLay);
    document.getElementById('title-surv-rat').textContent =
      `Survival: never below ${{minRatio}}% target-kpt inlier ratio @ ${{thrPx}} px (${{nWithGT}} seqs with GT)`;
    renderPlot('plot-surv-rat', [
      {{x:sx,y:ss(survRat),    mode:'lines',type:'scatter',name:'survival (never failed)',    line:{{color:'#4CAF50',width:2}},hoverinfo:'skip'}},
      {{x:sx,y:ss(fracAboveRat),mode:'lines',type:'scatter',name:'currently above N (snapshot)',line:{{color:'#4CAF50',width:1.5,dash:'dot'}},hoverinfo:'skip'}},
    ], sLay);
  }}

  // Heatmap
  const hmData = buildHeatmapData(thrIdx, hmMetric, hmSort, xIsRel);
  const metricLabels = {{ frac:`inlier ratio @ ${{thrPx}} px (target kpts)`, cnt:`inlier count @ ${{thrPx}} px (target kpts)`, n:'total n_matches', objfrac:'kptA obj fraction' }};
  const isRatio = (hmMetric==='frac'||hmMetric==='objfrac');
  document.getElementById('title-heatmap').textContent =
    `Per-sequence heatmap — ${{metricLabels[hmMetric]}}  (sorted: ${{hmSort}})`;
  const hmH = Math.max(300, Math.min(nSeq*14+80, 1200));
  document.getElementById('plot-heatmap').style.height = hmH + 'px';
  const hmTrace = {{ type:'heatmap', x:hmData.x, y:hmData.y, z:hmData.z,
    colorscale: isRatio?'RdYlGn':'Viridis', zmid: isRatio?0.5:undefined,
    zmin: isRatio?0:undefined, zmax: isRatio?1:undefined,
    showscale:true, hoverongaps:false,
    hovertemplate:'%{{y}}<br>frame %{{x}}<br>value: %{{z:.3f}}<extra></extra>' }};
  const hmLay = {{ margin:{{l:150,r:60,t:10,b:48}},
    xaxis:{{title:xIsRel?'Relative position':'Frame index',gridcolor:'#eee',zeroline:false}},
    yaxis:{{tickfont:{{size:9}},automargin:true}},
    paper_bgcolor:'white', plot_bgcolor:'#fafafa', height:hmH }};
  renderPlot('plot-heatmap', [hmTrace], hmLay);

  initialized = true;
}}

wireControls();
updatePlots();
</script>
</body>
</html>
"""
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate interactive HTML report for LoMa analysis results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "analysis_dir",
        type=Path,
        help="Directory containing *_analysis.json files (e.g. LoMa/LoMaB/POT-210/).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output HTML file path.  Defaults to <analysis_dir>/report.html.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Report title.  Defaults to the analysis dir name.",
    )
    p.add_argument(
        "--cdn",
        action="store_true",
        help="Load Plotly from CDN instead of inlining it (requires internet).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    analysis_dir: Path = args.analysis_dir.resolve()
    if not analysis_dir.is_dir():
        print(f"ERROR: {analysis_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output: Path = args.output or analysis_dir / "report.html"
    title: str = args.title or f"LoMa analysis — {analysis_dir.name}"

    print(f"Loading analysis data from {analysis_dir} …")
    data = load_analysis_dir(analysis_dir)

    n_seq = len(data["sequences"])
    n_frames = sum(len(s["frames"]) for s in data["sequences"])
    print(f"  {n_seq} sequences, {n_frames} total frames, thresholds: {data['thresholds']} px")

    if n_seq == 0:
        print("ERROR: no sequences loaded.", file=sys.stderr)
        sys.exit(1)

    summary_text = _format_summary_from_data(data)
    print(summary_text)

    html = _build_html(data, title, summary_text=summary_text, cdn=args.cdn)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Report written → {output}")


if __name__ == "__main__":
    main()
