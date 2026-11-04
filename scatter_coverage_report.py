#!/usr/bin/env python3
"""Interactive scatter report: hull coverage vs alignment error (or inlier count).

Two views:
  1. Error vs Coverage  — X=hull/peeled coverage, Y=alignment error, color=inlier count
  2. Inliers vs Coverage — X=hull/peeled coverage, Y=inlier count,  color=alignment error

Requires ``*_spatial_stats.json`` alongside kp tracker results.  Pass
``--auto-spatial-stats`` to compute them automatically from the NPZ files.

Usage::

    python scatter_coverage_report.py \\
        --kp-tracker loma_r_2000obj_infl10 \\
        --output coverage_scatter.html \\
        --auto-spatial-stats
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from einops import rearrange

from flatsam2.datasets import DATASETS, get_sequences, pot_load_gt, pt_load_gt, pti_load_gt
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.report_utils import get_plotly_script_tag


# ---------------------------------------------------------------------------
# Auto-compute spatial stats
# ---------------------------------------------------------------------------


def _ensure_spatial_stats(results_dir: Path, dataset: str, masks_dir: Optional[Path] = None) -> None:
    """Run compute_spatial_stats for sequences with missing or outdated *_spatial_stats.json.

    "Outdated" means the file exists but lacks the ``peeled_hull_coverage`` key
    (written by an older version of compute_spatial_stats.py).  Those files are
    recomputed with ``--force``.

    Args:
        results_dir: Dataset-level results dir (contains *.npz files).
        dataset: Dataset CLI key.
        masks_dir: Optional init-mask base dir.
    """
    npz_files = sorted(results_dir.glob("*.npz"))
    if not npz_files:
        print(f"  WARNING: no .npz files found in {results_dir}")
        print(f"    hull_coverage requires keypoint NPZ files saved by the loma tracker.")
        print(f"    Spatial stats cannot be computed without them.")
        return

    n_missing = n_outdated = 0
    for p in npz_files:
        spatial_path = results_dir / (p.stem + "_spatial_stats.json")
        if not spatial_path.exists():
            n_missing += 1
            continue
        try:
            data = json.loads(spatial_path.read_text())
            frames = data.get("frames", [])
            # outdated if none of the first 10 frames have the peeled key at all
            if frames and not any("peeled_hull_coverage" in f for f in frames[:10]):
                n_outdated += 1
        except Exception:
            n_missing += 1

    if n_missing == 0 and n_outdated == 0:
        return

    force = n_outdated > 0
    print(f"  {n_missing} missing + {n_outdated} outdated spatial stats "
          f"— computing (force={force})...")
    try:
        repo_root = Path(__file__).parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from compute_spatial_stats import main as css_main  # type: ignore[import]

        argv = [str(results_dir), "--dataset", dataset]
        if masks_dir is not None:
            argv += ["--masks-dir", str(masks_dir)]
        if force:
            argv.append("--force")
        print(f"  Running: compute_spatial_stats.py {' '.join(argv)}")
        rc = css_main(argv)
        if rc != 0:
            print("  WARNING: compute_spatial_stats returned non-zero exit code")
    except Exception as exc:
        print(f"  WARNING: could not auto-compute spatial stats: {exc}")
        print(traceback.format_exc())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_predictions(results_dir: Path, seq_name: str) -> Optional[np.ndarray]:
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        return None
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(pred_flat, "f (c xy) -> f xy c", c=4, xy=2)


def _load_frame_stats(results_dir: Path, seq_name: str) -> dict[int, dict]:
    """Merge per-frame data from *_stats.json and *_spatial_stats.json."""
    frame_map: dict[int, dict] = {}

    for suffix in ("_stats.json", "_spatial_stats.json"):
        path = results_dir / f"{seq_name}{suffix}"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            for r in data.get("frames", []):
                fidx = int(r["frame_idx"])
                frame_map.setdefault(fidx, {}).update(r)
        except Exception:
            pass

    return frame_map


def _make_gt_loader(dataset: str, base_dir: Path, initfix_dir: Optional[Path], woft_compat: bool):
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return lambda sn: pot_load_gt(base_dir, sn, woft_compat=woft_compat)
    if loader == "pt":
        return lambda sn: pt_load_gt(base_dir, sn)
    if initfix_dir is None:
        raise ValueError(f"initfix_dir required for dataset {dataset}")
    return lambda sn: pti_load_gt(base_dir, sn, initfix_dir)


def collect_dataset_frames(
    dataset: str,
    base_dir: Path,
    sequences: list[str],
    initfix_dir: Optional[Path],
    results_dir: Path,
    ds_id: int,
    woft_compat: bool = True,
) -> tuple[list[str], list[list]]:
    """Collect per-frame scatter data for one dataset.

    Returns:
        (seq_names, rows) where each row is
        [hull, peeled, inliers, err, ds_id, seq_idx, fidx].
        hull/peeled/err may be None; inliers may be None.
    """
    load_gt = _make_gt_loader(dataset, base_dir, initfix_dir, woft_compat)
    seq_names: list[str] = []
    rows: list[list] = []
    n_ok = n_fail = 0

    for seq_name in sequences:
        try:
            gt_corners, valid = load_gt(seq_name)
            pred = _load_predictions(results_dir, seq_name)
            if pred is None:
                n_fail += 1
                continue

            min_len = min(len(gt_corners), len(pred))
            gt_corners = gt_corners[:min_len]
            valid = valid[:min_len]
            pred = pred[:min_len]

            eval_mask = valid.copy()
            if len(eval_mask) > 0:
                eval_mask[0] = False
            eval_idx = np.where(eval_mask)[0]
            if len(eval_idx) == 0:
                n_ok += 1
                continue

            errors = np.full(min_len, np.nan)
            gt_eval = gt_corners[eval_idx]
            pred_eval = pred[eval_idx]
            finite = (
                np.all(np.isfinite(gt_eval), axis=(1, 2))
                & np.all(np.isfinite(pred_eval), axis=(1, 2))
            )
            if finite.any():
                errs = alignment_error(gt_eval[finite], pred_eval[finite])
                errors[eval_idx[finite]] = errs

            frame_stats = _load_frame_stats(results_dir, seq_name)
            seq_idx = len(seq_names)
            seq_names.append(seq_name)

            for fidx in eval_idx:
                stats = frame_stats.get(int(fidx), {})
                hull = stats.get("hull_coverage")
                peeled = stats.get("peeled_hull_coverage")
                n_inliers = stats.get("n_inliers_used") or stats.get("inliers_total")
                err_val = float(errors[fidx])
                rows.append([
                    round(hull, 4) if hull is not None else None,
                    round(peeled, 4) if peeled is not None else None,
                    int(n_inliers) if n_inliers is not None else None,
                    round(err_val, 2) if np.isfinite(err_val) else None,
                    ds_id,
                    seq_idx,
                    int(fidx),
                ])
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"    ERROR {seq_name}: {exc}")
            if n_fail <= 3:
                print(traceback.format_exc())

    print(f"    {n_ok} ok, {n_fail} failed → {len(rows)} evaluated frames")
    return seq_names, rows


def resolve_results_dir(results_roots: list[Path], tracker_spec: str, dataset: str) -> Optional[Path]:
    name = Path(tracker_spec).stem
    label = DATASETS[dataset].label
    for root in results_roots:
        d = root / name / label
        if d.exists():
            return d
    searched = ", ".join(str(r / name / label) for r in results_roots)
    print(f"  WARNING: results dir not found (searched: {searched})")
    return None


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _build_payload(
    all_rows: list[list],
    all_seq_names: list[str],
    ds_labels: list[str],
    rng_seed: int = 42,
) -> dict:
    n = len(all_rows)
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n).tolist()
    return {
        "n": n,
        "hull":    [r[0] for r in all_rows],
        "peeled":  [r[1] for r in all_rows],
        "inliers": [r[2] for r in all_rows],
        "err":     [r[3] for r in all_rows],
        "ds":      [r[4] for r in all_rows],
        "seq_idx": [r[5] for r in all_rows],
        "fidx":    [r[6] for r in all_rows],
        "perm": perm,
        "seq_names": all_seq_names,
        "ds_labels": ds_labels,
    }


def write_report(output_path: Path, payload: dict, tracker_name: str, cdn: bool = False) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"))
    plotly_tag = get_plotly_script_tag(cdn)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coverage scatter — {tracker_name}</title>
  {plotly_tag}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 14px 18px 24px;
      font-family: system-ui, sans-serif; font-size: 13px;
      background: #eef2f7; color: #1f2937;
    }}
    h1 {{ font-size: 20px; margin: 0 0 10px; }}
    .controls {{
      background: #fff; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
      padding: 12px 16px; margin-bottom: 10px;
      display: flex; flex-wrap: wrap; gap: 16px 22px; align-items: flex-end;
    }}
    .ctrl-group {{ display: flex; flex-direction: column; gap: 4px; }}
    .ctrl-label {{
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: .06em; color: #6b7280;
    }}
    .btn-row {{ display: flex; gap: 4px; }}
    .btn {{
      padding: 4px 10px; border: 1px solid #d1d5db; border-radius: 5px;
      background: #f9fafb; cursor: pointer; font-size: 12px; user-select: none;
    }}
    .btn.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
    .stats-bar {{
      font-size: 12px; color: #374151; background: #fff;
      border-radius: 8px; padding: 7px 14px; margin-bottom: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }}
    #plot-div {{ min-height: 680px; background: #fff; border-radius: 10px;
                 box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
    label {{ font-size: 12px; color: #374151; }}
    input[type=range] {{ width: 110px; vertical-align: middle; cursor: pointer; }}
    input[type=number] {{
      width: 58px; padding: 3px 5px; border: 1px solid #d1d5db;
      border-radius: 4px; font-size: 12px;
    }}
    .ds-check {{ display: flex; gap: 10px; align-items: center; }}
    .dyn-val {{ font-weight: 700; color: #2563eb; min-width: 2.5em; display: inline-block; }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <h1>Coverage scatter — {tracker_name}</h1>

  <div class="controls">

    <div class="ctrl-group">
      <div class="ctrl-label">X metric</div>
      <div class="btn-row">
        <div class="btn active" id="btn-hull"   onclick="setMetric('hull')">hull_coverage</div>
        <div class="btn"        id="btn-peeled" onclick="setMetric('peeled')">peeled_hull_coverage</div>
      </div>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">View</div>
      <div class="btn-row">
        <div class="btn active" id="btn-view-error"  onclick="setView('error')">Error vs Coverage</div>
        <div class="btn"        id="btn-view-inlier" onclick="setView('inlier')">Inliers vs Coverage</div>
      </div>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">Subset: <span class="dyn-val" id="subset-val">100%</span></div>
      <input type="range" id="subset-slider" min="1" max="100" step="1" value="100"
             oninput="document.getElementById('subset-val').textContent=this.value+'%';
                      rebuildSubset(); redraw();">
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">Error cap (px)</div>
      <input type="number" id="err-cap" value="30" min="5" max="500" step="5" onchange="redraw()">
    </div>

    <div class="ctrl-group" id="grp-inlier-thr">
      <div class="ctrl-label">Inlier threshold: <span class="dyn-val" id="inlier-thr-val">35</span></div>
      <input type="range" id="inlier-thr" min="0" max="300" step="1" value="35"
             oninput="document.getElementById('inlier-thr-val').textContent=this.value; redraw()">
    </div>

    <div class="ctrl-group hidden" id="grp-err-thr">
      <div class="ctrl-label">Error threshold (px)</div>
      <input type="number" id="err-thr" value="15" min="1" max="500" step="1" onchange="redraw()">
    </div>

    <div class="ctrl-group hidden" id="grp-inlier-cap">
      <div class="ctrl-label">Inlier cap</div>
      <input type="number" id="inlier-cap" value="100" min="5" max="2000" step="5" onchange="redraw()">
    </div>

    <div class="ctrl-group hidden" id="grp-color-mode">
      <div class="ctrl-label">Color mode</div>
      <div class="btn-row">
        <div class="btn active" id="btn-binary" onclick="setColorMode('binary')">Binary</div>
        <div class="btn"        id="btn-smooth" onclick="setColorMode('smooth')">Smooth</div>
      </div>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">Dataset</div>
      <div class="ds-check" id="ds-checks"></div>
    </div>

  </div>

  <div class="stats-bar" id="stats-bar">Loading…</div>
  <div id="plot-div"></div>

  <script>
    const D = {payload_json};

    let metric    = 'hull';
    let view      = 'error';
    let colorMode = 'binary';
    let subsetIndices = null;
    let dsChecks = {{}};

    // Build dataset checkboxes
    const dsChecksDiv = document.getElementById('ds-checks');
    D.ds_labels.forEach((lbl, i) => {{
      dsChecks[i] = true;
      const id = 'ds-chk-' + i;
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.id = id; cb.checked = true;
      cb.onchange = () => {{ dsChecks[i] = cb.checked; rebuildSubset(); redraw(); }};
      const la = document.createElement('label');
      la.htmlFor = id; la.textContent = ' ' + lbl;
      dsChecksDiv.appendChild(cb); dsChecksDiv.appendChild(la);
    }});

    function setMetric(m) {{
      metric = m;
      document.getElementById('btn-hull').classList.toggle('active', m === 'hull');
      document.getElementById('btn-peeled').classList.toggle('active', m === 'peeled');
      redraw();
    }}

    function setView(v) {{
      view = v;
      document.getElementById('btn-view-error').classList.toggle('active', v === 'error');
      document.getElementById('btn-view-inlier').classList.toggle('active', v === 'inlier');
      // show/hide view-specific controls
      document.getElementById('grp-inlier-thr').classList.toggle('hidden', v !== 'error');
      document.getElementById('grp-err-thr').classList.toggle('hidden', v !== 'inlier');
      document.getElementById('grp-inlier-cap').classList.toggle('hidden', v !== 'inlier');
      document.getElementById('grp-color-mode').classList.toggle('hidden', v !== 'inlier');
      redraw();
    }}

    function setColorMode(m) {{
      colorMode = m;
      document.getElementById('btn-binary').classList.toggle('active', m === 'binary');
      document.getElementById('btn-smooth').classList.toggle('active', m === 'smooth');
      redraw();
    }}

    function rebuildSubset() {{
      const pct = parseInt(document.getElementById('subset-slider').value);
      const n = Math.floor(D.n * pct / 100);
      const base = D.perm.slice(0, n);
      const allActive = Object.values(dsChecks).every(v => v);
      subsetIndices = allActive ? base : base.filter(i => dsChecks[D.ds[i]]);
    }}

    // ── diverging colorscale: green (0) → yellow (mid) → red (max) ──────────
    const CSCALE = [
      [0.0,  '#16a34a'],
      [0.25, '#86efac'],
      [0.5,  '#fef08a'],
      [0.75, '#fca5a5'],
      [1.0,  '#dc2626'],
    ];

    function redraw() {{
      if (!subsetIndices) rebuildSubset();

      const errCap    = parseFloat(document.getElementById('err-cap').value)    || 30;
      const inlierThr = parseInt(document.getElementById('inlier-thr').value)   || 35;
      const errThr    = parseFloat(document.getElementById('err-thr').value)    || 15;
      const inlierCap = parseFloat(document.getElementById('inlier-cap').value) || 100;
      const xArr = metric === 'hull' ? D.hull : D.peeled;

      // Split into two groups for binary coloring; also collect for smooth.
      const gx = [], gy = [], gtext = [], gcol = [];
      const bx = [], by = [], btext = [], bcol = [];
      let nShown = 0, nFinite = 0, nCorrect = 0;

      for (const i of subsetIndices) {{
        const x = xArr[i];
        if (x === null) continue;
        const rawErr  = D.err[i];
        const ni      = D.inliers[i];
        nShown++;
        if (rawErr !== null) {{
          nFinite++;
          if (rawErr <= errThr) nCorrect++;
        }}
        const seqLbl = D.ds_labels[D.ds[i]] + '/' + D.seq_names[D.seq_idx[i]];
        const errStr = rawErr !== null ? rawErr.toFixed(1) + 'px' : '∞';
        const niStr  = ni   !== null ? ni : '?';

        if (view === 'error') {{
          const y = rawErr === null ? errCap : Math.min(rawErr, errCap);
          const txt = seqLbl + '<br>frame ' + D.fidx[i] +
            '<br>err=' + errStr + '<br>inliers=' + niStr +
            '<br>' + (metric==='hull'?'hull':'peeled') + '=' + x.toFixed(4);
          if (ni !== null && ni > inlierThr) {{
            gx.push(x); gy.push(y); gtext.push(txt);
          }} else {{
            bx.push(x); by.push(y); btext.push(txt);
          }}
        }} else {{
          // inlier view
          const y = ni !== null ? Math.min(ni, inlierCap) : 0;
          const txt = seqLbl + '<br>frame ' + D.fidx[i] +
            '<br>inliers=' + niStr + '<br>err=' + errStr +
            '<br>' + (metric==='hull'?'hull':'peeled') + '=' + x.toFixed(4);
          const isCorrect = rawErr !== null && rawErr <= errThr;
          if (isCorrect) {{
            gx.push(x); gy.push(y); gtext.push(txt);
            gcol.push(rawErr);
          }} else {{
            bx.push(x); by.push(y); btext.push(txt);
            bcol.push(rawErr !== null ? rawErr : errCap);
          }}
        }}
      }}

      const pctK = nFinite > 0 ? (100 * nCorrect / nFinite).toFixed(1) : '—';
      document.getElementById('stats-bar').innerHTML =
        '<b>' + nShown.toLocaleString() + '</b> frames shown' +
        ' &nbsp;|&nbsp; P@' + (view==='error'?errThr:errThr).toFixed(0) +
        ' = <b>' + pctK + '%</b>' +
        ' &nbsp;|&nbsp; green <b>' + gx.length.toLocaleString() +
        '</b> / red <b>' + bx.length.toLocaleString() + '</b>';

      const xLabel = metric === 'hull' ? 'hull_coverage' : 'peeled_hull_coverage';
      const MAIN_X = [0, 0.82], MAIN_Y = [0, 0.82];
      const MARG_X = [0.84, 1.0], MARG_Y = [0.84, 1.0];
      const binSizeX = 0.02;

      let traces;
      if (view === 'error') {{
        const yLabel = 'alignment error (px)';
        const binSizeY = errCap / 40;
        traces = [
          // scatter
          {{ x:gx, y:gy, text:gtext, name:'inliers > '+inlierThr,
             type:'scattergl', mode:'markers', xaxis:'x', yaxis:'y',
             hovertemplate:'%{{text}}<extra></extra>',
             marker:{{ color:'#16a34a', size:3, opacity:0.35 }} }},
          {{ x:bx, y:by, text:btext, name:'inliers ≤ '+inlierThr,
             type:'scattergl', mode:'markers', xaxis:'x', yaxis:'y',
             hovertemplate:'%{{text}}<extra></extra>',
             marker:{{ color:'#dc2626', size:3, opacity:0.35 }} }},
          // top marginal (X distribution)
          {{ x:gx, type:'histogram', xaxis:'x', yaxis:'y2', showlegend:false, opacity:0.55,
             xbins:{{start:0, end:1, size:binSizeX}}, marker:{{color:'#16a34a'}},
             hovertemplate:'coverage=%{{x:.3f}}<br>count=%{{y}}<extra></extra>' }},
          {{ x:bx, type:'histogram', xaxis:'x', yaxis:'y2', showlegend:false, opacity:0.55,
             xbins:{{start:0, end:1, size:binSizeX}}, marker:{{color:'#dc2626'}},
             hovertemplate:'coverage=%{{x:.3f}}<br>count=%{{y}}<extra></extra>' }},
          // right marginal (Y distribution, horizontal)
          {{ y:gy, type:'histogram', xaxis:'x2', yaxis:'y', showlegend:false, opacity:0.55,
             ybins:{{start:0, end:errCap+binSizeY, size:binSizeY}}, marker:{{color:'#16a34a'}},
             hovertemplate:'err=%{{y:.1f}}<br>count=%{{x}}<extra></extra>' }},
          {{ y:by, type:'histogram', xaxis:'x2', yaxis:'y', showlegend:false, opacity:0.55,
             ybins:{{start:0, end:errCap+binSizeY, size:binSizeY}}, marker:{{color:'#dc2626'}},
             hovertemplate:'err=%{{y:.1f}}<br>count=%{{x}}<extra></extra>' }},
        ];
        const layout = {{
          barmode:'overlay',
          margin:{{ l:56, r:8, t:8, b:80 }},
          xaxis:{{  title:xLabel,  range:[0,1],      domain:MAIN_X }},
          yaxis:{{  title:yLabel,  range:[0,errCap], domain:MAIN_Y }},
          yaxis2:{{ domain:MARG_Y, showticklabels:false, fixedrange:true }},
          xaxis2:{{ domain:MARG_X, showticklabels:false, fixedrange:true }},
          legend:{{ orientation:'h', yanchor:'top', y:-0.16, xanchor:'center', x:0.41 }},
          shapes:[{{ type:'line', x0:0, x1:1, xref:'x',
                    y0:errThr, y1:errThr, yref:'y',
                    line:{{ color:'#f59e0b', width:1.5, dash:'dot' }} }}],
          annotations:[{{ x:0.82, y:errThr, xref:'x', yref:'y',
                          text:'P@'+errThr.toFixed(0), showarrow:false, xanchor:'right',
                          font:{{ size:10, color:'#f59e0b' }} }}],
        }};
        Plotly.react('plot-div', traces, layout, {{responsive:true, displayModeBar:true}});

      }} else {{
        // inlier view
        const yLabel = 'inlier count';
        const binSizeY = inlierCap / 40;
        if (colorMode === 'binary') {{
          traces = [
            {{ x:gx, y:gy, text:gtext, name:'err ≤ '+errThr+'px',
               type:'scattergl', mode:'markers', xaxis:'x', yaxis:'y',
               hovertemplate:'%{{text}}<extra></extra>',
               marker:{{ color:'#16a34a', size:3, opacity:0.35 }} }},
            {{ x:bx, y:by, text:btext, name:'err > '+errThr+'px',
               type:'scattergl', mode:'markers', xaxis:'x', yaxis:'y',
               hovertemplate:'%{{text}}<extra></extra>',
               marker:{{ color:'#dc2626', size:3, opacity:0.35 }} }},
            // top marginal
            {{ x:gx, type:'histogram', xaxis:'x', yaxis:'y2', showlegend:false, opacity:0.55,
               xbins:{{start:0, end:1, size:binSizeX}}, marker:{{color:'#16a34a'}},
               hovertemplate:'coverage=%{{x:.3f}}<br>count=%{{y}}<extra></extra>' }},
            {{ x:bx, type:'histogram', xaxis:'x', yaxis:'y2', showlegend:false, opacity:0.55,
               xbins:{{start:0, end:1, size:binSizeX}}, marker:{{color:'#dc2626'}},
               hovertemplate:'coverage=%{{x:.3f}}<br>count=%{{y}}<extra></extra>' }},
            // right marginal
            {{ y:gy, type:'histogram', xaxis:'x2', yaxis:'y', showlegend:false, opacity:0.55,
               ybins:{{start:0, end:inlierCap+binSizeY, size:binSizeY}}, marker:{{color:'#16a34a'}},
               hovertemplate:'inliers=%{{y}}<br>count=%{{x}}<extra></extra>' }},
            {{ y:by, type:'histogram', xaxis:'x2', yaxis:'y', showlegend:false, opacity:0.55,
               ybins:{{start:0, end:inlierCap+binSizeY, size:binSizeY}}, marker:{{color:'#dc2626'}},
               hovertemplate:'inliers=%{{y}}<br>count=%{{x}}<extra></extra>' }},
          ];
        }} else {{
          // smooth: one trace with colorscale, marginals still use binary split
          const allX = gx.concat(bx), allY = gy.concat(by);
          const allC = gcol.concat(bcol), allT = gtext.concat(btext);
          traces = [
            {{ x:allX, y:allY, text:allT,
               name:'alignment error',
               type:'scattergl', mode:'markers', xaxis:'x', yaxis:'y',
               hovertemplate:'%{{text}}<extra></extra>',
               marker:{{
                 color:allC, colorscale:CSCALE, cmin:0, cmax:errCap,
                 colorbar:{{ title:'error (px)', thickness:14, len:0.78,
                             x:0.79, xanchor:'right',
                             tickfont:{{size:10}} }},
                 size:3, opacity:0.45,
               }} }},
            // top marginal (still binary by errThr)
            {{ x:gx, type:'histogram', xaxis:'x', yaxis:'y2', showlegend:false, opacity:0.55,
               xbins:{{start:0, end:1, size:binSizeX}}, marker:{{color:'#16a34a'}},
               hovertemplate:'coverage=%{{x:.3f}}<br>count=%{{y}}<extra></extra>' }},
            {{ x:bx, type:'histogram', xaxis:'x', yaxis:'y2', showlegend:false, opacity:0.55,
               xbins:{{start:0, end:1, size:binSizeX}}, marker:{{color:'#dc2626'}},
               hovertemplate:'coverage=%{{x:.3f}}<br>count=%{{y}}<extra></extra>' }},
            // right marginal (still binary)
            {{ y:gy, type:'histogram', xaxis:'x2', yaxis:'y', showlegend:false, opacity:0.55,
               ybins:{{start:0, end:inlierCap+binSizeY, size:binSizeY}}, marker:{{color:'#16a34a'}},
               hovertemplate:'inliers=%{{y}}<br>count=%{{x}}<extra></extra>' }},
            {{ y:by, type:'histogram', xaxis:'x2', yaxis:'y', showlegend:false, opacity:0.55,
               ybins:{{start:0, end:inlierCap+binSizeY, size:binSizeY}}, marker:{{color:'#dc2626'}},
               hovertemplate:'inliers=%{{y}}<br>count=%{{x}}<extra></extra>' }},
          ];
        }}
        const layout = {{
          barmode:'overlay',
          margin:{{ l:56, r:8, t:8, b:80 }},
          xaxis:{{  title:xLabel,    range:[0,1],         domain:MAIN_X }},
          yaxis:{{  title:yLabel,    range:[0,inlierCap], domain:MAIN_Y }},
          yaxis2:{{ domain:MARG_Y,   showticklabels:false, fixedrange:true }},
          xaxis2:{{ domain:MARG_X,   showticklabels:false, fixedrange:true }},
          legend:{{ orientation:'h', yanchor:'top', y:-0.16, xanchor:'center', x:0.41 }},
        }};
        Plotly.react('plot-div', traces, layout, {{responsive:true, displayModeBar:true}});
      }}
    }}

    rebuildSubset();
    redraw();
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hull-coverage vs alignment-error/inlier-count scatter report."
    )
    parser.add_argument("--kp-tracker", required=True,
                        help="Keypoint tracker name or config path.")
    parser.add_argument("--dataset", choices=["pot", "pti", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("coverage_scatter.html"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path(paths.OUTPUT_DIRECTORY))
    parser.add_argument("--extra-results-dirs", type=Path, nargs="+",
                        default=[Path("/mnt/personal/serycjon/flatsam2/loma_v2/")],
                        metavar="DIR")
    parser.add_argument("--initfix-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="Init-mask base dir for spatial stats computation.")
    parser.add_argument("--auto-spatial-stats", action="store_true",
                        help="Auto-run compute_spatial_stats if *_spatial_stats.json missing.")
    parser.add_argument("--cdn", action="store_true")
    parser.add_argument("--orig-pot-gt", action="store_true")
    args = parser.parse_args(argv)

    args.output = args.output.expanduser()
    args.results_dir = args.results_dir.expanduser()
    args.extra_results_dirs = [d.expanduser() for d in args.extra_results_dirs]
    results_roots = [*args.extra_results_dirs, args.results_dir]

    tracker_name = Path(args.kp_tracker).stem
    woft_compat = not args.orig_pot_gt
    initfix_dir = args.initfix_dir or paths.PT_INITFIX_DIR

    raw = [("pti", None), ("pot", None)] if args.dataset == "both" else [(args.dataset, None)]

    all_seq_names: list[str] = []
    all_rows: list[list] = []
    ds_labels: list[str] = []

    for ds_key, base_dir_override in raw:
        ds_info = DATASETS[ds_key]
        base_dir = Path(base_dir_override or getattr(paths, ds_info.base_dir_attr)).expanduser()
        ds_label = ds_info.label
        split = "test" if ds_key == "pti" else None

        try:
            seqs = get_sequences(ds_key, base_dir, split=split)
        except Exception as exc:
            print(f"[{ds_label}] cannot list sequences: {exc} — skipping")
            continue

        print(f"[{ds_label}] {len(seqs)} sequences")
        results_dir = resolve_results_dir(results_roots, args.kp_tracker, ds_key)
        if results_dir is None:
            continue

        print(f"  Results dir: {results_dir}")

        if args.auto_spatial_stats:
            _ensure_spatial_stats(results_dir, ds_key, args.masks_dir)

        # Check coverage of spatial stats
        n_spatial = sum(
            1 for s in seqs
            if (results_dir / f"{s}_spatial_stats.json").exists()
        )
        if n_spatial == 0:
            if args.auto_spatial_stats:
                print(f"  WARNING: spatial stats still absent after auto-compute (see messages above)")
            else:
                print(f"  WARNING: no *_spatial_stats.json found — run with --auto-spatial-stats")
        else:
            print(f"  Spatial stats found for {n_spatial}/{len(seqs)} sequences")

        ds_id = len(ds_labels)
        ds_labels.append(ds_label)

        seq_names, rows = collect_dataset_frames(
            dataset=ds_key,
            base_dir=base_dir,
            sequences=seqs,
            initfix_dir=initfix_dir if ds_key == "pti" else None,
            results_dir=results_dir,
            ds_id=ds_id,
            woft_compat=woft_compat,
        )
        offset = len(all_seq_names)
        for row in rows:
            row[5] += offset
        all_seq_names.extend(seq_names)
        all_rows.extend(rows)

    if not all_rows:
        print("No frames collected; aborting.")
        return 1

    n_hull = sum(1 for r in all_rows if r[0] is not None)
    n_peeled = sum(1 for r in all_rows if r[1] is not None)
    print(f"\nTotal frames: {len(all_rows)}")
    print(f"  with hull_coverage:        {n_hull}")
    print(f"  with peeled_hull_coverage: {n_peeled}")
    if n_hull == 0:
        print("  NOTE: hull_coverage missing for all frames — run with --auto-spatial-stats")

    payload = _build_payload(all_rows, all_seq_names, ds_labels)
    write_report(args.output, payload, tracker_name, cdn=args.cdn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
