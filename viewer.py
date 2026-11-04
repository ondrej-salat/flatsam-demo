"""Interactive viewer for FlatSAM2 tracking results.

Serves a FastAPI web application for exploring heatmaps and raw frame data.
Click on a sequence in the heatmap to inspect its trajectory and navigate
frame-by-frame with homography overlays.

Usage:
    python viewer.py pti --config configs/tracker/flow_hough_symmetry_nn.py
    python viewer.py pti --config flow_hough_symmetry_nn --output-dir ./analysis --port 8000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import socket

import cv2
import numpy as np
from einops import rearrange
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from analyze_results import (
    CAND_COLORS, CAND_PATHS, THRESHOLD,
    _build_sequence_data, _load_debug, _load_predictions,
    _make_load_gt_fn, _p_at_k,
)
from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES, DATASETS, get_sequences,
    pot_image_paths, pt_image_paths,
    warp_corners,
)
from flatsam2.path_config import paths


# ---------------------------------------------------------------------------
# Embedded HTML frontend
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FlatSAM2 Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; font-size: 13px; background: #1e1e1e; color: #ddd; height: 100vh; overflow: hidden; }
    #app { display: flex; height: 100vh; }

    /* Left panel — heatmap */
    #left { width: 35%; border-right: 1px solid #444; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
    #left.hidden { display: none; }
    #heatmap-toolbar { padding: 6px 8px; display: flex; gap: 6px; align-items: center; background: #252525;
                       border-bottom: 1px solid #444; flex-shrink: 0; }
    #heatmap-toolbar button { padding: 3px 8px; background: #333; color: #ccc; border: 1px solid #555; cursor: pointer; font: inherit; }
    #heatmap-toolbar button.active { background: #555; color: #fff; border-color: #888; }
    #heatmap-status { color: #666; margin-left: 6px; font-size: 11px; }
    #heatmap-wrap { flex: 1; overflow: auto; cursor: crosshair; }
    /* image natural size is 1px/frame × 1px/seq; CSS scales it to fill panel width at zoom=1 */
    #heatmap-img { display: block; image-rendering: pixelated; width: 100%; }
    #zoom-hint { font-size: 10px; color: #555; padding: 2px 8px; flex-shrink: 0; }
    #tooltip { position: fixed; background: rgba(0,0,0,0.9); color: #fff; padding: 4px 8px; border-radius: 3px;
               font-size: 12px; pointer-events: none; display: none; white-space: pre; z-index: 100; line-height: 1.6; }

    /* Right panel */
    #right { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
    #seq-header { padding: 6px 10px; background: #252525; border-bottom: 1px solid #444;
                  flex-shrink: 0; display: flex; align-items: center; gap: 8px; }
    #btn-toggle-heatmap { padding: 3px 8px; background: #333; color: #ccc; border: 1px solid #555;
                          cursor: pointer; font: inherit; flex-shrink: 0; }
    #btn-toggle-heatmap:hover { background: #444; }
    #seq-header-text { flex: 1; min-width: 0; }
    #seq-name { font-size: 14px; font-weight: bold; color: #4fc3f7; }
    #seq-stats { font-size: 12px; color: #aaa; margin-top: 2px; min-height: 16px; }
    #right-content { flex: 1; overflow: auto; display: flex; flex-direction: column; min-height: 0; }
    #traj-section { padding: 4px 8px; border-bottom: 1px solid #2a2a2a; flex-shrink: 0; }
    #traj-img { max-width: 100%; display: none; }
    #frame-section { flex: 1; padding: 6px 8px; display: flex; flex-direction: column; gap: 5px; min-height: 0; }
    #frame-img-wrap { flex: 1; display: flex; align-items: center; justify-content: center; background: #111;
                      min-height: 200px; overflow: hidden; }
    #frame-img { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
    #no-seq-msg { color: #555; }
    #timeline-wrap { flex-shrink: 0; background: #111; border-top: 1px solid #2a2a2a;
                     border-bottom: 1px solid #2a2a2a; cursor: crosshair; position: relative; }
    #timeline-canvas { display: block; width: 100%; height: 56px; }
    #frame-controls { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
    #frame-slider { flex: 1; accent-color: #4fc3f7; }
    #frame-num { min-width: 90px; color: #aaa; text-align: right; font-size: 12px; }
    #overlay-controls { display: flex; gap: 14px; flex-shrink: 0; flex-wrap: wrap; }
    #overlay-controls label { display: flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; }
    #frame-info { color: #aaa; font-size: 12px; min-height: 18px; flex-shrink: 0; }
    #rmse-info { font-size: 12px; min-height: 18px; flex-shrink: 0; display: flex; gap: 16px; flex-wrap: wrap; }
    .rmse-cand { display: flex; align-items: center; gap: 5px; }
    .rmse-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .rmse-label { color: #888; }
    .rmse-val { color: #ccc; }
    .rmse-sel { font-weight: bold; color: #fff; }
    .rmse-sigma { color: #666; font-size: 11px; }
    #legend { padding: 5px 8px; border-top: 1px solid #2a2a2a; font-size: 11px; color: #888;
              display: flex; gap: 10px; flex-wrap: wrap; flex-shrink: 0; }
    .ls { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; }
  </style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="heatmap-toolbar">
      <button id="btn-regret" class="active" onclick="switchHeatmap('regret')">Regret</button>
      <button id="btn-taxonomy" onclick="switchHeatmap('taxonomy')">Taxonomy</button>
      <span id="heatmap-status"></span>
    </div>
    <div id="heatmap-wrap">
      <img id="heatmap-img" src="" alt="">
    </div>
    <div id="zoom-hint">scroll to zoom · click to select</div>
    <div id="tooltip"></div>
  </div>

  <div id="right">
    <div id="seq-header">
      <button id="btn-toggle-heatmap" onclick="toggleHeatmap()" title="Toggle heatmap (H)">&#8644;</button>
      <div id="seq-header-text">
        <div id="seq-name">Click a sequence in the heatmap</div>
        <div id="seq-stats"></div>
      </div>
    </div>
    <div id="right-content">
      <div id="traj-section">
        <img id="traj-img" alt="trajectory">
      </div>
      <div id="frame-section">
        <div id="frame-img-wrap">
          <span id="no-seq-msg">No sequence selected</span>
          <img id="frame-img" alt="frame">
        </div>
        <div id="timeline-wrap">
          <canvas id="timeline-canvas" height="56"></canvas>
        </div>
        <div id="frame-controls">
          <button onclick="navFrame(-1)">&#9664;</button>
          <input type="range" id="frame-slider" min="0" max="0" value="0" oninput="goToFrame(+this.value)">
          <button onclick="navFrame(+1)">&#9654;</button>
          <span id="frame-num">—</span>
        </div>
        <div id="overlay-controls">
          <label><input type="checkbox" id="chk-pred" checked onchange="refreshFrame()"> Pred</label>
          <label><input type="checkbox" id="chk-gt" checked onchange="refreshFrame()"> GT</label>
          <label><input type="checkbox" id="chk-cand" checked onchange="refreshFrame()"> Candidates</label>
          <label><input type="checkbox" id="chk-mask" onchange="refreshFrame()"> Mask</label>
        </div>
        <div id="frame-info">—</div>
        <div id="rmse-info"></div>
      </div>
    </div>
    <div id="legend">
      <span>Cands:</span>
      <span><span class="ls" style="background:#2196F3"></span> flow</span>
      <span><span class="ls" style="background:#FF9800"></span> mask_h</span>
      <span><span class="ls" style="background:#4CAF50"></span> flow+mask</span>
      <span><span class="ls" style="background:#F44336"></span> last_H</span>
      <span style="margin-left:4px">Pred:<span class="ls" style="background:#00ff00;margin-left:4px"></span></span>
      <span>GT:<span class="ls" style="background:#fff;margin-left:4px"></span> dashed</span>
      <span>Mask:<span class="ls" style="background:rgba(255,80,80,0.6);margin-left:4px"></span></span>
    </div>
  </div>
</div>

<script>
  let currentType  = 'regret';
  let heatmapMeta  = null;
  let currentSeq   = null;
  let currentFrame = 0;
  let seqData      = null;
  let frameEvalMap = {};
  let heatmapZoom  = 1.0;   // 1 = fit panel width
  let leftVisible  = true;

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.key === 'ArrowLeft')  { navFrame(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { navFrame(+1); e.preventDefault(); }
    if (e.key === 'h' || e.key === 'H') toggleHeatmap();
  });

  // Back/forward button
  let _restoringFromHistory = false;
  window.addEventListener('popstate', e => {
    if (e.state && e.state.seq) {
      _restoringFromHistory = true;
      selectSequence(e.state.seq, e.state.frame || 0);
    }
  });

  // Boot
  loadHeatmap('regret');
  const _initParams = new URLSearchParams(location.search);
  const _initSeq    = _initParams.get('seq');
  const _initFrame  = parseInt(_initParams.get('frame') || '0', 10);
  if (_initSeq) selectSequence(_initSeq, _initFrame);

  // ── Heatmap panel toggle ──────────────────────────────────────────────────
  function toggleHeatmap() {
    leftVisible = !leftVisible;
    document.getElementById('left').classList.toggle('hidden', !leftVisible);
    document.getElementById('btn-toggle-heatmap').style.opacity = leftVisible ? '1' : '0.45';
  }

  // ── Heatmap type ─────────────────────────────────────────────────────────
  function switchHeatmap(type) {
    currentType = type;
    document.getElementById('btn-regret').className   = type === 'regret'   ? 'active' : '';
    document.getElementById('btn-taxonomy').className = type === 'taxonomy' ? 'active' : '';
    loadHeatmap(type);
  }

  async function loadHeatmap(type) {
    document.getElementById('heatmap-status').textContent = 'Loading…';
    const res = await fetch('/heatmaps/' + type + '/meta');
    if (!res.ok) { document.getElementById('heatmap-status').textContent = 'Error'; return; }
    heatmapMeta = await res.json();
    // Reset zoom when switching type
    heatmapZoom = 1.0;
    applyZoom();
    const img = document.getElementById('heatmap-img');
    img.src = '/heatmaps/' + type + '.png?t=' + Date.now();
    img.onload = () => {
      document.getElementById('heatmap-status').textContent =
        heatmapMeta.n_seqs + ' × ' + heatmapMeta.max_T;
    };
  }

  // ── Heatmap scroll zoom ───────────────────────────────────────────────────
  const heatmapWrap = document.getElementById('heatmap-wrap');
  const heatmapImg  = document.getElementById('heatmap-img');
  const tooltip     = document.getElementById('tooltip');

  heatmapWrap.addEventListener('wheel', e => {
    e.preventDefault();
    // Zoom around the mouse position
    const wrap    = heatmapWrap;
    const rect    = heatmapImg.getBoundingClientRect();
    const mouseX  = e.clientX - rect.left;   // px from left edge of image
    const scrollX = wrap.scrollLeft + (e.clientX - wrap.getBoundingClientRect().left);

    const factor  = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    const newZoom = Math.max(1.0, Math.min(40.0, heatmapZoom * factor));
    const ratio   = newZoom / heatmapZoom;
    heatmapZoom   = newZoom;
    applyZoom();

    // Keep the pixel under the cursor stationary
    wrap.scrollLeft = scrollX * ratio - (e.clientX - wrap.getBoundingClientRect().left);
  }, { passive: false });

  function applyZoom() {
    // width% = zoom * 100% of the wrapper's client width
    heatmapImg.style.width = (heatmapZoom * 100) + '%';
  }

  // ── Heatmap interaction ───────────────────────────────────────────────────
  function heatmapCoords(e) {
    if (!heatmapMeta) return null;
    const rect  = heatmapImg.getBoundingClientRect();
    const x     = e.clientX - rect.left;
    const y     = e.clientY - rect.top;
    const col   = Math.floor(x / rect.width  * heatmapMeta.max_T);
    const row   = Math.floor(y / rect.height * heatmapMeta.n_seqs);
    if (row < 0 || row >= heatmapMeta.n_seqs || col < 0) return null;
    const seqName = heatmapMeta.seq_names[row];
    const T       = heatmapMeta.row_lengths[row];
    const p15     = heatmapMeta.p15s[row];
    const frame   = Math.max(0, Math.min(col, T - 1));
    return { row, frame, seqName, T, p15 };
  }

  heatmapImg.addEventListener('mousemove', e => {
    const c = heatmapCoords(e);
    if (!c) { tooltip.style.display = 'none'; return; }
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 10) + 'px';
    tooltip.textContent = c.seqName + '\nFrame ' + c.frame + ' / ' + (c.T - 1) + '\nP@15: ' + c.p15.toFixed(1) + '%';
  });

  heatmapImg.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

  heatmapImg.addEventListener('click', e => {
    const c = heatmapCoords(e);
    if (!c) return;
    selectSequence(c.seqName, c.frame);
    // Auto-hide heatmap after selecting so frame gets more space
    if (leftVisible) toggleHeatmap();
  });

  // ── Sequence selection ────────────────────────────────────────────────────
  async function selectSequence(seqName, startFrame) {
    document.getElementById('seq-name').textContent  = seqName;
    document.getElementById('seq-stats').textContent = 'Loading…';
    const res = await fetch('/api/sequence/' + encodeURIComponent(seqName));
    if (!res.ok) { document.getElementById('seq-stats').textContent = 'Error'; return; }
    seqData    = await res.json();
    currentSeq = seqName;
    if (_restoringFromHistory) {
      _restoringFromHistory = false;
    } else {
      history.pushState({seq: seqName, frame: startFrame || 0}, '',
                        '?seq=' + encodeURIComponent(seqName) + '&frame=' + (startFrame || 0));
    }

    document.getElementById('seq-stats').textContent =
      'P@5: ' + seqData.p5.toFixed(1) + '%  P@15: ' + seqData.p15.toFixed(1) + '%' +
      '  frames: ' + seqData.num_frames + (seqData.has_debug ? '  [debug]' : '');

    frameEvalMap = {};
    seqData.eval_frame_idx.forEach((fi, i) => { frameEvalMap[fi] = i; });

    const trajImg = document.getElementById('traj-img');
    trajImg.style.display = 'block';
    trajImg.src = '/trajectories/' + encodeURIComponent(seqName) + '.png?t=' + Date.now();
    trajImg.onerror = () => { trajImg.style.display = 'none'; };

    document.getElementById('frame-slider').max = seqData.num_frames - 1;
    drawTimeline();
    goToFrame(startFrame || 0);
  }

  // ── Frame navigation ──────────────────────────────────────────────────────
  async function goToFrame(idx) {
    if (!currentSeq || !seqData) return;
    currentFrame = Math.max(0, Math.min(idx, seqData.num_frames - 1));
    history.replaceState({seq: currentSeq, frame: currentFrame}, '',
                         '?seq=' + encodeURIComponent(currentSeq) + '&frame=' + currentFrame);
    document.getElementById('frame-slider').value = currentFrame;
    document.getElementById('frame-num').textContent = currentFrame + ' / ' + (seqData.num_frames - 1);
    updateFrameInfo(currentFrame);
    drawTimeline();

    const pred = document.getElementById('chk-pred').checked ? 1 : 0;
    const gt   = document.getElementById('chk-gt').checked   ? 1 : 0;
    const cand = document.getElementById('chk-cand').checked ? 1 : 0;
    const mask = document.getElementById('chk-mask').checked ? 1 : 0;
    const url  = '/api/frame/' + encodeURIComponent(currentSeq) + '/' + currentFrame +
                 '?pred=' + pred + '&gt=' + gt + '&candidates=' + cand + '&mask=' + mask + '&t=' + Date.now();

    const frameImg = document.getElementById('frame-img');
    document.getElementById('no-seq-msg').style.display = 'none';
    frameImg.style.display = 'block';
    frameImg.src = url;
  }

  const CAND_COLORS = { flow: '#2196F3', mask_h: '#FF9800', flow_maskH: '#4CAF50', last_H: '#F44336' };
  const CAND_PATHS  = ['flow', 'mask_h', 'flow_maskH'];
  const CAND_LABELS = { flow: 'flow', mask_h: 'mask_h', flow_maskH: 'flow+mask' };

  function updateFrameInfo(frameIdx) {
    const rmseEl = document.getElementById('rmse-info');
    const infoEl = document.getElementById('frame-info');
    const pos = frameEvalMap[frameIdx];

    if (pos !== undefined && seqData) {
      const err    = seqData.errors[pos];
      const oracle = seqData.oracle_errors ? seqData.oracle_errors[pos] : null;
      const sel    = seqData.selected      ? seqData.selected[pos]      : null;
      let info = 'err: ' + (isFinite(err) ? err.toFixed(1) + 'px' : 'N/A');
      if (oracle !== null && oracle !== undefined && isFinite(oracle)) {
        info += '   oracle: ' + oracle.toFixed(1) + 'px';
        info += '   regret: ' + Math.max(0, err - oracle).toFixed(1) + 'px';
      }
      if (sel) info += '   sel: ' + sel;
      infoEl.textContent = info;

      // Per-candidate predicted RMSE row
      if (seqData.pred_rmse) {
        const parts = CAND_PATHS.map(p => {
          const val   = seqData.pred_rmse[p]  ? seqData.pred_rmse[p][pos]  : null;
          const sigma = seqData.pred_sigma && seqData.pred_sigma[p] ? seqData.pred_sigma[p][pos] : null;
          const isSel = (p === sel);
          const color = CAND_COLORS[p] || '#888';
          const label = CAND_LABELS[p] || p;
          const valStr   = (val   !== null && val   !== undefined) ? val.toFixed(1)   + 'px' : '—';
          const sigmaStr = (sigma !== null && sigma !== undefined) ? '±' + sigma.toFixed(1) + 'px' : '';
          return `<span class="rmse-cand ${isSel ? 'rmse-sel' : ''}">` +
                 `<span class="rmse-dot" style="background:${color}"></span>` +
                 `<span class="rmse-label">${label}</span>` +
                 `<span class="rmse-val">${valStr}` +
                 (sigmaStr ? `<span class="rmse-sigma"> ${sigmaStr}</span>` : '') +
                 `</span>` +
                 `</span>`;
        });
        rmseEl.innerHTML = '<span style="color:#666;margin-right:4px">pred RMSE:</span>' + parts.join('');
      } else {
        rmseEl.innerHTML = '';
      }
    } else {
      infoEl.textContent = 'frame ' + frameIdx + ' (not evaluated)';
      rmseEl.innerHTML = '';
    }
  }

  function navFrame(delta)  { if (currentSeq) goToFrame(currentFrame + delta); }
  function refreshFrame()   { if (currentSeq) goToFrame(currentFrame); }

  // ── Timeline plot ─────────────────────────────────────────────────────────
  const timelineCanvas = document.getElementById('timeline-canvas');
  const timelineWrap   = document.getElementById('timeline-wrap');

  timelineCanvas.addEventListener('click', e => {
    if (!seqData) return;
    const rect = timelineCanvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    goToFrame(Math.round(frac * (seqData.num_frames - 1)));
  });

  // Redraw on resize
  new ResizeObserver(() => { if (seqData) drawTimeline(); }).observe(timelineWrap);

  function drawTimeline() {
    if (!seqData) return;
    const W = timelineWrap.clientWidth || 600;
    const H = 56;
    timelineCanvas.width  = W;
    timelineCanvas.height = H;
    const ctx = timelineCanvas.getContext('2d');

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);

    const T = seqData.num_frames;
    if (!T) return;

    // Build dense per-frame error array
    const frameErrors = new Float32Array(T).fill(NaN);
    seqData.eval_frame_idx.forEach((fi, i) => { frameErrors[fi] = seqData.errors[i]; });

    // Scale: max of data or 15 so threshold line is always meaningful
    const validVals = seqData.errors.filter(isFinite);
    const trim = 30
    const maxErr = trim // validVals.length ? Math.max(Math.max(...validVals), 15) : 15;
    const pad = 4;  // bottom padding so zero bars are still visible

    const barW = W / T;
    for (let f = 0; f < T; f++) {
      const err = frameErrors[f];
      if (!isFinite(err)) continue;
      const norm = Math.min(err / maxErr, 1.0);
      // Green → yellow → red gradient
      const r = norm < 0.5 ? Math.round(norm * 2 * 255) : 255;
      const g = norm < 0.5 ? 255 : Math.round((1 - norm) * 2 * 255);
      ctx.fillStyle = `rgb(${r},${g},0)`;
      const bh = Math.max(1, norm * (H - pad));
      ctx.fillRect(f / T * W, H - pad - bh, Math.max(1, barW + 0.5), bh);
    }

    // Dashed threshold line at 15 px error
    const threshY = H - pad - (15 / maxErr) * (H - pad);
    if (threshY > 0 && threshY < H) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(0, threshY); ctx.lineTo(W, threshY); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Current-frame indicator
    const cx = T > 1 ? (currentFrame / (T - 1)) * W : 0;
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class _State:
    dataset: str = ''
    base_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    initfix_dir: Optional[Path] = None
    load_gt_fn = None

    seq_cache: Dict[str, dict] = {}          # name -> full sequence data (lazy)
    image_path_cache: Dict[str, Optional[List[Path]]] = {}

    # Built at startup
    sorted_seq_names: List[str] = []         # worst p15 first
    heatmap_store: Dict[str, dict] = {}      # type -> {png_bytes, meta}


_st = _State()
app = FastAPI(title='FlatSAM2 Viewer')


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def _image_paths(seq_name: str) -> Optional[List[Path]]:
    if seq_name in _st.image_path_cache:
        return _st.image_path_cache[seq_name]
    try:
        if DATASETS[_st.dataset].gt_loader == 'pot':
            result = pot_image_paths(_st.base_dir, seq_name)
        else:
            result = pt_image_paths(_st.base_dir, seq_name)
        _st.image_path_cache[seq_name] = result
        return result
    except Exception:
        _st.image_path_cache[seq_name] = None
        return None


# ---------------------------------------------------------------------------
# Sequence data (lazy)
# ---------------------------------------------------------------------------

def _ensure_seq(seq_name: str) -> Optional[dict]:
    """Return cached sequence data, loading if necessary."""
    if seq_name in _st.seq_cache:
        return _st.seq_cache[seq_name]
    try:
        gt, valid = _st.load_gt_fn(_st.base_dir, seq_name)
        pred      = _load_predictions(_st.results_dir, seq_name)
        debug     = _load_debug(_st.results_dir, seq_name)
        T         = min(len(gt), len(pred))
        data      = _build_sequence_data(seq_name, gt[:T], valid[:T], pred[:T], debug)
        data['gt']        = gt[:T]
        data['valid']     = valid[:T]
        data['pred']      = pred[:T]
        data['raw_debug'] = debug   # cache for frame-level candidate rendering
        _st.seq_cache[seq_name] = data
        return data
    except Exception as e:
        print(f'  Warning loading {seq_name}: {e}', file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _hex_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _draw_poly(img: np.ndarray, corners: np.ndarray, color: tuple,
               thickness: int, dashed: bool = False) -> None:
    """Draw a closed quadrilateral from (2, 4) corners."""
    if dashed:
        n = corners.shape[1]
        for i in range(n):
            p1 = (int(corners[0, i]), int(corners[1, i]))
            p2 = (int(corners[0, (i + 1) % n]), int(corners[1, (i + 1) % n]))
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = max(1, int(np.hypot(dx, dy)))
            dash, gap = 8, 6
            for s in range(0, dist, dash + gap):
                e = min(s + dash, dist)
                xs = p1[0] + dx * s // dist
                ys = p1[1] + dy * s // dist
                xe = p1[0] + dx * e // dist
                ye = p1[1] + dy * e // dist
                cv2.line(img, (xs, ys), (xe, ye), color, thickness)
        cv2.circle(img, (int(corners[0, 0]), int(corners[1, 0])), 8, color, -1)
    else:
        pts = rearrange(corners, 'xy c -> c 1 xy').astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        for corner_i, (x, y) in enumerate(corners.T):
            if corner_i == 0:
                radius = 8
            else:
                radius = 4
            cv2.circle(img, (int(x), int(y)), radius, color, -1)


def _placeholder(text: str = 'Image not available', h: int = 480, w: int = 640) -> np.ndarray:
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    for i, line in enumerate(text.split('\n')):
        cv2.putText(img, line, (20, h // 2 - 10 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    return img


def render_frame(
    seq_name: str, frame_idx: int,
    show_pred: bool = True, show_gt: bool = True,
    show_candidates: bool = True, show_mask: bool = False,
) -> bytes:
    """Render frame with overlays and return JPEG bytes."""
    seq = _ensure_seq(seq_name)

    # Load raw image
    img: Optional[np.ndarray] = None
    paths_list = _image_paths(seq_name)
    if paths_list and frame_idx < len(paths_list):
        p = paths_list[frame_idx]
        if p.exists():
            img = cv2.imread(str(p))
    if img is None:
        msg = f'{seq_name}  frame {frame_idx}'
        if paths_list is None:
            msg += '\n(dataset not mounted)'
        img = _placeholder(msg)

    if seq is None:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return bytes(buf)

    gt    = seq['gt']     # (T, 2, 4)
    pred  = seq['pred']   # (T, 2, 4)
    valid = seq['valid']  # (T,)

    # GT corners — white dashed
    if show_gt and frame_idx < len(gt) and valid[frame_idx]:
        _draw_poly(img, gt[frame_idx], (255, 255, 255), 2, dashed=True)

    # Candidate homographies — coloured thin lines
    if show_candidates and seq.get('has_debug'):
        raw_debug = seq.get('raw_debug')
        if raw_debug is not None:
            init_corners = gt[0]
            frame_to_row = {int(fi): i for i, fi in enumerate(raw_debug['frame_idx'])}
            row = frame_to_row.get(frame_idx, -1)
            if row >= 0:
                for p in CAND_PATHS:
                    H = raw_debug[f'{p}_H'][row]
                    if not np.isnan(H[0, 0]):
                        warped = warp_corners(H, init_corners)
                        _draw_poly(img, warped, _hex_bgr(CAND_COLORS[p]), 1)

    # Segmentation mask — semi-transparent red overlay
    if show_mask:
        masks_root = _st.masks_dir if _st.masks_dir is not None else _st.results_dir
        mask_path = masks_root / 'masks' / seq_name / f'{frame_idx + 1:05d}.png'
        if mask_path.exists():
            seg_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if seg_mask is not None:
                # Resize mask to match frame if needed
                if seg_mask.shape[:2] != img.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                colored = img.copy()
                colored[seg_mask > 0] = (80, 80, 255)   # BGR red
                img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)

    # Predicted corners — green solid
    if show_pred and frame_idx < len(pred):
        _draw_poly(img, pred[frame_idx], (0, 255, 0), 2)

    # Text overlay
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    eval_idx = np.where(eval_mask)[0]
    pos_arr  = np.where(eval_idx == frame_idx)[0]

    lines = [f'Frame {frame_idx}']
    if len(pos_arr):
        pos = pos_arr[0]
        err = seq['errors'][pos] if pos < len(seq['errors']) else float('nan')
        lines.append(f'Err: {err:.1f} px')
        if seq.get('has_debug'):
            oracle = seq['oracle_errors'][pos] if pos < len(seq['oracle_errors']) else float('nan')
            if np.isfinite(oracle):
                lines.append(f'Oracle: {oracle:.1f} px')
            sel = seq['selected'][pos] if pos < len(seq['selected']) else ''
            if sel:
                lines.append(f'Sel: {sel}')

    y = 22
    for ln in lines:
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf)


# ---------------------------------------------------------------------------
# Heatmap image generation
# ---------------------------------------------------------------------------

def _magma_rgb(values: np.ndarray, vmax: float = 20.0) -> np.ndarray:
    """Apply hot_r colormap to a 1-D array. Returns (N, 3) uint8 RGB."""
    import matplotlib.cm as cm
    normed = np.clip(values / vmax, 0.0, 1.0)
    rgba   = cm.magma(normed)
    return (rgba[:, :3] * 255).astype(np.uint8)


_TAXONOMY_RGB: Dict[str, Tuple[int, int, int]] = {
    'both_good': (0x4C, 0xAF, 0x50),
    'flow_only': (0x21, 0x96, 0xF3),
    'mask_only': (0xFF, 0x98, 0x00),
    'neither':   (0xF4, 0x43, 0x36),
    'no_debug':  (0x99, 0x99, 0x99),
    'padding':   (0xCC, 0xCC, 0xCC),
}
_TAXONOMY_KEYS = ('both_good', 'flow_only', 'mask_only', 'neither')


def _build_heatmap(htype: str, sorted_seqs: List[dict]) -> Tuple[bytes, dict]:
    """Build a raw-pixel PNG + metadata dict for *htype* ('regret'|'taxonomy')."""
    n_seqs = len(sorted_seqs)
    max_T  = max(len(s['errors']) for s in sorted_seqs)
    PAD_RGB = (0xCC, 0xCC, 0xCC)

    img = np.full((n_seqs, max_T, 3), PAD_RGB, dtype=np.uint8)

    for row, s in enumerate(sorted_seqs):
        T = len(s['errors'])
        if T == 0:
            continue

        if htype == 'regret':
            if s.get('has_debug') and len(s.get('oracle_errors', [])) == T:
                oracle = s['oracle_errors']
                regret = np.clip(
                    s['errors'] - np.where(np.isfinite(oracle), oracle, 0.0),
                    0, None,
                )
            else:
                regret = s['errors'].copy()
            img[row, :T] = _magma_rgb(regret, vmax=20.0)

        else:  # taxonomy
            if not s.get('has_debug'):
                img[row, :T] = _TAXONOMY_RGB['no_debug']
                continue
            fe = s['cand_errors']['flow']
            me = s['cand_errors']['mask']
            fg = np.isfinite(fe) & (fe <= THRESHOLD)
            mg = np.isfinite(me) & (me <= THRESHOLD)
            # Build row colours in a local array to avoid chained-index pitfalls
            row_rgb = np.full((T, 3), _TAXONOMY_RGB['neither'], dtype=np.uint8)
            row_rgb[~fg &  mg] = _TAXONOMY_RGB['mask_only']
            row_rgb[ fg & ~mg] = _TAXONOMY_RGB['flow_only']
            row_rgb[ fg &  mg] = _TAXONOMY_RGB['both_good']
            img[row, :T] = row_rgb

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf   = cv2.imencode('.png', img_bgr)
    png_bytes = bytes(buf)

    meta = {
        'seq_names':   [s['name'] for s in sorted_seqs],
        'row_lengths': [len(s['errors']) for s in sorted_seqs],
        'p15s':        [float(_p_at_k(s['errors'])) for s in sorted_seqs],
        'max_T':       max_T,
        'n_seqs':      n_seqs,
    }
    return png_bytes, meta


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get('/', response_class=HTMLResponse)
async def root():
    return HTML_PAGE


@app.get('/api/sequences')
async def api_sequences():
    out = []
    for name in _st.sorted_seq_names:
        s = _st.seq_cache.get(name)
        if s is None:
            continue
        out.append({
            'name':      name,
            'p15':       float(_p_at_k(s['errors'])),
            'p5':        float(_p_at_k(s['errors'], 5.0)),
            'num_valid': len(s['errors']),
        })
    return JSONResponse(out)


@app.get('/heatmaps/{htype}.png')
async def heatmap_png(htype: str):
    if htype not in _st.heatmap_store:
        raise HTTPException(404, f'Unknown heatmap type: {htype}')
    png = _st.heatmap_store[htype]['png_bytes']
    return StreamingResponse(iter([png]), media_type='image/png')


@app.get('/heatmaps/{htype}/meta')
async def heatmap_meta(htype: str):
    if htype not in _st.heatmap_store:
        raise HTTPException(404, f'Unknown heatmap type: {htype}')
    return JSONResponse(_st.heatmap_store[htype]['meta'])


@app.get('/api/sequence/{name}')
async def api_sequence(name: str):
    seq = _ensure_seq(name)
    if seq is None:
        raise HTTPException(404, f'Sequence not found or failed to load: {name}')

    eval_mask = seq['valid'].copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    eval_idx = np.where(eval_mask)[0]

    out: dict = {
        'name':           name,
        'num_frames':     len(seq['gt']),
        'p15':            float(_p_at_k(seq['errors'])),
        'p5':             float(_p_at_k(seq['errors'], 5.0)),
        'has_debug':      bool(seq.get('has_debug', False)),
        'eval_frame_idx': eval_idx.tolist(),
        'errors':         [float(e) for e in seq['errors']],
    }

    if seq.get('has_debug'):
        out['oracle_errors'] = [
            float(v) if np.isfinite(v) else None for v in seq['oracle_errors']
        ]
        out['selected'] = seq['selected'].tolist()
        out['pred_rmse'] = {
            p: [float(v) if np.isfinite(v) else None for v in seq['pred_rmse'][p]]
            for p in CAND_PATHS
        }
        out['pred_sigma'] = {
            p: [float(v) if np.isfinite(v) else None for v in seq['pred_sigma'][p]]
            for p in CAND_PATHS
        }

    return JSONResponse(out)


@app.get('/api/frame/{name}/{frame_idx}')
async def api_frame(
    name: str, frame_idx: int,
    pred: int = 1, gt: int = 1, candidates: int = 1, mask: int = 0,
):
    jpeg = render_frame(
        name, frame_idx,
        show_pred=bool(pred),
        show_gt=bool(gt),
        show_candidates=bool(candidates),
        show_mask=bool(mask),
    )
    return StreamingResponse(iter([jpeg]), media_type='image/jpeg')


@app.get('/trajectories/{name}.png')
async def trajectory_png(name: str):
    traj_file = _st.output_dir / 'trajectories' / f'{name}.png'
    if not traj_file.exists():
        raise HTTPException(404, f'Trajectory file not found: {traj_file}')
    return StreamingResponse(
        iter([traj_file.read_bytes()]),
        media_type='image/png',
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup(
    dataset: str,
    base_dir: Path,
    results_dir: Path,
    output_dir: Path,
    sequences: List[str],
    initfix_dir: Optional[Path],
    masks_dir: Optional[Path] = None,
) -> None:
    """Load all sequence data and build heatmap images."""
    _st.dataset     = dataset
    _st.base_dir    = base_dir
    _st.results_dir = results_dir
    _st.masks_dir   = masks_dir
    _st.output_dir  = output_dir
    _st.initfix_dir = initfix_dir
    _st.load_gt_fn  = _make_load_gt_fn(dataset, initfix_dir)

    print(f'Loading {len(sequences)} sequences from {results_dir} …')
    loaded = []
    n_fail = 0

    for seq_name in sequences:
        try:
            gt, valid = _st.load_gt_fn(base_dir, seq_name)
            pred  = _load_predictions(results_dir, seq_name)
            debug = _load_debug(results_dir, seq_name)
            T     = min(len(gt), len(pred))
            data  = _build_sequence_data(seq_name, gt[:T], valid[:T], pred[:T], debug)
            data['gt']        = gt[:T]
            data['valid']     = valid[:T]
            data['pred']      = pred[:T]
            data['raw_debug'] = debug
            if len(data['errors']) > 0:
                loaded.append(data)
                _st.seq_cache[seq_name] = data
        except Exception as e:
            print(f'  WARNING: {seq_name}: {e}', file=sys.stderr)
            n_fail += 1

    n_debug = sum(1 for s in loaded if s.get('has_debug'))
    print(f'  {len(loaded)} ok, {n_fail} failed, {n_debug} with debug NPZ')

    # Sort worst-first
    loaded.sort(key=lambda s: _p_at_k(s['errors']))
    _st.sorted_seq_names = [s['name'] for s in loaded]

    # Build heatmap images
    if loaded:
        print('Building heatmap images …')
        for htype in ('regret', 'taxonomy'):
            png_bytes, meta = _build_heatmap(htype, loaded)
            _st.heatmap_store[htype] = {'png_bytes': png_bytes, 'meta': meta}
        print('Ready.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Interactive FlatSAM2 results viewer')
    parser.add_argument('dataset', choices=CLI_CHOICES, help='Dataset to inspect')
    parser.add_argument('--config', type=Path, help='Config file (for results directory structure)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Analysis output directory (trajectory PNGs etc.)')
    parser.add_argument('--results-dir', type=Path, default=None,
                        help='Top-level results directory (default: from path_config)')
    parser.add_argument('--base-dir', type=Path, default=None,
                        help='Dataset base directory (default: from path_config)')
    parser.add_argument('--initfix-dir', type=Path, default=None,
                        help='Initfix directory for pti dataset (default: from path_config)')
    parser.add_argument('--masks-dir', type=Path, default=None,
                        help='Directory containing masks/ subdirectory (overrides results-dir for mask loading)')
    parser.add_argument('--sequences', nargs='+', help='Specific sequences to load')
    parser.add_argument('--subset', help='Subset file name (configs/subsets/<name>.txt)')
    parser.add_argument('--split', choices=['train', 'test'], help='Dataset split')
    parser.add_argument('--port', type=int, default=8123)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]

    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == 'pti' and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY

    results_dir = compute_output_dir(args.results_dir, args.dataset, args.config)

    if args.output_dir is None:
        args.output_dir = results_dir  # look for trajectories/ next to predictions

    if not results_dir.exists():
        print(f'ERROR: Results directory not found: {results_dir}')
        sys.exit(1)

    try:
        sequences = get_sequences(
            args.dataset, args.base_dir,
            sequences=args.sequences,
            subset=args.subset,
            split=args.split,
        )
    except ValueError as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    _startup(
        dataset=args.dataset,
        base_dir=args.base_dir,
        results_dir=results_dir,
        output_dir=args.output_dir,
        sequences=sequences,
        initfix_dir=args.initfix_dir,
        masks_dir=args.masks_dir,
    )

    print(f'Serving at http://{args.host}:{args.port}')
    hostname = socket.gethostname()
    print(f'ssh -L 8080:{hostname}:{args.port} rci')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
