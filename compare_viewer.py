"""Interactive comparison viewer for two FlatSAM2 tracking methods.

Shows a clickable per-sequence/per-frame heatmap of error differences and a frame
viewer with overlays for both methods.

A comparison report (dominance curves + p@15 scatter plot + HTML) is saved
automatically to --report-dir/<label_a>_vs_<label_b>/<dataset>/.

Usage:
    python compare_viewer.py pti --config-a configs/tracker/default.py \
        --config-b configs/tracker/flow_hough_symmetry.py \
        --label-a baseline --label-b new
    python compare_viewer.py pti --results-dir-a /path/to/method_a \
        --results-dir-b /path/to/method_b --label-a baseline --label-b new
"""

import argparse
import atexit
import base64
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from analyze_results import _load_debug, _load_predictions, _make_load_gt_fn
from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_image_paths,
    pt_image_paths,
)
from flatsam2.metrics import alignment_error, best_rotation_error
from flatsam2.path_config import paths


THRESHOLD = 15.0
WIN_MARGIN_PX = 1.0
DEFAULT_GOOD_THR = 15.0
DEFAULT_BAD_THR = 30.0
SORT_MODES = ("severity", "gap")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FlatSAM2 Method Compare Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; font-size: 13px; background: #1e1e1e; color: #ddd; height: 100vh; overflow: hidden; }
    #app { display: flex; height: 100vh; }

    #left { width: 38%; border-right: 1px solid #444; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
    #left.hidden { display: none; }
    #heatmap-toolbar { padding: 6px 8px; display: flex; gap: 8px; align-items: center; background: #252525;
                       border-bottom: 1px solid #444; flex-shrink: 0; flex-wrap: wrap; }
    #heatmap-toolbar button { padding: 3px 8px; background: #333; color: #ccc; border: 1px solid #555; cursor: pointer; font: inherit; }
    #heatmap-toolbar button.active { background: #555; color: #fff; border-color: #888; }
    .toolbar-field { display: flex; align-items: center; gap: 4px; color: #aaa; font-size: 11px; }
    .toolbar-field select, .toolbar-field input {
      background: #1f1f1f; color: #ddd; border: 1px solid #555; font: inherit; padding: 2px 4px;
    }
    .toolbar-field input { width: 58px; }
    #heatmap-status { color: #888; margin-left: 4px; font-size: 11px; }
    #heatmap-wrap { flex: 1; overflow: auto; cursor: crosshair; }
    #heatmap-img { display: block; image-rendering: pixelated; width: 100%; }
    #scatter-svg { display: none; width: 100%; cursor: crosshair; }
    #zoom-hint { font-size: 10px; color: #555; padding: 2px 8px; flex-shrink: 0; }
    #tooltip { position: fixed; background: rgba(0,0,0,0.92); color: #fff; padding: 4px 8px; border-radius: 3px;
               font-size: 12px; pointer-events: none; display: none; white-space: pre; z-index: 100; line-height: 1.45; }

    #right { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
    #seq-header { padding: 6px 10px; background: #252525; border-bottom: 1px solid #444;
                  flex-shrink: 0; display: flex; align-items: center; gap: 8px; }
    #btn-toggle-heatmap { padding: 3px 8px; background: #333; color: #ccc; border: 1px solid #555;
                          cursor: pointer; font: inherit; flex-shrink: 0; }
    #btn-toggle-heatmap:hover { background: #444; }
    #seq-header-text { flex: 1; min-width: 0; }
    #seq-name { font-size: 14px; font-weight: bold; color: #4fc3f7; }
    #seq-stats { font-size: 12px; color: #aaa; margin-top: 2px; min-height: 16px; white-space: pre-wrap; }

    #frame-section { flex: 1; padding: 8px; display: flex; flex-direction: column; gap: 6px; min-height: 0; }
    #frame-img-wrap { flex: 1; display: flex; align-items: center; justify-content: center; background: #111;
                      min-height: 220px; overflow: hidden; }
    #frame-img { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
    #no-seq-msg { color: #555; }

    #timeline-wrap { flex-shrink: 0; background: #111; border-top: 1px solid #2a2a2a;
                     border-bottom: 1px solid #2a2a2a; cursor: crosshair; position: relative; }
    #timeline-canvas { display: block; width: 100%; height: 72px; }

    #frame-controls { display: flex; align-items: center; gap: 8px; flex-shrink: 0; flex-wrap: wrap; }
    #frame-slider { flex: 1; min-width: 220px; accent-color: #4fc3f7; }
    #frame-num { min-width: 110px; color: #aaa; text-align: right; font-size: 12px; }
    #jump-controls { display: flex; gap: 6px; }
    #jump-controls button { padding: 3px 8px; background: #333; color: #ccc; border: 1px solid #555; cursor: pointer; font: inherit; }

    #overlay-controls { display: flex; gap: 14px; flex-shrink: 0; flex-wrap: wrap; }
    #overlay-controls label { display: flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; }
    #frame-info { color: #bbb; font-size: 12px; min-height: 18px; flex-shrink: 0; }

    #legend { padding: 6px 8px; border-top: 1px solid #2a2a2a; font-size: 11px; color: #888;
              display: flex; gap: 12px; flex-wrap: wrap; flex-shrink: 0; }
    .ls { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; }
  </style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="heatmap-toolbar">
      <button id="btn-delta" class="active" onclick="switchHeatmap('delta')">Delta</button>
      <button id="btn-winner" onclick="switchHeatmap('winner')">Winner</button>
      <button id="btn-scatter" onclick="switchHeatmap('scatter')">Scatter</button>
      <label class="toolbar-field">Sort
        <select id="sort-mode" onchange="changeSortMode()">
          <option value="severity">Severity first</option>
          <option value="gap">Largest gap</option>
        </select>
      </label>
      <label class="toolbar-field">Good
        <select id="good-preset" onchange="onWinnerSettingChange()">
          <option value="5">5</option>
          <option value="15" selected>15</option>
          <option value="custom">custom</option>
        </select>
        <input id="good-custom" type="number" min="0" step="1" value="15" onchange="onWinnerSettingChange()">
      </label>
      <label class="toolbar-field">Bad
        <select id="bad-preset" onchange="onWinnerSettingChange()">
          <option value="15">15</option>
          <option value="30" selected>30</option>
          <option value="custom">custom</option>
        </select>
        <input id="bad-custom" type="number" min="0" step="1" value="30" onchange="onWinnerSettingChange()">
      </label>
      <button id="btn-br" onclick="toggleBestRotation()" title="Rotation-invariant metric: tries all 4 corner orderings">BR</button>
      <label class="toolbar-field" id="attr-color-field" style="display:none">Color by
        <select id="attr-color-select" onchange="renderScatter()">
          <option value="">winner</option>
        </select>
      </label>
      <span id="heatmap-status"></span>
    </div>
    <div id="heatmap-wrap">
      <img id="heatmap-img" src="" alt="">
      <svg id="scatter-svg" viewBox="0 0 420 420" preserveAspectRatio="xMidYMid meet"></svg>
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

    <div id="frame-section">
      <div id="frame-img-wrap">
        <span id="no-seq-msg">No sequence selected</span>
        <img id="frame-img" alt="frame">
      </div>

      <div id="timeline-wrap">
        <canvas id="timeline-canvas" height="72"></canvas>
      </div>

      <div id="frame-controls">
        <button onclick="navFrame(-1)">&#9664;</button>
        <input type="range" id="frame-slider" min="0" max="0" value="0" oninput="goToFrame(+this.value)">
        <button onclick="navFrame(+1)">&#9654;</button>
        <span id="frame-num">—</span>
      <div id="jump-controls">
          <button id="btn-jump-a" onclick="jumpWinner(-1)">Next A better</button>
          <button id="btn-jump-b" onclick="jumpWinner(+1)">Next B better</button>
          <button id="btn-copy-loc" onclick="copyLocation()" title="Copy sequence name and frame index to clipboard (C)">Copy loc</button>
          <button id="btn-detail-a" onclick="openDetail('a')" style="display:none;border-color:#ff7a00;color:#ff7a00" title="Open current frame in detail viewer for method A">Detail <span id="detail-lbl-a">A</span> ↗</button>
          <button id="btn-detail-b" onclick="openDetail('b')" style="display:none;border-color:#39ff14;color:#39ff14" title="Open current frame in detail viewer for method B">Detail <span id="detail-lbl-b">B</span> ↗</button>
        </div>
      </div>

      <div id="overlay-controls">
        <label><input type="checkbox" id="chk-a" checked onchange="refreshFrame()"> <span id="lbl-a">Method A</span></label>
        <label><input type="checkbox" id="chk-b" checked onchange="refreshFrame()"> <span id="lbl-b">Method B</span></label>
        <label><input type="checkbox" id="chk-gt" checked onchange="refreshFrame()"> GT</label>
      </div>

      <div id="frame-info">—</div>
    </div>

    <div id="legend">
      <span>Winner map:</span>
      <span><span class="ls" style="background:#17a974"></span> <span id="legend-both-good">both good</span></span>
      <span><span class="ls" style="background:#7f1d1d"></span> <span id="legend-both-bad">both bad</span></span>
      <span><span class="ls" style="background:#ff7a00"></span> <span id="legend-a-better">A better</span></span>
      <span><span class="ls" style="background:#39ff14"></span> <span id="legend-b-better">B better</span></span>
      <span>Overlay <span id="legend-overlay-a">A</span>:<span class="ls" style="background:#ff7a00;margin-left:4px"></span></span>
      <span>Overlay <span id="legend-overlay-b">B</span>:<span class="ls" style="background:#39ff14;margin-left:4px"></span></span>
      <span>GT dashed:<span class="ls" style="background:#fff;margin-left:4px"></span></span>
      <span>Mask corrected (timeline): <span id="legend-corr-a">A</span>&#9650; top · <span id="legend-corr-b">B</span>&#9660; bottom</span>
    </div>
  </div>
</div>

<script>
  let methodInfo   = null;
  let currentType  = 'delta';
  let heatmapMeta  = null;
  let currentSeq   = null;
  let currentFrame = 0;
  let seqData      = null;
  let frameEvalMap = {};
  let heatmapZoom  = 1.0;
  let leftVisible  = true;
  let currentSort  = 'severity';
  let attrData     = { attributes: [], annotations: {} };

  const WIN_MARGIN = 1.0;
  const DEFAULT_GOOD = 15.0;
  const DEFAULT_BAD = 30.0;

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.key === 'ArrowLeft')  { navFrame(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { navFrame(+1); e.preventDefault(); }
    if (e.key === 'h' || e.key === 'H') toggleHeatmap();
    if (e.key === 'c' || e.key === 'C') copyLocation();
  });

  let _restoringFromHistory = false;
  window.addEventListener('popstate', e => {
    if (e.state && e.state.seq) {
      _restoringFromHistory = true;
      selectSequence(e.state.seq, e.state.frame || 0);
    }
  });

  async function boot() {
    const infoRes = await fetch('/api/info');
    methodInfo = infoRes.ok ? await infoRes.json() : { label_a: 'A', label_b: 'B' };

    document.getElementById('lbl-a').textContent = methodInfo.label_a;
    document.getElementById('lbl-b').textContent = methodInfo.label_b;
    document.getElementById('btn-jump-a').textContent =
      'Next ' + methodInfo.label_a + ' better';
    document.getElementById('btn-jump-b').textContent =
      'Next ' + methodInfo.label_b + ' better';
    document.getElementById('legend-a-better').textContent = methodInfo.label_a + ' better';
    document.getElementById('legend-b-better').textContent = methodInfo.label_b + ' better';
    document.getElementById('legend-overlay-a').textContent = methodInfo.label_a;
    document.getElementById('legend-overlay-b').textContent = methodInfo.label_b;
    document.getElementById('legend-corr-a').textContent = methodInfo.label_a;
    document.getElementById('legend-corr-b').textContent = methodInfo.label_b;
    updateWinnerLegend();
    updateWinnerControlsVisibility();

    useBestRotation = !!methodInfo.use_best_rotation;
    document.getElementById('btn-br').className = useBestRotation ? 'active' : '';
    if (methodInfo.has_detail) {
      document.getElementById('detail-lbl-a').textContent = methodInfo.label_a;
      document.getElementById('detail-lbl-b').textContent = methodInfo.label_b;
      document.getElementById('btn-detail-a').style.display = '';
      document.getElementById('btn-detail-b').style.display = '';
    }

    const attrRes = await fetch('/api/attributes');
    if (attrRes.ok) {
      attrData = await attrRes.json();
      const sel = document.getElementById('attr-color-select');
      for (const attr of attrData.attributes) {
        const opt = document.createElement('option');
        opt.value = attr;
        opt.textContent = attr;
        sel.appendChild(opt);
      }
    }

    loadHeatmap('delta');
    const params = new URLSearchParams(location.search);
    const initSeq = params.get('seq');
    const initFrame = parseInt(params.get('frame') || '0', 10);
    if (initSeq) selectSequence(initSeq, initFrame);
  }

  let useBestRotation = false;

  async function toggleBestRotation() {
    useBestRotation = !useBestRotation;
    document.getElementById('btn-br').className = useBestRotation ? 'active' : '';
    document.getElementById('heatmap-status').textContent = 'Recomputing…';
    await fetch('/api/best-rotation/' + (useBestRotation ? 1 : 0), { method: 'POST' });
    // reload current view
    if (currentType === 'scatter') {
      const res = await fetch('/heatmaps/delta/meta?sort_mode=' + currentSort);
      if (res.ok) heatmapMeta = await res.json();
      renderScatter();
    } else {
      loadHeatmap(currentType);
    }
    // refresh current sequence data if one is selected
    if (currentSeq) {
      const res = await fetch('/api/sequence/' + encodeURIComponent(currentSeq));
      if (res.ok) {
        seqData = await res.json();
        frameEvalMap = {};
        seqData.eval_frame_idx.forEach((fi, i) => { frameEvalMap[fi] = i; });
        drawTimeline();
        updateFrameInfo(currentFrame);
        document.getElementById('seq-stats').textContent =
          methodInfo.label_a + ': P@15 ' + seqData.p15_a.toFixed(1) + '% · mean ' + seqData.mean_a.toFixed(2) + 'px\n' +
          methodInfo.label_b + ': P@15 ' + seqData.p15_b.toFixed(1) + '% · mean ' + seqData.mean_b.toFixed(2) + 'px\n' +
          'ΔP@15 (B-A): ' + seqData.delta_p15.toFixed(1) + '% · Δmean (A-B): ' + seqData.mean_delta.toFixed(2) + 'px';
      }
    }
  }

  function toggleHeatmap() {
    leftVisible = !leftVisible;
    document.getElementById('left').classList.toggle('hidden', !leftVisible);
    document.getElementById('btn-toggle-heatmap').style.opacity = leftVisible ? '1' : '0.45';
  }

  function switchHeatmap(type) {
    currentType = type;
    document.getElementById('btn-delta').className   = type === 'delta'   ? 'active' : '';
    document.getElementById('btn-winner').className  = type === 'winner'  ? 'active' : '';
    document.getElementById('btn-scatter').className = type === 'scatter' ? 'active' : '';
    const isScatter = type === 'scatter';
    document.getElementById('heatmap-img').style.display  = isScatter ? 'none'  : 'block';
    document.getElementById('scatter-svg').style.display  = isScatter ? 'block' : 'none';
    document.getElementById('zoom-hint').textContent = isScatter
      ? 'click to select sequence'
      : 'scroll to zoom \u00b7 click to select';
    document.getElementById('attr-color-field').style.display =
      (isScatter && attrData.attributes.length > 0) ? '' : 'none';
    updateWinnerControlsVisibility();
    if (isScatter) {
      renderScatter();
      if (heatmapMeta)
        document.getElementById('heatmap-status').textContent = heatmapMeta.n_seqs + ' sequences';
    } else {
      loadHeatmap(type);
    }
  }

  function changeSortMode() {
    currentSort = document.getElementById('sort-mode').value;
    loadHeatmap(currentType);
  }

  function getThresholdValue(presetId, customId, fallback) {
    const preset = document.getElementById(presetId).value;
    if (preset !== 'custom') return parseFloat(preset);
    const val = parseFloat(document.getElementById(customId).value);
    return Number.isFinite(val) ? val : fallback;
  }

  function winnerThresholds() {
    const goodRaw = getThresholdValue('good-preset', 'good-custom', DEFAULT_GOOD);
    const badRaw = getThresholdValue('bad-preset', 'bad-custom', DEFAULT_BAD);
    const good = Math.max(0.0, goodRaw);
    const bad = Math.max(good + 0.01, badRaw);
    return { good: good, bad: bad };
  }

  function updateWinnerLegend() {
    const thr = winnerThresholds();
    document.getElementById('legend-both-good').textContent =
      'both ≤ ' + thr.good.toFixed(0) + 'px';
    document.getElementById('legend-both-bad').textContent =
      'both ≥ ' + thr.bad.toFixed(0) + 'px';
  }

  function onWinnerSettingChange(reload=true) {
    const goodPreset = document.getElementById('good-preset').value;
    const badPreset = document.getElementById('bad-preset').value;
    document.getElementById('good-custom').disabled = goodPreset !== 'custom' || currentType !== 'winner';
    document.getElementById('bad-custom').disabled = badPreset !== 'custom' || currentType !== 'winner';
    updateWinnerLegend();
    if (reload && currentType === 'winner') loadHeatmap(currentType);
  }

  function updateWinnerControlsVisibility() {
    const show = currentType === 'winner';
    document.getElementById('good-preset').disabled = !show;
    document.getElementById('bad-preset').disabled = !show;
    document.getElementById('sort-mode').disabled = currentType === 'scatter';
    onWinnerSettingChange(false);
  }

  function heatmapQueryParams(type) {
    const params = new URLSearchParams();
    params.set('sort_mode', currentSort);
    if (type === 'winner') {
      const thr = winnerThresholds();
      params.set('good_thr', String(thr.good));
      params.set('bad_thr', String(thr.bad));
    }
    return params.toString();
  }

  async function loadHeatmap(type) {
    document.getElementById('heatmap-status').textContent = 'Loading…';
    const query = heatmapQueryParams(type);
    const res = await fetch('/heatmaps/' + type + '/meta?' + query);
    if (!res.ok) {
      document.getElementById('heatmap-status').textContent = 'Error';
      return;
    }
    heatmapMeta = await res.json();
    heatmapZoom = 1.0;
    applyZoom();
    const img = document.getElementById('heatmap-img');
    img.src = '/heatmaps/' + type + '.png?' + query + '&t=' + Date.now();
    img.onload = () => {
      document.getElementById('heatmap-status').textContent =
        heatmapMeta.n_seqs + ' × ' + heatmapMeta.max_T + ' · ' + currentSort;
    };
  }

  const heatmapWrap = document.getElementById('heatmap-wrap');
  const heatmapImg = document.getElementById('heatmap-img');
  const tooltip = document.getElementById('tooltip');

  heatmapWrap.addEventListener('wheel', e => {
    e.preventDefault();
    const wrap = heatmapWrap;
    const rect = heatmapImg.getBoundingClientRect();
    const scrollX = wrap.scrollLeft + (e.clientX - wrap.getBoundingClientRect().left);

    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    const newZoom = Math.max(1.0, Math.min(40.0, heatmapZoom * factor));
    const ratio = newZoom / heatmapZoom;
    heatmapZoom = newZoom;
    applyZoom();
    wrap.scrollLeft = scrollX * ratio - (e.clientX - wrap.getBoundingClientRect().left);
  }, { passive: false });

  function applyZoom() {
    heatmapImg.style.width = (heatmapZoom * 100) + '%';
  }

  function heatmapCoords(e) {
    if (!heatmapMeta) return null;
    const rect = heatmapImg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / rect.width * heatmapMeta.max_T);
    const row = Math.floor(y / rect.height * heatmapMeta.n_seqs);
    if (row < 0 || row >= heatmapMeta.n_seqs || col < 0) return null;
    const seqName = heatmapMeta.seq_names[row];
    const T = heatmapMeta.row_lengths[row];
    const frame = Math.max(0, Math.min(col, T - 1));
    return { row, frame, seqName, T };
  }

  heatmapImg.addEventListener('mousemove', e => {
    const c = heatmapCoords(e);
    if (!c) {
      tooltip.style.display = 'none';
      return;
    }
    const p15A = heatmapMeta.p15_a[c.row];
    const p15B = heatmapMeta.p15_b[c.row];
    const dP15 = heatmapMeta.delta_p15[c.row];
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.textContent = c.seqName + '\n' +
      'Frame ' + c.frame + ' / ' + (c.T - 1) + '\n' +
      methodInfo.label_a + ' P@15: ' + p15A.toFixed(1) + '%\n' +
      methodInfo.label_b + ' P@15: ' + p15B.toFixed(1) + '%\n' +
      'ΔP@15 (B-A): ' + dP15.toFixed(1) + '%';
  });

  heatmapImg.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

  heatmapImg.addEventListener('click', e => {
    const c = heatmapCoords(e);
    if (!c) return;
    selectSequence(c.seqName, c.frame);
    if (leftVisible) toggleHeatmap();
  });

  async function selectSequence(seqName, startFrame) {
    document.getElementById('seq-name').textContent = seqName;
    document.getElementById('seq-stats').textContent = 'Loading…';
    const res = await fetch('/api/sequence/' + encodeURIComponent(seqName));
    if (!res.ok) {
      document.getElementById('seq-stats').textContent = 'Error';
      return;
    }
    seqData = await res.json();
    currentSeq = seqName;
    if (currentType === 'scatter') renderScatter();

    if (_restoringFromHistory) {
      _restoringFromHistory = false;
    } else {
      history.pushState({ seq: seqName, frame: startFrame || 0 }, '',
                        '?seq=' + encodeURIComponent(seqName) + '&frame=' + (startFrame || 0));
    }

    document.getElementById('seq-stats').textContent =
      methodInfo.label_a + ': P@15 ' + seqData.p15_a.toFixed(1) + '% · mean ' + seqData.mean_a.toFixed(2) + 'px\n' +
      methodInfo.label_b + ': P@15 ' + seqData.p15_b.toFixed(1) + '% · mean ' + seqData.mean_b.toFixed(2) + 'px\n' +
      'ΔP@15 (B-A): ' + seqData.delta_p15.toFixed(1) + '% · Δmean (A-B): ' + seqData.mean_delta.toFixed(2) + 'px';

    frameEvalMap = {};
    seqData.eval_frame_idx.forEach((fi, i) => { frameEvalMap[fi] = i; });
    document.getElementById('frame-slider').max = seqData.num_frames - 1;
    drawTimeline();
    goToFrame(startFrame || 0);
  }

  async function goToFrame(idx) {
    if (!currentSeq || !seqData) return;
    currentFrame = Math.max(0, Math.min(idx, seqData.num_frames - 1));
    history.replaceState({ seq: currentSeq, frame: currentFrame }, '',
                         '?seq=' + encodeURIComponent(currentSeq) + '&frame=' + currentFrame);
    document.getElementById('frame-slider').value = currentFrame;
    document.getElementById('frame-num').textContent =
      currentFrame + ' / ' + (seqData.num_frames - 1);
    updateFrameInfo(currentFrame);
    drawTimeline();

    const showA = document.getElementById('chk-a').checked ? 1 : 0;
    const showB = document.getElementById('chk-b').checked ? 1 : 0;
    const showGT = document.getElementById('chk-gt').checked ? 1 : 0;
    const url = '/api/frame/' + encodeURIComponent(currentSeq) + '/' + currentFrame +
                '?a=' + showA + '&b=' + showB + '&gt=' + showGT + '&t=' + Date.now();

    const frameImg = document.getElementById('frame-img');
    document.getElementById('no-seq-msg').style.display = 'none';
    frameImg.style.display = 'block';
    frameImg.src = url;
  }

  function updateFrameInfo(frameIdx) {
    const infoEl = document.getElementById('frame-info');
    const corr = correctionNote(frameIdx);
    const pos = frameEvalMap[frameIdx];
    if (pos === undefined || !seqData) {
      infoEl.textContent = 'frame ' + frameIdx + ' (not evaluated)' + corr;
      return;
    }

    const eA = seqData.errors_a[pos];
    const eB = seqData.errors_b[pos];
    const d = seqData.delta[pos];
    let winner = 'tie';
    if (d > WIN_MARGIN) winner = methodInfo.label_b + ' better';
    else if (d < -WIN_MARGIN) winner = methodInfo.label_a + ' better';

    infoEl.textContent =
      methodInfo.label_a + ': ' + eA.toFixed(2) + 'px · ' +
      methodInfo.label_b + ': ' + eB.toFixed(2) + 'px · ' +
      'Δ(' + methodInfo.label_a + '-' + methodInfo.label_b + '): ' + d.toFixed(2) + 'px · ' + winner + corr;
  }

  function correctionNote(frameIdx) {
    if (!seqData) return '';
    const inA = (seqData.corrected_a || []).includes(frameIdx);
    const inB = (seqData.corrected_b || []).includes(frameIdx);
    if (!inA && !inB) return '';
    const who = [inA ? methodInfo.label_a : null, inB ? methodInfo.label_b : null]
      .filter(Boolean).join(' + ');
    return ' · mask corrected: ' + who;
  }

  function openDetail(label) {
    if (!currentSeq) return;
    window.open('/report-' + label + '/?seq=' + encodeURIComponent(currentSeq) + '&frame=' + currentFrame, '_blank');
  }

  function copyLocation() {
    if (!currentSeq) return;
    navigator.clipboard.writeText(currentSeq + ' ' + currentFrame);
    const btn = document.getElementById('btn-copy-loc');
    const prev = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = prev; }, 1200);
  }

  function navFrame(delta) {
    if (currentSeq) goToFrame(currentFrame + delta);
  }

  function refreshFrame() {
    if (currentSeq) goToFrame(currentFrame);
  }

  function jumpWinner(sign) {
    if (!seqData) return;
    const target = sign > 0 ? 1 : -1;
    const start = currentFrame + 1;
    for (let f = start; f < seqData.num_frames; f++) {
      const pos = frameEvalMap[f];
      if (pos === undefined) continue;
      const d = seqData.delta[pos];
      if ((target > 0 && d > WIN_MARGIN) || (target < 0 && d < -WIN_MARGIN)) {
        goToFrame(f);
        return;
      }
    }
  }

  const timelineCanvas = document.getElementById('timeline-canvas');
  const timelineWrap = document.getElementById('timeline-wrap');

  timelineCanvas.addEventListener('click', e => {
    if (!seqData) return;
    const rect = timelineCanvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    goToFrame(Math.round(frac * (seqData.num_frames - 1)));
  });

  new ResizeObserver(() => { if (seqData) drawTimeline(); }).observe(timelineWrap);

  function drawTimeline() {
    if (!seqData) return;
    const W = timelineWrap.clientWidth || 700;
    const H = 72;
    timelineCanvas.width = W;
    timelineCanvas.height = H;
    const ctx = timelineCanvas.getContext('2d');

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);

    const T = seqData.num_frames;
    if (!T) return;

    const denseDelta = new Float32Array(T).fill(NaN);
    seqData.eval_frame_idx.forEach((fi, i) => { denseDelta[fi] = seqData.delta[i]; });

    const maxAbs = Math.max(5.0, Math.max(...seqData.delta.map(v => Math.abs(v))));
    const mid = H / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(W, mid);
    ctx.stroke();

    const barW = W / T;
    for (let f = 0; f < T; f++) {
      const d = denseDelta[f];
      if (!Number.isFinite(d)) continue;
      const mag = Math.min(Math.abs(d) / maxAbs, 1.0);
      const h = Math.max(1, mag * (H * 0.48));
      const x = f / T * W;
      if (d >= 0) {
        ctx.fillStyle = `rgb(${Math.round(255*mag)},${Math.round(80*(1-mag))},${Math.round(80*(1-mag))})`;
        ctx.fillRect(x, mid - h, Math.max(1, barW + 0.5), h);
      } else {
        ctx.fillStyle = `rgb(${Math.round(80*(1-mag))},${Math.round(130*(1-mag))},${Math.round(255*mag)})`;
        ctx.fillRect(x, mid, Math.max(1, barW + 0.5), h);
      }
    }

    // Mask-correction markers: A along the top edge, B along the bottom edge.
    function drawCorrections(frames, color, atTop) {
      if (!frames || !frames.length) return;
      ctx.fillStyle = color;
      const mh = 6;
      for (const f of frames) {
        if (f < 0 || f >= T) continue;
        const x = (f + 0.5) / T * W;
        ctx.beginPath();
        if (atTop) {
          ctx.moveTo(x, 0); ctx.lineTo(x - 3, mh); ctx.lineTo(x + 3, mh);
        } else {
          ctx.moveTo(x, H); ctx.lineTo(x - 3, H - mh); ctx.lineTo(x + 3, H - mh);
        }
        ctx.closePath();
        ctx.fill();
      }
    }
    drawCorrections(seqData.corrected_a, '#ff7a00', true);
    drawCorrections(seqData.corrected_b, '#39ff14', false);

    const cx = T > 1 ? (currentFrame / (T - 1)) * W : 0;
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, H);
    ctx.stroke();
  }

  function renderScatter() {
    if (!heatmapMeta || !heatmapMeta.seq_names) return;
    const svg = document.getElementById('scatter-svg');
    svg.innerHTML = '';

    const W = 420, H = 420;
    const M = { top: 28, right: 18, bottom: 52, left: 54 };
    const IW = W - M.left - M.right;
    const IH = H - M.top - M.bottom;

    function toX(v) { return M.left + (v / 100) * IW; }
    function toY(v) { return M.top  + (1 - v / 100) * IH; }

    function el(tag, attrs, text) {
      const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
      if (text !== undefined) e.textContent = text;
      return e;
    }

    svg.appendChild(el('rect', { x: 0, y: 0, width: W, height: H, fill: '#1e1e1e' }));
    svg.appendChild(el('rect', { x: M.left, y: M.top, width: IW, height: IH, fill: '#111' }));

    for (let v = 0; v <= 100; v += 20) {
      const x = toX(v), y = toY(v);
      svg.appendChild(el('line', { x1: x, y1: M.top, x2: x, y2: M.top + IH, stroke: '#2a2a2a', 'stroke-width': 1 }));
      svg.appendChild(el('line', { x1: M.left, y1: y, x2: M.left + IW, y2: y, stroke: '#2a2a2a', 'stroke-width': 1 }));
      svg.appendChild(el('text', { x: x, y: M.top + IH + 16, 'text-anchor': 'middle', fill: '#666', 'font-size': 11 }, String(v)));
      svg.appendChild(el('text', { x: M.left - 5, y: y + 4, 'text-anchor': 'end', fill: '#666', 'font-size': 11 }, String(v)));
    }

    svg.appendChild(el('line', {
      x1: toX(0), y1: toY(0), x2: toX(100), y2: toY(100),
      stroke: '#444', 'stroke-width': 1, 'stroke-dasharray': '4 3'
    }));

    const lblA = methodInfo ? methodInfo.label_a : 'A';
    const lblB = methodInfo ? methodInfo.label_b : 'B';
    svg.appendChild(el('text', {
      x: M.left + IW / 2, y: H - 6, 'text-anchor': 'middle', fill: '#aaa', 'font-size': 12
    }, lblA + ' p@15 (%)'));
    const yLabel = el('text', {
      x: 13, y: M.top + IH / 2, 'text-anchor': 'middle', fill: '#aaa', 'font-size': 12,
      transform: 'rotate(-90 13 ' + (M.top + IH / 2) + ')'
    }, lblB + ' p@15 (%)');
    svg.appendChild(yLabel);

    const selectedAttr = document.getElementById('attr-color-select').value;

    const n = heatmapMeta.seq_names.length;
    for (let i = 0; i < n; i++) {
      const pa = heatmapMeta.p15_a[i];
      const pb = heatmapMeta.p15_b[i];
      const name = heatmapMeta.seq_names[i];
      if (!Number.isFinite(pa) || !Number.isFinite(pb)) continue;

      let color;
      if (selectedAttr) {
        const flags = attrData.annotations[name];
        if (!flags || !(selectedAttr in flags)) {
          color = '#555';  // unannotated
        } else {
          color = flags[selectedAttr] ? '#e040fb' : '#aaa';
        }
      } else {
        color = pa > pb ? '#ff7a00' : (pb > pa ? '#4caf50' : '#888');
      }

      const isSel = name === currentSeq;
      const c = el('circle', {
        cx: toX(pa), cy: toY(pb), r: isSel ? 7 : 4,
        fill: color, 'fill-opacity': isSel ? 1 : 0.65,
        stroke: isSel ? '#fff' : 'none', 'stroke-width': isSel ? 2 : 0
      });
      c.style.cursor = 'pointer';
      c.dataset.name = name;
      c.dataset.pa = pa;
      c.dataset.pb = pb;
      if (selectedAttr) {
        const flags = attrData.annotations[name];
        c.dataset.attrVal = (!flags || !(selectedAttr in flags)) ? 'unknown'
          : flags[selectedAttr] ? 'yes' : 'no';
      }
      svg.appendChild(c);
    }

    svg.onmousemove = e => {
      const t = e.target;
      if (!t.dataset || !t.dataset.name) { tooltip.style.display = 'none'; return; }
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top  = (e.clientY - 10) + 'px';
      const pa = parseFloat(t.dataset.pa), pb = parseFloat(t.dataset.pb);
      const selAttr = document.getElementById('attr-color-select').value;
      let tip = t.dataset.name + '\n' +
        lblA + ' p@15: ' + pa.toFixed(1) + '%\n' +
        lblB + ' p@15: ' + pb.toFixed(1) + '%\n' +
        '\u0394p@15 (B\u2212A): ' + (pb - pa).toFixed(1) + '%';
      if (selAttr && t.dataset.attrVal !== undefined)
        tip += '\n' + selAttr + ': ' + t.dataset.attrVal;
      tooltip.textContent = tip;
    };
    svg.onmouseleave = () => { tooltip.style.display = 'none'; };
    svg.onclick = e => {
      const t = e.target;
      if (!t.dataset || !t.dataset.name) return;
      selectSequence(t.dataset.name, 0);
      if (leftVisible) toggleHeatmap();
    };
  }

  boot();
</script>
</body>
</html>
"""


class _State:
    dataset: str = ""
    base_dir: Optional[Path] = None
    results_dir_a: Optional[Path] = None
    results_dir_b: Optional[Path] = None
    label_a: str = "Method A"
    label_b: str = "Method B"
    load_gt_fn = None

    seq_cache: Dict[str, dict] = {}
    image_path_cache: Dict[str, Optional[List[Path]]] = {}
    loaded_sequences: List[dict] = []

    attributes: List[str] = []
    annotations: Dict[str, Dict[str, bool]] = {}
    use_best_rotation: bool = False

    detail_procs: Dict[str, subprocess.Popen] = {}
    detail_ports: Dict[str, int] = {}


_st = _State()
app = FastAPI(title="FlatSAM2 Method Compare Viewer")


# ── detail server helpers ──────────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _infer_config_from_results_dir(results_dir: Path) -> Optional[Path]:
    """Try to find configs/tracker/<stem>.py from a results directory."""
    stem = results_dir.parent.name
    candidate = Path(__file__).parent / "configs" / "tracker" / f"{stem}.py"
    return candidate if candidate.exists() else None


def _spawn_detail_server(
    label: str,
    config_path: Optional[Path],
    results_dir: Path,
    split: Optional[str],
    output_path: Path,
) -> int:
    if config_path is None:
        config_path = _infer_config_from_results_dir(results_dir)
        if config_path is not None:
            print(f"  Inferred config for {label}: {config_path}")
        else:
            print(f"  WARNING: could not infer config for {label}, Hough params will use defaults")
    port = _find_free_port()
    script = Path(__file__).parent / "experiments" / "pti_sam3_mask_homography_report.py"
    cmd = [sys.executable, str(script)]
    cmd += ["--tracker-results-dir", str(results_dir)]
    if config_path is not None:
        cmd += ["--tracker-config", str(config_path)]
    cmd += ["--host", "127.0.0.1", "--port", str(port), "--output", str(output_path)]
    if split:
        cmd += ["--split", split]
    print(f"  Spawning detail server {label} on port {port}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
    _st.detail_procs[label] = proc
    _st.detail_ports[label] = port
    return port


def _wait_ready(label: str, port: int, timeout: float = 120.0) -> None:
    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = _st.detail_procs.get(label)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Detail server {label} exited early (returncode={proc.returncode})")
        try:
            urllib.request.urlopen(url, timeout=3.0)
            print(f"  Detail server {label} ready.")
            return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"Detail server {label} at port {port} did not start within {timeout}s")


def _rewrite_html(content: bytes, prefix: str, label: str, color: str) -> bytes:
    badge = (
        f'<div style="position:fixed;top:6px;right:8px;background:{color};color:#000;'
        f'padding:2px 10px;font-family:monospace;font-size:13px;font-weight:bold;'
        f'border-radius:4px;z-index:99999;opacity:0.92;pointer-events:none">'
        f'{label}</div>'
    ).encode()
    content = (
        content
        .replace(b"'/api/", f"'/{prefix}/api/".encode())
        .replace(b'"/api/', f'"/{prefix}/api/'.encode())
        .replace(b"</body>", badge + b"</body>", 1)
    )
    return content


@app.api_route("/report-{label}/{path:path}", methods=["GET", "POST"])
async def _detail_proxy(label: str, path: str, request: Request) -> Response:
    port = _st.detail_ports.get(label)
    if port is None:
        raise HTTPException(503, f"Detail server '{label}' not running")
    qs = request.url.query
    target = f"http://127.0.0.1:{port}/{path}" + (f"?{qs}" if qs else "")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            upstream = await client.request(
                method=request.method,
                url=target,
                content=await request.body(),
                headers={k: v for k, v in request.headers.items()
                         if k.lower() not in ("host", "content-length")},
            )
    except httpx.RequestError as exc:
        raise HTTPException(502, f"Detail server error: {exc}")
    content = upstream.content
    ct = upstream.headers.get("content-type", "")
    if "text/html" in ct:
        method_label = _st.label_a if label == "a" else _st.label_b
        color = "#ff7a00" if label == "a" else "#39ff14"
        content = _rewrite_html(content, f"report-{label}", method_label, color)
    skip = {"content-encoding", "content-length", "transfer-encoding"}
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in skip}
    return Response(content=content, status_code=upstream.status_code,
                    headers=headers, media_type=ct.split(";")[0] or None)


def _p_at_k(errors: np.ndarray, k: float = THRESHOLD) -> float:
    if len(errors) == 0:
        return float("nan")
    return 100.0 * float(np.mean(errors <= k))


def _image_paths(seq_name: str) -> Optional[List[Path]]:
    if seq_name in _st.image_path_cache:
        return _st.image_path_cache[seq_name]
    try:
        if DATASETS[_st.dataset].gt_loader == "pot":
            result = pot_image_paths(_st.base_dir, seq_name)
        else:
            result = pt_image_paths(_st.base_dir, seq_name)
        _st.image_path_cache[seq_name] = result
        return result
    except Exception:
        _st.image_path_cache[seq_name] = None
        return None


def _corrected_frames(results_dir: Path, seq_name: str) -> List[int]:
    """Frame indices where a SAM3 mask self-correction fired this frame.

    Reads the ``corrected`` / ``frame_idx`` arrays from ``{seq_name}_debug.npz``
    (written by run.py). Empty when no sidecar or no correction events — e.g. a
    config without ``correction.enabled``.

    Args:
        results_dir: Per-method results directory holding the debug sidecar.
        seq_name: Sequence name.

    Returns:
        Sorted 0-based frame indices where the mask was corrected.
    """
    debug = _load_debug(results_dir, seq_name)
    if debug is None or "corrected" not in debug or "frame_idx" not in debug:
        return []
    corrected = debug["corrected"].astype(bool)
    frame_idx = debug["frame_idx"].astype(int)
    n = min(len(corrected), len(frame_idx))
    return [int(f) for f, c in zip(frame_idx[:n], corrected[:n]) if c]


def _build_sequence_data(
    seq_name: str,
    gt: np.ndarray,
    valid: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:
    """Build per-sequence comparison tensors.

    Args:
        seq_name: Sequence name.
        gt: Ground-truth corners with shape (T, 2, 4).
        valid: Valid-frame mask with shape (T,).
        pred_a: Method-A corners with shape (T, 2, 4).
        pred_b: Method-B corners with shape (T, 2, 4).

    Returns:
        Dict with sequence metrics and per-evaluated-frame arrays.
    """
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False
    eval_idx = np.where(eval_mask)[0]

    if len(eval_idx) == 0:
        return {
            "name": seq_name,
            "num_frames": len(gt),
            "eval_frame_idx": eval_idx,
            "errors_a": np.array([]),
            "errors_b": np.array([]),
            "delta": np.array([]),
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "mean_delta": float("nan"),
            "p15_a": float("nan"),
            "p15_b": float("nan"),
            "delta_p15": float("nan"),
        }

    if _st.use_best_rotation:
        errors_a, _ = best_rotation_error(gt[eval_idx], pred_a[eval_idx])
        errors_b, _ = best_rotation_error(gt[eval_idx], pred_b[eval_idx])
    else:
        errors_a = alignment_error(gt[eval_idx], pred_a[eval_idx])
        errors_b = alignment_error(gt[eval_idx], pred_b[eval_idx])
    delta = errors_a - errors_b

    p15_a = _p_at_k(errors_a)
    p15_b = _p_at_k(errors_b)
    return {
        "name": seq_name,
        "num_frames": len(gt),
        "eval_frame_idx": eval_idx,
        "errors_a": errors_a,
        "errors_b": errors_b,
        "delta": delta,
        "mean_a": float(np.mean(errors_a)),
        "mean_b": float(np.mean(errors_b)),
        "mean_delta": float(np.mean(delta)),
        "p15_a": float(p15_a),
        "p15_b": float(p15_b),
        "delta_p15": float(p15_b - p15_a),
    }


def _hex_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def _draw_poly(
    img: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    dashed: bool = False,
) -> None:
    """Draw a closed quadrilateral from (2, 4) corners."""
    if dashed:
        n = corners.shape[1]
        for i in range(n):
            p1 = (int(corners[0, i]), int(corners[1, i]))
            p2 = (int(corners[0, (i + 1) % n]), int(corners[1, (i + 1) % n]))
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = max(1, int(np.hypot(dx, dy)))
            dash = 8
            gap = 6
            for s in range(0, dist, dash + gap):
                e = min(s + dash, dist)
                xs = p1[0] + dx * s // dist
                ys = p1[1] + dy * s // dist
                xe = p1[0] + dx * e // dist
                ye = p1[1] + dy * e // dist
                cv2.line(img, (xs, ys), (xe, ye), color, thickness)
        cv2.circle(img, (int(corners[0, 0]), int(corners[1, 0])), 7, color, -1)
        return

    pts = rearrange(corners, "xy c -> c 1 xy").astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    for corner_i, (x, y) in enumerate(corners.T):
        radius = 7 if corner_i == 0 else 4
        cv2.circle(img, (int(x), int(y)), radius, color, -1)


def _placeholder(
    text: str = "Image not available",
    h: int = 480,
    w: int = 640,
) -> np.ndarray:
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(
            img,
            line,
            (20, h // 2 - 10 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1,
        )
    return img


def render_frame(
    seq_name: str,
    frame_idx: int,
    show_a: bool = True,
    show_b: bool = True,
    show_gt: bool = True,
) -> bytes:
    """Render a frame with GT + method overlays and return JPEG bytes."""
    seq = _st.seq_cache.get(seq_name)
    if seq is None:
        raise ValueError(f"Unknown sequence: {seq_name}")

    img: Optional[np.ndarray] = None
    paths_list = _image_paths(seq_name)
    if paths_list and frame_idx < len(paths_list):
        p = paths_list[frame_idx]
        if p.exists():
            img = cv2.imread(str(p))
    if img is None:
        msg = f"{seq_name}  frame {frame_idx}"
        if paths_list is None:
            msg += "\n(dataset not mounted)"
        img = _placeholder(msg)

    gt = seq["gt"]
    pred_a = seq["pred_a"]
    pred_b = seq["pred_b"]
    valid = seq["valid"]

    if show_gt and frame_idx < len(gt) and valid[frame_idx]:
        _draw_poly(img, gt[frame_idx], (255, 255, 255), 2, dashed=True)

    if show_a and frame_idx < len(pred_a):
        _draw_poly(img, pred_a[frame_idx], _hex_bgr("#ff7a00"), 2)

    if show_b and frame_idx < len(pred_b):
        _draw_poly(img, pred_b[frame_idx], _hex_bgr("#39ff14"), 2)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf)


def _delta_to_rgb(values: np.ndarray, vmax: float = 20.0) -> np.ndarray:
    """Map signed deltas to a blue-white-red palette.

    Positive values mean method B has lower error, negative values mean method A
    has lower error.
    """
    clipped = np.clip(values / vmax, -1.0, 1.0)
    out = np.empty((len(values), 3), dtype=np.uint8)

    pos = clipped >= 0
    neg = ~pos

    if np.any(pos):
        t = clipped[pos][:, None]
        low = np.array([0xF7, 0xF7, 0xF7], dtype=np.float32)
        high = np.array([0xB4, 0x04, 0x26], dtype=np.float32)
        rgb = (1.0 - t) * low + t * high
        out[pos] = rgb.astype(np.uint8)

    if np.any(neg):
        t = (-clipped[neg])[:, None]
        low = np.array([0xF7, 0xF7, 0xF7], dtype=np.float32)
        high = np.array([0x3B, 0x4C, 0xC0], dtype=np.float32)
        rgb = (1.0 - t) * low + t * high
        out[neg] = rgb.astype(np.uint8)

    return out


def _sort_sequences(sequences: List[dict], sort_mode: str) -> List[dict]:
    """Sort sequence rows according to the selected comparison mode."""
    if sort_mode == "gap":
        return sorted(
            sequences,
            key=lambda s: (
                -abs(float(s["mean_delta"])),
                float(0.5 * (s["p15_a"] + s["p15_b"])),
            ),
        )

    return sorted(
        sequences,
        key=lambda s: (
            float(0.5 * (s["p15_a"] + s["p15_b"])),
            -float(0.5 * (s["mean_a"] + s["mean_b"])),
            -abs(float(s["mean_delta"])),
        ),
    )


def _build_heatmap(
    htype: str,
    sorted_seqs: List[dict],
    good_thr: float,
    bad_thr: float,
) -> Tuple[bytes, dict]:
    """Build a raw-pixel PNG + metadata dict for heatmap type."""
    n_seqs = len(sorted_seqs)
    max_t = max(len(s["delta"]) for s in sorted_seqs)
    pad_rgb = (0xCC, 0xCC, 0xCC)
    img = np.full((n_seqs, max_t, 3), pad_rgb, dtype=np.uint8)

    for row, seq in enumerate(sorted_seqs):
        t = len(seq["delta"])
        if t == 0:
            continue
        delta = seq["delta"]
        if htype == "delta":
            img[row, :t] = _delta_to_rgb(delta, vmax=20.0)
            continue

        errors_a = seq["errors_a"]
        errors_b = seq["errors_b"]
        row_rgb = np.full((t, 3), (0x8A, 0x8A, 0x8A), dtype=np.uint8)
        both_good = (errors_a <= good_thr) & (errors_b <= good_thr)
        both_bad = (errors_a >= bad_thr) & (errors_b >= bad_thr)
        a_better = (~both_good) & (~both_bad) & (errors_a <= errors_b)
        b_better = (~both_good) & (~both_bad) & (errors_a > errors_b)
        row_rgb[both_good] = (0x17, 0xA9, 0x74)
        row_rgb[both_bad] = (0x7F, 0x1D, 0x1D)
        row_rgb[a_better] = (0xFF, 0x7A, 0x00)
        row_rgb[b_better] = (0x39, 0xFF, 0x14)
        img[row, :t] = row_rgb

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    png_bytes = bytes(buf)

    meta = {
        "seq_names": [s["name"] for s in sorted_seqs],
        "row_lengths": [len(s["delta"]) for s in sorted_seqs],
        "max_T": max_t,
        "n_seqs": n_seqs,
        "p15_a": [float(s["p15_a"]) for s in sorted_seqs],
        "p15_b": [float(s["p15_b"]) for s in sorted_seqs],
        "delta_p15": [float(s["delta_p15"]) for s in sorted_seqs],
    }
    return png_bytes, meta


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/info")
async def api_info() -> JSONResponse:
    return JSONResponse(
        {
            "label_a": _st.label_a,
            "label_b": _st.label_b,
            "default_good_thr": DEFAULT_GOOD_THR,
            "default_bad_thr": DEFAULT_BAD_THR,
            "sort_modes": list(SORT_MODES),
            "use_best_rotation": _st.use_best_rotation,
            "has_detail": bool(_st.detail_ports),
        }
    )


@app.post("/api/best-rotation/{enabled}")
async def api_set_best_rotation(enabled: int) -> JSONResponse:
    _st.use_best_rotation = bool(enabled)
    for seq in _st.loaded_sequences:
        updated = _build_sequence_data(seq["name"], seq["gt"], seq["valid"], seq["pred_a"], seq["pred_b"])
        seq.update(updated)
    return JSONResponse({"use_best_rotation": _st.use_best_rotation})


@app.get("/api/attributes")
async def api_attributes() -> JSONResponse:
    return JSONResponse({"attributes": _st.attributes, "annotations": _st.annotations})


def _resolve_heatmap_params(
    sort_mode: str,
    good_thr: float,
    bad_thr: float,
) -> Tuple[str, float, float]:
    """Validate and sanitize heatmap query parameters."""
    selected_sort = sort_mode if sort_mode in SORT_MODES else "severity"
    safe_good = max(0.0, float(good_thr))
    safe_bad = max(safe_good + 1e-3, float(bad_thr))
    return selected_sort, safe_good, safe_bad


def _render_heatmap_payload(
    htype: str,
    sort_mode: str,
    good_thr: float,
    bad_thr: float,
) -> Tuple[bytes, dict]:
    """Render heatmap PNG and metadata for selected query params."""
    if htype not in ("delta", "winner"):
        raise HTTPException(404, f"Unknown heatmap type: {htype}")

    selected_sort, safe_good, safe_bad = _resolve_heatmap_params(
        sort_mode=sort_mode,
        good_thr=good_thr,
        bad_thr=bad_thr,
    )
    sorted_seqs = _sort_sequences(_st.loaded_sequences, selected_sort)
    png_bytes, meta = _build_heatmap(
        htype=htype,
        sorted_seqs=sorted_seqs,
        good_thr=safe_good,
        bad_thr=safe_bad,
    )
    meta["sort_mode"] = selected_sort
    meta["good_thr"] = safe_good
    meta["bad_thr"] = safe_bad
    return png_bytes, meta


@app.get("/heatmaps/{htype}.png")
async def heatmap_png(
    htype: str,
    sort_mode: str = "severity",
    good_thr: float = DEFAULT_GOOD_THR,
    bad_thr: float = DEFAULT_BAD_THR,
) -> StreamingResponse:
    png, _ = _render_heatmap_payload(
        htype=htype,
        sort_mode=sort_mode,
        good_thr=good_thr,
        bad_thr=bad_thr,
    )
    return StreamingResponse(iter([png]), media_type="image/png")


@app.get("/heatmaps/{htype}/meta")
async def heatmap_meta(
    htype: str,
    sort_mode: str = "severity",
    good_thr: float = DEFAULT_GOOD_THR,
    bad_thr: float = DEFAULT_BAD_THR,
) -> JSONResponse:
    _, meta = _render_heatmap_payload(
        htype=htype,
        sort_mode=sort_mode,
        good_thr=good_thr,
        bad_thr=bad_thr,
    )
    return JSONResponse(meta)


@app.get("/api/sequence/{name}")
async def api_sequence(name: str) -> JSONResponse:
    seq = _st.seq_cache.get(name)
    if seq is None:
        raise HTTPException(404, f"Sequence not found: {name}")

    out = {
        "name": seq["name"],
        "num_frames": int(seq["num_frames"]),
        "eval_frame_idx": [int(v) for v in seq["eval_frame_idx"]],
        "errors_a": [float(v) for v in seq["errors_a"]],
        "errors_b": [float(v) for v in seq["errors_b"]],
        "delta": [float(v) for v in seq["delta"]],
        "mean_a": float(seq["mean_a"]),
        "mean_b": float(seq["mean_b"]),
        "mean_delta": float(seq["mean_delta"]),
        "p15_a": float(seq["p15_a"]),
        "p15_b": float(seq["p15_b"]),
        "delta_p15": float(seq["delta_p15"]),
        "corrected_a": [int(f) for f in seq.get("corrected_a", [])],
        "corrected_b": [int(f) for f in seq.get("corrected_b", [])],
    }
    return JSONResponse(out)


@app.get("/api/frame/{name}/{frame_idx}")
async def api_frame(
    name: str,
    frame_idx: int,
    a: int = 1,
    b: int = 1,
    gt: int = 1,
) -> StreamingResponse:
    try:
        jpeg = render_frame(
            seq_name=name,
            frame_idx=frame_idx,
            show_a=bool(a),
            show_b=bool(b),
            show_gt=bool(gt),
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


def _save_dominance_plot(
    sequences: List[dict],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Save a per-sequence dominance-curve plot comparing two methods on p@15.

    Splits sequences into those where A wins and those where B wins, then plots
    two non-increasing curves of |p15_A − p15_B|.

    Args:
        sequences: List of sequence data dicts containing 'p15_a' and 'p15_b'.
        label_a: Display label for method A.
        label_b: Display label for method B.
        output_path: Path to write the PNG figure.
    """
    a_better: List[float] = []
    b_better: List[float] = []

    for s in sequences:
        p_a = s.get("p15_a", float("nan"))
        p_b = s.get("p15_b", float("nan"))
        if np.isnan(p_a) or np.isnan(p_b):
            continue
        diff = abs(p_a - p_b)
        if p_a > p_b:
            a_better.append(diff)
        elif p_b > p_a:
            b_better.append(diff)

    a_better.sort(reverse=True)
    b_better.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    if a_better:
        ax.plot(
            range(1, len(a_better) + 1),
            a_better,
            color="#ff7a00",
            label=f"{label_a} better (n={len(a_better)})",
        )
    if b_better:
        ax.plot(
            range(1, len(b_better) + 1),
            b_better,
            color="#4caf50",
            label=f"{label_b} better (n={len(b_better)})",
        )

    ax.set_xlabel("Sequence rank")
    ax.set_ylabel("|p@15(A) \u2212 p@15(B)| (%)")
    ax.set_title(f"Per-sequence gap: {label_a} vs {label_b}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_scatter_plot(
    sequences: List[dict],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Save a per-sequence scatter plot of p@15_A vs p@15_B.

    Points above the diagonal are sequences where B wins; below are A wins.

    Args:
        sequences: List of sequence data dicts containing 'p15_a' and 'p15_b'.
        label_a: Display label for method A.
        label_b: Display label for method B.
        output_path: Path to write the PNG figure.
    """
    xs: List[float] = []
    ys: List[float] = []

    for s in sequences:
        p_a = s.get("p15_a", float("nan"))
        p_b = s.get("p15_b", float("nan"))
        if np.isnan(p_a) or np.isnan(p_b):
            continue
        xs.append(p_a)
        ys.append(p_b)

    colors = [
        "#ff7a00" if x > y else ("#4caf50" if y > x else "#888888")
        for x, y in zip(xs, ys)
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.5, s=12, linewidths=0)
    ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{label_a} p@15 (%)")
    ax.set_ylabel(f"{label_b} p@15 (%)")
    ax.set_title(f"Per-sequence p@15: {label_a} vs {label_b}")
    ax.grid(True, alpha=0.3)
    n_a = sum(1 for x, y in zip(xs, ys) if x > y)
    n_b = sum(1 for x, y in zip(xs, ys) if y > x)
    ax.legend(
        handles=[
            mpatches.Patch(color="#ff7a00", label=f"{label_a} better (n={n_a})"),
            mpatches.Patch(color="#4caf50", label=f"{label_b} better (n={n_b})"),
        ],
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_comparison_report(
    sequences: List[dict],
    label_a: str,
    label_b: str,
    dataset: str,
    out_dir: Path,
) -> None:
    """Save dominance + scatter plots and a self-contained HTML report.

    Args:
        sequences: Loaded sequence data dicts with 'p15_a' / 'p15_b'.
        label_a: Display label for method A.
        label_b: Display label for method B.
        dataset: Dataset key (used in report title).
        out_dir: Directory to write dominance.png, scatter.png, report.html.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dominance_path = out_dir / "dominance.png"
    scatter_path = out_dir / "scatter.png"
    report_path = out_dir / "report.html"

    _save_dominance_plot(sequences, label_a, label_b, dominance_path)
    _save_scatter_plot(sequences, label_a, label_b, scatter_path)

    def _to_b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode()

    dominance_b64 = _to_b64(dominance_path)
    scatter_b64 = _to_b64(scatter_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{label_a} vs {label_b} \u2014 {dataset}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #f5f7fa;
            color: #1a202c; padding: 24px; margin: 0; }}
    h1 {{ font-size: 20px; margin: 0 0 4px; }}
    .meta {{ color: #718096; font-size: 13px; margin-bottom: 20px; }}
    .plots {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }}
    .plot-box {{ background: #fff; padding: 12px; border-radius: 10px;
                 box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
    img {{ display: block; max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>{label_a} vs {label_b}</h1>
  <p class="meta">Dataset: {dataset} &nbsp;|&nbsp; {timestamp} &nbsp;|&nbsp; {len(sequences)} sequences</p>
  <div class="plots">
    <div class="plot-box">
      <img src="data:image/png;base64,{dominance_b64}" alt="Dominance curves">
    </div>
    <div class="plot-box">
      <img src="data:image/png;base64,{scatter_b64}" alt="Scatter plot">
    </div>
  </div>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    print(f"Comparison report saved to: {report_path}")


def _load_annotations(annotations_file: Optional[Path]) -> None:
    """Load sequence-level attribute annotations from JSON into _st."""
    if annotations_file is None or not annotations_file.exists():
        if annotations_file is not None:
            print(f"WARNING: annotations file not found: {annotations_file}", file=sys.stderr)
        return
    try:
        data = json.loads(annotations_file.read_text(encoding="utf-8"))
        _st.attributes = data.get("attributes", [])
        _st.annotations = {
            seq: {k: bool(v) for k, v in flags.items()}
            for seq, flags in data.get("annotations", {}).items()
        }
        print(f"Loaded annotations: {len(_st.attributes)} attributes, {len(_st.annotations)} sequences")
    except Exception as e:
        print(f"WARNING: failed to load annotations: {e}", file=sys.stderr)


def _startup(
    dataset: str,
    base_dir: Path,
    results_dir_a: Path,
    results_dir_b: Path,
    sequences: List[str],
    initfix_dir: Optional[Path],
    label_a: str,
    label_b: str,
    config_a: Optional[Path] = None,
    config_b: Optional[Path] = None,
    split: Optional[str] = None,
    annotations_file: Optional[Path] = None,
    use_best_rotation: bool = False,
    spawn_detail: bool = False,
) -> None:
    """Load comparison data and build heatmaps."""
    _st.dataset = dataset
    _st.base_dir = base_dir
    _st.results_dir_a = results_dir_a
    _st.results_dir_b = results_dir_b
    _st.label_a = label_a
    _st.label_b = label_b
    _st.use_best_rotation = use_best_rotation
    _st.load_gt_fn = _make_load_gt_fn(dataset, initfix_dir)
    _load_annotations(annotations_file)

    if spawn_detail:
        tmpdir = Path(tempfile.mkdtemp(prefix="compare_viewer_detail_"))
        atexit.register(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
        print("Spawning detail servers…")
        _spawn_detail_server("a", config_a, results_dir_a, split, tmpdir / "report_a.html")
        _spawn_detail_server("b", config_b, results_dir_b, split, tmpdir / "report_b.html")
        print("Waiting for detail servers to become ready…")
        _wait_ready("a", _st.detail_ports["a"])
        _wait_ready("b", _st.detail_ports["b"])

    print(
        f"Loading {len(sequences)} sequences\n"
        f"  A ({label_a}): {results_dir_a}\n"
        f"  B ({label_b}): {results_dir_b}"
    )

    loaded = []
    n_fail = 0
    for seq_name in sequences:
        try:
            gt, valid = _st.load_gt_fn(base_dir, seq_name)
            pred_a = _load_predictions(results_dir_a, seq_name)
            pred_b = _load_predictions(results_dir_b, seq_name)
            t = min(len(gt), len(pred_a), len(pred_b))
            data = _build_sequence_data(
                seq_name=seq_name,
                gt=gt[:t],
                valid=valid[:t],
                pred_a=pred_a[:t],
                pred_b=pred_b[:t],
            )
            if len(data["delta"]) == 0:
                continue
            data["gt"] = gt[:t]
            data["valid"] = valid[:t]
            data["pred_a"] = pred_a[:t]
            data["pred_b"] = pred_b[:t]
            data["corrected_a"] = _corrected_frames(results_dir_a, seq_name)
            data["corrected_b"] = _corrected_frames(results_dir_b, seq_name)
            _st.seq_cache[seq_name] = data
            loaded.append(data)
        except Exception as e:
            print(f"  WARNING: {seq_name}: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            n_fail += 1

    print(f"Loaded {len(loaded)} sequences ({n_fail} failed)")
    if not loaded:
        raise ValueError("No overlapping sequences could be loaded from both methods")

    _st.loaded_sequences = loaded
    print("Ready.")


def _resolve_results_dir(
    dataset: str,
    root_results_dir: Path,
    config: Optional[Path],
    explicit_results_dir: Optional[Path],
    method_name: str,
) -> Path:
    """Resolve per-method results path."""
    if explicit_results_dir is not None:
        return explicit_results_dir
    if config is None:
        raise ValueError(
            f"{method_name}: provide either --results-dir-{method_name.lower()} "
            f"or --config-{method_name.lower()}"
        )
    return compute_output_dir(root_results_dir, dataset, config)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive FlatSAM2 comparison viewer for two methods"
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to inspect")
    parser.add_argument(
        "--config-a",
        type=Path,
        default=None,
        help="Config for method A (used with --results-dir)",
    )
    parser.add_argument(
        "--config-b",
        type=Path,
        default=None,
        help="Config for method B (used with --results-dir)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Top-level results root (default: from path_config)",
    )
    parser.add_argument(
        "--results-dir-a",
        type=Path,
        default=None,
        help="Explicit method-A results directory",
    )
    parser.add_argument(
        "--results-dir-b",
        type=Path,
        default=None,
        help="Explicit method-B results directory",
    )
    parser.add_argument(
        "--label-a", type=str, default="Method A", help="Display label for method A"
    )
    parser.add_argument(
        "--label-b", type=str, default="Method B", help="Display label for method B"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Dataset base directory (default: from path_config)",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="Initfix directory for pti dataset",
    )
    parser.add_argument("--sequences", nargs="+", help="Specific sequences to load")
    parser.add_argument(
        "--subset", help="Subset file name (configs/subsets/<name>.txt)"
    )
    parser.add_argument("--split", choices=["train", "test"], help="Dataset split")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=paths.COMPARISONS_DIRECTORY,
        help=(
            "Root directory for comparison reports. A subdirectory "
            "<label_a>_vs_<label_b>/<dataset>/ is created automatically. "
            "Pass an empty string to skip report generation."
        ),
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help=(
            "Spawn per-method pti_sam3_mask_homography_report servers on loopback "
            "and proxy them at /report-a/ and /report-b/. Only valid for pti dataset."
        ),
    )
    parser.add_argument(
        "--best-rotation",
        action="store_true",
        help="Use rotation-invariant metrics (try all 4 corner orderings, take minimum error)",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path("/mnt/personal/serycjon/flatsam2/annotations/planartrack_sequence_attributes.json"),
        help="JSON with sequence-level attribute annotations (from planartrack_sequence_annotator.py)",
    )
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    if args.detail and args.dataset not in ("pti", "planartrack"):
        raise ValueError("--detail only supported for pti/planartrack dataset")
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY

    results_dir_a = _resolve_results_dir(
        dataset=args.dataset,
        root_results_dir=args.results_dir,
        config=args.config_a,
        explicit_results_dir=args.results_dir_a,
        method_name="A",
    )
    results_dir_b = _resolve_results_dir(
        dataset=args.dataset,
        root_results_dir=args.results_dir,
        config=args.config_b,
        explicit_results_dir=args.results_dir_b,
        method_name="B",
    )

    for method_name, p in (("A", results_dir_a), ("B", results_dir_b)):
        if not p.exists():
            raise ValueError(f"Method {method_name} results directory not found: {p}")

    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        split=args.split,
    )

    _startup(
        dataset=args.dataset,
        base_dir=args.base_dir,
        results_dir_a=results_dir_a,
        results_dir_b=results_dir_b,
        sequences=sequences,
        initfix_dir=args.initfix_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        config_a=args.config_a,
        config_b=args.config_b,
        split=args.split,
        annotations_file=args.annotations_file,
        use_best_rotation=args.best_rotation,
        spawn_detail=args.detail,
    )

    if args.report_dir:
        safe = lambda s: re.sub(r"[^\w.-]", "_", s)
        out_dir = args.report_dir / f"{safe(args.label_a)}_vs_{safe(args.label_b)}" / args.dataset
        _save_comparison_report(
            sequences=_st.loaded_sequences,
            label_a=args.label_a,
            label_b=args.label_b,
            dataset=args.dataset,
            out_dir=out_dir,
        )

    print(f"Serving at http://{args.host}:{args.port}")
    hostname = socket.gethostname()
    print(f"ssh -L 8080:{hostname}:{args.port} rci")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
