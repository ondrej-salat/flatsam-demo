#!/usr/bin/env python3
"""Interactive viewer for loma_v2 experiment outputs.

Shows one selected sequence at time, with frame navigation and overlays for:
  - current-frame SAM mask
  - matched keypoints
  - RANSAC inliers
  - homography-warped target polygon

Usage examples:
    python loma_v2_viewer.py pot --config configs/loma/loma_b_default.py
    python loma_v2_viewer.py pti --config configs/loma/loma_r_MAGSAC.py \
        --sequences book1 --masks-dir /path/to/masks --port 8124
"""

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from analyze_results import _load_predictions
from flatsam2.config import LoMaTrackerConfig, compute_output_dir, load_config_from_file
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
    warp_corners,
)
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>LoMa v2 Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; font-size: 13px; background: #1e1e1e; color: #ddd; height: 100vh; overflow: hidden; }
    #app { display: flex; flex-direction: column; height: 100vh; }
    #topbar { padding: 8px 10px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
              background: #252525; border-bottom: 1px solid #444; }
    #topbar select, #topbar button, #topbar input { background: #1f1f1f; color: #ddd; border: 1px solid #555;
              font: inherit; padding: 3px 8px; }
    #topbar button { cursor: pointer; }
    .field { display: flex; gap: 6px; align-items: center; color: #aaa; }
    #summary { color: #888; }
    #content { flex: 1; min-height: 0; display: flex; flex-direction: column; }
    #frame-wrap { flex: 1; min-height: 220px; background: #111; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    #frame-img { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
    #empty-msg { color: #555; }
    #timeline-wrap { background: #111; border-top: 1px solid #2a2a2a; border-bottom: 1px solid #2a2a2a; }
    #timeline { display: block; width: 100%; height: 62px; cursor: crosshair; }
    #controls { padding: 8px 10px; display: flex; flex-direction: column; gap: 8px; background: #1a1a1a; }
    #nav { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    #frame-slider { flex: 1; min-width: 240px; accent-color: #4fc3f7; }
    #frame-num { min-width: 96px; text-align: right; color: #aaa; }
    #overlays { display: flex; gap: 16px; flex-wrap: wrap; color: #bbb; }
    #overlays label { display: flex; align-items: center; gap: 4px; cursor: pointer; }
    #info { display: grid; grid-template-columns: minmax(220px, 2fr) minmax(220px, 3fr); gap: 12px; }
    #stats { color: #bbb; min-height: 72px; line-height: 1.5; white-space: pre-wrap; }
    #matrix { background: #101010; border: 1px solid #333; padding: 8px; color: #9cdcfe; white-space: pre; overflow: auto; }
    #legend { padding: 6px 10px; border-top: 1px solid #2a2a2a; color: #888; display: flex; gap: 12px; flex-wrap: wrap; }
    .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; }
    @media (max-width: 900px) {
      #info { grid-template-columns: 1fr; }
      #summary { width: 100%; }
    }
  </style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <label class="field">Sequence
      <select id="seq-select" onchange="onSequenceChange()"></select>
    </label>
    <button onclick="navFrame(-1)">&#9664;</button>
    <button onclick="navFrame(+1)">&#9654;</button>
    <span id="summary">Loading...</span>
  </div>

  <div id="content">
    <div id="frame-wrap">
      <span id="empty-msg">No sequence loaded</span>
      <img id="frame-img" alt="frame view">
    </div>

    <div id="timeline-wrap">
      <canvas id="timeline" height="62"></canvas>
    </div>

    <div id="controls">
      <div id="nav">
        <input type="range" id="frame-slider" min="0" max="0" value="0" oninput="goToFrame(+this.value)">
        <span id="frame-num">-</span>
      </div>

      <div id="overlays">
        <label><input type="checkbox" id="chk-mask" checked onchange="refreshFrame()"> mask</label>
        <label><input type="checkbox" id="chk-gt" checked onchange="refreshFrame()"> GT</label>
        <label><input type="checkbox" id="chk-h" checked onchange="refreshFrame()"> homography</label>
        <label><input type="checkbox" id="chk-kpts" checked onchange="refreshFrame()"> keypoints</label>
        <label><input type="checkbox" id="chk-matches" checked onchange="refreshFrame()"> inlier matches</label>
      </div>

      <div id="info">
        <div id="stats">-</div>
        <pre id="matrix">-</pre>
      </div>
    </div>
  </div>

  <div id="legend">
    <span>GT:<span class="sw" style="background:#ffffff;margin-left:4px"></span> dashed</span>
    <span>Homography:<span class="sw" style="background:#00ff7f;margin-left:4px"></span></span>
    <span>Mask:<span class="sw" style="background:rgba(255,80,80,0.75);margin-left:4px"></span></span>
    <span>Inliers:<span class="sw" style="background:#ffd700;margin-left:4px"></span></span>
    <span>Outliers:<span class="sw" style="background:#ffb000;margin-left:4px"></span></span>
  </div>
</div>

<script>
  let appInfo = null;
  let seqData = null;
  let currentSeq = null;
  let currentFrame = 0;

  const frameImg = document.getElementById('frame-img');
  const emptyMsg = document.getElementById('empty-msg');
  const slider = document.getElementById('frame-slider');
  const frameNum = document.getElementById('frame-num');
  const stats = document.getElementById('stats');
  const matrix = document.getElementById('matrix');
  const summary = document.getElementById('summary');
  const seqSelect = document.getElementById('seq-select');
  const timeline = document.getElementById('timeline');
  const timelineWrap = document.getElementById('timeline-wrap');

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.key === 'ArrowLeft') { navFrame(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { navFrame(+1); e.preventDefault(); }
  });

  timeline.addEventListener('click', (e) => {
    if (!seqData) return;
    const rect = timeline.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    goToFrame(Math.round(frac * Math.max(seqData.num_frames - 1, 0)));
  });

  new ResizeObserver(() => { if (seqData) drawTimeline(); }).observe(timelineWrap);

  async function boot() {
    const res = await fetch('/api/info');
    if (!res.ok) {
      summary.textContent = 'failed to load viewer state';
      return;
    }
    appInfo = await res.json();
    seqSelect.innerHTML = '';
    for (const name of appInfo.sequences) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      seqSelect.appendChild(opt);
    }

    const params = new URLSearchParams(location.search);
    const initialSeq = params.get('seq') || appInfo.default_sequence;
    const initialFrame = parseInt(params.get('frame') || '0', 10);
    summary.textContent = appInfo.dataset + ' | ' + appInfo.ransac_method +
      ' thr=' + appInfo.ransac_threshold.toFixed(1) + 'px';
    if (initialSeq) {
      seqSelect.value = initialSeq;
      await selectSequence(initialSeq, initialFrame);
    }
  }

  async function onSequenceChange() {
    await selectSequence(seqSelect.value, 0);
  }

  async function selectSequence(seqName, startFrame) {
    const res = await fetch('/api/sequence/' + encodeURIComponent(seqName));
    if (!res.ok) {
      summary.textContent = 'failed to load ' + seqName;
      return;
    }
    seqData = await res.json();
    currentSeq = seqName;
    slider.max = Math.max(seqData.num_frames - 1, 0);
    summary.textContent = seqName + ' | frames=' + seqData.num_frames +
      ' | valid=' + seqData.num_valid_gt + ' | masks=' + seqData.num_masks;
    drawTimeline();
    goToFrame(startFrame || 0);
  }

  function navFrame(delta) {
    if (!seqData) return;
    goToFrame(currentFrame + delta);
  }

  function refreshFrame() {
    if (!seqData) return;
    goToFrame(currentFrame);
  }

  function overlayParam(id) {
    return document.getElementById(id).checked ? 1 : 0;
  }

  function goToFrame(idx) {
    if (!seqData) return;
    currentFrame = Math.max(0, Math.min(idx, seqData.num_frames - 1));
    slider.value = currentFrame;
    frameNum.textContent = currentFrame + ' / ' + (seqData.num_frames - 1);
    const url = '/api/frame/' + encodeURIComponent(currentSeq) + '/' + currentFrame +
      '?mask=' + overlayParam('chk-mask') +
      '&gt=' + overlayParam('chk-gt') +
      '&homography=' + overlayParam('chk-h') +
      '&keypoints=' + overlayParam('chk-kpts') +
      '&matches=' + overlayParam('chk-matches') +
      '&t=' + Date.now();
    frameImg.src = url;
    frameImg.style.display = 'block';
    emptyMsg.style.display = 'none';
    updateFrameInfo(currentFrame);
    drawTimeline();

    const params = new URLSearchParams(location.search);
    params.set('seq', currentSeq);
    params.set('frame', String(currentFrame));
    history.replaceState({}, '', '?' + params.toString());
  }

  function updateFrameInfo(frameIdx) {
    if (!seqData) return;
    const s = seqData.frame_stats[frameIdx];
    const errStr = s.error === null ? 'n/a' : s.error.toFixed(2) + ' px';
    const ratioStr = s.inlier_ratio === null ? 'n/a' : (100 * s.inlier_ratio).toFixed(1) + '%';
    const initDetStr = (s.init_detected_object === null || s.init_detected_background === null)
      ? 'n/a'
      : s.init_detected_object + ' / ' + s.init_detected_background;
    const curDetStr = (s.detected_current_object === null || s.detected_current_background === null)
      ? 'n/a'
      : s.detected_current_object + ' / ' + s.detected_current_background;
    const matchedSplitStr = (s.matched_object === null || s.matched_background === null)
      ? 'n/a'
      : s.matched_object + ' / ' + s.matched_background;
    const inlierSplitStr = (s.inliers_object === null || s.inliers_background === null)
      ? 'n/a'
      : s.inliers_object + ' / ' + s.inliers_background;
    stats.textContent =
      'frame: ' + frameIdx + '\n' +
      'det init obj/bg: ' + initDetStr + '\n' +
      'det cur  obj/bg: ' + curDetStr + '\n' +
      'matches: ' + s.n_matches + '\n' +
      'matched obj/bg: ' + matchedSplitStr + '\n' +
      'inliers: ' + s.n_inliers + ' (' + ratioStr + ')\n' +
      'inlier obj/bg: ' + inlierSplitStr + '\n' +
      'mask: ' + (s.mask_available ? 'yes' : 'no') + '\n' +
      'GT valid: ' + (s.valid_gt ? 'yes' : 'no') + '\n' +
      'error: ' + errStr + '\n' +
      'homography source: ' + s.source;
    matrix.textContent = formatHomography(s.homography);
  }

  function formatHomography(H) {
    if (!H) return '-';
    return H.map((row) => row.map((v) => {
      const abs = Math.abs(v);
      if (abs >= 1000 || (abs > 0 && abs < 0.001)) return v.toExponential(3);
      return v.toFixed(5);
    }).join('  ')).join('\n');
  }

  function drawTimeline() {
    if (!seqData) return;
    const W = timelineWrap.clientWidth || 600;
    const H = 62;
    timeline.width = W;
    timeline.height = H;
    const ctx = timeline.getContext('2d');
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);

    const errors = seqData.frame_stats.map((s) => s.error);
    const valid = errors.filter((v) => v !== null);
    const maxErr = valid.length ? Math.max(15, Math.min(40, Math.max(...valid))) : 15;
    const pad = 4;
    const barW = W / Math.max(seqData.num_frames, 1);

    for (let i = 0; i < seqData.num_frames; i++) {
      const err = errors[i];
      if (err === null) continue;
      const norm = Math.min(err / maxErr, 1.0);
      const r = norm < 0.5 ? Math.round(norm * 2 * 255) : 255;
      const g = norm < 0.5 ? 255 : Math.round((1 - norm) * 2 * 255);
      ctx.fillStyle = 'rgb(' + r + ',' + g + ',0)';
      const bh = Math.max(1, norm * (H - pad));
      ctx.fillRect(i / seqData.num_frames * W, H - pad - bh, Math.max(1, barW + 0.5), bh);
    }

    const threshY = H - pad - (15 / maxErr) * (H - pad);
    if (threshY > 0 && threshY < H) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, threshY);
      ctx.lineTo(W, threshY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const x = seqData.num_frames > 1 ? (currentFrame / (seqData.num_frames - 1)) * W : 0;
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  boot();
</script>
</body>
</html>
"""


class _State:
    """Process-global viewer state."""

    dataset: str = ""
    base_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None
    initfix_dir: Optional[Path] = None
    sequences: List[str] = []
    default_sequence: Optional[str] = None
    ransac_method_name: str = "RANSAC"
    ransac_method_value: int = int(cv2.RANSAC)
    ransac_threshold: float = 5.0
    load_gt_fn = None
    image_paths_fn = None
    sequence_cache: Dict[str, dict] = {}


app = FastAPI(title="LoMa v2 Viewer")
_st = _State()


def _make_gt_loader(
    dataset: str,
    initfix_dir: Optional[Path],
) -> Callable[[Path, str], Tuple[np.ndarray, np.ndarray]]:
    """Build GT loader for dataset.

    Args:
        dataset: Dataset CLI name.
        initfix_dir: Optional PTI initfix directory.

    Returns:
        Callable GT loader.
    """
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir=initfix_dir)


def _make_image_paths_fn(dataset: str) -> Callable[[Path, str], List[Path]]:
    """Build image-path resolver for dataset.

    Args:
        dataset: Dataset CLI name.

    Returns:
        Callable image-path resolver.
    """
    if DATASETS[dataset].gt_loader == "pot":
        return pot_image_paths
    return pt_image_paths


def _draw_poly(
    img: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    dashed: bool = False,
) -> None:
    """Draw quadrilateral from `(2, 4)` corners.

    Args:
        img: Target BGR image.
        corners: Corner array `(2, 4)`.
        color: BGR color.
        thickness: Line thickness.
        dashed: Draw dashed outline.
    """
    if dashed:
        for i in range(corners.shape[1]):
            p1 = (int(corners[0, i]), int(corners[1, i]))
            p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
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


def _draw_points(
    img: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 3,
) -> None:
    """Draw point set.

    Args:
        img: Target BGR image.
        pts: Point array `(N, 2)`.
        color: BGR color.
        radius: Circle radius.
    """
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def _blend_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (80, 80, 255),
) -> np.ndarray:
    """Blend binary mask into image.

    Args:
        img: Target BGR image.
        mask: Binary mask `(H, W)`.
        color: Overlay color in BGR.

    Returns:
        Image with blended mask.
    """
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    colored = img.copy()
    colored[mask > 0] = color
    return cv2.addWeighted(img, 0.65, colored, 0.35, 0)


def _put_text_block(img: np.ndarray, lines: List[str], x: int, y: int) -> None:
    """Draw multi-line text with dark outline.

    Args:
        img: Target BGR image.
        lines: Text lines.
        x: Left position.
        y: Top baseline position.
    """
    yy = y
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        yy += 20


def _load_mask(seq_name: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load current-frame SAM mask if present.

    Args:
        seq_name: Sequence name.
        frame_idx: Zero-based frame index.

    Returns:
        Mask image or `None`.
    """
    if _st.masks_dir is None:
        return None
    mask_path = _st.masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


def _has_mask(seq_name: str, frame_idx: int) -> bool:
    """Check whether mask file exists.

    Args:
        seq_name: Sequence name.
        frame_idx: Zero-based frame index.

    Returns:
        True if mask file exists.
    """
    if _st.masks_dir is None:
        return False
    mask_path = _st.masks_dir / seq_name / f"{frame_idx + 1:05d}.png"
    return mask_path.exists()


def _homography_from_corners(
    init_corners: np.ndarray,
    pred_corners: np.ndarray,
) -> np.ndarray:
    """Compute homography from saved predicted corners.

    Args:
        init_corners: Init-frame corners `(2, 4)`.
        pred_corners: Predicted frame corners `(2, 4)`.

    Returns:
        Homography `(3, 3)` mapping init frame to current frame.

    Raises:
        ValueError: If homography cannot be estimated.
    """
    src = init_corners.T.astype(np.float64)
    dst = pred_corners.T.astype(np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("failed to estimate homography from saved corners")
    return H.astype(np.float64)


def _compute_inliers_from_homography(
    H: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mark matches consistent with given homography.

    Args:
        H: Homography `(3, 3)` mapping `pts_a -> pts_b`.
        pts_a: Init-frame points `(N, 2)`.
        pts_b: Current-frame points `(N, 2)`.
        threshold: Reprojection error threshold in pixels.

    Returns:
        Boolean inlier mask `(N,)`.
    """
    if len(pts_a) == 0:
        return np.zeros(0, dtype=bool)

    pts_a_h = np.concatenate(
        [pts_a.astype(np.float64), np.ones((len(pts_a), 1), dtype=np.float64)],
        axis=1,
    )
    warped = (H @ pts_a_h.T).T
    warped_xy = warped[:, :2] / warped[:, 2:3]
    reproj_err = np.linalg.norm(warped_xy - pts_b.astype(np.float64), axis=1)
    return reproj_err <= threshold


def _load_sequence(seq_name: str) -> dict:
    """Load and cache full sequence data.

    Args:
        seq_name: Sequence name.

    Returns:
        Cached sequence dict.

    Raises:
        ValueError: If sequence files are missing.
    """
    if seq_name in _st.sequence_cache:
        return _st.sequence_cache[seq_name]

    pred_path = _st.results_dir / f"{seq_name}.txt"
    npz_path = _st.results_dir / f"{seq_name}.npz"
    stats_path = _st.results_dir / f"{seq_name}_stats.json"
    if not pred_path.exists() or not npz_path.exists():
        raise ValueError(f"missing results for {seq_name}")

    gt, valid = _st.load_gt_fn(_st.base_dir, seq_name)
    image_paths = _st.image_paths_fn(_st.base_dir, seq_name)
    pred = _load_predictions(_st.results_dir, seq_name)
    keypoint_stats = None
    if stats_path.exists():
        try:
            keypoint_stats = json.loads(stats_path.read_text())
        except Exception:
            keypoint_stats = None

    with np.load(npz_path, allow_pickle=False) as data:
        T = min(len(gt), len(valid), len(pred), len(image_paths))
        gt = gt[:T]
        valid = valid[:T]
        pred = pred[:T]
        image_paths = image_paths[:T]
        init_corners = gt[0]

        matches: List[dict] = []
        homographies = np.zeros((T, 3, 3), dtype=np.float64)
        homographies[0] = np.eye(3, dtype=np.float64)
        homography_corners = np.zeros((T, 2, 4), dtype=np.float64)
        homography_corners[0] = init_corners
        errors = np.full(T, np.nan, dtype=np.float64)
        n_matches = np.zeros(T, dtype=np.int32)
        n_inliers = np.zeros(T, dtype=np.int32)
        inlier_ratio = np.full(T, np.nan, dtype=np.float64)
        source = np.full(T, "init", dtype="U16")
        mask_available = np.zeros(T, dtype=bool)
        last_H = np.eye(3, dtype=np.float64)

        for frame_idx in range(T):
            mask_available[frame_idx] = _has_mask(seq_name, frame_idx)

        for frame_idx in range(1, T):
            key_a = f"kptsA_{frame_idx:05d}"
            key_b = f"kptsB_{frame_idx:05d}"
            pts_a = (
                data[key_a].astype(np.float64)
                if key_a in data
                else np.zeros((0, 2), dtype=np.float64)
            )
            pts_b = (
                data[key_b].astype(np.float64)
                if key_b in data
                else np.zeros((0, 2), dtype=np.float64)
            )
            inliers = np.zeros(len(pts_a), dtype=bool)
            H_source = "corners"

            n_matches[frame_idx] = len(pts_a)
            try:
                H = _homography_from_corners(init_corners, pred[frame_idx])
                last_H = H
            except (ValueError, cv2.error):
                H = last_H.copy()
                H_source = "fallback"

            inliers = _compute_inliers_from_homography(
                H,
                pts_a,
                pts_b,
                _st.ransac_threshold,
            )

            homographies[frame_idx] = H
            homography_corners[frame_idx] = warp_corners(H, init_corners)
            n_inliers[frame_idx] = int(inliers.sum())
            if len(pts_a) > 0:
                inlier_ratio[frame_idx] = float(n_inliers[frame_idx] / len(pts_a))
            source[frame_idx] = H_source
            matches.append(
                {
                    "frame_idx": frame_idx,
                    "pts_a": pts_a.astype(np.float32),
                    "pts_b": pts_b.astype(np.float32),
                    "inliers": inliers,
                }
            )

        matches.insert(
            0,
            {
                "frame_idx": 0,
                "pts_a": np.zeros((0, 2), dtype=np.float32),
                "pts_b": np.zeros((0, 2), dtype=np.float32),
                "inliers": np.zeros(0, dtype=bool),
            },
        )

    valid_eval = valid.copy()
    if len(valid_eval) > 0:
        valid_eval[0] = False
    if valid_eval.any():
        errors[valid_eval] = alignment_error(gt[valid_eval], pred[valid_eval])

    seq = {
        "name": seq_name,
        "gt": gt,
        "valid": valid,
        "pred": pred,
        "image_paths": image_paths,
        "matches": matches,
        "homographies": homographies,
        "homography_corners": homography_corners,
        "errors": errors,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "source": source,
        "mask_available": mask_available,
        "keypoint_stats": keypoint_stats,
    }
    _st.sequence_cache[seq_name] = seq
    return seq


def _sequence_json(seq_name: str) -> dict:
    """Build JSON payload for one sequence.

    Args:
        seq_name: Sequence name.

    Returns:
        JSON-serializable sequence payload.
    """
    seq = _load_sequence(seq_name)
    keypoint_stats = seq.get("keypoint_stats") or {}
    stats_by_frame = {
        int(frame["frame_idx"]): frame for frame in keypoint_stats.get("frames", [])
    }
    frame_stats = []
    for frame_idx in range(len(seq["gt"])):
        err = seq["errors"][frame_idx]
        kp_stats = stats_by_frame.get(frame_idx, {})
        n_matches = kp_stats.get("matched_total")
        if n_matches is None:
            n_matches = int(seq["n_matches"][frame_idx])
        n_inliers = kp_stats.get("inliers_total")
        if n_inliers is None:
            n_inliers = int(seq["n_inliers"][frame_idx])
        ratio = kp_stats.get("inlier_ratio")
        if ratio is None:
            raw_ratio = seq["inlier_ratio"][frame_idx]
            ratio = float(raw_ratio) if np.isfinite(raw_ratio) else None
        frame_stats.append(
            {
                "frame_idx": frame_idx,
                "valid_gt": bool(seq["valid"][frame_idx]),
                "error": float(err) if np.isfinite(err) else None,
                "n_matches": int(n_matches),
                "n_inliers": int(n_inliers),
                "inlier_ratio": ratio,
                "init_detected_object": keypoint_stats.get("init_detected_object"),
                "init_detected_background": keypoint_stats.get(
                    "init_detected_background"
                ),
                "detected_current_object": kp_stats.get("detected_current_object"),
                "detected_current_background": kp_stats.get(
                    "detected_current_background"
                ),
                "matched_object": kp_stats.get("matched_object"),
                "matched_background": kp_stats.get("matched_background"),
                "inliers_object": kp_stats.get("inliers_object"),
                "inliers_background": kp_stats.get("inliers_background"),
                "source": str(seq["source"][frame_idx]),
                "mask_available": bool(seq["mask_available"][frame_idx]),
                "homography": seq["homographies"][frame_idx].tolist(),
            }
        )
    return {
        "name": seq_name,
        "num_frames": len(seq["gt"]),
        "num_valid_gt": int(seq["valid"][1:].sum()) if len(seq["valid"]) > 1 else 0,
        "num_masks": int(seq["mask_available"].sum()),
        "frame_stats": frame_stats,
    }


def render_frame(
    seq_name: str,
    frame_idx: int,
    show_mask: bool = True,
    show_gt: bool = True,
    show_homography: bool = True,
    show_keypoints: bool = True,
    show_matches: bool = True,
) -> bytes:
    """Render combined init/current frame view.

    Args:
        seq_name: Sequence name.
        frame_idx: Zero-based frame index.
        show_mask: Overlay current SAM mask.
        show_gt: Overlay GT polygons.
        show_homography: Overlay homography-warped polygon.
        show_keypoints: Overlay matched points.
        show_matches: Overlay inlier match lines.

    Returns:
        PNG bytes.
    """
    seq = _load_sequence(seq_name)
    frame_idx = int(np.clip(frame_idx, 0, len(seq["gt"]) - 1))

    init_img = cv2.imread(str(seq["image_paths"][0]))
    cur_img = cv2.imread(str(seq["image_paths"][frame_idx]))
    if init_img is None or cur_img is None:
        raise HTTPException(404, f"failed to read frame for {seq_name}")

    if show_mask:
        cur_mask = _load_mask(seq_name, frame_idx)
        if cur_mask is not None:
            cur_img = _blend_mask(cur_img, cur_mask)

    gap = 24
    left_h, left_w = init_img.shape[:2]
    right_h, right_w = cur_img.shape[:2]
    canvas_h = max(left_h, right_h)
    canvas_w = left_w + gap + right_w
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
    canvas[:left_h, :left_w] = init_img
    canvas[:right_h, left_w + gap : left_w + gap + right_w] = cur_img

    left_roi = canvas[:left_h, :left_w]
    right_x = left_w + gap
    right_roi = canvas[:right_h, right_x : right_x + right_w]

    if show_gt:
        _draw_poly(left_roi, seq["gt"][0], (255, 255, 255), 2, dashed=True)
        if seq["valid"][frame_idx]:
            _draw_poly(right_roi, seq["gt"][frame_idx], (255, 255, 255), 2, dashed=True)

    if show_homography:
        _draw_poly(right_roi, seq["homography_corners"][frame_idx], (127, 255, 0), 2)

    match_data = seq["matches"][frame_idx]
    pts_a = match_data["pts_a"]
    pts_b = match_data["pts_b"]
    inliers = match_data["inliers"]

    if show_matches and len(pts_a) > 0:
        for idx, (pt_a, pt_b) in enumerate(zip(pts_a, pts_b)):
            if idx >= len(inliers) or not inliers[idx]:
                continue
            p1 = (int(pt_a[0]), int(pt_a[1]))
            p2 = (int(pt_b[0]) + right_x, int(pt_b[1]))
            cv2.line(canvas, p1, p2, (255, 215, 0), 1, cv2.LINE_AA)

    if show_keypoints and len(pts_a) > 0:
        outlier_mask = (
            ~inliers if len(inliers) == len(pts_a) else np.ones(len(pts_a), dtype=bool)
        )
        if outlier_mask.any():
            _draw_points(left_roi, pts_a[outlier_mask], (0, 176, 255), radius=3)
            _draw_points(right_roi, pts_b[outlier_mask], (0, 176, 255), radius=3)
        if inliers.any():
            _draw_points(left_roi, pts_a[inliers], (255, 215, 0), radius=4)
            _draw_points(right_roi, pts_b[inliers], (255, 215, 0), radius=4)

    err = seq["errors"][frame_idx]
    ratio = seq["inlier_ratio"][frame_idx]
    lines_left = ["init frame"]
    lines_right = [f"frame {frame_idx}"]
    if frame_idx > 0:
        lines_right.append(f"matches: {int(seq['n_matches'][frame_idx])}")
        lines_right.append(f"inliers: {int(seq['n_inliers'][frame_idx])}")
        if np.isfinite(ratio):
            lines_right.append(f"ratio: {100.0 * float(ratio):.1f}%")
        lines_right.append(f"H: {seq['source'][frame_idx]}")
    if np.isfinite(err):
        lines_right.append(f"err: {float(err):.2f}px")
    elif not seq["valid"][frame_idx]:
        lines_right.append("err: GT invalid")

    _put_text_block(left_roi, lines_left, 8, 22)
    _put_text_block(right_roi, lines_right, 8, 22)

    _, buf = cv2.imencode(".png", canvas)
    return bytes(buf)


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve viewer page."""
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/info")
async def api_info() -> JSONResponse:
    """Return static viewer info."""
    return JSONResponse(
        {
            "dataset": _st.dataset,
            "sequences": _st.sequences,
            "default_sequence": _st.default_sequence,
            "ransac_method": _st.ransac_method_name,
            "ransac_threshold": _st.ransac_threshold,
        }
    )


@app.get("/api/sequence/{name}")
async def api_sequence(name: str) -> JSONResponse:
    """Return per-sequence frame stats.

    Args:
        name: Sequence name.

    Returns:
        Sequence JSON.
    """
    if name not in _st.sequences:
        raise HTTPException(404, f"unknown sequence: {name}")
    return JSONResponse(_sequence_json(name))


@app.get("/api/frame/{name}/{frame_idx}")
async def api_frame(
    name: str,
    frame_idx: int,
    mask: int = 1,
    gt: int = 1,
    homography: int = 1,
    keypoints: int = 1,
    matches: int = 1,
) -> StreamingResponse:
    """Return rendered frame PNG."""
    if name not in _st.sequences:
        raise HTTPException(404, f"unknown sequence: {name}")
    png = render_frame(
        name,
        frame_idx,
        show_mask=bool(mask),
        show_gt=bool(gt),
        show_homography=bool(homography),
        show_keypoints=bool(keypoints),
        show_matches=bool(matches),
    )
    return StreamingResponse(iter([png]), media_type="image/png")


def _startup(args: argparse.Namespace) -> None:
    """Resolve paths and preload viewer state.

    Args:
        args: Parsed CLI arguments.

    Raises:
        ValueError: If no sequences with results are found.
    """
    ds_info = DATASETS[args.dataset]

    base_dir = args.base_dir or getattr(paths, ds_info.base_dir_attr)
    initfix_dir = args.initfix_dir
    if ds_info.uses_split and args.split is None:
        args.split = "test"
    if ds_info.gt_loader == "pti" and initfix_dir is None:
        initfix_dir = paths.PT_INITFIX_DIR

    raw_cfg = load_config_from_file(args.config)
    if not isinstance(raw_cfg, LoMaTrackerConfig):
        raise ValueError(
            f"{args.config} defines {type(raw_cfg).__name__}, expected LoMaTrackerConfig"
        )

    results_dir = compute_output_dir(args.results_dir, args.dataset, args.config)
    if not results_dir.exists():
        raise ValueError(f"results dir not found: {results_dir}")

    sequences = get_sequences(
        args.dataset,
        base_dir,
        sequences=args.sequences,
        subset=args.subset,
        split=args.split,
    )
    available_sequences = [
        seq_name
        for seq_name in sequences
        if (results_dir / f"{seq_name}.txt").exists()
        and (results_dir / f"{seq_name}.npz").exists()
    ]
    if not available_sequences:
        raise ValueError(f"no completed loma_v2 sequences in {results_dir}")

    ransac_method_value = getattr(cv2, raw_cfg.ransac_method, None)
    if ransac_method_value is None:
        raise ValueError(f"unknown cv2 RANSAC method: {raw_cfg.ransac_method}")

    _st.dataset = args.dataset
    _st.base_dir = base_dir
    _st.results_dir = results_dir
    _st.masks_dir = args.masks_dir
    _st.initfix_dir = initfix_dir
    _st.sequences = available_sequences
    _st.default_sequence = available_sequences[0]
    _st.ransac_method_name = raw_cfg.ransac_method
    _st.ransac_method_value = int(ransac_method_value)
    _st.ransac_threshold = float(raw_cfg.ransac_threshold)
    _st.load_gt_fn = _make_gt_loader(args.dataset, initfix_dir)
    _st.image_paths_fn = _make_image_paths_fn(args.dataset)
    _st.sequence_cache = {}


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint.

    Args:
        argv: Optional CLI argv override.
    """
    parser = argparse.ArgumentParser(
        description="Interactive viewer for loma_v2 results"
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to inspect")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="LoMaTrackerConfig file used for loma_v2 run",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/mnt/personal/serycjon/flatsam2/loma_v2/"),
        help="Base output dir before config/dataset expansion",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Dataset base dir override",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="PTI initfix dir override",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Precomputed masks dir with layout <dir>/<seq>/<frame>.png",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Specific sequence names to expose",
    )
    parser.add_argument(
        "--subset",
        help="Subset file name from configs/subsets",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        help="Dataset split when dataset uses split",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args(argv)

    try:
        _startup(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Serving at http://{args.host}:{args.port}")
    hostname = socket.gethostname()
    print(f"ssh -L 8080:{hostname}:{args.port} rci")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
