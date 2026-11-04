"""Interactive viewer for SAM3 candidate-mask dumps.

Explores the output of ``configs/tracker/sam3_candidates.py`` (see
``FlatSAMTracker.track_candidates`` / ``run.py``): per-frame SAM3 candidate
masks under ``<results>/masks/<seq>/`` plus the ``ious.json`` sidecar.

Main goal: find *where the multiple candidates disagree*. The overview is a
heatmap (one row per sequence, one pixel per frame) coloured by candidate
disagreement (``1 - min pairwise mask IoU``) or predicted-IoU spread; sequences
are sorted most-ambiguous first. Click a sequence to open a detail view with a
frame scrubber (arrow keys step frames), an IoU timeline (SAM3 predicted IoUs +
pairwise mask agreement), and the frame image with each candidate overlay
individually togglable. Ground truth (PTi/POT corners) is available as a
secondary, togglable overlay plus an optional "regret" heatmap layer showing
where SAM3's confidence-selected candidate is not the one closest to GT.

Usage:
    python candidate_viewer.py pot --config configs/tracker/sam3_candidates.py
    python candidate_viewer.py pti --config sam3_candidates --port 8200 --no-gt
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from analyze_results import _make_load_gt_fn
from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    corners_to_mask,
    get_sequences,
    pot_image_paths,
    pt_image_paths,
)
from flatsam2.path_config import paths


# Candidate overlay colours, RGB. c0 cyan, c1 magenta, c2 yellow, c3 spring-green.
CAND_RGB: List[Tuple[int, int, int]] = [
    (0x00, 0xE5, 0xFF),
    (0xFF, 0x3D, 0xDB),
    (0xFF, 0xD5, 0x00),
    (0x76, 0xFF, 0x03),
]
# Fixed pair order for pairwise mask IoUs (indices into the candidate list).
PAIR_ORDER: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]


# ---------------------------------------------------------------------------
# Embedded HTML frontend
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SAM3 Candidate Viewer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; font-size: 13px; background: #1e1e1e; color: #ddd; height: 100vh; overflow: hidden; }
    #app { display: flex; height: 100vh; }
    #left { width: 34%; border-right: 1px solid #444; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
    #toolbar { padding: 6px 8px; display: flex; gap: 6px; align-items: center; background: #252525; border-bottom: 1px solid #444; flex-wrap: wrap; }
    #toolbar button { padding: 3px 8px; background: #333; color: #ccc; border: 1px solid #555; cursor: pointer; font: inherit; }
    #toolbar button.active { background: #555; color: #fff; border-color: #888; }
    #status { color: #888; margin-left: auto; font-size: 11px; white-space: nowrap; }
    #hm-wrap { flex: 1; overflow: auto; padding: 4px; }
    #hm { image-rendering: pixelated; cursor: crosshair; display: block; }
    #right { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    #right.empty { align-items: center; justify-content: center; color: #666; }
    #detail-top { padding: 6px 8px; background: #252525; border-bottom: 1px solid #444; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    #detail-top .title { font-weight: bold; color: #fff; }
    #frame-wrap { flex: 1; overflow: auto; display: flex; align-items: center; justify-content: center; background: #111; }
    #frame-img { max-width: 100%; max-height: 100%; image-rendering: auto; }
    #controls { padding: 6px 8px; background: #252525; border-top: 1px solid #444; display: flex; flex-direction: column; gap: 6px; }
    #toggles { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .chip { display: inline-flex; align-items: center; gap: 4px; padding: 2px 6px; border: 1px solid #555; cursor: pointer; user-select: none; }
    .chip.off { opacity: 0.4; }
    .sw { width: 11px; height: 11px; display: inline-block; border: 1px solid #000; }
    #scrub-row { display: flex; gap: 8px; align-items: center; }
    #scrub { flex: 1; }
    #readout { font-size: 11px; color: #bbb; white-space: pre; min-height: 30px; }
    #timeline { width: 100%; height: 150px; background: #161616; border: 1px solid #333; cursor: pointer; display: block; }
    .legend { font-size: 11px; color: #999; display: flex; gap: 12px; flex-wrap: wrap; }
    .legend span { display: inline-flex; align-items: center; gap: 3px; }
    .ln { width: 16px; height: 0; border-top-width: 2px; border-top-style: solid; display: inline-block; }
  </style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="toolbar"><span>layer:</span><span id="layers"></span><span id="status"></span></div>
    <div id="hm-wrap"><canvas id="hm"></canvas></div>
  </div>
  <div id="right" class="empty"><div>Click a sequence in the heatmap.</div></div>
</div>

<script>
const S = { layer: 'disagreement', layers: [], meta: null, hmImg: null,
            seq: null, data: null, frame: 0, cands: [true,true,true,true],
            showGt: false, showSel: true, showPairwise: false };

function el(t, props, kids){ const e=document.createElement(t); Object.assign(e, props||{});
  (kids||[]).forEach(k=>e.appendChild(typeof k==='string'?document.createTextNode(k):k)); return e; }
function rgb(c){ return `rgb(${c[0]},${c[1]},${c[2]})`; }
const CAND_RGB = [[0,229,255],[255,61,219],[255,213,0],[118,255,3]];
const PAIR_RGB = [[255,136,0],[136,255,0],[0,136,255]];

async function boot(){
  const r = await fetch('/api/layers'); S.layers = await r.json();
  S.layer = S.layers[0];
  const box = document.getElementById('layers');
  S.layers.forEach(l=>{ const b=el('button',{textContent:l}); b.onclick=()=>setLayer(l); b.dataset.layer=l; box.appendChild(b); });
  await setLayer(S.layer);
}

async function setLayer(l){
  S.layer = l;
  document.querySelectorAll('#layers button').forEach(b=>b.classList.toggle('active', b.dataset.layer===l));
  const meta = await (await fetch('/heatmaps/'+l+'/meta')).json();
  S.meta = meta;
  const img = new Image();
  img.onload = ()=>{ S.hmImg = img; drawHeatmap(); };
  img.src = '/heatmaps/'+l+'.png?t='+Date.now();
}

const HM_SCALE = 3, ROW_H = 3;
function drawHeatmap(){
  const c = document.getElementById('hm'), m = S.meta;
  c.width = m.max_T * HM_SCALE; c.height = m.n_seqs * ROW_H;
  const ctx = c.getContext('2d'); ctx.imageSmoothingEnabled=false;
  ctx.drawImage(S.hmImg, 0,0, c.width, c.height);
  if (S.seq){ const row = m.seq_names.indexOf(S.seq);
    if (row>=0){ ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.strokeRect(0.5,row*ROW_H+0.5, c.width-1, ROW_H-1);
      if (S.data){ const x=S.frame*HM_SCALE; ctx.strokeStyle='#0ff'; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,c.height); ctx.stroke(); } } }
}
document.getElementById('hm').addEventListener('click', e=>{
  const c=e.target, r=c.getBoundingClientRect();
  const col=Math.floor((e.clientX-r.left)/HM_SCALE), row=Math.floor((e.clientY-r.top)/ROW_H);
  const name=S.meta.seq_names[row]; if(!name) return; openSeq(name, col);
});
document.getElementById('hm').addEventListener('mousemove', e=>{
  const c=e.target, r=c.getBoundingClientRect();
  const col=Math.floor((e.clientX-r.left)/HM_SCALE), row=Math.floor((e.clientY-r.top)/ROW_H);
  const name=S.meta.seq_names[row];
  document.getElementById('status').textContent = name? `${name}  f${col}` : '';
});

async function openSeq(name, frame){
  S.seq=name; S.data = await (await fetch('/api/sequence/'+encodeURIComponent(name))).json();
  S.frame = Math.min(Math.max(frame||0,0), S.data.num_frames-1);
  buildDetail(); drawHeatmap();
}

function buildDetail(){
  const d=S.data; const right=document.getElementById('right'); right.className='';
  right.innerHTML='';
  const top = el('div',{id:'detail-top'});
  top.appendChild(el('span',{className:'title', textContent:S.seq}));
  top.appendChild(el('span',{textContent:`${d.num_frames} frames · ${d.has_gt?'GT':'no GT'}`, style:'color:#888'}));
  right.appendChild(top);

  const fw=el('div',{id:'frame-wrap'}); const img=el('img',{id:'frame-img'}); fw.appendChild(img); right.appendChild(fw);

  const ctr=el('div',{id:'controls'});
  const toggles=el('div',{id:'toggles'});
  for(let k=0;k<4;k++){
    if(d.max_candidates<=k) break;
    const chip=el('span',{className:'chip'+(S.cands[k]?'':' off')});
    chip.appendChild(el('span',{className:'sw', style:'background:'+rgb(CAND_RGB[k])}));
    chip.appendChild(el('span',{textContent:'c'+k}));
    chip.onclick=()=>{ S.cands[k]=!S.cands[k]; chip.classList.toggle('off'); refreshImg(); };
    toggles.appendChild(chip);
  }
  const selChip=el('span',{className:'chip'+(S.showSel?'':' off')}); selChip.appendChild(el('span',{textContent:'★ selected outline'}));
  selChip.onclick=()=>{ S.showSel=!S.showSel; selChip.classList.toggle('off'); refreshImg(); }; toggles.appendChild(selChip);
  if(d.has_gt){ const g=el('span',{className:'chip'+(S.showGt?'':' off')}); g.appendChild(el('span',{className:'sw',style:'background:#fff'})); g.appendChild(el('span',{textContent:'GT'}));
    g.onclick=()=>{ S.showGt=!S.showGt; g.classList.toggle('off'); refreshImg(); }; toggles.appendChild(g); }
  const pwChip=el('span',{className:'chip'+(S.showPairwise?'':' off')}); pwChip.appendChild(el('span',{textContent:'pairwise IoU'}));
  pwChip.onclick=()=>{ S.showPairwise=!S.showPairwise; pwChip.classList.toggle('off'); drawTimeline(); updateLegend(); }; toggles.appendChild(pwChip);
  ctr.appendChild(toggles);

  const scrub=el('div',{id:'scrub-row'});
  const slider=el('input',{type:'range', id:'scrub', min:0, max:d.num_frames-1, value:S.frame});
  slider.oninput=()=>{ S.frame=+slider.value; refreshImg(); drawTimeline(); drawHeatmap(); updateReadout(); };
  scrub.appendChild(el('button',{textContent:'◀', onclick:()=>step(-1)}));
  scrub.appendChild(slider);
  scrub.appendChild(el('button',{textContent:'▶', onclick:()=>step(1)}));
  ctr.appendChild(scrub);

  const legend=el('div',{className:'legend', id:'legend'});
  ctr.appendChild(legend); updateLegend();
  const tl=el('canvas',{id:'timeline'}); ctr.appendChild(tl);
  const ro=el('div',{id:'readout'}); ctr.appendChild(ro);
  right.appendChild(ctr);

  tl.addEventListener('click', e=>{ const r=tl.getBoundingClientRect(); const f=Math.round((e.clientX-r.left)/tl.clientWidth*(d.num_frames-1)); S.frame=Math.min(Math.max(f,0),d.num_frames-1); document.getElementById('scrub').value=S.frame; refreshImg(); drawTimeline(); drawHeatmap(); updateReadout(); });
  refreshImg(); drawTimeline(); updateReadout();
}
const PAIR_ORDER_JS=[[0,1],[0,2],[1,2]];
function updateLegend(){
  const legend=document.getElementById('legend'); if(!legend) return;
  let h='predicted IoU: '+[0,1,2].map(k=>`<span><i class="ln" style="border-top-color:${rgb(CAND_RGB[k])}"></i>c${k}</span>`).join(' ');
  if(S.showPairwise) h+=' &nbsp; pairwise IoU: '+PAIR_ORDER_JS.map((p,i)=>`<span><i class="ln" style="border-top-color:${rgb(PAIR_RGB[i])};border-top-style:dashed"></i>${p[0]}·${p[1]}</span>`).join(' ');
  h+=' <span><i class="ln" style="border-top-color:#fff;border-top-width:3px"></i>disagreement</span>';
  legend.innerHTML=h;
}

function step(dir){ if(!S.data) return; S.frame=Math.min(Math.max(S.frame+dir,0),S.data.num_frames-1);
  const s=document.getElementById('scrub'); if(s) s.value=S.frame; refreshImg(); drawTimeline(); drawHeatmap(); updateReadout(); }
window.addEventListener('keydown', e=>{ if(!S.data) return;
  if(e.key==='ArrowLeft'){ step(-1); e.preventDefault(); } else if(e.key==='ArrowRight'){ step(1); e.preventDefault(); } });

function refreshImg(){
  const cands = S.cands.map((v,i)=>v?i:-1).filter(i=>i>=0).join(',');
  const q = `cands=${cands}&gt=${S.showGt?1:0}&sel=${S.showSel?1:0}`;
  document.getElementById('frame-img').src = `/api/frame/${encodeURIComponent(S.seq)}/${S.frame}?${q}`;
}

function drawTimeline(){
  const c=document.getElementById('timeline'), d=S.data; if(!c) return;
  c.width=c.clientWidth; c.height=150; const ctx=c.getContext('2d');
  ctx.fillStyle='#161616'; ctx.fillRect(0,0,c.width,c.height);
  const W=c.width, H=c.height, T=d.num_frames;
  const X=f=>T>1? f/(T-1)*(W-1):0, Y=v=>H-4-(H-8)*Math.max(0,Math.min(1,v));
  ctx.strokeStyle='#333'; [0,0.5,1].forEach(v=>{ ctx.beginPath(); ctx.moveTo(0,Y(v)); ctx.lineTo(W,Y(v)); ctx.stroke(); });
  function line(arrGetter, color, dash, wid){
    ctx.strokeStyle=color; ctx.lineWidth=wid||1; ctx.setLineDash(dash||[]); ctx.beginPath(); let started=false;
    for(let f=0; f<T; f++){ const v=arrGetter(f); if(v==null||isNaN(v)){ started=false; continue; }
      const x=X(f), y=Y(v); if(!started){ ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  for(let k=0;k<d.max_candidates && k<3;k++) line(f=>{ const a=d.pred_ious[f]; return a&&a[k]!=null?a[k]:null; }, rgb(CAND_RGB[k]),[],1);
  if(S.showPairwise) for(let i=0;i<3;i++) line(f=>{ const a=d.pairwise_ious[f]; return a&&a[i]!=null?a[i]:null; }, rgb(PAIR_RGB[i]),[4,3],1);
  line(f=>d.disagreement[f], '#ffffff',[],3);
  const x=X(S.frame); ctx.strokeStyle='#0ff'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
}

function fmt(a){ return a==null? '—' : (Array.isArray(a)? '['+a.map(v=>v==null?'—':v.toFixed(2)).join(', ')+']' : a.toFixed(3)); }
function updateReadout(){
  const d=S.data, f=S.frame; const ro=document.getElementById('readout'); if(!ro) return;
  let s=`frame ${f}   M=${d.n_candidates[f]}   best=c${d.best[f]}\n`;
  s+=`pred IoU  ${fmt(d.pred_ious[f])}   pred-spread ${fmt(d.pred_spread[f])}\n`;
  s+=`pairwise  ${fmt(d.pairwise_ious[f])}   disagreement ${fmt(d.disagreement[f])}`;
  if(d.has_gt){ s+=`\nIoU→GT   ${fmt(d.iou_to_gt[f])}   oracle=c${d.oracle[f]}   regret ${fmt(d.regret[f])}`; }
  ro.textContent=s;
}
boot();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class _State:
    dataset: str = ""
    base_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None
    initfix_dir: Optional[Path] = None
    use_gt: bool = True
    load_gt_fn = None

    seq_stats: Dict[str, dict] = {}       # name -> per-frame stat arrays
    gt_cache: Dict[str, tuple] = {}       # name -> (gt (T,2,4), valid (T,))
    image_path_cache: Dict[str, Optional[List[Path]]] = {}
    mask_cache: Dict[str, np.ndarray] = {}  # small LRU-ish cache for rendering

    sorted_seq_names: List[str] = []
    layers: List[str] = []
    heatmap_store: Dict[str, dict] = {}   # layer -> {png_bytes, meta}


_st = _State()
app = FastAPI(title="SAM3 Candidate Viewer")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _image_paths(seq_name: str) -> Optional[List[Path]]:
    if seq_name in _st.image_path_cache:
        return _st.image_path_cache[seq_name]
    try:
        if DATASETS[_st.dataset].gt_loader == "pot":
            result = pot_image_paths(_st.base_dir, seq_name)
        else:
            result = pt_image_paths(_st.base_dir, seq_name)
    except Exception:
        result = None
    _st.image_path_cache[seq_name] = result
    return result


def _load_ious_json(seq_name: str) -> Optional[dict]:
    """Load the per-sequence ``ious.json`` sidecar written by run.py's dump."""
    p = _st.masks_dir / seq_name / "ious.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_cand_mask(seq_name: str, frame_idx: int, k: int) -> Optional[np.ndarray]:
    """Load candidate *k* bool mask for a frame (``{f+1:05d}_c{k}.png``)."""
    key = f"{seq_name}/{frame_idx}/c{k}"
    if key in _st.mask_cache:
        return _st.mask_cache[key]
    p = _st.masks_dir / seq_name / f"{frame_idx + 1:05d}_c{k}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    out = (m > 0) if m is not None else None
    if len(_st.mask_cache) > 256:
        _st.mask_cache.clear()
    _st.mask_cache[key] = out
    return out


def _gt(seq_name: str) -> Optional[tuple]:
    """Return (gt corners (T,2,4), valid (T,)) or None."""
    if not _st.use_gt:
        return None
    if seq_name in _st.gt_cache:
        return _st.gt_cache[seq_name]
    try:
        gt, valid = _st.load_gt_fn(_st.base_dir, seq_name)
        _st.gt_cache[seq_name] = (gt, valid)
        return gt, valid
    except Exception as e:
        print(f"  GT load failed for {seq_name}: {e}", file=sys.stderr)
        _st.gt_cache[seq_name] = None
        return None


def _bbox_area(m: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    """Tight (y0, y1, x0, x1) bounding box and foreground area of a bool mask.

    Returns ``(None, 0)`` for an empty mask. Computed with cheap 1-D row/col
    reductions so the expensive ``.sum()`` runs only over the bbox.
    """
    rows = np.any(m, axis=1)
    if not rows.any():
        return None, 0
    cols = np.any(m, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    area = int(m[y0:y1, x0:x1].sum())
    return (y0, y1, x0, x1), area


def _iou_bbox(a: np.ndarray, box_a, area_a: int,
              b: np.ndarray, box_b, area_b: int) -> float:
    """IoU of two bool masks via bbox precrop and precomputed areas.

    ``inter`` is counted only inside the overlap of the two bounding boxes;
    ``union = area_a + area_b - inter`` avoids a full-frame ``logical_or``.
    """
    if box_a is None or box_b is None:
        return 0.0
    y0 = max(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x0 = max(box_a[2], box_b[2])
    x1 = min(box_a[3], box_b[3])
    if y1 <= y0 or x1 <= x0:
        inter = 0
    else:
        inter = int(np.logical_and(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).sum())
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-sequence stat precompute (cached to disk)
# ---------------------------------------------------------------------------

def _cache_path() -> Path:
    return _st.masks_dir / "candidate_spread_cache.json"


# cv2 IMREAD flags that decode a PNG straight to 1/N resolution — much cheaper
# to decode and 1/N^2 the pixels for the boolean ops. IoU of blob masks is
# essentially unchanged. Used for STATS only; frame rendering stays full-res.
_REDUCE_FLAG = {
    1: cv2.IMREAD_GRAYSCALE,
    2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
    4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
    8: cv2.IMREAD_REDUCED_GRAYSCALE_8,
}


def _compute_seq_stats_pure(
    seq_name: str, masks_dir: Path, dataset: str, base_dir: Optional[Path],
    initfix_dir: Optional[Path], use_gt: bool, downscale: int,
) -> Optional[dict]:
    """Compute per-frame candidate stat arrays for a sequence (no shared state).

    Reads ``ious.json`` for candidate counts and predicted IoUs, then decodes the
    candidate mask PNGs (downscaled by ``downscale``) to compute pairwise mask
    IoU / disagreement and, when GT is enabled, per-candidate IoU-to-GT / oracle
    / regret. Pure so it can run in a worker process.

    Returns:
        Dict of frame-indexed arrays (length T), or None if the dump is missing.
    """
    ij = masks_dir / seq_name / "ious.json"
    if not ij.exists():
        return None
    try:
        meta = json.loads(ij.read_text(encoding="utf-8"))
    except Exception:
        return None

    flag = _REDUCE_FLAG.get(downscale, cv2.IMREAD_GRAYSCALE)

    def load_mask(f: int, k: int) -> Optional[np.ndarray]:
        p = masks_dir / seq_name / f"{f + 1:05d}_c{k}.png"
        if not p.exists():
            return None
        m = cv2.imread(str(p), flag)
        return (m > 0) if m is not None else None

    T = max(int(s) for s in meta.keys())  # frames are 1-based contiguous
    n_cand = np.zeros(T, dtype=np.int32)
    pred_ious: List[Optional[List[float]]] = [None] * T
    pred_spread = np.full(T, np.nan)
    pairwise: List[Optional[List[Optional[float]]]] = [None] * T
    disagreement = np.full(T, np.nan)
    best = np.full(T, -1, dtype=np.int32)

    gt_pack = None
    if use_gt:
        try:
            gt_pack = _make_load_gt_fn(dataset, initfix_dir)(base_dir, seq_name)
        except Exception:
            gt_pack = None
    has_gt = gt_pack is not None
    iou_to_gt: List[Optional[List[float]]] = [None] * T
    oracle = np.full(T, -1, dtype=np.int32)
    regret = np.full(T, np.nan)
    valid_arr = np.zeros(T, dtype=bool)

    for stem, rec in meta.items():
        f = int(stem) - 1
        if f < 0 or f >= T or rec.get("is_init"):
            continue
        M = int(rec.get("n_candidates", 0))
        n_cand[f] = M
        pious = [float(x) for x in rec.get("ious", [])]
        if pious:
            pred_ious[f] = pious
            best[f] = int(np.argmax(pious))
            if len(pious) >= 2:
                pred_spread[f] = float(max(pious) - min(pious))
        if M < 2:
            continue
        masks = [load_mask(f, k) for k in range(M)]
        # Per-mask bbox + area once; reused across all pairs and the GT compare.
        boxes = [None] * M
        areas = [0] * M
        for k in range(M):
            if masks[k] is not None:
                boxes[k], areas[k] = _bbox_area(masks[k])
        pw: List[Optional[float]] = []
        min_iou = 1.0
        for (i, j) in PAIR_ORDER:
            if i < M and j < M and masks[i] is not None and masks[j] is not None:
                v = _iou_bbox(masks[i], boxes[i], areas[i],
                              masks[j], boxes[j], areas[j])
                pw.append(v)
                min_iou = min(min_iou, v)
            else:
                pw.append(None)
        pairwise[f] = pw
        if any(v is not None for v in pw):
            disagreement[f] = 1.0 - min_iou
        if has_gt:
            gt, valid = gt_pack
            if f < len(valid) and valid[f] and masks and masks[0] is not None:
                valid_arr[f] = True
                h, w = masks[0].shape[:2]
                # Corners are in full-res pixels; scale to the downscaled mask.
                gt_mask = corners_to_mask(gt[f] / downscale, (h, w)) > 0
                gbox, garea = _bbox_area(gt_mask)
                ig = [
                    _iou_bbox(masks[k], boxes[k], areas[k], gt_mask, gbox, garea)
                    if masks[k] is not None else 0.0
                    for k in range(M)
                ]
                iou_to_gt[f] = ig
                oracle[f] = int(np.argmax(ig))
                if best[f] >= 0:
                    regret[f] = float(ig[oracle[f]] - ig[best[f]])

    return {
        "num_frames": int(T),
        "max_candidates": int(n_cand.max()) if T else 0,
        "has_gt": bool(has_gt),
        "n_candidates": n_cand.tolist(),
        "pred_ious": pred_ious,
        "pred_spread": [None if np.isnan(v) else float(v) for v in pred_spread],
        "pairwise_ious": pairwise,
        "disagreement": [None if np.isnan(v) else float(v) for v in disagreement],
        "best": best.tolist(),
        "iou_to_gt": iou_to_gt,
        "oracle": oracle.tolist(),
        "regret": [None if np.isnan(v) else float(v) for v in regret],
        "valid": valid_arr.tolist(),
    }


def _stats_worker(args: tuple) -> Tuple[str, Optional[dict]]:
    """Picklable wrapper around :func:`_compute_seq_stats_pure` for a pool."""
    (name, masks_dir, dataset, base_dir, initfix_dir, use_gt, downscale) = args
    return name, _compute_seq_stats_pure(
        name, masks_dir, dataset, base_dir, initfix_dir, use_gt, downscale
    )


def _flush_cache(cache: dict, cache_path: Path) -> None:
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(cache_path)


def _load_or_build_stats(sequences: List[str], workers: int, downscale: int) -> None:
    """Populate ``_st.seq_stats``, reusing a disk cache keyed by ious.json mtime.

    Uncached sequences are computed in parallel across ``workers`` processes;
    the cache is flushed incrementally so a Ctrl-C keeps completed work.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_path = _cache_path()
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    tag = f"{'gt' if _st.use_gt else 'nogt'}_d{downscale}"
    out: Dict[str, dict] = {}
    todo: List[Tuple[str, float]] = []
    for name in sequences:
        ij = _st.masks_dir / name / "ious.json"
        if not ij.exists():
            continue
        mtime = ij.stat().st_mtime
        entry = cache.get(name)
        if entry and entry.get("mtime") == mtime and entry.get("tag") == tag:
            out[name] = entry["stats"]
        else:
            todo.append((name, mtime))

    n_cached = len(out)
    print(f"  {n_cached} from cache, {len(todo)} to compute "
          f"({workers} workers, downscale {downscale})")

    if todo:
        t0 = time.time()
        args = [
            (name, _st.masks_dir, _st.dataset, _st.base_dir,
             _st.initfix_dir, _st.use_gt, downscale)
            for name, _ in todo
        ]
        mtime_by_name = {name: mtime for name, mtime in todo}
        done = 0
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(_stats_worker, a): a[0] for a in args}
            for fut in as_completed(futures):
                name, stats = fut.result()
                done += 1
                if stats is None:
                    continue
                out[name] = stats
                cache[name] = {"mtime": mtime_by_name[name], "tag": tag, "stats": stats}
                if done % 20 == 0 or done == len(todo):
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (len(todo) - done) / max(1e-6, rate)
                    print(f"  computed {done}/{len(todo)} "
                          f"({rate:.1f} seq/s, ETA {eta:.0f}s)")
                    _flush_cache(cache, cache_path)
        _flush_cache(cache, cache_path)

    _st.seq_stats = out
    print(f"  {len(out)} sequences with stats "
          f"({len(out) - n_cached} recomputed, {n_cached} from cache)")


def _mean_disagreement(stats: dict) -> float:
    vals = [v for v in stats["disagreement"] if v is not None]
    return float(np.mean(vals)) if vals else -1.0


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

_GRAY_SINGLE = (0x55, 0x55, 0x55)   # M == 1 (no pair to compare)
_GRAY_ABSENT = (0x22, 0x22, 0x22)   # M == 0 (object absent)
_PAD_RGB = (0xCC, 0xCC, 0xCC)       # off the end of a short sequence / init frame


def _magma_rgb(values: np.ndarray, vmax: float = 1.0) -> np.ndarray:
    import matplotlib.cm as cm
    normed = np.clip(values / vmax, 0.0, 1.0)
    rgba = cm.magma(normed)
    return (rgba[:, :3] * 255).astype(np.uint8)


def _build_heatmap(layer: str, ordered: List[str]) -> Tuple[bytes, dict]:
    """Build a raw-pixel PNG (row per sequence, col per frame) + click metadata."""
    n_seqs = len(ordered)
    max_T = max(_st.seq_stats[n]["num_frames"] for n in ordered)
    img = np.full((n_seqs, max_T, 3), _PAD_RGB, dtype=np.uint8)

    for row, name in enumerate(ordered):
        st = _st.seq_stats[name]
        T = st["num_frames"]
        n_cand = np.asarray(st["n_candidates"])
        vals = np.array(
            [np.nan if v is None else v for v in st[layer]], dtype=np.float64
        ) if layer in st else np.full(T, np.nan)

        row_rgb = np.full((T, 3), _GRAY_SINGLE, dtype=np.uint8)
        row_rgb[n_cand == 0] = _GRAY_ABSENT
        finite = np.isfinite(vals)
        if finite.any():
            row_rgb[finite] = _magma_rgb(vals[finite], vmax=1.0)
        img[row, :T] = row_rgb

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    meta = {
        "layer": layer,
        "seq_names": ordered,
        "row_lengths": [_st.seq_stats[n]["num_frames"] for n in ordered],
        "mean_disagreement": [_mean_disagreement(_st.seq_stats[n]) for n in ordered],
        "max_T": int(max_T),
        "n_seqs": int(n_seqs),
    }
    return bytes(buf), meta


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _placeholder(text: str, h: int = 480, w: int = 640) -> np.ndarray:
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (20, h // 2 - 10 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    return img


def _overlay_mask(img: np.ndarray, mask: np.ndarray, rgb: Tuple[int, int, int],
                  alpha: float = 0.4, outline: int = 2) -> np.ndarray:
    """Blend a translucent colour fill + solid contour for one bool mask."""
    color = (rgb[2], rgb[1], rgb[0])  # BGR
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    fill = img.copy()
    fill[mask] = color
    out = cv2.addWeighted(img, 1 - alpha, fill, alpha, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if outline:
        cv2.drawContours(out, cnts, -1, color, outline)
    return out


def _draw_gt(img: np.ndarray, corners: np.ndarray) -> None:
    pts = corners.T.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, (255, 255, 255), 2)


def render_frame(seq_name: str, frame_idx: int, cands: List[int],
                 show_gt: bool, show_sel: bool) -> bytes:
    """Render a frame with the requested candidate overlays. Returns JPEG bytes."""
    paths_list = _image_paths(seq_name)
    img = None
    if paths_list and frame_idx < len(paths_list) and paths_list[frame_idx].exists():
        img = cv2.imread(str(paths_list[frame_idx]))
    st = _st.seq_stats.get(seq_name)

    if img is None:
        # Fall back to the selected mask's canvas size so overlays still show.
        h, w = 480, 640
        m0 = _load_cand_mask(seq_name, frame_idx, 0)
        if m0 is not None:
            h, w = m0.shape[:2]
        msg = f"{seq_name}  frame {frame_idx}"
        if paths_list is None:
            msg += "\n(dataset not mounted)"
        img = _placeholder(msg, h, w)

    sel = st["best"][frame_idx] if st and frame_idx < st["num_frames"] else -1
    for k in cands:
        m = _load_cand_mask(seq_name, frame_idx, k)
        if m is not None:
            img = _overlay_mask(img, m, CAND_RGB[k % len(CAND_RGB)])
    if show_sel and sel >= 0 and sel in cands:
        m = _load_cand_mask(seq_name, frame_idx, sel)
        if m is not None:
            base = CAND_RGB[sel % len(CAND_RGB)]
            comp = (255 - base[0], 255 - base[1], 255 - base[2])  # complementary
            img = _overlay_mask(img, m, comp, alpha=0.0, outline=2)
    if show_gt:
        gt_pack = _gt(seq_name)
        if gt_pack is not None:
            gt, valid = gt_pack
            if frame_idx < len(gt) and valid[frame_idx]:
                _draw_gt(img, gt[frame_idx])

    # (text, BGR colour). Selected-candidate line is drawn in the candidate's
    # own colour so it ties to the overlay; a black box behind keeps it legible.
    lines: List[Tuple[str, Tuple[int, int, int]]] = [(f"f{frame_idx}", (255, 255, 255))]
    if st and frame_idx < st["num_frames"] and sel >= 0:
        pious = st["pred_ious"][frame_idx]
        pred = pious[sel] if pious is not None and sel < len(pious) else None
        txt = f"sel=c{sel}" + (f"  iou {pred:.2f}" if pred is not None else "")
        crgb = CAND_RGB[sel % len(CAND_RGB)]
        lines.append((txt, (crgb[2], crgb[1], crgb[0])))
    y = 22
    for ln, color in lines:
        (tw, th), base = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (5, y - th - 4), (10 + tw, y + base), (0, 0, 0), -1)
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += th + 10

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf)


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE


@app.get("/api/layers")
async def api_layers():
    return JSONResponse(_st.layers)


@app.get("/heatmaps/{layer}.png")
async def heatmap_png(layer: str):
    if layer not in _st.heatmap_store:
        raise HTTPException(404, f"Unknown heatmap layer: {layer}")
    return StreamingResponse(iter([_st.heatmap_store[layer]["png_bytes"]]),
                             media_type="image/png")


@app.get("/heatmaps/{layer}/meta")
async def heatmap_meta(layer: str):
    if layer not in _st.heatmap_store:
        raise HTTPException(404, f"Unknown heatmap layer: {layer}")
    return JSONResponse(_st.heatmap_store[layer]["meta"])


@app.get("/api/sequence/{name}")
async def api_sequence(name: str):
    st = _st.seq_stats.get(name)
    if st is None:
        raise HTTPException(404, f"No candidate stats for sequence: {name}")
    return JSONResponse(st)


@app.get("/api/frame/{name}/{frame_idx}")
async def api_frame(name: str, frame_idx: int, cands: str = "0,1,2,3",
                    gt: int = 0, sel: int = 1):
    idxs = [int(x) for x in cands.split(",") if x.strip() != "" and x.strip() != "-1"]
    jpeg = render_frame(name, frame_idx, idxs, show_gt=bool(gt), show_sel=bool(sel))
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Startup + CLI
# ---------------------------------------------------------------------------

def _startup(dataset: str, base_dir: Path, masks_dir: Path,
             sequences: List[str], initfix_dir: Optional[Path], use_gt: bool,
             workers: int, downscale: int) -> None:
    _st.dataset = dataset
    _st.base_dir = base_dir
    _st.masks_dir = masks_dir
    _st.initfix_dir = initfix_dir
    _st.use_gt = use_gt
    _st.load_gt_fn = _make_load_gt_fn(dataset, initfix_dir) if use_gt else None

    print(f"Loading candidate dumps from {masks_dir} …")
    _load_or_build_stats(sequences, workers=workers, downscale=downscale)
    if not _st.seq_stats:
        print("ERROR: no sequences with candidate dumps found "
              "(expected <masks>/<seq>/ious.json).", file=sys.stderr)
        sys.exit(1)

    # Most-ambiguous first.
    ordered = sorted(_st.seq_stats.keys(),
                     key=lambda n: _mean_disagreement(_st.seq_stats[n]), reverse=True)
    _st.sorted_seq_names = ordered

    _st.layers = ["disagreement", "pred_spread"]
    if use_gt and any(_st.seq_stats[n]["has_gt"] for n in ordered):
        _st.layers.append("regret")

    print("Building heatmaps …")
    for layer in _st.layers:
        png, meta = _build_heatmap(layer, ordered)
        _st.heatmap_store[layer] = {"png_bytes": png, "meta": meta}
    print("Ready.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SAM3 candidate-mask viewer")
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset to inspect")
    parser.add_argument("--config", type=Path, default=Path("configs/tracker/sam3_candidates.py"),
                        help="Config used for the dump (locates the results directory)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Top-level results directory (default: from path_config)")
    parser.add_argument("--masks-dir", type=Path, default=None,
                        help="Direct path to the masks/ directory (overrides results-dir/config)")
    parser.add_argument("--base-dir", type=Path, default=None,
                        help="Dataset base directory (default: from path_config)")
    parser.add_argument("--initfix-dir", type=Path, default=None,
                        help="Initfix directory for pti (default: from path_config)")
    parser.add_argument("--sequences", nargs="+", help="Specific sequences to load")
    parser.add_argument("--subset", help="Subset file name (configs/subsets/<name>.txt)")
    parser.add_argument("--split", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--no-gt", action="store_true", help="Disable GT loading / regret layer")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel processes for the one-time stat precompute "
                             "(default: allocated cores)")
    parser.add_argument("--precompute-only", action="store_true",
                        help="Build/refresh the stat cache and exit without serving. "
                             "Use as a batch job (e.g. slurm/submit_cmd.py) to bake the "
                             "cache, then launch the viewer later (instant from cache).")
    parser.add_argument("--stat-downscale", type=int, default=1, choices=[1, 2, 4, 8],
                        help="Decode candidate masks at 1/N resolution for the disagreement "
                             "stats (rendering stays full-res). Default 1 (full res): with the "
                             "bbox IoU the ops are cheap and PNG decode isn't reduced anyway. "
                             "Raise only if decode-bound and you accept minor IoU error.")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY

    if args.masks_dir is not None:
        masks_dir = args.masks_dir
    else:
        masks_dir = compute_output_dir(args.results_dir, args.dataset, args.config) / "masks"

    if not masks_dir.exists():
        print(f"ERROR: masks directory not found: {masks_dir}")
        sys.exit(1)

    try:
        sequences = get_sequences(args.dataset, args.base_dir,
                                  sequences=args.sequences, subset=args.subset,
                                  split=args.split)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    import os
    # sched_getaffinity respects the SLURM cpuset; cpu_count() would report the
    # whole node and oversubscribe. Fall back to cpu_count() off-Linux.
    try:
        n_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        n_avail = os.cpu_count() or 4
    workers = args.workers if args.workers else n_avail
    _startup(dataset=args.dataset, base_dir=args.base_dir, masks_dir=masks_dir,
             sequences=sequences, initfix_dir=args.initfix_dir, use_gt=not args.no_gt,
             workers=workers, downscale=args.stat_downscale)

    if args.precompute_only:
        print("Precompute complete (cache written). Exiting (--precompute-only).")
        return

    print(f"Serving at http://{args.host}:{args.port}")
    print(f"ssh -L 8200:{socket.gethostname()}:{args.port} rci")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
