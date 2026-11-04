"""Fast annotation UI for PlanarTrack sequence-level binary attributes.

Shows sampled frames in fixed grid, overlays GT quadrilateral, supports keyboard
navigation/toggling, and auto-persists annotation JSON.
"""

import socket
import argparse
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from einops import rearrange
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from flatsam2.datasets import pt_image_paths, pt_load_gt
from flatsam2.path_config import paths
from flatsam2.sequences import pt_sequences


DEFAULT_ATTRIBUTES: List[str] = [
    "appearance-changing",
    "box-shaped",
    "box-shaped-flat-init",
]


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PlanarTrack Sequence Annotator</title>
  <style>
    :root {
      --bg0: #0f1115;
      --bg1: #171a20;
      --bg2: #222736;
      --line: #2d3347;
      --txt: #e8edf6;
      --muted: #9aa5be;
      --accent: #7fcf8a;
      --accent2: #4db8d7;
      --danger: #f07a7a;
      --tile-zoom: 1;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--txt);
      font-family: "IBM Plex Sans", "Fira Sans", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(1200px 700px at 0% -10%, #2a3247 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #24353f 0%, transparent 45%),
        var(--bg0);
      min-height: 100vh;
    }

    #app {
      max-width: 1700px;
      margin: 0 auto;
      padding: 14px;
      display: grid;
      gap: 12px;
    }

    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: 12px;
      backdrop-filter: blur(2px);
    }

    #top {
      padding: 12px;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
    }

    #nav {
      display: flex;
      gap: 8px;
      align-items: center;
    }

    button {
      border: 1px solid #42506f;
      color: var(--txt);
      background: #20283a;
      border-radius: 8px;
      padding: 7px 10px;
      cursor: pointer;
      font-weight: 600;
    }

    button:hover { background: #2a3550; }
    button:active { transform: translateY(1px); }

    #seq-name {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }

    #seq-meta {
      color: var(--muted);
      font-family: "JetBrains Mono", "IBM Plex Mono", monospace;
      font-size: 13px;
      margin-top: 4px;
    }

    #zoom-wrap {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    #zoom-slider {
      width: 160px;
      accent-color: var(--accent2);
    }

    #attrs {
      padding: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .attr-btn {
      border: 1px solid #46526f;
      color: #dbe7f8;
      background: #20283a;
      border-radius: 20px;
      padding: 6px 11px;
      font-size: 13px;
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }

    .attr-btn .kbd {
      font-family: "JetBrains Mono", "IBM Plex Mono", monospace;
      font-size: 11px;
      color: #9db4e2;
      border: 1px solid #4e5f88;
      padding: 1px 5px;
      border-radius: 10px;
      background: #1a2132;
    }

    .attr-btn.on {
      background: linear-gradient(180deg, #2d5f3a, #244b30);
      border-color: #5db073;
      color: #e8ffe8;
    }

    #status {
      margin-left: auto;
      color: var(--muted);
      font-size: 13px;
      font-family: "JetBrains Mono", "IBM Plex Mono", monospace;
    }

    #grid {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }

    .frame-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--bg1);
      overflow: hidden;
      min-height: 190px;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .frame-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      padding: 6px 8px;
      border-bottom: 1px solid #283046;
      font-family: "JetBrains Mono", "IBM Plex Mono", monospace;
      font-size: 12px;
      color: var(--muted);
    }

    .init-chip {
      border: 1px solid #5887d8;
      color: #d2e4ff;
      padding: 1px 6px;
      border-radius: 12px;
      background: #1d2a46;
      font-size: 11px;
    }

    .frame-viewport {
      overflow: auto;
      background: #0f131d;
      max-height: 280px;
    }

    .frame-viewport img {
      display: block;
      width: calc(100% * var(--tile-zoom));
      height: auto;
      user-select: none;
    }

    #shortcuts {
      padding: 10px 12px;
      color: var(--muted);
      font-size: 12px;
      font-family: "JetBrains Mono", "IBM Plex Mono", monospace;
      border-top: 1px solid var(--line);
    }

    @media (max-width: 1200px) {
      #grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 760px) {
      #top {
        grid-template-columns: 1fr;
      }
      #zoom-wrap { justify-content: flex-start; }
      #grid { grid-template-columns: 1fr; }
      #status { width: 100%; margin-left: 0; }
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="top" class="panel">
      <div id="nav">
        <button id="btn-prev" title="ArrowLeft">&#8592; Prev</button>
        <button id="btn-next" title="ArrowRight">Next &#8594;</button>
      </div>
      <div>
        <div id="seq-name">Loading...</div>
        <div id="seq-meta"></div>
      </div>
      <div id="zoom-wrap">
        <label for="zoom-slider">Zoom</label>
        <input id="zoom-slider" type="range" min="0.6" max="3.2" step="0.1" value="1.0">
        <span id="zoom-value">1.0x</span>
      </div>
    </div>

    <div id="attrs" class="panel"></div>

    <div id="grid" class="panel"></div>

    <div id="shortcuts" class="panel">
      ArrowLeft/ArrowRight: navigate sequence | Numpad1..Numpad9: toggle attributes | Ctrl+Wheel: zoom
    </div>
  </div>

  <script>
    let sequences = [];
    let attributes = [];
    let currentIndex = 0;
    let currentData = null;

    function shortcutLabel(i) {
      if (i < 9) return 'Numpad' + (i + 1);
      return '---';
    }

    function prettyName(value) {
      return value.replaceAll('-', ' ');
    }

    function setZoom(zoom) {
      const bounded = Math.max(0.6, Math.min(3.2, zoom));
      document.documentElement.style.setProperty('--tile-zoom', String(bounded));
      document.getElementById('zoom-slider').value = String(bounded);
      document.getElementById('zoom-value').textContent = bounded.toFixed(1) + 'x';
    }

    async function init() {
      const stateRes = await fetch('/api/state');
      if (!stateRes.ok) {
        alert('Failed to load state: ' + stateRes.status);
        return;
      }
      const state = await stateRes.json();
      sequences = state.sequences;
      attributes = state.attributes;
      currentIndex = state.current_index;
      renderAttributeButtons({});
      setZoom(1.0);
      await loadSequence(currentIndex);
    }

    function renderAttributeButtons(flags) {
      const root = document.getElementById('attrs');
      root.innerHTML = '';

      attributes.forEach((attr, i) => {
        const btn = document.createElement('button');
        btn.className = 'attr-btn' + (flags[attr] ? ' on' : '');
        btn.dataset.attr = attr;
        btn.innerHTML = '<span>' + prettyName(attr) + '</span>' +
                        '<span class="kbd">' + shortcutLabel(i) + '</span>';
        btn.onclick = () => toggleAttribute(attr);
        root.appendChild(btn);
      });

      const status = document.createElement('div');
      status.id = 'status';
      status.textContent = sequences.length + ' sequences';
      root.appendChild(status);
    }

    function refreshStatus(payload) {
      const status = document.getElementById('status');
      status.textContent = [
        'seq ' + (currentIndex + 1) + '/' + sequences.length,
        'touched ' + payload.touched_sequences + '/' + sequences.length,
        payload.saved_at,
      ].join('  |  ');
    }

    async function loadSequence(index) {
      if (sequences.length === 0) return;
      currentIndex = (index + sequences.length) % sequences.length;

      await fetch('/api/current-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: currentIndex }),
      });

      const res = await fetch('/api/sequence/' + currentIndex);
      if (!res.ok) {
        alert('Failed to load sequence: ' + res.status);
        return;
      }

      currentData = await res.json();
      document.getElementById('seq-name').textContent = currentData.sequence_name;
      document.getElementById('seq-meta').textContent =
        'frames=' + currentData.num_frames +
        '  sample=' + currentData.sampled_frames.length;
      renderAttributeButtons(currentData.attributes);
      refreshStatus(currentData);
      renderGrid(currentData.sampled_frames, currentData.sequence_name);
    }

    function renderGrid(frames, sequenceName) {
      const grid = document.getElementById('grid');
      grid.innerHTML = '';

      frames.forEach(frame => {
        const card = document.createElement('div');
        card.className = 'frame-card';

        const head = document.createElement('div');
        head.className = 'frame-head';
        head.innerHTML = '<span>frame ' + frame.frame_idx + '</span>' +
                         (frame.is_init ? '<span class="init-chip">INIT</span>' : '<span></span>');

        const viewport = document.createElement('div');
        viewport.className = 'frame-viewport';

        const img = document.createElement('img');
        img.loading = 'lazy';
        img.draggable = false;
        img.src = '/api/frame/' + encodeURIComponent(sequenceName) + '/' + frame.frame_idx +
          '.jpg?max_side=1200';
        viewport.appendChild(img);

        card.appendChild(head);
        card.appendChild(viewport);
        grid.appendChild(card);
      });
    }

    async function toggleAttribute(attr) {
      if (!currentData) return;
      const nextValue = !Boolean(currentData.attributes[attr]);
      const res = await fetch('/api/sequence/' + currentIndex + '/attribute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attribute: attr, value: nextValue }),
      });
      if (!res.ok) {
        alert('Failed to save annotation: ' + res.status);
        return;
      }
      const payload = await res.json();
      currentData.attributes = payload.attributes;
      renderAttributeButtons(currentData.attributes);
      refreshStatus(payload);
    }

    document.getElementById('btn-prev').onclick = () => loadSequence(currentIndex - 1);
    document.getElementById('btn-next').onclick = () => loadSequence(currentIndex + 1);
    document.getElementById('zoom-slider').oninput = (e) => setZoom(Number(e.target.value));

    window.addEventListener('wheel', e => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const step = e.deltaY < 0 ? 0.1 : -0.1;
      const now = Number(document.getElementById('zoom-slider').value);
      setZoom(now + step);
    }, { passive: false });

    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        loadSequence(currentIndex - 1);
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        loadSequence(currentIndex + 1);
        return;
      }

      const code = e.code || '';
      let idx = -1;
      if (code.startsWith('Numpad')) {
        idx = parseInt(code.slice(6), 10) - 1;
      } else if (code.startsWith('Digit')) {
        idx = parseInt(code.slice(5), 10) - 1;
      }

      if (idx >= 0 && idx < attributes.length) {
        e.preventDefault();
        toggleAttribute(attributes[idx]);
      }
    });

    init();
  </script>
</body>
</html>
"""


class CurrentIndexUpdate(BaseModel):
    index: int


class AttributeUpdate(BaseModel):
    attribute: str
    value: bool


@dataclass
class SequenceData:
    image_paths: List[Path]
    gt: np.ndarray
    num_frames: int


class AnnotationStore:
    def __init__(self, annotation_file: Path, attributes: List[str]) -> None:
        self.annotation_file = annotation_file
        self.attributes = list(attributes)
        self.lock = threading.Lock()
        self.payload: Dict[str, Any] = {
            "version": 1,
            "attributes": list(attributes),
            "last_sequence_index": 0,
            "annotations": {},
            "updated_at": _utc_now_string(),
        }
        self._load_or_init()

    def _load_or_init(self) -> None:
        if not self.annotation_file.exists():
            self._save_locked()
            return

        text = self.annotation_file.read_text(encoding="utf-8")
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid annotation payload in {self.annotation_file}")

        annotations = loaded.get("annotations", {})
        if not isinstance(annotations, dict):
            raise ValueError(f"Invalid 'annotations' map in {self.annotation_file}")

        merged_attributes = list(
            dict.fromkeys(self.attributes + loaded.get("attributes", []))
        )
        self.attributes = merged_attributes
        self.payload = {
            "version": 1,
            "attributes": merged_attributes,
            "last_sequence_index": int(loaded.get("last_sequence_index", 0)),
            "annotations": annotations,
            "updated_at": loaded.get("updated_at", _utc_now_string()),
        }
        self._save_locked()

    def _save_locked(self) -> None:
        self.annotation_file.parent.mkdir(parents=True, exist_ok=True)
        self.payload["updated_at"] = _utc_now_string()
        tmp_path = self.annotation_file.with_suffix(
            f"{self.annotation_file.suffix}.tmp"
        )
        tmp_path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.annotation_file)

    def get_last_sequence_index(self) -> int:
        with self.lock:
            return int(self.payload.get("last_sequence_index", 0))

    def set_last_sequence_index(self, index: int) -> None:
        with self.lock:
            self.payload["last_sequence_index"] = int(index)
            self._save_locked()

    def get_flags(self, sequence_name: str) -> Dict[str, bool]:
        with self.lock:
            seq_flags = self.payload["annotations"].get(sequence_name, {})
            result = {
                attr: bool(seq_flags.get(attr, False)) for attr in self.attributes
            }
        return result

    def set_flag(
        self, sequence_name: str, attribute: str, value: bool
    ) -> Dict[str, bool]:
        with self.lock:
            annotations = self.payload["annotations"]
            if sequence_name not in annotations:
                annotations[sequence_name] = {}
            annotations[sequence_name][attribute] = bool(value)
            self._save_locked()
            seq_flags = annotations[sequence_name]
            result = {
                attr: bool(seq_flags.get(attr, False)) for attr in self.attributes
            }
        return result

    def touched_sequences(self) -> int:
        with self.lock:
            return len(self.payload["annotations"])

    def saved_at(self) -> str:
        with self.lock:
            return str(self.payload.get("updated_at", ""))


class AnnotatorBackend:
    def __init__(
        self,
        base_dir: Path,
        split: str,
        sample_count: int,
        attributes: List[str],
        annotation_file: Path,
    ) -> None:
        self.base_dir = base_dir
        self.split = split
        self.sample_count = sample_count
        self.attributes = list(attributes)
        self.sequence_names = pt_sequences(base_dir, split)
        if not self.sequence_names:
            raise ValueError(f"No PlanarTrack sequences found for split={split!r}")

        self.cache: Dict[str, SequenceData] = {}
        self.store = AnnotationStore(
            annotation_file=annotation_file, attributes=self.attributes
        )
        self.attributes = self.store.attributes

    def _load_sequence_data(self, sequence_name: str) -> SequenceData:
        if sequence_name in self.cache:
            return self.cache[sequence_name]

        image_paths = pt_image_paths(self.base_dir, sequence_name)
        gt, _valid = pt_load_gt(self.base_dir, sequence_name)
        if gt.ndim != 3 or gt.shape[1:] != (2, 4):
            raise ValueError(
                f"Unexpected GT shape for {sequence_name}: {gt.shape}. Expected (N, 2, 4)."
            )

        num_frames = min(len(image_paths), int(gt.shape[0]))
        if num_frames < 1:
            raise ValueError(f"Sequence {sequence_name} has no usable frames")

        data = SequenceData(
            image_paths=image_paths[:num_frames],
            gt=gt[:num_frames],
            num_frames=num_frames,
        )
        self.cache[sequence_name] = data
        return data

    def _sample_frame_indices(self, num_frames: int) -> List[int]:
        if num_frames < 1:
            raise ValueError(f"Invalid num_frames={num_frames}")

        if num_frames <= self.sample_count:
            return list(range(num_frames))

        sampled = np.linspace(0, num_frames - 1, self.sample_count, dtype=int)
        return [int(i) for i in sampled]

    def sequence_payload(self, seq_index: int) -> Dict[str, Any]:
        idx = seq_index % len(self.sequence_names)
        sequence_name = self.sequence_names[idx]
        data = self._load_sequence_data(sequence_name)
        sampled_frames = self._sample_frame_indices(data.num_frames)

        return {
            "sequence_index": idx,
            "sequence_name": sequence_name,
            "num_frames": data.num_frames,
            "attributes": self.store.get_flags(sequence_name),
            "sampled_frames": [
                {
                    "frame_idx": frame_idx,
                    "is_init": frame_idx == 0,
                }
                for frame_idx in sampled_frames
            ],
            "touched_sequences": self.store.touched_sequences(),
            "saved_at": self.store.saved_at(),
        }

    def frame_jpeg(self, sequence_name: str, frame_idx: int, max_side: int) -> bytes:
        data = self._load_sequence_data(sequence_name)
        if frame_idx < 0 or frame_idx >= data.num_frames:
            raise ValueError(
                f"Frame index out of range for {sequence_name}: {frame_idx} / {data.num_frames}"
            )

        image_path = data.image_paths[frame_idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        quad = data.gt[frame_idx]
        _draw_quad_overlay(image, quad, frame_idx)

        max_side = max(200, max_side)
        h, w = image.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise ValueError(
                f"Failed to JPEG-encode frame for {sequence_name} frame {frame_idx}"
            )
        return encoded.tobytes()


def _draw_quad_overlay(image: np.ndarray, quad: np.ndarray, frame_idx: int) -> None:
    if quad.shape != (2, 4):
        return

    if not np.any(np.isfinite(quad)):
        return

    if not np.any(quad != 0):
        return

    points = rearrange(quad, "xy corners -> corners xy").astype(np.float32)
    points = np.round(points).astype(np.int32)
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=(90, 255, 245),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    for corner_id, corner in enumerate(points):
        center = (int(corner[0]), int(corner[1]))
        cv2.circle(image, center, 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, center, 3, (245, 240, 80), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            str(corner_id),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        f"frame {frame_idx}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _utc_now_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_app(backend: AnnotatorBackend) -> FastAPI:
    app = FastAPI(title="PlanarTrack Sequence Annotator")

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return HTML_PAGE

    @app.get("/api/state", response_class=JSONResponse)
    def get_state() -> Dict[str, Any]:
        raw_index = backend.store.get_last_sequence_index()
        current_index = raw_index % len(backend.sequence_names)
        return {
            "sequences": backend.sequence_names,
            "attributes": backend.attributes,
            "current_index": current_index,
            "touched_sequences": backend.store.touched_sequences(),
            "saved_at": backend.store.saved_at(),
        }

    @app.get("/api/sequence/{seq_index}", response_class=JSONResponse)
    def get_sequence(seq_index: int) -> Dict[str, Any]:
        if len(backend.sequence_names) == 0:
            raise HTTPException(status_code=404, detail="No sequences")
        return backend.sequence_payload(seq_index)

    @app.post("/api/current-index", response_class=JSONResponse)
    def set_current_index(payload: CurrentIndexUpdate) -> Dict[str, Any]:
        if len(backend.sequence_names) == 0:
            raise HTTPException(status_code=404, detail="No sequences")
        idx = payload.index % len(backend.sequence_names)
        backend.store.set_last_sequence_index(idx)
        return {
            "current_index": idx,
            "saved_at": backend.store.saved_at(),
        }

    @app.post("/api/sequence/{seq_index}/attribute", response_class=JSONResponse)
    def update_attribute(seq_index: int, payload: AttributeUpdate) -> Dict[str, Any]:
        if payload.attribute not in backend.attributes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown attribute {payload.attribute!r}. "
                    f"Expected one of: {backend.attributes!r}"
                ),
            )
        idx = seq_index % len(backend.sequence_names)
        sequence_name = backend.sequence_names[idx]
        flags = backend.store.set_flag(sequence_name, payload.attribute, payload.value)
        backend.store.set_last_sequence_index(idx)
        return {
            "sequence_index": idx,
            "sequence_name": sequence_name,
            "attributes": flags,
            "touched_sequences": backend.store.touched_sequences(),
            "saved_at": backend.store.saved_at(),
        }

    @app.get("/api/frame/{sequence_name}/{frame_idx}.jpg")
    def get_frame(
        sequence_name: str,
        frame_idx: int,
        max_side: int = Query(default=1200, ge=200, le=4096),
    ) -> Response:
        try:
            data = backend.frame_jpeg(sequence_name, frame_idx, max_side=max_side)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=data, media_type="image/jpeg")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequence-level binary attribute annotation UI for PlanarTrack"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(paths.PT_BASE),
        help="PlanarTrack dataset root (contains sequences/, annos/, training-test-splits/)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="PlanarTrack split: train/test or path to custom split file",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path(
            "/mnt/personal/serycjon/flatsam2/annotations/planartrack_sequence_attributes.json"
        ),
        help="JSON file with saved sequence-level attributes",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=DEFAULT_ATTRIBUTES,
        help="Binary attributes toggled in UI (Numpad1..Numpad9 by order)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=16,
        help="Frames sampled per sequence for overview grid",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI host")
    parser.add_argument("--port", type=int, default=8010, help="FastAPI port")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {args.sample_count}")
    if len(args.attributes) < 1:
        raise ValueError("Need at least one attribute")

    backend = AnnotatorBackend(
        base_dir=args.base_dir,
        split=args.split,
        sample_count=int(args.sample_count),
        attributes=[str(a) for a in args.attributes],
        annotation_file=args.annotations_file,
    )
    app = create_app(backend)

    print(
        f"Loaded {len(backend.sequence_names)} sequences from split={args.split!r}. "
        f"Saved labels: {args.annotations_file}"
    )
    print(f"Open http://{args.host}:{args.port}")
    hostname = socket.gethostname()
    print(f"ssh -L 8080:{hostname}:{args.port} rci")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
