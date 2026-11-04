"""Reusable HTML/JS components for standalone single-file reports.

Each component is a pair of functions::

    foo_html() -> str   -- HTML fragment to embed in <body>
    foo_js()   -> str   -- JS to embed in a <script> block

The caller substitutes these into their template with simple string replacement.
No external dependencies beyond cv2 / numpy.

Usage pattern in a report script::

    from flatsam2.report_components import seq_preview_html, seq_preview_js

    html = TEMPLATE
    html = html.replace("<!--SEQ_PREVIEW_POPUP-->", seq_preview_html())
    html = html.replace("// __SEQ_PREVIEW_JS__ //", seq_preview_js(fps=2.0))
    # payload must contain: seq_previews: {seq_name: [b64_jpeg, ...]}
    # caller must call SeqPreview.init(data.seq_previews) after data is loaded
    # caller calls SeqPreview.show(seqName, clientX, clientY) on click/hover
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from flatsam2.datasets import pt_image_paths, pti_load_gt


# --------------------------------------------------------------------------- #
# Data builder
# --------------------------------------------------------------------------- #

def build_seq_previews(
    seq_names: list[str],
    base_dir: Path,
    initfix_dir: Path,
    n_frames: int = 10,
    max_side: int = 200,
    jpeg_quality: int = 60,
) -> dict[str, list[str]]:
    """Build animated preview frames (base64 JPEGs) with GT quad overlay.

    Args:
        seq_names: Sequence names to generate previews for.
        base_dir: PlanarTrack base directory.
        initfix_dir: PlanarTrack initfix directory for GT corners.
        n_frames: Frames to include per sequence (equally spaced; always
            includes index 0 and the last frame).
        max_side: Resize so the longest side <= this many pixels.
        jpeg_quality: JPEG encode quality (0–100).

    Returns:
        Mapping seq_name -> list of base64 JPEG strings (empty list on failure).
    """
    previews: dict[str, list[str]] = {}
    for seq in seq_names:
        try:
            gt_corners, valid = pti_load_gt(base_dir, seq, initfix_dir)
            image_paths = pt_image_paths(base_dir, seq)
            T = min(len(gt_corners), len(image_paths))
            if T == 0:
                continue

            if T <= n_frames:
                selected = list(range(T))
            else:
                selected = [round(i * (T - 1) / (n_frames - 1)) for i in range(n_frames)]
                selected = list(dict.fromkeys(selected))

            frames_b64: list[str] = []
            for fi in selected:
                frame = cv2.imread(str(image_paths[fi]))
                if frame is None:
                    continue
                h, w = frame.shape[:2]

                # GT quad overlay: draw on frames that have valid GT corners.
                fi_valid = fi == 0 or (fi < len(valid) and bool(valid[fi]))
                if fi_valid and fi < len(gt_corners):
                    corners = gt_corners[fi]  # (2, 4): row0=x row1=y
                    pts = corners.T.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], True, (0, 220, 60), 4)

                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(
                        frame,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                if ok:
                    frames_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))

            if frames_b64:
                previews[seq] = frames_b64
        except Exception:
            continue
    return previews


# --------------------------------------------------------------------------- #
# SeqPreview popup component
# --------------------------------------------------------------------------- #

def seq_preview_html() -> str:
    """Return the HTML for the animated sequence preview popup.

    Uses ``position:fixed`` so it layers over everything regardless of where
    in the DOM it is placed.  Put the result anywhere inside ``<body>``.

    Element IDs (fixed):
        seq-preview-popup, seq-preview-name, seq-preview-close,
        seq-preview-img, seq-preview-empty, seq-preview-dots
    """
    return (
        '<div id="seq-preview-popup" style="position:fixed;display:none;z-index:1000;'
        "background:rgba(255,255,255,0.96);border:1px solid #bbb;border-radius:4px;"
        'padding:8px;box-shadow:0 2px 10px rgba(0,0,0,0.25);min-width:140px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
        '<div id="seq-preview-name" style="font-weight:bold;font-size:12px;color:#222;'
        'white-space:nowrap;max-width:210px;overflow:hidden;text-overflow:ellipsis"></div>'
        '<button id="seq-preview-close" style="border:none;background:none;cursor:pointer;'
        'font-size:15px;color:#888;padding:0 0 0 8px;line-height:1;flex-shrink:0">&times;</button>'
        "</div>"
        '<img id="seq-preview-img" style="display:none;max-width:240px;max-height:180px;'
        'border:1px solid #eee"/>'
        '<div id="seq-preview-empty" style="color:#888;font-size:11px">&#8212;</div>'
        '<div id="seq-preview-dots" style="display:flex;justify-content:center;gap:4px;'
        'margin-top:6px;min-height:10px"></div>'
        "</div>"
    )


def seq_preview_js(fps: float = 2.0) -> str:
    """Return JS that defines ``window.SeqPreview``.

    Must be included in a ``<script>`` block after the DOM is parsed (or in a
    DOMContentLoaded handler).  Pair with ``seq_preview_html()``.

    Public API::

        SeqPreview.init(previews)          // {seq: [b64jpeg, ...]} — call once
        SeqPreview.show(seq, cx, cy)       // open popup; cx/cy = clientX/Y
        SeqPreview.hide()                  // close popup

    Args:
        fps: Animation frame rate in frames per second.
    """
    interval_ms = round(1000 / fps)
    return f"""\
window.SeqPreview=(function(){{
 var popup=document.getElementById("seq-preview-popup");
 var pName=document.getElementById("seq-preview-name");
 var pImg=document.getElementById("seq-preview-img");
 var pEmpty=document.getElementById("seq-preview-empty");
 var pDots=document.getElementById("seq-preview-dots");
 var pClose=document.getElementById("seq-preview-close");
 var previews={{}},animTimer=null,animIdx=0,animFrames=null;
 function makeDots(n){{
  pDots.innerHTML="";
  for(var i=0;i<n;i++){{
   var d=document.createElement("span");
   d.style.cssText="display:inline-block;width:7px;height:7px;border-radius:50%;background:#ccc;flex-shrink:0";
   pDots.appendChild(d);
  }}
 }}
 function highlightDot(i){{
  Array.from(pDots.children).forEach(function(d,j){{d.style.background=j===i?"#333":"#ccc";}});
 }}
 function stopAnim(){{if(animTimer){{clearInterval(animTimer);animTimer=null;}}}}
 function pos(cx,cy){{
  popup.style.display="block";
  var pw=popup.offsetWidth||260,ph=popup.offsetHeight||220;
  var vw=window.innerWidth,vh=window.innerHeight,gap=14;
  popup.style.left=(cx+gap+pw<vw?cx+gap:Math.max(0,cx-pw-gap))+"px";
  popup.style.top=(cy+gap+ph<vh?cy+gap:Math.max(0,cy-ph-gap))+"px";
 }}
 if(pClose)pClose.addEventListener("click",function(e){{
  e.stopPropagation();stopAnim();popup.style.display="none";
 }});
 return {{
  init:function(p){{previews=p||{{}};}},
  show:function(seq,cx,cy){{
   stopAnim();
   pName.textContent=seq;
   var frames=previews[seq];
   if(frames&&frames.length){{
    animFrames=frames;animIdx=0;
    pImg.src="data:image/jpeg;base64,"+frames[0];
    pImg.style.display="block";pEmpty.style.display="none";
    makeDots(frames.length);highlightDot(0);
    if(frames.length>1){{
     animTimer=setInterval(function(){{
      animIdx=(animIdx+1)%animFrames.length;
      pImg.src="data:image/jpeg;base64,"+animFrames[animIdx];
      highlightDot(animIdx);
     }},{interval_ms});
    }}
   }}else{{
    stopAnim();pImg.style.display="none";pImg.removeAttribute("src");
    pEmpty.textContent=seq?"no preview":"\\u2014";pEmpty.style.display="block";
    pDots.innerHTML="";
   }}
   pos(cx,cy);
  }},
  hide:function(){{stopAnim();popup.style.display="none";}}
 }};
}})();"""
