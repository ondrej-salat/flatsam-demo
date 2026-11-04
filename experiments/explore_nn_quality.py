#!/usr/bin/env python3
"""HTML report for nn_quality_eval.py results.

Generates a self-contained HTML report with:
- Aggregate metrics summary
- Sequences sorted by balanced accuracy (primary) and Spearman r (secondary)
- Per-sequence collapsible sections with per-frame results
- Optional rendered frame comparisons: warped-ref mask | seg mask | overlap

Usage:
    # Report only (no images)
    python experiments/explore_nn_quality.py results/nn_quality_eval_best.json

    # With rendered frame images
    python experiments/explore_nn_quality.py results/nn_quality_eval_best.json \\
        --render-frames --dataset pot --results-dir ~/data/flatsam2/results/ \\
        --output report/index.html

    # Render only specific sequences
    python experiments/explore_nn_quality.py results/nn_quality_eval_best.json \\
        --render-frames --dataset pot --results-dir ~/data/ --sequences seq1 seq2 \\
        --output report/index.html
"""

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    pot_load_gt,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.homography_quality import homography_from_corners
from flatsam2.path_config import paths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GT_ERROR_THRESHOLD: float = 15.0

# Classification colors (BGR for cv2)
_CLS_BGR = {
    'TP': (0,  200,   0),   # green
    'TN': (150, 150, 150),  # grey
    'FP': (0,    0, 230),   # red
    'FN': (0,  165, 255),   # orange
}
_CLS_HTML = {
    'TP': '#00c800',
    'TN': '#969696',
    'FP': '#e60000',
    'FN': '#ffa500',
}
_CLS_LONG = {
    'TP': 'TP (pred good, GT good)',
    'TN': 'TN (pred bad, GT bad)',
    'FP': 'FP (pred good, GT bad) ⚠',
    'FN': 'FN (pred bad, GT good)',
}

_CSS = """
body { font-family: sans-serif; margin: 20px; background: #f0f0f0; }
h1, h2 { color: #222; }
.summary { background: #fff; padding: 15px; border-radius: 8px; margin: 15px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,.12); }
.metric-grid { display: grid; grid-template-columns: auto auto; gap: 4px 24px;
               font-size: 14px; margin: 8px 0; }
.metric-label { color: #555; }
.metric-value { font-weight: bold; }
.good { color: #2d7d2d; }
.bad  { color: #c0392b; }
table { border-collapse: collapse; font-size: 13px; background: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 8px; }
th { background: #e0e0e0; padding: 5px 10px; text-align: left; white-space: nowrap; }
td { padding: 3px 10px; border-top: 1px solid #eee; vertical-align: middle; }
tr:hover td { background: #f0f8ff; }
a { color: #1a73e8; text-decoration: none; }
a:hover { text-decoration: underline; }
details { margin: 6px 0; }
summary { cursor: pointer; padding: 7px 12px; background: #fff; border-radius: 6px;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 11px; color: #888; }
details[open] summary::before { content: "▼ "; }
summary:hover { background: #f0f0f0; }
.seq-section { padding: 8px 4px; overflow-x: auto; }
.frame-img { max-height: 90px; max-width: 270px; display: block; cursor: pointer; }
.cls-TP { color: #2d7d2d; font-weight: bold; }
.cls-TN { color: #666; }
.cls-FP { color: #c0392b; font-weight: bold; }
.cls-FN { color: #e67e22; font-weight: bold; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_frame(predicted_rmse: float, gt_error: float) -> str:
    """Return 'TP', 'TN', 'FP', or 'FN' for one frame.

    Args:
        predicted_rmse: NN predicted RMSE in pixels.
        gt_error: GT alignment error in pixels.

    Returns:
        Two-character classification string.
    """
    pred_good = predicted_rmse < GT_ERROR_THRESHOLD
    gt_good = gt_error < GT_ERROR_THRESHOLD
    if pred_good and gt_good:
        return 'TP'
    if not pred_good and not gt_good:
        return 'TN'
    if pred_good and not gt_good:
        return 'FP'
    return 'FN'


def _fmt(val, digits: int = 3) -> str:
    if val is None:
        return 'N/A'
    if isinstance(val, float) and np.isnan(val):
        return 'N/A'
    if isinstance(val, float):
        return f'{val:.{digits}f}'
    return str(val)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame_comparison(
    init_mask: np.ndarray,
    seg_mask: np.ndarray,
    H: np.ndarray,
    predicted_rmse: float,
    sigma: float,
    gt_error: float,
    frame_class: str,
    target_height: int = 240,
) -> np.ndarray:
    """Render a 3-panel comparison image for one frame.

    Left panel:  reference mask warped by H into the current frame.
    Center panel: raw segmenter mask.
    Right panel:  overlap (warped=red channel, seg=green channel).

    Border color encodes classification (green=TP, grey=TN, red=FP, orange=FN).

    Args:
        init_mask: Reference mask from frame 0 (grayscale uint8).
        seg_mask: Segmenter mask for the current frame (grayscale uint8).
        H: (3, 3) homography matrix (init frame → current frame).
        predicted_rmse: NN predicted RMSE in pixels.
        sigma: NN uncertainty in pixels.
        gt_error: GT alignment error in pixels.
        frame_class: One of 'TP', 'TN', 'FP', 'FN'.
        target_height: Output image height in pixels.

    Returns:
        BGR uint8 composite image of shape ``(target_height, ~3*w_scaled, 3)``.
    """
    h, w = seg_mask.shape[:2]
    warped = cv2.warpPerspective(init_mask, H, (w, h), flags=cv2.INTER_NEAREST)

    scale = target_height / h
    tw = max(1, int(w * scale))
    th = target_height

    def _s(img):
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    warped_s = _s(warped)
    seg_s = _s(seg_mask)

    # Overlap panel: warped → red, seg → green
    overlap = np.zeros((th, tw, 3), dtype=np.uint8)
    overlap[:, :, 2] = warped_s  # R
    overlap[:, :, 1] = seg_s     # G

    gap = 2
    comp_w = tw * 3 + gap * 2
    comp = np.zeros((th, comp_w, 3), dtype=np.uint8)
    comp[:, :tw] = cv2.cvtColor(warped_s, cv2.COLOR_GRAY2BGR)
    comp[:, tw + gap: 2 * tw + gap] = cv2.cvtColor(seg_s, cv2.COLOR_GRAY2BGR)
    comp[:, 2 * tw + 2 * gap:] = overlap

    # Classification border
    border_color = _CLS_BGR[frame_class]
    cv2.rectangle(comp, (0, 0), (comp_w - 1, th - 1), border_color, 3)

    # Metrics text in top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"pred={predicted_rmse:.1f}\xb1{sigma:.1f}  gt={gt_error:.1f}  [{frame_class}]"
    (tx, ty), _ = cv2.getTextSize(text, font, 0.42, 1)
    cv2.rectangle(comp, (0, 0), (tx + 6, ty + 6), (0, 0, 0), cv2.FILLED)
    cv2.putText(comp, text, (3, ty + 3), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel labels at bottom
    for label, x0 in [("warped ref", 4), ("seg mask", tw + gap + 4), ("overlap", 2 * tw + 2 * gap + 4)]:
        (lx, ly), _ = cv2.getTextSize(label, font, 0.35, 1)
        cv2.rectangle(comp, (x0, th - ly - 5), (x0 + lx + 4, th - 1), (0, 0, 0), cv2.FILLED)
        cv2.putText(comp, label, (x0 + 2, th - 4), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return comp


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _make_load_gt_fn(dataset: str, initfix_dir: Optional[Path]):
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir must be provided for the 'pti' GT loader")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def _load_seq_masks(
    masks_dir: Path,
) -> Tuple[Optional[np.ndarray], Dict[int, np.ndarray]]:
    """Load the reference mask and all per-frame segmenter masks.

    File naming: ``{file_num:05d}.png`` where ``file_num = frame_idx + 1``.
    The init mask (file 1, frame_idx 0) is returned separately.

    Args:
        masks_dir: Directory containing the mask PNG files.

    Returns:
        ``(init_mask, {frame_idx: mask})`` where frame_idx >= 1.
    """
    init_mask = None
    seg_masks: Dict[int, np.ndarray] = {}

    if not masks_dir.exists():
        return None, {}

    for p in sorted(masks_dir.glob("*.png")):
        file_num = int(p.stem)           # 1-indexed
        frame_idx = file_num - 1         # 0-indexed
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if frame_idx == 0:
            init_mask = mask
        else:
            seg_masks[frame_idx] = mask

    return init_mask, seg_masks


def render_sequence_frames(
    data: Dict,
    seq_name: str,
    results_dir: Path,
    base_dir: Path,
    output_dir: Path,
    load_gt_fn,
    force: bool = False,
) -> Dict[int, Path]:
    """Render comparison PNGs for every evaluated frame in *seq_name*.

    Skips frames whose output PNG already exists unless *force* is True.

    Args:
        data: Parsed nn_quality_eval JSON.
        seq_name: Sequence name.
        results_dir: Tracker output directory (contains masks/ and *.txt).
        base_dir: Dataset root (for GT corners).
        output_dir: Root directory to write rendered PNGs.
        load_gt_fn: Callable ``(base_dir, seq_name) → (gt_corners, valid)``.
        force: If True, re-render even if the output PNG already exists.

    Returns:
        ``{frame_idx: image_path}`` for successfully rendered frames.
    """
    frame_results = data.get('per_frame', {}).get(seq_name, [])
    if not frame_results:
        return {}

    seq_out = output_dir / seq_name

    # Collect already-rendered frames without loading any heavy data
    rendered: Dict[int, Path] = {}
    if not force:
        for fd in frame_results:
            p = seq_out / f"{fd['frame_idx']:05d}.png"
            if p.exists():
                rendered[fd['frame_idx']] = p
        if len(rendered) == len(frame_results):
            print(f"  all {len(rendered)} frames already rendered, skipping")
            return rendered

    remaining = [fd for fd in frame_results if fd['frame_idx'] not in rendered]

    gt_corners, _ = load_gt_fn(base_dir, seq_name)
    init_corners = gt_corners[0]

    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    pred_corners = rearrange(pred_flat, 'f (c xy) -> f xy c', c=4, xy=2)

    masks_dir = results_dir / 'masks' / seq_name
    init_mask, seg_masks = _load_seq_masks(masks_dir)
    if init_mask is None:
        raise FileNotFoundError(f"Init mask not found in {masks_dir}")

    seq_out.mkdir(parents=True, exist_ok=True)

    for fd in remaining:
        fidx = fd['frame_idx']
        seg_mask = seg_masks.get(fidx)
        if seg_mask is None or fidx >= len(pred_corners):
            continue
        H = homography_from_corners(init_corners, pred_corners[fidx])
        if H is None:
            continue
        cls = classify_frame(fd['predicted_rmse'], fd['gt_error'])
        try:
            img = render_frame_comparison(
                init_mask, seg_mask, H,
                fd['predicted_rmse'], fd['sigma'], fd['gt_error'], cls,
            )
            out_path = seq_out / f"{fidx:05d}.png"
            cv2.imwrite(str(out_path), img)
            rendered[fidx] = out_path
        except Exception as exc:
            print(f"    frame {fidx}: render failed — {exc}")

    return rendered


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

SPEARMAN_BIN_SIZE: float = 2.0  # px


def recompute_binned_spearman(data: Dict, bin_size: float = SPEARMAN_BIN_SIZE) -> None:
    """Recompute spearman_r_good in-place using binned values.

    Overwrites ``spearman_r_good`` / ``spearman_p_good`` on every sequence dict
    and updates the aggregate mean/median, so the rest of the report uses the
    binned version without needing to re-run the eval script.

    Args:
        data: Parsed nn_quality_eval JSON (mutated in place).
        bin_size: Bin width in pixels; values within a bin get equal rank.
    """
    per_frame = data.get('per_frame', {})
    for seq in data.get('sequences', []):
        frames = per_frame.get(seq['sequence'], [])
        good = [f for f in frames if f['gt_error'] < GT_ERROR_THRESHOLD]
        if len(good) >= 3:
            pred = np.array([f['predicted_rmse'] for f in good])
            gt   = np.array([f['gt_error']        for f in good])
            r, p = stats.spearmanr(-np.floor(pred / bin_size),
                                   -np.floor(gt   / bin_size))
            seq['spearman_r_good'] = float(r)
            seq['spearman_p_good'] = float(p)
        else:
            seq['spearman_r_good'] = float('nan')
            seq['spearman_p_good'] = float('nan')

    # Update aggregate
    agg = data.get('aggregate', {})
    vals = [s['spearman_r_good'] for s in data.get('sequences', [])
            if not np.isnan(s['spearman_r_good'])]
    agg['mean_spearman_r_good']   = float(np.mean(vals))   if vals else float('nan')
    agg['median_spearman_r_good'] = float(np.median(vals)) if vals else float('nan')


def rank_sequences(data: Dict) -> List[Dict]:
    """Return sequences sorted by balanced accuracy (desc) then Spearman r.

    Args:
        data: Parsed eval result dict.

    Returns:
        Sorted list of sequence summary dicts (from ``data['sequences']``).
    """
    def _sort_key(s):
        bal = s.get('balanced_accuracy', float('nan'))
        spr = s.get('spearman_r_good', float('nan'))
        if isinstance(bal, float) and not np.isnan(bal):
            bal_v = bal
        elif s.get('n_bad_frames', 1) == 0:
            # All frames are good — TNR undefined, but TPR is the full story
            tpr = s.get('tpr', float('nan'))
            bal_v = tpr if (isinstance(tpr, float) and not np.isnan(tpr)) else 1.0
        else:
            bal_v = float('-inf')
        spr_v = spr if (isinstance(spr, float) and not np.isnan(spr)) else float('-inf')
        return (bal_v, spr_v)

    return sorted(data.get('sequences', []), key=_sort_key, reverse=True)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _seq_summary_row(seq: Dict, rank: int) -> str:
    name = seq['sequence']
    bal = seq.get('balanced_accuracy', float('nan'))
    bal_str = _fmt(bal)
    if isinstance(bal, float) and not np.isnan(bal):
        color = '#2d7d2d' if bal >= 0.7 else ('#e67e22' if bal >= 0.5 else '#c0392b')
    else:
        color = '#999'
    return (
        f'<tr>'
        f'<td>{rank}</td>'
        f'<td><a href="#{escape(name)}">{escape(name)}</a></td>'
        f'<td style="color:{color};font-weight:bold">{bal_str}</td>'
        f'<td>{_fmt(seq.get("tpr"))}</td>'
        f'<td>{_fmt(seq.get("tnr"))}</td>'
        f'<td>{_fmt(seq.get("spearman_r_good"))}</td>'
        f'<td>{seq.get("n_good_frames", 0)}</td>'
        f'<td>{seq.get("n_bad_frames", 0)}</td>'
        f'<td>{_fmt(seq.get("mean_predicted_rmse"), 1)}</td>'
        f'<td>{_fmt(seq.get("mean_gt_error"), 1)}</td>'
        f'</tr>\n'
    )


def _frame_row(
    frame: Dict,
    report_dir: Path,
    image_map: Optional[Dict[str, Dict[int, Path]]],
    seq_name: str,
) -> str:
    fidx = frame['frame_idx']
    pred = frame['predicted_rmse']
    sigma = frame['sigma']
    gt = frame['gt_error']
    cls = classify_frame(pred, gt)

    img_td = ''
    if image_map and seq_name in image_map and fidx in image_map[seq_name]:
        p = image_map[seq_name][fidx]
        try:
            rel = str(p.relative_to(report_dir)).replace('\\', '/')
        except ValueError:
            rel = str(p)
        img_td = (
            f'<td><a href="{escape(rel)}" target="_blank">'
            f'<img class="frame-img" src="{escape(rel)}" loading="lazy">'
            f'</a></td>'
        )

    return (
        f'<tr>'
        f'<td>{fidx}</td>'
        f'<td>{pred:.2f}</td>'
        f'<td>{sigma:.2f}</td>'
        f'<td>{gt:.2f}</td>'
        f'<td class="cls-{cls}">{escape(_CLS_LONG[cls])}</td>'
        f'{img_td}'
        f'</tr>\n'
    )


def generate_html_report(
    data: Dict,
    output_path: Path,
    image_map: Optional[Dict[str, Dict[int, Path]]] = None,
    max_seq_details: int = 50,
    max_frames_per_seq: int = 50,
) -> None:
    """Write the HTML report to *output_path*.

    Args:
        data: Parsed nn_quality_eval JSON.
        output_path: Destination HTML file.
        image_map: Optional ``{seq_name: {frame_idx: image_path}}``.
        max_seq_details: How many sequences to include expandable detail sections
            for (worst first — the interesting ones). The rest appear only in the
            summary table.
        max_frames_per_seq: Maximum frame rows per detail section.
    """
    agg = data.get('aggregate', {})
    ranked = rank_sequences(data)
    per_frame = data.get('per_frame', {})
    report_dir = output_path.parent

    total_frames = sum(len(v) for v in per_frame.values())
    counts: Dict[str, int] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for frames in per_frame.values():
        for f in frames:
            counts[classify_frame(f['predicted_rmse'], f['gt_error'])] += 1

    p = []  # html parts

    checkpoint_str = str(agg.get('checkpoint', 'N/A'))
    p += [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8">',
        f'<title>NN Quality — {escape(Path(checkpoint_str).name)}</title>',
        f'<style>{_CSS}</style>',
        '</head><body>',
        '<h1>NN Homography Quality Evaluation Report</h1>',
    ]

    # Aggregate summary box
    p += [
        '<div class="summary">',
        f'<p><strong>Checkpoint:</strong> {escape(checkpoint_str)}</p>',
        f'<p><strong>Architecture:</strong> {escape(str(agg.get("arch", "N/A")))}</p>',
        f'<p><strong>Sequences:</strong> {agg.get("num_sequences", 0)} evaluated, '
        f'{agg.get("num_failed", 0)} failed &nbsp;|&nbsp; '
        f'<strong>Frames:</strong> {total_frames}</p>',
        '<div class="metric-grid">',
        f'<span class="metric-label">Balanced accuracy (mean)</span>'
        f'<span class="metric-value">{_fmt(agg.get("mean_balanced_accuracy"))}</span>',
        f'<span class="metric-label">Mean TPR (sensitivity)</span>'
        f'<span class="metric-value">{_fmt(agg.get("mean_tpr"))}</span>',
        f'<span class="metric-label">Mean TNR (specificity)</span>'
        f'<span class="metric-value">{_fmt(agg.get("mean_tnr"))}</span>',
        f'<span class="metric-label">Mean Spearman r (good frames)</span>'
        f'<span class="metric-value">{_fmt(agg.get("mean_spearman_r_good"))}</span>',
        f'<span class="metric-label">Median Spearman r (good frames)</span>'
        f'<span class="metric-value">{_fmt(agg.get("median_spearman_r_good"))}</span>',
        '</div>',
        f'<p>'
        f'<span class="good">TP={counts["TP"]}</span> &nbsp;'
        f'<span class="good">TN={counts["TN"]}</span> &nbsp;'
        f'<span class="bad">FP={counts["FP"]}</span> &nbsp;'
        f'<span class="bad">FN={counts["FN"]}</span>'
        f' &nbsp;— threshold: pred_rmse &lt; {GT_ERROR_THRESHOLD:.0f} px = "good"</p>',
        '</div>',
    ]

    # Sequence ranking table
    p += [
        '<h2>Sequence Rankings</h2>',
        '<table>',
        '<tr><th>#</th><th>Sequence</th><th>Bal. Acc.</th><th>TPR</th><th>TNR</th>'
        '<th>Spearman r</th><th>Good frames</th><th>Bad frames</th>'
        '<th>Mean pred RMSE</th><th>Mean GT error</th></tr>',
    ]
    for rank, seq in enumerate(ranked, 1):
        p.append(_seq_summary_row(seq, rank))
    p.append('</table>')

    # Per-sequence detail sections — worst sequences first, capped at max_seq_details
    detail_seqs = ranked[::-1][:max_seq_details]  # worst → best, limited

    p += [
        '<h2>Per-Sequence Details</h2>',
        f'<p><em>Showing the {len(detail_seqs)} worst sequences '
        f'(of {len(ranked)} total). '
        'Click a sequence name to expand per-frame results '
        f'(up to {max_frames_per_seq} frames shown; FP first, then FN, then TP, TN). '
        'Images: warped ref mask | seg mask | overlap (red=warped, green=seg). '
        'Border: <span style="color:#00c800">green=TP</span>, '
        '<span style="color:#969696">grey=TN</span>, '
        '<span style="color:#e60000">red=FP</span>, '
        '<span style="color:#ffa500">orange=FN</span>.</em></p>',
    ]

    has_images = image_map is not None

    for seq in detail_seqs:
        seq_name = seq['sequence']
        frames = per_frame.get(seq_name, [])
        if not frames:
            continue

        bal = _fmt(seq.get('balanced_accuracy'))
        tpr = _fmt(seq.get('tpr'))
        tnr = _fmt(seq.get('tnr'))
        spr = _fmt(seq.get('spearman_r_good'))
        ng = seq.get('n_good_frames', 0)
        nb = seq.get('n_bad_frames', 0)

        sorted_frames = sorted(frames, key=lambda f: f['frame_idx'])
        shown = sorted_frames if max_frames_per_seq <= 0 else sorted_frames[:max_frames_per_seq]
        truncated = len(frames) - len(shown)

        p += [
            f'<details id="{escape(seq_name)}">',
            f'<summary>{escape(seq_name)}'
            f' &nbsp; bal_acc={bal}'
            f' &nbsp; tpr={tpr}  tnr={tnr}'
            f' &nbsp; spearman_r={spr}'
            f' &nbsp; good={ng}  bad={nb}'
            f'</summary>',
            '<div class="seq-section">',
            '<table>',
            '<tr><th>Frame</th><th>Pred RMSE (px)</th><th>Sigma (px)</th>'
            '<th>GT Error (px)</th><th>Classification</th>',
        ]
        if has_images:
            p.append('<th>Visualization</th>')
        p.append('</tr>')

        for frame in shown:
            p.append(_frame_row(frame, report_dir, image_map, seq_name))

        p += ['</table>']
        if truncated:
            p.append(f'<p><em>… {truncated} more frames not shown.</em></p>')
        p += ['</div>', '</details>']

    # Failed sequences
    failed = data.get('failed', [])
    if failed:
        p += [
            '<h2 class="bad">Failed Sequences</h2>',
            '<table><tr><th>Sequence</th><th>Error</th></tr>',
        ]
        for entry in failed:
            p.append(
                f'<tr><td>{escape(entry["sequence"])}</td>'
                f'<td>{escape(entry["error"])}</td></tr>'
            )
        p.append('</table>')

    p.append('</body></html>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(p), encoding='utf-8')
    print(f"Report written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="HTML report for nn_quality_eval.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('eval_json', type=Path,
                        help="Path to nn_quality_eval_*.json")
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('nn_quality_report.html'),
                        help="Output HTML file (default: nn_quality_report.html)")
    parser.add_argument('--max-seq-details', type=int, default=50, metavar='N',
                        help="Detail sections for the N worst sequences (default: 50)")
    parser.add_argument('--max-frames-per-seq', type=int, default=0, metavar='N',
                        help="Max frame rows per detail section (default: 50; 0 or -1 = unlimited)")

    render_grp = parser.add_argument_group('frame rendering (optional)')
    render_grp.add_argument('--render-frames', action='store_true',
                            help="Render per-frame comparison images alongside the report")
    render_grp.add_argument('--force-render', action='store_true',
                            help="Re-render frames even if the PNG already exists")
    render_grp.add_argument('--dataset', choices=CLI_CHOICES,
                            help="Dataset name (required for --render-frames)")
    render_grp.add_argument('--results-dir', type=Path,
                            help="Tracker results directory (required for --render-frames)")
    render_grp.add_argument('--base-dir', type=Path,
                            help="Dataset root directory override")
    render_grp.add_argument('--config', type=Path,
                            help="Tracker config path (for output sub-dir resolution)")
    render_grp.add_argument('--initfix-dir', type=Path,
                            help="PlanarTrack initfix directory (for 'pti' GT loader)")
    render_grp.add_argument('--frames-dir', type=Path,
                            help="Where to write rendered PNGs "
                                 "(default: <output_dir>/frames/)")
    render_grp.add_argument('--sequences', nargs='+',
                            help="Render only these sequences (default: all)")
    render_grp.add_argument('--render-n', type=int, default=None, metavar='N',
                            help="Render at most N sequences (worst first)")

    args = parser.parse_args(argv)

    with args.eval_json.open() as fh:
        data = json.load(fh)

    recompute_binned_spearman(data)

    image_map: Optional[Dict[str, Dict[int, Path]]] = None

    if args.render_frames:
        if args.dataset is None or args.results_dir is None:
            parser.error("--dataset and --results-dir are required for --render-frames")

        ds_info = DATASETS[args.dataset]
        base_dir: Path = args.base_dir or getattr(paths, ds_info.base_dir_attr)
        initfix_dir: Optional[Path] = args.initfix_dir
        if ds_info.gt_loader == "pti" and initfix_dir is None:
            initfix_dir = paths.PT_INITFIX_DIR

        load_gt_fn = _make_load_gt_fn(args.dataset, initfix_dir)
        results_dir = compute_output_dir(args.results_dir, args.dataset, args.config)
        frames_dir = args.frames_dir or (args.output.parent / 'frames')

        if args.sequences:
            seqs_to_render = args.sequences
        else:
            # Worst sequences first so early saves are the most interesting ones
            seqs_to_render = [s['sequence'] for s in rank_sequences(data)[::-1]]
            if args.render_n is not None:
                seqs_to_render = seqs_to_render[: args.render_n]

        image_map = {}

        # Save the report immediately so the user can open it right away
        generate_html_report(data, args.output, image_map=image_map,
                             max_seq_details=args.max_seq_details,
                             max_frames_per_seq=args.max_frames_per_seq)

        for i, seq_name in enumerate(seqs_to_render, 1):
            print(f"[{i}/{len(seqs_to_render)}] Rendering {seq_name} ...")
            try:
                rendered = render_sequence_frames(
                    data, seq_name, results_dir, base_dir, frames_dir, load_gt_fn,
                    force=args.force_render,
                )
                if rendered:
                    image_map[seq_name] = rendered
                    print(f"  {len(rendered)} frames written")
            except Exception as exc:
                print(f"  FAILED — {exc}")

            # Re-save after every sequence so the report reflects what's done so far
            generate_html_report(data, args.output, image_map=image_map,
                                 max_seq_details=args.max_seq_details,
                                 max_frames_per_seq=args.max_frames_per_seq)

    else:
        generate_html_report(data, args.output, image_map=image_map,
                             max_seq_details=args.max_seq_details,
                             max_frames_per_seq=args.max_frames_per_seq)

    return 0


if __name__ == '__main__':
    sys.exit(main())
