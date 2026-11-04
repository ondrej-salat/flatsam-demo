#!/usr/bin/env python3
"""Explore best and worst homography quality estimation results.

Loads the output of homography_quality_eval.py and identifies:
  - Frames where confidence agrees best with GT (true positives / negatives)
  - Frames where confidence disagrees most with GT (false positives / negatives)
  - Best and worst sequences by Spearman correlation

Produces an HTML report with sortable tables and optional frame visualizations.

Usage:
    # Generate HTML report from eval results
    python experiments/explore_quality.py results.json --output report.html

    # Also generate frame images for top-k best/worst
    python experiments/explore_quality.py results.json --output report.html \\
        --render-frames --top-k 20 --dataset pot --results-dir ~/data/flatsam2/results/
"""

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.datasets import CLI_CHOICES


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_eval_results(eval_json: Path) -> Dict:
    """Load the JSON output of homography_quality_eval.py.

    Args:
        eval_json: Path to the JSON file.

    Returns:
        Parsed dict with ``aggregate``, ``sequences``, ``per_frame`` keys.
    """
    with eval_json.open() as f:
        return json.load(f)


def rank_frames(data: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Classify all frames into agreement/disagreement categories.

    Threshold: confidence ≥ 0.5 = "predicted good"; GT error < 15 px = "actually good".

    Sorting rationale (two-tier):
    - False positives (worst case — we trust a wrong homography): sorted by
      confidence descending, so the most dangerously over-confident frames come
      first.
    - False negatives (we distrust a correct homography): sorted by GT error
      ascending, so the clearest near-perfect poses we missed come first.
    - True positives: sorted by GT error ascending (best alignments first) so
      we can inspect whether high-confidence frames truly are the tightest.
    - True negatives: sorted by confidence ascending (most-certain rejections
      first) — these are the easiest gross errors.

    Args:
        data: Parsed eval result dict.

    Returns:
        Four sorted lists: (true_pos, true_neg, false_pos, false_neg).
    """
    CONF_THRESHOLD = 0.5
    ERROR_THRESHOLD = 15.0

    true_pos: List[Dict] = []
    true_neg: List[Dict] = []
    false_pos: List[Dict] = []
    false_neg: List[Dict] = []

    for seq_name, frames in data.get('per_frame', {}).items():
        for f in frames:
            conf = f['confidence']
            gt_err = f['gt_error']
            entry = {
                'sequence': seq_name,
                'frame_idx': f['frame_idx'],
                'confidence': conf,
                'precision': f.get('precision', float('nan')),
                'alignment_loss': f.get('alignment_loss', float('nan')),
                'frac_aligned_A': f.get('frac_aligned_A', 0.0),
                'frac_occluded_A': f.get('frac_occluded_A', 0.0),
                'frac_interior_A': f.get('frac_interior_A', 0.0),
                'frac_out_of_frame_A': f.get('frac_out_of_frame_A', 0.0),
                'frac_excess_B': f.get('frac_excess_B', 0.0),
                'gt_error': gt_err,
                'explanation': f.get('explanation', ''),
            }
            pred_good = conf >= CONF_THRESHOLD
            gt_good = gt_err < ERROR_THRESHOLD

            if pred_good and gt_good:
                true_pos.append(entry)
            elif not pred_good and not gt_good:
                true_neg.append(entry)
            elif pred_good and not gt_good:
                false_pos.append(entry)
            else:
                false_neg.append(entry)

    # FP: highest confidence first — the most dangerous misses
    false_pos.sort(key=lambda x: -x['confidence'])
    # FN: lowest GT error first — the most obvious misses (near-perfect poses rejected)
    false_neg.sort(key=lambda x: x['gt_error'])
    # TP: lowest GT error first — best alignments at top
    true_pos.sort(key=lambda x: x['gt_error'])
    # TN: lowest confidence first — most clear-cut gross errors
    true_neg.sort(key=lambda x: x['confidence'])

    return true_pos, true_neg, false_pos, false_neg


def rank_sequences(data: Dict) -> List[Dict]:
    """Sort sequences by balanced accuracy (primary) then Spearman r on good frames.

    Balanced accuracy (TPR+TNR)/2 captures gross-regime correctness.
    Spearman r on good frames captures fine-regime ranking quality.
    Sequences with NaN balanced accuracy are sorted last.

    Args:
        data: Parsed eval result dict.

    Returns:
        List of sequence summary dicts, sorted descending.
    """
    seqs = data.get('sequences', [])
    return sorted(
        seqs,
        key=lambda s: (
            s.get('balanced_accuracy', float('-inf'))
            if not np.isnan(s.get('balanced_accuracy', float('nan')))
            else float('-inf'),
            s.get('spearman_r_good', float('-inf'))
            if not np.isnan(s.get('spearman_r_good', float('nan')))
            else float('-inf'),
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _table_html(
    rows: List[Dict],
    columns: List[Tuple[str, str]],
    max_rows: int = 50,
    report_dir: Optional[Path] = None,
) -> str:
    """Render a list of dicts as an HTML table.

    Args:
        rows: List of row dicts.
        columns: List of ``(key, label)`` pairs defining which columns to show.
        max_rows: Maximum rows to emit before truncating.
        report_dir: If provided, rows with an ``'image_path'`` key get an
            inline thumbnail column prepended.

    Returns:
        HTML string for the table.
    """
    has_images = report_dir is not None and any(r.get('image_path') for r in rows[:max_rows])

    lines = [
        '<table border="1" cellpadding="4" cellspacing="0" '
        'style="border-collapse:collapse; font-size:13px;">',
    ]
    lines.append('<tr style="background:#f0f0f0;">')
    if has_images:
        lines.append('  <th>Visualization</th>')
    for col_key, col_label in columns:
        lines.append(f'  <th>{escape(col_label)}</th>')
    lines.append('</tr>')

    for row in rows[:max_rows]:
        lines.append('<tr>')
        if has_images:
            img_path: Optional[Path] = row.get('image_path')
            if img_path is not None and img_path.exists():
                try:
                    rel = img_path.relative_to(report_dir)
                    rel_str = str(rel).replace('\\', '/')
                except ValueError:
                    rel_str = str(img_path)
                lines.append(
                    f'  <td><a href="{escape(rel_str)}" target="_blank">'
                    f'<img src="{escape(rel_str)}" style="max-height:120px; max-width:240px;">'
                    f'</a></td>'
                )
            else:
                lines.append('  <td></td>')
        for col_key, _ in columns:
            val = row.get(col_key, '')
            if isinstance(val, float):
                cell = f'{val:.3f}' if abs(val) < 1000 else f'{val:.1f}'
            else:
                cell = str(val)
            lines.append(f'  <td>{escape(cell)}</td>')
        lines.append('</tr>')

    lines.append('</table>')
    if len(rows) > max_rows:
        lines.append(f'<p><em>… and {len(rows) - max_rows} more rows</em></p>')
    return '\n'.join(lines)


def generate_html_report(
    data: Dict,
    output_path: Path,
    top_k: int = 30,
    image_map: Optional[Dict[str, Dict[Tuple[str, int], Path]]] = None,
) -> None:
    """Generate an HTML report exploring quality estimation results.

    Args:
        data: Output of :func:`load_eval_results`.
        output_path: Where to write the HTML file.
        top_k: Max entries per category.
        image_map: Optional mapping ``{category: {(seq, frame_idx): image_path}}``
            returned by :func:`render_extreme_frames`.
    """
    true_pos, true_neg, false_pos, false_neg = rank_frames(data)
    seq_ranking = rank_sequences(data)
    report_dir = output_path.parent

    agg = data.get('aggregate', {})
    total_frames = sum(len(frames) for frames in data.get('per_frame', {}).values())

    # Attach image paths to frame entries
    if image_map:
        for cat_name, entries in [
            ('false_positives', false_pos),
            ('false_negatives', false_neg),
            ('true_positives', true_pos),
            ('true_negatives', true_neg),
        ]:
            cat_imgs = image_map.get(cat_name, {})
            for entry in entries:
                key = (entry['sequence'], entry['frame_idx'])
                if key in cat_imgs:
                    entry['image_path'] = cat_imgs[key]

    frame_cols = [
        ('sequence',              'Sequence'),
        ('frame_idx',             'Frame'),
        ('confidence',            'Confidence'),
        ('gt_error',              'GT Error (px)'),
        ('precision',             'Precision (px)'),
        ('alignment_loss',        'Align. Loss'),
        ('frac_aligned_A',        '∂A\' aligned'),
        ('frac_occluded_A',       '∂A\' occluded'),
        ('frac_interior_A',       '∂A\' interior'),
        ('frac_out_of_frame_A',   '∂A\' OOF'),
        ('frac_excess_B',         '∂B excess'),
    ]

    seq_cols = [
        ('sequence',             'Sequence'),
        ('balanced_accuracy',    'Bal. Acc. (TPR+TNR)/2'),
        ('tpr',                  'TPR (sensitivity)'),
        ('tnr',                  'TNR (specificity)'),
        ('spearman_r_good',      'Spearman r (good frames)'),
        ('n_good_frames',        '# Good frames'),
        ('n_bad_frames',         '# Bad frames'),
        ('mean_confidence',      'Mean Confidence'),
        ('mean_gt_error',        'Mean GT Error (px)'),
        ('num_frames_evaluated', '# Frames total'),
    ]

    def _fmt(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        if isinstance(val, float):
            return f'{val:.3f}'
        return str(val)

    html_parts = [
        '<!DOCTYPE html>',
        '<html><head>',
        '<meta charset="utf-8">',
        '<title>Homography Quality Estimation Report</title>',
        '<style>',
        '  body { font-family: sans-serif; margin: 20px; }',
        '  h1, h2, h3 { color: #333; }',
        '  .summary { background: #f8f8f8; padding: 15px; border-radius: 5px; margin: 15px 0; }',
        '  .metric-grid { display: grid; grid-template-columns: repeat(3, auto); gap: 6px 20px;'
        '    font-size: 14px; margin: 8px 0; }',
        '  .metric-label { color: #555; }',
        '  .metric-value { font-weight: bold; }',
        '  .good { color: #2d7d2d; }',
        '  .bad { color: #c0392b; }',
        '  table img { display: block; }',
        '  .legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0; font-size: 12px; }',
        '  .legend-item { display: flex; align-items: center; gap: 4px; }',
        '  .swatch { width: 14px; height: 14px; display: inline-block; border: 1px solid #aaa; }',
        '</style>',
        '</head><body>',
        '<h1>Homography Quality Estimation Report</h1>',
        '<div class="summary">',
        f'<p><strong>Total frames evaluated:</strong> {total_frames} across '
        f'{agg.get("num_sequences", "N/A")} sequences</p>',
        # Two-tier metric summary
        '<p><strong>Tier 1 — Gross correctness</strong> '
        '(threshold: confidence ≥ 0.5, GT error &lt; 15 px)</p>',
        '<div class="metric-grid">',
        f'  <span class="metric-label">Balanced accuracy (TPR+TNR)/2</span>'
        f'  <span class="metric-value">{_fmt(agg.get("mean_balanced_accuracy"))}</span>'
        f'  <span></span>',
        f'  <span class="metric-label">Mean TPR (sensitivity — correct good predictions)</span>'
        f'  <span class="metric-value">{_fmt(agg.get("mean_tpr"))}</span>'
        f'  <span></span>',
        f'  <span class="metric-label">Mean TNR (specificity — correct bad predictions)</span>'
        f'  <span class="metric-value">{_fmt(agg.get("mean_tnr"))}</span>'
        f'  <span></span>',
        '</div>',
        '<p><strong>Tier 2 — Fine ranking</strong> '
        '(Spearman r between confidence and GT error, restricted to good frames only)</p>',
        '<div class="metric-grid">',
        f'  <span class="metric-label">Mean Spearman r (good frames)</span>'
        f'  <span class="metric-value">{_fmt(agg.get("mean_spearman_r_good"))}</span>'
        f'  <span></span>',
        f'  <span class="metric-label">Median Spearman r (good frames)</span>'
        f'  <span class="metric-value">{_fmt(agg.get("median_spearman_r_good"))}</span>'
        f'  <span></span>',
        '</div>',
        '<p>',
        f'<strong>Classification counts</strong> (conf ≥ 0.5 vs GT err &lt; 15 px): ',
        f'<span class="good">TP={len(true_pos)}</span>, ',
        f'<span class="good">TN={len(true_neg)}</span>, ',
        f'<span class="bad">FP={len(false_pos)}</span>, ',
        f'<span class="bad">FN={len(false_neg)}</span>',
        '</p>',
        '</div>',
        # Boundary class legend
        '<details><summary><strong>Boundary classification legend</strong></summary>',
        '<div class="legend">',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(50,220,50)"></span>∂A\' / ∂B ALIGNED</div>',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(30,100,220)"></span>∂A\' OCCLUDED (outside B)</div>',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(230,50,50)"></span>∂A\' INTERIOR (inside B — contradictory)</div>',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(130,130,130)"></span>∂A\' OUT_OF_FRAME</div>',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(255,140,30)"></span>∂B EXCESS (outside A\' — segmenter overshoot)</div>',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(50,200,220)"></span>∂B OCCL_EDGE (inside A\' — occlusion boundary)</div>',
        '  <div class="legend-item"><span class="swatch" style="background:rgb(220,220,50)"></span>∂B FRAME_EDGE (touching image border)</div>',
        '</div>',
        '</details>',
    ]

    # Sequence ranking
    html_parts.append(
        '<h2>Sequence Ranking</h2>'
        '<p>Sorted by balanced accuracy (primary), then Spearman r on good frames (secondary). '
        'TPR = sensitivity (catching correct homographies), '
        'TNR = specificity (catching wrong ones).</p>'
    )
    html_parts.append(_table_html(seq_ranking, seq_cols, max_rows=200))

    # False positives — gross regime failure: we trust a wrong homography
    html_parts.append('<h2 class="bad">False Positives (confidence ≥ 0.5, GT error ≥ 15 px)</h2>')
    html_parts.append(
        f'<p>Most dangerous: we trust the homography when it is actually wrong. '
        f'Sorted by confidence descending — the most over-confident errors first. '
        f'<strong>{len(false_pos)} frames total.</strong></p>'
    )
    html_parts.append(_table_html(false_pos, frame_cols, max_rows=top_k, report_dir=report_dir))

    # False negatives — gross regime failure: we reject a correct homography
    html_parts.append('<h2 class="bad">False Negatives (confidence &lt; 0.5, GT error &lt; 15 px)</h2>')
    html_parts.append(
        f'<p>We reject a correct homography. '
        f'Sorted by GT error ascending — the clearest near-perfect poses we rejected first. '
        f'<strong>{len(false_neg)} frames total.</strong></p>'
    )
    html_parts.append(_table_html(false_neg, frame_cols, max_rows=top_k, report_dir=report_dir))

    # True positives — fine regime: check confidence ranks well within good frames
    html_parts.append('<h2 class="good">True Positives (correctly predicted good)</h2>')
    html_parts.append(
        f'<p>Sorted by GT error ascending — inspect whether the tightest alignments '
        f'also get the highest confidence (Tier 2 ranking quality). '
        f'<strong>{len(true_pos)} frames total.</strong></p>'
    )
    html_parts.append(_table_html(true_pos, frame_cols, max_rows=top_k, report_dir=report_dir))

    # True negatives — gross regime success: we correctly rejected bad homographies
    html_parts.append('<h2 class="good">True Negatives (correctly predicted bad)</h2>')
    html_parts.append(
        f'<p>Sorted by confidence ascending — the most clear-cut gross errors first. '
        f'<strong>{len(true_neg)} frames total.</strong></p>'
    )
    html_parts.append(_table_html(true_neg, frame_cols, max_rows=top_k, report_dir=report_dir))

    html_parts.append('</body></html>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(html_parts))
    print(f"Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Optional: render frame images for the extreme cases
# ---------------------------------------------------------------------------

def render_extreme_frames(
    data: Dict,
    dataset: str,
    base_results_dir: Path,
    output_dir: Path,
    top_k: int = 10,
    base_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
) -> Dict[str, Dict[Tuple[str, int], Path]]:
    """Render visualization images for the most interesting frames.

    Args:
        data: Parsed eval result dict.
        dataset: Dataset identifier.
        base_results_dir: Root directory of tracker outputs.
        output_dir: Directory where rendered images will be saved.
        top_k: How many frames per category to render.
        base_dir: Dataset root directory override.
        config_path: Tracker config path for output sub-directory resolution.
        initfix_dir: Required for ``'pti'`` GT loader.

    Returns:
        Nested mapping ``{category_name: {(seq, frame_idx): image_path}}``.
    """
    from visualize_quality import visualize_single_frame

    true_pos, true_neg, false_pos, false_neg = rank_frames(data)

    categories = [
        ('false_positives', false_pos),
        ('false_negatives', false_neg),
        ('true_positives', true_pos),
        ('true_negatives', true_neg),
    ]

    image_map: Dict[str, Dict[Tuple[str, int], Path]] = {}

    for cat_name, frames in categories:
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_map: Dict[Tuple[str, int], Path] = {}
        for i, entry in enumerate(frames[:top_k]):
            seq = entry['sequence']
            fidx = entry['frame_idx']
            out_path = cat_dir / f"{i:03d}_{seq}_frame{fidx:05d}.png"
            try:
                visualize_single_frame(
                    dataset, seq, fidx,
                    base_results_dir, base_dir, config_path, initfix_dir,
                    output_path=out_path,
                )
                cat_map[(seq, fidx)] = out_path
            except Exception as e:
                print(f"  Could not render {seq} frame {fidx}: {e}")
        image_map[cat_name] = cat_map

    return image_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Explore homography quality estimation results"
    )
    parser.add_argument('eval_json', type=Path, help="Path to homography_quality_eval.json")
    parser.add_argument(
        '--output', type=Path, default=Path('quality_report.html'),
        help="Output HTML report path",
    )
    parser.add_argument('--top-k', type=int, default=30, help="Entries per category")

    parser.add_argument('--render-frames', action='store_true',
                        help="Also render vis images for extreme cases")
    parser.add_argument('--dataset', choices=CLI_CHOICES,
                        help="Dataset (required for --render-frames)")
    parser.add_argument('--results-dir', type=Path,
                        help="Results dir (required for --render-frames)")
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--initfix-dir', type=Path)
    parser.add_argument('--render-output-dir', type=Path,
                        help="Directory for rendered frame images")

    args = parser.parse_args(argv)

    data = load_eval_results(args.eval_json)

    image_map = None
    if args.render_frames:
        if args.dataset is None or args.results_dir is None:
            print("ERROR: --dataset and --results-dir are required for --render-frames")
            return 1
        render_dir = args.render_output_dir or args.output.parent / 'quality_frames'
        image_map = render_extreme_frames(
            data, args.dataset, args.results_dir, render_dir,
            top_k=args.top_k,
            base_dir=args.base_dir,
            config_path=args.config,
            initfix_dir=args.initfix_dir,
        )

    generate_html_report(data, args.output, top_k=args.top_k, image_map=image_map)
    return 0


if __name__ == '__main__':
    sys.exit(main())
