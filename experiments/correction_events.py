#!/usr/bin/env python3
"""A2 event-conditional analysis: what happens AFTER each SAM3 correction fires?

Global averages hide correction effects (events are sparse); this script pairs a
correction run (any oracle-grid E-cell or the deployed config) with its
no-correction baseline, aligns the two per-frame error trajectories on the
correction events, and reports:

1. **Post-event error deltas** — median/IQR of ``err_corrected(t0+k) −
   err_baseline(t0+k)`` for k = 0..horizon across all events: does a correction
   help downstream, and for how long?
2. **Segmentation persistence** — event-aligned ``corr_iou_gt`` (SAM-vs-GT-mask
   IoU from the corrected run's telemetry): does the corrected mask stick or
   does SAM3 drift back? (Available when the run recorded the new telemetry.)
3. **Re-trigger intervals** — histogram of frames between consecutive events per
   sequence: rapid re-triggering means corrections do not stick (H2).
4. **Per-sequence ΔP@5 vs #corrections** — did sequences that got corrected
   actually improve?
5. **Abstain counts** — ``corr_trigger & ~corrected`` (trigger fired, content
   unavailable / dry run).

Both runs must be finished tracker runs with ``{seq}.txt`` + ``{seq}_debug.npz``.
Events are read from the corrected run's ``corrected`` flags (rerun any config
predating the telemetry fix — old sidecars carry stale all-False flags).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
from einops import rearrange

from flatsam2.config import compute_output_dir
from flatsam2.datasets import get_sequences, pot_load_gt, pti_load_gt
from flatsam2.metrics import alignment_error, precision_at_threshold
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.report_utils import get_plotly_script_tag


def _load_pred_corners(results_dir: Path, seq: str) -> Optional[np.ndarray]:
    """Load predicted corners ``(frames, 2, 4)`` from a run's ``{seq}.txt``."""
    f = results_dir / f"{seq}.txt"
    if not f.exists():
        return None
    flat = np.loadtxt(f)
    if flat.ndim == 1:
        flat = flat[np.newaxis, :]
    return rearrange(flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2)


def _per_frame_errors(
    results_dir: Path,
    seq: str,
    gt_corners: np.ndarray,
    valid: np.ndarray,
) -> Optional[np.ndarray]:
    """Per-frame corner error (px) vs GT, NaN on invalid frames; None if run missing."""
    pred = _load_pred_corners(results_dir, seq)
    if pred is None:
        return None
    T = min(len(gt_corners), len(pred), len(valid))
    err = np.full(T, np.nan)
    idx = np.arange(1, T)[valid[1:T].astype(bool)]
    if len(idx):
        err[idx] = alignment_error(gt_corners[idx], pred[idx])
    return err


def _load_events(results_dir: Path, seq: str) -> Optional[dict]:
    """Correction events + telemetry from a run's ``{seq}_debug.npz``.

    Returns:
        Dict with ``fired`` (list of 0-based frame indices), ``abstains`` (int),
        and ``iou_gt`` (frame_idx -> corr_iou_gt, empty when telemetry absent);
        None when the sidecar is missing.
    """
    npz_path = results_dir / f"{seq}_debug.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    if "corrected" not in data or "frame_idx" not in data:
        return None
    frame_ids = data["frame_idx"]
    corrected = data["corrected"].astype(bool)
    fired = [int(f) for f, c in zip(frame_ids, corrected) if c]
    abstains = 0
    if "corr_trigger" in data:
        trig = data["corr_trigger"].astype(bool)
        abstains = int((trig & ~corrected).sum())
    iou_gt: dict[int, float] = {}
    if "corr_iou_gt" in data:
        for f, v in zip(frame_ids, data["corr_iou_gt"]):
            if np.isfinite(v):
                iou_gt[int(f)] = float(v)
    return {"fired": fired, "abstains": abstains, "iou_gt": iou_gt}


def main() -> None:
    args = build_parser().parse_args()
    label = DATASETS[args.dataset].label
    if args.run_dir is None:
        if args.config is None:
            raise SystemExit("provide --config <corrected run config> or --run-dir")
        args.run_dir = compute_output_dir(paths.OUTPUT_DIRECTORY, args.dataset, args.config)
    if args.baseline_dir is None:
        args.baseline_dir = paths.OUTPUT_DIRECTORY / "loma_samh_decoupled_rerun" / label
    if args.base_dir is None:
        args.base_dir = paths.POT_BASE if args.dataset == "pot" else paths.PT_BASE
    if args.output_dir is None:
        stem = args.config.stem if args.config is not None else args.run_dir.parent.name
        args.output_dir = (
            paths.OUTPUT_DIRECTORY / "experiments" / "correction_events" / stem / label
        )
    if args.dataset == "pot":
        load_gt = pot_load_gt
    else:
        load_gt = lambda bd, sn: pti_load_gt(bd, sn, args.initfix_dir)  # noqa: E731

    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split if args.dataset != "pot" else None,
    )
    print(f"Corrected run: {args.run_dir}")
    print(f"Baseline run : {args.baseline_dir}")
    print(f"Sequences    : {len(sequences)}")

    K = args.horizon
    # delta_curves[k] collects err_corr - err_base at offset k over all events.
    delta_curves: list[list[float]] = [[] for _ in range(K + 1)]
    iou_curves: list[list[float]] = [[] for _ in range(K + 1)]
    intervals: list[int] = []
    seq_scatter: list[dict] = []
    totals = {"events": 0, "abstains": 0, "seqs_with_events": 0, "seqs_skipped": 0}

    for seq in sequences:
        gt_corners, valid = load_gt(args.base_dir, seq)
        err_corr = _per_frame_errors(args.run_dir, seq, gt_corners, valid)
        err_base = _per_frame_errors(args.baseline_dir, seq, gt_corners, valid)
        events = _load_events(args.run_dir, seq)
        if err_corr is None or err_base is None or events is None:
            totals["seqs_skipped"] += 1
            continue
        T = min(len(err_corr), len(err_base))
        fired = [f for f in events["fired"] if f < T]
        totals["events"] += len(fired)
        totals["abstains"] += events["abstains"]
        if fired:
            totals["seqs_with_events"] += 1
            intervals.extend(int(b - a) for a, b in zip(fired, fired[1:]))
        for t0 in fired:
            for k in range(K + 1):
                t = t0 + k
                if t >= T:
                    break
                d = err_corr[t] - err_base[t]
                if np.isfinite(d):
                    delta_curves[k].append(float(d))
                iou = events["iou_gt"].get(t)
                if iou is not None:
                    iou_curves[k].append(iou)

        both = np.isfinite(err_corr[:T]) & np.isfinite(err_base[:T])
        if both.sum() > 0:
            seq_scatter.append(
                {
                    "seq": seq,
                    "n_corrections": len(fired),
                    "dp5": float(
                        precision_at_threshold(err_corr[:T][both], 5.0)
                        - precision_at_threshold(err_base[:T][both], 5.0)
                    ),
                    "dp15": float(
                        precision_at_threshold(err_corr[:T][both], 15.0)
                        - precision_at_threshold(err_base[:T][both], 15.0)
                    ),
                }
            )

    def _quantiles(curve: list[list[float]], q: float) -> list:
        return [float(np.quantile(c, q)) if c else None for c in curve]

    med = _quantiles(delta_curves, 0.5)
    lo = _quantiles(delta_curves, 0.25)
    hi = _quantiles(delta_curves, 0.75)
    iou_med = _quantiles(iou_curves, 0.5)

    def _at(k: int) -> Optional[float]:
        return med[k] if k <= K else None

    summary = {
        **totals,
        "median_delta_err": {f"k{k}": _at(k) for k in (0, 5, 20, min(50, K))},
        "median_retrigger_interval": (float(np.median(intervals)) if intervals else None),
        "mean_dp5_over_corrected_seqs": (
            float(np.mean([s["dp5"] for s in seq_scatter if s["n_corrections"] > 0]))
            if any(s["n_corrections"] > 0 for s in seq_scatter)
            else None
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "n_corrections", "dp5", "dp15"])
        w.writeheader()
        w.writerows(seq_scatter)
    print(json.dumps(summary, indent=2))

    payload = json.dumps(
        {
            "k": list(range(K + 1)),
            "med": med,
            "lo": lo,
            "hi": hi,
            "iou_med": iou_med,
            "intervals": intervals,
            "scatter_x": [s["n_corrections"] for s in seq_scatter],
            "scatter_y": [s["dp5"] for s in seq_scatter],
            "scatter_text": [s["seq"] for s in seq_scatter],
        }
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>correction events</title>
{get_plotly_script_tag()}
<style>body{{font-family:sans-serif;margin:2em}}div.plot{{width:900px;height:450px}}</style>
</head><body>
<h1>Event-conditional correction analysis (A2)</h1>
<p>corrected: <code>{args.run_dir}</code><br>baseline: <code>{args.baseline_dir}</code></p>
<pre>{json.dumps(summary, indent=2)}</pre>
<h2>&Delta;err after a correction (median, IQR band; &lt;0 = correction helped)</h2>
<div class="plot" id="delta"></div>
<h2>SAM-vs-GT mask IoU after a correction (median; needs telemetry run)</h2>
<div class="plot" id="iou"></div>
<h2>Re-trigger intervals (frames between consecutive events)</h2>
<div class="plot" id="intervals"></div>
<h2>Per-sequence &Delta;P@5 vs number of corrections</h2>
<div class="plot" id="scatter"></div>
<script>
var d = {payload};
Plotly.newPlot('delta', [
  {{x: d.k, y: d.hi, line: {{width: 0}}, showlegend: false}},
  {{x: d.k, y: d.lo, fill: 'tonexty', fillcolor: 'rgba(60,120,220,0.2)',
    line: {{width: 0}}, showlegend: false}},
  {{x: d.k, y: d.med, name: 'median', line: {{color: 'rgb(60,120,220)'}}}}
], {{xaxis: {{title: 'frames after event k'}}, yaxis: {{title: 'err_corr - err_base (px)'}}}});
Plotly.newPlot('iou', [{{x: d.k, y: d.iou_med, mode: 'lines'}}],
  {{xaxis: {{title: 'frames after event k'}}, yaxis: {{title: 'IoU(SAM, GT mask)'}}}});
Plotly.newPlot('intervals', [{{x: d.intervals, type: 'histogram'}}],
  {{xaxis: {{title: 'frames to next correction'}}, yaxis: {{title: 'count'}}}});
Plotly.newPlot('scatter', [{{x: d.scatter_x, y: d.scatter_y, text: d.scatter_text,
  mode: 'markers', type: 'scatter'}}],
  {{xaxis: {{title: '#corrections in sequence'}}, yaxis: {{title: 'dP@5 (corr - base)'}},
    hovermode: 'closest'}});
</script>
</body></html>"""
    out_html = args.output_dir / "report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Report: {out_html}")


def build_parser() -> argparse.ArgumentParser:
    """Build the single-shot analysis CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=None,
                   help="corrected run's config file (resolves the run dir)")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="corrected run dir (overrides --config)")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="no-correction baseline run dir "
                        "(default: <OUTPUT>/loma_samh_decoupled_rerun/<label>)")
    p.add_argument("--dataset", choices=["pti", "pot"], default="pti")
    p.add_argument("--base-dir", type=Path, default=None)
    p.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--sequences", nargs="+")
    p.add_argument("--subset")
    p.add_argument("--subset-file", type=Path)
    p.add_argument("--horizon", type=int, default=50,
                   help="frames after each event to trace")
    p.add_argument("--output-dir", type=Path, default=None)
    return p


if __name__ == "__main__":
    main()
