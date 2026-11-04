#!/usr/bin/env python3
"""A0 error-budget analysis: how much can SAM3 correction (or better fusion) buy?

Runs entirely offline on an EXISTING no-correction run (default:
``loma_samh_decoupled_rerun``) — no tracker rerun. Per frame it joins the
``{seq}_debug.npz`` candidate homographies with GT corners and the precomputed
SAM3 masks, then decomposes the error mass to answer, before any online oracle
experiment:

1. **Selection stats** — how often is LoMa vs SAM-H selected, and how good is
   each conditional on being selected.
2. **Selection ceiling** — P@5/P@15 of the run vs a per-frame oracle picking the
   best stored candidate (``loma_H``/``samh_H``/``mask_H``): the headroom of any
   smarter selection rule, within this run.
3. **Error-mass decomposition** — failed frames split by (selected branch x SAM
   mask state), where the mask state vs the GT-induced quad is one of
   ``ok / spill / shrink / both / jump``. "Spill & samh-selected" is the only
   cell an over-seg correction can ever recover: its size bounds the mechanism.
4. **Per-sequence headroom + shortlist** — sequences ranked by correction
   headroom (spill frames with a correct LoMa available) and selection headroom
   (failed frames where a stored candidate was fine), written both as CSV and as
   a plain subset file consumable by ``run.py --subset-file`` for cheap
   single-sequence oracle iteration.

Mask-state definitions (vs the GT-corner quad mask, boundary tolerance T px):
    jump   IoU(SAM, GT) < iou-jump         (SAM tracks something else)
    spill  dout_gt_T >= frac               (SAM area deep outside GT: over-seg)
    shrink din_gt_T >= frac                (GT interior uncovered: occlusion/loss)
    both   spill and shrink
    ok     none of the above

Subcommands:
    dump    Per-frame join -> per-sequence CSV (mask IO; run on the HPC).
    report  CSVs -> stdout markdown tables + standalone HTML + shortlist files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange

from flatsam2.datasets import (
    corners_to_mask,
    get_sequences,
    pot_image_paths,
    pot_load_gt,
    pt_image_paths,
    pti_load_gt,
    warp_corners,
)
from flatsam2.mask_metrics import cropped_signed_area_discrepancy
from flatsam2.metrics import alignment_error, precision_at_threshold
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.report_utils import get_plotly_script_tag


# Homography candidates read from the debug npz, in preference order for err_min.
_CANDIDATES = ("loma_H", "samh_H", "mask_H")

# Boundary tolerance (px) for the spill/shrink depth features (matches the
# sam_overseg_correction study's label threshold and the deployed trigger).
_T_STATE = 10

CSV_HEADER = [
    "frame_idx",
    "evaluated",       # valid GT + prediction row present
    "selected",        # candidate label the run used ('' when debug row missing)
    "corrected",       # correction fired (should be all-0 on the baseline substrate)
    "n_inliers",
    "err_loma",
    "err_samh",
    "err_mask_h",
    "err_selected",    # from the run's {seq}.txt — includes fallbacks/symmetry
    "err_min",         # min over available candidate errors
    "masked",          # SAM mask PNG was available for this frame
    "iou_gt",
    f"dout_gt_T{_T_STATE}",
    f"din_gt_T{_T_STATE}",
]


# --------------------------------------------------------------------------- #
# dataset plumbing
# --------------------------------------------------------------------------- #


def _dataset_fns(
    dataset: str,
    base_dir: Path,
    initfix_dir: Path,
) -> Tuple[Callable, Callable]:
    """Return ``(load_gt, image_paths)`` callables for 'pot' / 'pti'."""
    if dataset == "pot":
        return pot_load_gt, pot_image_paths
    if dataset == "pti":
        return (lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)), pt_image_paths
    raise ValueError(f"unsupported dataset {dataset!r} (use 'pot' or 'pti')")


def _load_pred_corners(results_dir: Path, seq: str) -> Optional[np.ndarray]:
    """Load predicted corners ``(frames, 2, 4)`` from the run's ``{seq}.txt``."""
    f = results_dir / f"{seq}.txt"
    if not f.exists():
        return None
    flat = np.loadtxt(f)
    if flat.ndim == 1:
        flat = flat[np.newaxis, :]
    return rearrange(flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2)


def _load_debug_full(results_dir: Path, seq: str) -> Optional[dict[int, dict]]:
    """Per-frame debug row keyed by 0-based frame index.

    Returns:
        ``frame_idx -> {candidate: (3,3) H or None, 'selected': str,
        'corrected': bool, 'n_inliers': int}`` or None when the sidecar is
        missing.
    """
    npz_path = results_dir / f"{seq}_debug.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    if "frame_idx" not in data:
        return None
    frame_ids = data["frame_idx"]
    n = len(frame_ids)
    out: dict[int, dict] = {}
    for k in range(n):
        row: dict = {}
        for cand in _CANDIDATES:
            H = data[cand][k] if cand in data else np.full((3, 3), np.nan)
            row[cand] = H if np.isfinite(H).all() else None
        row["selected"] = str(data["selected"][k]) if "selected" in data else ""
        row["corrected"] = bool(data["corrected"][k]) if "corrected" in data else False
        row["n_inliers"] = int(data["n_inliers"][k]) if "n_inliers" in data else -1
        out[int(frame_ids[k])] = row
    return out


def _load_mask(path: Path) -> Optional[np.ndarray]:
    """Load a binary mask PNG as bool (H, W), or None if missing/unreadable."""
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else (raw > 0)


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #


def _candidate_error(
    H: Optional[np.ndarray],
    init_corners: np.ndarray,
    gt_frame_corners: np.ndarray,
) -> float:
    """Corner alignment error (px) of one candidate H, NaN when unavailable."""
    if H is None:
        return float("nan")
    return float(alignment_error(gt_frame_corners, warp_corners(H, init_corners)))


def _dump_sequence(
    seq: str,
    load_gt: Callable,
    image_paths_fn: Callable,
    base_dir: Path,
    masks_dir: Path,
    results_dir: Path,
    out_csv: Path,
) -> int:
    """Join debug npz + GT + predictions + masks into one per-frame CSV.

    Returns:
        Number of evaluated frames written.
    """
    gt_corners, valid = load_gt(base_dir, seq)
    image_paths = image_paths_fn(base_dir, seq)
    pred = _load_pred_corners(results_dir, seq)
    debug = _load_debug_full(results_dir, seq)
    if pred is None or debug is None:
        raise FileNotFoundError(f"missing {seq}.txt or {seq}_debug.npz in {results_dir}")

    T = min(len(gt_corners), len(valid), len(image_paths), len(pred))
    init_corners = gt_corners[0]

    rows: list[list] = []
    n_eval = 0
    for i in range(1, T):  # frame 0 = template, never evaluated
        evaluated = bool(valid[i])
        rec = debug.get(i)
        selected = rec["selected"] if rec is not None else ""
        corrected = int(rec["corrected"]) if rec is not None else 0
        n_inl = rec["n_inliers"] if rec is not None else -1

        err_cand = {
            cand: (
                _candidate_error(rec[cand], init_corners, gt_corners[i])
                if evaluated and rec is not None
                else float("nan")
            )
            for cand in _CANDIDATES
        }
        err_selected = (
            float(alignment_error(gt_corners[i], pred[i])) if evaluated else float("nan")
        )
        finite = [e for e in err_cand.values() if np.isfinite(e)]
        err_min = float(min(finite)) if finite else float("nan")

        masked = 0
        iou_gt = dout_gt = din_gt = float("nan")
        if evaluated:
            sam_mask = _load_mask(masks_dir / seq / f"{i + 1:05d}.png")
            if sam_mask is not None:
                masked = 1
                h_img, w_img = sam_mask.shape
                gt_mask = corners_to_mask(gt_corners[i], (h_img, w_img)) > 0
                union = int((sam_mask | gt_mask).sum())
                if union > 0:
                    iou_gt = float(int((sam_mask & gt_mask).sum()) / union)
                if gt_mask.any():
                    dout, din = cropped_signed_area_discrepancy(
                        gt_mask, sam_mask, [_T_STATE]
                    )
                    dout_gt, din_gt = float(dout[_T_STATE]), float(din[_T_STATE])
        if evaluated:
            n_eval += 1

        rows.append(
            [
                i,
                int(evaluated),
                selected,
                corrected,
                n_inl,
                err_cand["loma_H"],
                err_cand["samh_H"],
                err_cand["mask_H"],
                err_selected,
                err_min,
                masked,
                iou_gt,
                dout_gt,
                din_gt,
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in r])
    tmp.replace(out_csv)
    return n_eval


def cmd_dump(args: argparse.Namespace) -> None:
    """Run the per-frame join over a sequence list."""
    load_gt, image_paths_fn = _dataset_fns(args.dataset, args.base_dir, args.initfix_dir)
    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split if args.dataset != "pot" else None,
    )
    out_dir = args.output_dir / "per_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset    : {args.dataset} ({DATASETS[args.dataset].label})")
    print(f"Sequences  : {len(sequences)}")
    print(f"Masks dir  : {args.masks_dir}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir : {out_dir}")

    total_eval = 0
    failed: list[str] = []
    for k, seq in enumerate(sequences, 1):
        out_csv = out_dir / f"{seq}.csv"
        if out_csv.exists() and args.skip_completed:
            print(f"[{k}/{len(sequences)}] skip {seq} (exists)")
            continue
        try:
            n = _dump_sequence(
                seq,
                load_gt,
                image_paths_fn,
                args.base_dir,
                args.masks_dir,
                args.results_dir,
                out_csv,
            )
            total_eval += n
            print(f"[{k}/{len(sequences)}] {seq}: {n} eval frames")
        except Exception as e:  # noqa: BLE001 - keep going, record failure
            failed.append(seq)
            print(f"[{k}/{len(sequences)}] {seq}: FAILED ({e})")
    print(f"\nDone. eval frames={total_eval} failed={len(failed)}")
    if failed:
        print("failed:", " ".join(failed))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def _load_frames(per_seq_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate evaluated frames from all per-sequence CSVs into columns."""
    numeric = [c for c in CSV_HEADER if c != "selected"]
    cols: dict[str, list] = {c: [] for c in numeric}
    selected_col: list[str] = []
    seq_col: list[str] = []
    files = sorted(per_seq_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-sequence CSVs in {per_seq_dir}")
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as f:
            for d in csv.DictReader(f):
                if int(d["evaluated"]) != 1:
                    continue
                for c in numeric:
                    cols[c].append(float(d[c]))
                selected_col.append(d["selected"])
                seq_col.append(path.stem)
    out = {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}
    out["selected"] = np.asarray(selected_col, dtype=object)
    out["seq"] = np.asarray(seq_col, dtype=object)
    return out


def _mask_state(
    iou_gt: np.ndarray,
    dout_gt: np.ndarray,
    din_gt: np.ndarray,
    masked: np.ndarray,
    iou_jump: float,
    frac: float,
) -> np.ndarray:
    """Classify each frame's SAM mask vs the GT quad (see module docstring)."""
    state = np.full(len(iou_gt), "nomask", dtype=object)
    has = masked > 0
    state[has] = "ok"
    jump = has & (iou_gt < iou_jump)
    spill = has & ~jump & (dout_gt >= frac)
    shrink = has & ~jump & (din_gt >= frac)
    state[spill & shrink] = "both"
    state[spill & ~shrink] = "spill"
    state[shrink & ~spill] = "shrink"
    state[jump] = "jump"
    return state


def _md_table(header: list[str], rows: list[list]) -> str:
    """Render a simple pipe table (stdout + HTML-embeddable)."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _group_metrics(err: np.ndarray) -> list[str]:
    """[n, p@5, p@15, mean, median] formatted for a table row."""
    if len(err) == 0:
        return ["0", "-", "-", "-", "-"]
    finite = err[np.isfinite(err)]
    if len(finite) == 0:
        return [str(len(err)), "-", "-", "-", "-"]
    return [
        str(len(err)),
        f"{precision_at_threshold(finite, 5.0):.1f}",
        f"{precision_at_threshold(finite, 15.0):.1f}",
        f"{np.mean(finite):.1f}",
        f"{np.median(finite):.1f}",
    ]


def cmd_report(args: argparse.Namespace) -> None:
    """CSVs -> stdout tables, HTML report, shortlist CSV + subset file."""
    data = _load_frames(args.per_seq_dir)
    n_frames = len(data["frame_idx"])
    err_sel = data["err_selected"]
    err_min = data["err_min"]
    err_loma = data["err_loma"]
    state = _mask_state(
        data["iou_gt"],
        data[f"dout_gt_T{_T_STATE}"],
        data[f"din_gt_T{_T_STATE}"],
        data["masked"],
        args.iou_jump,
        args.frac,
    )
    sections: list[str] = []

    # 1. selection stats -----------------------------------------------------
    rows = []
    for label in sorted(set(data["selected"])):
        sel = data["selected"] == label
        rows.append([label or "(missing)", *_group_metrics(err_sel[sel])])
    rows.append(["ALL", *_group_metrics(err_sel)])
    rows.append(["oracle(min of stored H)", *_group_metrics(err_min)])
    t1 = _md_table(["selected", "frames", "p@5", "p@15", "mean", "median"], rows)
    sections.append(("Selection stats + within-run selection ceiling", t1))

    # 2. error-mass decomposition (selected branch x mask state) --------------
    miss5 = np.isfinite(err_sel) & (err_sel > 5.0)
    rows = []
    for label in sorted(set(data["selected"])):
        for st in ("ok", "spill", "shrink", "both", "jump", "nomask"):
            cell = (data["selected"] == label) & (state == st)
            n_cell = int(cell.sum())
            if n_cell == 0:
                continue
            n_miss = int((cell & miss5).sum())
            mass = float(np.minimum(err_sel[cell & np.isfinite(err_sel)], 40.0).sum())
            recoverable = int((cell & miss5 & (err_loma <= 5.0)).sum())
            rows.append([label, st, n_cell, n_miss, f"{mass:.0f}", recoverable])
    t2 = _md_table(
        [
            "selected",
            "mask state",
            "frames",
            "p@5 misses",
            "err mass (capped 40)",
            "misses w/ loma<=5 (recoverable)",
        ],
        rows,
    )
    sections.append(("Error-mass decomposition (frames, misses, headroom)", t2))

    # 3. per-sequence headroom + shortlist ------------------------------------
    seq_rows: list[tuple] = []
    for seq in sorted(set(data["seq"])):
        s = data["seq"] == seq
        spill_frames = int(((state == "spill") & s).sum())
        corr_headroom = int(
            ((state == "spill") & s & np.isfinite(err_loma) & (err_loma <= 5.0)).sum()
        )
        sel_headroom = int(
            (s & np.isfinite(err_sel) & (err_sel > 5.0) & (err_min <= 5.0)).sum()
        )
        n_miss = int((s & miss5).sum())
        seq_rows.append((seq, int(s.sum()), n_miss, spill_frames, corr_headroom, sel_headroom))
    seq_rows.sort(key=lambda r: (r[4] + r[5]), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shortlist_csv = args.output_dir / "shortlist.csv"
    with shortlist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["sequence", "frames", "p5_misses", "spill_frames",
             "correction_headroom", "selection_headroom"]
        )
        w.writerows(seq_rows)
    subset_path = args.output_dir / f"shortlist_top{args.top_n}.txt"
    subset_path.write_text(
        "\n".join(r[0] for r in seq_rows[: args.top_n]) + "\n", encoding="utf-8"
    )

    t3 = _md_table(
        ["sequence", "frames", "p@5 misses", "spill", "corr headroom", "sel headroom"],
        [list(r) for r in seq_rows[: args.top_n]],
    )
    sections.append((f"Deep-dive shortlist (top {args.top_n}; full list in shortlist.csv)", t3))

    # 4. headline summary ------------------------------------------------------
    summary = {
        "frames": n_frames,
        "p5": float(precision_at_threshold(err_sel[np.isfinite(err_sel)], 5.0)),
        "p15": float(precision_at_threshold(err_sel[np.isfinite(err_sel)], 15.0)),
        "oracle_p5": float(precision_at_threshold(err_min[np.isfinite(err_min)], 5.0)),
        "oracle_p15": float(precision_at_threshold(err_min[np.isfinite(err_min)], 15.0)),
        "spill_frames": int((state == "spill").sum()),
        "jump_frames": int((state == "jump").sum()),
        "recoverable_misses": int((miss5 & (state == "spill") & (err_loma <= 5.0)).sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for title, table in sections:
        print(f"\n## {title}\n\n{table}")
    print(f"\nSummary: {json.dumps(summary)}")
    print(f"Shortlist subset file: {subset_path}")

    # 5. HTML (tables + per-sequence headroom scatter) -------------------------
    scatter_json = json.dumps(
        {
            "x": [r[3] for r in seq_rows],
            "y": [r[4] for r in seq_rows],
            "text": [r[0] for r in seq_rows],
        }
    )
    tables_html = "".join(
        f"<h2>{title}</h2><pre>{table}</pre>" for title, table in sections
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>correction headroom</title>
{get_plotly_script_tag()}
<style>body{{font-family:sans-serif;margin:2em}}pre{{font-size:12px}}</style>
</head><body>
<h1>Correction error budget (A0)</h1>
<p>substrate: <code>{args.per_seq_dir}</code></p>
<pre>{json.dumps(summary, indent=2)}</pre>
{tables_html}
<h2>Per-sequence: spill frames vs correction headroom</h2>
<div id="sc" style="width:900px;height:600px"></div>
<script>
var d = {scatter_json};
Plotly.newPlot('sc', [{{x: d.x, y: d.y, text: d.text, mode: 'markers', type: 'scatter'}}],
  {{xaxis: {{title: 'spill frames'}}, yaxis: {{title: 'correction headroom (spill & loma<=5)'}},
    hovermode: 'closest'}});
</script>
</body></html>"""
    out_html = args.output_dir / "report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Report: {out_html}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the dump/report CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset", choices=["pti", "pot"], default="pti",
        help="dataset key (default pti; pot needs --masks-dir/--results-dir to match)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def out_default(dataset: str) -> Path:
        return (
            paths.OUTPUT_DIRECTORY / "experiments" / "correction_headroom"
            / DATASETS[dataset].label
        )

    d = sub.add_parser("dump", help="per-frame join -> per-sequence CSV")
    d.add_argument("--results-dir", type=Path, default=None,
                   help="baseline run dir with {seq}.txt + {seq}_debug.npz "
                        "(default: <OUTPUT>/loma_samh_decoupled_rerun/<label>)")
    d.add_argument("--masks-dir", type=Path, default=None,
                   help="precomputed SAM3 masks (default: path_config per dataset)")
    d.add_argument("--base-dir", type=Path, default=None)
    d.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    d.add_argument("--split", choices=["train", "test"], default="test")
    d.add_argument("--sequences", nargs="+")
    d.add_argument("--subset")
    d.add_argument("--subset-file", type=Path)
    d.add_argument("--output-dir", type=Path, default=None)
    d.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    d.set_defaults(func=cmd_dump)

    r = sub.add_parser("report", help="per-sequence CSVs -> tables + HTML + shortlist")
    r.add_argument("--per-seq-dir", type=Path, default=None)
    r.add_argument("--output-dir", type=Path, default=None)
    r.add_argument("--iou-jump", type=float, default=0.1,
                   help="IoU below this = jump state")
    r.add_argument("--frac", type=float, default=0.10,
                   help="depth fraction for spill/shrink states")
    r.add_argument("--top-n", type=int, default=20, help="shortlist length")
    r.set_defaults(func=cmd_report)
    return p


def _fill_defaults(args: argparse.Namespace) -> None:
    """Resolve dataset-dependent path defaults left as None by the parser."""
    label = DATASETS[args.dataset].label
    out_default = paths.OUTPUT_DIRECTORY / "experiments" / "correction_headroom" / label
    if getattr(args, "output_dir", None) is None:
        args.output_dir = out_default
    if getattr(args, "per_seq_dir", None) is None and args.cmd == "report":
        args.per_seq_dir = out_default / "per_sequence"
    if args.cmd == "dump":
        if args.results_dir is None:
            args.results_dir = (
                paths.OUTPUT_DIRECTORY / "loma_samh_decoupled_rerun" / label
            )
        if args.masks_dir is None:
            args.masks_dir = (
                paths.SAM3_PRECOMPUTED_MASKS_POT
                if args.dataset == "pot"
                else paths.SAM3_PRECOMPUTED_MASKS_PTI
            )
        if args.base_dir is None:
            args.base_dir = paths.POT_BASE if args.dataset == "pot" else paths.PT_BASE


def main() -> None:
    args = build_parser().parse_args()
    _fill_defaults(args)
    args.func(args)


if __name__ == "__main__":
    main()
