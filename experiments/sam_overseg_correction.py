#!/usr/bin/env python3
"""Feasibility study: correcting SAM3 *over-segmentation* with LoMa homographies.

Target failure mode: SAM3 grows to cover a larger part of the object than the
tracked plane (e.g. the whole box instead of its front face). Two questions:

1. **Detect reliably?** Does a mask-vs-mask signal between the SAM mask and the
   LoMa-induced mask (``warp(init_mask, loma_H)``) separate true over-segmentation
   from everything else — using only signals available at inference time
   (``n_inliers`` + corr-vs-SAM geometry, no GT)?
2. **Fixable without collateral?** When the signal fires, is the LoMa estimate
   actually correct (so injecting its mask helps), and how often would it fire on
   a frame whose SAM mask was already fine (collateral damage)?

Why over-segmentation is the tractable case: the label is ``d_out`` = SAM area
lying > T px *outside* the GT plane mask. Occlusion shrinks SAM, out-of-frame
cannot inflate SAM\\GT, and the box's non-planar faces land squarely in SAM\\GT.
So ``d_out``-vs-GT is an honest "SAM over-segments" oracle, unlike the under-seg
direction which is confounded by occlusion.

Three mask pairs (cf. design discussion):
    SAM vs GT      -> LABEL: is SAM over-segmenting (table row)
    corr vs GT     -> LABEL: is LoMa correct (via corner alignment error)
    corr vs SAM    -> FEATURE: the only mask signal the trigger sees at inference

Substrate is the **no-correction** baseline (``loma_samh_decoupled_rerun``): its SAM
mask stream is unmutated (= the shared precomputed SAM3 masks), and its
``{seq}_debug.npz`` stores ``loma_H``/``n_inliers`` every frame regardless of
selection — so the whole study is offline, no rerun.

Subcommands:
    dump    Per-frame continuous features -> per-sequence CSV (heavy mask IO).
    report  CSVs -> standalone HTML with the 2x3 table, detection PR curves,
            live hyperparameter sliders, and grid-searched optimal defaults.
            With ``--config <deployed correction config>`` the localization map
            gains an expected/actual toggle: the static-analysis expectation vs
            where correction *actually* fired in the real run and whether it
            helped or harmed corner error against the no-correction baseline.
    viz     Render how the metrics count on a single (reference, other) mask pair
            (counted excess/deficit pixels + normal-probe coverage) -> PNG.
    compare Diff a correction run's *live* SAM3 masks against the study substrate
            (precomputed masks) per frame. Pre-fire frames should match; an IoU
            gap there means the study analysed masks the real run never saw.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from einops import rearrange
from tqdm import tqdm

from flatsam2.config import compute_output_dir, load_config_from_file
from flatsam2.report_components import (
    build_seq_previews,
    seq_preview_html,
    seq_preview_js,
)
from flatsam2.datasets import (
    corners_to_mask,
    get_sequences,
    pt_image_paths,
    pti_load_gt,
    warp_corners,
)
from flatsam2.mask_metrics import (
    cropped_signed_area_discrepancy,
    outline_normal_coverage,
    signed_area_discrepancy,
    signed_distance,
)
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS


# Boundary tolerance grid (px). A pixel counts as "outside region R" only when it
# is farther than T from R, so T ignores few-px boundary jitter. Both the GT
# over-seg label and the corr-vs-SAM feature are stored at every T -> T stays a
# slider in the report.
T_GRID: tuple[int, ...] = (0, 2, 5, 10, 15, 20, 30)

# Per-T columns, signed (out = over-seg, in = under-seg) for each reference
# (gt, corr), in two metric types:
#   d{out,in}_{gt,corr} = area DEPTH, vs the reference signed distance field:
#       out = SAM area with sdf_ref > T (deep outside the reference boundary);
#       in  = uncovered area with sdf_ref < −T (deep inside the reference). Both
#       as a fraction of the *reference* area.
#   b{out,in}_{gt,corr} = boundary COVERAGE: fraction of the reference outline
#       where SAM is present T px outward along the contour normal (out) or absent
#       T px inward (in) — a directional normal probe. Robust to thin lateral
#       spills and to wide/neck overshoots that an isotropic distance would cap.
_FAMILIES = [
    "dout_gt", "din_gt", "dout_corr", "din_corr",
    "bout_gt", "bin_gt", "bout_corr", "bin_corr",
]
_T_COLS = [f"{fam}_T{t}" for fam in _FAMILIES for t in T_GRID]
CSV_HEADER = [
    "frame_idx",
    "evaluated",
    "n_inliers",
    "loma_align_err",
    "n_corners_oob",
    "area_sam",
    "area_gt",
    "area_corr",
    "containment",  # frac of corr lying outside SAM (corr spill / superset guard)
    *_T_COLS,
]


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #


def _load_mask(path: Path) -> Optional[np.ndarray]:
    """Load a binary mask PNG as bool (H, W), or None if missing/unreadable."""
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else (raw > 0)


def _load_debug(results_dir: Path, seq: str) -> Optional[dict[int, tuple]]:
    """Load per-frame ``(loma_H, n_inliers)`` keyed by 0-based frame index.

    Args:
        results_dir: Baseline run dir holding ``{seq}_debug.npz``.
        seq: Sequence name.

    Returns:
        Mapping ``frame_idx -> (loma_H (3,3) or None, n_inliers int)``, or None
        when the sidecar is absent or lacks a ``loma_H`` field.
    """
    npz_path = results_dir / f"{seq}_debug.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    if "loma_H" not in data or "frame_idx" not in data:
        return None
    frame_ids = data["frame_idx"]
    loma_H = data["loma_H"]
    n_inliers = data["n_inliers"] if "n_inliers" in data else np.full(len(frame_ids), -1)
    out: dict[int, tuple] = {}
    for k, fid in enumerate(frame_ids):
        H = loma_H[k]
        out[int(fid)] = (None if not np.isfinite(H).all() else H, int(n_inliers[k]))
    return out


def _dump_sequence(
    seq: str,
    base_dir: Path,
    initfix_dir: Path,
    masks_dir: Path,
    results_dir: Path,
    out_csv: Path,
) -> int:
    """Compute and write per-frame features for one sequence.

    Returns:
        Number of evaluated (written, masked, loma-valid) frames.
    """
    gt_corners, valid = pti_load_gt(base_dir, seq, initfix_dir)
    image_paths = pt_image_paths(base_dir, seq)
    T = min(len(gt_corners), len(valid), len(image_paths))
    gt_corners, valid = gt_corners[:T], valid[:T]
    debug = _load_debug(results_dir, seq)
    if debug is None:
        raise FileNotFoundError(f"no usable debug npz for {seq}")

    # init_mask: the runtime template warped by loma_H. No-correction baseline ->
    # frame-0 SAM mask == GT init mask; fall back to the GT quad if absent.
    init_mask = _load_mask(masks_dir / seq / "00001.png")
    init_corners = gt_corners[0]

    rows: list[list] = []
    n_eval = 0
    for i in range(1, T):  # frame 0 is the template, never evaluated
        evaluated = False
        n_inl = -1
        loma_err = float("nan")
        n_oob = -1
        sam_area = gt_area = corr_area = 0
        containment = float("nan")
        t_vals = {c: float("nan") for c in _T_COLS}

        sam_mask = _load_mask(masks_dir / seq / f"{i + 1:05d}.png")
        rec = debug.get(i)
        loma_H = rec[0] if rec is not None else None
        if rec is not None:
            n_inl = rec[1]

        if bool(valid[i]) and sam_mask is not None and loma_H is not None:
            h_img, w_img = sam_mask.shape
            if init_mask is None:
                init_mask = corners_to_mask(init_corners, (h_img, w_img)) > 0
            gt_mask = corners_to_mask(gt_corners[i], (h_img, w_img)) > 0
            corr_mask = cv2.warpPerspective(
                init_mask.astype(np.uint8), loma_H, (w_img, h_img)
            ) > 0

            sam_area = int(sam_mask.sum())
            gt_area = int(gt_mask.sum())
            corr_area = int(corr_mask.sum())
            if sam_area > 0 and corr_area > 0:
                evaluated = True
                n_eval += 1
                loma_err = float(
                    alignment_error(gt_corners[i], warp_corners(loma_H, init_corners))
                )
                cx, cy = gt_corners[i][0], gt_corners[i][1]
                n_oob = int(
                    ((cx < 0) | (cx >= w_img) | (cy < 0) | (cy >= h_img)).sum()
                )
                containment = float((corr_mask & ~sam_mask).sum()) / corr_area

                # Area discrepancy: outside (SAM excess past ref) and inside
                # (ref interior uncovered by SAM), as a fraction of reference area.
                # reference = GT / corr, other = SAM. The util crops to the
                # relevant window internally — same code the tracker trigger uses.
                fam = {}
                fam["dout_gt"], fam["din_gt"] = cropped_signed_area_discrepancy(
                    gt_mask, sam_mask, T_GRID
                )
                fam["dout_corr"], fam["din_corr"] = cropped_signed_area_discrepancy(
                    corr_mask, sam_mask, T_GRID
                )
                # Boundary coverage: outward/inward normal probe of each reference
                # outline against SAM (reference = GT / corr, other = SAM). Crop to
                # the union bbox (+margin > max T) so the normal probes run on an
                # object-sized window; the pad keeps every probe exact.
                union = sam_mask | gt_mask | corr_mask
                ys, xs = np.where(union.any(axis=1))[0], np.where(union.any(axis=0))[0]
                pad = max(T_GRID) + 2
                y0, y1 = max(0, ys[0] - pad), min(h_img, ys[-1] + pad + 1)
                x0, x1 = max(0, xs[0] - pad), min(w_img, xs[-1] + pad + 1)
                sam_crop = sam_mask[y0:y1, x0:x1]
                gt_crop = gt_mask[y0:y1, x0:x1]
                corr_crop = corr_mask[y0:y1, x0:x1]
                fam["bout_gt"], fam["bin_gt"] = outline_normal_coverage(
                    gt_crop, sam_crop, T_GRID
                )
                fam["bout_corr"], fam["bin_corr"] = outline_normal_coverage(
                    corr_crop, sam_crop, T_GRID
                )
                for name in _FAMILIES:
                    for t in T_GRID:
                        t_vals[f"{name}_T{t}"] = fam[name][t]

        rows.append(
            [
                i,
                int(evaluated),
                n_inl,
                loma_err,
                n_oob,
                sam_area,
                gt_area,
                corr_area,
                containment,
                *[t_vals[c] for c in _T_COLS],
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
    """Run the per-frame feature dump over a sequence list."""
    sequences = get_sequences(
        "pti",
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        subset_file=args.subset_file,
        split=args.split,
    )
    out_dir = args.output_dir / "per_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
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
                args.base_dir,
                args.initfix_dir,
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
def _load_all_frames(per_seq_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate evaluated frames from all per-sequence CSVs into columns.

    Args:
        per_seq_dir: Directory of ``{seq}.csv`` files written by ``dump``.

    Returns:
        Columnar dict of numpy arrays over evaluated frames (one entry per CSV
        field), plus ``seq`` (object array of sequence names).
    """
    cols: dict[str, list] = {c: [] for c in CSV_HEADER}
    seq_col: list[str] = []
    files = sorted(per_seq_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-sequence CSVs in {per_seq_dir}")
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as f:
            for d in csv.DictReader(f):
                if int(d["evaluated"]) != 1:
                    continue
                for c in CSV_HEADER:
                    cols[c].append(float(d[c]))
                seq_col.append(path.stem)
    out = {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}
    out["seq"] = np.asarray(seq_col, dtype=object)
    return out


def _load_per_seq_all(per_seq_dir: Path) -> dict[str, list[dict]]:
    """Load all CSV rows (including non-evaluated) per sequence, preserving order."""
    result: dict[str, list[dict]] = {}
    for path in sorted(per_seq_dir.glob("*.csv")):
        rows: list[dict] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for d in csv.DictReader(f):
                rows.append(dict(d))
        result[path.stem] = rows
    return result


# Heatmap geometry + palette, shared by the expected (static) and actual (real
# run) localization maps so the two images register cell-for-cell under the toggle.
_HM_CW, _HM_CH, _HM_GAP = 2, 3, 1
_HM_C_BG = (50, 50, 50)         # not evaluated / beyond sequence length
_HM_C_SKIP = (110, 110, 110)    # evaluated row but feature/run data unreadable
_HM_C_CLEAN = (200, 200, 200)   # evaluated, no notable event
_HM_C_MISSED = (200, 160, 40)   # expected map: correctable but trigger missed
_HM_C_BENEFIT = (50, 200, 80)   # expected: benefit / actual: err < baseline - eps
_HM_C_HARM = (220, 60, 60)      # expected: harm    / actual: err > baseline + eps
_HM_C_FIRED = (255, 255, 255)   # actual: a correction fired on this frame (marker)


def _heatmap_blank(n_seqs: int, max_frames: int) -> np.ndarray:
    """Background-filled RGB canvas sized for (n_seqs rows, max_frames cols)."""
    img_h = max(1, n_seqs * (_HM_CH + _HM_GAP) - _HM_GAP)
    img_w = (max_frames + 2) * _HM_CW
    img = np.zeros((img_h, img_w, 3), np.uint8)
    img[:] = _HM_C_BG
    return img


def _heatmap_paint(img: np.ndarray, si: int, fi: int, color: tuple) -> None:
    """Paint the (seq row ``si``, frame col ``fi``) cell ``color``."""
    x0 = fi * _HM_CW
    if x0 >= img.shape[1]:
        return
    y0 = si * (_HM_CH + _HM_GAP)
    img[y0:y0 + _HM_CH, x0:min(x0 + _HM_CW, img.shape[1])] = color


def _trig_from_cfg(corr_cfg) -> Optional[dict]:
    """Map a ``CorrectionCfg`` onto the static heatmap trigger params.

    Returns None when the config does not use the ``sam_excess_depth`` trigger
    (the only one the static dout_corr feature reproduces).
    """
    if corr_cfg.trigger != "sam_excess_depth":
        return None
    if corr_cfg.excess_depth_px not in T_GRID:
        raise ValueError(
            f"excess_depth_px={corr_cfg.excess_depth_px} not in T_GRID {T_GRID}; "
            "re-dump with that T to compare"
        )
    return {
        "thr": int(corr_cfg.inlier_threshold),
        "tol": float(corr_cfg.superset_tolerance),
        "margin": float(corr_cfg.excess_margin),
        "t_feat_idx": T_GRID.index(corr_cfg.excess_depth_px),
    }


def _make_heatmap_img(
    per_seq: dict[str, list[dict]],
    seqs: list[str],
    max_frames: int,
    trig: dict,
    t_label_idx: int,
    frac_label: float,
    n_default: float,
) -> np.ndarray:
    """Expected (static-analysis) localization map, RGB uint8 (rows=seqs, cols=frames).

    Colors encode the static trigger outcome at ``trig`` params:
      green  = benefit (fires & over-seg & loma_ok)
      red    = harm    (fires & ~(over-seg & loma_ok))
      amber  = correctable (over-seg & loma_ok) but trigger missed
      lgrey  = evaluated, no notable event
      dgrey  = not evaluated / beyond sequence length
    """
    img = _heatmap_blank(len(seqs), max_frames)
    if not seqs:
        return img

    thr = float(trig["thr"])
    tol = float(trig["tol"])
    margin = float(trig["margin"])
    t_feat = T_GRID[int(trig["t_feat_idx"])]
    t_label = T_GRID[int(t_label_idx)]
    dout_col = f"dout_gt_T{t_label}"
    dexc_col = f"dout_corr_T{t_feat}"

    for si, seq in enumerate(seqs):
        for fd in per_seq[seq]:
            fi = int(float(fd["frame_idx"]))
            if int(fd["evaluated"]) != 1:
                _heatmap_paint(img, si, fi, _HM_C_SKIP)
                continue
            try:
                n_inl = float(fd["n_inliers"])
                dout = float(fd[dout_col])
                dexc = float(fd[dexc_col])
                loma_e = float(fd["loma_align_err"])
                cont = float(fd["containment"])
            except (KeyError, ValueError):
                _heatmap_paint(img, si, fi, _HM_C_SKIP)
                continue
            over = np.isfinite(dout) and dout >= frac_label
            lok = np.isfinite(loma_e) and loma_e < n_default
            fires = (
                n_inl >= thr
                and np.isfinite(cont) and cont <= tol
                and np.isfinite(dexc) and dexc >= margin
            )
            if fires:
                color: tuple = _HM_C_BENEFIT if (over and lok) else _HM_C_HARM
            elif over and lok:
                color = _HM_C_MISSED
            else:
                color = _HM_C_CLEAN
            _heatmap_paint(img, si, fi, color)

    return img


def _load_pred_corners(results_dir: Path, seq: str) -> Optional[np.ndarray]:
    """Load predicted corners ``{seq}.txt`` as ``(frames, 2, 4)``, or None."""
    f = results_dir / f"{seq}.txt"
    if not f.exists():
        return None
    flat = np.loadtxt(f)
    if flat.ndim == 1:
        flat = flat[np.newaxis, :]
    return rearrange(flat, "frames (corners xy) -> frames xy corners", corners=4, xy=2)


def _seq_align_errors(
    results_dir: Path, seq: str, gt_corners: np.ndarray, valid: np.ndarray
) -> dict[int, float]:
    """Per-frame corner alignment error (px) vs GT, keyed by frame index."""
    pred = _load_pred_corners(results_dir, seq)
    if pred is None:
        return {}
    T = min(len(gt_corners), len(pred), len(valid))
    return {
        i: float(alignment_error(gt_corners[i], pred[i]))
        for i in range(1, T)
        if bool(valid[i])
    }


def _load_corrected_flags(results_dir: Path, seq: str) -> Optional[dict[int, bool]]:
    """Per-frame ``corrected`` flag from ``{seq}_debug.npz``, keyed by frame index."""
    npz = results_dir / f"{seq}_debug.npz"
    if not npz.exists():
        return None
    data = np.load(npz)
    if "corrected" not in data or "frame_idx" not in data:
        return None
    return {int(fi): bool(c) for fi, c in zip(data["frame_idx"], data["corrected"])}


def _make_actual_heatmap_img(
    per_seq: dict[str, list[dict]],
    seqs: list[str],
    max_frames: int,
    corr_dir: Path,
    baseline_dir: Path,
    base_dir: Path,
    initfix_dir: Path,
    eps: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Actual (real-run) localization map + summary counts.

    *Every* evaluated frame is colored by the per-frame corner-error delta of the
    corrected run vs the no-correction baseline — so the long-lasting downstream
    effect of a correction (a green or red streak following a fire) is visible,
    not only the frame it fired on. Frames where a correction actually fired
    (``corrected`` in the corrected run's debug npz) are overlaid with a marker.

      white   = a correction fired on this frame (marker overrides the cell)
      green   = corrected err < baseline err - eps  (corrected better here)
      red     = corrected err > baseline err + eps  (corrected worse here)
      lgrey   = |Δerr| <= eps (trajectories agree) or err unavailable
      dgrey   = not evaluated / run data missing
    """
    img = _heatmap_blank(len(seqs), max_frames)
    counts = {"fired": 0, "better": 0, "worse": 0, "same": 0, "missing": 0}
    for si, seq in enumerate(seqs):
        try:
            gt_corners, valid = pti_load_gt(base_dir, seq, initfix_dir)
        except Exception:  # noqa: BLE001 - missing GT -> mark row as skipped
            gt_corners, valid = None, None
        corr_flags = _load_corrected_flags(corr_dir, seq)
        base_err = (
            _seq_align_errors(baseline_dir, seq, gt_corners, valid)
            if gt_corners is not None else {}
        )
        corr_err = (
            _seq_align_errors(corr_dir, seq, gt_corners, valid)
            if gt_corners is not None else {}
        )
        for fd in per_seq[seq]:
            fi = int(float(fd["frame_idx"]))
            if int(fd["evaluated"]) != 1:
                _heatmap_paint(img, si, fi, _HM_C_SKIP)
                continue
            if corr_flags is None:
                _heatmap_paint(img, si, fi, _HM_C_SKIP)
                counts["missing"] += 1
                continue
            # Fire marker overrides the cell — we care about *when* it fired, not
            # its same-frame effect (the effect shows up on the following frames).
            if bool(corr_flags.get(fi, False)):
                counts["fired"] += 1
                _heatmap_paint(img, si, fi, _HM_C_FIRED)
                continue
            # Per-frame error delta (downstream effect), on every non-fire frame.
            be, ce = base_err.get(fi), corr_err.get(fi)
            if be is not None and ce is not None and ce < be - eps:
                color = _HM_C_BENEFIT
                counts["better"] += 1
            elif be is not None and ce is not None and ce > be + eps:
                color = _HM_C_HARM
                counts["worse"] += 1
            else:
                color = _HM_C_CLEAN
                counts["same"] += 1
            _heatmap_paint(img, si, fi, color)

    return img, counts


def _grid_search_trigger(
    fires_inputs: dict, overseg: np.ndarray, loma_ok: np.ndarray, enough_fn
) -> dict:
    """Find trigger params maximizing (benefit - harm) at the default labels.

    Args:
        fires_inputs: Dict with arrays ``dexc`` (per-T list), ``containment``,
            ``n_inliers`` and the ``t_grid``.
        overseg: Bool label array (SAM over-segments).
        loma_ok: Bool label array (LoMa estimate correct).
        enough_fn: ``thr -> bool array`` (n_inliers >= thr).

    Returns:
        Best params + counts: ``thr, tol, margin, t_feat_idx, benefit, harm, net``.
    """
    dexc = fires_inputs["dexc"]  # list of arrays, one per T
    cont = fires_inputs["containment"]
    thr_grid = [30, 40, 50, 60, 70, 80, 100]
    tol_grid = [0.01, 0.02, 0.05, 0.10]
    margin_grid = [0.05, 0.10, 0.20, 0.30, 0.50]
    best = {"net": -1, "thr": 70, "tol": 0.02, "margin": 0.20, "t_feat_idx": 0}
    for thr in thr_grid:
        enough = enough_fn(thr)
        for ti in range(len(fires_inputs["t_grid"])):
            de = dexc[ti]
            for tol in tol_grid:
                guard = enough & (cont <= tol)
                for margin in margin_grid:
                    fires = guard & (de >= margin)
                    benefit = int((fires & overseg & loma_ok).sum())
                    harm = int((fires & ~(overseg & loma_ok)).sum())
                    net = benefit - harm
                    if net > best["net"]:
                        best = {
                            "net": net,
                            "thr": thr,
                            "tol": tol,
                            "margin": margin,
                            "t_feat_idx": ti,
                            "benefit": benefit,
                            "harm": harm,
                        }
    return best



def cmd_report(args: argparse.Namespace) -> None:
    """Build the standalone HTML report from per-sequence CSVs."""
    data = _load_all_frames(args.per_seq_dir)
    m = len(data["frame_idx"])
    if m == 0:
        raise ValueError("no evaluated frames found")

    dout = [data[f"dout_gt_T{t}"] for t in T_GRID]  # SAM excess vs GT (label)
    dexc = [data[f"dout_corr_T{t}"] for t in T_GRID]  # SAM excess vs corr (feature)
    din = [data[f"din_gt_T{t}"] for t in T_GRID]  # SAM deficit vs GT (under-seg)
    bout = [data[f"bout_gt_T{t}"] for t in T_GRID]  # GT-outline overshoot
    binn = [data[f"bin_gt_T{t}"] for t in T_GRID]  # GT-outline shortfall
    n_inliers = data["n_inliers"]
    loma_err = data["loma_align_err"]
    cont = data["containment"]

    # Label defaults (also slider start positions): T_label px, over-seg excess
    # fraction (default matches CorrectionCfg.area_margin), N px alignment-correct.
    # Overridable via CLI to re-grid-search the trigger for a different label.
    if args.t_label not in T_GRID:
        raise ValueError(f"--t-label {args.t_label} not in T_GRID {T_GRID}")
    t_label_idx = T_GRID.index(args.t_label)
    frac_label_default = args.frac_label
    n_default = args.n_align
    overseg = dout[t_label_idx] >= frac_label_default
    loma_ok = loma_err < n_default

    best = _grid_search_trigger(
        {"dexc": dexc, "containment": cont, "n_inliers": n_inliers, "t_grid": T_GRID},
        overseg,
        loma_ok,
        enough_fn=lambda thr: n_inliers >= thr,
    )

    # Heatmap + previews -----------------------------------------------------
    per_seq = _load_per_seq_all(args.per_seq_dir)
    seq_names = sorted(per_seq.keys())
    max_frames = max((len(v) for v in per_seq.values()), default=1)

    # Localization-map trigger. With --config the expected map reflects the
    # *deployed* trigger (so expected vs actual compare the same rule); without
    # it, the grid-search optimum. The interactive sliders/table keep ``best``.
    heat_trig = best
    corr_cfg = None
    corr_dir = None
    if args.config is not None:
        corr_cfg = load_config_from_file(args.config).correction
        corr_dir = compute_output_dir(paths.OUTPUT_DIRECTORY, "pti", args.config)
        cfg_trig = _trig_from_cfg(corr_cfg)
        if cfg_trig is not None:
            heat_trig = cfg_trig

    heatmap_rgb = _make_heatmap_img(
        per_seq, seq_names, max_frames, heat_trig, t_label_idx, frac_label_default, n_default
    )
    ok_enc, buf = cv2.imencode(".png", cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
    heatmap_b64 = base64.b64encode(bytes(buf)).decode() if ok_enc else ""

    # Actual (real-run) localization map, only when a deployed config is given.
    actual_b64 = ""
    actual_counts: Optional[dict[str, int]] = None
    if corr_dir is not None:
        actual_rgb, actual_counts = _make_actual_heatmap_img(
            per_seq, seq_names, max_frames,
            corr_dir, args.baseline_dir, args.base_dir, args.initfix_dir, args.helped_eps,
        )
        ok2, buf2 = cv2.imencode(".png", cv2.cvtColor(actual_rgb, cv2.COLOR_RGB2BGR))
        actual_b64 = base64.b64encode(bytes(buf2)).decode() if ok2 else ""

    seq_list_html = "".join(f"<li>{s}</li>" for s in seq_names)
    seq_previews = build_seq_previews(seq_names, args.base_dir, args.initfix_dir)
    # ------------------------------------------------------------------------

    def col(a: np.ndarray, nd: int = 4) -> list:
        return [round(float(x), nd) for x in a]

    payload = {
        "t_grid": list(T_GRID),
        "n_inliers": [int(x) for x in n_inliers],
        "loma_err": col(loma_err, 3),
        "containment": col(cont),
        "dout": [col(a) for a in dout],
        "dexc": [col(a) for a in dexc],
        "din": [col(a) for a in din],
        "bout": [col(a) for a in bout],
        "binn": [col(a) for a in binn],
        "defaults": {
            "thr": int(best["thr"]),
            "tol": float(best["tol"]),
            "margin": float(best["margin"]),
            "t_feat_idx": int(best["t_feat_idx"]),
            "t_label_idx": int(t_label_idx),
            "frac_label": frac_label_default,
            "n": n_default,
        },
        "n_frames": m,
        "inlier_max": int(n_inliers.max()),
        "seq_names": seq_names,
        "seq_previews": seq_previews,
        "has_actual": bool(actual_b64),
        "actual_stats": (
            f"fired on {actual_counts['fired']} frames. Across all evaluated frames "
            f"(downstream effect vs baseline, eps={args.helped_eps:g}px): "
            f"better {actual_counts['better']}, worse {actual_counts['worse']}, "
            f"unchanged {actual_counts['same']}."
            if actual_counts is not None else ""
        ),
    }
    actual_src = f"data:image/png;base64,{actual_b64}" if actual_b64 else ""
    html = _HTML_TEMPLATE.replace("/*__PAYLOAD__*/null", json.dumps(payload))
    html = html.replace("<!--HEATMAP_SRC-->", f"data:image/png;base64,{heatmap_b64}")
    html = html.replace("<!--ACTUAL_HEATMAP_SRC-->", actual_src)
    html = html.replace("<!--SEQ_LIST-->", seq_list_html)
    html = html.replace("<!--SEQ_PREVIEW_POPUP-->", seq_preview_html())
    html = html.replace("// __SEQ_PREVIEW_JS__ //", seq_preview_js(fps=2.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    # Flat combined CSV (evaluated frames, all sequences) for ad-hoc analysis.
    combined = args.output.parent / "combined.csv"
    with combined.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", *CSV_HEADER])
        for i in range(m):
            w.writerow(
                [data["seq"][i]]
                + [
                    f"{data[c][i]:.6g}" if c not in ("frame_idx", "evaluated", "n_inliers")
                    else int(data[c][i])
                    for c in CSV_HEADER
                ]
            )
    print(f"Frames     : {m}")
    print(
        f"Over-seg @ default label: {int(overseg.sum())} "
        f"({100 * overseg.mean():.1f}%)"
    )
    print(
        f"Feasibility ceiling (over-seg & loma-correct): "
        f"{int((overseg & loma_ok).sum())}"
    )
    print(
        f"Optimal trigger: thr={best['thr']} tol={best['tol']} "
        f"margin={best['margin']} T_feat={T_GRID[best['t_feat_idx']]}px "
        f"-> benefit={best.get('benefit')} harm={best.get('harm')}"
    )
    if actual_counts is not None:
        print(
            f"Actual run @ {args.config.stem} (eps={args.helped_eps}px): "
            f"fired={actual_counts['fired']} | downstream over all frames: "
            f"better={actual_counts['better']} worse={actual_counts['worse']} "
            f"unchanged={actual_counts['same']} missing={actual_counts['missing']}"
        )
    print(f"Report     : {args.output}")
    print(f"Combined   : {combined}")


_HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAM over-seg correction feasibility</title>
<style>
 body{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a;max-width:1100px}
 h1{font-size:20px} h2{font-size:15px;margin-top:28px}
 .ctrl{display:grid;grid-template-columns:170px 1fr 70px;gap:6px 12px;align-items:center;
   max-width:620px;margin:8px 0}
 .ctrl label{font-size:13px} .ctrl output{font-variant-numeric:tabular-nums;text-align:right}
 table{border-collapse:collapse;margin:10px 0} td,th{border:1px solid #ccc;padding:7px 12px;text-align:center}
 th{background:#f3f3f3} .benefit{background:#d6f5d6} .harm{background:#f7d6d6} .muted{color:#888}
 .stat{display:inline-block;min-width:150px;margin:4px 18px 4px 0;font-size:14px}
 .stat b{font-size:18px} .note{font-size:12px;color:#666;max-width:760px;line-height:1.5}
 code{background:#f0f0f0;padding:1px 4px;border-radius:3px}
</style></head><body>
<h1>SAM3 over-segmentation &rarr; LoMa correction feasibility</h1>
<p class="note">Substrate: no-correction baseline, <span id="nframes"></span> evaluated frames.
Move sliders to threshold the stored per-frame metrics. <b>Row label</b> (SAM over-segments)
= fraction of SAM area &gt; T<sub>label</sub> px outside GT &ge; frac. <b>LoMa correct</b>
= corner alignment error &lt; N px. <b>Trigger</b> (inference-available) fires when
n_inliers &ge; THR <span class="muted">and</span> corr-spill &le; tol <span class="muted">and</span>
SAM-deep-outside-corr &ge; margin. Defaults are grid-searched to maximize benefit&minus;harm.</p>

<h2>Quantity definitions</h2>
<details><summary style="cursor:pointer;font-size:13px;color:#444;user-select:none">&#9658; expand glossary</summary>
<div style="margin-top:8px;font-size:13px;line-height:1.65;max-width:920px">
<p style="margin:0 0 6px"><b>Three mask pairs</b> &mdash; "SAM (<i>other</i>) vs <i>reference</i>":</p>
<table style="font-size:12px;margin:0 0 10px">
 <tr><th>Pair</th><th>Reference</th><th>Other (always SAM3)</th><th>Role</th></tr>
 <tr><td>SAM vs GT</td><td>GT corners &rarr; mask</td><td>SAM3 mask</td><td><b>Label</b>: is SAM over-segmenting?</td></tr>
 <tr><td>LoMa-corr vs GT</td><td>GT corners &rarr; mask</td><td>LoMa init-mask warped by loma_H</td><td><b>Label</b>: is LoMa&rsquo;s homography correct?</td></tr>
 <tr><td>SAM vs LoMa-corr</td><td>LoMa init-mask warped by loma_H</td><td>SAM3 mask</td><td><b>Feature</b>: does SAM exceed LoMa? <span class="muted">(inference-available)</span></td></tr>
</table>
<div style="display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;margin-bottom:10px">
 <div style="min-width:230px">
  <b>d &mdash; area depth</b>
  <svg width="200" height="110" viewBox="0 0 200 110" style="display:block;margin:4px 0">
   <rect width="200" height="110" fill="#f8f8f8" rx="4"/>
   <rect x="52" y="22" width="116" height="72" fill="#ddd" rx="3" opacity=".85"/>
   <rect x="130" y="22" width="38" height="72" fill="rgba(220,60,60,.42)" rx="3"/>
   <rect x="18" y="27" width="38" height="62" fill="rgba(210,140,30,.48)" rx="3"/>
   <rect x="18" y="27" width="112" height="62" fill="none" stroke="#2060c0" stroke-width="2" stroke-dasharray="5,3" rx="3"/>
   <rect x="52" y="22" width="116" height="72" fill="none" stroke="#999" stroke-width="1.5" rx="3"/>
   <line x1="130" y1="60" x2="168" y2="60" stroke="#c03030" stroke-width="1" stroke-dasharray="3,2"/>
   <text x="132" y="56" font-size="9" fill="#c03030">T px</text>
   <text x="20" y="17" font-size="9" fill="#2060c0">reference</text>
   <text x="75" y="106" font-size="9" fill="#555">SAM</text>
   <text x="133" y="78" font-size="9" fill="#c03030" font-weight="bold">excess</text>
   <text x="20" y="78" font-size="9" fill="#b07020" font-weight="bold">deficit</text>
  </svg>
  <code>dout_gt</code>: fraction of SAM area &gt;&nbsp;T&nbsp;px outside GT boundary &rarr; over-seg depth label.<br>
  <code>din_gt</code>: fraction of GT area uncovered by SAM at depth T &rarr; under-seg label.<br>
  <code>dout_corr</code>: same vs LoMa mask &rarr; inference-time over-seg feature (<code>dexc</code>).<br>
  Normalized by reference area. T=0 counts any outside pixel; larger T ignores edge jitter.<br>
  <i>Depth view</i>: a thin shell all around scores high; one isolated deep bulge scores lower.
 </div>
 <div style="min-width:230px">
  <b>b &mdash; boundary coverage</b>
  <svg width="200" height="110" viewBox="0 0 200 110" style="display:block;margin:4px 0">
   <rect width="200" height="110" fill="#f8f8f8" rx="4"/>
   <rect x="42" y="22" width="116" height="72" fill="#ddd" rx="3" opacity=".85"/>
   <rect x="18" y="22" width="116" height="72" fill="none" stroke="#2060c0" stroke-width="2" stroke-dasharray="5,3" rx="3"/>
   <rect x="42" y="22" width="116" height="72" fill="none" stroke="#999" stroke-width="1.5" rx="3"/>
   <line x1="134" y1="36" x2="154" y2="36" stroke="#888" stroke-width="1"/>
   <circle cx="154" cy="36" r="3.5" fill="#32c832"/>
   <line x1="134" y1="54" x2="154" y2="54" stroke="#888" stroke-width="1"/>
   <circle cx="154" cy="54" r="3.5" fill="#32c832"/>
   <line x1="134" y1="72" x2="154" y2="72" stroke="#888" stroke-width="1"/>
   <circle cx="154" cy="72" r="3.5" fill="#32c832"/>
   <line x1="18" y1="36" x2="-2" y2="36" stroke="#888" stroke-width="1"/>
   <circle cx="-2" cy="36" r="3.5" fill="#dc3c3c"/>
   <line x1="18" y1="54" x2="-2" y2="54" stroke="#888" stroke-width="1"/>
   <circle cx="-2" cy="54" r="3.5" fill="#dc3c3c"/>
   <line x1="18" y1="72" x2="-2" y2="72" stroke="#888" stroke-width="1"/>
   <circle cx="-2" cy="72" r="3.5" fill="#dc3c3c"/>
   <text x="157" y="58" font-size="9" fill="#32c832">hit</text>
   <text x="2" y="88" font-size="9" fill="#dc3c3c">miss</text>
   <text x="20" y="17" font-size="9" fill="#2060c0">reference</text>
   <text x="134" y="106" font-size="9" fill="#555">&larr; T px outward</text>
  </svg>
  <code>bout_gt</code>: fraction of GT outline points where SAM protrudes outward T&nbsp;px.<br>
  <code>bin_gt</code>: fraction of GT outline points where SAM falls short inward T&nbsp;px.<br>
  Normal probes perpendicular to the outline, T&nbsp;px outward (for overshoot) / inward (shortfall).<br>
  <i>Breadth view</i>: complementary to depth &mdash; a thin ring all round is high-coverage/low-depth;
  one deep isolated bulge is low-coverage/high-depth.
 </div>
</div>
<b>Scalar fields:</b>
<ul style="margin:4px 0 8px;padding-left:18px">
 <li><code>n_inliers</code>: LoMa RANSAC inlier count. Higher = more confident homography estimate.</li>
 <li><code>loma_align_err</code> (px): corner alignment error of LoMa-predicted corners vs GT. <b>GT label only</b> (not available at inference) &mdash; used to classify whether LoMa is &ldquo;correct&rdquo;.</li>
 <li><code>containment</code>: fraction of the LoMa mask lying outside SAM. Guard: if LoMa spills well beyond SAM, injecting it would <em>expand</em> the mask, which is the wrong direction for an over-seg fix.</li>
 <li><code>area_sam / area_gt / area_corr</code>: absolute pixel areas of each mask.</li>
</ul>
<p style="margin:4px 0"><b>Trigger logic</b> (all inference-available):
 <code>n_inliers &ge; THR</code> (LoMa has sufficient evidence)
 <b>and</b> <code>containment &le; tol</code> (LoMa mask sits inside SAM, so replacing SAM shrinks it)
 <b>and</b> <code>dout_corr &ge; margin</code> (SAM meaningfully exceeds LoMa prediction).
 When fired: replace the SAM mask with the LoMa-warped init mask.
</p>
</div></details>

<h2>Labels</h2>
<div class="ctrl">
 <label>N: align-correct (px)</label><input id="n" type="range" min="0" max="30" step="0.5"><output id="n_o"></output>
 <label>T<sub>label</sub> (px)</label><input id="tl" type="range" min="0" step="1"><output id="tl_o"></output>
 <label>over-seg frac</label><input id="fl" type="range" min="0" max="1" step="0.01"><output id="fl_o"></output>
</div>
<h2>Trigger (inference signals)</h2>
<div class="ctrl">
 <label>THR: min inliers</label><input id="thr" type="range" min="0" step="1"><output id="thr_o"></output>
 <label>corr-spill tol</label><input id="tol" type="range" min="0" max="0.3" step="0.005"><output id="tol_o"></output>
 <label>SAM-excess margin</label><input id="mg" type="range" min="0" max="1" step="0.01"><output id="mg_o"></output>
 <label>T<sub>feat</sub> (px)</label><input id="tf" type="range" min="0" step="1"><output id="tf_o"></output>
</div>

<h2>2&times;3 table <span class="muted">(rows: SAM; cols: LoMa)</span></h2>
<table id="tab">
 <tr><th></th><th>LoMa correct &amp; &ge;THR</th><th>LoMa wrong &amp; &ge;THR</th><th>&lt;THR (abstain)</th><th>row total</th></tr>
 <tr><th>SAM over-seg</th><td class="benefit" id="c00"></td><td id="c01"></td><td id="c02"></td><td id="r0"></td></tr>
 <tr><th>SAM not-over-seg</th><td id="c10"></td><td id="c11"></td><td id="c12"></td><td id="r1"></td></tr>
</table>
<p class="note">Green = correctable cell (SAM over-seg, LoMa confident-correct) — the upside ceiling.
"SAM not-over-seg" lumps clean + under-seg frames (under-seg is out of scope here):
<span id="under"></span> of them are actually under-seg (GT deep outside SAM &ge; frac).</p>
<p class="note"><b>GT-outline coverage</b> (over-seg frames, at T<sub>label</sub>):
overshoot (SAM past GT) median <b id="br_med"></b> / mean <b id="br_mean"></b>;
shortfall (SAM short of GT) median <b id="bi_med"></b> / mean <b id="bi_mean"></b> — signed
fractions of the GT outline deviating outward / inward by &gt; T. Robust to thin lateral spills.
This is <i>coverage</i>; the area fractions are <i>depth</i> (signed excess/deficit of GT area):
a thin shell all around is high-coverage/low-depth, one deep bulge is low-coverage/high-depth.</p>

<h2>Trigger outcome</h2>
<div>
 <span class="stat">fires: <b id="fires"></b></span>
 <span class="stat benefit">benefit: <b id="ben"></b></span>
 <span class="stat harm">harm: <b id="harm"></b></span>
 <span class="stat">net: <b id="net"></b></span>
</div>
<div>
 <span class="stat">precision: <b id="prec"></b></span>
 <span class="stat">recall (of correctable): <b id="rec"></b></span>
 <span class="stat">harm: LoMa wrong <b id="hw"></b> / SAM was fine <b id="hf"></b></span>
</div>
<p class="note">benefit = fires &cap; over-seg &cap; LoMa-correct. harm = fires &cap; &not;(over-seg &amp; LoMa-correct):
either LoMa wrong (injects bad mask) or SAM was already fine (needless shrink). precision = benefit/fires;
recall = benefit / (over-seg &amp; LoMa-correct &amp; &ge;THR).</p>

<h2>Fix / harm localization</h2>
<p class="note">Each row = one sequence (alphabetical; see list or hover). Each column = one frame.
Hover any row to see its name; click to open animated preview. Rows/columns register cell-for-cell
between the two views.</p>
<div id="heatmap-toggle" style="display:none;margin:6px 0">
 <button id="hm-btn-exp" type="button">&larr; expected (static)</button>
 <button id="hm-btn-act" type="button">actual (real run) &rarr;</button>
 <span id="hm-mode" class="muted" style="margin-left:10px"></span>
 <span class="muted" style="margin-left:6px">(&larr;/&rarr; keys)</span>
</div>
<div style="position:relative">
 <img id="heatmap-img" src="<!--HEATMAP_SRC-->"
      style="image-rendering:pixelated;width:100%;border:1px solid #ddd;cursor:crosshair;display:block"/>
 <img id="heatmap-img-actual" src="<!--ACTUAL_HEATMAP_SRC-->"
      style="image-rendering:pixelated;width:100%;border:1px solid #ddd;cursor:crosshair;display:none"/>
 <div id="heatmap-hover-label" style="font-size:11px;color:#666;margin-top:2px;height:16px">&nbsp;</div>

</div>
<p id="expected-legend" class="note">Expected (static analysis at the deployed/optimal trigger):
<span style="display:inline-block;width:12px;height:12px;background:#32c850;vertical-align:middle;margin-left:4px"></span>&nbsp;benefit &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#dc3c3c;vertical-align:middle;margin-left:4px"></span>&nbsp;harm &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#c8a028;vertical-align:middle;margin-left:4px"></span>&nbsp;correctable&nbsp;/&nbsp;missed &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#c8c8c8;vertical-align:middle;margin-left:4px"></span>&nbsp;evaluated,&nbsp;no event &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#6e6e6e;vertical-align:middle;margin-left:4px"></span>&nbsp;not&nbsp;evaluated</p>
<p id="actual-legend" class="note" style="display:none">Actual (real run). White marks frames where a correction fired; every other frame is
colored by corner-error vs the no-correction baseline — so a correction's <i>downstream</i> effect shows as a colored streak after the marker:
<span style="display:inline-block;width:12px;height:12px;background:#fff;border:1px solid #999;vertical-align:middle;margin-left:4px"></span>&nbsp;fired &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#32c850;vertical-align:middle;margin-left:4px"></span>&nbsp;better&nbsp;(err&nbsp;down) &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#dc3c3c;vertical-align:middle;margin-left:4px"></span>&nbsp;worse&nbsp;(err&nbsp;up) &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#c8c8c8;vertical-align:middle;margin-left:4px"></span>&nbsp;unchanged&nbsp;(&le;eps) &ensp;
<span style="display:inline-block;width:12px;height:12px;background:#6e6e6e;vertical-align:middle;margin-left:4px"></span>&nbsp;not&nbsp;evaluated<br><b id="actual-stats" style="font-weight:normal"></b></p>
<details style="font-size:11px;margin-top:4px">
 <summary style="cursor:pointer;color:#444;user-select:none">&#9658; sequence list (row # &rarr; name)</summary>
 <ol style="margin:4px 0;padding-left:18px;line-height:1.35;column-count:3"><!--SEQ_LIST--></ol>
</details>

<!--SEQ_PREVIEW_POPUP-->
<script>
// __SEQ_PREVIEW_JS__ //
const D=/*__PAYLOAD__*/null;
const $=id=>document.getElementById(id);
document.getElementById("nframes").textContent=D.n_frames;
const G=D.t_grid, M=D.n_frames;
function setup(){
 $("n").value=D.defaults.n;
 $("tl").max=G.length-1; $("tl").value=D.defaults.t_label_idx;
 $("fl").value=D.defaults.frac_label;
 $("thr").max=D.inlier_max; $("thr").value=D.defaults.thr;
 $("tol").value=D.defaults.tol;
 $("mg").value=D.defaults.margin;
 $("tf").max=G.length-1; $("tf").value=D.defaults.t_feat_idx;
}
function compute(){
 const N=+$("n").value, tl=+$("tl").value, fl=+$("fl").value;
 const thr=+$("thr").value, tol=+$("tol").value, mg=+$("mg").value, tf=+$("tf").value;
 $("n_o").textContent=N.toFixed(1); $("tl_o").textContent=G[tl]+"px"; $("fl_o").textContent=fl.toFixed(2);
 $("thr_o").textContent=thr; $("tol_o").textContent=tol.toFixed(3); $("mg_o").textContent=mg.toFixed(2);
 $("tf_o").textContent=G[tf]+"px";
 const dout=D.dout[tl], dexc=D.dexc[tf], din=D.din[tl], bout=D.bout[tl], binn=D.binn[tl];
 let c=[[0,0,0],[0,0,0]], under=0, fires=0, ben=0, hw=0, hf=0, correctable=0;
 let bo=[], bi=[];
 for(let i=0;i<M;i++){
  const over=dout[i]>=fl, ok=D.loma_err[i]<N, enough=D.n_inliers[i]>=thr;
  const r=over?0:1, col=enough?(ok?0:1):2;
  c[r][col]++;
  if(!over && din[i]>=fl) under++;
  if(over){ if(bout[i]===bout[i]) bo.push(bout[i]); if(binn[i]===binn[i]) bi.push(binn[i]); }
  if(over && ok && enough) correctable++;
  if(enough && D.containment[i]<=tol && dexc[i]>=mg){
   fires++;
   if(over && ok) ben++; else { if(!ok) hw++; if(over===false && ok) hf++; }
  }
 }
 const stat=(a,me,mn)=>{ if(a.length){ a.sort((x,y)=>x-y);
   $(me).textContent=(100*a[Math.floor(a.length/2)]).toFixed(1)+"%";
   $(mn).textContent=(100*a.reduce((x,y)=>x+y,0)/a.length).toFixed(1)+"%";
  } else { $(me).textContent="–"; $(mn).textContent="–"; } };
 stat(bo,"br_med","br_mean"); stat(bi,"bi_med","bi_mean");
 const ids=[["c00","c01","c02"],["c10","c11","c12"]];
 for(let r=0;r<2;r++){for(let k=0;k<3;k++)$(ids[r][k]).textContent=c[r][k];}
 $("r0").textContent=c[0][0]+c[0][1]+c[0][2];
 $("r1").textContent=c[1][0]+c[1][1]+c[1][2];
 $("under").textContent=under;
 const harm=fires-ben;
 $("fires").textContent=fires; $("ben").textContent=ben; $("harm").textContent=harm;
 $("net").textContent=ben-harm;
 $("prec").textContent=fires?(100*ben/fires).toFixed(1)+"%":"–";
 $("rec").textContent=correctable?(100*ben/correctable).toFixed(1)+"%":"–";
 $("hw").textContent=hw; $("hf").textContent=hf;
}
setup();
document.querySelectorAll("input[type=range]").forEach(e=>e.addEventListener("input",compute));
compute();

// Heatmap wiring — delegates to SeqPreview component
(function(){
 const lbl=$("heatmap-hover-label");
 const imgs=[$("heatmap-img"),$("heatmap-img-actual")].filter(Boolean);
 const CH=3, GAP=1;
 function rowAt(img,e){
  const r=img.getBoundingClientRect();
  const yNat=(e.clientY-r.top)/r.height*img.naturalHeight;
  return Math.max(0,Math.min((D.seq_names||[]).length-1,Math.floor(yNat/(CH+GAP))));
 }
 imgs.forEach(function(img){
  img.addEventListener("mousemove",function(e){lbl.textContent=(D.seq_names||[])[rowAt(img,e)]||" ";});
  img.addEventListener("mouseleave",function(){lbl.textContent=" ";});
  img.addEventListener("click",function(e){
   const seq=(D.seq_names||[])[rowAt(img,e)];
   if(seq) SeqPreview.show(seq,e.clientX,e.clientY);
  });
 });
 if(D.has_actual){
  $("heatmap-toggle").style.display="block";
  $("actual-stats").textContent=D.actual_stats||"";
  const exp=$("heatmap-img"), act=$("heatmap-img-actual");
  function mode(isAct){
   exp.style.display=isAct?"none":"block"; act.style.display=isAct?"block":"none";
   $("actual-legend").style.display=isAct?"block":"none";
   $("expected-legend").style.display=isAct?"none":"block";
   $("hm-mode").textContent=isAct?"showing real-run outcome":"showing static-analysis expectation";
  }
  $("hm-btn-exp").addEventListener("click",function(){mode(false);});
  $("hm-btn-act").addEventListener("click",function(){mode(true);});
  document.addEventListener("keydown",function(e){
   if(e.key==="ArrowLeft"){mode(false);e.preventDefault();}
   else if(e.key==="ArrowRight"){mode(true);e.preventDefault();}
  });
  mode(false);
 }
})();
SeqPreview.init(D.seq_previews);
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# viz
# --------------------------------------------------------------------------- #
# Panel colours, BGR.
_VIZ = {
    "ref_line": (235, 120, 40),     # reference outline (blue)
    "other_line": (170, 170, 170),  # other outline (grey)
    "other_fill": (210, 210, 210),  # other interior (light grey)
    "excess": (60, 60, 235),        # area: other beyond reference (red)
    "deficit": (40, 150, 245),      # area: reference uncovered (orange)
    "hit": (60, 185, 75),           # coverage overshoot hit (green)
    "miss": (155, 155, 155),        # coverage miss (grey)
    "short": (60, 60, 235),         # coverage shortfall = other absent (red)
    "normal": (120, 120, 120),      # normal probe segment
}


def _viz_base(reference: np.ndarray, other: np.ndarray, scale: int) -> np.ndarray:
    """White canvas with ``other`` filled + both outlines, upscaled ``scale``×."""
    h, w = reference.shape
    img = np.full((h, w, 3), 255, np.uint8)
    img[other] = _VIZ["other_fill"]
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    for mask, col, th in ((other, _VIZ["other_line"], 1), (reference, _VIZ["ref_line"], 2)):
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(big, [c * scale for c in cnts], -1, col, th)
    return big


def _viz_recolor(big: np.ndarray, mask: np.ndarray, col: tuple) -> None:
    """Paint ``col`` where the (native-res) bool ``mask`` upscales to in ``big``."""
    up = cv2.resize(
        mask.astype(np.uint8), (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    big[up > 0] = col


def _viz_label(img: np.ndarray, text: str) -> np.ndarray:
    """Stack a caption band above ``img``."""
    band = np.full((26, img.shape[1], 3), 245, np.uint8)
    cv2.putText(band, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def _viz_area_panel(
    reference: np.ndarray, other: np.ndarray, sdf: np.ndarray, t: int, ref_area: int, scale: int
) -> np.ndarray:
    """Area panel: counted excess (red) and deficit (orange) pixels at T."""
    outside, inside, dbg = signed_area_discrepancy(sdf, other, ref_area, [t], debug=True)
    big = _viz_base(reference, other, scale)
    if dbg is not None:
        _viz_recolor(big, dbg.inside_counted[t], _VIZ["deficit"])
        _viz_recolor(big, dbg.outside_counted[t], _VIZ["excess"])
        cnts, _ = cv2.findContours(
            reference.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(big, [c * scale for c in cnts], -1, _VIZ["ref_line"], 2)
    return _viz_label(big, f"area T{t}  ex={outside[t]:.3f} df={inside[t]:.3f}")


def _viz_coverage_panels(
    reference: np.ndarray, other: np.ndarray, t: int, scale: int
) -> list[np.ndarray]:
    """Two coverage panels (overshoot, shortfall): normal probes coloured by hit."""
    overshoot, shortfall, dbg = outline_normal_coverage(reference, other, [t], debug=True)
    if dbg is None:
        base = _viz_base(reference, other, scale)
        return [_viz_label(base, "overshoot: n/a"), _viz_label(base, "shortfall: n/a")]
    n = len(dbg.points)
    idx = np.arange(0, n, max(1, n // 80))  # subsample so arrows are legible
    pts = dbg.points[idx]
    dot = max(2, scale)
    panels = []
    specs = (
        ("overshoot", dbg.overshoot_probe[t][idx], dbg.overshoot_hit[t][idx],
         _VIZ["hit"], overshoot[t]),
        ("shortfall", dbg.shortfall_probe[t][idx], dbg.shortfall_hit[t][idx],
         _VIZ["short"], shortfall[t]),
    )
    for name, probe, hit, hit_col, frac in specs:
        big = _viz_base(reference, other, scale)
        for (px, py), (qx, qy), is_hit in zip(pts, probe, hit):
            a = (int(round(px * scale)), int(round(py * scale)))
            b = (int(qx * scale), int(qy * scale))
            cv2.line(big, a, b, _VIZ["normal"], 1)
            cv2.circle(big, b, dot, hit_col if is_hit else _VIZ["miss"], -1)
        panels.append(_viz_label(big, f"{name} T{t} = {frac:.3f}"))
    return panels


def _viz_build_pair(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    """Build (reference, other) full-frame bool masks from CLI args + a title.

    Either ``--reference-mask``/``--other-mask`` PNGs, or ``--seq``/``--frame``
    pulling SAM (=other) and GT or LoMa-corr (=reference) from the dataset.
    """
    if args.reference_mask and args.other_mask:
        reference = _load_mask(args.reference_mask)
        other = _load_mask(args.other_mask)
        if reference is None or other is None:
            raise FileNotFoundError("could not load --reference-mask / --other-mask")
        if reference.shape != other.shape:
            raise ValueError(f"shape mismatch {reference.shape} vs {other.shape}")
        return reference, other, f"ref={args.reference_mask.name} other={args.other_mask.name}"

    if args.seq is None or args.frame is None:
        raise ValueError("provide --reference-mask+--other-mask, or --seq+--frame")
    gt_corners, _ = pti_load_gt(args.base_dir, args.seq, args.initfix_dir)
    other = _load_mask(args.masks_dir / args.seq / f"{args.frame + 1:05d}.png")
    if other is None:
        raise FileNotFoundError(f"no SAM mask for {args.seq} frame {args.frame}")
    h, w = other.shape
    gt = corners_to_mask(gt_corners[args.frame], (h, w)) > 0
    if args.reference == "gt":
        return gt, other, f"{args.seq} f{args.frame} ref=GT other=SAM"
    init_mask = _load_mask(args.masks_dir / args.seq / "00001.png")
    if init_mask is None:
        init_mask = corners_to_mask(gt_corners[0], (h, w)) > 0
    debug = _load_debug(args.results_dir, args.seq)
    if debug is None or debug.get(args.frame) is None or debug[args.frame][0] is None:
        raise ValueError(f"no LoMa homography for {args.seq} frame {args.frame}")
    corr = cv2.warpPerspective(init_mask.astype(np.uint8), debug[args.frame][0], (w, h)) > 0
    return corr, other, f"{args.seq} f{args.frame} ref=LoMa-corr other=SAM"


def cmd_viz(args: argparse.Namespace) -> None:
    """Render how the metrics count on a single (reference, other) mask pair."""
    reference, other, title = _viz_build_pair(args)
    t = int(args.threshold)
    pad = max(t + 4, 6)
    box = _crop_union(reference, other, pad=pad)
    if box is None:
        raise ValueError("both masks are empty")
    y0, y1, x0, x1 = box
    ref_crop = reference[y0:y1, x0:x1]
    other_crop = other[y0:y1, x0:x1]
    ref_area = int(ref_crop.sum())

    sdf = signed_distance(ref_crop)
    panels = [
        _viz_label(_viz_base(ref_crop, other_crop, args.scale), "masks (ref=blue other=grey)"),
        _viz_area_panel(ref_crop, other_crop, sdf, t, ref_area, args.scale),
        *_viz_coverage_panels(ref_crop, other_crop, t, args.scale),
    ]
    gap = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
    strip = panels[0]
    for panel in panels[1:]:
        strip = np.hstack([strip, gap, panel])
    header = np.full((24, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    out = np.vstack([header, strip])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), out)
    print(f"reference area : {ref_area}")
    print(f"Wrote          : {args.output}")


def _crop_union(*masks: np.ndarray, pad: int) -> Optional[tuple[int, int, int, int]]:
    """Union bounding box of bool ``masks`` padded by ``pad`` (clamped)."""
    union = np.zeros_like(masks[0])
    for m in masks:
        union |= m
    ys = np.where(union.any(axis=1))[0]
    xs = np.where(union.any(axis=0))[0]
    if ys.size == 0:
        return None
    h, w = masks[0].shape
    return (
        max(0, ys[0] - pad), min(h, ys[-1] + pad + 1),
        max(0, xs[0] - pad), min(w, xs[-1] + pad + 1),
    )


# --------------------------------------------------------------------------- #
# compare — live-run masks vs the static-study substrate
# --------------------------------------------------------------------------- #
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two bool masks (1.0 when both empty)."""
    a, b = a > 0, b > 0
    union = int((a | b).sum())
    return 1.0 if union == 0 else int((a & b).sum()) / union


def _diff_panel(base: np.ndarray, corr: np.ndarray, scale: int, caption: str) -> Optional[np.ndarray]:
    """Captioned overlay of two masks: overlap grey, baseline-only red, run-only blue."""
    box = _crop_union(base, corr, pad=6)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    b, c = base[y0:y1, x0:x1], corr[y0:y1, x0:x1]
    h, w = b.shape
    img = np.full((h, w, 3), 255, np.uint8)
    img[b & c] = (200, 200, 200)   # overlap (grey)
    img[b & ~c] = (60, 60, 235)    # baseline/substrate only (red, BGR)
    img[c & ~b] = (235, 120, 40)   # run only (blue, BGR)
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return _viz_label(big, caption)


def cmd_compare(args: argparse.Namespace) -> None:
    """Diff the live-run SAM3 masks against the static-study substrate per frame.

    The feasibility study treats the precomputed SAM3 masks as the run's mask
    stream. The deployed correction run instead saves the *live* SAM3 masks it
    actually used (``<corr_dir>/masks/<seq>/<frame+1>.png``). On frames before a
    correction fired the run injected nothing, so its mask should equal the
    substrate — IoU < 1 there means the study analysed masks the real run never
    saw, which breaks the trigger's premise independently of any correction.
    """
    sequences = get_sequences(
        "pti", args.base_dir,
        sequences=args.sequences, subset=args.subset,
        subset_file=args.subset_file, split=args.split,
    )
    corr_dir = compute_output_dir(paths.OUTPUT_DIRECTORY, "pti", args.config)
    corr_masks_root = corr_dir / "masks"
    baseline_root = args.baseline_masks_dir

    rows: list[tuple] = []
    pre_ious: list[float] = []
    post_ious: list[float] = []
    seqs_pre_mismatch = 0
    seqs_compared = 0
    bar = tqdm(sequences, desc="compare", unit="seq")
    for seq in bar:
        seq_corr_dir = corr_masks_root / seq
        if not seq_corr_dir.exists():
            continue
        seqs_compared += 1
        flags = _load_corrected_flags(corr_dir, seq) or {}
        fire_frames = sorted(f for f, c in flags.items() if c)
        first_fire = fire_frames[0] if fire_frames else None
        seq_pre_bad = False
        mask_files = sorted(
            p for p in seq_corr_dir.glob("*.png") if not p.stem.endswith("_corr")
        )
        for p in mask_files:
            try:
                fi = int(p.stem) - 1  # filename is 1-based frame number
            except ValueError:
                continue
            corr_mask = _load_mask(p)
            base_mask = _load_mask(baseline_root / seq / p.name)
            if corr_mask is None or base_mask is None or corr_mask.shape != base_mask.shape:
                continue
            iou = _iou(base_mask, corr_mask)
            pre = first_fire is None or fi < first_fire
            phase = "nofire" if first_fire is None else ("pre" if pre else "post")
            rows.append((seq, fi, phase, iou, int(bool(flags.get(fi, False)))))
            if pre:
                pre_ious.append(iou)
                if iou < args.iou_thr:
                    seq_pre_bad = True
            else:
                post_ious.append(iou)
        if seq_pre_bad:
            seqs_pre_mismatch += 1
        bar.set_postfix(
            cmp=seqs_compared,
            pre_mism=seqs_pre_mismatch,
            pre_meanIoU=(f"{np.mean(pre_ious):.3f}" if pre_ious else "n/a"),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mask_compare.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame_idx", "phase", "iou", "fired"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6g}", r[4]])

    def _stat(name: str, vals: list[float]) -> None:
        if not vals:
            print(f"{name}: no frames")
            return
        a = np.asarray(vals)
        print(
            f"{name}: n={len(a)} meanIoU={a.mean():.4f} medianIoU={np.median(a):.4f} "
            f"minIoU={a.min():.4f} frac<{args.iou_thr}={float((a < args.iou_thr).mean()):.3f}"
        )

    print(f"Config         : {args.config.stem}")
    print(f"Run masks      : {corr_masks_root}")
    print(f"Substrate masks: {baseline_root}")
    print(f"Seqs compared  : {seqs_compared}")
    _stat("PRE-fire (run never injected here; should match substrate)", pre_ious)
    _stat("POST-fire (after a correction was injected)", post_ious)
    print(f"Seqs with a pre-fire mismatch (IoU<{args.iou_thr}): "
          f"{seqs_pre_mismatch}/{seqs_compared}")
    print(f"Per-frame CSV  : {csv_path}")

    if args.seq is not None:
        seq_rows = sorted(
            (r for r in rows if r[0] == args.seq and r[2] in ("pre", "nofire")),
            key=lambda r: r[3],
        )[: args.num]
        panels = []
        for seq, fi, _phase, iou, _fired in seq_rows:
            name = f"{fi + 1:05d}.png"
            base_mask = _load_mask(baseline_root / seq / name)
            corr_mask = _load_mask(corr_masks_root / seq / name)
            if base_mask is None or corr_mask is None:
                continue
            panel = _diff_panel(
                base_mask, corr_mask, args.scale, f"f{fi} IoU={iou:.3f}"
            )
            if panel is not None:
                panels.append(panel)
        if panels:
            wmax = max(pn.shape[1] for pn in panels)
            padded = [
                np.pad(pn, ((0, 0), (0, wmax - pn.shape[1]), (0, 0)),
                       constant_values=255)
                for pn in panels
            ]
            montage = np.vstack(padded)
            header = np.full((24, montage.shape[1], 3), 255, np.uint8)
            cv2.putText(
                header, f"{args.seq}: worst pre-fire frames  red=substrate-only blue=run-only",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
            out_png = args.output_dir / f"mask_compare_{args.seq}.png"
            cv2.imwrite(str(out_png), np.vstack([header, montage]))
            print(f"Montage        : {out_png}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the dump/report/viz/compare CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    out_default = paths.OUTPUT_DIRECTORY / "experiments" / "sam_overseg_correction"

    decoupled_default = (
        paths.OUTPUT_DIRECTORY / "loma_samh_decoupled_rerun" / DATASETS["pti"].label
    )

    d = sub.add_parser("dump", help="per-frame features -> per-sequence CSV")
    d.add_argument("--results-dir", type=Path, default=decoupled_default,
                   help="baseline (no-correction) run dir with {seq}_debug.npz "
                        f"(default: {decoupled_default})")
    d.add_argument("--masks-dir", type=Path, default=paths.SAM3_PRECOMPUTED_MASKS_PTI,
                   help="SAM3 masks: <dir>/<seq>/{frame+1:05d}.png")
    d.add_argument("--base-dir", type=Path, default=paths.PT_BASE)
    d.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    d.add_argument("--split", choices=["train", "test"], default="test")
    d.add_argument("--sequences", nargs="+")
    d.add_argument("--subset")
    d.add_argument("--subset-file", type=Path)
    d.add_argument("--output-dir", type=Path, default=out_default)
    d.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    d.set_defaults(func=cmd_dump)

    r = sub.add_parser("report", help="per-sequence CSVs -> standalone HTML")
    r.add_argument("--per-seq-dir", type=Path,
                   default=out_default / "per_sequence",
                   help="directory of {seq}.csv from dump")
    r.add_argument("--output", type=Path, default=out_default / "report.html")
    r.add_argument("--base-dir", type=Path, default=paths.PT_BASE,
                   help="PlanarTrack base dir for animated previews "
                        f"(default: {paths.PT_BASE})")
    r.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR,
                   help="PlanarTrack initfix dir for GT corners "
                        f"(default: {paths.PT_INITFIX_DIR})")
    r.add_argument("--n-align", type=float, default=15.0,
                   help="N: LoMa-correct corner alignment error threshold (px)")
    r.add_argument("--t-label", type=int, default=10,
                   help=f"T_label: over-seg boundary tolerance (px); one of {T_GRID}")
    r.add_argument("--frac-label", type=float, default=0.10,
                   help="over-seg area fraction threshold for the GT label")
    r.add_argument("--config", type=Path, default=None,
                   help="deployed correction tracker config (e.g. "
                        "configs/tracker/loma_samh_correct_decoupled_excessdepth.py). "
                        "Enables the expected-vs-actual localization toggle and aligns "
                        "the expected map to the config's sam_excess_depth trigger.")
    r.add_argument("--baseline-dir", type=Path, default=decoupled_default,
                   help="no-correction baseline run dir for the per-frame error "
                        f"comparison (default: {decoupled_default})")
    r.add_argument("--helped-eps", type=float, default=5.0,
                   help="alignment-error delta (px) above which a frame counts "
                        "better/worse vs the baseline run")
    r.set_defaults(func=cmd_report)

    v = sub.add_parser("viz", help="render how the metrics count on one mask pair")
    v.add_argument("--reference-mask", type=Path, help="reference mask PNG")
    v.add_argument("--other-mask", type=Path, help="other mask PNG (vs reference)")
    v.add_argument("--seq", help="instead of PNGs: dataset sequence name")
    v.add_argument("--frame", type=int, help="0-based frame index (>=1) for --seq")
    v.add_argument("--reference", choices=["gt", "corr"], default="gt",
                   help="reference for --seq mode (other is always SAM)")
    v.add_argument("--threshold", "-T", type=int, default=10, help="tolerance T (px)")
    v.add_argument("--scale", type=int, default=3, help="upscale factor for the render")
    v.add_argument("--masks-dir", type=Path, default=paths.SAM3_PRECOMPUTED_MASKS_PTI)
    v.add_argument("--results-dir", type=Path, default=decoupled_default)
    v.add_argument("--base-dir", type=Path, default=paths.PT_BASE)
    v.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    v.add_argument("--output", type=Path, default=out_default / "viz.png")
    v.set_defaults(func=cmd_viz)

    c = sub.add_parser(
        "compare",
        help="diff a correction run's live SAM3 masks vs the study substrate",
    )
    c.add_argument("--config", type=Path, required=True,
                   help="deployed correction tracker config; its run dir holds the "
                        "live masks at <run_dir>/masks/<seq>/")
    c.add_argument("--baseline-masks-dir", type=Path,
                   default=paths.SAM3_PRECOMPUTED_MASKS_PTI,
                   help="substrate masks the study assumed: <dir>/<seq>/{frame+1:05d}.png "
                        f"(default: {paths.SAM3_PRECOMPUTED_MASKS_PTI}). Point at another "
                        "run's masks/ dir to compare two runs directly.")
    c.add_argument("--iou-thr", type=float, default=0.95,
                   help="IoU below which a pre-fire frame counts as a mismatch")
    c.add_argument("--seq", help="render a worst-pre-fire-frames diff montage for this seq")
    c.add_argument("--num", type=int, default=12, help="montage frame count (--seq)")
    c.add_argument("--scale", type=int, default=2, help="montage upscale factor")
    c.add_argument("--base-dir", type=Path, default=paths.PT_BASE)
    c.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    c.add_argument("--split", choices=["train", "test"], default="test")
    c.add_argument("--sequences", nargs="+")
    c.add_argument("--subset")
    c.add_argument("--subset-file", type=Path)
    c.add_argument("--output-dir", type=Path, default=out_default)
    c.set_defaults(func=cmd_compare)
    return p


def main() -> None:
    """Parse args and dispatch the chosen subcommand."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
