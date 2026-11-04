#!/usr/bin/env python3
"""Catastrophic-LoMA gallery: list + visualize high-inlier correspondence failures.

The selection headroom analysis found the dominant recoverable error mass on
frames where the LoMA branch passes the inlier gate — often with hundreds of
RANSAC inliers — yet its homography is catastrophically wrong (median 30 px,
up to hundreds), while the stored SAM-H candidate is fine. No online signal
tried so far separates these frames. This tool prepares them for manual
inspection: where the keypoint-sampling (densify) mask was, where the matches
and RANSAC inliers landed, and how the estimate failed.

Design: the ``render`` step does the *expensive* work once — recompute LoMA
keypoints/matches/RANSAC exactly as the tracker did — and dumps raw data
(frames, sampling masks, correspondences, homographies, quads) into
``gallery_data.js``.  ALL visualization happens in the browser (no baked PNGs,
no JS dependencies); ``index.html`` holds only the viewer and loads the data via
``<script src>``, so the gallery can be restyled or extended with new inspection
tools by editing ``index.html`` and reloading the browser — no GPU rerun.

Subcommands (PTi only — the phenomenon is PTi-specific):
    list    selector_study per-frame CSVs -> ranked ``breakdown_frames.csv``
            (sorted by inlier count, then LoMA error) + a per-sequence error-
            floor triage table.  Cheap, login-node OK.
    sample  ``breakdown_frames.csv`` -> an *unbiased* ``breakdown_frames.csv``
            in a sibling directory: a seeded random draw stratified over
            disagreement deciles, capped per sequence.  The ``list`` ranking is
            inlier-descending, so tags collected on it describe each failure
            class but cannot estimate class shares; this draw can.  Cheap,
            login-node OK; ``render`` then runs on the sample directory.
    render  Dump data + interactive gallery for the top frames, at most
            ``--max-per-seq`` per sequence (default 5) so the gallery surveys
            the failure-mode variety instead of collapsing onto the few
            sequences whose inlier counts dominate the ranking.
            Recomputes LoMA per frame -> needs a GPU (on CPU it is minutes per
            frame; the render aborts unless --allow-cpu). e.g.:
            ``python slurm/submit_cmd.py --job-name breakdown-gallery --gpus 1 \\
              --time 08:00:00 -- \\
              python experiments/loma_breakdown_gallery.py render``

Gallery thumbnail (drawn client-side): left = init frame (sampling-mask tint,
template match points, GT quads — initfix green / original-PT magenta),
right = current frame (sampling-mask tint, matches inliers green / outliers red,
quads GT green / loma red / samh blue).  Clicking opens the overlay inspector:
both frames superimposed via a selectable homography (identity / loma / samh /
GT / recomputed-loma), in the template or current coordinate frame, with
green/magenta anaglyph, blink, single-image, or alpha compositing, and a
togglable, subsamplable overlay of the correspondences drawn as residual
vectors under the selected homography.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from flatsam2.datasets import (
    corners_to_mask,
    pt_image_paths,
    pt_load_gt,
    pti_load_gt,
    warp_corners,
)
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.tracking.common import _inflate_mask


_LABEL = DATASETS["pti"].label

LIST_HEADER = [
    "seq",
    "frame_idx",
    "n_inliers",
    "err_loma",
    "err_samh",
    "disagreement",
    "loma_speed",
]

# Cap correspondences embedded per pair in the viewer (even stride).
_STORE_PTS_CAP = 200
_JPEG_QUALITY = 82


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #


def cmd_list(args: argparse.Namespace) -> None:
    """Rank loma-selected catastrophic frames from the selector-study CSVs."""
    files = sorted(args.selector_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no selector_study CSVs in {args.selector_dir} — run "
            "`selector_study.py dump` first"
        )
    rows: list[dict] = []
    for path in files:
        with path.open("r", newline="", encoding="utf-8") as f:
            for d in csv.DictReader(f):
                if d["selected"] != "loma_H" or int(d["evaluated"]) != 1:
                    continue
                err_loma = float(d["err_loma"])
                err_samh = float(d["err_samh"])
                n_inl = int(float(d["n_inliers"]))
                if not np.isfinite(err_loma) or err_loma < args.min_err_loma:
                    continue
                if n_inl < args.min_inliers:
                    continue
                if args.max_err_samh is not None and not (
                    np.isfinite(err_samh) and err_samh <= args.max_err_samh
                ):
                    continue
                rows.append(
                    {
                        "seq": path.stem,
                        "frame_idx": int(float(d["frame_idx"])),
                        "n_inliers": n_inl,
                        "err_loma": err_loma,
                        "err_samh": err_samh,
                        "disagreement": float(d["disagreement"]),
                        "loma_speed": float(d["loma_speed"]),
                    }
                )
    rows.sort(key=lambda r: (-r["n_inliers"], -r["err_loma"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "breakdown_frames.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LIST_HEADER)
        w.writeheader()
        w.writerows(rows)

    n_seqs = len({r["seq"] for r in rows})
    print(
        f"{len(rows)} catastrophic loma-selected frames "
        f"(err_loma >= {args.min_err_loma}, inliers >= {args.min_inliers}, "
        f"samh <= {args.max_err_samh}) across {n_seqs} sequences"
    )
    print(f"List: {out_csv}\n")

    # Per-sequence triage: the min LoMA error is the sequence's error "floor".
    # A high floor (every flagged frame already >= ~15 px) is the signature of a
    # mis-defined template / imprecise init GT, not a genuine per-frame breakdown.
    by_seq: dict[str, list[dict]] = {}
    for r in rows:
        by_seq.setdefault(r["seq"], []).append(r)
    floors = sorted(
        ((s, min(x["err_loma"] for x in rr), len(rr)) for s, rr in by_seq.items()),
        key=lambda t: -t[1],
    )
    print("Per-seq error floor (min err_loma) — high floor ~ template/init-GT problem:")
    print("| seq | n_frames | err_loma floor |")
    print("|---|---|---|")
    for s, floor, n in floors[: args.top_n]:
        print(f"| {s} | {n} | {floor:.1f} |")

    print("\nTop frames by inlier count:")
    print("| seq | frame | inliers | err_loma | err_samh |")
    print("|---|---|---|---|---|")
    for r in rows[: args.top_n]:
        print(
            f"| {r['seq']} | {r['frame_idx']} | {r['n_inliers']} "
            f"| {r['err_loma']:.0f} | {r['err_samh']:.1f} |"
        )


# --------------------------------------------------------------------------- #
# sample — unbiased draw for class-share estimation
# --------------------------------------------------------------------------- #


def cmd_sample(args: argparse.Namespace) -> None:
    """Draw a disagreement-stratified random sample of the breakdown frames."""
    src = args.list_dir / "breakdown_frames.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run the `list` subcommand first")
    with src.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{src} is empty")

    # Strata over disagreement deciles: the tagged failure classes separate
    # sharply on this axis (identity failures high, geometry failures low), so
    # stratifying it keeps every class represented at a known sampling rate.
    dis = np.array([float(r["disagreement"]) for r in rows])
    finite = np.isfinite(dis)
    edges = np.quantile(dis[finite], np.linspace(0, 1, args.n_strata + 1))
    strata = np.digitize(dis, edges[1:-1])
    strata[~finite] = args.n_strata  # NaNs form their own stratum

    rng = np.random.default_rng(args.seed)
    n_nonempty = sum(1 for s in range(args.n_strata + 1) if (strata == s).any())
    per_stratum = max(1, -(-args.n_frames // max(1, n_nonempty)))
    picked: list[int] = []
    taken_idx: set[int] = set()
    per_seq: dict[str, int] = {}

    def take(i: int) -> bool:
        """Accept row ``i`` unless its sequence already hit the per-sequence cap."""
        seq = rows[i]["seq"]
        if i in taken_idx or per_seq.get(seq, 0) >= args.max_per_seq:
            return False
        picked.append(i)
        taken_idx.add(i)
        per_seq[seq] = per_seq.get(seq, 0) + 1
        return True

    for s in range(args.n_strata + 1):
        idx = np.nonzero(strata == s)[0]
        rng.shuffle(idx)
        taken = 0
        for i in idx:
            if take(int(i)) and (taken := taken + 1) >= per_stratum:
                break

    # Top up to the requested size from whatever the per-sequence cap left over,
    # so a stratum that runs dry does not shrink the whole sample.
    spare = np.arange(len(rows))
    rng.shuffle(spare)
    for i in spare:
        if len(picked) >= args.n_frames:
            break
        take(int(i))

    picked.sort(key=lambda i: (rows[i]["seq"], int(rows[i]["frame_idx"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "breakdown_frames.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LIST_HEADER)
        w.writeheader()
        w.writerows({k: rows[i][k] for k in LIST_HEADER} for i in picked)

    rate = len(picked) / len(rows)
    print(
        f"sampled {len(picked)} of {len(rows)} breakdown frames "
        f"({rate * 100:.1f}%) across {len({rows[i]['seq'] for i in picked})} sequences, "
        f"{args.n_strata} disagreement strata, seed {args.seed}, "
        f"max {args.max_per_seq}/sequence"
    )
    print("Per-stratum draw (tag counts scale back to the population by 1/rate):")
    print("| stratum | disagreement range | population | sampled |")
    print("|---|---|---|---|")
    for s in range(args.n_strata + 1):
        n_pop = int((strata == s).sum())
        n_smp = sum(1 for i in picked if strata[i] == s)
        rng_txt = (
            "non-finite" if s == args.n_strata else f"{edges[s]:.1f} – {edges[s + 1]:.1f} px"
        )
        print(f"| {s} | {rng_txt} | {n_pop} | {n_smp} |")
    print(f"\nSample list: {out_csv}")


# --------------------------------------------------------------------------- #
# render — data dump only; all drawing happens in the browser
# --------------------------------------------------------------------------- #


def _load_tracker_config(path: Path):
    """Load a tracker config file's ``config`` object (the configs' own idiom)."""
    spec = importlib.util.spec_from_file_location("gallery_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def _load_mask(path: Path) -> Optional[np.ndarray]:
    """Binary mask PNG as uint8 {0,1}, or None when missing."""
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else (raw > 0).astype(np.uint8)


def _stored_candidates(npz_path: Path) -> dict[int, dict[str, Optional[np.ndarray]]]:
    """frame_idx -> stored candidate homographies from a run's debug sidecar."""
    if not npz_path.exists():
        return {}
    data = np.load(npz_path)
    if "frame_idx" not in data:
        return {}
    out: dict[int, dict[str, Optional[np.ndarray]]] = {}
    for k, i in enumerate(data["frame_idx"]):
        row = {}
        for cand in ("loma_H", "samh_H"):
            H = data[cand][k] if cand in data else np.full((3, 3), np.nan)
            row[cand] = H if np.isfinite(H).all() else None
        out[int(i)] = row
    return out


def _select_rows(
    rows: list[dict], max_per_seq: int, max_total: int, sequences: Optional[list[str]]
) -> list[dict]:
    """Take up to ``max_per_seq`` highest-ranked frames per sequence (ranking is
    the CSV order: inlier count desc, then LoMA error), bounded by ``max_total``.
    """
    if sequences:
        keep = set(sequences)
        rows = [r for r in rows if r["seq"] in keep]
    out: list[dict] = []
    per_seq: dict[str, int] = {}
    for r in rows:
        s = r["seq"]
        if per_seq.get(s, 0) >= max_per_seq:
            continue
        out.append(r)
        per_seq[s] = per_seq.get(s, 0) + 1
        if len(out) >= max_total:
            break
    return out


def _gt_homography(c0: np.ndarray, ci: np.ndarray) -> Optional[np.ndarray]:
    """4-point homography mapping init corners c0 -> current corners ci; (2,4)."""
    if not (np.isfinite(c0).all() and np.isfinite(ci).all()):
        return None
    try:
        H = cv2.getPerspectiveTransform(
            c0.T.astype(np.float32), ci.T.astype(np.float32)
        )
    except cv2.error:
        return None
    return H if np.isfinite(H).all() else None


def _h_list(H: Optional[np.ndarray]) -> Optional[list]:
    """3x3 homography -> nested python list rounded for compact JSON, or None."""
    if H is None:
        return None
    return [[round(float(v), 8) for v in row] for row in np.asarray(H)]


def _quad_list(corners: Optional[np.ndarray]) -> Optional[list]:
    """(2,4) corners -> [[x..],[y..]] rounded, or None."""
    if corners is None or not np.isfinite(corners).all():
        return None
    return [[round(float(v), 2) for v in row] for row in np.asarray(corners)]


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Save a {0,1} mask as an 8-bit PNG (0/255)."""
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def cmd_render(args: argparse.Namespace) -> None:
    """Recompute LoMA on the selected frames and dump the interactive gallery."""
    list_csv = args.output_dir / "breakdown_frames.csv"
    if not list_csv.exists():
        raise FileNotFoundError(f"{list_csv} missing — run the `list` subcommand first")
    with list_csv.open("r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = _select_rows(all_rows, args.max_per_seq, args.max_total, args.sequences)
    if not rows:
        print("nothing to render")
        return

    cfg = _load_tracker_config(args.config)
    from flatsam2.loma_models import build_loma_model
    from flatsam2.loma_tracker import LoMaTracker

    loma = LoMaTracker(build_loma_model(cfg.loma.model))
    device = loma._device()
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit(
            f"LoMA model is on {device} — the render recomputes LoMA per frame and "
            "would take minutes each on CPU. Submit with a GPU "
            "(slurm/submit_cmd.py --gpus 1 ...), or pass --allow-cpu to force."
        )
    print(f"LoMA device: {device}")

    by_seq: dict[str, list[dict]] = {}
    for r in rows:
        by_seq.setdefault(r["seq"], []).append(r)

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []  # gallery order (id, seq, caption)
    pairs: list[dict] = []    # full per-pair payload for the viewer
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]

    for seq in sorted(by_seq):
        seq_rows = by_seq[seq]
        try:
            gt_corners, _valid = pti_load_gt(args.base_dir, seq, args.initfix_dir)
        except (ValueError, FileNotFoundError) as exc:
            print(f"skip {seq}: {exc}")
            continue
        try:
            orig_corners, _ = pt_load_gt(args.base_dir, seq)
        except (ValueError, FileNotFoundError):
            orig_corners = None

        image_paths = pt_image_paths(args.base_dir, seq)
        init_frame = cv2.imread(str(image_paths[0]))
        if init_frame is None:
            print(f"skip {seq}: init frame unreadable")
            continue
        hw = init_frame.shape[:2]
        init_mask = corners_to_mask(gt_corners[0], hw)
        init_sampling = _inflate_mask(init_mask, cfg.loma.init_mask_inflate_px)
        sampling_cfg = (
            cfg.loma.sampling.model_copy(update={"total_budget": 0})
            if cfg.loma.exact_gt_mask_on_init
            else cfg.loma.sampling
        )
        feats0 = loma.get_keypoints(init_frame, mask=init_sampling, cfg=sampling_cfg)
        stored = _stored_candidates(args.results_dir / f"{seq}_debug.npz")

        init_name = f"{seq}__init.jpg"
        init_mask_name = f"{seq}__initmask.png"
        cv2.imwrite(str(data_dir / init_name), init_frame, jpeg)
        _write_mask(data_dir / init_mask_name, init_sampling)
        init_quads = {
            "initfix": _quad_list(gt_corners[0]),
            "orig": _quad_list(orig_corners[0]) if orig_corners is not None else None,
        }

        for r in seq_rows:
            i = int(r["frame_idx"])
            frame = cv2.imread(str(image_paths[i])) if i < len(image_paths) else None
            sam_mask = _load_mask(args.masks_dir / seq / f"{i + 1:05d}.png")
            if frame is None or sam_mask is None:
                print(f"skip {seq} f{i}: frame or mask missing")
                continue
            sampling = _inflate_mask(sam_mask, cfg.loma.current_mask_inflate_px)
            feats_i = loma.get_keypoints(frame, mask=sampling, cfg=cfg.loma.sampling)
            pts_a, pts_b, _idx_a, _ = loma.match_with_indices(feats0, feats_i)
            if len(pts_a) < 4:
                print(f"skip {seq} f{i}: {len(pts_a)} matches")
                continue
            if cfg.loma.ransac_seed is not None:
                cv2.setRNGSeed(cfg.loma.ransac_seed)
            H_recomp, inliers = cv2.findHomography(
                pts_a,
                pts_b,
                getattr(cv2, cfg.loma.ransac_method),
                cfg.loma.ransac_threshold,
            )
            inlier_mask = (
                inliers.ravel().astype(bool)
                if inliers is not None
                else np.zeros(len(pts_a), dtype=bool)
            )

            cand = stored.get(i, {})
            quads = {
                "gt": _quad_list(gt_corners[i]),
                "loma": _quad_list(
                    warp_corners(cand["loma_H"], gt_corners[0])
                    if cand.get("loma_H") is not None else None
                ),
                "samh": _quad_list(
                    warp_corners(cand["samh_H"], gt_corners[0])
                    if cand.get("samh_H") is not None else None
                ),
            }

            cur_name = f"{seq}__f{i:05d}.jpg"
            cur_mask_name = f"{seq}__f{i:05d}mask.png"
            cv2.imwrite(str(data_dir / cur_name), frame, jpeg)
            _write_mask(data_dir / cur_mask_name, sampling)

            pts_a = np.asarray(pts_a, dtype=float)
            pts_b = np.asarray(pts_b, dtype=float)
            n = len(pts_a)
            sel = (
                np.unique(np.linspace(0, n - 1, _STORE_PTS_CAP).astype(int))
                if n > _STORE_PTS_CAP else np.arange(n)
            )
            H_gt = _gt_homography(gt_corners[0], gt_corners[i])

            caption = (
                f"{seq} f{i}  inl {r['n_inliers']} (recomp {int(inlier_mask.sum())})  "
                f"loma {float(r['err_loma']):.0f}px  samh {float(r['err_samh']):.1f}px"
            )
            pair_id = f"{seq}_f{i:05d}"
            pairs.append(
                {
                    "id": pair_id,
                    "seq": seq,
                    "frame": i,
                    "initImg": f"data/{init_name}",
                    "curImg": f"data/{cur_name}",
                    "initMask": f"data/{init_mask_name}",
                    "curMask": f"data/{cur_mask_name}",
                    "w0": int(init_frame.shape[1]),
                    "h0": int(init_frame.shape[0]),
                    "w1": int(frame.shape[1]),
                    "h1": int(frame.shape[0]),
                    "H": {
                        "loma": _h_list(cand.get("loma_H")),
                        "samh": _h_list(cand.get("samh_H")),
                        "gt": _h_list(H_gt),
                        "loma_recomp": _h_list(H_recomp),
                    },
                    "ptsA": [[round(float(x), 1), round(float(y), 1)]
                             for x, y in pts_a[sel]],
                    "ptsB": [[round(float(x), 1), round(float(y), 1)]
                             for x, y in pts_b[sel]],
                    "inl": [int(bool(v)) for v in inlier_mask[sel]],
                    "quads": quads,
                    "initQuads": init_quads,
                    "metrics": {
                        "n_inliers": int(r["n_inliers"]),
                        "n_recomp": int(inlier_mask.sum()),
                        "n_matches": int(n),
                        "err_loma": round(float(r["err_loma"]), 1),
                        "err_samh": round(float(r["err_samh"]), 1),
                    },
                }
            )
            entries.append({"id": pair_id, "seq": seq, "caption": caption})
            print(f"dumped {pair_id}")

    _write_gallery_html(args.output_dir, entries, pairs, all_rows)
    print(
        f"\nGallery: {args.output_dir / 'index.html'} "
        f"({len(entries)} panels, {len({e['seq'] for e in entries})} seqs)"
    )


# --------------------------------------------------------------------------- #
# interactive HTML (all rendering client-side)
# --------------------------------------------------------------------------- #


def _write_gallery_html(
    output_dir: Path,
    entries: list[dict],
    pairs: list[dict],
    all_rows: list[dict],
) -> None:
    """Emit the interactive gallery: index.html (viewer) + gallery_data.js (dump)."""
    floor: dict[str, float] = {}
    for d in all_rows:
        s = d["seq"]
        e = float(d["err_loma"])
        floor[s] = min(floor.get(s, e), e)
    seq_meta = {
        s: {"floor": round(floor.get(s, float("nan")), 1)}
        for s in {e["seq"] for e in entries}
    }
    payload = json.dumps(
        {"pairs": pairs, "entries": entries, "seqMeta": seq_meta},
        separators=(",", ":"),
    )
    # Data and view are decoupled: the (expensive) dump lands in gallery_data.js,
    # loaded via <script src>. index.html holds only the viewer code, so the
    # visualization can be edited and reloaded in the browser with no GPU rerun.
    (output_dir / "gallery_data.js").write_text(
        f"window.GALLERY_DB={payload};\n", encoding="utf-8"
    )
    (output_dir / "index.html").write_text(_GALLERY_HTML, encoding="utf-8")


_GALLERY_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LoMA breakdown gallery</title>
<style>
  :root{--bg:#141414;--fg:#e8e8e8;--mut:#9aa0a6;--card:#1e1e1e;--accent:#4ea1ff}
  *{box-sizing:border-box}
  body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
  header{padding:14px 20px;border-bottom:1px solid #333}
  h1{font-size:19px;margin:0 0 4px}
  header p{margin:0;color:var(--mut);font-size:13px;max-width:70em}
  .seqhdr{padding:10px 20px 2px;font-size:14px;color:var(--accent)}
  .seqhdr .floor{color:var(--mut);font-weight:normal;margin-left:8px}
  #gallery{padding:6px 20px 30px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px;margin:4px 0 10px}
  figure{margin:0;background:var(--card);border:1px solid #2a2a2a;border-radius:8px;overflow:hidden;cursor:pointer}
  figure canvas{width:100%;display:block;background:#000}
  figcaption{font-size:12px;color:#bbb;padding:6px 8px;font-family:ui-monospace,monospace}
  figure:hover{border-color:var(--accent)}
  figure.tagged{border-color:#4a7a4a}
  .tags{display:flex;flex-wrap:wrap;gap:4px;padding:5px 8px;border-top:1px solid #2a2a2a}
  .chip{font-size:11px;padding:1px 8px;border-radius:11px;border:1px solid #3a3a3a;background:#222;color:#c8c8c8;cursor:pointer;user-select:none}
  .chip.on{background:var(--accent);color:#000;border-color:var(--accent)}
  #tagbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 20px;border-bottom:1px solid #333;font-size:13px;background:#181818}
  #tagbar input[type=text]{background:#222;border:1px solid #3a3a3a;color:#eee;border-radius:5px;padding:3px 7px}
  #tagbar button,#tagbar select{background:#262626;color:var(--fg);border:1px solid #3a3a3a;border-radius:5px;padding:3px 8px;font-size:13px;cursor:pointer}
  #tagbar .spacer{flex:1}
  #tagcounts{color:var(--mut);font-size:12px;font-family:ui-monospace,monospace}
  #dtags{padding:6px 14px;background:#111;display:flex;flex-wrap:wrap;gap:4px;align-items:center}
  #detail{position:fixed;inset:0;background:#0c0c0c;display:none;flex-direction:column;z-index:20}
  #detail.open{display:flex}
  #bar{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:8px 14px;border-bottom:1px solid #333;font-size:13px}
  #bar .grp{display:flex;gap:6px;align-items:center}
  #bar label{color:var(--mut)}
  #bar button,#bar select{background:#262626;color:var(--fg);border:1px solid #3a3a3a;border-radius:5px;padding:3px 8px;font-size:13px;cursor:pointer}
  #bar button.on{background:var(--accent);color:#000;border-color:var(--accent)}
  #bar .title{font-family:ui-monospace,monospace;color:#ddd}
  #bar .spacer{flex:1}
  #stagewrap{flex:1;overflow:auto;display:flex;align-items:flex-start;justify-content:center;padding:14px;background:#080808}
  #stage{position:relative;transform-origin:0 0}
  #stage img{position:absolute;left:0;top:0;transform-origin:0 0;-webkit-user-drag:none}
  #ov{position:absolute;left:0;top:0;overflow:visible;pointer-events:none;z-index:5}
  .legend{font-size:12px;color:var(--mut);padding:2px 14px 8px}
  .legend b{color:#ddd}
  kbd{background:#262626;border:1px solid #3a3a3a;border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:11px}
</style></head>
<body>
<header>
  <h1>Catastrophic LoMA failures — high inliers, wrong homography</h1>
  <p>Thumbnails (drawn in-browser): left = init frame (sampling mask tinted; template match points; GT quads —
  <b style="color:#0c0">initfix green</b>, <b style="color:#f0f">original-PT magenta</b>), right = current frame
  (inliers green / outliers red; quads GT green, loma red, samh blue). Click for the interactive overlay inspector.
  A high per-sequence error <i>floor</i> (per group) means every flagged frame is already ≥ that error — the
  signature of an imprecise init annotation, not a per-frame breakdown.</p>
</header>
<div id="tagbar">
  <b>tags:</b>
  <input id="newTag" type="text" placeholder="new failure type…" size="16">
  <button id="addTag">+ add</button>
  <label>filter</label><select id="tagFilter"></select>
  <span class="spacer"></span>
  <button id="expTags">⭳ export json</button>
  <button id="impBtn">⭱ import</button>
  <input id="impFile" type="file" accept=".json,application/json" style="display:none">
  <span id="tagcounts"></span>
</div>
<div id="gallery"></div>

<div id="detail">
  <div id="bar">
    <span class="title" id="dTitle"></span>
    <div class="grp"><label>H</label><select id="cH"></select></div>
    <div class="grp"><label>frame</label>
      <button data-frame="template" class="on">template</button>
      <button data-frame="current">current</button></div>
    <div class="grp"><label>mode</label>
      <button data-mode="anaglyph" class="on">anaglyph</button>
      <button data-mode="blink">blink</button>
      <button data-mode="A">init</button>
      <button data-mode="B">current</button>
      <button data-mode="alpha">alpha</button>
      <input id="cAlpha" type="range" min="0" max="1" step="0.02" value="0.5" style="width:90px">
    </div>
    <div class="grp"><label>points</label>
      <button id="bPts" class="on">on</button>
      <button id="bInl">inliers only</button>
      <button id="bLines" class="on">residual lines</button>
      <button id="bQuads" class="on">quads</button>
      <button id="bMask">mask</button>
      <label>sub</label><input id="cSub" type="range" min="1" max="20" step="1" value="1" style="width:80px">
      <span id="subv" style="color:#ddd">1</span>
    </div>
    <span id="dWarn" style="color:#ffb020;font-size:12px"></span>
    <span class="spacer"></span>
    <button id="bClose">✕ close</button>
  </div>
  <div class="legend">
    <b>keys:</b> <kbd>←</kbd><kbd>→</kbd> prev/next · <kbd>space</kbd> flip init/current ·
    <kbd>a</kbd> anaglyph · <kbd>t</kbd>/<kbd>c</kbd> template/current frame · <kbd>Esc</kbd> close &nbsp;·&nbsp;
    overlay: <b style="color:#3f6">init pt</b> → <b style="color:#f66">current pt</b>, line = residual under the selected H;
    quads: <b style="color:#0e0">GT</b> · <b style="color:#f33">loma</b> · <b style="color:#08f">samh</b> · <b style="color:#0ee">init (dashed)</b>.
  </div>
  <div id="dtags"></div>
  <div id="stagewrap">
    <div id="stage">
      <img id="imgA"><img id="imgB"><img id="mskA"><img id="mskB">
      <svg id="ov" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
  </div>
</div>

<svg width="0" height="0" style="position:absolute">
  <filter id="chGreen" color-interpolation-filters="sRGB">
    <feColorMatrix type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
  <filter id="chMag" color-interpolation-filters="sRGB">
    <feColorMatrix type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
</svg>

<script src="gallery_data.js"></script>
<script>
"use strict";
const DB = window.GALLERY_DB || {};
const PAIRS = DB.pairs || [], ENTRIES = DB.entries || [], SEQMETA = DB.seqMeta || {};
const byId = {}; PAIRS.forEach(p => byId[p.id] = p);
const MASK_RGB = [0, 200, 200];

// ---- homography algebra (row-major 3x3) ----
function matI(){return [[1,0,0],[0,1,0],[0,0,1]];}
function inv(H){
  const a=H[0][0],b=H[0][1],c=H[0][2],d=H[1][0],e=H[1][1],f=H[1][2],g=H[2][0],h=H[2][1],i=H[2][2];
  const A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g,D=-(b*i-c*h),E=a*i-c*g,F=-(a*h-b*g),
        G=b*f-c*e,Hh=-(a*f-c*d),I=a*e-b*d;
  let det=a*A+b*B+c*C; if(Math.abs(det)<1e-12)det=1e-12;
  return [[A/det,D/det,G/det],[B/det,E/det,Hh/det],[C/det,F/det,I/det]];
}
function apply(H,x,y){const w=H[2][0]*x+H[2][1]*y+H[2][2]||1e-12;
  return [(H[0][0]*x+H[0][1]*y+H[0][2])/w,(H[1][0]*x+H[1][1]*y+H[1][2])/w];}
function css3d(H){
  const a=H[0][0],b=H[0][1],c=H[0][2],d=H[1][0],e=H[1][1],f=H[1][2],g=H[2][0],h=H[2][1],i=H[2][2];
  return `matrix3d(${a},${d},0,${g}, ${b},${e},0,${h}, 0,0,1,0, ${c},${f},0,${i})`;
}
function wAt(H,x,y){ return H[2][0]*x+H[2][1]*y+H[2][2]; }
// A homography is defined up to scale, but CSS matrix3d is not: it rasterises
// only the half-plane with w>0 and drops everything behind it. inv(H) routinely
// comes back with w<0 over the whole image — same map, nothing drawn. Flip the
// sign so the region of interest (the tracked quad) lands in front.
function orient(H, ax, ay){
  return wAt(H,ax,ay) < 0 ? H.map(r=>r.map(v=>-v)) : H;
}
function quadCenter(q){
  if(!q) return null;
  let sx=0, sy=0; for(let k=0;k<4;k++){ sx+=q[0][k]; sy+=q[1][k]; }
  return [sx/4, sy/4];
}
// True when the horizon line of H crosses the source image: part of it maps
// behind the camera and is unrenderable no matter which sign we pick.
function horizonCuts(H,w,h){
  const ws=[wAt(H,0,0),wAt(H,w,0),wAt(H,w,h),wAt(H,0,h)];
  return ws.some(v=>v>0) && ws.some(v=>v<0);
}

// ---- image cache ----
const imgCache = {};
function loadImg(src){
  if(!src) return Promise.resolve(null);
  if(imgCache[src]) return imgCache[src];
  const pr = new Promise(res=>{const im=new Image(); im.onload=()=>res(im); im.onerror=()=>res(null); im.src=src;});
  imgCache[src] = pr; return pr;
}

// ---- shared canvas drawing primitives ----
function tintMask(ctx, maskImg, w, h, ox, oy, rgb, alpha){
  // Additive tint of the (white-on-black) mask region — pure compositing, no
  // getImageData, so it works on file:// where the canvas is cross-origin tainted.
  if(!maskImg) return;
  const off=document.createElement("canvas"); off.width=w; off.height=h;
  const o=off.getContext("2d");
  o.drawImage(maskImg,0,0,w,h);                 // region white, background black
  o.globalCompositeOperation="multiply";        // region -> rgb, background -> black
  o.fillStyle=`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`; o.fillRect(0,0,w,h);
  ctx.save();
  ctx.globalAlpha=alpha; ctx.globalCompositeOperation="lighter";  // black adds nothing
  ctx.drawImage(off,ox,oy);
  ctx.restore();
}
function drawQuad(ctx, q, color, wdt, ox){
  if(!q) return;
  ctx.strokeStyle=color; ctx.lineWidth=wdt; ctx.beginPath();
  for(let k=0;k<4;k++){const x=q[0][k]+ox, y=q[1][k]; if(k===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
  ctx.closePath(); ctx.stroke();
}
function drawPts(ctx, pts, inl, ox){
  for(let k=0;k<pts.length;k++){
    ctx.fillStyle = inl[k]===1 ? "#20dd20" : "#e02020";
    ctx.beginPath(); ctx.arc(pts[k][0]+ox, pts[k][1], inl[k]===1?2.5:2, 0, 6.2832); ctx.fill();
  }
}

// ---- gallery (lazy canvas thumbnails), ordered worst-first ----
function buildGallery(){
  const gal = document.getElementById("gallery");
  const order = ENTRIES.slice().sort((a,b)=>{
    const ma=byId[a.id].metrics, mb=byId[b.id].metrics;
    return (mb.err_loma-ma.err_loma) || (mb.n_inliers-ma.n_inliers);
  });
  let html = `<div class="grid">`;
  for(const e of order){
    const fl = SEQMETA[e.seq] ? SEQMETA[e.seq].floor : "";
    html += `<figure data-id="${e.id}"><canvas data-id="${e.id}"></canvas>`
          + `<figcaption>${e.caption}  ·  seq floor ${fl}px</figcaption>`
          + `<div class="tags" data-id="${e.id}"></div></figure>`;
  }
  html += `</div>`;
  gal.innerHTML = html;
  gal.addEventListener("click", ev=>{
    const chip=ev.target.closest(".chip");
    if(chip){ toggleTag(chip.dataset.id, chip.dataset.tag); return; }  // tag, don't open
    const f=ev.target.closest("figure"); if(f) openDetail(f.dataset.id);
  });
  const io = new IntersectionObserver((ents)=>{
    for(const en of ents){ if(en.isIntersecting){ const cv=en.target; io.unobserve(cv); renderThumb(cv, byId[cv.dataset.id]); } }
  }, {rootMargin:"300px"});
  document.querySelectorAll("#gallery canvas").forEach(cv=>io.observe(cv));
  buildFilter(); renderAllChips(); applyFilter();
}
async function renderThumb(cv, p){
  const [ia,ib,ma,mb] = await Promise.all([loadImg(p.initImg),loadImg(p.curImg),loadImg(p.initMask),loadImg(p.curMask)]);
  const gap=8, full=p.w0+p.w1+gap, scale=Math.min(1, 760/full);
  cv.width=Math.round(full*scale); cv.height=Math.round(Math.max(p.h0,p.h1)*scale);
  const ctx=cv.getContext("2d"); ctx.save(); ctx.scale(scale,scale);
  if(ia) ctx.drawImage(ia,0,0);
  tintMask(ctx,ma,p.w0,p.h0,0,0,MASK_RGB,0.32);
  drawPts(ctx,p.ptsA,p.inl,0);
  drawQuad(ctx,p.initQuads.orig,"#ff00ff",2,0);
  drawQuad(ctx,p.initQuads.initfix,"#00cc00",2,0);
  const xo=p.w0+gap;
  if(ib) ctx.drawImage(ib,xo,0);
  tintMask(ctx,mb,p.w1,p.h1,xo,0,MASK_RGB,0.32);
  drawPts(ctx,p.ptsB,p.inl,xo);
  drawQuad(ctx,p.quads.gt,"#00cc00",3,xo);
  drawQuad(ctx,p.quads.loma,"#ff3333",2,xo);
  drawQuad(ctx,p.quads.samh,"#0088ff",2,xo);
  ctx.restore();
}

// ---- detail overlay inspector ----
const state = {idx:0, H:"gt", frame:"template", mode:"anaglyph",
               showPts:true, inliersOnly:false, showLines:true, showQuads:true, showMask:false, alpha:0.5, sub:1, blinkAB:"A"};
const D=document.getElementById("detail");
const dtags=document.getElementById("dtags");
const imgA=document.getElementById("imgA"), imgB=document.getElementById("imgB");
const mskA=document.getElementById("mskA"), mskB=document.getElementById("mskB");
const stage=document.getElementById("stage"), stagewrap=document.getElementById("stagewrap");
const ov=document.getElementById("ov"), cH=document.getElementById("cH");
const H_ORDER=[["identity","identity"],["gt","GT"],["loma","loma"],["samh","samh"],["loma_recomp","loma (recomp)"]];

function openDetail(id){ state.idx=PAIRS.findIndex(p=>p.id===id); if(state.idx<0)state.idx=0;
  D.classList.add("open"); buildHselect(); render(); renderDetailTags(); }
function closeDetail(){ D.classList.remove("open"); stopBlink(); }
function buildHselect(){
  const p=PAIRS[state.idx];
  cH.innerHTML=H_ORDER.filter(([k])=>k==="identity"||p.H[k]).map(([k,l])=>`<option value="${k}">${l}</option>`).join("");
  if(![...cH.options].some(o=>o.value===state.H)) state.H=cH.value;
  cH.value=state.H;
}
function selH(){ if(state.H==="identity")return matI(); const p=PAIRS[state.idx]; return p.H[state.H]||matI(); }
// Which frame's annotations are relevant to what's on screen.
function effShown(){ return state.mode==="A"?"A":state.mode==="B"?"B":state.mode==="blink"?state.blinkAB:"both"; }

function render(){
  const p=PAIRS[state.idx];
  document.getElementById("dTitle").textContent =
    `${p.id}  ·  inliers ${p.metrics.n_inliers} (recomp ${p.metrics.n_recomp}) / ${p.metrics.n_matches} matches  ·  err loma ${p.metrics.err_loma} samh ${p.metrics.err_samh}`;
  setSrc(imgA,p.initImg); setSrc(imgB,p.curImg); setSrc(mskA,p.initMask); setSrc(mskB,p.curMask);
  layout();
}
function setSrc(el,src){ if(el.getAttribute("src")!==src){ el.onload=layout; el.src=src; } }

function layout(){
  const p=PAIRS[state.idx], Hs=selH();
  const disp = state.frame==="template" ? [p.w0,p.h0] : [p.w1,p.h1];
  let Ta,Tb;
  if(state.frame==="template"){ Ta=matI(); Tb=inv(Hs); }
  else                        { Ta=Hs;    Tb=matI();  }
  // Anchor the w>0 half-plane on the tracked object, not on the image centre:
  // with a grazing view the two can sit on opposite sides of the horizon.
  const aA=quadCenter(p.initQuads.initfix||p.initQuads.orig) || [p.w0/2,p.h0/2];
  const aB=quadCenter(p.quads.gt||p.quads.loma||p.quads.samh) || [p.w1/2,p.h1/2];
  Ta=orient(Ta,aA[0],aA[1]); Tb=orient(Tb,aB[0],aB[1]);
  const cut=[horizonCuts(Ta,p.w0,p.h0)?"init":"", horizonCuts(Tb,p.w1,p.h1)?"current":""].filter(Boolean);
  document.getElementById("dWarn").textContent =
    cut.length ? `⚠ horizon crosses ${cut.join(" & ")} frame — part of the warp is unrenderable` : "";
  const ca=css3d(Ta), cb=css3d(Tb);
  imgA.style.transform=ca; mskA.style.transform=ca; imgB.style.transform=cb; mskB.style.transform=cb;
  imgA.style.width=mskA.style.width=p.w0+"px"; imgA.style.height=mskA.style.height=p.h0+"px";
  imgB.style.width=mskB.style.width=p.w1+"px"; imgB.style.height=mskB.style.height=p.h1+"px";
  stage.style.width=disp[0]+"px"; stage.style.height=disp[1]+"px";
  ov.setAttribute("width",disp[0]); ov.setAttribute("height",disp[1]); ov.setAttribute("viewBox",`0 0 ${disp[0]} ${disp[1]}`);
  const avail=stagewrap.clientWidth-28, s=Math.min(1,avail/disp[0]);
  stage.style.transform=`scale(${s})`;
  applyMode(); drawOverlay();
}
function applyMode(){
  [imgA,imgB].forEach(el=>{el.style.display="block";el.style.opacity="1";el.style.filter="";el.style.mixBlendMode="normal";});
  imgA.style.zIndex="1"; imgB.style.zIndex="2";
  if(state.mode==="anaglyph"){ imgA.style.filter="url(#chGreen)"; imgB.style.filter="url(#chMag)"; imgB.style.mixBlendMode="screen"; }
  else if(state.mode==="A"){ imgB.style.display="none"; }
  else if(state.mode==="B"){ imgA.style.display="none"; }
  else if(state.mode==="alpha"){ imgB.style.opacity=String(state.alpha); }
  else if(state.mode==="blink"){ if(state.blinkAB==="A")imgB.style.display="none"; else imgA.style.display="none"; }
  // sampling-mask tint layers (green init, magenta current). The masks are
  // white-on-black; `screen` makes the black background a no-op so only the
  // region tints, and the channel filters colour it.
  mskA.style.zIndex=mskB.style.zIndex="4"; mskA.style.mixBlendMode=mskB.style.mixBlendMode="screen";
  const eff=effShown();
  mskA.style.display=(state.showMask && eff!=="B")?"block":"none";  // init mask only when init shows
  mskB.style.display=(state.showMask && eff!=="A")?"block":"none";  // current mask only when current shows
  mskA.style.opacity=mskB.style.opacity="0.85";
  mskA.style.filter="url(#chGreen)"; mskB.style.filter="url(#chMag)";
}
// Map a point from its native frame (init-frame if isInit, else current-frame)
// into the active display frame under the selected homography.
function mapPt(x,y,isInit,Hs,Hi){
  if(isInit) return state.frame==="template" ? [x,y] : apply(Hs,x,y);
  return state.frame==="template" ? apply(Hi,x,y) : [x,y];
}
function drawOverlay(){
  const p=PAIRS[state.idx];
  if(!state.showPts && !state.showLines && !state.showQuads){ ov.innerHTML=""; return; }
  const Hs=selH(), Hi=inv(Hs); let parts=[];
  // Only annotate the frame(s) currently visible: init-frame overlays when the
  // init image shows, current-frame overlays when the current image shows, both
  // (plus the residual line joining them) when the frames are superimposed.
  const eff=effShown(), showInit=eff!=="B", showCur=eff!=="A";
  if(state.showPts || state.showLines){
    for(let k=0;k<p.ptsA.length;k+=state.sub){
      const inl=p.inl[k]===1; if(state.inliersOnly && !inl) continue;
      // Both endpoints always computed so the residual vector is available even
      // in single-frame views; endpoint dots are gated to the visible frame.
      const a=mapPt(p.ptsA[k][0],p.ptsA[k][1],true,Hs,Hi);
      const b=mapPt(p.ptsB[k][0],p.ptsB[k][1],false,Hs,Hi);
      const col=inl?"#25d366":"#ff5252";
      if(state.showLines) parts.push(`<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" stroke="${col}" stroke-width="1" opacity="0.8"/>`);
      if(state.showPts){
        if(showInit) parts.push(`<circle cx="${a[0].toFixed(1)}" cy="${a[1].toFixed(1)}" r="2" fill="#3f6" opacity="0.9"/>`);
        if(showCur) parts.push(`<circle cx="${b[0].toFixed(1)}" cy="${b[1].toFixed(1)}" r="2" fill="#f66" opacity="0.9"/>`);
      }
    }
  }
  if(state.showQuads){
    const poly=(q,isInit,color,w,dash)=>{
      if(!q) return; let pp=[];
      for(let k=0;k<4;k++){const m=mapPt(q[0][k],q[1][k],isInit,Hs,Hi); pp.push(m[0].toFixed(1)+","+m[1].toFixed(1));}
      parts.push(`<polygon points="${pp.join(' ')}" fill="none" stroke="${color}" stroke-width="${w}"${dash?` stroke-dasharray="${dash}"`:''} opacity="0.95"/>`);
    };
    if(showInit){
      poly(p.initQuads.initfix, true, "#00e0e0", 2, "7 5");  // init template quad (dashed cyan)
      poly(p.initQuads.orig,    true, "#ff00ff", 2, "7 5");  // original-PT init quad (dashed magenta)
    }
    if(showCur){
      poly(p.quads.gt,   false, "#00cc00", 3, null);         // current GT
      poly(p.quads.loma, false, "#ff3333", 2, null);         // predicted (loma)
      poly(p.quads.samh, false, "#0088ff", 2, null);         // predicted (samh)
    }
  }
  ov.innerHTML=parts.join("");
}

let blinkTimer=null;
function startBlink(){ stopBlink(); blinkTimer=setInterval(()=>{state.blinkAB=state.blinkAB==="A"?"B":"A";applyMode();drawOverlay();},600); }
function stopBlink(){ if(blinkTimer){clearInterval(blinkTimer);blinkTimer=null;} }

document.getElementById("bClose").onclick=closeDetail;
cH.onchange=()=>{ state.H=cH.value; layout(); };
document.querySelectorAll("#bar button[data-frame]").forEach(b=>b.onclick=()=>{state.frame=b.dataset.frame;setActive("[data-frame]",b);layout();});
document.querySelectorAll("#bar button[data-mode]").forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;setActive("[data-mode]",b);if(state.mode==="blink")startBlink();else stopBlink();applyMode();drawOverlay();});
document.getElementById("cAlpha").oninput=e=>{state.alpha=parseFloat(e.target.value);if(state.mode==="alpha")applyMode();};
document.getElementById("bPts").onclick=e=>{state.showPts=!state.showPts;e.target.classList.toggle("on",state.showPts);drawOverlay();};
document.getElementById("bInl").onclick=e=>{state.inliersOnly=!state.inliersOnly;e.target.classList.toggle("on",state.inliersOnly);drawOverlay();};
document.getElementById("bLines").onclick=e=>{state.showLines=!state.showLines;e.target.classList.toggle("on",state.showLines);drawOverlay();};
document.getElementById("bQuads").onclick=e=>{state.showQuads=!state.showQuads;e.target.classList.toggle("on",state.showQuads);drawOverlay();};
document.getElementById("bMask").onclick=e=>{state.showMask=!state.showMask;e.target.classList.toggle("on",state.showMask);applyMode();};
document.getElementById("cSub").oninput=e=>{state.sub=parseInt(e.target.value);document.getElementById("subv").textContent=state.sub;drawOverlay();};
function setActive(sel,btn){ document.querySelectorAll("#bar button"+sel).forEach(x=>x.classList.remove("on")); btn.classList.add("on"); }
function step(d){ state.idx=(state.idx+d+PAIRS.length)%PAIRS.length; buildHselect(); render(); renderDetailTags(); }

window.addEventListener("keydown", ev=>{
  if(!D.classList.contains("open")) return;
  if(ev.key==="Escape") closeDetail();
  else if(ev.key==="ArrowRight") step(1);
  else if(ev.key==="ArrowLeft") step(-1);
  else if(ev.key===" "){ ev.preventDefault(); state.blinkAB=state.blinkAB==="A"?"B":"A"; if(state.mode!=="blink"){state.mode="blink";setActive("[data-mode]",document.querySelector('[data-mode="blink"]'));} applyMode(); drawOverlay(); }
  else if(ev.key==="a"){ state.mode="anaglyph"; setActive("[data-mode]",document.querySelector('[data-mode="anaglyph"]')); stopBlink(); applyMode(); drawOverlay(); }
  else if(ev.key==="t"){ state.frame="template"; setActive("[data-frame]",document.querySelector('[data-frame="template"]')); layout(); }
  else if(ev.key==="c"){ state.frame="current"; setActive("[data-frame]",document.querySelector('[data-frame="current"]')); layout(); }
  else if(ev.key>="1" && ev.key<="9"){ const t=store.palette[parseInt(ev.key)-1]; if(t) toggleTag(PAIRS[state.idx].id, t); }
  else if(ev.key==="0"){ const t=store.palette[9]; if(t) toggleTag(PAIRS[state.idx].id, t); }
});
window.addEventListener("resize", ()=>{ if(D.classList.contains("open")) layout(); });

// ---- tag annotation subsystem (localStorage + JSON export/import) ----
const TAGKEY="loma_gallery_tags_v1";
// Seeded from the first annotation round so a fresh gallery needs no setup.
// "occlusion" is new and deliberately separate from "bad distribution": few,
// clustered inliers on a partly hidden target can still give a good homography,
// so the two must not be conflated when the tags are used as feature labels.
const SEED_PALETTE=["distractor","submatch","orientation","ok","screen","transparent",
                    "mirror","tless","bad distribution","occlusion","blur",
                    "background motion","weird"];
let store={palette:[],ann:{}}, storageOK=true;
try{ const raw=localStorage.getItem(TAGKEY); if(raw) store=JSON.parse(raw); }catch(e){ storageOK=false; }
if(!store.palette) store.palette=[]; if(!store.ann) store.ann={};
for(const t of SEED_PALETTE) if(!store.palette.includes(t)) store.palette.push(t);
function saveTags(){ if(!storageOK) return; try{ localStorage.setItem(TAGKEY, JSON.stringify(store)); }catch(e){ storageOK=false; updateCounts(); } }
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function cssq(s){ return (window.CSS&&CSS.escape)?CSS.escape(s):s; }
function tagsOf(id){ return store.ann[id]||[]; }
function chipHTML(id){
  if(!store.palette.length) return `<span style="color:#666;font-size:11px">no tags yet — add one above</span>`;
  return store.palette.map(t=>{
    const on=tagsOf(id).includes(t);
    return `<span class="chip${on?' on':''}" data-id="${esc(id)}" data-tag="${esc(t)}">${esc(t)}</span>`;
  }).join("");
}
function toggleTag(id,tag){
  const a=store.ann[id]||(store.ann[id]=[]);
  const i=a.indexOf(tag); if(i<0) a.push(tag); else a.splice(i,1);
  if(!a.length) delete store.ann[id];
  saveTags(); refreshChips(id); updateCounts(); applyFilter();
}
function figTagsEl(id){ return document.querySelector(`#gallery .tags[data-id="${cssq(id)}"]`); }
function refreshChips(id){
  const el=figTagsEl(id);
  if(el){ el.innerHTML=chipHTML(id); const fig=el.closest("figure"); if(fig) fig.classList.toggle("tagged", tagsOf(id).length>0); }
  if(D.classList.contains("open") && PAIRS[state.idx].id===id) renderDetailTags();
}
function renderAllChips(){
  document.querySelectorAll("#gallery .tags").forEach(el=>{
    const id=el.dataset.id; el.innerHTML=chipHTML(id);
    const fig=el.closest("figure"); if(fig) fig.classList.toggle("tagged", tagsOf(id).length>0);
  });
  if(D.classList.contains("open")) renderDetailTags();
  updateCounts();
}
function renderDetailTags(){
  if(!D.classList.contains("open")) return;
  dtags.innerHTML=`<b style="color:#9aa0a6">tags (keys 1-9, 0 = 10th):</b> `+chipHTML(PAIRS[state.idx].id);
}
function addNewTag(name){
  name=(name||"").trim(); if(!name || store.palette.includes(name)) return;
  store.palette.push(name); saveTags(); buildFilter(); renderAllChips();
}
function updateCounts(){
  let tagged=0; const per={};
  for(const p of PAIRS){ const a=tagsOf(p.id); if(a.length)tagged++; for(const t of a) per[t]=(per[t]||0)+1; }
  const perStr=store.palette.map(t=>`${t}:${per[t]||0}`).join("  ");
  document.getElementById("tagcounts").textContent=
    (storageOK?"":"[localStorage off — Export to save] ")+`${tagged}/${PAIRS.length} tagged   ${perStr}`;
}
function buildFilter(){
  const sel=document.getElementById("tagFilter"), cur=sel.value;
  sel.innerHTML=`<option value="__all">all</option><option value="__untag">(untagged)</option>`
    + store.palette.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join("");
  sel.value=[...sel.options].some(o=>o.value===cur)?cur:"__all";
}
function applyFilter(){
  const f=document.getElementById("tagFilter").value;
  document.querySelectorAll("#gallery figure").forEach(fig=>{
    const a=tagsOf(fig.dataset.id);
    let show=true;
    if(f==="__untag") show=a.length===0; else if(f!=="__all") show=a.includes(f);
    fig.style.display=show?"":"none";
  });
}
function exportTags(){
  const blob=new Blob([JSON.stringify(store,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob), a=document.createElement("a");
  a.href=url; a.download="loma_tags.json"; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}
function importTags(file){
  const r=new FileReader();
  r.onload=()=>{ try{
    const d=JSON.parse(r.result);
    if(Array.isArray(d.palette)) for(const t of d.palette) if(!store.palette.includes(t)) store.palette.push(t);
    if(d.ann) for(const k in d.ann) store.ann[k]=d.ann[k].slice();
    saveTags(); buildFilter(); renderAllChips(); applyFilter();
  }catch(e){ alert("bad tags file: "+e.message); } };
  r.readAsText(file);
}
document.getElementById("addTag").onclick=()=>{ const i=document.getElementById("newTag"); addNewTag(i.value); i.value=""; i.focus(); };
document.getElementById("newTag").addEventListener("keydown", e=>{ if(e.key==="Enter"){ addNewTag(e.target.value); e.target.value=""; } });
document.getElementById("tagFilter").onchange=applyFilter;
document.getElementById("expTags").onclick=exportTags;
document.getElementById("impBtn").onclick=()=>document.getElementById("impFile").click();
document.getElementById("impFile").onchange=e=>{ if(e.target.files[0]) importTags(e.target.files[0]); e.target.value=""; };
dtags.addEventListener("click", ev=>{ const c=ev.target.closest(".chip"); if(c) toggleTag(c.dataset.id, c.dataset.tag); });

buildGallery();
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the list/render CLI."""
    out_default = paths.OUTPUT_DIRECTORY / "experiments" / "loma_breakdowns" / _LABEL
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list", help="rank catastrophic loma frames -> breakdown_frames.csv")
    l.add_argument("--selector-dir", type=Path,
                   default=paths.OUTPUT_DIRECTORY / "experiments" / "selector_study"
                   / _LABEL / "per_sequence",
                   help="selector_study per-sequence CSVs (the frame-level substrate)")
    l.add_argument("--min-err-loma", type=float, default=15.0,
                   help="catastrophic threshold on the selected LoMA error (px)")
    l.add_argument("--min-inliers", type=int, default=35,
                   help="only frames at/above the deployed inlier gate")
    l.add_argument("--max-err-samh", type=float, default=15.0,
                   help="require a fine SAM-H candidate for clean contrast "
                        "(pass a huge value to disable)")
    l.add_argument("--top-n", type=int, default=40, help="stdout table length")
    l.add_argument("--output-dir", type=Path, default=out_default)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("sample", help="stratified random draw -> unbiased tagging list")
    s.add_argument("--list-dir", type=Path, default=out_default,
                   help="directory holding the full ranked breakdown_frames.csv")
    s.add_argument("--n-frames", type=int, default=150,
                   help="target sample size (split evenly over the strata)")
    s.add_argument("--n-strata", type=int, default=10,
                   help="disagreement quantile strata (plus one for non-finite)")
    s.add_argument("--max-per-seq", type=int, default=4,
                   help="cap per sequence so long breakdowns cannot dominate")
    s.add_argument("--seed", type=int, default=0, help="RNG seed (keep it fixed)")
    s.add_argument("--output-dir", type=Path, default=out_default.parent / f"{_LABEL}_sample")
    s.set_defaults(func=cmd_sample)

    r = sub.add_parser("render", help="dump data + interactive gallery (GPU)")
    r.add_argument("--config", type=Path,
                   default=Path("configs/tracker/loma_samh_decoupled_rerun.py"),
                   help="tracker config supplying the LoMA settings to replicate")
    r.add_argument("--results-dir", type=Path,
                   default=paths.OUTPUT_DIRECTORY / "loma_samh_decoupled_rerun" / _LABEL,
                   help="run dir with {seq}_debug.npz for the stored candidate quads")
    r.add_argument("--masks-dir", type=Path, default=paths.SAM3_PRECOMPUTED_MASKS_PTI)
    r.add_argument("--base-dir", type=Path, default=paths.PT_BASE)
    r.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    r.add_argument("--sequences", nargs="+", help="restrict to these sequences")
    r.add_argument("--max-per-seq", type=int, default=5,
                   help="cap panels rendered per sequence (variety over volume)")
    r.add_argument("--max-total", type=int, default=300,
                   help="overall cap on rendered panels")
    r.add_argument("--allow-cpu", action="store_true",
                   help="permit running with the LoMA model on CPU (very slow)")
    r.add_argument("--output-dir", type=Path, default=out_default)
    r.set_defaults(func=cmd_render)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
