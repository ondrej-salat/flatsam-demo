#!/usr/bin/env python3
"""Offline benchmark for mask -> homography estimators (the SAM-H branch).

The tracker's SAM-H branch answers one self-contained question: *given the
template silhouette and a segmentation mask of the object in this frame, where
is the object?* That question needs no tracker, no GPU and no SAM3 rerun — the
masks are already on disk (``sam3_candidates``) and GT corners come from the
dataset — so it can be measured directly, on a fixed test set, with every
estimator seeing byte-identical inputs.

Why this exists: ``experiments/branch_p_at_k.py`` measures branches *inside a
run*, so every comparison costs a full grid and confounds the estimator with
the tracker state feeding it. Here the initialization is a controlled variable:

  regimes  good  = GT pose + 5 px per-corner noise   (healthy tracking)
           drift = GT pose + 30 px per-corner noise  (degraded tracking)
           lost  = GT pose of a random *other* frame (estimate off the object)
           cold  = the template pose itself          (no information at all)

An estimator that only wins under ``good`` is an aligner; one that also wins
under ``lost``/``cold`` can recover from occlusion and drift, which is what the
deployed branch actually lacks (docs/2026-08-07.md, 2026-08-11 entries).

Test set (``build``): every GT-annotated frame of POT-210 (6 per sequence, so
all of them) and a seeded uniform sample of PlanarTrackInitfix frames. Cases
carry covariates measured at build time (mask/GT IoU, occlusion-style coverage
split, component count, GT scale / rotation / perspective, out-of-view share)
so ``run`` can slice results by difficulty without re-reading masks. Sequences
are split dev/test by name hash: tune on ``dev``, report on ``test``.

Imports cv2 — submit to the cpu partition, do not run on the login node.

Usage:
    python slurm/submit_cmd.py --gpus 0 -- python experiments/maskh_bench.py build
    python slurm/submit_cmd.py --gpus 0 -- python experiments/maskh_bench.py run \\
        --methods hough align align_redetect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from flatsam2.datasets import corners_to_mask, pot_load_gt, pti_load_gt, warp_corners
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.sequences import get_sequences

_BENCH_DIR = paths.OUTPUT_DIRECTORY / "experiments" / "maskh_bench"
_MASK_RUN = "sam3_candidates"
_REGIMES = ("good", "drift", "lost", "cold")
_NOISE_PX = {"good": 5.0, "drift": 30.0}
_THRESHOLDS = (5.0, 15.0)


# ---------------------------------------------------------------------------
# Test-set construction
# ---------------------------------------------------------------------------


def _mask_dir(dataset: str, seq_name: str) -> Path:
    """Directory holding the precomputed SAM3 masks of one sequence."""
    return paths.OUTPUT_DIRECTORY / _MASK_RUN / DATASETS[dataset].label / "masks" / seq_name


def _available_frames(mask_dir: Path) -> set:
    """0-based frame indices with a selected-mask PNG, from a single readdir."""
    if not mask_dir.is_dir():
        return set()
    frames = set()
    with os.scandir(mask_dir) as it:
        for entry in it:
            name = entry.name
            if len(name) == 9 and name.endswith(".png") and name[:5].isdigit():
                frames.add(int(name[:5]) - 1)
    return frames


def _load_mask(dataset: str, seq_name: str, frame_idx: int) -> Optional[np.ndarray]:
    """Read one selected SAM3 mask as a bool array, or None when unreadable."""
    path = _mask_dir(dataset, seq_name) / f"{frame_idx + 1:05d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else img > 127


def _split_of(seq_name: str) -> str:
    """Deterministic dev/test assignment of a sequence (stable across runs)."""
    digest = hashlib.md5(seq_name.encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def _pose_stats(init_corners: np.ndarray, gt_corners: np.ndarray, shape: Tuple[int, int]) -> Dict:
    """Scale / rotation / perspective / out-of-view descriptors of a GT pose."""
    src = init_corners.T.astype(np.float32)
    dst = gt_corners.T.astype(np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return {"gt_scale": float("nan"), "gt_rot_deg": float("nan"),
                "gt_persp": float("nan"), "oov_frac": float("nan")}
    A = H[:2, :2] / (H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0)
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    rot = float(np.degrees(np.arctan2(A[1, 0] + A[0, 1], A[0, 0] - A[1, 1])))
    persp = float(np.linalg.norm(H[2, :2]) * max(shape) / max(abs(H[2, 2]), 1e-12))
    h, w = shape
    inside = (
        (gt_corners[0] >= 0) & (gt_corners[0] < w) & (gt_corners[1] >= 0) & (gt_corners[1] < h)
    )
    return {"gt_scale": scale, "gt_rot_deg": rot, "gt_persp": persp,
            "oov_frac": float(1.0 - inside.mean())}


def _mask_stats(mask: np.ndarray, gt_corners: np.ndarray) -> Dict:
    """Mask-vs-GT-quad agreement: IoU, the two one-sided coverages, components.

    ``covered`` (share of the GT quad the mask explains) drops under occlusion
    while ``precision`` (share of the mask inside the GT quad) stays high; the
    reverse pattern is over-segmentation. Their split is what distinguishes the
    two failure modes an estimator must survive.
    """
    gt_mask = corners_to_mask(gt_corners, mask.shape) > 0
    inter = float(np.logical_and(mask, gt_mask).sum())
    area_mask = float(mask.sum())
    area_gt = float(gt_mask.sum())
    union = area_mask + area_gt - inter
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    big = [s for s in stats[1:, cv2.CC_STAT_AREA] if s > 0.05 * max(area_mask, 1.0)]
    return {
        "iou": inter / union if union > 0 else 0.0,
        "precision": inter / area_mask if area_mask > 0 else 0.0,
        "covered": inter / area_gt if area_gt > 0 else 0.0,
        "area_mask": area_mask,
        "area_gt": area_gt,
        "n_components": len(big),
        "n_labels": int(n_labels - 1),
    }


def _init_corners_for_regimes(
    gt_corners: np.ndarray,
    template_corners: np.ndarray,
    other_pose: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, List[List[float]]]:
    """Per-regime initial corner positions (the estimators' starting guess)."""
    inits = {}
    for regime, sigma in _NOISE_PX.items():
        inits[regime] = (gt_corners + rng.normal(0.0, sigma, size=(2, 4))).tolist()
    inits["lost"] = (other_pose if other_pose is not None else template_corners).tolist()
    inits["cold"] = template_corners.tolist()
    return inits


def _build_sequence(args: Tuple) -> List[Dict]:
    """Build the cases of one sequence (worker body of ``build``)."""
    dataset, seq_name, frames_per_seq, seed = args
    ds_info = DATASETS[dataset]
    base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pot":
        gt_corners, valid = pot_load_gt(base_dir, seq_name)
    else:
        gt_corners, valid = pti_load_gt(base_dir, seq_name, paths.PT_INITFIX_DIR)

    available = _available_frames(_mask_dir(dataset, seq_name))
    if 0 not in available:
        return []
    template_mask = _load_mask(dataset, seq_name, 0)
    if template_mask is None or template_mask.sum() < 50:
        return []

    candidates = [
        int(f) for f in np.flatnonzero(valid)
        if f > 0 and int(f) in available and np.isfinite(gt_corners[f]).all()
    ]
    if not candidates:
        return []

    rng = np.random.default_rng(seed + int(hashlib.md5(seq_name.encode()).hexdigest()[:8], 16))
    if frames_per_seq > 0 and len(candidates) > frames_per_seq:
        chosen = sorted(rng.choice(candidates, size=frames_per_seq, replace=False).tolist())
    else:
        chosen = candidates

    template_corners = gt_corners[0]
    split = _split_of(seq_name)
    cases = []
    for frame_idx in chosen:
        mask = _load_mask(dataset, seq_name, frame_idx)
        if mask is None or mask.sum() < 50:
            continue
        others = [f for f in candidates if abs(f - frame_idx) > 30]
        other_pose = gt_corners[int(rng.choice(others))] if others else None
        case = {
            "dataset": dataset,
            "seq": seq_name,
            "frame_idx": frame_idx,
            "split": split,
            "shape": list(mask.shape),
            "template_corners": template_corners.tolist(),
            "gt_corners": gt_corners[frame_idx].tolist(),
            "inits": _init_corners_for_regimes(
                gt_corners[frame_idx], template_corners, other_pose, rng
            ),
        }
        case.update(_mask_stats(mask, gt_corners[frame_idx]))
        case.update(_pose_stats(template_corners, gt_corners[frame_idx], mask.shape))
        cases.append(case)
    return cases


def build(argv: argparse.Namespace) -> int:
    """Assemble the test-set manifest and write it as JSON."""
    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    all_cases: List[Dict] = []
    for dataset in argv.datasets:
        ds_info = DATASETS[dataset]
        base_dir = getattr(paths, ds_info.base_dir_attr)
        sequences = get_sequences(dataset, base_dir, split=argv.split if dataset != "pot" else None)
        if argv.max_seqs:
            sequences = sequences[: argv.max_seqs]
        per_seq = 0 if dataset == "pot" else argv.frames_per_seq
        jobs = [(dataset, s, per_seq, argv.seed) for s in sequences]
        with ProcessPoolExecutor(max_workers=argv.workers) as pool:
            for cases in pool.map(_build_sequence, jobs, chunksize=1):
                all_cases.extend(cases)
        n_ds = sum(1 for c in all_cases if c["dataset"] == dataset)
        print(f"{ds_info.label}: {len(sequences)} sequences -> {n_ds} cases")

    out = _BENCH_DIR / argv.name
    out.write_text(json.dumps({"cases": all_cases}))
    n_dev = sum(1 for c in all_cases if c["split"] == "dev")
    print(f"\nWrote {len(all_cases)} cases ({n_dev} dev / {len(all_cases) - n_dev} test) -> {out}")
    return 0


# ---------------------------------------------------------------------------
# Estimators under test
# ---------------------------------------------------------------------------


def _method_hough(mask, template_mask, template_corners, init_corners, init_H, ctx):
    """Baseline: the deployed Hough lines -> quad pipeline (quadrilateral prior)."""
    from flatsam2.hough_homography import mask_to_homography

    H, n_corr = mask_to_homography(mask.astype(np.uint8) * 255, template_corners, init_H)
    return (H if n_corr > 0 else None), {"n": float(n_corr)}


def _method_align(mask, template_mask, template_corners, init_corners, init_H, ctx):
    """Baseline: damped-GN boundary alignment from the given init (quad-free)."""
    from flatsam2.mask_align import align_mask_to_homography

    H, n_ok, residual, _, _ = align_mask_to_homography(
        mask, template_corners, init_H, template_mask=template_mask
    )
    return H, {"n": float(n_ok), "residual": residual}


def _method_align_redetect(mask, template_mask, template_corners, init_corners, init_H, ctx):
    """Baseline: boundary alignment plus mask-moment redetection on a bad solve."""
    from flatsam2.mask_align import align_mask_to_homography

    H, n_ok, residual, _, _ = align_mask_to_homography(
        mask, template_corners, init_H, template_mask=template_mask,
        redetect_above_residual_px=6.0,
    )
    return H, {"n": float(n_ok), "residual": residual}


METHODS: Dict[str, Callable] = {
    "hough": _method_hough,
    "align": _method_align,
    "align_redetect": _method_align_redetect,
}


def _keypoint_homographies(dataset: str, seq_name: str, run: str) -> Dict[int, np.ndarray]:
    """Per-frame keypoint-branch homography from an existing run's telemetry.

    The mask says where the object is but not which corner is which, so the
    keypoint branch is the natural second opinion. Rather than re-running LoMa,
    read the ``loma_H`` a completed run already stored per frame — it is exactly
    what the tracker would hand the registrar online.

    Args:
        dataset: Dataset key.
        seq_name: Sequence name.
        run: Config stem of the run holding ``{seq}_debug.npz``.

    Returns:
        ``{frame_idx: (3, 3) homography}``; empty when the run has no telemetry
        for this sequence.
    """
    npz = paths.OUTPUT_DIRECTORY / run / DATASETS[dataset].label / f"{seq_name}_debug.npz"
    if not npz.exists():
        return {}
    with np.load(npz, allow_pickle=False) as data:
        if "loma_H" not in data or "frame_idx" not in data:
            return {}
        out = {}
        for t, fidx in enumerate(data["frame_idx"]):
            H = data["loma_H"][t]
            if np.all(np.isfinite(H)):
                out[int(fidx)] = H
    return out


def _image_paths(dataset: str, seq_name: str) -> Optional[List[Path]]:
    """Frame paths of one sequence, or None when they cannot be listed."""
    from flatsam2.datasets import pot_image_paths, pt_image_paths

    base_dir = getattr(paths, DATASETS[dataset].base_dir_attr)
    try:
        return (
            pot_image_paths(base_dir, seq_name)
            if DATASETS[dataset].gt_loader == "pot"
            else pt_image_paths(base_dir, seq_name)
        )
    except ValueError:
        return None


def _needs_frames(method_names: List[str]) -> bool:
    """Whether any requested method uses image appearance (frames cost IO).

    Decided from each method's own configuration (``needs_frames``, set by
    :func:`flatsam2.mask_register.bench_methods` from ``appearance_power``), not
    from its name. A name-based rule silently turned every ``abl_*`` method into
    its shape-only twin whenever no method called ``register*`` happened to be
    in the same invocation — which reads as "the appearance term does nothing"
    rather than as a missing input.
    """
    return any(
        getattr(METHODS.get(n), "needs_frames", n.startswith("register"))
        for n in method_names
    )


def _register_extra_methods() -> None:
    """Add the developing quad-free registration methods when the module exists."""
    try:
        from flatsam2 import mask_register
    except ImportError:
        return
    METHODS.update(mask_register.bench_methods())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _run_sequence(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """Evaluate every method x regime on all cases of one sequence.

    Returns:
        ``(rows, hyp_rows)`` — one row per (case, regime, method), plus one row
        per individual hypothesis for methods that report their search trace.
    """
    seq_key, cases, method_names, regimes, extra_h_run = args
    _register_extra_methods()
    dataset, seq_name = seq_key
    template_mask = _load_mask(dataset, seq_name, 0)
    if template_mask is None:
        return [], []
    template_mask_u8 = template_mask.astype(np.uint8) * 255
    image_paths = _image_paths(dataset, seq_name) if _needs_frames(method_names) else None
    keypoint_H = (
        _keypoint_homographies(dataset, seq_name, extra_h_run) if extra_h_run else {}
    )
    init_frame = (
        cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR) if image_paths else None
    )

    rows: List[Dict] = []
    hyp_rows: List[Dict] = []
    for case in cases:
        mask = _load_mask(dataset, seq_name, case["frame_idx"])
        if mask is None:
            continue
        frame = None
        if image_paths is not None and case["frame_idx"] < len(image_paths):
            frame = cv2.imread(str(image_paths[case["frame_idx"]]), cv2.IMREAD_COLOR)
        ctx = {
            "frame": frame,
            "init_frame": init_frame,
            "extra_H": keypoint_H.get(case["frame_idx"]),
        }
        template_corners = np.array(case["template_corners"], dtype=np.float64)
        gt_corners = np.array(case["gt_corners"], dtype=np.float64)
        for regime in regimes:
            init_corners = np.array(case["inits"][regime], dtype=np.float64)
            try:
                init_H = cv2.getPerspectiveTransform(
                    template_corners.T.astype(np.float32), init_corners.T.astype(np.float32)
                )
            except cv2.error:
                continue
            for name in method_names:
                t0 = time.perf_counter()
                try:
                    H, extra = METHODS[name](
                        mask, template_mask_u8, template_corners, init_corners, init_H, ctx
                    )
                except Exception as exc:  # a broken estimator must not kill the sweep
                    H, extra = None, {"error": repr(exc)[:80]}
                ms = 1000.0 * (time.perf_counter() - t0)
                if H is None or not np.isfinite(H).all():
                    err = float("inf")
                else:
                    err = float(alignment_error(gt_corners, warp_corners(H, template_corners)))
                for k, hyp in enumerate(extra.pop("hypotheses", [])):
                    hyp_rows.append({
                        "dataset": dataset, "seq": seq_name, "frame_idx": case["frame_idx"],
                        "split": case["split"], "method": name, "regime": regime, "k": k,
                        "source": hyp["source"], "score": hyp["score"],
                        "precision": hyp.get("precision", float("nan")),
                        "recall": hyp.get("recall", float("nan")),
                        "iou": hyp.get("iou", float("nan")),
                        "area_ratio": hyp.get("area_ratio", float("nan")),
                        "residual_px": hyp.get("residual_px", float("nan")),
                        "ncc": hyp.get("ncc", float("nan")),
                        "err": float(alignment_error(gt_corners, hyp["control"])),
                        # Error under the best cyclic corner relabeling: when it
                        # is small while `err` is large, the pose is right and
                        # only the corner labels are permuted — the silhouette
                        # alias, which no mask-shape score can resolve.
                        "err_rot": min(
                            float(alignment_error(gt_corners, np.roll(hyp["control"], k, axis=1)))
                            for k in range(4)
                        ),
                    })
                rows.append({
                    "dataset": dataset, "seq": seq_name, "frame_idx": case["frame_idx"],
                    "split": case["split"], "method": name, "regime": regime,
                    "err": err, "ms": ms, "iou": case["iou"], "covered": case["covered"],
                    "precision": case["precision"], "n_components": case["n_components"],
                    "gt_scale": case["gt_scale"], "gt_rot_deg": case["gt_rot_deg"],
                    "oov_frac": case["oov_frac"],
                    "n": extra.get("n", float("nan")),
                    "residual": extra.get("residual", float("nan")),
                    "score": extra.get("score", float("nan")),
                    "src": extra.get("source", ""),
                })
    return rows, hyp_rows


def _write_csv(path: Path, rows: List[Dict]) -> None:
    """Write a list of flat dicts as CSV (all rows share the first row's keys)."""
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")


def _summarize_hypotheses(hyp_rows: List[Dict]) -> None:
    """Oracle vs deployed selection over the search trace.

    The gap between ``oracle`` (best hypothesis the search generated) and
    ``picked`` (the one the verifier's score chose) says which half of the
    pipeline to work on: proposals or verification.
    """
    import collections

    by_case = collections.defaultdict(list)
    for r in hyp_rows:
        by_case[(r["dataset"], r["split"], r["regime"], r["seq"], r["frame_idx"])].append(r)

    stats = collections.defaultdict(lambda: {"oracle": [], "picked": [], "n": []})
    for (dataset, split, regime, _, _), rs in by_case.items():
        errs = np.array([r["err"] for r in rs])
        scores = np.array([r["score"] for r in rs])
        key = (dataset, split, regime)
        stats[key]["oracle"].append(float(errs.min()))
        stats[key]["picked"].append(float(errs[int(np.argmax(scores))]))
        stats[key]["n"].append(len(rs))

    print("\n=== hypothesis search: oracle vs verifier " + "=" * 25)
    print(f"  {'dataset':<6} {'split':<5} {'regime':<6} {'orc@5':>6} {'orc@15':>7} "
          f"{'pick@5':>7} {'pick@15':>8} {'n_hyp':>6}  cases")
    for key in sorted(stats):
        s = stats[key]
        orc, pick = np.array(s["oracle"]), np.array(s["picked"])
        print(f"  {key[0]:<6} {key[1]:<5} {key[2]:<6} "
              f"{100 * np.mean(orc <= 5):6.1f} {100 * np.mean(orc <= 15):7.1f} "
              f"{100 * np.mean(pick <= 5):7.1f} {100 * np.mean(pick <= 15):8.1f} "
              f"{np.mean(s['n']):6.1f}  {len(orc)}")


def _summarize(rows: List[Dict], methods: List[str], regimes: List[str]) -> None:
    """Print P@k / median error / runtime per dataset x split x regime x method."""
    import collections

    keyed = collections.defaultdict(list)
    for r in rows:
        keyed[(r["dataset"], r["split"], r["regime"], r["method"])].append(r)

    for dataset in sorted({r["dataset"] for r in rows}):
        for split in ("dev", "test"):
            print(f"\n=== {dataset}  split={split} " + "=" * 40)
            print(f"  {'regime':<7} {'method':<20} {'P@5':>6} {'P@15':>6} "
                  f"{'med_err':>8} {'p90_err':>8} {'ms':>7}  n")
            for regime in regimes:
                for method in methods:
                    rs = keyed.get((dataset, split, regime, method), [])
                    if not rs:
                        continue
                    err = np.array([r["err"] for r in rs])
                    ms = np.array([r["ms"] for r in rs])
                    finite = err[np.isfinite(err)]
                    med = float(np.median(err)) if np.isfinite(err).all() else (
                        float(np.median(finite)) if finite.size else float("inf"))
                    p90 = float(np.percentile(err, 90)) if np.isfinite(err).all() else float("inf")
                    print(f"  {regime:<7} {method:<20} "
                          f"{100 * np.mean(err <= 5):6.1f} {100 * np.mean(err <= 15):6.1f} "
                          f"{med:8.1f} {p90:8.1f} {np.mean(ms):7.1f}  {len(rs)}")


def run(argv: argparse.Namespace) -> int:
    """Evaluate the requested methods on the test set and write per-case rows."""
    _register_extra_methods()
    manifest = json.loads((_BENCH_DIR / argv.name).read_text())
    cases = manifest["cases"]
    if argv.datasets:
        cases = [c for c in cases if c["dataset"] in argv.datasets]
    if argv.split:
        cases = [c for c in cases if c["split"] == argv.split]
    if argv.limit:
        cases = cases[: argv.limit]

    methods = argv.methods or list(METHODS)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        print(f"Unknown methods: {unknown}. Available: {list(METHODS)}")
        return 1
    regimes = argv.regimes or list(_REGIMES)

    # A keypoint method without --extra-h-run receives no extra proposal and is
    # silently identical to its non-keypoint twin — which reads as a null result
    # rather than as a missing input. Refuse instead.
    if not argv.extra_h_run:
        needs_extra = [m for m in methods if getattr(METHODS[m], "needs_extra_h", False)]
        if needs_extra:
            print(
                f"ERROR: {needs_extra} need keypoint homographies, but --extra-h-run was "
                f"not given. Without it they run identically to their non-keypoint twin. "
                f"Pass e.g. --extra-h-run corner_fusion_register."
            )
            return 1

    by_seq: Dict[Tuple[str, str], List[Dict]] = {}
    for c in cases:
        by_seq.setdefault((c["dataset"], c["seq"]), []).append(c)
    jobs = [(k, v, methods, regimes, argv.extra_h_run) for k, v in by_seq.items()]
    print(f"Evaluating {len(methods)} methods x {len(regimes)} regimes on {len(cases)} cases "
          f"({len(jobs)} sequences, {argv.workers} workers)")

    rows: List[Dict] = []
    hyp_rows: List[Dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=argv.workers) as pool:
        for i, (out, hyp_out) in enumerate(pool.map(_run_sequence, jobs, chunksize=1)):
            rows.extend(out)
            hyp_rows.extend(hyp_out)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} sequences  ({time.perf_counter() - t0:.0f}s)",
                      flush=True)

    _write_csv(_BENCH_DIR / argv.out, rows)
    print(f"\nWrote {len(rows)} rows -> {_BENCH_DIR / argv.out}  "
          f"({time.perf_counter() - t0:.0f}s)")
    if hyp_rows:
        hyp_csv = _BENCH_DIR / f"hyp_{argv.out}"
        _write_csv(hyp_csv, hyp_rows)
        print(f"Wrote {len(hyp_rows)} hypothesis rows -> {hyp_csv}")
        _summarize_hypotheses(hyp_rows)
    _summarize(rows, methods, regimes)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="assemble the test-set manifest")
    p_build.add_argument("--datasets", nargs="+", default=["pti", "pot"])
    p_build.add_argument("--frames-per-seq", type=int, default=6)
    p_build.add_argument("--split", default="test", help="dataset split for PlanarTrack")
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--workers", type=int, default=16)
    p_build.add_argument("--max-seqs", type=int, default=0, help="smoke-test cap per dataset")
    p_build.add_argument("--name", default="testset_v1.json")

    p_run = sub.add_parser("run", help="evaluate estimators on the test set")
    p_run.add_argument("--methods", nargs="+", default=None)
    p_run.add_argument("--regimes", nargs="+", default=None, choices=list(_REGIMES))
    p_run.add_argument("--datasets", nargs="+", default=None)
    p_run.add_argument("--split", default=None, choices=["dev", "test"])
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--workers", type=int, default=16)
    p_run.add_argument("--name", default="testset_v1.json")
    p_run.add_argument("--out", default="results.csv")
    p_run.add_argument(
        "--extra-h-run",
        default=None,
        help=(
            "Config stem of a completed run whose per-frame loma_H is fed to "
            "register_kp as an extra pose proposal (e.g. corner_fusion_register)"
        ),
    )

    args = parser.parse_args(argv)
    return build(args) if args.cmd == "build" else run(args)


if __name__ == "__main__":
    sys.exit(main())
