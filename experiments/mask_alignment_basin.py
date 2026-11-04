#!/usr/bin/env python3
"""P3' feasibility: convergence basins of direct mask-to-mask homography alignment.

Could a direct aligner (template mask -> current SAM mask) replace the Hough
quad extraction in SAM-H? Its viability hinges on the convergence basin: the
online tracker would initialize from the last-frame (or LoMa) homography, so the
aligner only ever needs to recover the *remaining* perturbation. This study
measures that basin offline on the precomputed SAM3 masks, with **structured,
attributable perturbation families** — iid corner noise alone neither spans the
realistic failure modes nor tells us *which* challenge breaks the aligner:

    translation  all corners shifted by m px (random direction)
    rotation     corners rotated about the quad centroid by m degrees
                 (wide grid up to 180 deg -> probes capture by wrong symmetry poses)
    scale        corners scaled about the centroid by (1 + m/100)
    shear        x-only scaling about the centroid by (1 + m/100)
    tilt         trapezoid: one edge pulled in, the opposite pushed out by m/100
                 (perspective-like foreshortening)
    noise        iid Gaussian corner noise, sigma = m px
    mix          translation + rotation + scale + noise composed at scale m

Aligner: damped Gauss-Newton on the **4-point parameterization** (the 4 image
positions of the template corners; never raw 3x3 coefficients), minimizing a
Huber loss on the target-mask signed distance sampled along the warped template
boundary. Numeric Jacobian (8 params) — simple and adequate for a feasibility
study.

Outcome per trial (tau = --tau px, default 5):
    correct     final corner error <= tau
    symmetric   best-rotation corner error <= tau but plain error > tau
                (converged onto a wrong cyclic pose — the symmetry trap)
    diverged    neither

``--target gt`` aligns against clean GT-quad rasterizations instead of SAM masks
(upper bound of the aligner itself; the gap to ``--target sam`` isolates the
cost of SAM boundary noise/spill). Doubles as a training-data generator recipe
for a learned two-mask aligner if classical basins prove too small.

Subcommands:
    dump    Run perturbation trials -> per-sequence CSV.
    report  CSVs -> basin-rate tables (stdout) + standalone HTML curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from einops import rearrange

from flatsam2.corner_covariance import corners_to_theta, theta_to_corners
from flatsam2.datasets import (
    corners_to_mask,
    get_sequences,
    pt_image_paths,
    pti_load_gt,
)
from flatsam2.mask_metrics import signed_distance
from flatsam2.metrics import alignment_error, best_rotation_error
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.report_utils import get_plotly_script_tag


# Perturbation family -> magnitude grid (units in the family's own scale, see
# module docstring). Wide rotation range deliberately reaches the symmetry poses.
FAMILIES: dict[str, tuple[float, ...]] = {
    "translation": (2, 5, 10, 20, 40, 80),
    "rotation": (5, 10, 20, 45, 90, 135, 180),
    "scale": (2, 5, 10, 20, 40),
    "shear": (2, 5, 10, 20, 40),
    "tilt": (2, 5, 10, 20, 40),
    "noise": (1, 2, 5, 10, 20),
    "mix": (0.5, 1.0, 2.0, 4.0),
}

CSV_HEADER = [
    "frame_idx",
    "family",
    "magnitude",
    "repeat",
    "iou_gt",       # SAM-vs-GT-quad IoU of the target mask (mask quality bucket)
    "init_err",     # corner error of the perturbed start vs GT
    "final_err",    # plain corner error after alignment
    "final_br_err", # best-cyclic-rotation corner error after alignment
    "outcome",      # correct | symmetric | diverged
    "n_iters",
]


# --------------------------------------------------------------------------- #
# perturbations (all operate on corner arrays (2, 4) in image coordinates)
# --------------------------------------------------------------------------- #


def _perturb(
    corners: np.ndarray,
    family: str,
    magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one structured perturbation to GT corners.

    Args:
        corners: GT corners (2, 4) at the current frame.
        family: Perturbation family name (see FAMILIES).
        magnitude: Family-specific magnitude.
        rng: Seeded generator (direction/sign randomness).

    Returns:
        Perturbed corners (2, 4).
    """
    c = corners.astype(np.float64).copy()
    centroid = c.mean(axis=1, keepdims=True)  # (2, 1)
    if family == "translation":
        angle = rng.uniform(0, 2 * np.pi)
        return c + magnitude * np.array([[np.cos(angle)], [np.sin(angle)]])
    if family == "rotation":
        theta = np.deg2rad(magnitude) * rng.choice([-1.0, 1.0])
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return centroid + R @ (c - centroid)
    if family == "scale":
        s = 1.0 + (magnitude / 100.0) * rng.choice([-1.0, 1.0])
        return centroid + s * (c - centroid)
    if family == "shear":
        s = 1.0 + (magnitude / 100.0) * rng.choice([-1.0, 1.0])
        out = c - centroid
        out[0] *= s
        return centroid + out
    if family == "tilt":
        # Trapezoid: edge (0,1) pulled toward the centroid, edge (2,3) pushed out.
        f = magnitude / 100.0
        out = c - centroid
        out[:, [0, 1]] *= 1.0 - f
        out[:, [2, 3]] *= 1.0 + f
        return centroid + out
    if family == "noise":
        return c + rng.normal(0.0, magnitude, size=c.shape)
    if family == "mix":
        # Compose small instances of the base families at a common scale.
        out = _perturb(c, "translation", 8.0 * magnitude, rng)
        out = _perturb(out, "rotation", 5.0 * magnitude, rng)
        out = _perturb(out, "scale", 4.0 * magnitude, rng)
        return _perturb(out, "noise", 1.5 * magnitude, rng)
    raise ValueError(f"unknown perturbation family {family!r}")


# --------------------------------------------------------------------------- #
# aligner: damped GN on the 4-point parameterization
# --------------------------------------------------------------------------- #


def _template_boundary(template_mask: np.ndarray, n_points: int = 150) -> np.ndarray:
    """Subsampled outer-contour points of the template mask.

    Args:
        template_mask: Binary template mask (H, W) in init-frame coordinates.
        n_points: Approximate number of boundary samples.

    Returns:
        Boundary points (N, 2) float64 as (x, y).
    """
    contours, _ = cv2.findContours(
        template_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("template mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    pts = rearrange(contour, "n 1 xy -> n xy").astype(np.float64)
    step = max(1, len(pts) // n_points)
    return pts[::step]


def _sample_bilinear(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of a scalar field at (x, y) points, edge-clamped.

    Args:
        field: Scalar field (H, W) float.
        pts: Sample points (N, 2) as (x, y).

    Returns:
        Sampled values (N,).
    """
    h, w = field.shape
    x = np.clip(pts[:, 0], 0.0, w - 1.001)
    y = np.clip(pts[:, 1], 0.0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    v00 = field[y0, x0]
    v01 = field[y0, x0 + 1]
    v10 = field[y0 + 1, x0]
    v11 = field[y0 + 1, x0 + 1]
    return (
        v00 * (1 - fx) * (1 - fy)
        + v01 * fx * (1 - fy)
        + v10 * (1 - fx) * fy
        + v11 * fx * fy
    )


def _warp_boundary(
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control: np.ndarray,
) -> Optional[np.ndarray]:
    """Warp template-boundary points by the 4-point homography init -> control.

    Args:
        boundary: Template boundary points (N, 2).
        init_corners: Init-frame corner positions (2, 4).
        control: Current control-point positions (2, 4).

    Returns:
        Warped points (N, 2), or None when the 4-point fit is degenerate.
    """
    src = init_corners.T.astype(np.float32)
    dst = control.T.astype(np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    pts = np.concatenate([boundary, np.ones((len(boundary), 1))], axis=1)
    warped = pts @ H.T
    z = warped[:, 2:3]
    if not np.all(np.abs(z) > 1e-9):
        return None
    return warped[:, :2] / z


def _residuals(
    sdf: np.ndarray,
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control: np.ndarray,
) -> Optional[np.ndarray]:
    """Signed-distance residuals of the warped template boundary; None if degenerate."""
    warped = _warp_boundary(boundary, init_corners, control)
    if warped is None:
        return None
    return _sample_bilinear(sdf, warped)


def align_masks(
    sdf: np.ndarray,
    boundary: np.ndarray,
    init_corners: np.ndarray,
    control0: np.ndarray,
    max_iters: int = 60,
    huber_delta: float = 5.0,
    step_tol: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Damped Gauss-Newton mask alignment on the 4-point parameterization.

    Minimizes ``sum huber(sdf_target(warp(boundary, H(control))))`` over the 8
    control-point coordinates. Numeric forward-difference Jacobian.

    Args:
        sdf: Signed distance field of the target mask (H, W), positive outside.
        boundary: Template boundary points (N, 2) in init-frame coordinates.
        init_corners: Init-frame corners (2, 4) defining the parameterization.
        control0: Initial control points (2, 4) (the perturbed GT corners).
        max_iters: Iteration cap.
        huber_delta: Huber transition (px).
        step_tol: Stop when the max control-point step falls below this (px).

    Returns:
        Tuple ``(control, n_iters)`` — final control points (2, 4) and the
        number of iterations run.
    """
    control = control0.astype(np.float64).copy()
    lam = 1.0
    r = _residuals(sdf, boundary, init_corners, control)
    if r is None:
        return control, 0
    for it in range(1, max_iters + 1):
        # Huber IRLS weights.
        absr = np.abs(r)
        w = np.where(absr <= huber_delta, 1.0, huber_delta / np.maximum(absr, 1e-9))

        # Numeric Jacobian over the 8 control coordinates.
        J = np.zeros((len(r), 8))
        flat = corners_to_theta(control)
        delta = 0.5
        degenerate = False
        for p in range(8):
            trial = flat.copy()
            trial[p] += delta
            r_p = _residuals(sdf, boundary, init_corners, theta_to_corners(trial))
            if r_p is None:
                degenerate = True
                break
            J[:, p] = (r_p - r) / delta
        if degenerate:
            break

        JW = J * w[:, None]
        A = J.T @ JW + lam * np.eye(8)
        g = J.T @ (w * r)
        try:
            step = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            break
        trial = theta_to_corners(flat + step)
        r_trial = _residuals(sdf, boundary, init_corners, trial)
        cost = float((w * r * r).sum())
        if r_trial is not None and float((w * r_trial * r_trial).sum()) < cost:
            control, r = trial, r_trial
            lam = max(lam * 0.5, 1e-3)
            if np.abs(step).max() < step_tol:
                return control, it
        else:
            lam = min(lam * 4.0, 1e6)
            if lam >= 1e6:
                return control, it
    return control, max_iters


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #


def _load_mask(path: Path) -> Optional[np.ndarray]:
    """Load a binary mask PNG as bool (H, W), or None if missing/unreadable."""
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else (raw > 0)


def _classify(final_err: float, final_br_err: float, tau: float) -> str:
    if final_err <= tau:
        return "correct"
    if final_br_err <= tau:
        return "symmetric"
    return "diverged"


def _dump_sequence(
    seq: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_csv: Path,
) -> int:
    """Run all perturbation trials for one sequence; returns the trial count."""
    gt_corners, valid = pti_load_gt(args.base_dir, seq, args.initfix_dir)
    image_paths = pt_image_paths(args.base_dir, seq)
    T = min(len(gt_corners), len(valid), len(image_paths))
    init_corners = gt_corners[0]

    # Template: frame-0 SAM mask when available (the shape the tracker actually
    # holds), else the GT quad.
    template = _load_mask(args.masks_dir / seq / "00001.png")

    rows: list[list] = []
    n_trials = 0
    for i in range(1, T, args.frame_stride):
        if not bool(valid[i]):
            continue
        sam_mask = _load_mask(args.masks_dir / seq / f"{i + 1:05d}.png")
        if sam_mask is None:
            continue
        h_img, w_img = sam_mask.shape
        template_i = (
            template
            if template is not None
            else corners_to_mask(init_corners, (h_img, w_img)) > 0
        )
        try:
            boundary = _template_boundary(template_i)
        except ValueError:
            continue
        gt_mask = corners_to_mask(gt_corners[i], (h_img, w_img)) > 0
        union = int((sam_mask | gt_mask).sum())
        iou_gt = float(int((sam_mask & gt_mask).sum()) / union) if union else float("nan")

        target = gt_mask if args.target == "gt" else sam_mask
        if not target.any():
            continue
        sdf = signed_distance(target)

        gt_c = gt_corners[i]
        for family, magnitudes in FAMILIES.items():
            if args.families and family not in args.families:
                continue
            for magnitude in magnitudes:
                for rep in range(args.repeats):
                    control0 = _perturb(gt_c, family, magnitude, rng)
                    init_err = float(alignment_error(gt_c, control0))
                    control, n_iters = align_masks(sdf, boundary, init_corners, control0)
                    final_err = float(alignment_error(gt_c, control))
                    # (2, 4) inputs -> best_rotation_error returns scalars.
                    br_err, _ = best_rotation_error(gt_c, control)
                    final_br = float(br_err)
                    rows.append(
                        [
                            i,
                            family,
                            magnitude,
                            rep,
                            iou_gt,
                            init_err,
                            final_err,
                            final_br,
                            _classify(final_err, final_br, args.tau),
                            n_iters,
                        ]
                    )
                    n_trials += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([f"{x:.6g}" if isinstance(x, float) else x for x in r])
    tmp.replace(out_csv)
    return n_trials


def cmd_dump(args: argparse.Namespace) -> None:
    """Run the basin trials over a sequence list."""
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
    print(f"Sequences: {len(sequences)}  target={args.target}  stride={args.frame_stride}")

    total = 0
    failed: list[str] = []
    for k, seq in enumerate(sequences, 1):
        out_csv = out_dir / f"{seq}.csv"
        if out_csv.exists() and args.skip_completed:
            print(f"[{k}/{len(sequences)}] skip {seq} (exists)")
            continue
        rng = np.random.default_rng(args.seed + k)
        try:
            n = _dump_sequence(seq, args, rng, out_csv)
            total += n
            print(f"[{k}/{len(sequences)}] {seq}: {n} trials")
        except Exception as e:  # noqa: BLE001 - keep going, record failure
            failed.append(seq)
            print(f"[{k}/{len(sequences)}] {seq}: FAILED ({e})")
    print(f"\nDone. trials={total} failed={len(failed)}")
    if failed:
        print("failed:", " ".join(failed))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def cmd_report(args: argparse.Namespace) -> None:
    """CSVs -> per-family basin-rate tables + HTML curves."""
    rows: list[dict] = []
    for path in sorted(args.per_seq_dir.glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            for d in csv.DictReader(f):
                d["seq"] = path.stem
                rows.append(d)
    if not rows:
        raise FileNotFoundError(f"no per-sequence CSVs in {args.per_seq_dir}")

    def _bucket(row: dict) -> str:
        iou = float(row["iou_gt"])
        return "clean" if np.isfinite(iou) and iou >= args.clean_iou else "degraded"

    def _outcome(row: dict) -> str:
        # --tau re-thresholds from the stored per-trial errors, so the decision
        # metric can be applied without re-running the (expensive) dump.
        if args.tau is None:
            return row["outcome"]
        fe, fbr = float(row["final_err"]), float(row["final_br_err"])
        if np.isfinite(fe) and fe <= args.tau:
            return "correct"
        if np.isfinite(fbr) and fbr <= args.tau:
            return "symmetric"
        return "diverged"

    if args.tau is not None:
        print(f"(outcomes re-thresholded at tau = {args.tau} px)")
    lines: list[str] = []
    curves: dict[str, dict] = {}
    for family, magnitudes in FAMILIES.items():
        fam_rows = [r for r in rows if r["family"] == family]
        if not fam_rows:
            continue
        lines.append(f"\n## {family}\n")
        lines.append(
            "| magnitude | bucket | trials | correct % | symmetric % | diverged % "
            "| med final err |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        curves[family] = {"x": [], "correct": [], "symmetric": []}
        for magnitude in magnitudes:
            m_rows = [r for r in fam_rows if float(r["magnitude"]) == magnitude]
            for bucket in ("clean", "degraded", "all"):
                b_rows = (
                    m_rows if bucket == "all" else [r for r in m_rows if _bucket(r) == bucket]
                )
                n = len(b_rows)
                if n == 0:
                    continue
                outs = [_outcome(r) for r in b_rows]
                pc = 100.0 * sum(o == "correct" for o in outs) / n
                ps = 100.0 * sum(o == "symmetric" for o in outs) / n
                pd = 100.0 - pc - ps
                errs = [
                    float(r["final_err"])
                    for r in b_rows
                    if np.isfinite(float(r["final_err"]))
                ]
                med = f"{np.median(errs):.1f}" if errs else "-"
                lines.append(
                    f"| {magnitude} | {bucket} | {n} | {pc:.1f} | {ps:.1f} | {pd:.1f} "
                    f"| {med} |"
                )
                if bucket == "all":
                    curves[family]["x"].append(magnitude)
                    curves[family]["correct"].append(pc)
                    curves[family]["symmetric"].append(ps)

    report_md = "\n".join(lines)
    print(report_md)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(curves)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>mask alignment basins</title>
{get_plotly_script_tag()}
<style>body{{font-family:sans-serif;margin:2em}}div.plot{{width:900px;height:500px}}</style>
</head><body>
<h1>Mask-alignment convergence basins (P3')</h1>
<p>substrate: <code>{args.per_seq_dir}</code></p>
<h2>Convergence-to-correct rate per family</h2>
<div class="plot" id="correct"></div>
<h2>Symmetric-capture rate per family (converged to a wrong cyclic pose)</h2>
<div class="plot" id="symmetric"></div>
<pre>{report_md}</pre>
<script>
var d = {payload};
function traces(key) {{
  return Object.keys(d).map(function (fam) {{
    return {{x: d[fam].x, y: d[fam][key], name: fam, mode: 'lines+markers'}};
  }});
}}
Plotly.newPlot('correct', traces('correct'),
  {{xaxis: {{title: 'magnitude (family units)', type: 'log'}},
    yaxis: {{title: 'correct %', range: [0, 100]}}}});
Plotly.newPlot('symmetric', traces('symmetric'),
  {{xaxis: {{title: 'magnitude (family units)', type: 'log'}},
    yaxis: {{title: 'symmetric %', range: [0, 100]}}}});
</script>
</body></html>"""
    out_html = args.output_dir / "report.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\nReport: {out_html}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the dump/report CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    out_default = (
        paths.OUTPUT_DIRECTORY / "experiments" / "mask_alignment_basin"
        / DATASETS["pti"].label
    )

    d = sub.add_parser("dump", help="perturbation trials -> per-sequence CSV")
    d.add_argument("--masks-dir", type=Path, default=paths.SAM3_PRECOMPUTED_MASKS_PTI)
    d.add_argument("--base-dir", type=Path, default=paths.PT_BASE)
    d.add_argument("--initfix-dir", type=Path, default=paths.PT_INITFIX_DIR)
    d.add_argument("--split", choices=["train", "test"], default="test")
    d.add_argument("--sequences", nargs="+")
    d.add_argument("--subset")
    d.add_argument("--subset-file", type=Path)
    d.add_argument("--target", choices=["sam", "gt"], default="sam",
                   help="align against SAM masks (realistic) or GT quads (aligner "
                        "upper bound)")
    d.add_argument("--families", nargs="+", choices=list(FAMILIES),
                   help="restrict to these perturbation families")
    d.add_argument("--frame-stride", type=int, default=60,
                   help="trial every Nth valid frame")
    d.add_argument("--repeats", type=int, default=1,
                   help="random repeats per (frame, family, magnitude)")
    d.add_argument("--tau", type=float, default=5.0,
                   help="corner-error convergence threshold (px)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--output-dir", type=Path, default=out_default)
    d.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    d.set_defaults(func=cmd_dump)

    r = sub.add_parser("report", help="per-sequence CSVs -> basin tables + HTML")
    r.add_argument("--per-seq-dir", type=Path, default=out_default / "per_sequence")
    r.add_argument("--output-dir", type=Path, default=out_default)
    r.add_argument("--clean-iou", type=float, default=0.7,
                   help="SAM-vs-GT IoU above this = 'clean' mask bucket")
    r.add_argument("--tau", type=float, default=None,
                   help="re-threshold outcomes from the stored final errors "
                        "(px; default: keep dump-time outcomes). Use 15 for "
                        "the decision metric.")
    r.set_defaults(func=cmd_report)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
