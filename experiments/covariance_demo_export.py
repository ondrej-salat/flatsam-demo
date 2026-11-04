"""Export real per-frame covariance inputs for the interactive corner-covariance demo.

Both branches' *inputs* (not just their answers) for a handful of frames, as JSON
an HTML page can recompute from: the keypoint branch's RANSAC inliers and the mask
branch's template boundary plus the frame's SAM3 mask. A browser that re-runs
``sigma^2 (J^T W J)^-1`` on these arrays reproduces the numbers the tracker stored.

Substrate, all already on disk (no GPU, no tracker rerun):

- ``{LOMA_V2}/loma_r_allobj_inflate10/<label>/<seq>.npz`` — per-frame LoMa matches
- ``{RESULTS}/sam3_candidates/<label>/masks/<seq>/<frame+1:05d}.png`` — SAM3 masks
- the dataset's frames and GT corners

Two modes:

    scan    rank frames by how much the per-corner covariances differ from each
            other (the interesting cases: packed evidence, occlusion, drift)
    export  write the full payload for the chosen ``seq:frame`` pairs

Usage:
    python experiments/covariance_demo_export.py scan --sequences V09_5 V15_6
    python experiments/covariance_demo_export.py export --frames V09_5:120 V15_6:88
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from flatsam2.corner_covariance import bootstrap_corner_covs, homography_corner_covs
from flatsam2.datasets import make_gt_loader, make_image_paths, pot_load_gt, warp_corners
from flatsam2.corner_covariance import mahalanobis_sq
from flatsam2.mask_align import align_mask_to_homography, template_boundary
from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.tracking.corner_fusion import fuse_corners

LOMA_DIR = Path("/mnt/personal/serycjon/flatsam2/loma_v2/loma_r_allobj_inflate10")
MASKS_DIR = Path("/mnt/personal/serycjon/flatsam2/results/sam3_candidates")
OUT_DIR = Path("/mnt/personal/serycjon/flatsam2/results/experiments/covariance_demo")

RANSAC_METHOD = cv2.USAC_MAGSAC
RANSAC_THRESHOLD = 5.0
RANSAC_SEED = 0
N_BOUNDARY_POINTS = 150
HUBER_DELTA_PX = 5.0
JPEG_MAX_WIDTH = 900
JPEG_QUALITY = 72



def gt_loader(dataset: str):
    """GT loader for *dataset*, dense for POT.

    The default POT loader reads the WOFT coarse re-annotation, which marks only
    a handful of frames per sequence valid. The demo needs a pose on *every*
    frame — to initialize the aligner where the tracker would have been, and to
    draw a reference quad — so it reads the dataset's own dense annotation.
    """
    if dataset == "pot":
        return lambda bd, sn: pot_load_gt(bd, sn, woft_compat=False)
    return make_gt_loader(dataset)


def keypoint_branch(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    init_corners: np.ndarray,
) -> Optional[dict]:
    """RANSAC + per-corner covariance, exactly as ``tracker._estimate_loma_H`` does.

    Args:
        pts_a: (N, 2) template-frame matches.
        pts_b: (N, 2) current-frame matches, row-aligned.
        init_corners: (2, 4) template corners.

    Returns:
        Dict with the fit, its inliers and its covariance, or None on failure.
    """
    if len(pts_a) < 4:
        return None
    cv2.setRNGSeed(RANSAC_SEED)
    H, inliers = cv2.findHomography(pts_a, pts_b, RANSAC_METHOD, RANSAC_THRESHOLD)
    if H is None or inliers is None or not np.isfinite(H).all():
        return None
    inlier_mask = inliers.ravel().astype(bool)
    corners = warp_corners(H, init_corners)
    cov = homography_corner_covs(
        pts_a[inlier_mask], pts_b[inlier_mask], init_corners, corners
    )
    return {
        "H": H,
        "corners": corners,
        "cov": cov,
        "inlier_mask": inlier_mask,
        "n_inliers": int(inlier_mask.sum()),
        "pts_a": pts_a,
        "pts_b": pts_b,
    }


def mask_branch(
    mask: np.ndarray,
    template_mask: np.ndarray,
    init_corners: np.ndarray,
    last_H: np.ndarray,
) -> Optional[dict]:
    """Boundary alignment + per-corner covariance, as the SAM-H branch runs it."""
    boundary = template_boundary(template_mask > 0, N_BOUNDARY_POINTS)
    if boundary is None:
        return None
    H, n_ok, residual, n_iters, cov = align_mask_to_homography(
        mask,
        init_corners,
        last_H,
        template_mask=template_mask,
        boundary=boundary,
        n_boundary_points=N_BOUNDARY_POINTS,
        huber_delta_px=HUBER_DELTA_PX,
    )
    if H is None:
        return None
    return {
        "H": H,
        "corners": warp_corners(H, init_corners),
        "cov": cov,
        "boundary": boundary,
        "n_ok": int(n_ok),
        "residual": float(residual),
        "n_iters": int(n_iters),
    }


# Deployed fusion settings (configs/tracker/corner_fusion_wholeframe.py and the
# FusionCfg defaults it inherits): one decision per frame, no covariance rescaling.
FUSION_GATE = 20.0
FUSION_SCOPE = "whole_frame"


def fusion_outcome(
    init_corners: np.ndarray,
    loma_H: np.ndarray,
    loma_cov: np.ndarray,
    samh_H: np.ndarray,
    samh_cov: np.ndarray,
) -> dict:
    """What the deployed rule does with these two branch estimates.

    Args:
        init_corners: (2, 4) template corners.
        loma_H: (3, 3) keypoint-branch homography.
        loma_cov: (4, 2, 2) keypoint-branch per-corner covariance.
        samh_H: (3, 3) mask-branch homography.
        samh_cov: (4, 2, 2) mask-branch per-corner covariance.

    Returns:
        Dict with the per-corner squared Mahalanobis disagreements, the gate, the
        decision mode, and the corners the tracker would emit.
    """
    c_loma, c_samh = warp_corners(loma_H, init_corners), warp_corners(samh_H, init_corners)
    d2 = [
        float(mahalanobis_sq(c_loma[:, k] - c_samh[:, k], loma_cov[k] + samh_cov[k]))
        for k in range(4)
    ]
    fused, telemetry = fuse_corners(
        init_corners, loma_H, loma_cov, samh_H, samh_cov, c_loma,
        FUSION_GATE, 1.0, 1.0, FUSION_SCOPE, 0.0,
    )
    return {
        "d2": d2,
        "gate": FUSION_GATE,
        "scope": FUSION_SCOPE,
        "modes": list(telemetry.modes),
        "corners": fused.tolist(),
        "gap_px": [
            float(np.linalg.norm(c_loma[:, k] - c_samh[:, k])) for k in range(4)
        ],
    }


def corner_sigmas(cov: np.ndarray) -> np.ndarray:
    """(4,) mean per-corner standard deviation (px) from a (4, 2, 2) covariance."""
    return np.array([float(np.sqrt(np.trace(c) / 2.0)) for c in cov])


def scan(args: argparse.Namespace) -> None:
    """Rank frames of the given sequences by per-corner covariance spread."""
    info = DATASETS[args.dataset]
    base_dir = getattr(paths, info.base_dir_attr)
    load_gt = gt_loader(args.dataset)
    image_paths_fn = make_image_paths(args.dataset)

    rows = []
    for seq in args.sequences:
        npz_path = LOMA_DIR / info.label / f"{seq}.npz"
        if not npz_path.exists():
            print(f"  {seq}: no LoMa npz, skipped")
            continue
        gt_corners, valid = load_gt(base_dir, seq)
        init_corners = gt_corners[0]
        n_frames = len(image_paths_fn(base_dir, seq))
        with np.load(npz_path) as data:
            for i in range(1, n_frames, args.stride):
                key_a, key_b = f"kptsA_{i:05d}", f"kptsB_{i:05d}"
                if key_a not in data:
                    continue
                if args.require_gt and not (valid[i] and np.any(gt_corners[i] != 0)):
                    continue
                fit = keypoint_branch(data[key_a], data[key_b], init_corners)
                if fit is None or fit["cov"] is None:
                    continue
                sig = corner_sigmas(fit["cov"])
                err = (
                    float(np.linalg.norm(fit["corners"] - gt_corners[i], axis=0).mean())
                    if valid[i]
                    else float("nan")
                )
                rows.append(
                    {
                        "seq": seq,
                        "frame": i,
                        "n_inliers": fit["n_inliers"],
                        "sigma_min": float(sig.min()),
                        "sigma_max": float(sig.max()),
                        "spread": float(sig.max() / max(sig.min(), 1e-6)),
                        "err_px": err,
                    }
                )

    rows.sort(key=lambda r: -r["spread"])
    print(f"\n{'seq':<10} {'frame':>6} {'inl':>5} {'sig_min':>8} {'sig_max':>8} "
          f"{'spread':>7} {'err_px':>7}")
    for r in rows[: args.top]:
        print(f"{r['seq']:<10} {r['frame']:>6} {r['n_inliers']:>5} "
              f"{r['sigma_min']:>8.3f} {r['sigma_max']:>8.3f} {r['spread']:>7.2f} "
              f"{r['err_px']:>7.2f}")


def _jpeg_b64(image: np.ndarray) -> tuple[str, float]:
    """Downscaled JPEG as base64, plus the scale factor applied."""
    scale = min(1.0, JPEG_MAX_WIDTH / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf).decode("ascii"), scale


def _png_b64(mask: np.ndarray) -> str:
    """Binary mask as a base64 PNG at full resolution."""
    ok, buf = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf).decode("ascii")


def export(args: argparse.Namespace) -> None:
    """Write one JSON payload per requested ``seq:frame``."""
    info = DATASETS[args.dataset]
    base_dir = getattr(paths, info.base_dir_attr)
    load_gt = gt_loader(args.dataset)
    image_paths_fn = make_image_paths(args.dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = []
    for spec in args.frames:
        seq, frame_str = spec.split(":")
        frame = int(frame_str)
        gt_corners, valid = load_gt(base_dir, seq)
        init_corners = gt_corners[0]
        image_paths = image_paths_fn(base_dir, seq)

        with np.load(LOMA_DIR / info.label / f"{seq}.npz") as data:
            pts_a = data[f"kptsA_{frame:05d}"]
            pts_b = data[f"kptsB_{frame:05d}"]
        fit = keypoint_branch(pts_a, pts_b, init_corners)
        if fit is None or fit["cov"] is None:
            print(f"{spec}: keypoint branch failed, skipped")
            continue

        mask_dir = MASKS_DIR / info.label / "masks" / seq
        template_mask = cv2.imread(str(mask_dir / "00001.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir / f"{frame + 1:05d}.png"), cv2.IMREAD_GRAYSCALE)
        align = None
        if template_mask is not None and mask is not None:
            # Init the aligner where the tracker would have been: the previous
            # frame's GT pose, so the demo shows a converged fit, not a lost one.
            prev = frame - 1 if valid[frame - 1] else 0
            last_H = cv2.getPerspectiveTransform(
                init_corners.T.astype(np.float32),
                gt_corners[prev].T.astype(np.float32),
            )
            align = mask_branch(mask, template_mask, init_corners, last_H)

        frame_bgr = cv2.imread(str(image_paths[frame]))
        init_bgr = cv2.imread(str(image_paths[0]))
        frame_jpeg, frame_scale = _jpeg_b64(frame_bgr)
        init_jpeg, init_scale = _jpeg_b64(init_bgr)

        boot = bootstrap_corner_covs(
            fit["pts_a"][fit["inlier_mask"]],
            fit["pts_b"][fit["inlier_mask"]],
            init_corners,
        )

        payload = {
            "seq": seq,
            "frame": frame,
            "dataset": info.label,
            "image_size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
            "init_corners": init_corners.tolist(),
            "gt_corners": gt_corners[frame].tolist() if valid[frame] else None,
            "frame_jpeg": frame_jpeg,
            "frame_scale": frame_scale,
            "init_jpeg": init_jpeg,
            "init_scale": init_scale,
            "keypoint": {
                "pts_src": fit["pts_a"].tolist(),
                "pts_dst": fit["pts_b"].tolist(),
                "inlier": fit["inlier_mask"].tolist(),
                "corners": fit["corners"].tolist(),
                "cov": fit["cov"].tolist(),
                "cov_bootstrap": boot.tolist() if boot is not None else None,
                "n_inliers": fit["n_inliers"],
            },
        }
        if align is not None and align["cov"] is not None:
            payload["fusion"] = fusion_outcome(
                init_corners, fit["H"], fit["cov"], align["H"], align["cov"]
            )
            if valid[frame]:
                gt = gt_corners[frame]
                payload["errors_px"] = {
                    "keypoint": [
                        float(np.linalg.norm(fit["corners"][:, k] - gt[:, k])) for k in range(4)
                    ],
                    "mask": [
                        float(np.linalg.norm(align["corners"][:, k] - gt[:, k])) for k in range(4)
                    ],
                    "fused": [
                        float(np.linalg.norm(np.array(payload["fusion"]["corners"])[:, k] - gt[:, k]))
                        for k in range(4)
                    ],
                }
            payload["mask"] = {
                "boundary": align["boundary"].tolist(),
                "corners": align["corners"].tolist(),
                "cov": align["cov"].tolist(),
                "residual_px": align["residual"],
                "n_ok": align["n_ok"],
                "n_iters": align["n_iters"],
                "huber_delta_px": HUBER_DELTA_PX,
                "mask_png": _png_b64(mask),
                "template_png": _png_b64(template_mask),
            }
        examples.append(payload)

        sig_kp = corner_sigmas(np.array(fit["cov"]))
        print(f"{spec}: {fit['n_inliers']} inliers, keypoint sigma "
              f"{np.array2string(sig_kp, precision=2)}", flush=True)
        if "errors_px" in payload:
            e = payload["errors_px"]
            print(f"    err px: keypoint {np.mean(e['keypoint']):.1f}  mask {np.mean(e['mask']):.1f}"
                  f"  fused {np.mean(e['fused']):.1f}   decision {payload['fusion']['modes'][0]}"
                  f"  mean d2 {np.mean(payload['fusion']['d2']):.1f}")
        if align is not None and align["cov"] is not None:
            print(f"    mask branch sigma "
                  f"{np.array2string(corner_sigmas(np.array(align['cov'])), precision=2)}"
                  f"  residual {align['residual']:.2f} px, {align['n_iters']} iters")

    out_path = OUT_DIR / args.out
    out_path.write_text(json.dumps({"examples": examples}))
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pot")
    sub = parser.add_subparsers(dest="mode", required=True)

    scan_p = sub.add_parser("scan", help="rank frames by per-corner covariance spread")
    scan_p.add_argument("--sequences", nargs="+", required=True)
    scan_p.add_argument("--stride", type=int, default=10)
    scan_p.add_argument("--top", type=int, default=40)
    scan_p.add_argument(
        "--require-gt",
        action="store_true",
        help="Only frames carrying a usable GT pose (POT annotates sparsely).",
    )
    scan_p.set_defaults(func=scan)

    export_p = sub.add_parser("export", help="write the demo payload")
    export_p.add_argument("--frames", nargs="+", required=True, help="seq:frame pairs")
    export_p.add_argument("--out", default="examples.json")
    export_p.set_defaults(func=export)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
