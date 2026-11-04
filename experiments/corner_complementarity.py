"""Does per-corner fusion have anything to fuse? Measure branch complementarity.

The per-corner rule is only worth its complexity if the two branches are
actually right about *different corners* often enough to matter. This script
measures that on a completed grid, from the per-frame branch homographies the
run already stored in ``<seq>_debug.npz``:

For every frame where both branches produced a pose, it computes each branch's
four per-corner distances to the ground truth, and classifies the frame:

``both_good``      every corner of both branches is within tau
``one_branch``     exactly one branch has all four corners within tau
``split``          neither branch is fully within tau, but the per-corner
                   minimum over the two branches is — i.e. the corners the two
                   branches get right are complementary, which is the case the
                   per-corner rule exists for
``neither``        the per-corner minimum still misses

It also reports the achievable gain: the error of the per-corner oracle (take
each corner from whichever branch is closer) against the better whole branch,
and against the deployed fused pose that the same run produced.

CPU-only, no model, no dataset images::

    python experiments/corner_complementarity.py --run corner_fusion_register_final
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from flatsam2.path_config import paths
from flatsam2.registry import DATASETS
from flatsam2.sequences import get_sequences


def _warp(H: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply a homography to (2, 4) corners."""
    proj = H @ np.vstack([corners, np.ones((1, 4))])
    return proj[:2] / proj[2]


def _load_gt(base_dir: Path, seq_name: str, initfix_dir: Optional[Path]):
    """Load GT corners and validity without importing cv2.

    Args:
        base_dir: PlanarTrack root.
        seq_name: Sequence name.
        initfix_dir: Directory of reannotated first-frame corners, or None.

    Returns:
        ``(gt (F, 2, 4), valid (F,), init_corners (2, 4))``.
    """
    annot = np.loadtxt(base_dir / "annos" / f"{seq_name}.txt")
    flags = np.loadtxt(base_dir / "annos" / f"{seq_name}_flag.txt")
    n = min(len(annot), len(flags))
    annot, flags = annot[:n], flags[:n]
    gt = annot.reshape(-1, 4, 2).transpose(0, 2, 1)
    valid = np.any(annot != 0, axis=1) & (flags == 0)
    init = gt[0].copy()
    if initfix_dir is not None:
        init = np.loadtxt(initfix_dir / f"{seq_name}.txt").reshape(4, 2).T
    return gt, valid, init


def analyse_sequence(npz_path: Path, gt: np.ndarray, valid: np.ndarray,
                     init: np.ndarray, tau: float) -> Dict[str, np.ndarray]:
    """Per-frame branch corner errors and the resulting classification."""
    data = np.load(npz_path, allow_pickle=True)
    frames = data["frame_idx"].astype(int)
    keep = (frames < len(gt)) & valid[np.clip(frames, 0, len(gt) - 1)]
    out = {k: [] for k in ("cls", "kp_err", "mask_err", "oracle_err", "fused_err",
                           "best_branch_err", "n_kp_good", "n_mask_good", "frame")}
    # Signed per-corner error vectors, kept so the independence the fusion
    # arithmetic assumes can be tested rather than asserted (see
    # branch_error_correlation).
    out["kp_vec"] = []
    out["mask_vec"] = []
    for row, frame in enumerate(frames):
        if not keep[row]:
            continue
        H_kp, H_mask, H_fused = data["loma_H"][row], data["samh_H"][row], data["fused_H"][row]
        if not (np.isfinite(H_kp).all() and np.isfinite(H_mask).all()):
            continue
        c_kp, c_mask = _warp(H_kp, init), _warp(H_mask, init)
        g = gt[frame]
        d_kp = np.linalg.norm(c_kp - g, axis=0)          # (4,) per-corner distance
        d_mask = np.linalg.norm(c_mask - g, axis=0)
        d_best = np.minimum(d_kp, d_mask)
        kp_ok, mask_ok = d_kp <= tau, d_mask <= tau

        if kp_ok.all() and mask_ok.all():
            cls = "both_good"
        elif kp_ok.all() or mask_ok.all():
            cls = "one_branch"
        elif (d_best <= tau).all():
            cls = "split"
        else:
            cls = "neither"

        rms = lambda d: float(np.sqrt(np.mean(d ** 2)))   # noqa: E731
        out["cls"].append(cls)
        out["frame"].append(int(frame))
        out["kp_err"].append(rms(d_kp))
        out["mask_err"].append(rms(d_mask))
        out["oracle_err"].append(rms(d_best))
        out["best_branch_err"].append(min(rms(d_kp), rms(d_mask)))
        out["fused_err"].append(
            rms(np.linalg.norm(_warp(H_fused, init) - g, axis=0))
            if np.isfinite(H_fused).all() else float("nan")
        )
        out["kp_vec"].append((c_kp - g).T.copy())      # (4, 2)
        out["mask_vec"].append((c_mask - g).T.copy())
        out["n_kp_good"].append(int(kp_ok.sum()))
        out["n_mask_good"].append(int(mask_ok.sum()))
    return {k: np.array(v) for k, v in out.items()}


def branch_error_correlation(kp_vec: np.ndarray, mask_vec: np.ndarray) -> Dict[str, float]:
    """Are the two branches' errors independent, as the fusion arithmetic assumes?

    Precision-weighted averaging of two estimates is only the right combination
    when their errors are uncorrelated; with correlation rho the combined
    variance is understated by roughly 2*rho*sqrt(va*vb), so the fused pose is
    reported as more certain than it is and the chi-square gate is read against
    the wrong distribution. That is an assumption, and it is measurable: the
    branches' signed corner errors against GT are on disk for every frame.

    Reported per coordinate (x and y pooled separately would hide a common
    translation, so both are given) and as the magnitude correlation, which
    catches "both branches fail on the same hard frames" even when the
    directions are unrelated.

    Args:
        kp_vec: (F, 4, 2) signed keypoint-branch corner errors, px.
        mask_vec: (F, 4, 2) signed mask-branch corner errors, px.

    Returns:
        Correlation coefficients keyed by what was correlated.
    """
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.ravel(), b.ravel()
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[keep], b[keep])[0, 1])

    return {
        "signed_x": corr(kp_vec[..., 0], mask_vec[..., 0]),
        "signed_y": corr(kp_vec[..., 1], mask_vec[..., 1]),
        "magnitude": corr(
            np.linalg.norm(kp_vec, axis=-1), np.linalg.norm(mask_vec, axis=-1)
        ),
        "n_frames": float(len(kp_vec)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="corner_fusion_register_final",
                        help="Results directory name holding <seq>_debug.npz.")
    parser.add_argument("--dataset", default="pti", help="Dataset key.")
    parser.add_argument("--tau", type=float, default=15.0,
                        help="Per-corner distance (px) counted as good.")
    parser.add_argument("--limit", type=int, default=0, help="Cap sequences (0 = all).")
    parser.add_argument("--out", type=Path,
                        default=paths.COMPARISONS_DIRECTORY / "reports" / "complementarity.json")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    base_dir = getattr(paths, ds.base_dir_attr)
    initfix = paths.PT_INITFIX_DIR if ds.gt_loader == "pti" else None
    run_dir = paths.OUTPUT_DIRECTORY / args.run / ds.label

    sequences = get_sequences(args.dataset, base_dir)
    if args.limit:
        sequences = sequences[: args.limit]

    agg = {k: [] for k in ("cls", "kp_err", "mask_err", "oracle_err", "fused_err",
                           "best_branch_err", "n_kp_good", "n_mask_good",
                           "kp_vec", "mask_vec")}
    per_seq_examples = []
    for seq in sequences:
        npz_path = run_dir / f"{seq}_debug.npz"
        if not npz_path.exists():
            continue
        gt, valid, init = _load_gt(base_dir, seq, initfix)
        res = analyse_sequence(npz_path, gt, valid, init, args.tau)
        if len(res["cls"]) == 0:
            continue
        for k in agg:
            agg[k].append(res[k])
        # Keep the clearest "split" frame of this sequence: complementary
        # corners, and the largest gain of the per-corner oracle over the
        # better whole branch.
        split = np.where(res["cls"] == "split")[0]
        if len(split):
            gain = res["best_branch_err"][split] - res["oracle_err"][split]
            j = split[int(np.argmax(gain))]
            per_seq_examples.append({
                "seq": seq, "frame": int(res["frame"][j]),
                "kp_err": float(res["kp_err"][j]), "mask_err": float(res["mask_err"][j]),
                "oracle_err": float(res["oracle_err"][j]), "fused_err": float(res["fused_err"][j]),
                "n_kp_good": int(res["n_kp_good"][j]), "n_mask_good": int(res["n_mask_good"][j]),
                "gain": float(gain.max()),
            })

    cls = np.concatenate(agg["cls"])
    n = len(cls)
    counts = {c: int((cls == c).sum()) for c in ("both_good", "one_branch", "split", "neither")}
    shares = {c: 100.0 * v / n for c, v in counts.items()}

    oracle = np.concatenate(agg["oracle_err"])
    best_branch = np.concatenate(agg["best_branch_err"])
    fused = np.concatenate(agg["fused_err"])
    kp_good = np.concatenate(agg["n_kp_good"])
    mask_good = np.concatenate(agg["n_mask_good"])

    # The specific pattern the rule is sold on, stated precisely: one branch has
    # exactly three corners within tau, and the per-corner minimum has all four
    # — i.e. the other branch supplies a usable fourth corner. That is a subset
    # of the "split" class, and it is what per-corner mixing rescues.
    rescue = ((cls == "split") & ((kp_good == 3) | (mask_good == 3)))
    three_plus_one = int(rescue.sum())
    fused_beats_best = float(100.0 * np.mean(fused[np.isfinite(fused)]
                                             < best_branch[np.isfinite(fused)]))

    correlation = branch_error_correlation(
        np.concatenate(agg["kp_vec"]), np.concatenate(agg["mask_vec"])
    )

    summary = {
        "branch_error_correlation": correlation,
        "run": args.run, "dataset": args.dataset, "tau_px": args.tau,
        "n_frames": n, "n_sequences": len(agg["cls"]),
        "counts": counts, "shares_pct": shares,
        "three_plus_one_pct": 100.0 * three_plus_one / n,
        "fused_beats_better_branch_pct": fused_beats_best,
        "n_good_hist": {
            "keypoint": [int((kp_good == k).sum()) for k in range(5)],
            "mask": [int((mask_good == k).sum()) for k in range(5)],
        },
        "p_at_15_by_pose": {
            "keypoint_branch": float(100.0 * (np.concatenate(agg["kp_err"]) <= 15).mean()),
            "mask_branch": float(100.0 * (np.concatenate(agg["mask_err"]) <= 15).mean()),
            "better_branch_oracle": float(100.0 * (best_branch <= 15).mean()),
            "per_corner_oracle": float(100.0 * (oracle <= 15).mean()),
            "deployed_fusion": float(100.0 * (fused[np.isfinite(fused)] <= 15).mean()),
        },
        "median_err_px": {
            "keypoint_branch": float(np.median(np.concatenate(agg["kp_err"]))),
            "mask_branch": float(np.median(np.concatenate(agg["mask_err"]))),
            "better_branch_oracle": float(np.median(best_branch)),
            "per_corner_oracle": float(np.median(oracle)),
            "deployed_fusion": float(np.nanmedian(fused)),
        },
        # Frames where both branches are already good: here fusion cannot
        # rescue anything, it can only average two estimates of the same thing,
        # which is where a precision-weighted mean is supposed to reduce noise.
        "both_good_median_err_px": {
            "keypoint_branch": float(np.median(np.concatenate(agg["kp_err"])[cls == "both_good"])),
            "mask_branch": float(np.median(np.concatenate(agg["mask_err"])[cls == "both_good"])),
            "better_branch_oracle": float(np.median(best_branch[cls == "both_good"])),
            "deployed_fusion": float(np.nanmedian(fused[cls == "both_good"])),
        },
        "examples": sorted(per_seq_examples, key=lambda e: -e["gain"])[:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    print(f"\nTop complementary frames -> {args.out}")
    for e in summary["examples"][:5]:
        print(f"  {e['seq']} frame {e['frame']}: kp {e['kp_err']:.1f} px "
              f"({e['n_kp_good']}/4 good), mask {e['mask_err']:.1f} px "
              f"({e['n_mask_good']}/4), per-corner oracle {e['oracle_err']:.1f}, "
              f"deployed fusion {e['fused_err']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
