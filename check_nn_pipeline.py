"""Verbose diagnostic for the NN homography scorer pipeline.

Runs end-to-end on synthetic inputs and prints the value/shape/range of every
intermediate result.  Designed to catch silent failures (wrong thresholds,
None returns, inf scores) without needing a real checkpoint.

Optionally loads a trained checkpoint to check that scores are sensible:
    python check_nn_pipeline.py --checkpoint checkpoints/nn_quality/best.pt

By default the script exercises all three architectures (option_e, option_c,
point_baseline).  Use --arch to restrict to one.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _fmt(arr, name=""):
    """One-liner summary of a numpy array."""
    if arr is None:
        return f"{name}: None"
    finite = np.isfinite(arr)
    n_inf = (~finite).sum()
    vals = arr[finite]
    if vals.size == 0:
        return f"{name}: shape={arr.shape} dtype={arr.dtype} ALL NON-FINITE"
    s = (f"{name}: shape={arr.shape} dtype={arr.dtype}"
         f"  min={vals.min():.3g}  max={vals.max():.3g}"
         f"  mean={vals.mean():.3g}  std={vals.std():.3g}")
    if n_inf:
        s += f"  [WARNING: {n_inf} non-finite values]"
    return s


def _section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Synthetic input factory
# ---------------------------------------------------------------------------

def make_rect_mask(h=480, w=640, value=255):
    """0/255 rectangle mask (reference shape)."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[h//5 : 4*h//5, w//5 : 4*w//5] = value
    return m


def make_shifted_mask(h=480, w=640, dx=10, dy=8, value=255):
    """Slightly shifted rectangle, same size."""
    m = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = h//5 + dy, 4*h//5 + dy
    x0, x1 = w//5 + dx, 4*w//5 + dx
    m[max(0,y0):min(h,y1), max(0,x0):min(w,x1)] = value
    return m


def translation_H(dx, dy):
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def bad_H(scale=5.0):
    """Homography with a large random perturbation — should score worse."""
    rng = np.random.default_rng(7)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] += rng.uniform(-scale, scale, size=(2, 3)) * 0.05
    H[0, 2] += scale * 10
    H[1, 2] += scale * 10
    return H


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def check_reference_contour(mask_255):
    _section("1. ReferenceContour.build()")
    from flatsam2.homography_quality import ReferenceContour

    print(f"  Input mask: shape={mask_255.shape} dtype={mask_255.dtype}"
          f"  unique values={np.unique(mask_255).tolist()}")
    ref = ReferenceContour.build(mask_255)
    print(f"  ref.points:       {_fmt(ref.points)}")
    print(f"  ref.curvature:    {_fmt(ref.curvature)}")
    print(f"  ref.weights:      {_fmt(ref.weights)}")
    print(f"  ref.mask:         {'not None, shape=' + str(ref.mask.shape) if ref.mask is not None else 'None  ← problem'}")
    print(f"  ref.boundary_pts: {_fmt(ref.boundary_pts)}")
    print(f"  ref.area:         {ref.area:.0f} px")

    pts, ts, tc = ref.get_nn_points(128)
    norms = np.sqrt(ts**2 + tc**2)
    print(f"  get_nn_points(128): pts={pts.shape}  tangent norm min={norms.min():.4f} max={norms.max():.4f}")
    return ref


def check_preprocess(ref, mask_B, H, arch, image_shape):
    from flatsam2.nn.preprocess import preprocess

    print(f"\n  -- arch={arch!r}  mask_B unique={np.unique(mask_B).tolist()}")
    prep = preprocess(ref, mask_B, H, image_shape, n_points=128, architecture=arch)
    if prep is None:
        print(f"  preprocess() returned None  ← PROBLEM")
        return None

    fields_to_check = [
        "pts_Ap", "pts_B", "sdf_B_at_Ap", "sdf_Ap_at_B",
        "tan_sin_A", "tan_cos_A", "tan_sin_B", "tan_cos_B",
        "is_in_view", "is_on_boundary_B", "norm_xy_A", "norm_xy_B",
    ]
    for f in fields_to_check:
        v = prep.get(f)
        if isinstance(v, np.ndarray):
            print(f"  {_fmt(v, f)}")
        else:
            print(f"  {f}: {v}")

    # SDF images (may be None for option_e / point_baseline)
    for f in ("sdf_Ap", "sdf_B", "mask_Ap"):
        v = prep.get(f)
        if v is not None:
            print(f"  {_fmt(v, f)}")
        else:
            print(f"  {f}: None (expected for {arch})")
    return prep


def check_features(prep, arch):
    from flatsam2.nn.models import (
        build_option_e_features, build_option_c_input,
        build_point_baseline_features,
    )

    if arch == "option_e":
        feat_A, feat_B = build_option_e_features(prep)
        print(f"  {_fmt(feat_A, 'feat_A')}")
        print(f"  {_fmt(feat_B, 'feat_B')}")
        return feat_A, feat_B
    elif arch == "point_baseline":
        features = build_point_baseline_features(prep)
        print(f"  {_fmt(features, 'features_combined')}")
        return features
    else:  # option_c
        crop, s = build_option_c_input(prep)
        print(f"  {_fmt(crop.numpy(), 'crop')}  scale={s:.3f}")
        return crop, s


def check_model_forward(arch, features, device="cpu"):
    import torch
    from flatsam2.nn.models import (
        BoundaryTransformer, SDFCropCNN, PointSetBaseline,
    )

    if arch == "option_e":
        model = BoundaryTransformer()
        feat_A, feat_B = features
        fa = torch.from_numpy(feat_A).unsqueeze(0)
        fb = torch.from_numpy(feat_B).unsqueeze(0)
        with torch.no_grad():
            pred, log_sigma = model(fa, fb)
    elif arch == "point_baseline":
        model = PointSetBaseline()
        feat = features
        ft = torch.from_numpy(feat).unsqueeze(0)
        with torch.no_grad():
            pred, log_sigma = model(ft)
    else:  # option_c
        model = SDFCropCNN()
        crop, s = features
        with torch.no_grad():
            pred, log_sigma = model(crop.unsqueeze(0))
        pred = pred / s
        log_sigma = log_sigma - math.log(s)

    rmse = pred.item()
    sigma = math.exp(float(log_sigma.clamp(-5, 5)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")
    print(f"  predicted RMSE: {rmse:.3f} px   sigma: {sigma:.3f} px")
    return rmse, sigma


def check_full_scorer(arch, checkpoint_path, ref, mask_B, H_good, H_bad, image_shape):
    from flatsam2.nn.inference import NNHomographyScorer

    scorer = NNHomographyScorer(
        str(Path(checkpoint_path).expanduser()),
        arch=arch,
        n_points=128,
    )
    scorer.init_reference(ref)

    rmse_good, sigma_good = scorer.score(mask_B, H_good, image_shape)
    rmse_bad,  sigma_bad  = scorer.score(mask_B, H_bad,  image_shape)

    print(f"  good H:  predicted_rmse={rmse_good:.2f}  sigma={sigma_good:.2f}")
    print(f"  bad  H:  predicted_rmse={rmse_bad:.2f}   sigma={sigma_bad:.2f}")
    if math.isfinite(rmse_good) and math.isfinite(rmse_bad):
        if rmse_bad > rmse_good:
            print(f"  ✓ bad H correctly scored higher ({rmse_bad:.2f} > {rmse_good:.2f})")
        else:
            print(f"  ✗ bad H NOT scored higher — model may be poorly calibrated")
    else:
        print(f"  ✗ at least one score is non-finite — scoring failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verbose diagnostic for the NN scorer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained .pt checkpoint (optional).")
    parser.add_argument("--arch", choices=["option_e", "option_c", "point_baseline"],
                        default=None, help="Restrict to one architecture.")
    args = parser.parse_args()

    archs = [args.arch] if args.arch else ["option_e", "option_c", "point_baseline"]

    H_good = translation_H(dx=10, dy=8)
    H_bad  = bad_H(scale=5.0)
    h, w   = 480, 640
    image_shape = (h, w)

    mask_ref_255  = make_rect_mask(h, w, value=255)   # 0/255 — init mask
    mask_B_255    = make_shifted_mask(h, w, value=255) # 0/255 — seg mask (training format)
    mask_B_01     = make_shifted_mask(h, w, value=1)   # 0/1   — SAM3 output format

    # ------------------------------------------------------------------
    # Step 1: ReferenceContour
    # ------------------------------------------------------------------
    ref = check_reference_contour(mask_ref_255)

    # ------------------------------------------------------------------
    # Step 2 & 3: preprocess + features + random-weight forward
    # ------------------------------------------------------------------
    for arch in archs:
        _section(f"2-4. preprocess → features → forward  [{arch}]")

        # 2a. correct mask format (0/255)
        prep_255 = check_preprocess(ref, mask_B_255, H_good, arch, image_shape)
        if prep_255 is not None:
            print()
            feats_255 = check_features(prep_255, arch)
            print()
            print("  [random weights, good H, mask 0/255]")
            check_model_forward(arch, feats_255)

        # 2b. SAM3 mask format (0/1) — the bug we fixed
        print()
        prep_01 = check_preprocess(ref, mask_B_01, H_good, arch, image_shape)
        if prep_01 is not None:
            feats_01 = check_features(prep_01, arch)
            print()
            print("  [random weights, good H, mask 0/1]")
            check_model_forward(arch, feats_01)
        else:
            print("  ← mask_B 0/1 caused None — mask threshold bug not fixed")

        # 2c. bad homography
        if prep_255 is not None:
            print()
            prep_bad = check_preprocess(ref, mask_B_255, H_bad, arch, image_shape)
            if prep_bad is not None:
                feats_bad = check_features(prep_bad, arch)
                print()
                print("  [random weights, bad H, mask 0/255]")
                rmse_bad, _ = check_model_forward(arch, feats_bad)
                rmse_good_val, _ = check_model_forward(arch, feats_255)
                note = ("good < bad ✓" if rmse_good_val < rmse_bad
                        else "good >= bad (random weights, so this is expected)")
                print(f"  good_H RMSE={rmse_good_val:.3f}  bad_H RMSE={rmse_bad:.3f}  → {note}")

    # ------------------------------------------------------------------
    # Step 5: full scorer with trained checkpoint (optional)
    # ------------------------------------------------------------------
    if args.checkpoint:
        _section(f"5. NNHomographyScorer with checkpoint  [{args.checkpoint}]")
        ckpt = Path(args.checkpoint).expanduser()
        if not ckpt.exists():
            print(f"  Checkpoint not found: {ckpt}")
        else:
            for arch in archs:
                if arch == "option_c":
                    continue  # skip if checkpoint is for a different arch
                print(f"\n  arch={arch}")
                try:
                    check_full_scorer(
                        arch, str(ckpt), ref,
                        mask_B_01,   # use SAM3 format to test the fix
                        H_good, H_bad, image_shape,
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()
    else:
        print("\n(no --checkpoint given; skip trained-model scoring)")

    print("\nDone.")


if __name__ == "__main__":
    main()
