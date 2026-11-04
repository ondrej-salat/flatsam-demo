"""Smoke test for the NN homography quality estimation implementation.

Run: python test_nn_smoke.py
"""

import numpy as np
import torch


def test_reference_contour_extension():
    from flatsam2.homography_quality import ReferenceContour

    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:150, 80:220] = 255
    ref = ReferenceContour.build(mask)

    assert ref.mask is not None, "mask should be stored"
    assert ref.tangent_sin is not None, "tangent_sin should be computed"
    assert ref.tangent_cos is not None, "tangent_cos should be computed"
    assert ref.tangent_sin.shape == ref.points.shape[:1]
    assert ref.tangent_cos.shape == ref.points.shape[:1]

    pts, ts, tc = ref.get_nn_points(128)
    assert pts.shape == (128, 2), f"Expected (128, 2), got {pts.shape}"
    assert ts.shape == (128,), f"Expected (128,), got {ts.shape}"
    assert tc.shape == (128,), f"Expected (128,), got {tc.shape}"

    # Tangent vectors should be roughly unit length
    norms = np.sqrt(ts ** 2 + tc ** 2)
    assert np.allclose(norms, 1.0, atol=0.01), "Tangents should be unit vectors"

    print("  ReferenceContour extension: OK")


def test_preprocess():
    from flatsam2.homography_quality import ReferenceContour
    from flatsam2.nn.preprocess import preprocess

    mask_A = np.zeros((200, 300), dtype=np.uint8)
    mask_A[50:150, 80:220] = 255
    ref = ReferenceContour.build(mask_A)

    mask_B = np.zeros((200, 300), dtype=np.uint8)
    mask_B[55:155, 85:225] = 255  # Slightly shifted

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 5.0  # Small translation
    H[1, 2] = 5.0

    result = preprocess(ref, mask_B, H, (200, 300), n_points=128)
    assert result is not None, "preprocess should succeed"
    assert result["pts_Ap"].shape == (128, 2)
    assert result["pts_B"].shape == (128, 2)
    assert result["sdf_B_at_Ap"].shape == (128,)
    assert result["sdf_Ap_at_B"].shape == (128,)
    assert result["norm_xy_A"].shape == (128, 2)
    assert result["norm_xy_B"].shape == (128, 2)
    assert result["mask_Ap"].shape == (200, 300)
    assert result["sdf_B"].shape == (200, 300)

    print("  Preprocessing pipeline: OK")


def test_boundary_transformer():
    from flatsam2.nn.models import BoundaryTransformer

    model = BoundaryTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BoundaryTransformer params: {n_params:,}")

    feat_A = torch.randn(2, 128, 20)
    feat_B = torch.randn(2, 128, 20)
    pred_rmse, log_sigma = model(feat_A, feat_B)

    assert pred_rmse.shape == (2,), f"Expected (2,), got {pred_rmse.shape}"
    assert log_sigma.shape == (2,), f"Expected (2,), got {log_sigma.shape}"
    assert (pred_rmse >= 0).all(), "softplus output should be non-negative"
    assert (log_sigma >= -5).all() and (log_sigma <= 5).all(), \
        "log_sigma should be clamped to [-5, 5]"

    print("  BoundaryTransformer forward: OK")


def test_sdf_crop_cnn():
    from flatsam2.nn.models import SDFCropCNN

    model = SDFCropCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SDFCropCNN params: {n_params:,}")

    x = torch.randn(2, 4, 128, 128)
    pred_rmse, log_sigma = model(x)

    assert pred_rmse.shape == (2,), f"Expected (2,), got {pred_rmse.shape}"
    assert log_sigma.shape == (2,), f"Expected (2,), got {log_sigma.shape}"
    assert (pred_rmse >= 0).all(), "softplus output should be non-negative"

    print("  SDFCropCNN forward: OK")


def test_feature_construction():
    from flatsam2.nn.models import build_option_e_features, build_option_c_input

    # Fake preprocess output
    n = 128
    prep = {
        "sdf_B_at_Ap": np.random.randn(n).astype(np.float32),
        "sdf_Ap_at_B": np.random.randn(n).astype(np.float32),
        "tan_sin_A": np.random.randn(n).astype(np.float32),
        "tan_cos_A": np.random.randn(n).astype(np.float32),
        "tan_sin_B": np.random.randn(n).astype(np.float32),
        "tan_cos_B": np.random.randn(n).astype(np.float32),
        "is_in_view": np.ones(n, dtype=np.float32),
        "norm_xy_A": np.random.rand(n, 2).astype(np.float32),
        "norm_xy_B": np.random.rand(n, 2).astype(np.float32),
        "mask_Ap": np.zeros((200, 300), dtype=np.uint8),
        "sdf_Ap": np.zeros((200, 300), dtype=np.float32),
        "mask_B": np.zeros((200, 300), dtype=np.uint8),
        "sdf_B": np.zeros((200, 300), dtype=np.float32),
    }
    # Fill some area so bbox computation works
    prep["mask_Ap"][50:150, 80:220] = 255
    prep["mask_B"][55:155, 85:225] = 255

    feat_A, feat_B = build_option_e_features(prep)
    assert feat_A.shape == (128, 20), f"Expected (128, 20), got {feat_A.shape}"
    assert feat_B.shape == (128, 20), f"Expected (128, 20), got {feat_B.shape}"

    crop, s = build_option_c_input(prep)
    assert crop.shape == (4, 128, 128), f"Expected (4, 128, 128), got {crop.shape}"
    assert s > 0

    print("  Feature construction: OK")


def test_loss():
    from flatsam2.nn.loss import censored_heteroscedastic_loss

    pred = torch.tensor([5.0, 10.0, 25.0, 50.0])
    ls = torch.tensor([0.0, 0.0, 0.0, 0.0])
    gt = torch.tensor([4.0, 12.0, 35.0, 100.0])

    loss = censored_heteroscedastic_loss(pred, ls, gt)
    assert loss.shape == (), "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"

    # Perfect prediction should have low uncensored loss
    pred_perfect = torch.tensor([4.0, 12.0])
    ls_perfect = torch.tensor([0.0, 0.0])
    gt_perfect = torch.tensor([4.0, 12.0])
    loss_perfect = censored_heteroscedastic_loss(
        pred_perfect, ls_perfect, gt_perfect)
    assert loss_perfect.item() < loss.item(), \
        "Perfect prediction should have lower loss"

    print(f"  Loss function: OK (loss={loss.item():.4f})")


def test_data_gen():
    from flatsam2.nn.data_gen import (
        generate_shape, extract_contour, sample_scene_transform,
        augment_mask_B, sample_h_candidate, generate_sample,
    )

    rng = np.random.default_rng(42)

    # Shape generation
    mask = generate_shape(rng)
    assert mask.shape == (720, 1280), f"Expected (720, 1280), got {mask.shape}"
    assert np.count_nonzero(mask) > 100, "Shape should have > 100 pixels"

    # Contour extraction
    contour = extract_contour(mask)
    assert contour is not None, "Should find contour"
    assert contour.shape == (256, 2), f"Expected (256, 2), got {contour.shape}"

    # Scene transform
    T = sample_scene_transform(rng)
    assert T.shape == (3, 3), f"Expected (3, 3), got {T.shape}"
    assert abs(T[2, 2] - 1.0) < 0.01, "T[2,2] should be ~1"

    # Mask augmentation
    import cv2
    mask_B_clean = cv2.warpPerspective(
        mask, T, (1280, 720), flags=cv2.INTER_NEAREST)
    if np.count_nonzero(mask_B_clean) > 100:
        mask_B_aug = augment_mask_B(mask_B_clean, rng)
        assert mask_B_aug.shape == mask_B_clean.shape

    # Full sample
    sample = None
    for _ in range(10):  # May need a few tries for valid sample
        sample = generate_sample(rng)
        if sample is not None:
            break

    if sample is not None:
        assert sample["features_A"].shape == (128, 20)
        assert sample["features_B"].shape == (128, 20)
        assert sample["crop_4ch"].shape == (4, 128, 128)
        assert isinstance(sample["gt_rmse"], float)
        assert isinstance(sample["censored"], bool)
        print("  Data generation: OK")
    else:
        print("  Data generation: OK (no valid sample in 10 tries, but code ran)")


def test_dataset():
    from flatsam2.nn.data_gen import SyntheticHomographyDataset
    import itertools

    ds = SyntheticHomographyDataset(
        architecture="option_e", seed=42, max_samples=3)

    samples = list(itertools.islice(ds, 3))
    assert len(samples) <= 3
    if samples:
        feat_A, feat_B, gt_rmse, censored = samples[0]
        assert feat_A.shape == (128, 20)
        assert feat_B.shape == (128, 20)
        print(f"  Dataset iteration: OK ({len(samples)} samples)")
    else:
        print("  Dataset iteration: OK (no samples generated)")


def test_end_to_end():
    """Full pipeline: generate data → preprocess → model → loss."""
    from flatsam2.homography_quality import ReferenceContour
    from flatsam2.nn.preprocess import preprocess
    from flatsam2.nn.models import (
        BoundaryTransformer, build_option_e_features,
        SDFCropCNN, build_option_c_input,
    )
    from flatsam2.nn.loss import censored_heteroscedastic_loss

    # Create test data
    mask_A = np.zeros((200, 300), dtype=np.uint8)
    mask_A[50:150, 80:220] = 255
    ref = ReferenceContour.build(mask_A)

    mask_B = np.zeros((200, 300), dtype=np.uint8)
    mask_B[55:155, 85:225] = 255

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = 5.0
    H[1, 2] = 5.0

    prep = preprocess(ref, mask_B, H, (200, 300))
    assert prep is not None

    # Option E
    feat_A, feat_B = build_option_e_features(prep)
    model_e = BoundaryTransformer()
    pred_e, ls_e = model_e(
        torch.from_numpy(feat_A).unsqueeze(0),
        torch.from_numpy(feat_B).unsqueeze(0),
    )
    loss_e = censored_heteroscedastic_loss(
        pred_e, ls_e,
        torch.tensor([5.0]),
    )

    # Option C
    crop, s = build_option_c_input(prep)
    model_c = SDFCropCNN()
    pred_c, ls_c = model_c(crop.unsqueeze(0))
    loss_c = censored_heteroscedastic_loss(
        pred_c / s, ls_c - np.log(s),
        torch.tensor([5.0]),
    )

    print(f"  End-to-end Option E: pred={pred_e.item():.2f}, loss={loss_e.item():.4f}")
    print(f"  End-to-end Option C: pred={pred_c.item()/s:.2f}, loss={loss_c.item():.4f}")
    print("  End-to-end pipeline: OK")


if __name__ == "__main__":
    print("Running NN homography quality estimation smoke tests...\n")

    tests = [
        ("ReferenceContour extension", test_reference_contour_extension),
        ("Preprocessing pipeline", test_preprocess),
        ("BoundaryTransformer", test_boundary_transformer),
        ("SDFCropCNN", test_sdf_crop_cnn),
        ("Feature construction", test_feature_construction),
        ("Loss function", test_loss),
        ("Data generation", test_data_gen),
        ("Dataset iteration", test_dataset),
        ("End-to-end pipeline", test_end_to_end),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[{name}]")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
