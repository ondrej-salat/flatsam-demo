# Inlier Spatial Coverage Metrics for Ensemble Thresholding

**Date**: 2026-04-23

## Rationale

The threshold ensemble (`threshold_ensemble.py`) decides per-frame whether to use the keypoint tracker (LoMa) or a fallback tracker (e.g. Hough symmetry) based on a scalar signal from `*_stats.json`. Currently only inlier counts are available (`inliers_total`, `inliers_object`, `inlier_ratio`).

Scalar counts miss geometric quality: 1000 collinear inliers or 1000 inliers all bunched in the object center yield a high count but a degenerate / poorly-conditioned homography. Adding spatial coverage signals should give a better per-frame reliability estimate.

## New Signals

All computed in the **init-frame coordinate system** (pts_a side of matched pairs), using RANSAC inliers only.

| Signal | Description |
|---|---|
| `hull_coverage` | `ConvexHull_area(inliers) / ConvexHull_area(init_mask_pixels)` — fraction of object area spanned by inliers |
| `hull_stability` | Hull area after removing hull-vertex (boundary) points / hull area before — robustness to outlier inliers driving coverage |
| `spatial_conditioning` | λ_min / (λ_max + ε) of 2×2 covariance of inlier pts — 0 = collinear, 1 = isotropic |

Reference region for `hull_coverage`: foreground pixels of init-frame mask sampled at stride 8 (precomputed once per sequence). Fallback: `kpts0` from NPZ if mask unavailable.

## Implementation

### New files
- **`spatial_coverage.py`** — pure metric functions + benchmark (`--benchmark`) + visualizer (`--visualize`)
- **`compute_spatial_stats.py`** — post-processor: reads NPZ + TXT + GT, writes `*_spatial_stats.json`

### Modified files
- **`experiments/loma_experiment_v2.py`** — adds `inlier_mask_{i:05d}` (uint8) to NPZ alongside existing `kptsA/B`
- **`threshold_ensemble.py`** — extends `THRESHOLD_METRIC_CHOICES`, merges `*_spatial_stats.json` into stats loading

## Commands

```bash
# 1. Benchmark convex hull timing on this machine (run before integrating)
python spatial_coverage.py --benchmark

# 2. Visualize coverage on one sequence frame (sanity check)
python spatial_coverage.py --visualize \
    --npz /path/to/results/loma_r_2000obj_infl10/POT-210/seq001.npz \
    --masks-dir /path/to/masks/POT-210 \
    --seq seq001 --frame 50

# 3. Post-process existing results to compute spatial stats
python compute_spatial_stats.py /path/to/results/loma_r_2000obj_infl10 \
    --dataset pot \
    --masks-dir /path/to/masks/POT-210

# 4. Run threshold ensemble sweep with hull_coverage metric
python threshold_ensemble.py pot \
    --kp-tracker loma_r_2000obj_infl10 \
    --other-tracker hough_symmetry_only \
    --ensemble-name loma_hough_hull \
    --threshold-metric hull_coverage

# 5. Compare against inlier count baseline
python threshold_ensemble.py both \
    --kp-tracker loma_r_2000obj_infl10 \
    --other-tracker hough_symmetry_only \
    --ensemble-name loma_hough_hull_both \
    --threshold-metric hull_coverage
```

## Notes on Approximate Inliers (Fallback)

For NPZ files without `inlier_mask_*` (runs before the loma_v2 patch), `compute_spatial_stats.py` approximates inliers by:
1. Recovering H from 4 GT-init-corners → predicted-corners[i] via `getPerspectiveTransform`
2. Reprojecting `kptsA_i` through H, thresholding error at `ransac_threshold_px` from stats JSON

This differs from MAGSAC inliers (different scoring) but is a reasonable proxy for initial experiments.
