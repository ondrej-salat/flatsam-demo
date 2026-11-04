# Handoff — quad-free (arbitrary-shape) tracker thread

For a Claude Code session running **on the HPC cluster**. Repo tip: `8a62c3ec`.

## The question

Can we drop the quadrilateral target assumption from SAM-H (replace the Hough
lines→cyclic-quad pipeline with direct mask alignment) at an acceptable accuracy
cost? The user's bar: a drop vs our own quad system is fine if we still beat
published SOTA, because the generality claim is the paper.

Reference points (user-supplied published numbers; ours per-seq mean):

| system | POT p@5 | POT p@15 | PTi p@5 | PTi p@15 |
|---|---|---|---|---|
| WOFTSAM (published) | 91.9 | 97.5 | 57.4 | 77.6 |
| SAM-H (published) | 64.4 | 89.0 | 62.0 | 80.1 |
| ours quad (`loma_samh_decoupled_rerun`) | **?** | 97.66 | 65.44 | 87.86 |
| ours alignh (fallback ON, invalid — see below) | **?** | 97.37 | 62.12 | 85.45 |

POT p@5 was never read out for either of our configs. It is the cell where
quad-free could actually lose to WOFTSAM. Get it.

## Why the existing alignh run does not count

`loma_samh_decoupled_alignh` kept `align_fallback_hough=True` and
`symmetry.enabled=True`, and logged **nothing** about which estimator produced
mask-H. Its −2.41pp PTi P@15 may be measuring the Hough quad. Not recoverable
from the existing npz files — the telemetry did not exist. Treat that run as
superseded, not as evidence.

## What is now in the repo (commit 8a62c3ec)

- **`configs/tracker/loma_samh_decoupled_alignh_pure.py`** — `method="align"`,
  `align_fallback_hough=False`, `symmetry.enabled=False`. No quad machinery in
  mask-H. Both settings propagate to the nested SAM-H sub-tracker.
- **Branch-usage telemetry** — `MaskHTelemetry` → `TrackerDebugInfo.mask_h` →
  `{seq}_debug.npz` keys: `maskh_source` (`align`/`hough`/`none`),
  `maskh_align_attempted`, `maskh_align_accepted`, `maskh_align_residual_px`,
  `maskh_align_n_ok`, `maskh_align_iters`.
- **`scripts/hpc_readout.py`** — Part 4 prints branch shares, residual
  median/p90, GN-iteration median/p90/max/at-cap%, and flags `>5%` Hough as
  NOT quad-free. Parts 1–3 are the older completeness + P@15 + per-sequence-diff
  read-out.
- **Speed fixes** — template contour hoisted out of the per-frame loop
  (`template_boundary()` public, tracker caches it); GN iteration count returned
  and logged. `align_mask_to_homography` now returns a **4-tuple**.

**Nothing above was ever executed.** Syntax-checked only — the authoring
workstation has no deps installed. First job on the cluster is to actually run it.

**Anything you change here must be carried back.** The workstation pushes this
repo with `make sjo_sync` (rsync `--delete`, and the second pass overwrites
`.git`), so HPC-side edits *and commits* are destroyed by the next sync. If you
change code or docs, say so explicitly and have the user pull it back before
they sync again. Results belong under `OUTPUT_DIRECTORY`, never inside the repo.

## Run order

`source .envrc` first — nothing runs in a bare shell. Light steps then work on
the login node; anything loading a model or a dataset is submitted. See
CLAUDE.md "HPC rules".

```bash
source .envrc

# 0. prove the code works — nothing in 8a62c3ec has ever been executed
pytest tests/test_mask_align.py            # CPU-only, login node is fine

# 1. smoke: one sequence, confirm telemetry lands (GPU -> submit)
python slurm/submit.py pot --config configs/tracker/loma_samh_decoupled_alignh_pure.py \
    --jobs 1                               # see submit.py --help to pin one sequence
# then inspect the npz on the login node:
python - <<'PY'
import numpy as np
d = np.load("<OUTPUT_DIRECTORY>/loma_samh_decoupled_alignh_pure/POT-210/<seq>_debug.npz")
print(np.unique(d["maskh_source"], return_counts=True))
it = d["maskh_align_iters"]
print("iters median", np.median(it[it >= 0]))
PY

# 2. speed: aligner vs Hough on the same sequence (GPU -> submit)
python slurm/submit_cmd.py -- python profile_tracker.py pot \
    --config configs/tracker/loma_samh_decoupled_alignh_pure.py \
    --sequence V01_1 --max-frames 60 --warmup 5
python slurm/submit_cmd.py -- python profile_tracker.py pot \
    --config configs/tracker/loma_samh_decoupled_rerun.py \
    --sequence V01_1 --max-frames 60 --warmup 5
# compare the mask_H.align section against mask_H in slurm/logs/

# 3. full runs (only if 1 and 2 are acceptable)
python slurm/submit.py pti --config configs/tracker/loma_samh_decoupled_alignh_pure.py --jobs 10
python slurm/submit.py pot --config configs/tracker/loma_samh_decoupled_alignh_pure.py --jobs 10
python slurm/submit_cmd.py --gpus 0 -- python evaluate.py pti \
    --results-dir <OUTPUT_DIRECTORY> --config configs/tracker/loma_samh_decoupled_alignh_pure.py
python slurm/submit_cmd.py --gpus 0 -- python evaluate.py pot \
    --results-dir <OUTPUT_DIRECTORY> --config configs/tracker/loma_samh_decoupled_alignh_pure.py

# 4. read out — Part 4 BEFORE Part 2
python scripts/hpc_readout.py              # login node; -> /tmp/hpc_readout.txt
```

Job state: `squeue -u $USER`. Logs: `slurm/logs/`.

Add `loma_samh_decoupled_rerun` to `BRANCH_USAGE_CONFIGS` / `METRIC_CONFIGS` in
`hpc_readout.py` as needed; `METRIC_CONFIGS` currently pairs rerun vs the old
alignh, so point it at `_pure` for the new comparison.

## Decision gates, in order

1. **Branch shares.** If `none` dominates, the aligner is rejecting constantly
   and the metric is measuring last-H coasting plus the LoMA branch — not mask
   alignment. Loosen `align_residual_ok_px` (6px default) and rerun. Any
   nonzero `hough` share means the config is wrong (fallback should be off).
2. **Speed.** If `mask_H.align` ≫ `mask_H`, see the optimization list below
   before committing cluster time to the full grid.
3. **Accuracy.** Only then read P@15. Bar: beat published SOTA (PTi P@15 > 80.1,
   POT P@15 near 97.5, POT p@5 vs 91.9). Losing to our own quad baseline is
   acceptable per the user; losing to published SOTA is not.

## If the aligner is too slow

In order of payoff per risk:

1. **Analytic Jacobian.** `_gn_align` builds the Jacobian by 8 numeric forward
   differences, so each iteration costs 9 residual evaluations — up to 540 tiny
   numpy round-trips per frame at `max_iters=60`. Chain rule: ∇sdf (one Sobel
   per frame) sampled at the warped boundary × ∂warped/∂control. The fiddly part
   is differentiating `getPerspectiveTransform` w.r.t. the control points. ~9×
   on the loop.
2. **Crop the SDF** to a dilated ROI around the predicted quad — the two
   `cv2.distanceTransform` calls are full-frame. Watch validity near crop edges.
3. **`DIST_MASK_5`** instead of `DIST_MASK_PRECISE` in
   `mask_metrics.signed_distance` — faster, approximate; needs a basin re-check.
4. **Lower `align_max_iters`** — free if `maskh_align_iters` says the cap is
   rarely binding.

## Open, beyond this run

- **Align residual as a selector feature.** alignh's per-sequence split was 71
  improved / 148 regressed — a real per-frame conditional win applied blind.
  The residual is now in telemetry, so `selector_study` can finally test it
  against the high-inlier LoMA catastrophes that all previous features missed
  (oracle selection ceiling ≈ +3.55pp P@15).
- **Non-quad evidence does not exist.** POT and PT are all quads, so generality
  currently rests on unit tests. A real non-quad sequence set with annotation is
  the gating cost for the paper. Conics/ellipses are explicitly out of scope for
  now (user), so the homography-stabilizer identifiability caveat can wait.
- **`docs/correction_research.md` §0/§8 is stale** — still says the alignh runs
  are in flight. Another session is editing that file; coordinate before writing.

## Do not touch

Parallel work is uncommitted in the tree: `docs/correction_research.md`,
`experiments/loma_breakdown_gallery.py`, `experiments/selection_state.py`,
`experiments/run_breakdown_sample.sh`, `flatsam2/corner_covariance.py`,
`tests/test_corner_covariance.py`, two new blocks in `experiments/pipeline.yml`,
and `scripts/hpc_check.py` (a generated bundle that goes stale on manifest
changes — regenerate rather than edit).
