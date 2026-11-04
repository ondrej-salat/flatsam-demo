# Handoff — the candidate final quad-free tracker

Written 2026-08-13, to pick this thread up cold. Companion to
`docs/state-of-the-tracker.md` (what the method is and why) — this file is only
*where things stand and what to do next*.

---

## In flight right now

The `corner_fusion_register_recheck` grids (alias re-check on the final pose,
journal 2026-08-17). POT is in; PTi is still running as of 2026-08-17 16:15.

| what | jobs | state |
|---|---|---|
| `corner_fusion_register_recheck` POT grid | 11361395 (+eval 11361402) | **done**, 210/210 — parity with `_final`, see below |
| `corner_fusion_register_recheck` PTi grid | 11361403 (+eval 11361404) | **running**, 5/20 array tasks at a time (QOSMaxGRESPerUser) |
| `corner_fusion_simple` POT grid | 11362126 (+eval 11362127) | **queued** behind the above |
| `corner_fusion_simple` PTi grid | 11362128 (+eval 11362129) | **queued** |

`corner_fusion_simple` is `_recheck` with the two simplifications that came out of the
2026-08-17 review, both decided offline before spending a GPU hour (journal, same date):
one fusion decision per frame instead of four (`fusion.decision_scope='whole_frame'`,
replayed at +1.20 pp POT / +0.56 pp PTi pooled) and the verification score reduced to the
plain product IoU · (1+NCC)/2 (bench: POT +0.87, PTi +0.02 mean P@15 over regimes). Both
*remove* parameters. Expect the grids to confirm the replay, not to surprise.

An earlier submission of the same pair (11361235 POT / 11361243 PTi) was
cancelled and resubmitted — ignore those job ids.

Finished earlier in this thread:

| what | jobs | state |
|---|---|---|
| `corner_fusion_register_final` PTi grid | 11344539 (+eval 11344540) | **done**, 300/300 |
| `corner_fusion_register_final` POT grid | 11344541 (+eval 11344542) | **done**, 210/210 |
| end-to-end tracker profile, `register_kp` on POT V01_1 | 11343486 | done — see journal 2026-08-13 |
| offline kp re-run **with** `--extra-h-run` | 11343605 | done — reproduces documented numbers |

### POT: the re-check changes nothing, which is the correct outcome

| config | POT P@5 | POT P@15 | br-gap | >5pp | pooled P@15 |
|---|---|---|---|---|---|
| `corner_fusion_register_final` | 91.98 | 96.77 | 0.00 | 0/210 | 97.61 |
| `corner_fusion_register_recheck` | 91.98 | 96.77 | 0.00 | 0/210 | 97.61 |

Identical to two decimals (median error 1.70 → 1.69 px). POT already had a zero
br-gap and few lost frames, so there was nothing on this dataset for an alias
re-check to fix — it had to be neutral or it was harming healthy frames. It is
neutral. **PTi is the dataset that decides the knob**, since that is where the
offline bench put the gain (`lost` 79.3 → 80.1, p90 error when lost 151 → 129 px).

Read the grids with:

```bash
python scripts/hpc_status.py --pipeline corner_fusion
python evaluate.py pti --results-dir $OUTPUT_DIRECTORY --config configs/tracker/corner_fusion_register_final.py
```

### Result: parity was the bar, it cleared it

Per-sequence means (`mean_p@k`, not the pooled `overall_*`); br-gap =
`mean_br_p@15 − mean_p@15`, and `>5pp` is the share of sequences whose own gap
exceeds 5 pp.

| config | PTi P@5 | PTi P@15 | br-gap | >5pp | POT P@5 | POT P@15 | br-gap | >5pp |
|---|---|---|---|---|---|---|---|---|
| `corner_fusion_register` | 67.94 | 86.52 | 2.83 | 11.3% (34/300) | 92.07 | 96.77 | 0.10 | 0.5% (1/210) |
| `corner_fusion_register_kp` | 68.63 | 87.32 | 2.25 | 9.3% (28/300) | 91.70 | 96.65 | 0.00 | 0.0% (0/210) |
| **`corner_fusion_register_final`** | **69.00** | **87.87** | **1.71** | **8.3% (25/300)** | **91.98** | **96.77** | **0.00** | **0.0% (0/210)** |

The three knobs do not change what the tracker decides on a healthy frame, so
parity with `register_kp` was the expectation. PTi came in **+0.55 pp P@15 /
+0.37 pp P@5** above it, the br-gap fell to **1.71** (best of the three) and the
>5pp share to 8.3% — consistent with a real gain on the lost-frame search rather
than noise, since that is the only path the knobs touch. POT recovered the
0.12 pp P@15 that `register_kp` had given up against `corner_fusion_register`, back
to 96.77 with a zero br-gap. No regression anywhere.

`corner_fusion_register_final` is now the reference version of the quad-free
tracker; ablations should be measured against this row.

---

## What `corner_fusion_register_final` is

`corner_fusion_register_kp` (covariance fusion + quad-free propose/refine/verify
+ keypoint proposal) plus two changes that passed the offline dev benchmark:

| knob | value | measured effect |
|---|---|---|
| `register_sdf_roi_margin` | 1.0 | neutral to within 0.2 pp P@15 in every regime, both datasets; **−40 to −46% time** in the healthy regime |
| `register_alias_shortcut` | True | +1.6 pp PTi `lost` on its own, but slower alone |
| `register_sweep_stop_score` | 0.70 | with the shortcut: **+1.1 pp PTi lost/cold at slightly less time** |

Deliberately excluded: `sdf_precise=False` (3×3 chamfer). Another ~20% off the
field and accuracy-neutral, but it produced **one hard failure in 513 POT dev
cases** that the exact transform did not. Kept as ablation `abl_res_sdf_fast`.

---

## Where the time actually goes end-to-end (read this before optimizing anything)

`profile_tracker.py pot --config corner_fusion_register_kp --sequence V01_1`,
74 frames, CUDA-synced, 8 CPU cores — i.e. how the tracker really runs:

| section | ms/frame | share |
|---|---|---|
| `loma.keypoints` | 116.48 | **38.7%** (describe 62.89, prep 34.38, detect 15.94) |
| `sam3` | 96.28 | **32.0%** |
| `loma.match` | 55.99 | **18.6%** |
| `mask_H.register` | 27.30 | 9.1% |
| ↳ `register.prepare` (the SDF) | 7.38 | **2.4%** |
| `loma.ransac` | 1.90 | 0.6% |
| **total** | **301.05** | **3.32 FPS** |

**This corrects the "SDF is 47% of a frame" framing.** That 47% is of the
*isolated mask→H branch measured single-threaded*, because
`experiments/maskh_timing.py` deliberately pins cv2 and BLAS to one thread to
produce quotable per-call numbers. In the tracker, `cv2.distanceTransform` gets
8 cores and drops from 33 ms to 7.4 ms, and the whole registration branch is
only 9.1% of a frame. So:

* The SDF crop is still free and correct, but its **end-to-end effect is ~1-2%**,
  not 40%. It was worth doing; it is not a throughput answer.
* **Throughput is LoMa + SAM3**: 57.3% and 32.0%, together ~89% of the frame.
  The single largest item is LoMa's descriptor pass (20.9%). Nothing in the
  quad-free thread touches any of it.
* Peak GPU memory 6.45 GB.

Any future "make it faster" work should start here, not in `mask_register.py`.

## Why the SDF crop is the interesting one *within the branch*

Profiling (`experiments/maskh_timing.py --stages`) put the cost in one place:

| stage | ms/call | share of a fast-path frame |
|---|---|---|
| `register.prepare` (signed-distance field) | 32.99 | **46.9%** |
| `solve_full` | 9.90 | 22.3% |
| `solve_coarse` | 8.66 | 14.4% |
| `verify_appearance` | 2.43 | 9.5% |
| `verify_overlap` | 0.33 | 4.6% |
| `propose` | 0.54 | 0.2% |

The field was computed over the whole 1280×720 frame but is only ever sampled at
warped template-boundary points, which sit on the mask. Cropping it to the mask
bbox + margin is exactly equivalent when the crop covers the samples (verified:
identical corner error and score to four decimals), and outside the crop it
clamps at the border exactly as the uncropped field already does at the image
edge, with the Huber loss saturating past a few px anyway.

**Corollary worth keeping**: the search is *not* where the time goes. Only ~11%
of frames reach the sweep at all, so search-side optimizations barely move the
average frame. Anything further on speed should target `prepare` or the two GN
solves.

---

## Traps found the hard way (do not re-learn these)

1. **`register_kp` in the offline bench needs `--extra-h-run`.** Without it the
   bench never fills `ctx["extra_H"]` and the method is byte-identical to plain
   `register` — which reads as "the keypoint proposal does nothing". This
   invalidated one dev run in this session. `maskh_bench.py` now refuses to run
   a keypoint method without the flag, and `_make` tags such methods with
   `needs_extra_h`.

2. **`p90_err = inf` in the bench table means "at least one non-finite error",
   not "10% failed"** (`maskh_bench.py:519` uses `inf` as a sentinel when
   `np.isfinite(err).all()` fails). One case in 513 triggers it. Count the
   failures before drawing a conclusion.

3. **`br_p@15 − p@15` (the "br-gap") is a free alias detector** already in every
   `eval_results.json`. Read it before believing any headline from a method that
   re-detects pose. `alignh_redetect` looked fine on 4 probe sequences and lost
   6 pp at grid scale entirely to corner labelling.

4. **`squeue` truncates the NAME column** (`corner_fusion_register_final_pti` →
   `corner_f`), so watching a job by name via `grep` silently matches nothing.
   Watch by job id.

5. **`/tmp` is node-local.** A script written to `/tmp` on the login node is not
   there on the compute node. Use `$HOME` or the scratchpad on shared storage.

6. **The moment proposal is affine, the refinement is projective.** Second-moment
   normalization cannot represent perspective (`cov_mask = A cov_tpl Aᵀ` does not
   hold under a homography); the 8 DOF come from the GN solve over 4 control
   points. Measured consequence: accuracy falls with GT perspective strength, and
   the cold-minus-good gap widens from −4.6 pp (weakest half) to −11.1 pp (top
   5%) — i.e. the *stateless* path is where affine-only initialization hurts.
   The keypoint proposal helps most in exactly that corner (+5.6 pp).

---

## Open threads, roughly in order of expected value

1. **Verify the final config at grid scale** — the jobs above. Until they land,
   nothing else should branch off this.
2. **The residual corner-labelling error**: 2.25 pp br-gap on 9.3% of PTi
   sequences. Unknown whether reducible from mask + keypoint evidence, or
   genuinely ambiguous (near-square outline, near-symmetric texture). A first
   step is to list those sequences and look at them.
3. **Perspective-aware proposals** — the affine limitation above. Unexplored:
   seeding the sweep with vanishing-line-consistent warps, or unlocking the
   projective terms earlier in the coarse solve.
4. **Occlusion**: quad-free is 6.33 pp behind the quad baseline on POT's
   occlusion split. Offline says nothing recovers below 85% mask coverage,
   including the oracle — so this may be segmenter failure misattributed to
   registration. Not yet separated at grid scale.
5. **The four new datasets** — POIC (20), TMT (92), UCSB (96), MPOT-3K (573
   tasks) are wired, verified and unrun. `cross_dataset` pipeline in
   `experiments/pipeline.yml`. MPOT's official eval is the MOT-Challenge MATLAB
   devkit and our per-instance corner predictions concatenate into its
   submission format directly; MATLAB is available (`module avail MATLAB`).
6. **Chase the one chamfer-SDF failure case** if the last 20% of the field cost
   is wanted.

---

## Conventions in play

* Always submit with `--no-mail` (`slurm/submit.py`, `slurm/submit_cmd.py`).
* New experiment → new config stem; the output dir derives from it, so a fresh
  stem cannot collide with an existing run. Never overwrite `OUTPUT_DIRECTORY`.
* Offline first: `experiments/maskh_bench.py` measures a mask→H idea on CPU in
  minutes with initialization as a controlled variable. Tune on `--split dev`,
  report on `--split test`. A GPU grid is for confirming, not exploring.
