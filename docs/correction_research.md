# SAM3 correction & corr/seg fusion — research notes

Living document. Status of the online SAM3 `correct()` line, the controlled
oracle experiments built to debug it, and the reviewed method proposals for the
paper-grade combination of correspondence-based and segmentation-based
homography estimation.

## 0. State of play — plain-language summary (updated 2026-08-07)

*Self-contained; no shorthand. Section references only point at the detailed
evidence.*

The tracker combines two homography sources per frame: a **correspondence
branch** (LoMA keypoint matching + RANSAC — precise, dies under blur / low
texture) and a **segmentation branch** (SAM3 mask → homography — robust,
coarser). The combination rule: use the correspondence branch when it has
≥ 35 RANSAC inliers, else the segmentation branch. Benchmark: PlanarTrack-
initfix (PTi) is the target, POT-210 the regression guard. Decision metric is
**P@15** — the PlanarTrack annotations average ~5.7 px error, so 5-px-level
differences mostly measure annotation noise.

Findings, in order of importance:

1. **Online SAM3 mask correction is a dead end as a metric lever** (§8).
   Even with perfect, GT-derived decisions about *when* to correct and *what*
   mask to inject, the dataset-level gain is ~+1 pp P@15; the deployed
   version gains +0.0. Three independently verified reasons: (a) most output
   frames never route through the segmentation branch, so a repaired mask
   rarely changes the output; (b) the deployed trigger fires when the mask is
   already fine (median mask-IoU 0.975 at fire time) and reaches only half of
   the truly broken sequences; (c) on 88% of frames where the mask IS broken,
   the correspondence branch is broken too — there is nothing to inject.
   The injection mechanism itself works (SAM3 accepts corrections and they
   persist). Deployed correction stays off; the controlled diagnosis is
   paper material.
2. **The big recoverable error mass is in *choosing between* the two
   branches** (§8, A0). A per-frame oracle picking the better of the two
   already-computed homographies gains **+3.3 pp P@15**. Two thirds of that
   sits on frames where the correspondence branch reports many inliers
   (median ~137) yet is catastrophically wrong (median 30 px error) — the
   inlier count cannot detect these, and the current threshold is already
   optimal among inlier rules. Threshold rules over every online signal
   available so far (branch disagreement, mask containment, temporal
   jumpiness, shake-normalized variants) recover only +0.2 pp of the +3.0 pp
   held-out ceiling (§8 verdict 3): **selection needs a genuinely new quality
   signal, not a better threshold.** The catastrophic high-inlier failures
   are under manual investigation — `experiments/loma_breakdown_gallery.py`
   builds the inspection list + visual gallery (sampling mask, matches,
   inliers, failed quad).
3. **Replacing the segmentation branch's Hough-quad step with direct mask
   alignment failed as a blanket swap** (§8 verdict 4, read out 2026-08-03).
   The offline basin study was green — 93% convergence on real SAM masks from
   perturbations far beyond tracking scale, landing at ~3 px — but deployed
   (`configs/tracker/loma_samh_decoupled_alignh.py`) it costs **−2.41 pp PTi
   P@15** (87.86 → 85.45; P@5 65.44 → 62.12; POT flat at 97.66 → 97.37). The
   damage is broad, not a tail: 148 sequences regress against 71 that improve,
   and the worst five carry only 27% of the negative mass, so no gate rescues
   it. The 71 improvements are large (+34.9, +27.8, +18.5), which says the
   aligner is a real *conditional* win applied blind — the same
   branch-selection gap as point 2, one level down. Blanket swap is dead; the
   open salvage is the aligner's boundary residual as a per-branch quality
   feature, which needs a telemetry addition (the residual is not in the npz).

4. **The high-inlier catastrophes have been named** (§9). Manual tagging of the
   breakdown gallery splits them into identity failures (distractor / rotated
   alias / same-looking part — 53% of tagged frames, separable at
   `disagreement > 60 px`) and geometry-or-material failures. Crucially they are
   *episodes*, not frames: most of the mass sits in runs tens of frames long,
   which is why per-frame arbitration self-neutered. Current direction: detect
   the onset and latch (`experiments/selection_state.py`), plus per-corner
   position covariance as the first genuinely new quality signal
   (`flatsam2/corner_covariance.py`).
5. **Per-corner covariance fusion built, calibrated, and run at full scale —
   real progress, success criterion not yet met** (§11, 2026-08-07 to
   2026-08-11). Replaced the hard inlier-threshold switch with per-corner
   precision-weighted fusion of the two branches' own fit covariances.
   Uncalibrated fusion regressed some sequences alignh_pure had gotten right
   (both branches turned out badly overconfident, SAM-H more so — median
   chi-square(2) stat 44/96 vs nominal 1.4, confirmed at n=80 not just n=20);
   an empirical covariance-scale correction fixed that direction cleanly
   (better-or-equal on every probed sequence). Full 300+210 grid:
   **PTi 80.59 p@15 — past published SAM-H SOTA (80.1), but still 7.3 pp below
   the plain-switch baseline (87.86) and the §0 success bar (~88)**; POT 95.99,
   a small regression vs alignh_pure (96.90), root-caused (§11) to a real gap —
   a coasted (unmeasured-this-frame) homography gets zero fusion credit even
   when it is usually still accurate — not the gross-error mechanism point 4
   already flagged as out of scope for covariance fusion.

Priority queue: (1) run the verification debt in §10 — nothing in §9/§10 has
been executed; (2) read out `selection_state.py` and decide whether the latch
survives contact with real data; (3) the covariance plumbing spec in §10.

**2026-08-11 update**: (1) and the covariance-plumbing half of (3) are done
(§11) — corner-fusion is built, calibrated, and confirmed at full scale. (2),
`selection_state.py`'s episode-latch read-out, is still the open item, and is
now sharper: point 5's full-grid PTi gap to the switch baseline (-7.3 pp) is
plausibly mostly the gross-error/episode class point 4 already named, which
covariance fusion cannot fix by construction (see §10 "Two things not to
re-derive" — still true, now with a data point behind it).

Success criterion (agreed 2026-07-08): **improve PTi (p@5 ≈ 65.x, p@15 ≈ 88)
without cratering POT (93.5 / 98.5)**. POT gains are not required. Frame all
read-outs against PTi first; POT is the regression guard.

Metric amendment (2026-07-11): PlanarTrack GT is imprecise — mean annotation
error ≈ 5.71 px, ~10% of annotations exceed 15 px — so **P@15 is the decision
metric**; P@5 differences of a few pp sit at the annotation-noise floor and
are diagnostic only.

## 1. Problem status

`loma_samh` combines LoMA correspondences (precise, orientation-aware, dies
under blur/low texture) with SAM-H (SAM3 mask → Hough quad; robust, boundary-
quantized, orientation-blind). Decoupled variant: hard switch — LoMA when
RANSAC inliers ≥ 35, else the independent SAM-H sub-tracker. The `correct()`
mechanism (when SAM over-segments relative to the LoMA-induced mask, inject
`warp(init_mask, loma_H)` as an authoritative SAM3 conditioning frame) is
net-flat: POT p@5 93.5→93.5, PTi 65.6→65.4 (slightly worse), while the offline
feasibility study (`experiments/sam_overseg_correction.py`) predicted
benefit=2481 vs harm=387 frames. That gap is the thing to explain.
**Resolved 2026-07-11 — see §8: the E-grid attributes it (H1 dominant; trigger
and content both broken besides), and the lever moves to candidate selection.**

The two motivating failure cases:

- **Case 1 — SAM jumps to a distractor.** The superset guard makes correction
  abstain by construction (corr ∩ SAM ≈ ∅ → `outside` ≈ 1). LoMA *can* survive
  a jump (keypoint sampling covers the whole background besides the object), so
  "LoMA confident + SAM jumped" frames exist and are countable (A0), but no
  current trigger addresses them.
- **Case 2 — SAM spills from the planar face to the whole 3D object.** Correct
  from SAM's point of view (it saw only the face at init; the object turned out
  to be a box); we must further specify "only this face". This is the clean
  improvement story and the only case the current trigger targets.

## 2. Why correction is probably flat — hypotheses

- **H1 (headroom).** The causal path from a better SAM mask to the final metric
  is narrow: the output is LoMA-H whenever inliers ≥ 35 (most frames; POT LoMA
  alone = 93.2 vs 93.5 combined). Correcting SAM changes the output only via
  (a) SAM-H-selected frames, (b) the LoMA keypoint-sampling support, (c) future
  SAM-H state. The offline study counted frames where corr beats SAM — but on
  most of those the output was already LoMA-H, so fixing SAM was
  metric-invisible.
- **H2 (mechanism).** SAM3 ignores or reverts corrections: the memory bank
  still holds whole-box conditioning + non-cond memories; one corrected frame
  gets outvoted later. Testable via post-correction mask-agreement trajectories
  and re-trigger rates.
- **H3 (trigger).** The trigger fires rarely / at wrong times. The strict
  superset guard (`tol=0.02`) "almost never fires because the SAM3 mask is
  rarely a clean superset" (the `relaxed` config docstring); when it does fire
  it may be on frames where both branches are wrong.
- **H4 (content).** `warp(init_mask, loma_H)` is bad content: paints over
  occluders, boundary off by the LoMA error, fights SAM's own boundary sense —
  and it is injected as *authoritative* conditioning, so bad content poisons
  memory (the PTi regression).
- **H5 (case 1 unaddressed).** See above — jumps are outside the current
  trigger's reach by construction.

## 3. Diagnosis framework: the containment algebra

Treat both branches as measurements of one latent state H_t. The mask measures
the plane's image support *one-sidedly*: the plane's projection must lie
**inside** the object mask. Divergences then decompose without special cases:

| one-sided mass | reading | right reaction |
|---|---|---|
| SAM − plane large (spill) | object has other faces | fix seg (case 2) |
| plane − SAM large (deficit) | occlusion / partial loss | trust seg, don't "correct" |
| both large / IoU ≈ 0 (jump) | seg tracks something else | redetect (case 1) |

The deployed trigger already lives in this algebra (superset guard + one-sided
excess depth); the telemetry now records all of its components every frame, so
any trigger rule is evaluable offline from one run.

## 4. Branch A — controlled experiments (implemented)

### A0. Offline error budget — `experiments/correction_headroom.py`

No rerun needed; joins the existing `loma_samh_decoupled_rerun` sidecars with
GT and the precomputed SAM masks. Outputs (stdout tables + HTML + CSVs):

- selection stats + error conditional on selection;
- within-run **selection ceiling**: P@5/15 of the run vs per-frame best stored
  candidate (`loma_H`/`samh_H`/`mask_H`) — headroom of any smarter selection;
- **error-mass decomposition** by (selected branch × mask state
  ok/spill/shrink/both/jump); the "spill × samh-selected, LoMA-correct" cell is
  the *only* mass an over-seg correction can recover — H1 quantified;
- per-sequence headroom + `shortlist_top20.txt` (feed to
  `run.py --subset-file` for cheap single-sequence oracle iteration).

Cross-run complement (zero code): `oracle_ensemble.py` over the LMr-only and
SAM3_H result dirs gives the standalone-methods complementarity ceiling.

### A1. Oracle grid — configs + tracker plumbing (implemented)

GT homography `H_gt(t) = homography_from_corners(gt[0], gt[t])` is plumbed
per-frame by `run.py` into `tracker.track(frame, oracle_H=...)` whenever the
correction config needs it (invalid-GT frames → None → oracle legs abstain).
Gating rule: *the inlier gate applies to a leg iff that leg's source is
"loma"*; oracle legs need only valid GT; `periodic` needs neither.

| cell | config (`configs/tracker/`) | trigger | content | isolates |
|---|---|---|---|---|
| E0 | `loma_samh_decoupled_rerun` (exists) | — | — | baseline |
| E0+ | `..._correct_decoupled_dryrun` | deployed, dry-run | — | counterfactual trigger stats on unperturbed masks |
| E1 | `..._correct_decoupled_oracleTC` | vs GT mask | GT mask | **mechanism upper bound** |
| E2 | `..._correct_decoupled_oracleT` | vs GT mask | LoMA mask | content quality |
| E3 | `..._correct_decoupled_oracleC` | deployed | GT mask | trigger quality |
| E4 | `..._correct_decoupled_excessdepth_rerun` | deployed | LoMA mask | current method, fresh telemetry |
| E5 | `..._correct_decoupled_periodic30_oracleC` | every 30 frames | GT mask | continuous coaching; persistence |

Decision tree on the results:

- **E1 flat** → the `segmenter.correct` primitive is the bottleneck (H2).
  Next knobs already exist in the sam3 wrapper:
  `clear_recent_memory_on_correct`, `accumulate_corrections`. If E1 *hurts*,
  authoritative mask re-prompting is the wrong primitive altogether.
- **E1 good, E2 bad** → content is the bottleneck (H4) → better content
  (fused/joint H; carve occlusions; see §5).
- **E1 good, E3 bad** → trigger is the bottleneck (H3) → redesign from the
  E0+/A0 trigger ROC (thresholds, or a GT-side-calibrated rule).
- **E5 ≫ E1** → episodic correction is the wrong paradigm; SAM3 needs
  continuous conditioning, not interventions.
- **E1 good but final metric unchanged** → H1 confirmed: the seg branch barely
  feeds the output; correction is not the lever — selection/fusion is.

### A2. Event-conditional analysis — `experiments/correction_events.py`

Pairs any E-run with the baseline, aligned on correction events: Δerr(t0+k)
median/IQR curves (k = 0..50), post-event `corr_iou_gt` persistence (does the
correction stick or does SAM3 drift back — H2), re-trigger interval histogram,
per-sequence ΔP@5 vs #corrections. Global averages hide sparse events; these
don't.

### Telemetry (implemented; every correction-enabled run)

`{seq}_debug.npz` gains per-frame `corr_spill_{loma,gt}`, `corr_excess_{loma,gt}`,
`corr_area_ratio_{loma,gt}`, `corr_iou_gt`, `corr_loma_gt_err`, `corr_trigger`
(fired; `corr_trigger & ~corrected` = abstain/dry-run). NaN when a source is
unavailable. Old runs' `corrected` flags are stale (old binary) — analyze
reruns only.

## 5. Branch B — method proposals, reviewed (verdicts 2026-07-08)

### Active: P3′ — mask-alignment SAM-H, reframed

Direct DT/ICP-style alignment previously went badly: homography-space
alignment fights symmetries, occlusions, spills, and small mask inaccuracies.
Reframed approach:

1. **4-point parameterization only** (control-point offsets in pixels) — never
   raw 3×3 coefficients; isotropic, pixel-unit steps.
2. **Convergence-basin study first** (`experiments/mask_alignment_basin.py`,
   implemented): perturb H_gt with *structured, attributable families* —
   translation, wide rotation (±180°, probing capture by wrong symmetry
   poses), scale, shear, perspective tilt, iid corner noise, realistic mix —
   each over a magnitude grid; run a damped-GN Huber aligner on the target
   mask's signed distance field; classify **correct / symmetric-capture /
   diverged** per trial (`best_rotation_error` separates the symmetry trap).
   `--target gt` isolates the aligner's own limits from SAM boundary noise.
   The basin radii vs the actual init quality (last-H / LoMA-H — measurable
   from A0) decide viability.
3. **If classical basins are too small → small learned aligner**: two masks in,
   4-pt correction out, **multi-hypothesis** head for symmetric targets. Note
   circular targets carry a *continuous* rotation symmetry — the output must be
   a distribution/canonicalization, not just k hypotheses. The basin generator
   doubles as training-data synthesis.
4. Payoff if it works: drops the quadrilateral assumption, deletes Hough
   top-4-lines + cyclic intersections + bruteforce corner matching +
   convex-quad checks, and (with continuous last-H init) the DINO symmetry
   subsystem. Redetection-after-loss moves to LoMA, which is orientation-aware
   — a clean division of labor.

### Parked: P4 — one joint objective instead of the thr35 switch

H_t = argmin_H Σ_i ρ(‖x′_i − H·x_i‖/σ_c) + λ Σ_{x∈∂T(H)} ψ_asym(d_S(x)/σ_m):
robust correspondence term + asymmetric-robust mask-alignment term (boundary
noise both directions; an outlier/spill process only in the mask-exceeds-
template direction — asymmetry prevents shrink-collapse without tolerating
spill). Deletes the hard switch; degradation continuous; λ from noise scales,
not tuning. **Parked**: elegant but hard to debug (everything mixed opaquely),
low prior hopes. Revisit if P3′ produces a working mask term (its residual is
exactly the mask likelihood this needs).

### Rejected (kept for the record — negative knowledge)

- **P2 candidate selection by geometric consistency.** Empirically the 3 SAM
  candidates are near-identical, or one good + garbage (holes, noise,
  self-IoU ≈ 0 — the network effectively declined); two clean granularities
  (face vs whole box) essentially never co-occur. Nothing to select between.
- **P5 point-prompt correction.** SAM is highly sensitive to the number of
  correction points and behaves erratically; choosing +/− points is hard; the
  streaming wrapper would need new APIs. No.
- **P6 per-pixel plane-membership field (generative model, EM).** Too finnicky.
- **P7 sampling-support decoupling (union with last-H-warped mask).**
  Unnecessary: LoMA already samples the whole background besides the object, so
  it can re-catch a well-textured target after a SAM jump; and the last-H warp
  is no anchor precisely when jumps happen (after long occlusion/out-of-view
  with rotation).

## 6. Paper story + risks

Near-term story rides on Branch A: a controlled diagnosis of when/why the two
homography sources disagree (containment algebra: spill/deficit/jump), an
oracle-grid attribution of the correction mechanism (trigger vs content vs
SAM3-memory dynamics), and whichever repair the evidence selects — memory
knobs, trigger redesign, or P3′ replacing Hough SAM-H. Risks: (1) if E1 is
flat/hurts, mask re-prompting is the wrong primitive and the diagnosis itself +
selection improvements become the contribution; (2) benchmarks are quad
targets by construction — the arbitrary-shape claim needs a qualitative demo
elsewhere; (3) events are sparse — report event-conditional metrics, never just
aggregates.

## 7. How to run everything (cluster)

One driver, all defaults baked in (`DATASET=pti` default; `DATASET=pot`,
`OUTPUT_DIR=...`, `JOBS=...` env overrides):

```bash
experiments/run_correction_grid.sh status   # what is done / partial / missing
experiments/run_correction_grid.sh submit   # SLURM-submit every missing E-cell run
experiments/run_correction_grid.sh eval     # evaluate.py for each finished config
experiments/run_correction_grid.sh analyze  # A0 headroom + A2 event reports
```

Checklist / order of operations:

1. `pytest tests/` — the new unit tests must pass
   (`test_correction_oracle.py`, updated `test_correction_telemetry.py`,
   `test_config.py`).
2. Smoke one sequence before the full grid:
   `python run.py pti --config configs/tracker/loma_samh_correct_decoupled_oracleTC.py
   --sequences <seq> --save-masks --no-skip-completed --fail-hard`
   → expect `masks/<seq>/*_corr.png` and `corr_*` keys + True `corrected`
   entries in `<seq>_debug.npz`.
3. `submit`, wait, `status` until all cells complete (re-run `submit` any time —
   it skips complete configs; `run.py` skips complete sequences).
4. `eval`, then `python collect_results.py` for the cross-config table
   (add the new stems to a report view when needed).
5. `analyze` → read, in order: A0 `summary.json` + shortlist (H1 headroom),
   E1-vs-E0 events report (mechanism), E2/E3 (content vs trigger), E5
   persistence. Walk the §4 decision tree.
6. Optional / independent: the P3′ basin study —
   `python experiments/mask_alignment_basin.py dump && python
   experiments/mask_alignment_basin.py report` (add `--target gt` for the
   aligner-only upper bound; `--frame-stride`, `--families`, `--repeats` to
   scale cost).
7. A0 headroom dump is mask-IO heavy; to keep it off the login node:
   `python slurm/submit_cmd.py --job-name headroom-dump -- python
   experiments/correction_headroom.py dump`.

Implementation map (this round): `flatsam2/config.py` (`CorrectionCfg`
sources/periodic/dry_run), `flatsam2/tracking/common.py` (`CorrectionTelemetry`,
`_correction_telemetry`, `TrackerState.frame_idx`), `flatsam2/tracker.py`
(`track(oracle_H=)`, `_maybe_correct_sam3` per-leg sources), `run.py`
(`_gt_homographies`, GT plumbing, npz schema), 6 configs, `experiments/
correction_headroom.py`, `experiments/correction_events.py`,
`experiments/mask_alignment_basin.py`, `experiments/run_correction_grid.sh`,
tests, `experiments/pipeline.yml`.

## 8. E-grid read-out (2026-07-11, PTi; P@15-primary)

All cells complete; A0 + A2 analyzed from the exported reports, re-derived
under the P@15 metric amendment (§1). Decision-tree resolution: **H1 (narrow
causal path) is the dominant effect; H3 (trigger) and H4 (content) each
independently destroy most of the remaining oracle gain; H2 (SAM3 reverts
corrections) is rejected — corrected masks stick.** Correction is
positive-but-small where it fires; per-frame candidate selection holds ~3.7×
more P@15 mass.

### Final-metric deltas vs E0 (per-seq mean, pp)

| cell | events | seqs w/ events | ΔP@15 all | ΔP@15 corrected seqs | ΔP@5 all |
|---|---|---|---|---|---|
| E1 oracleTC | 408 | 115 | **+0.97** | +2.71 | +0.57 |
| E2 oracleT (LoMA content) | 298 (+2245 abstains) | 59 | +0.12 | +1.98 | −0.11 |
| E3 oracleC (deployed trigger) | 278 | 63 | +0.30 | +2.48 | +0.04 |
| E4 deployed (excessdepth) | 159 | 65 | +0.02 | +1.11 | −0.22 |
| E5 periodic30, GT content | 4270 | 300 | **+1.64** | +1.64 | −1.21 |

Event-conditional facts (A2, `correction_events` reports; IoU/Δerr statements
are threshold-free, so unaffected by the metric amendment):

- **E1 — mechanism works, coverage is the wall.** Corrected masks stick:
  median IoU_gt 0.735 at the event → 0.89 at k=1 → 0.94 at k=50; median
  retrigger interval 8. Paired Δerr −1.6 px at k=1 decaying to −0.04 px by
  k=50. Corrected sequences gain +2.71 pp P@15, but only 115/300 sequences
  ever qualify → dataset +0.97 pp. The mask→metric path is narrow (H1).
- **E3 — trigger fires when little is wrong (H3).** Deployed trigger's events
  sit at median IoU_gt **0.975**; the oracle trigger fires at 0.735 and
  reaches 115 sequences vs 63. 60% of re-triggers within 2 frames (bursts).
  Keeps +0.30 of E1's +0.97.
- **E2 — content starved (H4, sharpened).** With the oracle trigger, LoMA
  content is unavailable on **88% of trigger frames** (2245/2543 abstain):
  SAM and LoMA fail together, so `warp(init_mask, loma_H)` has nothing to
  offer exactly when needed. When available it barely lifts the mask
  (IoU 0.727 → 0.78 at k=2) and re-triggers immediately (median interval 1).
  Keeps +0.12 of +0.97.
- **E4 — fixes masks, not the metric.** IoU_gt 0.834 → 0.94 after events and
  persists; Δerr ≈ 0; ΔP@15 +0.02 overall (+1.11 on its 65 corrected seqs,
  dragged by occasional big losers, e.g. Seq_00341 −19.3). Event-level proof
  of H1. The earlier "actively harmful" reading (−0.22 P@5) was at the
  GT-noise floor — at the decision metric E4 is *flat*, not harmful.
- **E5 — continuous coaching is the best cell at P@15 (+1.64), high
  variance.** The P@5 harm (−1.21) is fine-alignment jitter below GT
  precision. Bimodal at P@15: 38 seqs > +5 / 18 seqs < −5 (extremes
  Seq_00350 +51, Seq_00276 −59); event-local Δerr ≈ 0 → gains are
  regime changes (sequences that avoid catastrophic loss), not local nudges.
  Oracle-content-only, hence undeployable as-is (E2: no real content source),
  but it shows continuous conditioning > episodic intervention for
  catastrophic-loss rescue.

### A0 headroom at P@15 — where the mass is

`correction_headroom` (133,320 eval frames): run P@15 87.68 → per-frame
candidate-selection oracle **91.23 (+3.55 pp)** (P@5: 65.6 → 75.8). Of 16,425
P@15 misses, 4,733 have a stored candidate ≤ 15 px: **3,207 (68%) are
loma-selected frames where samh_H was fine** — catastrophic LoMA breakdowns
passing the inlier gate (median err_loma 30 px, q90 487 px; median 137
inliers; 97% noise-robust with margin ≥ 5 px; spread over 137 sequences) —
plus 1,526 (32%) the reverse direction. The inlier sweep peaks exactly at the
deployed threshold 35 (P@15 87.68; X=200 → 86.58): **the gate is already
inlier-optimal; selector gains must come from other online signals.** The
static over-seg-correction cell (spill × samh-selected × loma ≤ 15) is 142
frames; the feasibility study's benefit=2481 counted frames whose output never
routes through SAM (H1, quantified). samh×ok misses at 15 px = 4,784 frames
(mask fine, Hough quad > 15 px off) — P3′'s direct target, robust at P@15.

Overlap check: 10/17 of E1's big winners (ΔP@15 > +5) also carry ≥ 10 static
selection-rescue frames; the largest correction wins (Seq_00350 +52) are
mostly *dynamic* (early SAM-state repair prevents later loss), which static
per-frame accounting undercounts on both sides. Seq_00791 (+27, zero
selection headroom) marks a small population only correction can save.

### Verdicts (P@15-primary)

1. **Remove deployed correction from the main line.** +0.02 pp at the
   decision metric, occasional −19 pp sequences, real complexity; its
   all-oracle ceiling is +0.97. Keep the always-on telemetry + dry-run config
   (selector-feature substrate).
2. **Correction line capped; the controlled diagnosis is the paper content**
   (trigger at IoU 0.975 + half the coverage, H3; content absent on 88% of
   true triggers, H4; masks fixed ≠ metric moved, H1; corrected masks
   persist, H2 rejected).
3. **New primary: S1 selection rule** — `experiments/selector_study.py`
   (pipeline `selector_study`): online features against the +3.55 pp P@15
   ceiling on the E0+ dry-run substrate; no GPU; rules picked and ranked by
   P@15 on a sequence-level train/test split. Simple rules ≥ +1.5–2 pp
   held-out P@15 → deploy; else this becomes the P4 mask-likelihood
   motivation.
   **S1 v1 result (2026-07-11): negative.** Threshold stumps over
   {inliers, disagreement, spill_loma, excess_loma, area_log_ratio} do not
   beat the baseline switch (test P@15 87.92 = baseline; oracle 90.93; best
   feature AUC 0.66). Root cause: disagreement is symmetric — it flags *that*
   the branches diverge, not *who* broke, and both failure tails live at high
   disagreement. **S1 v2** added per-branch temporal-smoothness features
   (`loma_speed`, `samh_speed`, `speed_gap`), shake-normalized excess
   variants (speed minus own lagged rolling median), and a two-sided
   smoothness-arbitration family.
   **S1 v2 result (2026-07-11): negative for deployment, useful
   diagnostically.** Best held-out rule (`speed_gap > 3.5 AND
   loma_speed > 10.4`) = **+0.20 pp** test P@15 (88.12 vs baseline 87.92;
   oracle 90.93). Feature ranking at the 3 px label margin: n_inliers 0.726,
   speed_gap 0.703, spill_loma 0.686 AUC — the signals *rank* frames but not
   sharply enough for a hard switch against the asymmetric cost of false
   switches (the rescue population is 2.4% of frames). Shake-normalized
   excess speeds were WORSE than raw (multi-frame breakdowns contaminate
   their own rolling baseline; AUC 0.57); the smoothness arbitration
   self-neutered (optimal τ → ∞, 0.1% changed frames). **Simple online
   threshold rules are exhausted.** Escalation: (a) the P3′ aligner residual
   as a new per-branch quality feature — still open, still needs the telemetry
   addition, and now the *only* surviving use of the aligner (verdict 4);
   (b) the breakdown gallery (`experiments/loma_breakdown_gallery.py`) to name
   the high-inlier catastrophic-LoMA failure mode before designing any further
   signal — **done 2026-08-07, see §9**; (c) failing both, the
   learned-selector / P4 route.
4. **P3′ — the basin study was green, the deployed run is NEGATIVE.**

   **Result (read out 2026-08-03; runs finished 2026-07-11).** `alignh` vs
   `loma_samh_decoupled_rerun`, per-sequence means: **PTi P@15 87.86 → 85.45
   (−2.41)**, P@5 65.44 → 62.12, branch-P@15 89.43 → 88.09; POT flat
   (97.66 → 97.37, branch +0.16). Per-sequence: **148 regressed / 71 improved /
   81 unchanged**, negative mass −937.7 pp with the worst five holding 27% of
   it — **broad damage, not gate-rescuable**. But the 71 improvements include
   +34.9, +27.8, +18.5, so the aligner is a genuine per-frame conditional win
   applied unconditionally: the same branch-selection-signal gap A0 identified,
   one level down. **Blanket SAM-H swap is dead.** Open salvage: the align
   residual as a selector feature — needs a telemetry addition, the residual is
   not currently in the npz. Read-out tooling: `scripts/hpc_readout.py`.

   The basin study below stands as measured and explains the gap: it predicted
   the aligner converges, and it does. What it could not predict is what
   happens on the frames where the *mask* is wrong rather than the quad, or how
   often the Hough quad was accidentally right. Convergence was the wrong
   success criterion for a blanket swap.

   Basin study, unchanged (2026-07-11, `--target sam`, stride 60 →
   2,324 frames; τ=15 re-read).** At the decision metric the picture is
   clean: **92.7% convergence on clean masks (IoU ≥ 0.7, 95.5% of frames) at
   small perturbations, median landing error 3.2 px** — the τ=5 read's "28%
   failure floor" was the below-GT-noise artifact (same trap as P@5). Basin
   widths (clean, τ=15): ≥ 86% correct out to ±20 px translation / ±20°
   rotation; 84–87% at ±40% scale/shear/tilt; 83% at σ=20 px corner noise.
   The symmetry trap is sharply localized and *reliable*: ≥ 45° rotations
   convert to symmetric captures (90°: 68%, 180°: 92% land on a cyclic pose)
   — so with external orientation disambiguation (LoMA / DINO vote over the 4
   cyclic starts), even large-rotation redetection is plausible later.
   Degraded masks (IoU < 0.7, 4.5% of frames, median final err ~55 px) never
   align — the aligner fixes wrong quads, not wrong masks. Remaining clean
   floor ~7%: decompose with the `--target gt` control only if the prototype
   underdelivers. **Prototype implemented**:
   `flatsam2/mask_align.py` (`align_mask_to_homography`, damped-GN port of
   the basin aligner), `hough.method='align'` config knob (+ `align_*`
   parameters, Hough as rejection fallback via `align_residual_ok_px`),
   wired inside `_compute_mask_h` so all algorithms including the decoupled
   SAM-H sub-tracker inherit it; config
   `configs/tracker/loma_samh_decoupled_alignh.py`; tests
   `tests/test_mask_align.py`. Run: pytest, single-seq smoke, then the config
   on PTi + POT guard; compare against `loma_samh_decoupled_rerun` at P@15
   (target population: 4,784 samh×ok misses → ceiling ≈ +3 pp).
5. Open, not urgent: the E5 population (continuous-conditioning rescue,
   +1.64) is blocked on a content source; revisit only if P3′ yields a
   trustworthy mask→H that could *become* correction content.
6. S1 v2 addendum (user note): PTi cameras shake — raw per-branch speed is
   not a clean signal. v2 therefore also derives, at report time,
   `*_speed_exc` = speed minus the branch's own lagged rolling median
   (causal, window 15) and runs the arbitration family on both raw and
   excess speeds; shake lifts both branches' baselines, the excess does not
   move.

## 9. Breakdown-gallery tags → stateful selection (2026-08-07)

Manual failure-type annotation of the breakdown gallery (121 frames / 56
sequences of the 3,176-frame, 142-sequence PTi list; the top of the
inlier-descending ranking, so **class counts here are not class shares**).
Tags: distractor, submatch, orientation, ok, screen, transparent, mirror,
tless, bad distribution, blur, background motion, weird.

### What the tags say

**Two populations, not one.** Identity failures (distractor 61, orientation 2,
submatch 1 = 53% of tagged frames) — LoMA locks onto a different instance, a
same-looking part, or a rotated alias — sit at median err_loma 348 px and
median disagreement 342 px. Everything else (geometry / material: bad
distribution, blur, tless, screen, transparent, mirror, background motion) sits
at median err_loma ≈ 28 px, disagreement ≈ 20 px. `disagreement > 60 px`
separates them at TP 63 / FP 2 / FN 0 on the tagged set and flags 36% of the
full breakdown list (it agrees with `err_loma > 100` on 96% of frames).

**Why S1 v2's arbitration self-neutered.** Breakdowns are *episodes*. Grouping
the breakdown list into runs of consecutive frames: identity-tagged sequences
carry 795 breakdown frames in 155 runs, 65% of that mass in runs ≥ 10 frames
(longest 161); geometry-tagged sequences 910 frames in 335 runs, 43% in runs
≥ 10. The temporal-smoothness evidence exists only on the *onset* frame — once
the tracker has sat on the distractor for ten frames, both branches move
smoothly again (distractor-tagged frames: median loma_speed 15.1 vs 14.0 for
the whole list). A per-frame arbiter therefore had to be right on every frame
of an episode while its evidence existed on the first one only, so the optimal
threshold ran to infinity. **Fix: detect the onset, then latch** — one correct
detection rescues the whole episode. Implemented as `experiments/selection_state.py`
(pipeline `selection_state`): episode run-length structure, onset-vs-sustained
feature AUCs, and a two-state rule family (enter distrust on `disagreement >
τ_hi` optionally gated by "the LoMA quad is the one that jumped", leave after
k frames below τ_lo), scored on the same sequence-level train/test split at
P@5/P@15/P@30, against two oracle-episode bounds (perfect episode boundaries
with always-samh and with a per-episode majority vote).

**GT-noise contamination of the ranking.** The `ok` tag (LoMA fine, GT wrong or
undecidable) covers 10 frames, and those have the *highest* median inlier count
of any class (526 vs 107 for the whole tagged set) — the inlier-descending
ranking's head is enriched for annotation error. No attempt is made to label
bad GT automatically (manual annotation is too costly): the control is the
metric itself, so every read-out from here reports **P@30 next to P@15**. A
rule that only trades annotation noise moves the two differently.

**Half the taxonomy is a sequence property.** screen (3 seqs), transparent (3),
mirror (3), tless (11): the target's material, covering ~23% of breakdown
frames by sequence propagation. Not reachable by any per-frame selector. Not
pursued as an up-front target-type classifier either — the properties are
*time-varying* (a screen showing static content, a reflective surface that only
misbehaves at certain viewing angles) and single-frame recognition of
mirror/transparent is unreliable, with too little data to train on without
overfitting to these sequences. If it becomes online at all it should be an
accumulating per-sequence statistic, i.e. the same latch machinery with a
longer time constant. Paper use for now: stratify PTi results by target type.

### Per-corner uncertainty: one measurement type for both branches

"bad distribution" (19 frames / 15 seqs, err ≈ 32 px, median 45 inliers) is a
real class, but the naive fix — penalize spatially concentrated inliers, as the
April 2026 `spatial_coverage.py` line did with `hull_coverage` — is wrong: under
heavy occlusion, correspondences on the small visible part give genuinely
reliable homographies and must not be suppressed. The distinction is not
coverage but *lever arm*: how far the corners are extrapolated beyond the
evidence, and how well-conditioned that evidence is.

Both branches can report the same thing — each predicted corner with a 2×2
covariance in pixels (`flatsam2/corner_covariance.py`, tests in
`tests/test_corner_covariance.py`):

- Segmentation branch: TLS fit of the two boundary lines meeting at the corner →
  angle/offset uncertainties → intersection Jacobian. Ported from the old
  flatsam `experiments/corner_confidences.py`.
- Correspondence branch: least squares on the inliers' reprojection residuals,
  in the four-corner parameterization `mask_align` already uses. The eight
  parameters *are* the corner positions, so `σ² (JᵀJ)⁻¹` is already the corner
  covariance — no second propagation, one linear solve per frame, deterministic.
  `bootstrap_corner_covs` re-fits on resampled correspondences and is kept as the
  offline cross-check only (~100× the cost, same answer to first order, but a
  different modelling assumption — resampling the inlier set rather than
  conditioning on it).

**Bug found in the ported line covariance.** The prototype assembled the line
parameter covariance as `diag(σ_φ², σ_ρ², σ_φ², σ_ρ²)`. The two fit
uncertainties are independent in the line's own frame (angle pivots about the
centroid, offset shifts perpendicular to it) but *not* in the normal form, where
`ρ = -n(φ)·centroid - s` picks up a lever arm equal to the centroid's distance
from the image origin. Treating them as independent double-counts the pivot: the
corner σ came out 4.7× too large for a well-conditioned corner and 34× for short
boundary arcs, growing with distance from the origin, so not even a constant
factor. Fixed by transforming the own-frame covariance into `(φ, ρ)` before
propagating. After the fix the analytic covariance matches Monte Carlo to within
1% in every regime tested (orthogonal, 15° shallow, and far-extrapolated
corners), and its Mahalanobis error is χ²₂-calibrated (median 1.34 vs 1.386,
5.4% above the 95% quantile). The prototype's own bootstrap cross-check could
never have caught this — it passed points where line parameters were expected.

What this is for, in order:

1. **Calibration on real data** — the same χ²₂ check per branch against GT
   corners. Both are expected to be over-confident in the field for known
   reasons: the bootstrap shrinks like 1/N with hundreds of inliers, and the
   segmentation branch's dominant error is systematic mask error (spill,
   occlusion, wrong object), not zero-mean boundary noise. A scalar inflation or
   a heavier-tailed model is a fine answer, but it has to be measured.
2. **Normalized disagreement** — the Mahalanobis distance between the branches'
   corner estimates under the sum of their covariances, replacing the hand-tuned
   60 px in the `selection_state` rule family with a scale-free χ² statistic.
3. **Fusion** — only if 1 and 2 hold up.

Hard boundary, to be stated plainly in the paper: covariance is a *precision*
model. A distractor produces a tight, confident, wrong ellipse, and the corner
labelling is presumed correct so the orientation-permutation failure is outside
the model too. Together those are the majority of the breakdown mass. Gross
errors stay the job of the episode-level rejection above; the uncertainty model
is what makes the surviving frames precise and the disagreement test principled.
Validation of the occlusion behaviour on real data needs an `occlusion` tag,
which is why it is now seeded into the gallery palette.

### Next annotation round

`experiments/run_breakdown_sample.sh` (pipeline `loma_breakdown_sample`):
refreshes the ranked list, draws a seeded random sample stratified over
disagreement deciles with a per-sequence cap, and submits the GPU render. Known
per-stratum sampling rates → its tag counts scale back to population shares,
which the ranked list's cannot. Palette pre-seeded (round-1 tags + `occlusion`).

## 10. Handoff to the HPC session (2026-08-07)

Everything below is committed and arrives via `make sjo_sync`. Nothing here has
been executed — the workstation has no environment, so **every claim marked
"unverified" is exactly that**. Verify before building on it.

### Verification debt (do this first, it is cheap)

```bash
python slurm/submit_cmd.py --gpus 0 -- pytest tests/test_corner_covariance.py
```

Six tests. The four line-covariance ones were checked numerically against Monte
Carlo on the workstation (pure numpy, no package import). The two that touch
`cv2` — `test_homography_cov_matches_the_bootstrap` and
`test_homography_cov_is_tighter_for_well_spread_correspondences` — have **never
run**. The first is the load-bearing one: it pins the analytic corner covariance
against resampling, which is the check whose absence let a factor-of-five
parameterization bug survive in the old flatsam prototype (§9).

Then the two new offline studies, both login-node-safe in principle but Python,
so both go through a partition:

```bash
python slurm/submit_cmd.py --gpus 0 -- python experiments/selection_state.py
python slurm/submit_cmd.py --gpus 0 -- experiments/run_breakdown_sample.sh status
```

`selection_state.py` takes no arguments and reads the existing `selector_study`
per-sequence CSVs. Its logic was exercised on synthetic CSVs on the workstation;
its numbers on real data are unknown. It answers whether the "detect the onset,
then latch" hypothesis (§9) survives contact with the data — specifically
whether the two oracle-episode bounds leave enough room to be worth chasing.

### The open work item: covariance plumbing

Design and rationale in §9; storage decisions settled with the user:

- **Corner covariances: always on**, both branches. 4 corners x 3 numbers
  (symmetric 2x2), float32 = 96 B/frame, ~13 MB for the whole PTi split.
- **Inlier correspondences on a fixed-stride sample: always on.** Every 25th
  frame, ~5.4 k frames, ~12 MB. Fixed stride rather than a random draw so it is
  reproducible and spread over every sequence.
- **Inlier correspondences on all frames: behind a config flag.** ~315 MB for
  PTi; the number to watch before enabling it for POT as well.
- Coordinates as **uint16 fixed-point at 1/16 px**, not float32. Quantization
  variance 8e-5 px^2 against 1-3 px residuals, so no covariance changes and the
  cost halves. float16 would be a mistake: 0.5 px spacing at coordinate 1000.
- Ragged per-frame counts go into the existing `{seq}_debug.npz` as one
  concatenated `(M, 4)` uint16 array plus a per-frame offset index.

Call sites to open up:

- `flatsam2/tracking/loma_samh.py` — `_estimate_loma_H` returns `(H, n_inliers)`;
  it also needs to hand back the inlier correspondences so
  `homography_corner_covs` can be called on them.
- `flatsam2/hough_homography.py` — `refine_lines` already returns per-line point
  lists; `mask_to_homography` drops them. Thread them out to reach
  `line_intersection_cov` for the two lines meeting at each corner.
- `flatsam2/config.py` — the all-frames dump flag.

Then the sequence in §9: calibration (chi-square-2 check per branch against GT
corners), normalized disagreement into `selection_state`'s feature set, and only
then any fusion.

### Two things not to re-derive

**Covariance is a precision model, not a robustness model.** A distractor
produces a tight, confident, wrong ellipse, and the corner labelling is presumed
correct so the orientation-permutation failure sits outside the model too.
Together those are the majority of the breakdown mass (§9). Gross errors stay
the job of episode-level rejection. Do not let the two mechanisms merge.

**The offline re-selection studies bound a re-ranking, not a re-run.** Both
`selector_study.py` and `selection_state.py` replay stored per-frame candidates,
so neither can model how a different choice would have changed the tracker's own
downstream state. Any headroom number from them carries that caveat.

## 11. Corner-fusion: built, calibrated, run at scale (2026-08-07 to 2026-08-11)

Picked up the §10 handoff's open item (covariance plumbing) and carried it
through to a full-grid result. Session-by-session detail lives in
`docs/2026-08-07.md` and the terse index in `docs/journal.md`; this section
is the settled state for anyone starting from this document.

### What was built

- **Per-corner fit covariance for both branches**, same currency
  (`sigma^2 (J^T W J)^-1` at the converged point of each branch's own fit):
  `homography_corner_covs` already existed for LoMa's RANSAC correspondences;
  `mask_align._corner_cov_from_boundary_fit` adds the analogue for the
  align-method boundary-SDF residual (`align_mask_to_homography` now returns
  a 5-tuple with the covariance). Every returned covariance is floored at a
  0.25px credible std (`corner_covariance._floor_cov_eigenvalues`) — a
  near-singular fit otherwise reports near-zero uncertainty on one axis,
  which blows up any downstream Mahalanobis distance to numerical nonsense.
- **`algorithm='corner_fusion'`** (`flatsam2/tracking/corner_fusion.py`,
  `cfg.loma.fusion=True`): replaces `loma_samh`'s hard 35-inlier switch with
  a per-corner blend — agree within `disagreement_gate` (squared Mahalanobis
  under the combined covariance) → precision-weighted average; disagree
  beyond it → trust whichever branch is more confident about that corner
  alone, no splitting the difference; missing covariance on one side → zero
  precision, degrades to "use whichever side has data."
- **Empirical calibration** (`corner_fusion_alignh_calibrated.py`,
  `loma_cov_scale=32.0` / `samh_cov_scale=70.0`): both branches' raw
  covariances were badly overconfident vs GT (median chi-square(2) stat 44.0
  LoMa / 96.3 SAM-H, nominal 1.386 — SAM-H is the *more* overconfident,
  which had biased the uncalibrated trace tie-break toward it regardless of
  which branch was actually right). The scale correction is a
  median-matching point estimate, first measured at n=20 PTi sequences,
  confirmed at n=80 (implied scales 33.8/68.4, within ~6%/2% — not overfit
  to the small sample). `experiments/fusion_calibration_check.py` makes this
  check reusable. `disagreement_gate` itself turned out not to matter in the
  6-40 range once calibrated (flat within 0.1pp) — the fix was in the
  tie-break bias, not the fuse/arbitrate boundary.

### Full-grid result (`corner_fusion_alignh_calibrated`, PTi n=300 + POT n=210)

| config | PTi p@15 | POT p@15 |
|---|---|---|
| `loma_samh_decoupled_rerun` (plain inlier switch, baseline) | 87.86 | 97.66 |
| `loma_samh_decoupled_alignh_pure` (align, no fusion — point 3's regression) | 77.95 | 96.90 |
| `corner_fusion_alignh_calibrated` | **80.59** | **95.99** |

Reads past **published SAM-H SOTA (80.1) on PTi** and recovers about a
third of alignh_pure's regression against the switch baseline — a real,
principled, verified result (two honest fixes: numerical floor, empirical
calibration, not parameter search). **Does not meet the §0 success
criterion** (PTi ≈ 88 without cratering POT): still -7.3pp below the switch
baseline on PTi, and a small (-0.91pp) regression vs alignh_pure on POT.

### POT regression, root-caused

11/210 POT sequences drop from perfect (100) to 60-80 under fusion; 3
recover from imperfect to perfect; net -0.91pp, invisible at the 20-sequence
probe scale that validated calibration. Traced precisely (not guessed): the
independent SAM-H sub-tracker is built with `flow_enabled=False`, so on a
rejected align frame it **coasts** (`_finalize_failure` returns the
unchanged last homography) — a real, usually-still-accurate estimate (it's
exactly why `alignh_pure` itself hits p@15=100 on these same sequences) but
with no fresh covariance, since nothing was measured that frame.
`fuse_corners` treats "no fresh covariance" identically to "branch absent,"
so a coasted-but-good `samh_H` gets discarded outright and LoMa wins by
default, even where LoMa is worse. Verified by exact cross-reference
(`fusion_samh_cov` validity == `maskh_align_accepted`, zero mismatches
either direction on the two worst-regressed sequences) — ruled out a first
(wrong) hypothesis that the covariance *computation* was numerically
failing on accepted fits; it never does.

This is a third case for the "Two things not to re-derive" list above:
**coasting loses information that covariance fusion, as currently scoped,
has no way to recover** — not the gross-error/distractor mechanism (point
4), a distinct and simpler gap. Natural remedy (not implemented — a design
choice, not a quick patch): a fallback covariance for a coasted `samh_H`
that decays with frames-since-last-measurement, so a 1-frame-stale coast
still casts a weak vote instead of none, while a long coast decays toward
"branch absent" as it should.

### Where this leaves the priority queue

The §10 handoff's items (1) verification debt and the covariance-plumbing
half of (3) are done. Item (2), `selection_state.py`'s episode-latch
read-out, is the clear next step — more so now: PTi's remaining -7.3pp gap
to the switch baseline is plausibly mostly the gross-error/episode class
(point 4), which covariance fusion cannot address by construction (both
branches under-report identity-failure errors as confident, not just
imprecise — a distractor produces a tight, confident, *wrong* ellipse, per
"Two things not to re-derive" above). Two smaller, well-scoped follow-ups
identified but not started: the coasting-credit gap just above, and
whether `disagreement_gate`/`*_cov_scale` should become per-dataset or stay
global constants (only ever calibrated on PTi; POT calibration untested).
