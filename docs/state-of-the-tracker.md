# State of the tracker

**The single entry point.** If you are picking this work up cold, read §0, then §1.
Everything else is reference. Last updated 2026-08-25.

---

## 0 · Start here

**What we are building.** A planar-object tracker that reports the four corners of a
flat surface in every frame of a video. Two lines of work exist; only one of them is the
paper.

**The method — `configs/tracker/loma_samh_matchnaming.py`, frozen since 2026-08-20.**
A keypoint matcher (LoMa) and a mask branch (SAM3 → Hough → corners) each estimate the
pose; the keypoint answer is taken when its RANSAC inlier count reaches 35, otherwise the
mask answer; the four cyclic namings of the resulting quad are then scored by how much of
the matcher's raw, pre-RANSAC correspondence set each explains
(`U = sum max(0, 1 - d²/τ²)`, τ = 30 px, `flatsam2/match_naming.py`). That last step is
the paper's contribution: it deletes DINOv2 and six parameters from the predecessor and
is free to the millisecond (§3.4).

**The quad-free line (`corner_fusion_*`) is PAUSED** (maintainer, 2026-08-20) — per-corner
covariance fusion of the same two branches, with a distance-transform registrar instead of
Hough. It is not in the paper. Its one valid post-fix number is
`corner_fusion_naming` PTi **89.21** (+1.03 pp over the shipped method); its POT half is
running as of 2026-08-25 (§0.1). Every *other* `corner_fusion_*` figure in this file and in
the journal predates the covariance and alias fixes of 2026-08-19 and **must not be cited**.

**State right now: the method is measured, the paper is being written.** All six datasets
are complete for the shipped config and for its DINOv2 predecessor; the three-arm naming
ablation and the controlled runtime measurement are done (§3). Remaining work is the paper
(`paper/`, WACV 2027; rolling review in `paper/draft_review.md`) plus the two sensitivity
questions in §0.1.

**How we got here, so it is not re-derived.**

1. **Naming is separable from localization.** Best-rotation P@15 forgives any cyclic
   relabelling, so a mechanism that only relabels corners must leave `br_p@15` untouched —
   and the no-naming and match-naming arms agree to two decimals at 89.58 over 300
   sequences while the headline moves 85.03 → 88.18. Read the br-gap before the headline.
2. **The matcher already holds the naming evidence.** 89.5% of mis-named PTi frames are
   frames the *mask* branch posed, and on them the keypoint branch's naming is right 72.3%
   of the time while its position is right only 44.5% — so borrow the naming, not the pose.
3. **One texture term cannot rank poses and name corners.** The support saturates over
   pose (a 10–20 px error still explains nearly every match at τ = 30) and separates
   namings ~180×; NCC does the opposite. `matchverify` (NCC deleted) lost 1.32 pp of
   localization on PTi, `matchncc` (a third factor in the score) lost 3.88 pp because the
   registrar's thresholds are not scale-free. The support belongs in the naming comparison
   and nowhere else.
4. **The segmenter is not the story.** Running each published baseline's own method on
   SAM3 gives SAM-H +2.71 pp (PTi) and WOFTSAM −0.22 pp; against the *best SAM3-equipped
   baseline* our margin is +5.13 PTi and +2.06 POT — wider, not narrower (§3.2).

### 0.1 · Resume here

**Publication grids and final readouts are complete.** Follow-up jobs submitted on
2026-08-25 at HEAD `55f63b9` now have these states:

| what | job / eval | output |
|---|---|---|
| τ-ablation telemetry re-run, PlanarTrack **train**, `pt_trn_50` | complete: 11408267 / 11408268 | `$OUT/tau_ablation/loma_samh_matchnaming/PlanarTrack/` |
| `corner_fusion_naming` POT — paused branch, not for publication | running: 11408275 / 11408276 | `$OUT/corner_fusion_naming/POT-210/` |
| cross-hardware reproducibility, `pti_fusion_probe20` re-run | complete: 11408394 | `$OUT/determinism_check/probe20/` |

**Final publication readouts.**

1. **Naming sensitivity is flat on train.** After retaining the deployed eight-match guard,
   naming accuracy varies by 0.04 pp over `tau=3..100`; sweeping the raw-match minimum from
   0 to 32 changes it by 0.10 pp. The deployed `tau=30`, minimum 8 setting lies on both
   plateaus, and replay reproduces recorded choices on 100.00% of 19,800 localized frames.
2. **Cross-GPU aggregate moves by +0.29 pp.** Two same-node re-runs are bit-identical. The
   20-sequence A100-grid to V100 re-run changes per-sequence mean P@15 from 91.01 to 91.30;
   trajectories are not bit-exact across GPU generations, but aggregate movement is small.
3. **`corner_fusion_naming` POT** may turn 89.21 into a pair. It does not change
   the paper unless the maintainer reopens the quad-free branch, so read it and record it,
   nothing more.

**Known open ends.**

- **Reproducibility across GPU generations is not bit-exact** (measured 2026-08-25). Seeds
  are fixed and `run.py --deterministic` exists, but SAM3/LoMa kernels differ between the
  A100 (`amdgpu`) and V100 (`gpu`) partitions and the tracker is autoregressive, so
  trajectories separate from frame 1. On 20 sequences, per-sequence mean P@15 moved
  +0.29 pp. Grids therefore must not be compared cell-by-cell across partitions;
  aggregates are what carry.
- **UCSB is closed, but its explanation is not.** 93.73 vs WOFTSAM's 96.30, with a measured
  oracle ceiling of 93.49 over the two branches — no selection rule can close it, so it is a
  target-definition mismatch and stays as an honest sentence in the paper. What the 2026-08-21
  figures complicate is *why*: on `bricks_motion8` the SAM3 mask sits on the GT photo region
  and the pose is still ~20 px out and visibly too large, which points at the mask→pose step
  rather than at the mask. Drawing LoMa's inliers would settle it and needs new telemetry —
  the debug npz stores `n_inliers` as a count only.
- **`samh_cov_scale=70` predates the covariance fix**, and the two branches are precision-
  averaged as if independent while the mask result is partly LoMa-derived. Both belong to
  the paused quad-free branch only. The decoupled baseline for the second question is
  measured: signed corner-error correlation 0.007 / 0.014 (magnitude 0.019) over 122195 PTi
  frames, so anything materially above that on a fusion run is induced double-counting.
- **Branch `hough-corner-cov`** (per-corner covariances for the Hough branch + fuse-when-
  branches-agree) is complete, tested and committed, and deliberately **not** merged: it is a
  method change, not a result, and the method is frozen. Worktree `$OUT/dev-hough-cov`. Its
  estimator's arithmetic is validated (Monte-Carlo std ratio 1.007) but its noise model needs
  `n_eff = min(n, cap)` — a mask boundary is systematically displaced, not noisy, so the true
  error is flat in n while the prediction shrinks as 1/√n and no constant scale factor fixes it.
- Two tests fail in `tests/test_cropped_area_discrepancy.py` and are **not** stale: a real
  defect on the SAM3 self-correction path, unrelated to the tracker. `pytest tests/` is
  otherwise green (184 passed, 2026-08-25).

**Working practice that keeps costing time if forgotten.**

```bash
source .envrc                                  # nothing works without this
squeue -u $USER -o "%.14i %.32j %.2t"          # what is alive
python scripts/hpc_status.py --pipeline cross_dataset
```

- Completeness is **not** proven by a job finishing: check the sequence count *and*
  `aggregate.num_sequences` in `eval_results.json`.
- Tasks read the **working tree**, not a pinned revision — do not edit `flatsam2/`,
  `run.py`, `evaluate.py` or `configs/` while a grid runs.
- `--time` and the partition set are **one decision**: size the limit for the slowest
  member (V100 is ~2.4× slower than A100) and the longest sequence. Three timeout waves on
  2026-08-19/20 cost ~135 task-slots.
- Always list both partitions (`--partition amdgpu,gpu`). Our QOS caps us at ~8 concurrent
  GPUs, below either partition's free capacity, so throughput is availability × speed.

---

## 1 · What the tracker does, and why it is built this way

You point at a flat thing in the first frame — a book cover, a poster, a facade. The
tracker follows it and reports its four corners in every later frame. Because the thing
is flat, its appearance in any frame is fixed by one 3×3 homography, so "track the
object" means "estimate that matrix, every frame".

Two experts answer that question every frame, and they fail in opposite conditions:

- **The keypoint expert (LoMa).** Matches texture between the first frame and now, fits
  a homography to the matches. Strong when the surface is textured and sharp. It knows
  *which corner is which*, because texture is not symmetric.
- **The mask expert (SAM-H).** Asks SAM3 for the silhouette, then finds the pose that
  puts the template's outline on it. Strong exactly where texture dies — blur, low
  light, extreme angle — because a silhouette survives what texture does not.

The interesting problems all come from combining them, and from what a silhouette
cannot say.

**Why not a threshold.** The predecessor took the keypoint answer when its RANSAC inlier
count exceeded 35, else the mask answer. That is a cliff — 34 and 36 inliers give
completely different outputs on nearly identical evidence — and the number is arbitrary.
*The shipped method keeps that switch*, and the objection is answered only in part: the
threshold is **inherited** from the predecessor system rather than fitted here, and T = 35
sits at the centre of a broad plateau (PTi moves 0.25 pp over T in [25, 80], POT not at all),
so nothing is tuned to it. The per-frame incoherence stands, and is what the quad-free branch
was for.
Both branches can instead state *how sure* they are, per corner, as an error ellipse
derived from their own fit (§2.1). That makes them comparable measurements rather than
two black boxes with a magic number between them, and the combination rule follows from
the uncertainties instead of from tuning.

**Why the mask branch needs a search.** A single solve anchored on the last frame only
works when the last frame was nearly right — which is when the branch is not needed. So
the mask branch answers the harder, stateless question ("where is the object, knowing
nothing") by whitening both silhouettes with their second moments, which leaves exactly
one free rotation, sweeping that angle, refining each candidate and scoring them.
Moments exist for any shape, so this is where the quad assumption disappears.

**Why texture is in the verifier.** When the search was instrumented, the correct pose
was already among the candidates **89%** of the time and the scorer was picking a
different one. A silhouette cannot say which way round it is: a wrong pose scored raster
IoU **0.94 — the same as the right one — while being 219 px wrong**, and **44.6%** of
all wrong hypotheses are pure cyclic corner relabelings. Shape cannot break that tie, so
the verifier also correlates the warped template's *texture* against the frame.

---

## 2 · The quad-free branch, as it stood when it was paused

> **This section documents the PAUSED `corner_fusion_*` line, not the shipped method.**
> The tracker in the paper is `loma_samh_matchnaming` (§0): the 35-inlier switch plus
> correspondence-support naming, implemented in `flatsam2/tracking/loma_samh.py` and
> `flatsam2/match_naming.py`. Keep this section for when the branch is resumed.

Implementation map: `flatsam2/tracking/corner_fusion.py` (the fusion),
`flatsam2/mask_register.py` (propose/refine/verify), `flatsam2/mask_align.py` (the GN
solve and the mask branch's covariance), `flatsam2/corner_covariance.py` (the keypoint
branch's covariance).

Full derivations, with figures generated by running the deployed code on real frames,
are in **`docs/tracker-cards.html`** — including the seven-step covariance recipe and
the drawing of the Jacobian that produces it.

### 2.1 Both branches report per-corner covariances

Each branch parameterizes its fit by θ = the eight numbers of the four corner positions,
writes one residual per piece of evidence (boundary signed-distance for the mask branch,
reprojection error for the keypoint branch), differentiates the residuals against θ by
central differences, and takes Σ_θ = σ̂²(JᵀWJ)⁻¹ with σ̂² from its own residuals. Because
θ *is* the corners, corner k's 2×2 covariance is just a diagonal block of Σ_θ — no
propagation step, and the same formula on both sides, which is what makes the two
comparable.

It is a **precision model**: it says how far a corner would move if the evidence were
resampled. It cannot know the evidence is the wrong object — a confident fit to a
distractor gives a tight, wrong ellipse. Validated by bootstrap on real data, where the
closed form matches an i.i.d. resample (1.32×) but is 2.92× narrower than a moving-block
resample, because boundary residuals are spatially correlated (τ_int = 5.1 samples, so
N_eff = 31 of 156). That derives ≈8.5× of the deployed inflation from first principles;
the rest is an empirical correction and is described as such.

### 2.2 One fusion decision per frame

Per corner, the squared Mahalanobis distance between the two branches' estimates under
the summed covariance. The **mean over the four corners** is compared against
`disagreement_gate`; then, for the whole quad at once:

- within the gate → every corner is the precision-weighted average of the two branches;
- beyond it → the entire pose comes from the branch with the smaller total corner
  variance;
- one branch missing → that branch; neither → coast on the last pose.

Before that decision, the mask branch's corners are re-*named* from the keypoint branch's:
its quad is permuted to the cyclic rotation nearest the keypoint quad, but only when that
rotation is `fusion.keypoint_naming_margin` (2.0) times closer than the branch's own
naming. A silhouette fixes a pose only up to which corner is called first, and 89.5% of the
mis-named frames are frames the mask branch supplied; texture can break that tie, so the
naming is borrowed without the position. The margin is the incumbent's protection — the
keypoint quad names correctly far more often than it localizes well, and taking the nearest
rotation unconditionally keeps only +0.28 pp of the rule's +1.01 pp, so the guard is worth 0.73 pp. The margin's *value* is not tuned: 1.35-3.0 spans a factor of 2.2 and moves pooled P@15 by 0.05 pp, and `br_p@15` is 89.82 at every margin including 1.0, so the rule never touches localization at any setting.

The output is therefore always either a genuine blend or a pose some branch actually
proposed. The earlier rule decided this **per corner**, which could emit a quad neither
branch had proposed; measurement said that freedom was worth 0.48 pp at the oracle on
0.66% of frames, and removing it *gained* accuracy on both datasets (§3.3).

### 2.3 The mask branch: propose → refine → verify

**Propose.** Whitening both silhouettes by the inverse square root of their second-moment
matrices makes them agree up to one rotation; sweeping 24 angles enumerates every affine
pose the two shapes are consistent with. The last-frame pose and the keypoint branch's
pose enter as proposals alongside the sweep.

**Refine.** The best proposals get a coarse damped Gauss-Newton solve against a
downscaled signed-distance field; the best of those get a full-resolution solve. The SDF
is built only on the mask's bounding box plus a margin, which is exactly equivalent
where it is sampled.

**Verify.** `score = IoU · (1+NCC)/2` — the raster overlap of the warped template
against the mask, times the normalized cross-correlation of the warped template's
texture against the frame, mapped to [0,1]. Both exponents are 1: it is the plain
product of two evidence terms, with nothing left to tune. NCC is computed only inside
`warp ∧ mask` so occluders and background never enter the sum.

**Aliasing defences.** After refining a proposal the three cyclic relabelings of its
control points are also scored (`alias_shortcut`) — they share a silhouette and differ
only in which corner is which, so they cost three verifications and no solve. The same
check is repeated on the final full-resolution winner (`alias_recheck`), because the
full-res solve moves the pose after the first check. Both require an *inherited* naming
to be beaten by `incumbent_margin`, since a naming carried from the previous frame
holds temporal evidence a single-frame score does not.

**Cost control.** When the last-frame proposal already verifies well the search is
skipped and the frame costs one GN solve. In deployment the incumbent wins **88.5%** of
frames and the stateless search runs on **8.3%**.

---

## 3 · Results

Numbers below are **per-sequence mean P@15** unless a column says otherwise — the
convention the paper's tables use. Pooled P@15 (every GT frame counted once) differs, and
the per-sequence mean is the more fragile of the two: on POT one sequence has zero GT frames
and the eval scores it 0, which alone moves the mean by 0.46 pp. The paper's POT branch and
oracle rows are therefore on the 209-sequence valid-GT basis; `results_tables._mean` is not.

**br-gap** = `br_p@15 − p@15`, the difference between precision under the best of the
four corner rotations and precision as-is. It isolates *corner-labelling* error from
*localization* error at zero cost, and should be read before believing any headline from
a method that re-detects pose.

### 3.1 The shipped method on the two deciding benchmarks (per-sequence mean P@15)

| config | PTi P@5 | PTi P@15 | br-gap | POT P@5 | POT P@15 |
|---|---|---|---|---|---|
| `loma_samh_nonaming` — no corner naming at all | — | 85.03 | 4.55 | — | 97.66 |
| `loma_samh_decoupled_rerun` — DINOv2 appearance vote | 65.44 | 87.86 | 1.57 | 92.74 | 97.66 |
| **`loma_samh_matchnaming` — correspondence support (shipped)** | 65.48 | **88.18** | **1.40** | **92.55** | **97.66** |

**The naming ablation is the paper's central claim, and the br-gap is how it is read.**
`br_p@15` is **89.58 for the no-naming arm and 89.58 for ours** over 300 sequences — a
mechanism that only relabels corners must leave best-rotation precision untouched, and it
does. So naming is worth **+3.15 pp on PTi and it is provably only naming**: of the 4.55 pp
of corner mislabelling that exists, DINOv2 recovers 2.83 (62%) and correspondence support
recovers 3.15 (69%), with a whole network and six parameters deleted. DINOv2's br-p@15
(89.43) is slightly *below* both other arms, i.e. appearance naming perturbs trajectories a
little while match naming does not.

**POT is a null result and worth stating as one.** Its br-gap is 0.00 for every arm, so
there is nothing to fix and naming changes the score by exactly nothing (97.66 either way).
A mechanism that helped there would be suspicious.

**Selection ceiling** (`experiments/branch_p_at_k.py::oracle_p_at_k`, per-frame oracle over
the two branches — no switch, threshold or covariance rule can beat it):

| | PTi P@15 | POT P@15 |
|---|---|---|
| keypoint branch alone | 78.26 | 97.07 |
| mask branch alone | 84.37 | 90.11 |
| shipped (35-inlier switch) | **88.18** | **97.66** |
| oracle over the two branches | 91.28 | 98.60 |

The switch already beats the better branch by 3.81 pp on PTi — it is not a fallback. PTi
still holds 3.10 pp of pure selection headroom (55% captured); POT holds 0.94 pp (39%), so
**POT cannot separate two candidate rules and PTi decides**.

### 3.2 Six datasets (per-sequence mean P@15)

Complete for the shipped config and its DINOv2 predecessor as of 2026-08-21.

| config | PTi (300) | POT (210) | POIC (20) | TMT (92) | UCSB (96) | MPOT-3K (573) | worst |
|---|---|---|---|---|---|---|---|
| **`loma_samh_matchnaming`** | **88.18** | **97.66** | 98.95 | **94.38** | 93.73 | **90.98** | **88.18** |
| `loma_samh_decoupled_rerun` | 87.86 | 97.66 | 98.97 | 93.99 | 93.76 | 90.98 | 87.86 |
| `woftsam_samfallback_nl` (prev. paper) | 77.37 | 96.75 | 98.82 | 89.20 | **96.30** | 89.74 | 77.37 |
| `sam_H` (prev. paper) | 80.34 | 88.38 | 63.04 | 58.07 | 68.61 | 80.46 | 58.07 |

We lead both baselines on five of six columns and beat WOFTSAM's worst column by 10.8 pp.
**Cross-dataset naming behaves as the mechanism claims**: it moves the br-gap, and the
headline follows only where the gap had room to move — TMT 0.53 → 0.06 and +0.40 pp, PTi
1.57 → 1.40 and +0.32, MPOT 0.08 → 0.00 and neutral, while POIC/UCSB/POT had nothing to fix
and it costs 0.02–0.03 pp, i.e. noise. **UCSB is the one hole** and is closed as a bounded,
explained gap, not a tuning target (§0.1).

**The segmenter control** (each published baseline's own method run on SAM3, per-sequence
mean P@15):

| | SAM2 | SAM3 | segmenter is worth |
|---|---|---|---|
| PTi SAM-H (mask → Hough) | 80.34 | 83.05 | +2.71 |
| PTi WOFTSAM (flow) | 77.37 | 77.15 | −0.22 |
| POT SAM-H | 88.38 | 89.37 | +0.99 |
| POT WOFTSAM | 96.75 | 95.60 | −1.15 |

SAM3 helps only the method that consumes masks directly, and on POT it makes the flow
baseline *worse*. Measured against the best SAM3-equipped baseline our margin is **+5.13
PTi** and **+2.06 POT**. **Provenance caveat:** the WOFTSAM POT SAM2 number is *imported*
from the previous paper; a fresh re-run of it scored 96.37, so the like-for-like fresh pair
is 95.60 vs 96.37 = −0.77, not −1.15. Quote the fresh pair.

### 3.3 The two simplifications of the paused branch, measured offline

Both *delete* parameters, and both were decided before spending a GPU hour. They belong to
the quad-free line (§2) and are recorded here so the branch can be resumed without redoing
the analysis.

**Fusion scope** (`experiments/fusion_rule_replay.py`, exact replay against completed
grids — neither branch reads the fused output, so re-fusing stored estimates reproduces
what the tracker would have output): the whole-frame decision beat the per-corner one by
1.19 pp (POT) and 0.56 pp (PTi) pooled, i.e. 86% and 40% of the entire gap to the oracle,
from deleting a degree of freedom. Genuinely complementary frames — where the two branches
are each better on different corners — are **0.60%** of all frames, which is why.
*(The absolute values in that replay predate the covariance fix; the ordering is the claim.)*

**Verification score** (`experiments/maskh_bench.py`, held-out test split, mean P@15
over the four initialization regimes): `IoU² · ((1+NCC)/2)^0.5` → `IoU · (1+NCC)/2`
takes POT 90.88 → 91.75 and PTi 84.38 → 84.40, with the worst regime's p90 error falling
14.5 → 11.6 px (POT) and 73.6 → 54.7 px (PTi). The only regression anywhere is PTi
`good`, −0.2 pp.

### 3.4 Speed, measured under control

All three configs and all sequences **inside one job on one node** (g11, A100-SXM4-40GB) —
mixing A100 and V100 across configs would have swamped the effect with a 2.4× hardware
difference. 100 frames after 5 warmup, CUDA sync on, mean over three PTi sequences:

| config | ms/frame | FPS | vs no naming |
|---|---|---|---|
| `loma_samh_nonaming` | 286.6 | 3.49 | — |
| **`loma_samh_matchnaming`** | **286.7** | **3.49** | **+0.1 ms (+0.03%)** |
| `loma_samh_decoupled_rerun` (DINOv2) | 299.5 | 3.34 | +12.9 ms (+4.51%) |

**Our naming is free to the millisecond**; the appearance vote costs 4.5%. The honest form
of the claim is not "we are faster" — the gain is entirely the deleted network. A second
forward pass was being bought for a 2-bit decision, and the matcher's existing
correspondences answer it for nothing.

Where the time goes (matchnaming, Seq_00006): LoMa keypoints 41.3% (describe alone 22.1%),
SAM3 34.4%, LoMa match 13.2%, mask-H 10.0%, RANSAC 0.3%; naming does not appear as a
measurable section. Peak GPU memory 6.45 GB. **At ~3.5 FPS the tracker is not real-time and
the paper should say so plainly** — the cost is the two networks, not the contribution.

---

## 4 · Settled, and open

### Settled

- **A silhouette alone cannot determine a planar pose.** Measured three independent
  ways: offline (44.6% of wrong hypotheses are cyclic relabelings; a wrong pose scores
  IoU 0.94 at 219 px error), at grid scale (the abandoned `redetect` variant's 10.50 pp
  br-gap on 35% of sequences), and by ablation (removing texture puts 86% of the loss
  into the br-gap). Texture breaks the tie; the keypoint branch is a second, cheaper
  source of the same disambiguation.
- **Quad-free tracking is not a sacrifice** on the datasets it was developed against:
  parity or better on PTi P@15, better P@5, better br_p@15, better under blur.
- **Covariance fusion beats the 35-inlier switch as a combination rule** on PTi — the one
  valid post-fix quad-free number is `corner_fusion_naming` 89.21 against 88.18 — but the
  branch was paused before that could be shown on a second benchmark (§0).
- **The per-corner decision was not earning its complexity** — and removing it gained
  accuracy (§3.3).
- **The verification score's exponents were not identifiable**; the plain product is as
  good or better everywhere that matters.
- **Corner naming is a correspondence-support problem, not a property of either
  architecture.** The same scalar helps both trackers: +0.37 pp on the quad-based one
  (with DINOv2 deleted) and +0.82 pp on the quad-free one.
- **T = 35 is not load-bearing.** Inherited from the predecessor system, and flat over
  T in [25, 80]. The claim is the flatness, never the peak.
- **The branches' errors really are uncorrelated when they are decoupled** — 0.007 / 0.014
  signed, 0.019 in magnitude, over 122195 PTi frames — so precision-weighted averaging is
  the right combination rule rather than a convenient approximation.
- **UCSB cannot be repaired by arbitration** (oracle 93.49 < WOFTSAM 96.30) and is closed.

### Open

- **Corner labelling still carries a 1.40 pp br-gap on PTi**, of which 12.8% of frames have
  no correspondence evidence at all to name from — and those evidence-less frames carry the
  large majority of all mis-naming. Whether that residue is reducible or genuinely ambiguous
  (near-square outline, near-symmetric texture) is unmeasured.
- **τ = 30 is unablated** until the train-split re-run lands (§0.1). It is the one parameter
  the method adds.
- **Cross-GPU reproducibility** (§0.1): bit-identical on one node, divergent across
  partitions from frame 1.
- **Occlusion is the attribute with the least margin** — 86.69 on PTi's occlusion subset
  against an 88.18 aggregate, and occlusion robustness outranks aggregate P@15 for this
  project. Not a regression against the baselines (they score 76–77 there), but the thinnest
  cushion we have.
- **Perspective.** The moment proposal is affine while the refinement is projective, so
  accuracy falls with ground-truth perspective strength; the cold-minus-good gap widens
  from −4.6 pp on the weakest half to −11.1 pp on the top 5%. Quad-free branch only.
- **Calibration.** The quad-free gate is χ²₂ in form but the covariances are inflated ×32
  and ×70 before it, so it is a relative precision model with an empirical scale correction,
  not calibrated probabilistic fusion — and the ×70 predates the covariance fix.

---

## 5 · Configs

| purpose | config |
|---|---|
| **the method** (frozen 2026-08-20) | `loma_samh_matchnaming` |
| naming ablation: DINOv2 appearance vote | `loma_samh_decoupled_rerun` |
| naming ablation: no naming at all | `loma_samh_nonaming` |
| segmenter control: SAM-H's own method on SAM3 | `hough_symmetry_only` |
| segmenter control: WOFTSAM's own method on SAM3 | `woftsam3_samfallback_nl` |
| published baselines | `sam_H`, `woftsam_samfallback_nl` |
| paused quad-free line: the live result | `corner_fusion_naming` |
| paused quad-free line: the reference rule | `corner_fusion_wholeframe` |
| quad-free ablation ladder (do not run as alternatives) | `corner_fusion_register`, `_kp`, `_final`, `_recheck`, `_simple`, `_simple_norecheck` |
| quad-free mechanism ablations (why the support goes in the alias comparison only) | `corner_fusion_register_noapp`, `_matchverify`, `_matchncc`, `_aliasname` |

Estimator-level ablations do **not** need configs or GPU: `mask_register.bench_methods`
exposes ~35 of them to `experiments/maskh_bench.py`, which runs the whole table on CPU
in minutes.

---

## 6 · Working rules, learned the hard way

1. **Offline first.** `maskh_bench` measures a mask→H idea on CPU in minutes with the
   initialization as a controlled variable; `fusion_rule_replay` re-scores a fusion rule
   exactly against a completed grid. Tune on `--split dev`, report on `--split test`. A
   GPU grid is for confirming, not exploring.
2. **`register_kp` in the offline bench needs `--extra-h-run`.** Without it the bench
   never fills `ctx["extra_H"]` and the method is byte-identical to plain `register` —
   which reads as "the keypoint proposal does nothing". The bench now refuses to run a
   keypoint method without it.
3. **`p90_err = inf` in the bench table means "at least one non-finite error"**, not
   "10% failed". Count the failures before drawing a conclusion.
4. **Read the br-gap** before believing any headline from a pose-redetecting method.
   `alignh_redetect` looked fine on 4 probe sequences and lost 6 pp at grid scale
   entirely to corner labelling.
5. **`squeue` truncates the NAME column**, so watching a job by config name silently
   matches nothing. Watch by job id.
6. **`/tmp` is node-local.** A script written to `/tmp` on the login node is not there
   on the compute node.
7. **A new experiment gets a new config stem.** The output directory derives from it, so
   a fresh stem cannot collide with an existing run. Never overwrite `OUTPUT_DIRECTORY`.
8. **Submit with `--no-mail`.**
9. **Offline regime means screen ideas; they do not predict grid deltas** for anything
   that only acts on the search path. `maskh_bench` weights `good`/`drift`/`lost`/`cold`
   equally, but in a real sequence the incumbent proposal wins 88.5% of frames — so a
   gain confined to `lost` is diluted roughly tenfold while any harm in `good` lands at
   full weight. Screen offline, then confirm on a grid before shipping. **Worked example:**
   the verification score's a=b=1 simplification measured +0.87 POT / +0.02 PTi over the
   regimes and cost **0.77 pp PTi** on the grid. (An earlier entry offered the alias
   re-check as this example; that was a 0.14 pp difference inside noise and does not
   support anything — see `dead-ends.md`.)
10. **Submit aggressively, then verify completeness.** `amdgpu` runs chronically full —
   27 running against 99 pending, the running jobs holding GPUs another 8-18 h — so work
   starts only when backfill finds a hole it fits, and the hole size is the one thing we
   control. Pack into many small tasks with a tight limit: `--jobs 75 --time 0:15:00` on
   PTi (~4 sequences per task), `--jobs 53 --time 0:15:00` on POT — measured 2026-08-18:
   tasks ran 8:40-11:24 against the 15-minute limit with zero kills, so this is calibrated
   rather than optimistic. Never the 5 h default,
   and "shorter than the default" is not the bar — 1:30:00 for 40-minute work still asks
   for a hole twice the size of the job.

   Over-packing is safe because `run.py` defaults to `skip_completed=True` and writes
   results per sequence: a task killed at the wall loses at most the sequence it was
   mid-way through, and a resubmit of the same command picks up exactly the remainder. So
   an under-requested limit costs a resubmit, not GPU-hours.

   The obligation that comes with it: **check that the whole thing finished.**
   `scripts/hpc_status.py --pipeline <name>` for per-config completeness, then resubmit
   the gap. And watch the chained eval — `slurm/submit.py` attaches it with
   `--dependency=afterok`, so if any array task is killed the eval never runs; after the
   final resubmit confirm `eval_results.json` exists and re-trigger it if not.
11. **Do not read 0.1–0.2 pp of PTi P@15 as a result.** Four configs in this thread span
   87.55–87.87 pooled with no reproducible ordering. What *is* reproducible at that scale
   is the **br-gap**, which moved monotonically 1.73 → 2.26 across the same four. Separate
   `br_p@15` (localization) from the gap (naming) before concluding anything; a change can
   improve one and pay it all back in the other, which is exactly what the two
   simplifications do on PTi.

---

## 7 · Where the detail lives

| document | what is in it |
|---|---|
| `docs/tracker-cards.html` | the mechanisms in full — derivations, MathML, and figures generated by running the deployed code on real frames. The technical reference. |
| `docs/dead-ends.md` | directions that were tried and abandoned, each with the measurement that killed it |
| `docs/journal.md` | append-only, one entry per working day — the index of what happened when |
| `docs/correction_research.md` | the SAM3 self-correction thread |
| `docs/2026-08-07.md` | full session log of the fusion thread's origin |
| `DETAILS.md` | architecture, data formats, dataset layouts |
| `experiments/pipeline.yml` | every runnable pipeline, its outputs and its completeness signal |
