# Journal

Append-only. Brief entries; see referenced docs for detail.

## 2026-08-07 — quad-free alignh_pure full grid: regression found
Full-grid run (PTi 300, POT 210) of the quad-free mask-align branch regresses
vs the plain inlier-threshold switch (PTi p@15 87.9→78.0, below published
SAM-H SOTA 80.1); root cause is the aligner rejecting ~half of frames
(residual p90 154px vs the 6px acceptance threshold). Details: docs/2026-08-07.md.

## 2026-08-10 — threshold sweep closed, not a fix
Loosening the align acceptance threshold (6→15/30px) on 4 probe sequences
makes things worse-or-flat, and res15 shows the fix can backfire
non-monotonically via the aligner's autoregressive GN warm-start
(accepting one marginal frame corrupts every later frame's starting point).
Not pursued further. Details: docs/2026-08-07.md.

## 2026-08-10 — corner_fusion built: per-corner covariance fusion
Replaced the hard inlier-count switch with per-corner precision-weighted
fusion of LoMa + SAM-H, each reporting its own covariance from its fit's
normal equations (`flatsam2/tracking/corner_fusion.py`). Fixed two bugs on
the way: stale `test_corner_covariance.py` API mismatch, and `model_copy()`
not re-running the algorithm-derivation validator (silently ran the old
algorithm for a full probe). First numbers beat alignh_pure on both smoke
sequences; mid-scale PTi 20-seq probe: alignh_pure 78.25 → fusion 80.44
p@15. Details: docs/2026-08-07.md.

## 2026-08-10 — covariance floor + calibration check
Found and fixed unfloored covariance blow-up (near-singular fits claim
absurd confidence) — floor is a correct defensive fix but didn't move the
regression (80.47 vs 80.44). Built the calibration check that should have
come first: both branches badly overconfident vs GT (LoMa 31.8x, SAM-H
69.5x implied covariance inflation), SAM-H more so — explains why the
uncalibrated trace tie-break was regressing some sequences. Applied an
empirical scale correction (`corner_fusion_alignh_calibrated`, n=20 PTi
sample): better-or-equal on 20/20 probe sequences, p@15 78.25→80.47→81.90
across alignh_pure→fusion→calibrated. Still short of the plain-switch
baseline (88.25) on this subset. Details: docs/2026-08-07.md.

## 2026-08-10 — POT probe confirms calibration fix generalizes; gate re-tune is a no-op
POT 20-seq probe: calibrated fusion matches alignh_pure (96.00 mean p@15)
and recovers uncalibrated fusion's one regression (V16_4: 60→100), same
pattern as PTi. `disagreement_gate` sweep (6.0/20.0/40.0) on PTi is flat
(81.88-81.90) — the gate isn't the lever, calibration's value was fixing
the trace tie-break's SAM-H bias, not the fuse/arbitrate boundary. Added
`experiments/fusion_calibration_check.py` (reusable calibration diagnostic)
and an 80-seq PTi subset to check the 32.0/70.0 scale constants hold beyond
the original n=20 sample. Details: docs/2026-08-07.md.

## 2026-08-11 — calibration confirmed at n=80, full grid launched
Implied cov_scale at n=80: 33.8 (LoMa) / 68.4 (SAM-H), vs the n=20 estimate
32.0/70.0 — within ~6%/2%, not overfit to the small sample; heavy tail
(p99 in 1e5-1e8) also matches n=20, a stable feature not a small-n
artifact. Cleared to launch the full 300+210 grid on
`corner_fusion_alignh_calibrated`. Details: docs/2026-08-07.md.

## 2026-08-11 — full grid result: beats SOTA on PTi, mixed on POT
Full-grid mean p@15: PTi rerun 87.86 / alignh_pure 77.95 / calibrated
fusion **80.59** (past published SAM-H SOTA 80.1, still -7.27pp vs the
switch baseline). POT rerun 97.66 / alignh_pure 96.90 / calibrated fusion
**95.99** (small regression vs alignh_pure, not predicted by the 20-seq POT
probe — POT is heavily saturated, not yet root-caused). Details: docs/2026-08-07.md.

## 2026-08-11 — POT full-grid regression root-caused: coasted estimates get zero credit
11/210 POT sequences drop from perfect (100) to 60-80 under calibrated
fusion vs alignh_pure, explaining the full-grid regression the 20-seq
probe missed. First hypothesis (numerical failure in the covariance
inversion) was checked directly and disproven: covariance validity matches
`maskh_align_accepted` exactly, 0 mismatches. Real cause: the SAM-H
sub-tracker has `flow_enabled=False`, so on a rejected align frame it
coasts (`_finalize_failure` returns the unchanged last homography) — a
valid, usually-still-decent position but with no fresh covariance.
`fuse_corners` treats "no fresh covariance" the same as "branch absent,"
discarding a coasted-but-good `samh_H` and handing the frame to LoMa. Not
a numerical bug — a missing case (stale-but-recent vs never-measured).
Natural fix (a covariance that decays with frames-since-last-measurement
rather than vanishing outright) is a real design choice, not implemented
this session. Details: docs/2026-08-07.md.

## 2026-08-11 — HPC-local commits wiped by a workstation sync, reconstructed
User ran `make sjo_sync` before pulling this session's commits back to the
workstation — exactly the destructive scenario CLAUDE.md warns about
(`--delete` rsync + full `.git` resync). All 7 session commits and every
uncommitted change were wiped; no local backup existed. Run outputs under
`OUTPUT_DIRECTORY` were untouched (outside the repo), so no numbers were
lost. All code/docs reconstructed from the session's own conversation
transcript (exact diffs/content had been printed for nearly everything
while splitting commits) and recommitted. Details: docs/2026-08-07.md.

## 2026-08-11 — coast theory refuted; quad-free gap is the aligner's missing redetection
Measured before building: a coasted SAM-H estimate is wrong, not stale (PTi
median coasted error 271px per sequence; 0/300 sequences under 15px), so the
previous entry's decaying-covariance fix was implemented and then reverted.
New per-branch readout shows where the quad-free deficit actually lives:
`samh_H` alone scores 82.90 p@15 with the Hough quad vs **41.14** with mask
align, while `loma_H` is identical (78.2) — and an exact offline fusion-rule
replay puts even *oracle* SAM-H acceptance at 82.77, 5pp short of the quad
baseline. Cause, verified frame-by-frame: on frames the aligner rejects it is
wrong 100% of the time while the stateless quad branch is right on 76.3% of
those same frames — the aligner has no way back once its last-frame init
drifts off the object. Implemented mask-moment redetection (any-shape, fires
only on would-be-rejected frames) as `hough.align_redetect_moments` +
`configs/tracker/corner_fusion_alignh_redetect.py`; probe running. New offline
tools: `experiments/{coast_covariance_study,samh_accept_study,branch_p_at_k,
fusion_rule_replay}.py`. Details: docs/2026-08-07.md.

## 2026-08-11 — moment redetection: first probe sequences strongly positive
On the first 4 PTi probe sequences the mask-align SAM-H branch goes 33.97 →
**66.66** p@15 and the fused output 76.18 → **84.68** (LoMa identical at 73.44,
so it is a controlled comparison); align acceptance rises from ~50% to 94-98%
of frames. Escalated straight to the full 300+210 grid
(`corner_fusion_alignh_redetect`, jobs 11329911/11329913) rather than
finishing the 20-sequence probe. Details: docs/2026-08-07.md.

## 2026-08-11 — offline mask→H benchmark built; quad-free registration beats the quad
Built `experiments/maskh_bench.py`: 2778 fixed cases (POT all GT frames + seeded
PTi sample), dev/test split by sequence hash, with the *initialization* as a
controlled variable (good/drift/lost/cold), so a mask→H estimator can be
measured in minutes on CPU instead of a GPU grid. Baselines showed the aligner
is competitive only from a good init (PTi cold: Hough 75.0 vs align+redetect
30.7). New `flatsam2/mask_register.py` (propose via second-moment affine
normalization / refine / verify) scores **79.6 cold, 79.6 lost** on held-out
PTi and **86.8 / 86.8** on POT — past the Hough quad by +4 to +15 there, -1.6
while healthy, and quad-free throughout. Key measured fact: proposals already
contained the answer (oracle 88.9) — the bottleneck was verification, because
silhouette aliases score identical IoU (0.94) while being 219 px wrong; texture
correlation breaks the tie (cold 65.9→81.5). Occlusion below 85% mask coverage
is unrecoverable for *every* method incl. the oracle (segmenter failure, not
registration). Integrated as `hough.method='register'` +
`configs/tracker/corner_fusion_register.py`; grid pending. Details:
docs/2026-08-07.md.

## 2026-08-11 — redetect full grid: regressed, and the cause is corner aliasing
`corner_fusion_alignh_redetect` completed (300+210) and **regressed**: PTi mean
p@15 80.59 → **74.47**, POT 95.99 → **92.46**, despite the 4-sequence probe
being strongly positive. The best-rotation metric names the cause exactly:
`mean_br_p@15` is 84.97 vs 74.47 direct, a **10.5 pp gap** affecting **35% of
PTi sequences** (predecessor configs: 0.05 pp gap, 0.3% of sequences). So
mask-moment redetection *does* find the object better than the plain aligner
(br 84.97 > calibrated's 80.65) but gets the corner labelling wrong a third of
the time — the identifiability caveat flagged when it was written, now
confirmed at grid scale. A silhouette cannot pin corner identity.
This is the same aliasing measured offline the same day (44.6% of wrong
hypotheses are pure cyclic relabelings) and is exactly what the texture
verification in `mask_register` exists to break; the 2-sequence register smoke
shows **zero** br-gap (85.5/71.0 vs redetect's 36.4/69.0, beating even the
quad-switch baseline 78.9/70.0). Full register grid running.

## 2026-08-11 — register full grid: best quad-free result, matches the quad baseline
`corner_fusion_register` completed 300+210. PTi mean p@15 **86.52** (p@5
**67.94**), POT **96.77**. Against the previous best quad-free config
(`alignh_calibrated` 80.59/95.99) that is **+5.93 / +0.78**, spread broadly
(better on 50.7% of PTi sequences, worse on 16.7%), and it clears published
SAM-H SOTA (80.1) by 6.4. Against the quad-based switch baseline
(`loma_samh_decoupled_rerun` 87.86/97.66) it is -1.34 / -0.89 — near parity,
27.7% of sequences better vs 30.3% worse — while **beating** it on PTi p@5
(67.94 vs 65.44), i.e. quad-free is now more precise where it lands. The
aliasing that sank redetect is largely gone: br-gap 2.83 / 11.3% of sequences
vs redetect's 10.50 / 35.0% (the quad baseline itself carries 1.57 / 6.7%).
Telemetry (120 seqs, 58k frames): incumbent wins 88.5% of frames, the stateless
search 8.3%, 3.2% produce nothing; 87.0% of frames accepted — so the fast path
dominates and cost stays near the plain aligner. Details: docs/2026-08-07.md.

## 2026-08-12 — keypoint proposal: quad parity on PlanarTrack, aliasing halved
`corner_fusion_register_kp` (registrar gets the keypoint branch's `loma_H` as an
extra proposal, verified not trusted) completed 300+210: PTi p@15 86.52 →
**87.32**, p@5 67.94 → **68.63**; POT flat at 96.65 (−0.12, three sequences
changed). It is an aliasing fix and behaves like one — br@15 barely moves
(89.36 → 89.57) while the best-rotation gap closes 2.83 → **2.25** on 9.3% of
sequences instead of 11.3%, i.e. the gain is corner *labelling*, not detection.
Against the quad switch baseline the quad-free tracker is now **−0.54 PTi p@15**
(87.32 vs 87.86) but **+3.19 p@5** and **higher br@15** (89.57 vs 89.43), so
what remains of the deficit is labelling alone. Kept as the best quad-free
config. `corner_fusion_register_noapp` (texture verifier off) running to size
how much of the disambiguation the appearance term carries.
Details: docs/2026-08-07.md.

## 2026-08-12 — evaluation widened: POIC, TMT, UCSB, MPOT-3K wired up
Added four datasets beyond POT/PT. POIC (20), TMT (**92** — fetched the 4 missing
`nl_bookI_s*` and the 8-sequence robot category that had never been downloaded)
and UCSB (**96**, new; its original UCSB home is dead, taken from the MTF mirror
at U. Alberta) share one annotation format, so one `corners_txt` loader
parametrised by the registry serves all three. MPOT-3K is multi-object: the 54
test videos expand to **573 tasks**, one per planar instance
(`<video>__<instance>`), each starting at its own first annotated frame. Its
official eval is the MOT-Challenge MATLAB devkit (MOTA/IDF1/SR on box IoU, MATLAB
available here as modules 9.7–24.1), and per-instance corner predictions
concatenate straight into its submission format — so compliance and one-run-per-
object are the same thing, with the caveat that GT init makes ID-switches
structurally zero and our MOTA not comparable to their detector-driven numbers.
Three data defects found and fixed: `nl_bookI_s1.txt` was truncated mid-line
(re-downloaded), TMT robot sequences pad image and annotation names to different
widths, and `robot_bookI` numbers frames from 0 but annotations from 1 — so GT
rows are matched to frames **by position** when the counts agree (189/208
sequences), which an earlier frame-number+trim approach got wrong by one frame
throughout that sequence. All 781 sequences/tasks load cleanly
(`experiments/check_new_datasets.py`, CPU-only). Nothing tracked yet; manifest
pipeline `cross_dataset` is registered with `corner_fusion_register_kp` and the
quad baseline. Details: DETAILS.md → Datasets.

## 2026-08-12 — appearance ablation confirms texture is what names the corners
`corner_fusion_register_noapp` (verifier ranks on shape IoU only,
`register_appearance_power=0`) completed 300+210: PTi p@15 86.52 → **81.65**,
POT 96.77 → 95.00. Detection is almost untouched — br@15 only 89.36 → 88.67 —
while the best-rotation gap blows out 2.83 → **7.02** pp on 27.0% of sequences
(vs 11.3%), so **86% of the appearance term's value is corner labelling, not
finding the object**. Matches the pre-registered prediction that the gap would
open toward redetect's 10.50 pp / 35%; it reached two-thirds of the way. Losses
are total rather than gradual (Seq_00795 −62.0, Seq_00247 −60.0), the signature
of a pose snapping to a symmetry. Texture verification is worth 4.18 pp of
labelling, the keypoint proposal a further 0.58 pp on top.
Details: docs/2026-08-07.md.

## 2026-08-12 — state-of-the-tracker document written
`docs/state-of-the-tracker.md` collects everything since the 35-inlier switch
baseline: an ELI5 half, then the algorithms (corner-covariance fusion, quad-free
propose/refine/verify registration, keypoint proposal), how it was measured (the
offline maskh_bench with initialization as a controlled variable, the br-gap
alias detector), and the full result tables — headline grids, POT by condition,
PTi by mask quality, and single-thread cost. Written to decide what to do next;
open questions listed at the end (the residual ~9.3% labelling sequences,
occlusion vs segmenter failure, the four unrun datasets).

## 2026-08-13 — speedups measured; candidate final config submitted
Profiling named the real cost: the signed-distance field is **47% of a
fast-path frame** (33 ms of 70), not the search. Two changes measured offline on
dev: cropping the SDF to the mask bbox + margin (`sdf_roi_margin`) is
**accuracy-neutral to within 0.2 pp in every regime on both datasets at up to
-46% time**, and is exactly equivalent by construction (verified: identical to 4
decimals when the crop covers the samples); the alias shortcut + early exit
(score the 3 cyclic relabelings of a refined pose — same silhouette, different
corner names, no extra GN solve) gives **+1.1 pp PTi lost/cold at slightly less
time**. The 3x3 chamfer SDF is another ~20% and looks neutral, but produced one
hard failure in 513 POT cases, so it stays an ablation. New config
`corner_fusion_register_final` = register_kp + ROI crop + alias/early-exit;
full grids submitted (11343591 PTi / 11343593 POT). Also: `--no-mail` added to
`slurm/submit{,_cmd}.py`, and `profile_tracker.py` fixed (it imported the two
loader factories removed in the dataset refactor).

## 2026-08-13 — kp bench discrepancy explained; handoff written
The offline `register_kp` rows that matched plain `register` exactly were a
missing input, not a null result: `maskh_bench` fills `ctx["extra_H"]` only when
given `--extra-h-run`, so without it the keypoint method is byte-identical to
its twin. The bench now refuses to run a keypoint method without the flag
(methods carry `needs_extra_h`). Re-run with the flag: job 11343605. Earlier
`kp_test.csv` numbers were produced with it and stand. Also recorded that a
`p90_err = inf` in the bench table is a sentinel for "at least one non-finite
error" (1 case in 513 here), not a 10% failure rate — an earlier reading of that
column was wrong. State + traps + next steps in `docs/HANDOFF-final-candidate.md`.

## 2026-08-13 — end-to-end profile corrects where the time goes
`profile_tracker.py` on `corner_fusion_register_kp` (POT V01_1, 74 frames, CUDA
synced): **301 ms/frame = 3.32 FPS**. LoMa keypoints **38.7%** (descriptors alone
20.9%), SAM3 **32.0%**, LoMa match **18.6%**, the whole mask->H registration
branch only **9.1%** — and the signed-distance field inside it just **2.4%**.
So the earlier "SDF is 47% of a frame" is 47% *of the isolated branch measured
single-threaded* (`maskh_timing` pins cv2/BLAS to one thread on purpose); with 8
cores `distanceTransform` drops 33 ms -> 7.4 ms. The ROI crop is still free and
correct but worth ~1-2% end-to-end, not 40%. **Throughput is LoMa + SAM3 (~89%
of the frame); nothing in the quad-free thread touches it.** Also: the corrected
kp bench (with `--extra-h-run`) reproduces the documented offline numbers — PTi
lost 80.6 -> 83.4, POT lost 81.1 -> 86.2 — so those stand.

## 2026-08-14 — `corner_fusion_register_final` grids land, parity cleared on both
Full PTi (300/300) + POT (210/210) grids for `corner_fusion_register_final`
(= `register_kp` + cropped SDF `register_sdf_roi_margin=1.0` + corner-alias
shortcut with `register_sweep_stop_score=0.70`) finished; all 20 array tasks and
both evals exit 0. **PTi 69.00 P@5 / 87.87 P@15, br-gap 1.71 (8.3% of sequences
over 5 pp); POT 91.98 / 96.77, br-gap 0.00 (0%).** Against `register_kp` (68.63 /
87.32 / 2.25 / 9.3%; POT 91.70 / 96.65) that is **+0.55 pp PTi P@15** with the
best br-gap of the three configs, and POT back to the 96.77 that plain
`corner_fusion_register` had. The knobs only touch the lost-frame search, so
parity was the bar and it cleared it — no regression on either dataset. This is
now the reference version; ablations compare against it. Table + context in
`docs/HANDOFF-final-candidate.md`.

## 2026-08-15 — legacy WOFTSAM / SAM-H runnable from flatsam2, parity confirmed
The frozen SAM2 trackers in `~/git/flatsam` are now driven from this repo by
`experiments/run_legacy.py`: flatsam2 supplies sequences, GT, init corners and the
output layout, the legacy package supplies the tracker. They run under the **legacy
venv** (the one with `sam2` + `kornia`, which also satisfies flatsam2's deps);
`slurm/submit.py --legacy-config` reuses the normal array/pack machinery via new
`FLATSAM2_ENV_FILE` / `FLATSAM2_ENTRY` hooks in `slurm/run.sbatch`, and
`evaluate.py --results-name` scores a directory that is not a flatsam2 config stem.
Parity vs the stored POT/PTi runs: **SAM-H is deterministic and reproduces `sam_H/`
bit-for-bit** (300/300 PTi sequences, POT within 1e-6 text rounding) — which also
shows `WACV2026-SAM-H.py` ≡ `sam_H.py` under current legacy code, while the older
`SAM_H/` and `WACV2026-SAM-H/` directories come from an earlier code state and sit
0.53 p@15 lower. WOFTSAM is stochastic (RANSAC + flow): POT 200/210 sequences
identical, p@15 96.37 vs 96.75; PTi p@15 78.20 vs 77.37, with divergence starting
at frame 2–70 and identical frame counts and init corners, i.e. scatter not wiring.
Resource note: the legacy API holds a whole sequence of decoded frames in RAM —
POT parts peak ~17 GB, PTi parts ~40 GB, so `--mem 48G` (`--mem 24G` for POT) and
`--pack 2` (GPU is 43% utilised, 5.7 GB of 40 GB). Full runs on POIC, TMT, UCSB and
MPOT-3K submitted for both trackers, results under
`{woftsam_samfallback_nl,sam_H}/<dataset label>/`. Manifest pipeline:
`legacy_baselines`; details in DETAILS.md → "Legacy baselines".

## 2026-08-15 — legacy baselines complete on POIC, TMT, UCSB, MPOT-3K
Both frozen trackers now have numbers on all four new benchmarks (mean p@15 / best-rotation p@15):

| dataset | n | WOFTSAM | SAM-H |
|---|---|---|---|
| POIC | 20 | 98.82 / 98.83 | 63.04 / 63.05 |
| TMT | 92 | 89.20 / 89.65 | 58.07 / 60.86 |
| UCSB | 96 | 96.30 / 96.31 | 68.61 / 68.88 |
| MPOT-3K | 573 | 89.74 / 89.97 | 80.46 / 81.97 |
| POT-210 | 210 | 96.75 / 96.75 | 88.38 / 89.54 |
| PTi | 300 | 77.37 / 78.97 | 80.34 / 82.95 |

**PlanarTrack is the only benchmark where the mask branch beats flow.** Everywhere else
SAM-H trails WOFTSAM by 8–36 pp, and the gap is largest exactly where the geometry is
tame: POIC (fixed geometry, varying illumination) −35.8, UCSB (short close-ups) −27.7,
TMT −31.1, against −8.4 on POT and −9.3 on MPOT-3K. Best-rotation barely moves SAM-H on
POIC/UCSB (+0.0/+0.3), so this is not corner labelling — the mask branch is losing pose
outright, unlike on PTi where br lifts it +2.6. Worth reading as a caution for the fusion
work: the SAM-H branch's competitiveness on PlanarTrack does not generalise, and a
tracker that leans on it should be checked on POIC/UCSB before it is called general.
Ops note: POIC/TMT have 2000–2666-frame sequences that need **128 G** per part (legacy
RAM ≈ 52 bytes/pixel/frame); 64 G OOMs on exactly those, so both datasets needed a
gap-fill pass. Results in `{woftsam_samfallback_nl,sam_H}/<label>/`; POT/PTi parity
reruns kept separately as `*_rerun/`.

## 2026-08-17 — the two current trackers run on all six datasets; UCSB is a regression
`loma_samh_decoupled_rerun` (the 35-inlier quad switch) and
`corner_fusion_register_final` (best quad-free) now have POIC/TMT/UCSB/MPOT-3K next to
their POT/PTi grids. Mean p@15 / best-rotation p@15:

| dataset | WOFTSAM | SAM-H | thr35 switch | quad-free final |
|---|---|---|---|---|
| POIC (20) | 98.82 / 98.83 | 63.04 / 63.05 | 98.97 / 98.98 | 98.94 / 98.95 |
| TMT (92) | 89.20 / 89.65 | 58.07 / 60.86 | **93.99 / 94.52** | 93.01 / 93.01 |
| UCSB (96) | **96.30 / 96.31** | 68.61 / 68.88 | 93.76 / 93.77 | 93.09 / 93.10 |
| MPOT-3K (573) | 89.74 / 89.97 | 80.46 / 81.97 | 90.98 / 91.06 | **91.38 / 91.40** |
| POT (210) | 96.75 | 88.38 / 89.54 | **97.66** | 96.77 |
| PTi (300) | 77.37 / 78.97 | 80.34 / 82.95 | 87.86 / 89.43 | **87.87 / 89.58** |

Two things the POT/PTi grids could not show. **UCSB is a regression**: plain legacy
WOFTSAM beats both current trackers (96.30 vs 93.76 / 93.09), the only dataset where the
old flow-only tracker wins, and br ≈ p@15 for everyone so it is not corner labelling —
on short close-up sequences the newer machinery loses accuracy the old tracker keeps.
And the **quad-free/switch ordering does not hold outside PlanarTrack**: quad-free is
ahead only on MPOT-3K (+0.40) and PTi (+0.01 p@15, +0.15 br), behind on TMT (−0.98),
UCSB (−0.67), POIC (−0.03) and POT (−0.89). PTi is where the two were chosen apart,
so the +0.55 pp that made `final` the reference is a PlanarTrack-local gain, not a
general one. Ops: `corner_fusion_register_final` OOMs at the default 16 G on TMT's three
2518-frame `nl_bookII` sequences (MaxRSS 16.6 G) — flatsam2's tracker does grow with
frame count despite streaming frames; those needed a 48 G gap-fill.

## 2026-08-17 — tracker cards: mechanisms written down, figures generated from the code
`docs/tracker-cards.html` is a standalone design record of every variant (WOFT/WOFTSAM/
SAM-H assumed known, everything else defined): the GN objective as equations
(`r_i = D_t(H(C)·b_i)`, Huber δ=5 px, IRLS + Levenberg damping on the 8 control-point
coordinates), both branches' corner covariances (`σ̂²(JᵀWJ)⁻¹` in corner space because θ
*is* the corners), the propose/refine/verify search, and the fusion rule as pseudocode.
Built by `experiments/build_tracker_cards.py`, which reads every cited
`eval_results.json` and the figure facts — **no number is transcribed by hand**.
Figures come from `experiments/register_visuals.py` running the deployed code
(`register_mask`, `_gn_align`) on PTi Seq_00006 frame 30 with real SAM3 masks: the SDF
with its boundary samples, the GN iterates and cost curve, the 24 sweep proposals (whose
score plot shows **four near-equal peaks 90° apart** — the aliasing, visible), the four
cyclic relabelings (IoU 0.937–0.966 vs NCC −0.294…+0.697), and the fusion decision read
back from `<seq>_debug.npz`.
Two things the writing forced into the open. **The metric**: the page reports pooled
`overall_p@k` (all valid frames counted once), which is what the benchmark success plots
do; `state-of-the-tracker.md` and older journal entries used per-sequence means, and the
two disagree — e.g. MPOT-3K, where the mean puts FINAL +0.40 pp ahead of SWITCH35 while
pooled has them level. Prefer pooled. **The calibration**: the deployed fusion inflates
the branch covariances ×32 (keypoint) and ×70 (mask) and then gates at d²=20 against a
χ²₂ 95% quantile of 5.99, so it is a relative precision model with an empirical scale
correction, not calibrated probabilistic fusion — stated as such in §5.2.
Also found while generating the verify panel: on that forced-lost frame the pipeline's
output scores 0.770 while its own +1 cyclic relabeling scores 0.859 and is 20.9 px from
GT against 342 px — the alias shortcut compares relabelings *before* the full-resolution
solve and never re-checks after it. One frame, a lead not a rate; cheap to test.

## 2026-08-17 — alias re-check fixed; the per-corner fusion story does not survive measurement
Four things came out of interrogating the tracker cards rather than polishing them.

**Alias re-check (fixed).** `alias_shortcut` compares relabelings of *coarsely* refined
proposals; the full-res solve then moves the pose and nothing re-examines its naming.
The naive fix (re-score the winner's three relabelings) **loses 3.0 pp P@15 in `good`**
while gaining 0.9 in `lost` — a naming inherited from the previous frame carries temporal
evidence the single-frame score does not. Requiring an inherited naming to be beaten by
the same `incumbent_margin` a challenger pose must clear fixes it: `good` unchanged 86.6,
`drift` 84.1→84.4, `lost` 79.3→80.1, `cold` 79.4→79.8, and p90 error when lost 151→129 px
(maskh_bench, PTi test, 961 cases/regime). Shipped as `hough.register_alias_recheck`,
config `corner_fusion_register_recheck`; full grids running.

**Per-corner fusion is mostly not doing what it is sold as doing** (`experiments/
corner_complementarity.py`, PTi, 122 195 frames). Frames where the two branches are
complementary — neither alone within 15 px on all four corners, together yes — are
**0.66%** (0.60% for "three good corners plus one borrowed"). Per-corner oracle vs
better-whole-branch oracle: **+0.48 pp** P@15 (94.33 vs 93.85), while deployed fusion is
91.35. Almost all headroom is *branch selection*, not corner mixing. What per-corner does
buy is variance reduction: on the 69.2% both-good frames the median error is 3.18 px
(keypoint), 2.81 (mask), **2.61 fused**, oracle 2.47. Also found: on **4.06%** of frames
the keypoint branch has a pose but no covariance and the rule discards it entirely
(Seq_00350 f275 costs 90 px that way) — a covariance fallback is the obvious fix.

**Covariance validated, and its optimism explained.** Bootstrap re-solves on real data:
the closed form matches an i.i.d. resample (median ratio 1.32×) but the *moving-block*
bootstrap is 2.92× wider, because boundary residuals are spatially correlated —
measured autocorrelation time τ_int = 5.1 samples, so N_eff = 31 of 156. √τ_int = 2.25×
predicted vs 2.92× measured. That derives ≈8.5× of variance from first principles; the
deployed ×70 mask fudge is not derived, and the rest is model error no resampling can see.

**The verification score's exponents are not identifiable.** score = IoU^a·((1+NCC)/2)^b
is a log-linear pool of two evidence terms. Over a:b from 2:0.5 to 1:1 the metric moves
≤0.7 pp anywhere; a=2,b=1 is as good or better everywhere except 0.1 pp in `good` and cuts
p90 when lost. Dropping the texture term entirely still costs 13 pp. Recommendation: fix
b=1 and stop calling the exponents tuned parameters.
Harness bug fixed on the way: `maskh_bench._needs_frames` keyed off the method *name*, so
`abl_*` methods silently ran shape-only whenever no `register*` method shared the
invocation — it now reads `needs_frames` off each method's config.

## 2026-08-17 — both fusion simplifications win; UCSB is a segmentation-scope failure
Acting on the review that the per-corner decision and the score's two exponents were
unjustified. Both simplifications *delete* parameters and both were decided offline.

**One fusion decision per frame beats four** (`fusion.decision_scope='whole_frame'`):
fuse all four corners, or take the whole pose from the branch with the smaller total
corner variance — never a quad neither branch proposed. Replayed exactly against the
stored branch estimates of `corner_fusion_register_final`
(`experiments/fusion_rule_replay.py`), pooled p@15: **POT 97.61 → 98.80**, **PTi 87.76 →
88.32**, against a whole-branch oracle of 99.00 / 89.14 — so it takes 86% of the
available headroom on POT and 40% on PTi. The covariance fallback for the 4.06% of
frames that discard a keypoint pose lacking a covariance was measured at the same time
and **loses** (−0.09 POT, −0.25 PTi): not shipped. Replay validated against the run's own
eval to 5 decimals on pooled p@15; the per-sequence mean differs only because one POT
sequence has zero GT frames and the eval scores it 0 — another reason to report pooled.

**The verification score is the plain product** IoU · (1+NCC)/2 (a=b=1). Offline bench,
test split, mean P@15 over the four regimes: POT 90.88 → **91.75**, PTi 84.38 → **84.40**;
worst-regime p90 error 14.5 → 11.6 px (POT) and 73.6 → 54.7 px (PTi). Only loss anywhere
is PTi `good`, −0.2. Two parameters gone, accuracy up.

Both ship as `corner_fusion_simple`; POT/PTi grids submitted (11362126/11362128).

**UCSB is not a registration regression** (`experiments/ucsb_failures.py`). The 3.21 pp
mean gap behind WOFTSAM is concentrated — 18 of 96 sequences, the worst 5 carrying 63% —
and lives in two categories: `unconstrained` (−12.73) and `perspective` (−9.86), while
`panning`/`zoom`/lighting are 100.00 for everyone. Conditioned on frames that actually
fail (err > 15 px), WOFTSAM's quad is the right size and displaced (scale 1.001) while
both mask-based trackers are **~18% too large** (1.174 quad-free, 1.179 thr35); on the
sequences they do not lose, all three sit at 1.000. The failure figure shows why: SAM3
segments the laminated sheet, GT annotates the picture printed on it. This is a target-
definition mismatch, not something `mask_register.py` can fix.

Docs: tracker-cards §3 rewritten as an explicit seven-step covariance recipe (what points,
what is done to them, why) with a new figure drawing J itself — boundary samples coloured
by their sensitivity to each corner, from the deployed code on real data.

## 2026-08-17 — docs consolidated onto one entry point
`docs/state-of-the-tracker.md` rewritten as **the** starting document: current config
(`corner_fusion_simple`), what is running, the numbered next steps, results global-first,
settled/open, and the working rules. The three `HANDOFF-*.md` files are deleted — their
live content moved into its "Start here" section and their traps into "Working rules";
per-thread handoffs are no longer a thing, the state doc is updated in place instead.
Abandoned directions moved to `docs/dead-ends.md`, each with the measurement that killed
it, so failures stay findable without cluttering the state doc. `CLAUDE.md`'s docs policy
updated to match. Entries above this line referencing `HANDOFF-*.md` are historical; the
content is in the two files named here.

## 2026-08-17 — the alias re-check does not survive its grid
`corner_fusion_register_recheck` PTi landed: **87.74 P@15 (pooled 87.62) against
`_final`'s 87.87 / 87.76**, with the br-gap *widening* 1.71 → 1.87; POT identical to the
decimal. So the knob is neutral-to-slightly-negative at grid scale, and the gap moved the
wrong way for something whose only job is corner naming. Offline had predicted `lost`
79.3 → 80.1 and p90-when-lost 151 → 129 px.

The disagreement is aggregation, not measurement error: `maskh_bench` weights `good`,
`drift`, `lost` and `cold` equally, while a real sequence takes the fast path on 88.5% of
frames — so a gain confined to `lost` is diluted roughly tenfold and any harm in `good`
is not. Recorded as working rule 9 (offline regime means screen ideas, they do not
predict grid deltas for search-path-only changes) and in `docs/dead-ends.md`.

Consequence: `corner_fusion_simple` inherits `register_alias_recheck` and therefore about
−0.14 pp, while the +1.20/+0.56 replay predictions for its two simplifications were made
against `_final`, which has no re-check. Next step after the simple grids land is to drop
the re-check and confirm — a third parameter deletion, now with a measurement against it.
Grids left running rather than resubmitted, so the two simplifications keep a clean
attribution against a known baseline.

## 2026-08-17 — `corner_fusion_simple`: POT confirms the replay exactly, PTi does not
Both grids landed. **POT: 98.80 pooled P@15 against `_final`'s 97.61** (mean 97.98 vs
96.77, median error 1.70 → 1.66 px) — `fusion_rule_replay` had predicted **98.80**, so the
grid reproduced the offline prediction to the decimal. That is the strongest validation
the replay method has had. **PTi: 87.70 pooled against a predicted 88.32**, i.e. no gain
over `_final`'s 87.76, and the br-gap widened 1.71 → 2.09 (>5pp sequences 25 → 27).

The PTi shortfall is not attributable from this run, and saying which change caused it
would be guessing: `_simple` moves three things at once against `_final` (alias re-check,
fusion scope, verification score). Worse, the replay's exactness assumption is violated by
one of them — re-fusing stored branch estimates is exact for a *fusion-rule* change, but
the score change alters which hypothesis the registrar returns, so `samh_H` and its
covariance are no longer the ones replayed. The two changes are not additive. Known
budget: 87.76 − 0.14 (re-check) + 0.56 (predicted scope) ≈ 88.18 expected vs 87.70
measured, so ≈0.5 pp is unexplained on PTi and none on POT.

`corner_fusion_simple_norecheck` submitted (PTi 11363239, POT 11363241): drops the knob
that already measured negative and cuts the confound from three deltas to two. If PTi
lands near 88.2 the design is confirmed on both datasets; if it stays near 87.7 then
whole-frame fusion's PTi gain does not transfer from replay to grid, and that becomes the
priority — the replay is the instrument much of this thread rests on.

## 2026-08-18 — the simplifications buy localization and pay it back in corner naming
`corner_fusion_simple_norecheck` landed, and splitting P@15 into localization and naming
resolves the PTi puzzle from yesterday. PTi pooled, across the four configs of this
thread: `br_p@15` (localization, naming ignored) **89.49 → 89.52 → 89.80 → 89.81**, while
the br-gap widens **1.73 → 1.89 → 2.10 → 2.26**, leaving `p@15` flat at 87.55–87.76. So
whole-frame fusion plus the product score genuinely improve *where the object is* by about
+0.32 pp and lose all of it to *which corner is which*. POT's br-gap is zero — it has no
naming problem — which is exactly why the same gain lands there undiluted as **+1.19 pp
(97.61 → 98.80)**, matching the replay's prediction to the decimal.

**Correction to yesterday's entry.** "The alias re-check does not survive its grid" was an
over-read of 0.14 pp. Adding the knob cost 0.14 pp and *removing* it cost 0.15 pp —
opposite operations, same sign, i.e. run-to-run noise. POT is 98.80 either way. Rule 9
keeps its a-priori reasoning but loses its worked example; new rule 10 says not to read
0.1–0.2 pp of PTi P@15 as a result at all, and to separate `br_p@15` from the gap first —
the gap is what actually moves reproducibly at this scale.

`corner_fusion_wholeframe` submitted (PTi 11363834): only `fusion.decision_scope` changed
against `_final`, deployed score kept. It says which of the two simplifications costs the
naming, and re-tests the replay's unmet PTi prediction of 88.32 under the one condition
the replay is exact for. POT deliberately not run — no naming problem there to diagnose.

## 2026-08-18 — the replay is exact on both datasets; the score simplification is not
`corner_fusion_wholeframe` (only `fusion.decision_scope` changed against `_final`) gives
**PTi 88.32 pooled P@15**, and `fusion_rule_replay` had predicted **88.32**. POT was
predicted 98.80 and measured 98.80. **The replay reproduces at grid scale to the decimal
on both datasets**, so a fusion rule can be decided against a finished grid for zero GPU.
Yesterday's "PTi did not reproduce" was my own contamination: the config under test also
changed the verification score, which the replay does not model, because it moves which
hypothesis the registrar returns.

**The product score is what costs PTi.** Adding a=b=1 on top of whole-frame fusion takes
88.32 → 87.55 (−0.77 pp), hurting localization (`br_p@15` 90.24 → 89.81) *and* naming (gap
1.93 → 2.26), while buying nothing on POT. The offline bench had called it +0.87 POT /
+0.02 PTi. That is working rule 9 with a genuine worked example at last, and the earlier
alias-re-check example is withdrawn as noise.

Shipped config is now `corner_fusion_wholeframe`: `corner_fusion_register_final` plus the
whole-frame fusion decision, deployed score kept at a=2, b=0.5. PTi 88.42 mean / 88.32
pooled, the best in the thread; POT grid submitted (11363864), predicted 98.80.

**Open for the maintainer:** the KISS argument for deleting the two score exponents does
not survive measurement — they are worth +0.77 pp PTi. Either keep them and report them as
tuned (which means running the a:b ablation the simplification was meant to avoid), or
accept the loss for the simpler form. Not decided here.

## 2026-08-18 — `corner_fusion_wholeframe` is the tracker; POT confirms at 98.80
The POT grid landed at **98.80 pooled / 97.98 mean P@15 / 92.69 mean P@5**, the figure the
replay predicted. With PTi at 88.32 pooled / 88.42 mean / 69.07 mean P@5, whole-frame
fusion is **the best config in this thread on every cell of both benchmarks** against
`_final`'s 97.61 / 87.76 pooled. No regression anywhere.

Docs and manifest updated to name it: `state-of-the-tracker.md` §0/§3.1/§5,
`experiments/pipeline.yml` active_configs, and `tracker-cards.html` §6.5 (which now records
the exact replay-vs-grid confirmation) plus a new §6.6 stating what the rule does *not*
fix — on PTi br-P@15 rises 89.49 → 90.24 while P@15 rises 87.76 → 88.32, so the widened
br-gap (1.73 → 1.93) is the whole remaining deficit, and the rule itself is a suspect:
taking the whole pose from the more confident branch adopts that branch's *naming*, and the
mask branch cannot name corners. Not yet tested.

`corner_fusion_simple` / `_simple_norecheck` are retained as the reverted score-simplification
rungs, not as alternatives.

## 2026-08-18 — the KISS score was already measured; corner naming is the next 0.82 pp
**No new grid was needed for the a=b=1 question.** `corner_fusion_simple_norecheck` differs
from the shipped `corner_fusion_wholeframe` by *exactly* the two exponents and both grids
are complete: PTi **88.32 → 87.55** pooled (br-pool 90.24 → 89.81, gap 1.91 → 2.26,
P@5 69.07 → 68.69), POT identical at 98.80 with P@5 92.69 → 92.50. The plain product is a
clean single-variable loss of 0.77 pp PTi and buys nothing anywhere. (`corner_fusion_simple`
is the same plus the alias re-check, which is noise.)

**Corner naming, the whole remaining PTi deficit, is repairable from the keypoint branch.**
New `experiments/naming_diagnosis.py` reads the shipped run: 2567 PTi GT frames (1.93%) on
68/300 sequences are mis-named, and **89.5% of them are frames where the mask branch
supplied the pose** (`gated_samh` 50.4%, `samh_only_cov` 37.7%) — the failure a silhouette
cannot avoid. On those frames LoMa's own naming is GT's on 72.3% while its *position* is
within 15 px on only 44.5%: naming and localization are separable.

So the mask branch now keeps its positions and borrows LoMa's naming —
`fusion.keypoint_naming_margin` permutes the SAM-H quad to the cyclic rotation nearest the
LoMa quad, but only when that rotation is `margin`× closer than the branch's own naming
(taking the nearest unconditionally keeps only +0.28 pp of the +1.01, i.e. the guard is worth 0.73 pp). Exact replay on the full PTi test
grid: **88.32 → 89.14 pooled (+0.82)**, br-pool 90.24 → 90.26, **gap 1.93 → 1.12** — the
gain is entirely in naming, and it *reaches the whole-branch oracle* (89.14), a ceiling
that cannot fix a naming both branches get wrong. POT is exactly neutral at every margin.
Margin plateaus over 1.5–3.0 on the full set and on both deterministic halves, so 2.0 is a
shoulder, not a fitted knob.

Shipped candidate `configs/tracker/corner_fusion_naming.py` (one field against
`wholeframe`); confirming grids submitted (PTi 11364231, POT 11364233) at `--time 1:30:00`
after measuring the previous grids at 40/28 min per array task — the default 5 h
reservation was pushing the start ~6 h out by defeating backfill.

Also new: a standalone teaching derivation of the per-corner covariance (published
artifact, not in-repo) — 1-D ruler → implicit differentiation of the normal equations →
sandwich → σ²(JᵀWJ)⁻¹, with every shape stated, a hand-checkable worked example, and §12
naming each assumption the deployed system violates.

## 2026-08-18 (later) — naming moves from appearance to correspondences, in both trackers
Two texture mechanisms were doing one job. The registrar's verifier already used NCC to
break naming ties, so overriding its naming afterwards with a *different* texture currency
(distance to the keypoint quad) was incoherent. Deleting NCC entirely
(`corner_fusion_register_noapp`) costs almost nothing in localization and destroys naming
— PTi br-P@15 89.49 → 88.64 while the gap goes 1.73 → 6.98 — i.e. **NCC is 86% a naming
mechanism**.

So: **shape decides where, correspondences decide which corner.** New
`flatsam2/match_naming.py` scores a hypothesis by how much of the matcher's raw,
pre-RANSAC correspondence set it explains, `U = Σ max(0, 1 − d²/τ²)`, τ = 30 px. Pre-RANSAC
because inliers are selected under a homography that already carries one of the four
namings; truncated rather than L1/Huber because a gross mismatch then saturates equally
under all four namings and cancels in the comparison.

Two uses, both off by default so every existing config reproduces byte-for-byte:
`hough.register_support_power=1` multiplies `U/N` into the registrar's verifier (with
`appearance_power=0` the score is `IoU · U/N` — one shape term, one texture term, no
exponents; and since IoU ties across cyclic relabelings by construction, naming stops
being a separate step), and `symmetry.method='matches'` replaces the DINOv2 vote in the
quad-based baseline, deleting six parameters and a second network.

Replay on the shape-only grid predicts PTi **88.73** (br 90.88, gap 2.15) / POT 98.41
against 88.56 / 98.80 for the NCC version — a wash on accuracy for a much smaller
mechanism, and a lower bound since the stand-in needed a keypoint *pose*.

Configs queued: `corner_fusion_matchverify`, `loma_samh_matchnaming` (both PTi+POT);
`corner_fusion_simple_naming` cancelled from the queue as superseded (its number is
exactly predicted). Every run now stores per-frame naming support + residual histograms
with per-bin sums of squares, so τ, a margin or an abstain rule can be re-tested offline
without a grid.

Also today: the margin figure of the superseded naming rule was corrected — dropping it
costs 0.73 pp, not the 0.44 pp first reported, and its plateau spans 1.35–3.0.

## 2026-08-18 (later still) — the SAM-H per-corner covariance was sliced in the wrong order

`_corner_cov_from_boundary_fit` flattened the control points xy-major
(`x1x2x3x4y1y2y3y4`) but sliced 2x2 blocks at rows/cols `2k, 2k+1`, the corner-major
convention, so the four "per-corner" covariances were really `(x1,x2)`, `(x3,x4)`,
`(y1,y2)`, `(y3,y4)`. Found by porting the estimator to JS for the demo export and
diffing against Python. Fixed the next day: both branches now go through
`corner_covariance.corners_to_theta` / `corner_blocks`, one definition of the layout,
with tests that fail on any block-to-corner mismatch. Every `corner_fusion` result on
disk was computed with the wrong slicing and was deleted; see 2026-08-19.

## 2026-08-18 (later still, 2) — the naming rule is confirmed on hardware, exactly as replayed
`corner_fusion_naming` (PTi, 300 seq) landed at **89.14 pooled / 90.26 br-pool / gap 1.12
/ mean 89.25 / P@5 69.57**, against `corner_fusion_wholeframe`'s 88.32 / 90.24 / 1.93 /
88.42 / 69.07. `fusion_rule_replay` had predicted 89.14 / 90.26 / 1.12 — **exact on every
figure**, the third time the replay has reproduced a grid to the decimal.

So borrowing the keypoint branch's naming is worth **+0.82 pp pooled, taken entirely out
of the naming gap** (1.93 → 1.12) with localization unmoved (90.24 → 90.26), plus +0.50 pp
of P@5. This is the *superseded* distance-based namer; the correspondence-support version
(`corner_fusion_matchverify`) is running now and should be at least as good, since it also
works on frames where the keypoint branch produced no pose.

Grids in flight: `corner_fusion_matchverify` and `loma_samh_matchnaming`, PTi+POT, packed
at ~4 sequences per task with a 15-minute limit. Both smoke runs passed (2m07s / 1m58s per
sequence; the baseline is *faster* now that DINOv2 is not loaded).

## 2026-08-19 — one texture term cannot do both jobs: `matchverify` loses 2.37 pp
`corner_fusion_matchverify` (shape-only verifier + correspondence support, NCC deleted)
came back at **86.77 pooled / 88.49 br-pool / gap 1.72 / P@5 69.10** on PTi 300, against
`corner_fusion_naming`'s 89.14 / 90.26 / 1.12 / 69.57.

**The loss is localization, not naming.** The gap *improved* over no-naming (1.93 → 1.72),
so correspondence naming works; but `br_p@15` fell 89.81 → 88.49, i.e. deleting NCC cost
**1.32 pp of pose accuracy** the support term did not replace.

**Why:** with τ=30 px a pose 10–20 px off still explains nearly all its matches, so `U/N`
is flat across nearby poses — it saturates. It discriminates *namings* superbly (aliases
are hundreds of px apart, ~180× separation) and *poses* barely at all. NCC degrades
continuously with pose error and so ranks poses. One term cannot do both jobs.

**The offline proxy was 1.96 pp optimistic** (predicted 88.73): it re-fused the `noapp`
run's stored branch estimates, which does not model a verifier that returns *different
hypotheses*. Working rule 9 again, in a case that looked safe.

Coverage finding from `naming_rule_replay` (26 seq, partial) predicted the residual gap
too: where correspondences exist naming is right 99.79% of the time, but the 5.7% of
localized frames without evidence are mis-named 17.26% and carry 84.5% of all mis-naming —
and with NCC gone those frames are verified by IoU alone, which ties exactly across the
four aliases.

**Next:** the natural config is both terms — `score = IoU · (1+NCC)/2 · U/N`, NCC ranking
poses and correspondences naming corners, each doing what it is good at. Not submitted yet;
waiting for `loma_samh_matchnaming`, which replaces DINO rather than NCC and so tests the
naming scalar without confounding it with pose ranking.

## 2026-08-19 — `matchverify` POT confirms it: NCC was doing naming work invisibly
POT 210: **98.21 pooled / 98.41 br-pool / gap 0.20**, against 98.80 / 98.80 / **0.00** for
both NCC configs. Two things repeat from PTi. Deleting NCC costs localization (−0.39 pp
here, −1.32 pp there). And it *creates* a naming gap where POT previously had none — 0.00
→ 0.20 — so NCC had been silently naming corners on POT as well, on the evidence-less
frames the coverage readout found (7.5% of localized frames, 21.79% mis-named, 74.9% of
all mis-naming).

The single-texture-term design is therefore worse on **both** datasets by **both**
mechanisms. `corner_fusion_matchverify` is settled as an ablation — a useful one, since it
isolates *why* the texture term cannot be deleted rather than merely showing that it
cannot.

Gap in the ablation table: `corner_fusion_naming` has no POT number (that half was
cancelled when it looked superseded; replay predicts 98.80, neutral, since POT's gap was
already 0.00). Needs running only if it ships rather than the two-term config.

## 2026-08-19 — the pairing claim holds: the same scalar helps the quad-based tracker too
`loma_samh_matchnaming` (PTi 300): **88.05 pooled / 89.41 br-pool / gap 1.36 / mean 88.18**,
against `loma_samh_decoupled_rerun`'s 87.68 / 89.25 / 1.57 / 87.86. So **+0.37 pp pooled
while deleting DINOv2 and six parameters**, mostly out of the naming gap (1.57 → 1.36) with
a little localization on top — it also removes DINO's occasional wrong flips.

`naming_shift` telemetry confirms it fired rather than no-op'd: the namer ran on **100%** of
57490 frames, kept the incumbent on 98.5% and re-named 1.50%. Adding that field was worth
it; a null result here would otherwise have been unreadable.

So the same scalar improves both trackers — quad-based +0.37 (and a network deleted),
quad-free +0.82 via naming — which is the paper claim: **corner naming is a
correspondence-support problem, not a property of either architecture**. It also isolates
`matchverify`'s failure: correspondence naming was never the problem there, deleting NCC
was. Here NCC was never in play (DINO was), so the naming gain shows up clean.

Submitted `corner_fusion_matchncc` (PTi 11368249, POT 11368251): `IoU · (1+NCC)/2 · U/N`,
one field against `corner_fusion_simple_norecheck`. IoU and NCC rank poses, `U/N` names
corners — each term on the job its sensitivity curve suits. Expected to beat 89.14: it adds
a namer measured at 99.4% accuracy where evidence exists, on top of a config that already
localizes at br 89.81, and unlike the distance-based namer it needs only *matches*, not a
successful keypoint pose.

## 2026-08-19 — quad-based tracker complete: DINOv2 deletable for free
`loma_samh_matchnaming` POT 210: **98.51 / 98.51 / gap 0.00**, identical to the DINO
baseline on P@15 and mean, with P@5 92.55 against 92.74 (−0.19, inside the noise band of
working rule 11). Expected: POT's naming gap was already 0.00, so there was nothing for a
namer to fix there.

Both datasets now read: PTi **+0.37 pp** (87.68 → 88.05, gap 1.57 → 1.36), POT neutral —
while deleting DINOv2 and six parameters (`only_when_lost`, `n_correspondences_enough`,
`dino_layer`, `abs_thr`, `fts_thr`, `run_length`). A simplification that costs nothing and
gains on the harder dataset, quotable independently of the quad-free thread.

## 2026-08-19 — `matchncc` fails on arithmetic, not evidence; support moves to alias-only
`corner_fusion_matchncc` (`IoU · (1+NCC)/2 · U/N`) came back **85.00 pooled / 85.93 br-pool
/ gap 0.93** — the **best naming gap measured anywhere in this thread**, on top of a
localization collapse (br 89.81 → 85.93, −3.88 pp).

The cause is arithmetic, not evidence. Every factor is ≤ 1, so a third one shifts the whole
score distribution down, and the registrar's thresholds are **not scale-free**. Measured
from the debug telemetry:

| config | median score | below `accept_score`=0.70 | incumbent wins | stateless search |
|---|---|---|---|---|
| `simple_norecheck` (2 factors) | 0.766 | 39.8% | 88.6% | 4.2% |
| `matchverify` (2 factors) | 0.821 | 28.7% | 86.4% | 4.4% |
| `matchncc` (3 factors) | 0.669 | 54.6% | 77.4% | 10.0% |

So the incumbent fast path collapsed and the stateless search ran 2.4× as often,
re-detecting poses that were already right.

Both failures now point the same way: **the support is evidence about naming, not about
pose**, so it must act only where naming is decided. `register_support_alias_only=True`
does that — `_alias_beats()` arbitrates the cyclic relabelings on support while
`verification_score` stays exactly `IoU · (1+NCC)/2`, keeping every calibrated threshold
intact. Falls back to the score when support is unavailable (<8 matches), which is the old
behaviour on precisely the frames with no naming evidence. `alias_recheck` keeps its
incumbent-margin asymmetry, now applied to support.

`corner_fusion_aliasname` submitted (PTi 11368401). POT held until PTi shows the
localization collapse is repaired.

## 2026-08-19 — placement fixed, coverage is the new bottleneck
`corner_fusion_aliasname` (support arbitrates relabelings, score untouched): **88.10 pooled
/ 89.75 br-pool / gap 1.65**, against its base `simple_norecheck`'s 87.55 / 89.81 / 2.26.

**The localization collapse is gone** — br 89.75 vs 89.81 is noise, where `matchncc` had
fallen 3.88 pp. So the diagnosis was right: the support must not touch the score, whose
thresholds are not scale-free.

But the gap only reached 1.65 against `matchncc`'s 0.93, and the reason is **coverage, not
evidence**: `alias_shortcut` runs *inside the stateless search*, which the incumbent
proposal makes unnecessary on ~88% of frames. On those the pose is accepted and its naming
is never re-examined — the best naming evidence we have is consulted on one frame in ten.
`matchncc` reached a better gap by accident: depressing the score below `accept_score`
forced the search 2.4× as often, buying naming at 3.88 pp of localization.

`alias_recheck` is the honest version — it scores the three relabelings of the *final*
winner every frame, whatever path produced it, for three verifications and no solve. It was
measured as noise (0.14 pp) when scored by `IoU · (1+NCC)/2`, because between relabelings
IoU is identical by construction and the comparison rested on NCC's weak naming signal
alone. The support separates namings ~180×, which is what should make the same check bite.

`corner_fusion_namecheck` submitted (PTi 11368658): built on `corner_fusion_wholeframe`
(a=2, b=0.5) rather than the plain product, because the two bases differ by 0.43 pp of
br_p@15 (90.24 vs 89.81) and with naming no longer the bottleneck that is what decides
whether it beats `corner_fusion_naming`'s 89.14. The a=b=1 variant is the ablation that
prices the exponents once naming is fixed — worth running, but second.

## 2026-08-19 — covariance layout fixed, every fusion result deleted and requeued

Both branches now build their 8-vector with `corner_covariance.corners_to_theta` and read
per-corner 2x2 blocks with `corner_blocks` — one definition of the layout, einops only, no
Fortran ravel. Five new tests fail on the old code: layout round-trip, block-to-corner
identity, per-corner starvation for each corner in turn, analytic-vs-bootstrap ranking
(bootstrap indexes corners independently of the layout), and mask-branch ellipse *shape*
with one outline edge deleted.

All 18 `corner_fusion_*` result directories (15093 files) were computed with the wrong
slicing and are deleted; `loma_samh_*`, `flow_hough_*`, maskh_bench and the LoMa/SAM3
dumps never read a covariance and are untouched. Requeued on the fixed code: `matchncc`,
`naming`, `wholeframe`, each on PTi + POT.

Also measured, on four GT-carrying frames picked for large per-corner spread: when the
mask branch converges cleanly onto the *wrong* silhouette (occlusion, residual < 1 px) it
reports sigma at the 0.25 px floor and wins the arbitration outright — keypoint 4.7 px vs
mask 132.4 px on V01_5:105, fused takes the mask. Where the mask branch fails loudly
(residual 69-147 px) its covariance is honest and the rule picks correctly. So the gate
detects the disagreement (mean d^2 ~ 1e4 against a gate of 20) but the tie-break prefers
confidence, and confidence is not correctness. A gross-error test, not a calibration
factor, is what that needs.


## 2026-08-19 — the covariance bug is fixed, and every corner_fusion number is void
The maintainer fixed the xy-major/corner-major mismatch, deleted the invalidated
`corner_fusion_*` results and re-queued `wholeframe`, `naming` and `matchncc`; the
`namecheck` array running against the old code was cancelled mid-flight.

The fix is structural rather than a patch: `corners_to_theta`, `theta_to_corners` and
`corner_blocks` in `corner_covariance.py` define the parameter layout once, and both
estimators build their Jacobians and read their per-corner blocks through them, so the two
sides cannot drift apart again. `_gn_align` now solves in the same layout too.

**So every corner_fusion figure recorded today is void** — wholeframe 88.32, naming 89.14,
matchverify 86.77, matchncc 85.00, aliasname 88.10. They shared the bug, so the *relative*
story (naming is worth ~0.8 pp; support saturates and cannot rank poses; the score's
thresholds are not scale-free; alias_shortcut only covers ~10% of frames) is probably
intact, but the absolute values are not and the ordering may not survive.

**Unaffected:** `loma_samh_*`, which selects by inlier count and never consumes these
covariances. DINOv2 deleting for free (+0.37 pp PTi, POT neutral) stands.

Still to re-run once the maintainer's grids land: `aliasname` and `namecheck`, whose
configs are unchanged by the fix.

## 2026-08-19 — external review: nine findings, all true, seven fixed
`docs/review-findings-2026-08-19.md` audited the covariance, naming, replay, occlusion and
SLURM work. Every finding checked out. Three changed conclusions rather than just code:

* **Alias enumeration skipped a naming.** Both alias loops rolled the *mutable* winner, so
  accepting shift 1 made the next rolls `roll(base,3)` and `base` — shift 2 was never
  scored. Fixed (`_best_naming`, frozen base, 3 tests). All results produced with it were
  deleted; only `corner_fusion_wholeframe` was affected (audit: every other stored result
  uses `hough`/`align` and never enters the registrar). **The 88.40 wholeframe figure is
  void.**
* **Naming telemetry conflated abstention with retention.** Recount:  the namer ran on
  **87.2%** of frames, not the 100% I reported — 12.8% abstained for fewer than 8 matches,
  85.7% looked and kept, 1.50% re-named. The +0.37 pp result stands (measured from
  predictions); the mechanism claim did not.
* **The covariance bug was worth +0.08 pp** on the shipped rule. Whole-frame arbitration
  compares the *summed* trace across four corners, and that sum is the full matrix trace
  however the blocks are cut — so the branch decision was nearly immune, and only the
  chi-square gate and per-corner weighting saw the error. That is why the thread's relative
  conclusions survived a genuinely wrong covariance.

Also fixed: `afterok` proving process exit rather than completeness (both scripts now exit
non-zero); the replay's `baseline` row being a rule the run never used (now read from the
config, **and checked against the run's own eval as MATCH/MISMATCH**); the occlusion screen
rasterizing a quad instead of warping the template silhouette, and predicting from the
previous frame; degenerate covariances floored into *maximum* confidence instead of being
refused; stale manifest and doc entries.

Left open deliberately: `samh_cov_scale=70` predates the fix and needs recomputing from
post-fix telemetry; and the two branches are fused as independent while the mask result is
partly LoMa-derived, which double-counts that evidence — a modelling question that deserves
a measurement, not a patch.

**Process note:** the degeneracy fix was edited while grids were running, splitting them
across two behaviours (3 sequences lost, restarted at HEAD `c73c080`). Jobs read the
*working tree*, not a revision, so any edit under `flatsam2/`, `run.py`, `evaluate.py` or
`configs/` changes what queued tasks do. Pinning the revision per array — recording it in
`.repro` and refusing a task whose revision differs — is the structural fix.

## 2026-08-19 — cross-dataset coverage for the quad-based candidate; calibration is unblocked
Two things while the re-queued `corner_fusion_*` grids run.

**`loma_samh_matchnaming` submitted on POIC / TMT / UCSB** (jobs 11370265/68/70, evals
11370267/69/71). It is the quad-based publication candidate and had results on PTi and POT
only, while its DINOv2 predecessor `loma_samh_decoupled_rerun` is complete on all four extra
datasets — so the comparison the paper needs did not exist off the two main benchmarks.
MPOT-3K (573 instances) is deliberately held back until the fusion grids clear the GPUs.
Added to `cross_dataset.active_configs` in the manifest.

**Recalibrating `samh_cov_scale` does not need a separate uncalibrated grid.**
`FusionTelemetry` stores the branches' *raw* covariances — the `*_cov_scale` factors are
applied only to the local copies used for the decision — so `fusion_calibration_check.py`
runs on the calibrated grids directly. Its docstring said otherwise; corrected.
Provisional read on the first 6 post-fix PTi sequences: implied `loma` 65.0, `samh` 146.8
against the deployed 32 / 70. Roughly 2x on both, but the *ratio* is 2.26 vs 2.19 deployed —
so the relative weighting is essentially unchanged and only the absolute chi-square gate is
mis-set, which is what the trace argument predicted. Re-measure at n=300 when the grid lands.

## 2026-08-19 — one table over all six benchmarks, and it names our weak column
`experiments/results_tables.py` only ever tabulated POT and PTi, so nothing showed the
claim the paper actually makes — *consistency*. Added `table_cross_dataset` (per-sequence
mean on all six, plus a `worst` column) and `table_cross_dataset_gap` (br-gap everywhere,
i.e. corner naming everywhere). Current state, p@15:

| config | PTi | POT | POIC | TMT | UCSB | MPOT | worst |
|---|---|---|---|---|---|---|---|
| `loma_samh_decoupled_rerun` | 87.86 | 97.66 | 98.97 | 93.99 | 93.76 | 90.98 | **87.86** |
| `loma_samh_matchnaming` | 88.18 | 97.66 | running | running | running | held | — |
| `woftsam_samfallback_nl` (prev. paper) | 77.37 | 96.75 | 98.82 | 89.20 | **96.30** | 89.74 | 77.37 |
| `sam_H` (prev. paper) | 80.34 | 88.38 | 63.04 | 58.07 | 68.61 | 80.46 | 58.07 |

We beat both baselines on five of six columns and beat WOFTSAM's worst column by 10.5 pp.
**The hole is UCSB: 93.76 vs WOFTSAM's 96.30.** It is the only benchmark where the previous
paper's tracker is ahead, so it is the one a reviewer will find. Re-check it when
`matchnaming`'s UCSB run lands; if the deficit survives, it needs an explanation in the
paper rather than a silence.

Also: the manifest pointed the legacy SAM-H baseline at `WACV2026-SAM-H.py`, whose stem is
a stale directory, so `hpc_status` reported 0/781 for a baseline that has been complete
since 2026-08-15 under `sam_H/`. Fixed.

## 2026-08-19 — UCSB is not an arbitration failure; closing it as a known, bounded gap
Asked the sharper version of the 2026-08-17 question. The scale argument (our quad ~18%
too large because SAM3 segments the laminated sheet) said the *mask* branch was at fault;
that is an argument, and the runs store both branch homographies every frame, so
`experiments/ucsb_failures.py::branch_attribution` measures it. Pooled over 6793 annotated
frames of `loma_samh_decoupled_rerun`:

| | within 15 px |
|---|---|
| shipped | 92.20% |
| keypoint branch alone | 90.65% |
| mask branch alone | 51.44% |
| **either branch (oracle selection)** | **93.49%** |

On the 530 frames the tracker gets wrong, **neither branch is within 15 px on 83.4%** of
them — keypoint 14.7%, mask 1.9%. So a perfect oracle over the two branches reaches 93.49
against WOFTSAM's 96.30: **no selection or fusion rule can close this gap**, because the
information is not in either branch to begin with. UCSB is a target-definition mismatch,
exactly as the scale measurement suggested, and not something arbitration can repair.

Per the maintainer (2026-08-19): UCSB is an old dataset and does not need a win — only
evidence of no breakdown. 93.76 vs 96.30 with a measured ceiling of 93.49 is that
evidence, and it is the honest sentence for the paper. **Closed; not to be reopened as a
tuning target.** Focus is PTi + POT; POIC/TMT/UCSB/MPOT are must-have coverage for the two
final methods only.

Also made `ucsb_failures.py` survive a config whose results were deleted, instead of
aborting the whole analysis.

## 2026-08-19 — the grids were queueing on one GPU family; they now use both
The six re-queued `corner_fusion_*` arrays plus the three cross-dataset ones were all
pinned to `amdgpu` (nodes g01-g12) and were getting 2-3 concurrent tasks, which put the
three PTi grids alone at roughly 40 h. The cluster also has an unused NVIDIA family
(`gpu`/`gpufast`, nodes n21-n32). Verified it runs our stack — torch 2.7.1, CUDA
available, Tesla V100-SXM2-32GB, `flatsam2` + `sam3` import, no flash attention (Ampere-
only, so a V100 task is slower per sequence but adds parallelism). Widened every pending
array with `scontrol update jobid=<id> Partition=amdgpu,gpu`; concurrency went 2 -> 5
immediately.

Consequence to state honestly: a grid can now span two GPU generations. Sequences are
independent and the pipeline is already stochastic (RANSAC), so this is scatter, not a
split behaviour — unlike the working-tree edit of earlier today, which genuinely changed
what queued tasks computed. Use `--partition amdgpu,gpu` on future submissions.

## 2026-08-19 — the selection ceiling for the published candidate, on PTi and POT
`experiments/branch_p_at_k.py` reported each branch's own accuracy but not the ceiling on
*choosing* between them, which is the quantity the whole two-candidate question turns on.
Added `oracle_p_at_k`: a per-frame oracle that takes whichever branch is closer. No switch,
threshold or covariance rule can beat it. Run on `loma_samh_matchnaming` (per-sequence mean):

| | PTi p@5 | PTi p@15 | POT p@5 | POT p@15 |
|---|---|---|---|---|
| keypoint branch alone | 56.59 | 78.26 | 92.13 | 97.07 |
| mask branch alone | 65.86 | 84.37 | 70.06 | 90.11 |
| **shipped (35-inlier switch)** | **65.48** | **88.18** | **92.55** | **97.66** |
| **oracle over the two branches** | **76.11** | **91.28** | **94.04** | **98.60** |

Three things worth carrying into the paper:

1. **The switch already beats both branches**, by 3.81 pp on PTi over the better one — it
   is not a fallback, it is extracting information neither branch has alone.
2. **PTi still has 3.10 pp of pure selection headroom** (88.18 -> 91.28), of which the
   switch currently captures 55%. That is precisely the case for the covariance-fusion
   branch, and the bar it has to clear: a fusion rule that does not beat 88.18 on PTi is
   not buying its complexity.
3. **POT is nearly exhausted** — 0.94 pp of headroom, 39% captured. POT will not
   distinguish the two candidates; PTi is the deciding benchmark.

Correction to the first version of this entry: it read the 10.63 pp PTi p@5 oracle gap as
"where the remaining selection value is". It is not. PlanarTrack GT carries ~5.7 px mean
annotation error with ~10% of annotations beyond 15 px, so PTi p@5 differences sit at the
noise floor — the reason this project already decided P@15 is the decision metric
(`experiments/selector_study.py`, A0/S1). The p@5 columns stay as diagnostics. Note also
that the shipped switch is *below* the mask branch alone at PTi p@5 (65.48 vs 65.86); at
P@15, the metric that decides, it leads by 3.81 pp.

## 2026-08-19 — the 35-inlier threshold is not load-bearing (but the cliff argument survives)
The published candidate switches branches at `inliers >= 35`, and the paper's own
motivation calls that an arbitrary cliff — so the number has to be defended, not just
used. `experiments/inlier_threshold_sweep.py` reconstructs the rule from stored branch
poses and inlier counts and sweeps it. **Self-check first**: the reconstruction reproduces
the run's own `selected` array on 100.00% of frames on both datasets (133320 PTi, 1004
POT), so it is the deployed rule, not a lookalike. Pooled P@15 at T=35 comes out 88.05 /
98.51 — the numbers already on record, from a completely different code path.

Per-sequence mean P@15 (`loma_samh_matchnaming`):

| T | 5 | 15 | 25 | **35** | 50 | 80 | 150 | 300 |
|---|---|---|---|---|---|---|---|---|
| PTi | 83.63 | 87.21 | 87.93 | **88.18** | 88.13 | 88.07 | 87.72 | 87.31 |
| POT | 97.84 | 98.13 | 98.13 | **98.13** | 98.13 | 98.13 | 97.72 | 96.29 |

**T=35 sits at the centre of a broad plateau.** Over T in [25, 80] PTi moves 0.25 pp and
POT does not move at all to two decimals; over [15, 100] the spread is 0.97 / 0.10 pp. The
tracker is not tuned to this number — anything from ~25 to ~80 gives the same tracker.

**What this does and does not answer.** It defuses "you tuned an arbitrary constant":
aggregate accuracy is insensitive across a 3x band, on both datasets, and the deployed
value is not perched on a peak. It does **not** answer the actual objection in §1, which is
about *per-frame coherence* — 34 and 36 inliers still produce completely different outputs
on near-identical evidence, whatever the aggregate does. That is an argument for the
covariance fusion branch and the sweep leaves it standing. Both sentences belong in the
paper; quoting only the first would be overclaiming.

Caveat recorded in the script: selection feeds back into the tracker, so off-deployment
columns are a first-order sensitivity, not what a re-run would score. The flatness is the
claim, not any single swept value.

## 2026-08-19 — the independence the fusion arithmetic assumes, measured instead of asserted
The 2026-08-19 review left open that the two branches are precision-averaged as if
independent while the mask result is partly LoMa-derived. That is testable: the branches'
signed corner errors against GT are on disk for every frame. Added
`branch_error_correlation` to `experiments/corner_complementarity.py`.

On `loma_samh_matchnaming` PTi, 122195 frames over 300 sequences — the **decoupled**
tracker, where the branches share nothing by construction, so this is the natural baseline:

| correlation | value |
|---|---|
| signed x error | **0.007** |
| signed y error | **0.014** |
| error magnitude | **0.019** |

Essentially zero on all three. So the independent-measurement arithmetic is not merely
convenient — when the branches *are* decoupled, their errors really are uncorrelated, and
precision-weighted averaging is the right combination rather than an approximation. The
magnitude column matters as much as the signed ones: it rules out the weaker worry that
both branches simply fail together on hard frames.

**This is the baseline the corner_fusion runs must be compared against.** There the mask
branch consumes LoMa's homography as a proposal, so any correlation above ~0.02 is
*induced by the coupling* and measures exactly the double-counting the review flagged. Run
the same function on the winning `corner_fusion_*` when its grid lands; that closes the
open end with a number instead of a caveat.

Two by-products from the same run, both supporting decisions already taken: the per-corner
oracle beats the whole-branch oracle by only 0.42 pp (94.98 vs 94.56), and genuinely
complementary `split` frames are **0.60%** of all frames — the per-corner rule has almost
nothing to fuse, which is why `decision_scope='whole_frame'` was the right simplification.

## 2026-08-19 — PTi by challenging factor: the robustness table POT cannot give
POT's condition table saturates (occlusion 100.00 for every serious tracker), so it cannot
show robustness. PlanarTrack ships 8 per-sequence challenging-factor flags and the official
toolkit's names for them are already in-repo (`third_party_examples/evaluation.py`) — no
external reference needed. Added `table_pti_attributes` to `experiments/results_tables.py`.
P@15, per-sequence mean:

| config | occl (87) | blur (160) | rot (244) | scale (285) | persp (291) | OOV (252) | low-res (17) | clutter (172) | all |
|---|---|---|---|---|---|---|---|---|---|
| `loma_samh_matchnaming` | 86.69 | 85.20 | 86.65 | 87.68 | 88.38 | 87.69 | **85.19** | 85.53 | 88.18 |
| `loma_samh_decoupled_rerun` | 86.85 | 85.05 | 86.71 | 87.35 | 87.96 | 87.39 | 81.52 | 84.57 | 87.86 |
| `sam_H` (prev. paper) | 76.27 | 76.95 | 78.24 | 79.61 | 80.37 | 79.99 | 76.09 | 75.94 | 80.34 |
| `woftsam_samfallback_nl` (prev. paper) | 77.36 | 75.69 | 77.46 | 76.58 | 77.47 | 77.08 | 58.52 | 72.73 | 77.37 |

**We lead both baselines on all eight factors**, by 8-10 pp typically and by 26.7 pp on
low-resolution against WOFTSAM. No factor collapses — which is the "no breakdown anywhere"
evidence the paper needs, on the hard benchmark rather than the saturated one.

Replacing DINOv2 with correspondence support also **helps most where appearance is least
reliable**: low-res +3.67 pp (n=17, so indicative not decisive), bg-clutter +0.96,
perspective +0.42. It is a wash on occlusion (-0.16) and rotation (-0.06), both within
noise. That is the mechanism behaving the way the story says it should — worth stating,
carefully, since the low-res cell is 17 sequences.

Watch item: occlusion is 1.5 pp below the aggregate for both current trackers, and
occlusion robustness ranks above aggregate P@15 for this project. It is not a regression
against the baselines, but it is the attribute with the least margin.

## 2026-08-19 — naming telemetry re-verified from the run itself; and a telemetry gap for the tau ablation
Two checks on the published candidate's naming mechanism, both from stored data.

**The quoted percentages hold.** Recomputed over all 147555 PTi frames of
`loma_samh_matchnaming` by classifying abstention from `naming_n_matches < 8` (the run
predates the `-1` abstention fix, so its `naming_shift` still lumps "abstained" with
"kept"): namer ran **87.21%**, abstained **12.79%** — matching the 87.2 / 12.8 on record
exactly. The other two differ slightly from the state doc: looked-and-kept **85.47%** (doc
says 85.7) and re-named **1.75%** (doc says 1.50), consistent with the doc counting over a
slightly different frame set. The mechanism claim is unaffected; the doc's two minor
figures should be taken as 85.5 / 1.75. Re-namings split almost evenly across the three
shifts (0.66 / 0.42 / 0.67 %), so they are not dominated by 180-degree flips.

**The tau ablation cannot be replayed, and that is a telemetry gap.** `matchnaming` trades
seven parameters for one, `symmetry.match_tau_px = 30`, so a reviewer will ask whether that
one is tuned — the same question the inlier-threshold sweep just answered for T=35. The
answer is offline-computable in principle: `naming_hist_counts` / `naming_hist_sumsq` are
exactly the per-bin sums needed to re-score the support at any tau on a bin edge (the edges
already include 30). **But only `corner_fusion.py` populates them** — the `hough` path that
the published candidate runs never does, so the arrays are structurally zero in all 300
sequences. Fixing this is a few lines in `flatsam2/tracker.py` next to where
`naming_shift` / `naming_support` are already recorded, then a re-run of a *subset* of
sequences (the ablation does not need the full grid). **Deferred: it edits `flatsam2/`,
which is frozen while the grids run.** Do it when they land, before the runtime table.

## 2026-08-19 — split hygiene on the threshold sweep
Checked where `inliers >= 35` came from before letting the sweep's framing stand: it is
**inherited from the predecessor system** (`LMr_allobj_infl10_X_SAM3H_thr35`), not fitted
on PTi and not fitted by us. That is what makes a sweep on the test split legitimate — it
is a post-hoc sensitivity check of a value chosen elsewhere, not a selection. Recorded in
the script so nobody later reads the argmax (T=35 on PTi) off the table and calls it a
tuned choice, which would be selecting on test. **The claim is the flatness, never the
peak.** The same rule applies to the tau ablation when its telemetry gap is closed: sweep
to show insensitivity, do not adopt an argmax.

## 2026-08-19 — the partition widening cost a grid restart: V100s do not fit a 15-minute task
The throughput fix earlier today was wrong in one respect, and it cost real GPU time. The
`corner_fusion_wholeframe` PTi array finished as **25 COMPLETED / 50 TIMEOUT**, 143 of 300
sequences, and its chained eval was cancelled by the failed `afterok`. The split is exactly
by node family:

| node | tasks | outcome | elapsed |
|---|---|---|---|
| `g*` (amdgpu) | 25 | COMPLETED | 10:36 - 13:01 |
| `n*` (V100) | 50 | **TIMEOUT** | 15:00 wall |

The arrays were submitted with `--time 0:15:00` — a tight budget deliberately chosen for
the fast nodes (see the SLURM packing note) — and a V100 without flash attention is more
than 1.4x slower, so *every* task that landed on the new partition died at the wall. I
widened the partition without re-examining the time limit that made the packing work.

Fixed: every pending array raised to `--time 0:35:00` and restored to `amdgpu,gpu`, and the
PTi gap resubmitted (11373141, eval 11373142; `submit.py` skips the 143 completed
sequences). Six already-running POT tasks could not be rescued — a user may not extend a
running job's TimeLimit — so they will time out and be refilled the same way.

**The slowdown factor, measured rather than guessed.** The POT array repeated the pattern
(25 more timeouts, all at the old 15:00 limit), but one task that started *after* the limit
was raised completed on a V100 in **27:34** against ~11:00 on an amdgpu node for the same
4-sequence packing. So the NVIDIA nodes are **~2.4x slower**, not the 1.4x I assumed from
the missing flash-attention warning alone. Every limit was resized against 2.4x: 50 min for
the 4-sequence fusion arrays, 1:30 for the resubmitted PTi array and the cross-dataset ones
(TMT contains 2518-frame sequences), 2:00 for MPOT. POT gap resubmitted as 11373701/2.

Cost: roughly 75 task-slots of wasted GPU time and two restarts. Nothing was lost
permanently, because completed sequences persist and are skipped on resubmit — which is the
property that makes tight packing safe in the first place. **The lesson is that `--time`
and the partition set are one decision, not two:** widening onto slower hardware silently
invalidates a limit tuned for the faster hardware. Any future `--partition amdgpu,gpu` must
carry a limit sized for the *slowest* member.

## 2026-08-19 — the Hough branch's corner covariance: it exists, it is correct, and its noise model is wrong
Acting on the maintainer's idea of giving the mask->Hough->corners branch per-corner
covariances so the fusion rule can be applied to the quad-based tracker too. It is already
written: `corner_covariance.line_intersection_cov` fits both boundary lines by TLS and
pushes the line uncertainties through the intersection Jacobian. It is unit tested (17
pass) and **called by nothing** — written for this branch and never wired in. The module
docstring warns this is where a factor-of-5 bug once survived because the Monte-Carlo
cross-check "was never run". So it was run: `experiments/line_covariance_validation.py`.

**A. The arithmetic is right.** 24 synthetic configurations (line angle 15-90 deg, 20-500
points, noise 0.5-2.0 px): predicted/empirical std ratio **0.925-1.075, median 1.007**,
median chi2 1.19-1.56 against a nominal 1.386. It also tracks the hard case — at 15 deg the
empirical scatter blows up to 3.5 px and the prediction follows. The old bug is gone.

**B. The noise model is wrong on real masks.** 11312 corners from stored SAM3 masks over
2940 PTi frames. The corners are *good* — median 2.88 px from GT — but the covariance is
absurdly over-confident: predicted std **0.097 px**, median chi2 **1125** (nominal 1.386).
And it is not a constant offset:

| n boundary pts | corners | predicted std | actual offset | median chi2 |
|---|---|---|---|---|
| 33-227 | 2837 | 0.125 px | 2.64 px | 563 |
| 227-401 | 2833 | 0.097 px | 2.95 px | 1065 |
| 401-678 | 2847 | 0.085 px | 2.93 px | 1337 |
| 678-2082 | 2837 | 0.074 px | 2.99 px | 1949 |

**The true error is flat in n while the prediction shrinks as 1/sqrt(n).** A mask boundary
is *systematically* displaced, not noisy: 2000 pixels along a spilled edge are one wrong
observation, not 2000 independent ones. **So a constant scale factor — the
`samh_cov_scale` approach used for the other branch — cannot fix this**, because the
required factor moves by 3.5x across the range of n seen in one dataset.

**The fix is to cap the effective sample size,** and it works. With `n_eff = min(n, cap)`
and a fixed boundary noise instead of the residual-derived one, drift across the n
quartiles collapses from 3.5x to **1.6x** at every setting tried. The two knobs are
degenerate — only `noise^2 / cap` enters — so it is **one** parameter, an effective
per-line offset variance. Calibrated value: `noise_px=3.0, n_eff_cap≈7`
(measured 0.98 median chi2 at cap 5, 1.97 at cap 10; chi2 is linear in the cap), i.e. an
effective boundary uncertainty of about **1.1 px per line regardless of contour length**.

Honest caveats: the tail stays heavy (14% beyond the 95% quantile at the calibrated
setting, against a nominal 5%) because gross mask failures are not Gaussian — arguably what
a disagreement gate *should* see as outliers — and 1.6x of n-dependence remains.

**Wiring plan (deferred: `flatsam2/` is frozen while grids run).** `mask_to_homography`
already computes everything needed and throws it away — `refine_lines` returns the per-line
point sets and `lines_to_points` returns `parents`, naming the two lines that made each
corner. It needs to return per-corner covariances alongside `(H, n_correspondences)`. Then
the maintainer's rule — *keep the 35-inlier switch, but fuse when the two branches agree* —
is the existing `corner_fusion` disagreement gate applied to the quad branch, with the
keypoint covariance already available from `homography_corner_covs`.

## 2026-08-20 — throughput reality check: PTi prioritised, POT and cross-dataset held
Measured rather than projected: 11 h of queue produced ~430 sequences, i.e. **~39
sequences/hour**, not the ~97/h estimated from the first ramp (the per-user GPU cap is ~6
concurrent, and V100 tasks take 2.4x longer). Remaining work at that point was ~1740
sequences, about **44 h** — more than the maintainer's window.

So the queue was reordered by what actually decides anything. From the branch-ceiling
measurement: **POT has 0.94 pp of selection headroom and will not separate the two
candidates; PTi has 3.10 pp and decides.** Three POT grids (630 sequences, ~16 h) were
queued ahead of PTi work.

`scontrol hold` (reversible, nothing lost) on: `corner_fusion_naming_pot`,
`corner_fusion_namecheck_pot`, and the four `loma_samh_matchnaming` cross-dataset arrays.
Left running: the three PTi arrays — `wholeframe` (157 left), `naming` (105), `namecheck`
(300) — 562 sequences, **~14 h**, which unblocks the candidate decision today instead of
in two days.

**Release the held jobs with `scontrol release <jobid>` once PTi is complete**
(11370101, 11370105, 11370348, 11370265, 11370268, 11370270). They are held, not cancelled;
a held array looks stuck in `squeue` and the REASON column says `JobHeldUser`.

Completion at the time of the reorder: `matchnaming` PTi 300/300 and POT 210/210 (the
published candidate is fully measured on both deciding benchmarks); `wholeframe` POT
201/210; `naming` PTi 195/300; `namecheck` POT 4/210; `matchnaming` MPOT 30/573.

## 2026-08-20 — a second timeout wave: PTi packing needs 2 h, not 50 min
`corner_fusion_naming` PTi finished 13 COMPLETED / 62 TIMEOUT, 230 of 300 sequences, eval
cancelled. The 50-minute limit set after the first wave was still too tight: **6 tasks
completed on a V100 at 50 min and 54 timed out at it.** The 2.4x slowdown I measured was
taken from POT, whose sequences are short and uniform; PTi sequences run to thousands of
frames and vary enormously, so a 4-sequence pack is not a fixed cost.

A timeout wastes the whole slot, while a generous walltime costs only a little scheduling
priority. So every remaining fusion array is now at **2 h** regardless of partition, and the
naming PTi gap is resubmitted (11377042 / eval 11377043, 70 sequences left).

Running total of waste from sizing walltime against the wrong hardware: ~135 task-slots
over three waves. The rule stands and should have been applied at full strength the first
time: **size the limit for the slowest hardware and the longest sequence, not the average
one** — the packing note's tight `--time` was calibrated for `amdgpu` + POT and does not
transfer.

## 2026-08-20 — corner_fusion paused; all effort on the quad-based branch
Maintainer's call: the SDF/registrar (mask->H by distance-transform optimisation) branch is
paused, everything goes to `loma_samh` (LoMa + SAM-Hough + 35 inliers + match naming). All
`corner_fusion_*` arrays cancelled, the four cross-dataset arrays released.

**Salvaged before cancelling: `corner_fusion_naming` PTi is complete and evaluated** — its
gap-fill finished minutes before the cancel. **P@15 89.21 per-sequence mean / 89.08 pooled /
br 90.29, n=300.** That is **+1.03 pp over the quad-based candidate's 88.18** and is a
valid post-covariance-fix, post-alias-fix number. It is the single most useful thing the
fusion thread produced and the reason the branch is *paused* rather than abandoned.

### State of the paused branch, for resumption

| config | PTi | POT | note |
|---|---|---|---|
| `corner_fusion_naming` | **300/300, evaluated: 89.21 / 89.08 pooled / br 90.29** | 0/210 | the live result; POT never started |
| `corner_fusion_wholeframe` | 143/300 | 201/210 | the reference rule |
| `corner_fusion_namecheck` | 88/300 | 4/210 | the alias-recheck hypothesis, barely started |

Partial `.txt`/`.npz` are on disk and `submit.py` skips completed sequences, so each line
resumes with one command, e.g.:

```bash
python slurm/submit.py pot --config configs/tracker/corner_fusion_naming.py \
    --jobs 53 --time 2:00:00 --partition amdgpu,gpu --no-mail
```

**Resume in this order** if time appears: (1) `naming` POT — one grid turns 89.21 into a
publishable pair; (2) `wholeframe` PTi/POT to finish the reference for the ablation;
(3) `namecheck` last, it is the least developed. Everything else the branch still owes is in
`docs/state-of-the-tracker.md` steps 3/3b/3c (recalibrate `samh_cov_scale`, measure induced
branch dependence against the 0.007/0.014 decoupled baseline, occlusion veto screen).

## 2026-08-20 — partition policy: keep both, always; and a bad-node failure mode
Two corrections to the scheduling, both learned by losing time.

**`amdgpu` is A100, `gpu` is V100, and A100 is ~2.4x faster — but pinning to it is wrong.**
On seeing 10 free A100s while all our tasks sat on V100 (SLURM takes the first free slot
when a job lists several partitions), the arrays were pinned to `amdgpu`. A100 then went to
100% allocated and our running count dropped to **zero** while ~12 V100s stood free. The
correct policy is the original one: **list both partitions and size `--time` for the
slowest**, because our QOS caps us at 8 concurrent GPUs — below either partition's typical
free capacity — so whichever is free wins. Throughput is availability x speed, not speed.

**A task can fail on a bad node, and it looks nothing like a timeout.** Two POIC tasks came
back `FAILED` with `RuntimeError: CUDA driver initialization failed, you might not have a
CUDA gpu` — a node-level fault, not our code. Worth recognising on sight, because the
instinct is to go looking for a bug in the tracker.

POIC also timed out at 90 min on 2-sequence packs, so its sequences are slower than the
size suggests; resubmitted at **one sequence per task** (11377639) so a single slow or
broken node cannot take several sequences down with it.

**Before merging the `hough-corner-cov` branch into the live tree** (which the running
cross-dataset arrays read), the new code must be shown to be *behaviourally identical* at
`fuse_when_agree_gate=0.0`, not merely assumed to be: job 11377644 runs three PTi sequences
through the worktree and diffs the corner files against the stored grid. Merge only on
IDENTICAL. The plan is then hold-pending / merge / release, which costs seconds rather than
the hours that waiting for the arrays would.

## 2026-08-20 — publication sequence for `loma_samh_matchnaming`
`corner_fusion` rejected; the quad-based tracker is the method. Everything below is queued
or staged, in dependency order, with watchers advancing the stages unattended.

**Stage 1 (running) — cross-dataset coverage.** POIC (11377639), TMT (11370268), UCSB
(11370270), MPOT-3K (11370348). Must-have: the paper claims no breakdown anywhere.

**Stage 2 (submitted, held behind stage 1) — the naming ablation.**
`configs/tracker/loma_samh_nonaming.py` (new): the same tracker with
`symmetry.enabled=False`, i.e. no corner naming at all. PTi 11377791, POT 11377793.
This completes the three-arm ablation the paper's central claim needs:

| arm | mechanism | PTi P@15 | br-gap |
|---|---|---|---|
| `loma_samh_nonaming` | none | queued | queued |
| `loma_samh_decoupled_rerun` | DINOv2 appearance vote | 87.86 | 1.57 |
| `loma_samh_matchnaming` | correspondence support | **88.18** | **1.40** |

Without the first arm we can only say "our naming beats their naming"; with it we can say
what naming is *worth*. Read the br-gap, not the headline — it is the direct measure of
corner labelling, and the no-naming arm should open it up markedly.

**Stage 3 (queued) — speed.** `profile_tracker.py` on all three configs **inside one job**
(11377799) so they share a node; mixing A100 and V100 would confound the comparison
entirely. Per-block breakdown (SAM3 / LoMa keypoints / match / RANSAC / Hough / symmetry)
on 120 frames after 5 warmup. The claim to support: match naming deletes a whole network
(DINOv2) relative to the predecessor, and that should be visible as time, not just as
parameter count.

**Stage 4 (staged, needs a merge window) — the tau ablation.** Branch
`naming-histogram-telemetry` (worktree `$OUT/dev-tau`, 184 tests pass) records the
per-alias residual histograms on the deployed `hough` path; they were only ever recorded
by the registrar path, which is why `symmetry.match_tau_px=30` — the one parameter the
method adds — could not be ablated. With them, the sweep is exact offline for any tau on a
bin edge. Needs: brief hold, merge, then a *subset* re-run into a scratch output dir (never
over the real grid) to collect histograms.

**Not in the publication path:** branch `hough-corner-cov` (per-corner covariances for the
Hough branch + fuse-when-branches-agree) is complete, tested and committed, but it is a
method *change* rather than a result, so it holds no GPU. Resume from the worktree if the
extension is wanted.

## 2026-08-20 — my own verification jobs reported success while doing nothing
The determinism (11377270) and old/new equivalence (11377644) jobs both exited **0** in 35 s
and 13 s, having tracked nothing. Two mistakes, one of them the exact failure mode this
project fixed in `run.py`/`evaluate.py` earlier: **a job that exits 0 is not a job that did
the work.**

1. The scripts `cd`'d into a worktree and re-sourced `.envrc`, but `slurm/cmd.sbatch`
   already sources it. The second source put the module tree's `typing-extensions` ahead of
   the venv's, so every `python run.py` died on
   `ImportError: cannot import name 'Sentinel' from 'typing_extensions'`.
2. `--no-debug` is not a `run.py` flag.

Neither killed the job, because the *last* command in the `bash -c` was the comparison,
which cheerfully printed `MISSING new` and returned 0. Resubmitted as
`$OUT/scratch/determinism.sh` with `set -euo pipefail`, no re-sourcing, real flags, and a
non-zero exit when any sequence produced no output (11378036).

The equivalence check is **not** being resubmitted: `hough-corner-cov` is parked out of the
publication path, and its safety was in any case established by construction (the diff shows
`return chosen` at gate 0) plus 2720 frames of the production covariance path with zero
exceptions — the 3-sequence run was only ever confirmation.

## 2026-08-20 — two stale files deleted
`tests/test_tracker.py` and `docs/review-findings-2026-08-19.md` removed at the
maintainer's request. Both were untracked leftovers.

`tests/test_tracker.py` was an April file that commit `80c843e` had already deleted from
git; the copy on disk called an API the tracker no longer has (`t._state`,
`t.update(frame=...)`), so a bare `pytest tests/` errored on it. `pytest tests/` now runs
clean: **184 passed** (excluding `test_cropped_area_discrepancy.py`, whose two failures are
a known, real SAM3 self-correction defect). Recoverable if ever wanted:
`git show 80c843e^:tests/test_tracker.py`.

`docs/review-findings-2026-08-19.md` was never committed, so deleting it is permanent — its
substance is preserved in the 2026-08-19 "external review: nine findings, all true, seven
fixed" entry above, and a verbatim copy is parked outside the repo at
`$OUT/scratch/review-findings-2026-08-19.md.bak` in case the detail is wanted.

## 2026-08-20 — "you just used a better segmenter" — answered, with both controls
The reviewer objection that our gain is really SAM3-vs-SAM2 has a direct answer already on
disk, and a `collect_results` view (`story-00-flagship` in
`configs/reports/example_views.py`) now carries it so the report makes the argument itself.

Two controls, both the *published baseline's own method* run on **our** segmenter:

| PTi, per-seq mean p@15 | SAM2 | SAM3 | segmenter is worth |
|---|---|---|---|
| SAM-H (mask -> Hough) | 80.34 (`sam_H`) | 83.05 (`hough_symmetry_only`) | **+2.71** |
| WOFTSAM (flow) | 77.37 (`woftsam_samfallback_nl`) | 77.15 (`woftsam3_samfallback_nl`) | **-0.22** |
| ours | — | **88.18** | — |

So the segmenter upgrade helps only the method that consumes masks directly, and does
**nothing** for the flow-based baseline. Decomposing our margin over SAM-H: PTi
80.34 -> 83.05 -> 88.18 = segmenter +2.71 (35%) + method +5.13 (65%); POT
88.38 -> 89.37 -> 97.66 = segmenter +0.99 (11%) + method +8.29 (89%). The method carries the
majority on both and almost all of it on POT.

`hough_symmetry_only` is a genuinely clean control — SAM3, `hough.method='hough'`,
`loma.enabled=False` — i.e. the mask expert alone. `woftsam3_samfallback_nl` was verified to
load `configs/sam/SAM3.py` rather than trusting the name.

**The one gap, now running:** `woftsam3` has no POT grid, and POT is exactly where our margin
over WOFTSAM is thinnest (97.66 vs 96.75, +0.91). Submitted on A100 (11378376/7, legacy
`--mem 24G`). Until it lands, that single cell is the honest hole in this argument.

## 2026-08-21 — UCSB failure figures, and the mask overlay complicates the diagnosis
`experiments/ucsb_failures.py` now writes **one PNG per failing sequence** (18 of 96, every
sequence with a p@15 deficit >= 3 pp against WOFTSAM), template beside the most diagnostic
frame — the frame where our error exceeds 15 px while WOFTSAM stays under it. Both panels
cropped around the object with context, GT drawn last so a near-miss cannot hide it, corner
0 dotted so cyclic mis-naming is visible, legend in a band below the imagery. CPU-only, about
a minute. SAM3 masks for those sequences were dumped separately (`ucsb_maskdump`, V100,
6/7 tasks) and are overlaid via `--masks-dir`.

**The overlay does not simply confirm the 2026-08-17 "target-definition mismatch" reading.**
On `bricks_motion8` frame 11 the SAM3 mask sits essentially on the GT photo region — it does
*not* grab the laminated sheet — yet our pose is still ~20 px out and visibly larger,
wrapping the checkerboard border. On `sunset_unconstrained` frame 465 the mask *does* spill
slightly onto the checkerboard, and our pose is again too large (41 px vs WOFTSAM's 12).

So the constant across failures is that **our quad is too large** (consistent with the
measured scale ratio 1.179 on failing frames), but "SAM3 segments the sheet" is not a
sufficient explanation — at least one clear case has a good mask and a bad pose, which
points at the mask->pose step rather than the mask itself. Two frames are not a conclusion;
the full set of 18 is on disk for the maintainer to judge.

Not drawn, and not available: **LoMa inliers**. The debug npz stores `n_inliers` as a count
only, with no point coordinates anywhere on disk, so drawing them needs either new telemetry
or a GPU re-run with the matcher loaded.

## 2026-08-21 — the segmenter objection is closed, and the cross-dataset grid is nearly done
**`woftsam3_samfallback_nl` on POT finished (210/210): mean p@15 95.60, pooled 96.61.** That
was the last cell of the "you only win because you use SAM3" argument, and it closes it in
our favour on both benchmarks. Per-sequence mean p@15, each baseline's own method run on
each segmenter:

| | SAM2 | SAM3 | segmenter is worth |
|---|---|---|---|
| **PTi** SAM-H (mask -> Hough) | 80.34 | 83.05 | +2.71 |
| **PTi** WOFTSAM (flow) | 77.37 | 77.15 | -0.22 |
| **PTi** ours | — | **88.18** | |
| **POT** SAM-H | 88.38 | 89.37 | +0.99 |
| **POT** WOFTSAM | 96.75 | **95.60** | **-1.15** |
| **POT** ours | — | **97.66** | |

SAM3 helps only the method that consumes masks directly, and on POT it makes the flow
baseline **worse**. The consequence worth quoting: measured against the *best SAM3-equipped
baseline*, our margin is **+5.13 PTi** and **+2.06 POT** — on POT that is more than double
the +0.91 we have over WOFTSAM-with-SAM2. Giving the baselines our segmenter widens our
lead rather than narrowing it.

**Provenance caveat, stated because it is the weak point.** The WOFTSAM POT SAM2 number
(96.75) was *imported* from the previous paper; woftsam3 POT was run fresh here. WOFTSAM is
stochastic (RANSAC + flow), and the 2026-08-15 parity check measured a fresh WOFTSAM POT run
at 96.37 against the imported 96.75. So the like-for-like fresh-vs-fresh comparison is
95.60 vs ~96.37 = **-0.77**, not -1.15. Still negative, still the same conclusion, but the
paper should quote the fresh pair rather than the imported one.

**Cross-dataset naming, three of four datasets in.** The mechanism behaves the way the story
claims — it moves the br-gap, and the headline follows only where the gap had room to move:

| dataset | matchnaming | br-gap | DINOv2 predecessor | br-gap | delta |
|---|---|---|---|---|---|
| POIC (20) | 98.95 | 0.03 | 98.97 | 0.01 | -0.02 |
| **TMT (92)** | **94.38** | **0.06** | 93.99 | **0.53** | **+0.40** |
| UCSB (96) | 93.73 | 0.00 | 93.76 | 0.00 | -0.03 |
| MPOT-3K | 542/573 | | 90.98 | 0.08 | |

TMT was the only extra with a real corner-labelling error (gap 0.53); the namer closes it to
0.06 and converts that into +0.40 p@15. POIC and UCSB had nothing to fix and it costs
0.02-0.03, i.e. noise. Same signature as PTi (1.57 -> 1.40, +0.32) and POT (0.00 both,
neutral). Four datasets, one consistent mechanism claim.

With woftsam3 done, the held A100 work was released automatically at 06:17: the naming
ablation (`loma_samh_nonaming`, PTi 11377791 / POT 11377793) is running, speed (11377799)
is queued behind it.

## 2026-08-21 — the naming ablation: what corner naming is actually worth
`configs/tracker/loma_samh_nonaming.py` (`symmetry.enabled=False`) completed on both
benchmarks, giving the three-arm ablation the paper's central claim needs. Per-sequence mean:

| PTi | p@15 | br-p@15 | br-gap |
|---|---|---|---|
| no naming | 85.03 | **89.58** | 4.55 |
| DINOv2 appearance vote | 87.86 | 89.43 | 1.57 |
| **correspondence support (ours)** | **88.18** | **89.58** | **1.40** |

| POT | p@15 | br-p@15 | br-gap |
|---|---|---|---|
| no naming | 97.66 | 97.66 | 0.00 |
| **ours** | **97.66** | 97.66 | 0.00 |

**Naming is worth +3.15 pp on PTi, and it is purely naming.** The best-rotation column is
**89.58 for the no-naming arm and 89.58 for ours** — identical to two decimals over 300
sequences. Best-rotation forgives any cyclic relabelling, so a mechanism that only relabels
corners must leave it untouched; it does. Localisation and the trajectory are unaffected,
which is exactly what the br-gap column was introduced to be able to show.

Read as recovery of the available error: 4.55 pp of corner mislabelling exists, DINOv2
recovers 2.83 (62%), correspondence support recovers **3.15 (69%)** — with a whole network
and six parameters deleted.

**POT is worth stating precisely because it is a null result.** Its br-gap is 0.00 for every
method, so there is no mislabelling to fix, and naming changes the score by exactly nothing
(97.66 either way). A mechanism that helped on POT would be suspicious; one that is free
there is behaving as claimed.

Small asymmetry worth keeping: DINOv2's br-p@15 (89.43) is slightly *below* the other two
arms, so appearance-based naming perturbs trajectories a little, while match naming does
not. It supports "the matcher already holds the evidence" over "add a second model".

## 2026-08-21 — speed: our naming is free, DINOv2's costs 4.5%
`profile_tracker.py`, all three configs and all sequences **inside one job on one node**
(g11, A100-SXM4-40GB) — mixing A100 and V100 across configs would have swamped the effect
with a 2.4x hardware difference. 100 frames after 5 warmup, cuda sync on. Mean over three
PTi sequences (Seq_00003 / 00005 / 00006):

| config | ms/frame | FPS | vs no naming |
|---|---|---|---|
| no naming | 286.6 | 3.49 | — |
| **correspondence support (ours)** | **286.7** | **3.49** | **+0.1 ms (+0.03%)** |
| DINOv2 appearance vote | 299.5 | 3.34 | +12.9 ms (+4.51%) |

**Our naming is free to the millisecond** — 0.1 ms against a tracker with no naming
mechanism at all, i.e. inside the noise — while the appearance vote costs **4.5%**. The
ordering is identical on all three sequences, so it is not a single-sequence artifact.
Ours is **4.5% faster than the DINOv2 variant** it replaces.

That is the honest form of the claim. The tempting version — "we are faster" — is weaker and
partly wrong: the gain is entirely the deleted network, not anything our namer does better
at runtime. What the numbers support is that a second forward pass was bought for a 2-bit
decision, and the matcher's existing correspondences answer it for nothing.

Where the time actually goes (matchnaming, Seq_00006): LoMa keypoints 41.3% (describe alone
22.1%), SAM3 34.4%, LoMa match 13.2%, mask-H 10.0%, RANSAC 0.3%. Naming does not appear as a
measurable section. Peak GPU memory 6.45 GB. At ~3.5 FPS the tracker is not real-time and
the paper should say so plainly; the cost is dominated by the two networks, not by our
contribution.

Seq_00011 produced no timing report for any of the three configs and was dropped from the
mean; the other three agree closely enough that chasing it is not worth a GPU slot.

## 2026-08-25 — the blocked threads restart: tau telemetry merged, and reproducibility is not bit-exact
Queue was empty and the publication grids were all complete, so the three things that had
been deferred behind them were started.

**1. The tau-ablation telemetry is merged** (`55f63b9`, cherry-picked from
`naming-histogram-telemetry`). The deployed `hough` path now records
`naming_hist_counts` / `naming_hist_sumsq`; only the registrar path did before, which is
why `symmetry.match_tau_px = 30` — the one parameter the method adds — could not be
ablated. Telemetry only: `alias_supports` is a pure recomputation and the tracking output
does not change. `pytest tests/` 184 passed on the `cpu` partition afterwards.

Re-run submitted on the PlanarTrack **train** split (`pt_trn_50`, 50 sequences, job
11408267, output under `$OUT/tau_ablation/` so it cannot touch the real grid). **Train, not
test, deliberately**: the 35-inlier sweep was legitimate on test because that threshold is
*inherited* from the predecessor system, but tau is ours, so sweeping it on the reported
benchmark would be selecting on test. With the histograms the sweep is then exact offline at
any bin edge (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100 px) via
`experiments/naming_rule_replay.py`. The claim will be the flatness, never the argmax.

**2. `corner_fusion_naming` POT resubmitted** (11408275, 53 tasks, 2 h, `amdgpu,gpu`). It is
the paused branch's missing half: PTi is complete at 89.21, +1.03 pp over the shipped
method, and one grid turns that into a publishable pair. It changes nothing in the paper
unless the maintainer reopens the branch.

**3. The determinism job that was submitted on 2026-08-20 had landed and never been read,
and its result is not the reassuring one.** Two runs of the same three PTi sequences on the
same node are **bit-identical** — `ransac_seed` does its job — but both differ from the
stored grid by up to **1494 px**, and the divergence starts at **frame 1**. Cause, checked
rather than assumed: the stored grid ran on `g02` (A100) and the re-run on `n23` (V100), and
no code on the hough path changed in between. Every commit since the grid either touches the
registrar (`afceb74`, the alias fix — `mask_register.py` only, so the published numbers are
*not* affected by it), is telemetry (`5085c10`, `55f63b9`), or is not wired into the tracker
at all (`a3f22be` occlusion screen). So this is GPU-kernel nondeterminism across
generations, amplified by an autoregressive tracker.

Impact on the metric, per sequence (`$OUT/scratch/det_impact.py`):

| sequence | first diverging frame | P@15 grid | P@15 re-run | delta |
|---|---|---|---|---|
| Seq_00003 | 1 | 74.10 | 73.49 | −0.60 |
| Seq_00005 | 1 | 69.64 | 69.31 | −0.33 |
| Seq_00006 | 1 | 82.40 | 84.64 | +2.23 |

Signs are mixed and the aggregate should be far steadier than any single sequence, but three
sequences are not a number to put in a paper, so a 20-sequence re-run (`pti_fusion_probe20`,
11408394) is running to give the reproducibility sentence an aggregate figure.
**Consequence for how grids are read:** results must not be compared cell-by-cell across
partitions, only in aggregate.

Also: `docs/state-of-the-tracker.md` was rewritten to the publication state (it still opened
on the void-covariance banner and a queue from 2026-08-19), and `paper/draft_review.md`
gained a section listing the results that landed on 2026-08-21 and are now fillable.

## 2026-08-25 (later) — the cross-GPU divergence is a butterfly, not a suppressed error
The 2026-08-20 determinism job's "up to 1494 px" needed verifying before it could be
written down as anything, because a jump that size usually means a gross error being
swallowed. It is not that. Three sequences, published grid (A100, `g02`) vs re-run
(V100, `n23`), same code on the hough path (checked commit by commit: the 2026-08-19
alias fix touches `mask_register.py` only, `5085c10`/`55f63b9` are telemetry, and the
occlusion screen is wired into nothing).

**It starts sub-pixel and grows multiplicatively.** The difference at frame 1 is
**0.56 / 0.84 / 0.69 px**; while it is still under 5 px it grows by a median factor of
1.16 / 1.04 / 1.11 per frame, reaching 15 px only after 10 / 40 / 17 frames. Nothing
discrete leads: `maskh_source` and align-acceptance differ on **0** frames of 443 / 462 /
468, and the first branch-switch disagreement is at frame 58 / 43 / 38 — long after the
trajectories have separated. At the jumps both runs have picked the *same* branch.

**The spikes are hard frames, and they recover.** Divergence episodes (>= 15 px) are 26 /
36 / 24 per sequence with a median length of **1 frame**, and 26/26, 31/36, 24/24 of them
are followed by a return under 1 px within 30 frames. Rolling medians across a sequence do
not trend (2.45 -> 2.02 -> 1.07 -> 1.08 -> 1.55 -> 1.03 px on Seq_00003), and every window
still contains frames agreeing to 0.03-0.4 px. Conditioning says the same thing: on frames
the published run tracks within 5 px the difference is 0.63-0.83 px median (p90 1.1-2.6),
on frames it gets wrong it is 0.84-4.73 px median (p90 12-500), and below the 35-inlier
switch it is 226 px median on Seq_00005 against 1.51 px above it. Spearman(diff, own error)
+0.59 / +0.56 on two sequences — and **+0.00 on Seq_00003**, so "hard implies divergent" is
a strong tendency, not a law.

**Neither run is the broken one.** At jump frames with GT, both runs are wrong on 2/9/1 of
them (median GT error 18-43 px) and only 0/1/5 have one right and one wrong. Overall the
pair is indistinguishable: median error 9.55/9.55, 9.46/9.34, 8.35/8.31 px; P@15
74.17/73.57, 69.74/69.41, 82.45/84.68.

So the tracker is chaotic exactly where the evidence is degenerate and self-correcting
everywhere else — a ~0.7 px per-frame injection (SAM3 and LoMa forward passes are not
bit-reproducible across GPU generations), amplified only while a hard stretch lasts. **The
operational rule is unchanged: compare grids in aggregate, never cell-by-cell across
partitions.** What is still open is *where* the 0.7 px enters; `$OUT/scratch/det_probe.py`
takes one step from a bit-identical state on both partitions and dumps the SAM3 mask
itself, and is queued.

## 2026-08-25 (later still) — tau is not tuned either, and the replay had a frame-of-reference bug
The train-split re-run (50 sequences, `pt_trn_50`) landed with the new histograms, so
`symmetry.match_tau_px` could finally be swept. **Train, not test**: tau is our own
parameter, unlike the inherited 35-inlier threshold.

**The replay was scoring the wrong comparison.** `naming_rule_replay` compared a rule's
pick, which lives in the *pre-naming* frame the histograms were computed in, against GT's
correction of `samh_H`, which is the pose *after* the run applied its own naming. The two
differ by exactly the applied shift, so every re-named frame was mis-scored — which is why
the first run of the sweep had the incumbent (99.37%) beating every rule, an impossible
ordering for a run whose incumbent *is* the rule. Fixed by composing with the recorded
`naming_shift`, and the script now refuses telemetry that predates that field rather than
reporting a wrong table. It also self-checks, like the inlier sweep: replaying the deployed
rule reproduces the run's own naming on **99.51%** of frames -> MATCH. Rule tables printed
by this script before today are suspect on re-named frames (order of the re-naming rate,
~1-2%); coverage figures from it are unaffected.

**Naming accuracy vs tau** (19800 localized GT frames, all with correspondence evidence):

| rule | 3 | 5 | 10 | 15 | 20 | **30** | 50 | 100 | incumbent |
|---|---|---|---|---|---|---|---|---|---|
| correct | 99.32 | 99.37 | 99.35 | 99.41 | 99.37 | **99.28** | 99.25 | 99.13 | 99.24 |

**Flat over a 17x band**: 99.25-99.41% for tau in [3, 50], i.e. 0.16 pp, and every setting
beats never re-naming. Only tau=100 degrades (99.13), which is the expected failure — at
that truncation a gross mismatch still contributes and the aliases stop separating. The
deployed 30 sits inside the plateau. **The claim is the flatness; the argmax (15) is not to
be adopted** — that would be tuning on data, and the flatness is what answers the reviewer.

Two things to state honestly with it: on this train sample *every* localized frame has
correspondence evidence, against 87.2% on the PTi test grid, so the coverage limitation is
not visible here; and the sweep scores the naming decision, not end-to-end P@15.

## 2026-08-25 (final readout) — deployed guards complete the sensitivity and reproducibility results
The first corrected replay still omitted the deployed eight-raw-match structural minimum.
That made its nominal `tau=30` rule differ from the tracker on low-match frames and limited
the self-check to 99.51%. The replay now holds the eight-match guard fixed while sweeping
tau, sweeps the raw-match minimum separately, and reproduces the recorded deployed choices
on **100.00%** of 19,800 localized train frames.

With that controlled comparison, naming accuracy is 99.34--99.38% over the full
`tau=3..100` range, only **0.04 pp**, and 99.28--99.38% for raw-match minima from 0 to 32,
only **0.10 pp**. The deployed `tau=30`, minimum 8 setting is inside both plateaus. These are
decision-level scores on the 50-sequence PlanarTrack train sample, not end-to-end P@15.

The 20-sequence cross-GPU readout also landed. Per-sequence mean P@15 changes from 91.01 on
the published A100 grid to 91.30 on the V100 re-run, **+0.29 pp**; per-sequence deltas have
standard deviation 0.84 pp and range from -0.67 to +2.41 pp. Seeds therefore give
bit-identical same-hardware runs, while cross-generation trajectories are not bit-exact but
the measured aggregate is stable within 0.3 pp.

## 2026-08-25 (later still, 2) — the injection is LoMa's matcher, not SAM3, and not TF32
`$OUT/scratch/det_probe.py` removes history from the question: for each probe frame it
builds a *fresh* tracker, inits on frame 0 and takes exactly **one** step, so every run
sees a bit-identical state. Run on both partitions, twice each.

**Within an architecture the tracker is bit-exact — on different machines.** A100 g04 vs
A100 g11 and V100 n21 vs V100 n30 both give mask IoU **1.000000**, identical mask pixel
counts, identical match and inlier counts and 0.000 px on both branch homographies. So this
is not "another run" or "another node"; it is specifically **another GPU architecture**.

**Across architectures, SAM3 is not the carrier.** A100 vs V100, one step, Seq_00003:

| frame | mask IoU | mask px A/V | centroid shift | matches A/V | loma_H diff | samh_H diff | output |
|---|---|---|---|---|---|---|---|
| 1 | 0.999686 | 70119/70117 | 0.02 px | 773/680 | 0.56 px | 0.04 px | 0.56 px |
| 2 | 0.999521 | 75061/75091 | 0.02 | 749/681 | 1.11 | 0.10 | 1.11 |
| 5 | 0.999546 | 118932/118922 | 0.02 | 796/739 | 0.84 | 0.11 | 0.84 |
| 10 | 0.999363 | 185164/185098 | 0.07 | 337/387 | 3.49 | 0.63 | 3.49 |
| 20 | 0.999595 | 61665/61662 | 0.02 | 70/86 | 5.60 | 0.05 | 5.60 |
| 50 | 0.997848 | 48337/48235 | 0.21 | 3/0 | — | **119.66** | 119.66 |

The mask agrees to **IoU 0.9994** and a 0.02 px centroid; the *matcher* disagrees by
**8-19% of its matches**. The pose difference tracks the matcher, not the mask.

**And it is LoMa itself, not LoMa reacting to a different mask.** `det_loma_probe.py`
feeds *the A100's stored mask* to keypoint extraction and matching on both partitions:
keypoints 4096 either way (a fixed budget), matches **773 / 749 / 796 / 337 / 70 / 3** on
A100 against **680 / 683 / 740 / 394 / 86 / 1** on V100. Same input, same seeds, ~10%
different correspondence sets.

**The obvious suspects are not it.** Disabling TF32 (`matmul`/`cudnn`) *and* forcing the
MATH attention kernel — the two things an Ampere card does differently — leaves the A100 at
774 / 743 / 796 / 339 / 68 / 3, i.e. it perturbs by a match or two and does not move toward
the V100 at all. So the difference is ordinary fp32 kernel/reduction-order and algorithm
selection differing per architecture, not a reduced-precision path we could switch off.
Each machine is internally deterministic; the two disagree at the last bits, and a
mutual-nearest-neighbour threshold turns last-bit differences into different match sets.

**One result here is worth more than the reproducibility question.** At frame 50 the
keypoint branch is effectively dead (3 vs 1 matches) and the mask branch is alone: a mask
differing by **0.2% IoU** produces a **119.7 px** pose difference — and the best-rotation
column is *also* 119.7, so it is a genuine pose disagreement, not a cyclic relabelling. The
mask -> pose step is ill-conditioned exactly where it is the only evidence, which is the
same phenomenon as `maskh_silhouette_aliasing` and is the shape of the open UCSB question
(good mask, bad pose). Worth a supplementary sentence, and worth remembering before anyone
proposes trusting the mask branch further.

**Aggregate effect, for the paper's reproducibility sentence** (`pti_fusion_probe20`, 20
sequences, V100 re-run vs the A100 grid): per-sequence mean P@15 **91.01 -> 91.30**, delta
**+0.29 pp**, per-sequence deltas sd 0.84, median |delta| 0.25, range -0.67 to +2.41, while
individual trajectories differ by up to 1395 px. Say: seeds fixed, bit-identical on the same
GPU architecture, aggregate stable to a few tenths of a point across architectures — never
"bit-exact reproducibility".

## 2026-08-26 — the paused branch is no longer half a result: quad-free wins PTi, loses occlusion
`corner_fusion_naming` POT finished (210/210, evaluated), so the quad-free candidate finally
has a pair. Against the shipped quad-based tracker, per-sequence mean:

| | PTi p@15 | PTi p@5 | br-gap | POT p@15 | POT p@5 |
|---|---|---|---|---|---|
| `loma_samh_matchnaming` (shipped) | 88.18 | 65.48 | 1.40 | **97.66** | **92.55** |
| **`corner_fusion_naming`** | **89.21** | **69.71** | **1.08** | 97.58 | 92.50 |

**+1.03 pp on the benchmark that decides, +4.23 pp at p@5, a smaller naming gap, and no
35-inlier cliff** — which is the paper's own stated motivation — against **-0.08 pp on POT**,
i.e. nothing. On aggregate this is the better tracker.

**The breakdowns say it is not that simple.** PTi by challenging factor: quad-free wins
**seven of eight** — occlusion 88.05 vs 86.69, blur 87.13 vs 85.20, rotation +1.39, scale
+1.07, perspective +1.02, out-of-view +1.10, bg-clutter +1.23 — and **loses low-resolution
by 10.37 pp (74.82 vs 85.19, n=17)**, a collapse rather than a slip. POT by condition:
occlusion **97.83 vs 100.00**, and unconstrained 94.67 vs 95.33; blur goes the other way,
90.56 vs 88.28.

Occlusion robustness ranks above aggregate p@15 for this project, so breaking POT's perfect
occlusion score is not a rounding detail — this is a maintainer decision, not mine. Both
regressions are diagnosable offline from stored telemetry (17 and 6 sequences), and that is
the next thing to do.

**Everything measurable is now queued.** The GPU queue was empty and the QOS submit cap is
~500 array tasks, so it is filled to the cap in decision order: `corner_fusion_naming` on
POIC / TMT / UCSB / MPOT-3K (781 sequences, the coverage any ship decision needs),
`corner_fusion_wholeframe` gap-fill (PTi 157, POT 9 — the ablation reference),
`corner_fusion_namecheck` PTi (212), and a single-node speed job pairing the quad-free
candidate against the shipped one on the same three sequences (11416980). What did not fit
under the cap — namecheck POT, and wholeframe's own cross-dataset coverage if its two grids
complete — is submitted automatically as the queue drains by
`$OUT/scratch/drain_submit.sh` (nohup'd on the login node, log beside it). If that process
dies, submit those by hand.

## 2026-08-26 (later) — the POT occlusion regression is three frames, and it is the known bug
Both of the quad-free candidate's regressions are concentrated, and they are not the same
phenomenon.

**POT occlusion: 3 sequences of 30, one frame each.** V19_5, V01_5, V30_5 fall from 100.00
to 75/80/80; the other 27 are unchanged at 100.00. POT annotates these sequences sparsely —
**5 or 6 GT frames each** — so one wrong frame is 16-25 pp of a sequence's score and 2.17 pp
of the condition. All three wrong frames have `fusion_modes = gated_samh` (the disagreement
gate fired and the rule took the mask branch whole) and on **100%** of them the mask branch
was the more confident of the two.

That is precisely the failure diagnosed on 2026-08-19 and never fixed: *when the mask branch
converges cleanly onto the wrong silhouette under occlusion it reports sigma at the 0.25 px
floor and wins the arbitration outright* — the example in that entry was V01_5 frame 105,
keypoint 4.7 px against mask 132.4 px, fused takes the mask. **The fix is a gross-error
test, not a calibration constant**, and `flatsam2/occlusion.py` (partial-view detection so
the mask branch abstains, written 2026-08-19, `a3f22be`) is on disk and **wired into
nothing**. Wiring it and re-running POT is one config and 210 sequences, and it targets
exactly the 3 frames that cost the perfect occlusion score.

**PTi low-resolution is a different, less flattering story.** 3 of 17 sequences carry 76% of
the -10.37 pp (Seq_00254 -54.12, Seq_00352 -46.69, Seq_00296 -32.96) while 5 of 17 improve.
But the bad frames there are *not* mask overconfidence — the mask branch is the more
confident one on only 3 / 6 / 24% of them, and the dominant mode on wrong frames is
`loma_only` on two of the three sequences. These are ordinary tracking losses on hard,
low-texture footage, where the quad-based inlier switch happens to survive and the fusion
rule does not. No known mechanism, and n=17 makes the aggregate cell fragile in both
directions.

So the honest summary for the decision: **one regression is a known bug with an unwired fix
sitting in the tree; the other is unexplained and rests on three sequences.**

## 2026-08-31 — the queue drained: quad-free is measured on all six datasets, and namecheck is a null
Everything submitted on 2026-08-26 finished; `squeue` is empty. Nothing new was learned about
the paper's method — the shipped config was already complete — but the paused quad-free branch
now has the coverage a ship decision needs.

**`corner_fusion_naming` vs the shipped `loma_samh_matchnaming`, mean P@15, six datasets:**
PTi **89.21 / 88.18** (+1.03), MPOT-3K **92.46 / 90.98** (+1.48), POIC 98.96 / 98.95 (+0.01),
POT 97.58 / 97.66 (-0.08), TMT 93.55 / 94.38 (-0.83), UCSB 93.27 / 93.73 (-0.46). It wins the
two large benchmarks and loses the two small classical ones by under a point. Aggregate case
unchanged from 2026-08-26; the new information is that the losses do not spread.

**Cost: ~16% slower.** Same node (A100, g11), three PTi sequences: 350 ms/frame against
307 ms/frame. The distance-transform registrar (`mask_H.register`, 74 ms/frame) is where it
goes; SAM3 and LoMa are identical between the two.

**`corner_fusion_namecheck` is a null.** Re-scoring the three relabelings of the final winner
on *every* frame gives PTi 89.15 / br 90.19 against `_naming`'s 89.21 / 90.29 — the br-gap
moves 1.08 -> 1.04. The 88%-of-frames coverage hole it was built to close is not where the
remaining naming error lives. POT is bit-identical to `_naming` (97.58). Direction closed.

**`corner_fusion_wholeframe` gap-fill** (the ablation reference) is complete on PTi/POT/POIC/
TMT/UCSB; MPOT-3K was never submitted and is the one computational loose end. Its PTi 88.30
against `_naming`'s 89.21 is the isolated value of the naming step, +0.91 pp.

**Still open, unchanged:** `flatsam2/occlusion.py` remains wired into nothing, and it is the
fix for the 3-frame POT occlusion regression (V19_5 / V01_5 / V30_5, one frame each) diagnosed
on 2026-08-26. That is the cheapest remaining move on the quad-free branch — one config plus
210 sequences — and occlusion robustness outranks aggregate P@15 here.

## 2026-08-31 (later) — the occlusion screen is wired, fixes the frames it targeted, and loses POT anyway
`flatsam2/occlusion.py` now reaches the tracker (`a6b15f2`): the mask branch abstains when the
SAM mask is an almost-perfect subset (precision >= 0.90, recall <= 0.80) of the template
silhouette warped by the **keypoint** branch's current-frame pose. Off by default,
`configs/tracker/corner_fusion_occl.py` turns it on over `corner_fusion_naming`. The branch
estimate is still stored and a per-frame `fusion_occluded` flag is dumped, so the replay stays
exact.

**It does exactly what it was built to do, and it is still a net loss.** POT-210, 210/210,
mean P@15 **97.16** against the base's 97.58 (shipped 97.66):

| condition (n=30) | shipped | quad-free | +screen | screen − base |
|---|---|---|---|---|
| occlusion | 100.00 | 97.83 | **100.00** | **+2.17** |
| unconstrained | 100.00 | 100.00 | 95.33 | **−4.67** |
| scale | 88.28 | 90.56 | 89.44 | −1.11 |
| unconstrained-2 | 95.33 | 94.67 | 95.33 | +0.67 |
| rotation / perspective / out-of-view | 100.00 | 100.00 | 100.00 | 0.00 |
| **all 210** | 97.66 | 97.58 | **97.16** | **−0.42** |

All three target sequences are repaired outright — V19_5 75 → 100, V01_5 80 → 100, V30_5
80 → 100, plus V10_7 80 → 100 — and POT occlusion is perfect again. The diagnosis of
2026-08-26 was right and the mechanism works.

**The trigger is what is wrong: it fires on 3.80% of frames (3992/105000) in 82 of 210
sequences**, and two *unconstrained* sequences pay for it — V02_1 100 → 20, V08_1 100 → 40.
Nothing is occluded there. High precision with low recall also describes a frame where the
keypoint pose is simply too *large*, and since the screen judges the mask against that pose,
a bad LoMa scale makes an honest mask look like a partial view — and the rule then throws away
the branch that was right. The screen needs a credibility guard on the pose it predicts from,
not a different threshold; both `occl0.7` and `occl0.8` land on the same −0.40.

**The offline replay predicted this exactly, and I ran the grid before reading it.**
`fusion_rule_replay.py` on the stored POT telemetry: baseline self-check MATCH (98.41 vs the
run's own eval), `whole_frame+occl0.7` and `+occl0.8` both −0.40 pp mean, against the GPU
run's measured −0.42. It also bounds the whole exercise: the **acceptance oracle is +0.50 pp**
on POT, so no screen of any kind can buy more than half a point here. Read the replay first
next time.

## 2026-09-03 — the paper's open holes are closed: mask-right/pose-wrong gets a figure, supplementary A and C are written
Nothing new was run for the method; this is paper work plus one small offline instrument.

**The mask→pose failure is now shown, not asserted.** `experiments/mask_pose_failure.py`
scans the shipped run's stored `loma_H`/`samh_H` for POT frames where the mask branch is far
from GT while the keypoint branch is on it — 42 such frames across 210 sequences — and
renders them. Masks are not kept by a normal run, so a `--save-masks` re-run of
`loma_samh_matchnaming` on six sequences went to `$OUT/maskpose_maskdump` (branch errors
reproduce exactly, same architecture). The paper figure is `V22_5` frame 171 and `V29_3`
frame 251: correct SAM3 mask, mask-derived pose off by **184 and 365 px** while the keypoint
branch is within **0.7 and 1.7 px**; an occluder truncates the silhouette on one, the image
border on the other (GT corners at y = −30). Both frames selected `loma_H`, so the tracker
was right and only the branch was wrong. This replaces the 119.7 px / 0.2 % IoU number from
2026-08-25 in the paper — the maintainer's call: a picture beats an unclear number.

**Supplementary A and C written; the naming section got its two caveats.** A restates the
SAM-H mask branch (contours → gradient Hough → line refit → intersections → assignment
enumeration → 4/3-2/1-corner transformations) and locates the cyclic ambiguity in the
nearest-to-incumbent rule. C carries the complete cross-dataset table (p@5, p@15, br-gap,
six benchmarks, four methods) and states the MPOT-3K protocol honestly: 573 single-target
tasks under alignment-error precision, **not** the benchmark's MOTA/IDF1/success rate, which
is still not computed. B now says the claim is the plateau not the argmax, and that this
train sample has ≥8 matches on every localized frame against 87.2 % on PTi.

**Also:** the determinism sentence names the carrier (matcher, not SAM3) and reports deltas
of both signs; `overview.png` deleted; `dinov2color` retired from the preamble in favour of
`maskposecolor` for the new overlay.

**Still open:** the paper has never been compiled — no LaTeX on the HPC — so table fit,
float placement and the review-ruler patch remain unverified on the workstation.

## 2026-09-03 (later) — the mask branch does not fail at the lines: it fails at the four-corner requirement
Maintainer looked at the new failure figure and asked why the fitted quad shares no side with
an obviously good mask. Probed it (`$OUT/scratch/maskh_probe*.py`, replay of the stored run's
masks and `samh_H`; the estimator was `hough` on 500/500 frames of both sequences, so no
aligner and no iterative fit is involved).

**Line extraction is fine.** On `V22_5` f171 three of the four detected lines sit on true
target edges at 0.6°/1.2 px, 0.0°/0.8 px and 0.2°/1.4 px; the fourth is the occluder's edge
(29.9°, 76 px off). On `V29_3` f251 the mask is 0.987 IoU against the annotated quad and three
lines match true edges at 0.1°/0.3 px, 0.4°/1.2 px, 0.0°/2.0 px; the fourth is a 7 px stub at
the image border.

**The loss happens at `lines_to_points` → `find_closest_H`.** Consecutive line pairs give at
most four candidate corners; the truncation edge destroys one or two, the 80 px near-contour
filter drops them, and with 2--3 surviving corners `_find_transformation` takes the
similarity branch — a 4-DOF TRS composed onto the branch's own incumbent — **discarding the
three good lines entirely**.

**The correlation is exact, and it is a cliff, not drift.** Per-frame replay: every frame with
4 corners estimates by DLT and overlaps its own mask at 0.98--0.99 IoU; every frame with 2--3
corners drops to 0.00--0.40 in one step and recovers just as abruptly (V22_5 171--185 sick,
186 healthy; V29_3 196--311 with brief healthy islands at 215--222, 245--249, 257--259).
Replay reproduces the stored pose to 0.0 px except on frames where `naming_shift != 0`, which
is match-naming re-labelling applied afterwards.

**Consequence for the paper:** the caption's old claim ("the boundary lines are fitted to a
smaller quadrilateral") was wrong and is fixed — the lines are right, the corner-count
fallback is what loses the pose. Limitations now names the fix direction: estimate from the
visible lines rather than from four corner points. `experiments/mask_pose_failure.py --stages`
reproduces the per-frame table.

## 2026-09-03 (later still) — mask-branch line fitting: three numerical failure modes found and fixed, one info-theoretic limit remains
Investigated why the mask-pose figure's fitted quad shares no side with its own mask (maintainer
question). Root cause was already known (`lines_to_points` only intersects *consecutive* detected
lines, so losing one line costs two corner intersections and the pipeline falls back to a 4-DOF
similarity, discarding 2-3 correctly-fit lines) but the obvious fix — solve from lines directly —
needed three rounds before it held up, each caught by looking at the actual case image, not a
metric:

1. **Null-space projection was in the wrong space.** Solving lines exactly and projecting the
   incumbent onto the unconstrained null space is nearest in matrix (Frobenius) coordinates, not
   pixel coordinates. Measured disagreement on real cases was only a few px, smaller than feared,
   but the principle is unsound.
2. **Two-line candidates blow up.** Switched to weighted soft-point regularization (incumbent
   prediction as a low-weight point in the same normalized least-squares system) — sound, but with
   only 2 lines the Gauss-Newton refinement had nothing holding it down and produced a corner
   1500+ px outside the frame. Fixed: require >=3 lines.
3. **A thin sliver quad passes coverage+area gates.** A convex quad can be narrow far from the
   mask and wide where it matters, covering 99.5% of the mask while a corner sits at x=-899 in a
   1280px frame. Fixed: reject any corner beyond 0.5 frame-diagonals outside the image.
4. **Refinement can still make a verified fit worse.** Final design computes linear + refined,
   re-verifies both independently against the mask, keeps whichever is valid and better.

**Result on 29 frozen POT-210 cases** (`experiments/maskh_cases.py`, `--save-masks` re-run of
`loma_samh_matchnaming`, 6 sequences): 12 originally-failing cases go from 125.6 to 80.7 px mean
GT error, silhouette explanation 0.70 -> 0.95; 17 healthy cases roughly hold (2.1 -> 2.8 px), one
crosses the 15px line by 2.7px (per-corner soft pull loses the rigid-shape consistency the
deployed similarity fallback gets for free). 3 genuinely 2-line cases correctly return "no fit"
rather than guess wrong.

**The remaining floor is information-theoretic, not numerical.** `V22_5` frame 171: three lines
recover within 1-2 px, but the object moved ~70px between frames in exactly the missing corner's
direction, and nothing here can see it — only the keypoint branch's own estimate could. Points at
fusing branches on the missing corner rather than pushing the mask branch further alone.

New code (`flatsam2/line_homography.py`, `experiments/maskh_cases.py`,
`experiments/maskh_line_fit.py`) is diagnostic only — not wired into the tracker, no config
references it, every existing config runs unchanged. Report with case images:
https://claude.ai/code/artifact/0dacf056-99a6-4aae-b7d4-62fb938b7974

## 2026-09-04 — the WOFTSAM+SAM3 "segmenter control" measures bf16 autocast, not SAM3

Chased the draft-review TODO: why does the SAM3 segmenter control cost \woftsam 9.6 pp
pooled p@5 on POT and 13.3 pp on PTi while p@15 barely moves. It is not a segmentation
effect. Median alignment error rises globally (POT 1.00 -> 2.96 px, PTi 4.15 -> 5.63 px, worse
on 96% / 82% of sequences), a fresh SAM2 replica reproduces the SAM2 run, and the SAM2/SAM3
runs already differ at the *first tracked frame* in 94% of POT sequences — where the mask
cannot influence anything, since `woftsam_samfallback_nl` uses the mask only in the
redetection/fallback branches and sets no `nn_scorer`.

Cause: `sam3/model/sam3_tracking_predictor.py` enters a bf16 `torch.autocast` in its
constructor and **never exits it** ("keep using for the entire model process"), so once a SAM3
predictor exists every later CUDA op in the process runs in bfloat16 — the legacy tracker's
RAFT flow and its least-squares homography included. `sam_track` guards the SAM2 path by
yielding inside `autocast(enabled=False)`; the `sam3_track` adapter does not. An 8-bit
mantissa there is exactly a small global pose degradation: fatal at p@5, nearly invisible at
p@15. (TF32, also enabled process-wide by `sam3.model_builder`, was the first suspect and is
*not* the cause: a live SAM3 run with TF32 forced off scores 82.17 p@5 vs 82.37 with it on.)

Verified by replaying the dumped SAM3 masks through the SAM2 code path (new
`experiments/run_legacy.py --precomputed-masks`, plumbed through `slurm/submit.py`). Full
POT-210: replay 92.13 p@5 / 97.21 p@15 / 1.05 px median against the same-cluster SAM2 rerun's
91.93 / 97.21 / 1.06 and live SAM3's 82.37 / 96.61 / 2.96. 118 of 210 sequences are
bit-identical to the SAM2 rerun and per-sequence p@5 moves by >1 point in 3 of 210 — SAM3
masks change essentially nothing once the bf16 context is gone; if anything p@5 is 0.2 pp
better. Confirmed independently by the new `--guard-autocast` (wraps `sam3_track` the way
`sam_track` wraps itself, legacy repo untouched): live SAM3 guarded scores 92.13 / 97.21 on
POT, agreeing with the replay to 0.000 px median over all 210 sequences.

**With the leak closed, SAM3 masks help both baselines** (per-sequence p@15): legacy WOFTSAM
78.20 -> 79.09 on PTi and 96.37 -> 96.37 on POT; legacy SAM-H 88.38 -> 89.25 on POT and 80.34 -> 82.48 on PTi. Pooled p@5: WOFTSAM 91.93 -> 92.13 (POT), 57.74 -> 58.26 (PTi); SAM-H
68.63 -> 70.92 (POT). The draft's "newer masks do not uniformly improve homography precision"
has the sign backwards. Also measured: legacy SAM-H with SAM3 masks (70.92 / 90.34 POT) vs
flatsam2's own `hough_symmetry_only` (70.42 / 90.34), so the reimplementation confound in the
SAM-H control row is ~0.5 pp at p@5 and zero at p@15.

Consequences: Tab. 1/Tab. 2's \woftsam SAM3 rows must be replaced by guarded numbers and the
"newer masks" sentence dropped; the \flatsam-vs-best-SAM3-baseline margin was computed against
a crippled baseline; and **flatsam2 itself has no autocast guard** (`flatsam2/tracker.py`
imports `SAM3StreamingTracker`), so every reported \flatsam number ran its flow, matcher and
homography solves in bf16 too — uniform across our rows, but worth measuring guarded. Details and the paper's remaining open items: `paper/draft_review.md`
(trimmed from 769 to ~160 lines; history in git).

## 2026-09-14 — paper: buggy SAM3 control rows replaced by the corrected ones

All legacy arms complete (SAM-H + SAM3 masks on PTi finished: 65.03 / 82.39 pooled). Paper
updated: Tab. 1's \woftsam SAM3 row 82.4/96.6 POT and 44.1/77.2 PTi -> 92.1/97.2 and
58.3/79.2; its \samh SAM3 row now the legacy replay (70.9/90.3, 65.0/82.4) instead of
flatsam2's `hough_symmetry_only`, so both halves of both control rows share one codebase.
Tab. 2 switched to same-cluster legacy pairs (\woftsam SAM2 = parity re-run): \samh +2.1 PTi
/ +0.9 POT, \woftsam +0.9 PTi / +0.0 POT. "Newer masks do not uniformly improve homography
precision" dropped — sign was backwards; the \flatsam margin against the best SAM3-equipped
baseline is 5.7 pp PTi / 1.3 pp POT, not 5.1 / 2.1. Added a paragraph disclosing the bf16
autocast leak and the two independent fixes. M1 rounding fixed (96.8 -> 96.7, 92.6 -> 92.5).
Open: \flatsam itself is still unguarded (A1 in `paper/draft_review.md`), and the guarded
*live* SAM3 \woftsam run on PTi is missing (A2). Next: rewrite abstract/intro around the
real contribution — dropping pre-warp-dependent flow for mask-focused LoMa-R correspondences.

## 2026-09-14 — paper story rewritten around dropping the pre-warp

New framing: the contribution is removing pre-warp-dependent optical flow (bad pre-warp =
unrecoverable, and the flow field cannot see it) and replacing it with rotation-invariant
LoMaR correspondences matched init->current + MAGSAC, which carries no pose prior at all.
Naive whole-image matching is not competitive (`LoMaR__USAC_MAGSAC`: POT 65.1/76.1, PTi
43.6/64.5 pooled) because the target is a median 3.2% of a PTi frame and gets a median 14.8%
of the correspondences (<20 target matches on 27.8% of frames), so the robust consensus can
fit the background; density on target is 3-6x higher, still outnumbered (stats from
`$OUTPUT_DIRECTORY/../LoMa/LoMaR/*/**_analysis.json`). Mask-focused sampling (in-mask first
to the full 4096 budget, background fills the remainder) is the fix: 97.3 / 78.2, i.e. +21.2
POT / +13.7 PTi. New Tab. (\ref{tab:sampling}) is the ladder whole-image -> n_object=500 ->
full budget -> full tracker, from `LoMaR__USAC_MAGSAC`, `loma_r_nogrid_MAGSAC`,
`loma_r_allobj_inflate10`, `loma_samh_matchnaming`. Corner naming demoted to contribution
(iii). Abstract, intro, related work, conclusion rewritten; new §4.3. Known gap, now in
Limitations: the background *fallback* is never ablated (no object-only run exists).

## 2026-09-14 — one aggregation, and the story retargeted at "best on both benchmarks"

Read-through found the paper mixing pooled and per-sequence means (same tracker appearing as
98.5 / 98.1 / 97.7 POT p@15) and a POT per-sequence basis that scored zero-GT `V26_7` as 0.
Resolved by going pooled everywhere — the POT toolkit's and the ECCV paper's own aggregation
(`np.concatenate(all_errors); mean(err <= thr)`), which also sidesteps the V26_7 basis.
Recomputed on the cluster: `experiments/branch_p_at_k.py` now prints pooled (jobs 11596963/65),
`inlier_threshold_sweep.py` already did (11596966/67); everything else from `overall_*` in the
eval JSONs; naming CIs re-bootstrapped pooled. Key pooled numbers: branches POT 97.41 (kp) /
90.64 (mask) / 98.51 (selected) / 99.00 (oracle), PTi 78.01 / 84.33 / 88.05 / 91.23.
Story per the user: the claim is simplification + independent branches with one comparison
instead of WOFTSAM's cascade + **leading POT-210 and PlanarTrack at once**, which neither
WOFTSAM (POT) nor SAM-H (PT) did. SAM3-vs-SAM2 out of the abstract, into implementation
details + its ablation. Title now "FlatSAM2: Segmentation-Focused Correspondences for Precise
and Robust Planar Tracking". Measured rather than guessed: UCSB targets are small (median 3.2%
of the frame, same as PTi), so the UCSB paragraph now says what is true — continuous,
unoccluded, repetitively textured planes are where dense flow still wins.

## 2026-09-15

Paper: abstract/intro rewritten (pre-warp explained via WOFT; keypoint failure = top-k
proposal lands on background, not consensus); sections merged into single `paper/main.tex`;
future-work section added. Diagnosed the mask-branch step failure (Fig. mask_pose_failure):
replayed `mask_to_homography` on V22_5 f150–189 and V29_3 f230–269 with the branch's own
incumbent (`slurm/logs/maskh_diag3.py`, scratch). Chaining math (`trs @ last_H`) is correct.
Cause is `find_closest_H`: it only tries max-cardinality matchings and has no per-point gate
against the incumbent, so a truncation-edge intersection (on the contour, so it survives the
80 px filter) 44–211 px from the predicted corner is forced into the 2–3-point similarity
fit. A fit on the good corners alone would have degraded gracefully. Candidate fix: gate
correspondences by distance to incumbent-predicted corner, try all cardinalities. Untested.

## 2026-10-23

Demo setup: `.env.example` + `.env` loader in `flatsam2/path_config.py` (all paths env-overridable, HPC defaults unchanged), `demo.py` (dataset sequence / own video, `--slurm`), SLURM partitions + mail-user from env in `slurm/submit_cmd.py`. Untested; LoMa dep (`github.com/davnords/LoMa@8fb59c4`) not yet in `pyproject.toml` pending dist name. Next: HTML demo page.
SLURM: hardcoded personal email removed (FAIL mails → submitter; summary mail from `SLURM_MAIL_USER`), partitions from `.env`. `demo.py --set key=value` + `params` (tweaked configs flattened to `OUTPUT_DIRECTORY/demo_configs/demo_<hash>.py`); LoMa added to `pyproject.toml` (dist name unverified). Tests: `tests/test_demo.py`.
setup.sh: one-time setup (.envrc from `MODULES`, venv, `pip install -e .`, HF login, config smoke check); flow is fill `.env` → `./setup.sh` → `python3 demo.py dataset --slurm`. Untested; weight pre-fetch for offline nodes pending.
setup.sh now follows SETUP.md: all-`--no-deps` install with pinned ptlflow/sam3/LoMa commits, module torch/numpy check, HF token + weight download on login node; `tracker.py` passes SAM3 `debug` only for candidate_dump (public sam3 fork lacks it). Untested.
setup.sh installs `requirements-lock.txt` (freeze of the working rci venv, minus numpy + private flatsam2 line) 1:1 with `--no-deps`; resolver-based rebuilds kept drifting (fork pins contradict modules).
Videos: legend + GT (green), SAM-H branch from `samh_H` debug telemetry (cyan), SAM3 mask (orange, 30%), LoMaSAM output (red) in `visualize.py`; own-video demo shows only the LoMaSAM legend. Fixed `--draw-masks` path (`results/masks/<seq>/`).

## 2026-10-28

Emails only from `.env` (`SLURM_MAIL_USER`): `migrate_legacy_results.py --eval-mail-to` default, author email dropped from `pyproject.toml`. setup.sh: one run (opens `.env` in `$EDITOR`), fails early if venv python >=3.13.

## 2026-11-03

Portable demo: `./setup.sh && source .envrc` now works on any Linux x86_64 + NVIDIA box (uv → own Python 3.12, full `requirements-lock.txt` freeze incl. pip torch 2.7.1+cu126, no modules, no `$EDITOR`, repo-relative `.env.example` defaults; video mode requires `--corners`, clicking removed: servers have no display). Tested from a clean clone on CMP `woska` (Debian 12, RTX 4000 Ada; dataset V01_7 P@5 75.0 = RCI, video mode) and on RCI (login-node setup + `--slurm`). README "Demo" section.
