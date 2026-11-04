# Dead ends

Directions that were tried and abandoned, each with the measurement that killed it and
where to find the detail. Kept so they are not silently re-tried, and so the paper can
say why the shipped design is the one it is.

Nothing here is live. The current tracker is in `docs/state-of-the-tracker.md`.

---

## Fusion and combination rules

**Per-corner fuse-or-arbitrate decisions** — *superseded 2026-08-17.*
Deciding fuse-vs-select independently for each corner lets the output take two corners
from one branch and two from the other: a quad neither branch proposed.
`experiments/corner_complementarity.py` (PTi, 122 195 frames) found genuinely
complementary frames are **0.66%** and a per-corner oracle beats a whole-branch oracle
by only **0.48 pp**. Replacing it with one decision per frame *gained* 1.20 pp (POT) and
0.56 pp (PTi) pooled P@15. Kept as `fusion.decision_scope='per_corner'` for
reproducibility of the runs already on disk.

**Covariance fallback for branches that report no covariance** — *rejected 2026-08-17.*
4.06% of PTi frames discard an existing keypoint pose because the branch produced no
covariance, and one such frame (Seq_00350 f275) costs 90 px. Letting the branch compete
under a fixed wide isotropic prior instead of dropping it was replayed at σ = 10 px and
**loses**: −0.09 pp POT, −0.25 pp PTi pooled. A branch that cannot say how sure it is
turns out to be a branch worth ignoring.

**Residual-threshold acceptance of the mask branch** — *rejected.* Gating the SAM-H
covariance on the mask-align residual (the sharpest observable separator available:
accepted-but-wrong fits sit at ~1.8 px vs ~0.95 px for accepted-and-right ones) is
neutral at best on POT and costs up to 4 pp on PTi at useful thresholds. See the
`residual<=T` rows of `experiments/fusion_rule_replay.py`.

**Re-tuning `disagreement_gate` to the nominal χ²₂ 95% quantile** — *rejected.*
Configs `corner_fusion_alignh_calibrated_gate6` (5.99) and `_gate40`. The branches are
too overconfident for the nominal value to be right; the gate is empirical and is
described as such rather than sold as calibrated.

---

## Mask → homography estimators

**Plain mask alignment without a search** (`loma_samh_decoupled_alignh_pure`) —
*superseded by propose/refine/verify.* Competitive only from a good initialization:
P@15 collapses to 15.6 (PTi) and 13.6 (POT) in the `lost` regime against the Hough
quad's 71.5 / 71.9. A branch that cannot recover is a branch that only helps when it is
not needed.

**Moment-based redetection inside the aligner** (`corner_fusion_alignh_redetect`) —
*rejected at grid scale.* Looked fine on four probe sequences, lost 6 pp on the full
grid, **entirely to corner labelling**: br-gap 10.50 pp affecting 35% of sequences. This
is the origin of working rule 4 (read the br-gap first).

**The earlier `loma_samh_decoupled_alignh` run** — *not evidence.* It kept
`align_fallback_hough=True` and `symmetry.enabled=True` and logged nothing about which
estimator produced mask-H, so its −2.41 pp may be measuring the Hough quad. The
telemetry to recover it does not exist in the npz files. Superseded, not disproven.

---

## Verification

**Ranking hypotheses by the fit's own residual** — *rejected.* The residual separates
right from wrong fits only weakly (1.77 px vs 0.95 px) because a silhouette can sit
convincingly on the wrong part of the image. Raster IoU beats it by a wide margin, and
beats every precision-weighted F-measure variant tried. `abl_verify_residual` in
`mask_register.bench_methods`.

**Shape-only verification** — *rejected, and it is the headline mechanism result.*
Removing the texture term costs 4.87 pp of PTi P@15, of which **86% lands in the
br-gap** (2.83 → 7.02 pp, 27% of sequences instead of 11%). Losses are total rather than
gradual, the signature of a pose snapping to a symmetry. Kept as the paper ablation
`corner_fusion_register_noapp` / `abl_verify_shape_only`.

**Retiring the verification score's exponents** — *attempted 2026-08-17, reverted
2026-08-18 on the grid.* Offline, `IoU^a · ((1+NCC)/2)^b` moves ≤0.7 pp anywhere over a:b
from 2:0.5 to 1:1, and a=b=1 measured equal or better in every regime except PTi `good`
(+0.87 POT, +0.02 PTi over the four regimes). It looked like two free parameters to
delete. At grid scale a=b=1 costs **0.77 pp PTi pooled P@15** (88.32 → 87.55 on top of
whole-frame fusion), hurting both localization (`br_p@15` 90.24 → 89.81) and naming (gap
1.93 → 2.26), and buys nothing on POT. The deployed a=2, b=0.5 stands. This is the worked
example behind working rule 9: the offline bench weights the four initialization regimes
equally, and its verdict was wrong by 0.8 pp about a real sequence.

**The alias re-check on the final pose** — *effect not established; within noise.*
Passed the offline bench (+0.8 pp `lost`, p90-when-lost 151 → 129 px) and shipped as
`register_alias_recheck`. Its grid comparisons do **not** resolve: adding it cost 0.14 pp
PTi pooled P@15 (`_final` 87.76 → `_recheck` 87.62), and *removing* it cost 0.15 pp
(`_simple` 87.70 → `_simple_norecheck` 87.55). Opposite operations, same sign — the
signature of run-to-run variation, not of an effect. POT is 98.80 either way, identical to
the decimal. An earlier journal entry called this "does not survive its grid"; that
over-read a 0.14 pp difference and is corrected here. The knob is neither justified nor
condemned by the grids; it costs three verifications and no GN solve, and it stays in the
shipped config until something measures it properly.

**A naive alias re-check on the final pose** — *rejected in that form.* Re-scoring the
winner's three cyclic relabelings after the full-resolution solve loses **3.0 pp P@15 in
the `good` regime** while gaining 0.9 when lost: a naming inherited from the previous
frame carries temporal evidence a single-frame score does not. Shipped only once the
inherited naming had to be beaten by `incumbent_margin`, like any other incumbent.

---

## Solver and cost

**Trimmed Gauss-Newton** (`trim_fraction`) — *not deployed.* It does fix the one
occlusion failure it was built for (a hidden third of the outline otherwise drags an
exactly-correct pose 86 px onto the occluder's edge), but a fit allowed to ignore its
worst quarter is also happy to explain only part of the silhouette: at 0.10, `good`
+1.7 but `drift` −3.8, `cold` −4.9, P@5 −10. Kept as a knob for a possible two-stage
design (converge untrimmed, then polish trimmed).

**The 3×3 chamfer distance transform** (`sdf_precise=False`) — *not deployed.* Another
~20% off the field cost and accuracy-neutral, but it produced **one hard failure in 513
POT dev cases** that the exact transform did not. Kept as `abl_res_sdf_fast`.

**"The SDF is 47% of a frame"** — *a framing error worth not repeating.* That figure is
of the *isolated mask→H branch measured single-threaded*, because `maskh_timing.py`
pins cv2 and BLAS to one thread for quotable per-call numbers. In the tracker
`cv2.distanceTransform` gets 8 cores and drops 33 ms → 7.4 ms, and the whole
registration branch is 9.1% of a frame. Cropping the SDF was still correct and free; its
end-to-end effect is ~1–2%, not 40%.

---

## Process

**Committing on the HPC as a way to preserve work** — *does not work, and cost a session
once.* The workstation's `make sjo_sync` pushes with `--delete` and does not exclude
`.git`, so an HPC-side commit is destroyed rather than merged. Work comes home with
`make backsync`. See `CLAUDE.md`.
