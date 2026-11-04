# Review of the WACV 2027 draft (paper/)

Rolling review, trimmed to what is still open. Pass history (passes 1–13, all the resolved
blockers, factual re-verifications and author responses) is in git — `git log -p
paper/draft_review.md`. Scope: adherence to the WOFTSAM/SAM-H ECCV camera-ready
(`~/papers/2026-eccv-camera_ready/970-paper.tex`), factual correctness against results on
disk, LLM-isms. Not a scientific review. **Keep this file under ~200 lines.**

---

## A · RESOLVED (2026-09-04): the WOFTSAM+SAM3 control measured bf16, not SAM3

SAM3's predictor enters a bf16 `torch.autocast` in its constructor and never exits it, so a
naive swap ran the legacy tracker's flow and homography solve in bfloat16 — fatal at p@5,
invisible at p@15, present from frame 1. Fixed two independent ways (`--guard-autocast`,
`--precomputed-masks`), agreeing to 0.000 px over 210 POT sequences. Corrected control rows
are in Tabs. 1–2; full account in `docs/journal.md` 2026-09-04. With the leak closed, SAM3
masks help both prior estimators (\samh +1.9 PTi / +0.9 POT, \woftsam +0.9 / +0.0 pooled
p@15), and both control rows now come from the legacy codebase, so the reimplementation
confound is gone too.

**Still open from that thread.** (A1) flatsam2 has no autocast guard either, so every
reported \flatsam number ran its matcher and homography solves in bf16 — uniform across our
rows, but it cost \woftsam ~10 pp of p@5; measure it guarded. (A2) The guarded *live* SAM3
\woftsam run on \pttst is missing; that row is replay-only.

---

## A2 · Resolved in the 2026-09-14 story pass

- **One aggregation.** Everything is now pooled precision over annotated frames, the POT
  toolkit's and the ECCV paper's convention (`third_party_examples/pot_eval.py`:
  `np.concatenate(all_errors); mean(errors <= thr)`). Branch and threshold tables were
  recomputed on the cluster (`experiments/branch_p_at_k.py` gained pooled output, jobs
  11596963/65/66/67); everything else came from `overall_*` in the eval JSONs. The naming
  bootstrap CIs were redone on the pooled metric (3.09 pp [1.85, 4.44] vs none; 0.37 pp
  [-0.37, 1.15] vs DINOv2).
- **The V26_7 basis issue is deliberately not fixed.** POT per-sequence means scored the
  zero-GT sequence as 0; pooled aggregation sidesteps it entirely and matches prior work.
- **Story.** Claim is now simplification + independent branches + leading POT-210 and
  \pttst simultaneously (\woftsam led POT, \samh led PT, neither both). Title changed to
  "Segmentation-Focused Correspondences for Precise and Robust Planar Tracking". The
  segmenter upgrade left the abstract for implementation details + `tab:segmenter-control`.
  "No pose prior" is scoped to the keypoint branch.
- **Verified rather than asserted:** UCSB target size (median 3.2% of frame, n=96) replaced
  the guessed "large, well-textured targets".

**Still open from the read-through.** (a) No experiment measures the pre-warp failure
directly -- deferred by the author; the WOFTSAM paper already argues WOFT's fragility, and
the oracle row plus the visuals carry it for now. (b) The sampling table's whole-image row
uses the matcher's own default keypoint selection, not our 4096 budget with the mask off --
that matched-budget run does not exist and a reviewer may ask. (c) The background-keypoint
*fallback* is still unablated. (d) Page budget still leans toward naming (its own
subsection, ablation, two supplementary tables, the whole runtime table).

---

## A3 · Reviewer verification of the pooled re-aggregation (2026-09-14)

Re-checked the conversion against `eval_results.json` rather than trusting it. **Everything
matches.**

- Naming ablation (PTi pooled): 84.96 / 89.41 / 4.45, 87.68 / 89.25 / 1.57,
  88.05 / 89.41 / 1.36 → printed 85.0/89.4/4.5, 87.7/89.3/1.6, 88.0/89.4/1.4 ✅
- Cross-dataset, **all 24 cells** ✅ (\woftsam 77.59/97.51/99.17/89.12/96.63/90.83; \samh
  80.53/89.44/80.78/54.28/54.34/82.03; DINOv2 87.68/98.51/99.29/94.46/92.20/92.62; ours
  88.05/98.51/99.27/94.73/92.18/92.63). "Five of six" and the worst-column claim hold.
- POT per-condition, **all 24 cells** ✅, recomputed as frame-weighted pools; blur +5.7 pp ✅.
- Main table and sampling ladder ✅.
- Pooled bootstrap CIs reproduce exactly: **3.09 pp [1.85, 4.44]** vs no naming and
  **0.37 pp [−0.39, 1.15]** vs DINOv2 (the draft prints [−0.37, 1.15]; resampling noise at
  the second decimal, not an error — but if it is quoted to two decimals, fix the seed and
  say so).
- No stale per-sequence text survives: "per-sequence", "valid ground truth" and the
  209-basis caveat are all gone, and the threshold table's POT column reads 98.5 in line
  with the headline. **The 209/210 hazard is retired** — remove it from §B's standing note.

Two additions to the open list:

- **P3 (carried, still open). The bf16 fact has no home in the paper.** Grepped again: no
  mention of precision context, autocast or bfloat16 anywhere. The SAM3 control rows were
  produced with a guard or a mask replay; a naive swap measures 44.1 pooled p@5 on \pttst.
  One sentence or a supplementary footnote — reproducibility, not process colour.
- **P4 (new, minor). "TMT, where p@15 increases by 0.2\,pp"** — exact is 94.73 − 94.46 =
  **0.27** → 0.3. The 0.2 comes from subtracting already-rounded values, the same habit that
  produced M1. Compute deltas from unrounded numbers.

On the author's own open item (b): the whole-image row uses the matcher's default keypoint
selection rather than our 4096 budget with the mask off. **This is the row a reviewer will
attack**, because contribution (ii) — the +21.2 / +13.7 pp headline — rests entirely on it,
and "no mask" is currently confounded with "different budget". The matched-budget run is
~510 sequences and would make the claim airtight; of the four open items it is the one worth
GPU time.

---

## A4 · Pass 17 — the budget-matched mask-off ablation is wired (not yet run)

`configs/loma/loma_r_allobj_nomask.py` plus `current_mask_sampling` in `config.py` is exactly
the control I asked for, and the implementation detail is right: an all-ones mask rather than
`mask=None`, so the budgeted sampling path is kept and the detector's own keypoint count does
not silently replace the 4096 budget. That is the difference between an ablation and a
confound.

**Q1. When it lands, the table will have three conditions, not two — label them precisely.**
The new config still uses the mask on the *initial* frame (`init_mask_inflate_px=10`, exact-GT
init sampling); only current-frame focusing is off. So:

| row | init frame | current frame | budget |
|---|---|---|---|
| whole image, no mask (existing) | no mask, detector's own selection | no mask | detector default |
| **new: mask off, budget matched** | mask, 10 px dilation | whole frame | 4096 |
| mask first, full budget | mask | mask first | 4096 |

If the new row is dropped in beside the old one without saying this, the two "no mask" rows
will differ for two reasons at once and a reviewer will ask which one supports the +13.7 pp
claim. Cleanest presentation: keep the off-the-shelf row as "the recipe without a tracker at
all", add the new row as the controlled ablation, and quote the *controlled* delta in the
contribution list. The off-the-shelf row is still worth keeping — it is the honest answer to
"why not just run a matcher?" — but it should not be the number the claim rests on.

**Q2. Two keypoint-branch numbers now coexist** — `tab:sampling` "mask first, full budget"
is 92.4 / 97.3 / 56.7 / 78.2, `tab:branch-complementarity` "keypoint branch" is
92.5 / 97.4 / 56.7 / **78.0**. Same branch, different harness (standalone run vs the branch
estimate stored inside the tracker). The gap is small but a reader who cross-reads will spot
it; one clause in the branch-table caption ("branch estimates recorded inside the full run,
so they differ marginally from the standalone run of \cref{tab:sampling}") settles it.

---

## A5 · Pass 19 — abstract tightened; one claim now rests on the wrong row

R1 is fixed ("estimates the homography", parallel with the mask clause), and the opening
sentence is better than either previous version — it maps the *target* from initial to
current pose rather than mapping a pose. The merged pre-warp sentence ("so that one wrong
pose derails every estimate after it") is the sharpest statement of the paper's motivation so
far. Numbers still check: 64.5 → 78.2 and 97.3 are the sampling ladder's rows.

**S1 (new, and the most attackable sentence in the paper). The abstract's headline lift is
measured against the row that is not budget-matched.** "lifts the same matcher and estimator
from 64.5 to 78.2 p@15 on PlanarTrack, and to 97.3 on POT-210" uses the off-the-shelf
whole-image row, where *two* things differ from the focused run: no mask **and** the
matcher's own keypoint count instead of the 4096 budget. `loma_r_allobj_nomask` exists
precisely to separate those, and has not run. Until it does, the abstract attributes to mask
focusing an effect that is partly selection/budget. Two options: run it before submission and
quote the controlled delta, or soften to "lifts ... from 64.5, as the matcher would be
applied without a tracker, to 78.2". The first is much better — it is one 510-sequence job
and it converts the paper's main contribution from arguable to demonstrated.

**R2 (escalated). Corner identity is now unmotivated everywhere.** The intro's clause
("they need the four resulting target corners to keep their identity --- corner one must stay
corner one") has been deleted too, so neither the abstract nor the intro ever says *why* a
mis-named corner is a failure. What survives is mechanical: a silhouette is invariant under
four relabelings, so the mask branch cannot tell which corner is first. That explains the
ambiguity but not the cost. Contribution (iii) is corner naming, §Experiments spends a
subsection and a metric (\brgap) on it, and the reader has been given no reason to care.
Restore one clause — registration, AR overlays and anything composited onto the plane break
under a cyclic relabelling even when all four corners are geometrically perfect — in the
intro's first paragraph. Six words of motivation for a third of the contributions.

**R3 (carried). "as well as three further datasets"** still omits that there are four and
that UCSB is the exception, which the body names openly. Six words restore the paper's own
standard of candour.

**T1. "dominates both POT-210 and PlanarTrack with a single model"** replaced "is, to our
knowledge, the first to lead both ... at once". The hedge went and the verb got stronger, in
the same edit. The margin over \woftsam on POT-210 is 1.0 pp p@15 and 1.5 pp p@5;
"dominates" invites a reviewer to measure it. "Leads both" was accurate and unattackable, and
the "to our knowledge" hedge on a first-ness claim is worth keeping.

**T2. Contribution (iv) — the controlled evaluation — was deleted.** Defensible on space,
but that bullet was the paper's standing answer to "you just used a better segmenter", and
the segmenter controls remain one of its strongest defensive assets (\cref{tab:segmenter-control}).
If it stays out of the list, make sure §Experiments' framing sentence still advertises it;
right now the argument survives only where a reader who reaches §4.2 will find it.

**T3. "no homography can be fit from what is left"** is absolute where the measurement is
statistical (fewer than 20 target correspondences on 27.8\% of frames). "The target's
correspondences cannot outvote the background" says the same thing and matches
\cref{tab:sampling}.

---

## A6 · Pass 21 — single-file merge, and a contradiction in the conclusion

**The merge into one `main.tex` (625 lines, `sec/` and the stale `sec/auto/` gone) matches
the reference paper's own structure** — the ECCV source is a single 705-line `.tex` that
inputs only figures. The Makefile still points at `main.tex`, the supplementary is still
guarded by `\ifcombined`, and my static check is clean: no duplicate labels, no unbalanced
environments or braces, all 26 `\cite` keys present in the bib. (The only "dangling refs" my
checker reports are an artefact of its regex on the nested `\xref{\cref{...}}{...}` form.)

**U1. FIXED.** The conclusion now reads "UCSB, where dense flow remains stronger", dropping
the "large well-textured target" description that contradicted the measured median in
§Failure Analysis. The mechanism sentence in §Failure Analysis (continuous, unoccluded
tracking of a repetitively textured plane) carries the explanation on its own.

**U2. Check the UCSB 3.2\% against its own script.** Measuring independently (GT quad area
over 640×480, per-sequence median over frames, then median over the 96 sequences) I get
**3.0\%**, range 2.9–13.4\%. Close but not equal — and 3.2\% is also exactly the
PlanarTrack number quoted three times elsewhere, a coincidence worth ruling out. Either
convention may be right (valid frames only, or area over annotated frames rather than
median-of-medians); just confirm which produced the printed number.

Carried, unchanged by this merge: R2 (corner identity unmotivated in both abstract and
intro), R3 (abstract still says "three further datasets" without naming UCSB, while the
conclusion now says "five of six ... UCSB" openly), T1 ("dominates"), T3 ("no homography can
be fit"), S1 (the abstract's 64.5 → 78.2 lift still rests on the non-budget-matched row;
`loma_r_allobj_nomask` has no results on disk yet), P3 (bf16 fact), P4 (0.2 vs 0.27 pp).

---

## A7 · Pass 22 — the new Future Work section

The content is good and some of it is genuinely useful to the field (the honest catalogue of
why general mask-to-homography is hard; the benchmark proposal). Three problems, all about
register and form rather than fact.

**V1. It is written in lab voice.** "This is the major one." / "Hierarchical, coarse-to-fine
schemes --- fit a translation first, then add rotation, and so on --- did not work well for
us either." / "We do not expect large gains, as SAM usually segments well, but **story-wise
it would be nice**." / "Ideally the decision is ``intelligent''". The last one would be
quoted back in a review. This is the first place in the draft where the internal voice of
the journal has reached the paper; everything else has been scrubbed of it. Rewrite in the
register of the rest of §Limitations, which is already right.

**V2. Form deviates from the reference paper.** The ECCV paper keeps one section,
"Limitations and Future Directions", as prose with First/Second/Third paragraphs — no
`itemize`. The draft now has two adjacent sections and a four-item bulleted list, in a paper
carrying 8 tables and 4 figures in the main text plus 3 more tables in the supplementary.
Merging back into one prose section is both closer to house style and the cheapest page
saving available.

**V3. Unsupported negative results.** "coarse-to-fine did not work well for us" and the
claim that the plain $3\times3$ parameterization is numerically poor are informal evidence
presented as fact. Either attribute them explicitly ("in our experiments", one clause each)
or drop them; a reviewer may ask for the experiment, and `docs/dead-ends.md` is not
citable.

Two smaller points: "Our SAM3 version already has a correction API" is an implementation
aside — say "SAM 3 exposes a mask-correction interface". And the benchmark proposal restates
a call the previous paper already made in its conclusion ("future planar tracking benchmarks
should include non-quadrilateral targets"); acknowledging that continuity costs one clause
and reads as a programme rather than a repeat.

---

## A8 · Pass 23 — TWO SUBMISSION HAZARDS (2026-09-15)

**W1. The build mode was switched to the arXiv/camera option, which de-anonymises the
paper.** `main.tex` now has `\usepackage[pagenumbers]{wacv}` active and
`% \usepackage[review,algorithms]{wacv}` commented out. In `wacv.sty` the `review` option is
what clears the `wacvfinal` toggle; with it gone the toggle stays true and `\maketitle`
prints `\@author` verbatim — **real names and institutional emails on page 1**, with no
review rulers and no Paper ID. The submission build must be
`\usepackage[review,algorithms]{wacv}`, which renders "Anonymous WACV Algorithms Track
submission" plus `\wacvPaperID` instead.

Two details: the `algorithms` option is gone too, and in review mode its absence triggers
`wacv.sty`'s literal ":-( YOU HAVE NOT YET SET THE DATASETS, APPLICATIONS, OR ALGORITHMS
TRACK OPTION" block on the title page — restore **both** options together. And the ruler
patch (L3) is guarded on `wacvfinal`, so it is silently inactive in this mode; a build made
now cannot check it either.

If the switch was deliberate — reading the paper with authors visible — keep it in the
working tree rather than a commit, or add a `make arxiv` target passing the option through
`-usepretex`, the way `combined` already does.

**W2. Raw brainstorm notes are now in the committed paper.** In §Future Work: "Maybe even
something crazy like wire EDM stuff" with **two YouTube URLs**, "(hard case for companions
and when to trust them, also pretty much texture-less and overall nicely crazy)", and an item
headed "This but with arbitrary other local features" whose body is "Maybe LoMa is not so
important here and a segmentation-mask-densified SIFT (or \dots) would work too."

The *ideas* are worth keeping — the SIFT one especially, because "is this about LoMa or about
the focusing?" is the first question contribution (i) invites, and the honest answer (we
expect focusing to help any local feature; we did not test it) belongs in Limitations. The
wire-EDM scenario is a vivid instance of the branch-trust problem: one clause, no links.
As written, none of it can ship.

This escalates V1 — Future Work is the only part of the draft still carrying internal voice,
and it is growing rather than shrinking.

**Both hazards are now committed** (`7b3ed9b`), not just sitting in the working tree, so they
will survive a `make backsync` and are easy to forget. The same commit carries three good
edits, which is why they are easy to miss in review: dropping "after the target has been out
of view" from the correspondence-failure list is **right** and worth saying out loud —
init-to-current matching needs no history, so an out-of-view period costs it nothing, which
is exactly the paper's own no-pose-prior argument applied consistently; unbolding the
selected-output 88.0 leaves the oracle as the only bold row, which is correct; and the
shortened p@k definition is still unambiguous.

One small loss in that last edit: the dropped clause also said this is the POT-210 toolkit's
aggregation and the one prior work uses. Some PlanarTrack numbers in the literature are
per-sequence means, so "as in the POT-210 toolkit" was doing work for a reader comparing
against other papers. Four words, worth restoring.

---

## B · Open correctness items

- **M1. Rounding, still unfixed.** `\woftsam` POT-210 p@15 is 96.746 → **96.7**, printed as
  96.8 in `main.tex` cross-dataset table (cross-dataset), `supplementary.tex:90`, and the same
  number appears as 96.8 in the segmenter-control table (`main.tex` segmenter-control table, where
  the Δ is then −1.1, not −1.2). `\flatsam` match-naming POT p@5 is 92.548 → **92.5**,
  printed 92.6 in `supplementary.tex:104`. POIC p@15 98.9495 vs the DINOv2 row's 98.974
  both print 99.0 — pick one rounding convention and apply it per column. Third
  rounding-consistency issue in the draft; the real fix is T5.
- **O1. `fig:mask-pose` caption reports two metrics without saying why** — 0.99 IoU on the
  right panel, 0.99 precision on the left. Add the clause: "precision rather than IoU on the
  left, where the occluder hides part of the annotated region."
- **O2. "within 2 pixels" sits exactly on the boundary** (probe: 1.2/0.8/1.4 px on `V22_5`,
  0.3/1.2/**2.0** px on `V29_3`). "At most 2 px" costs nothing and cannot be contested.
- **O3. Link the four-corner cliff to the UCSB residue.** §Failure Analysis leaves "mask
  aligns, quadrilateral fit inaccurate" unexplained; the four-corner requirement is a
  concrete candidate for exactly that residue, and UCSB has both occluders and
  border-crossing targets. Check corner counts on the UCSB failing frames before asserting
  it — if it holds, one sentence unifies two observations and the Limitations fix direction
  covers both.
- **Tab. 4 bolds `\frst{91.0}` twice** for MPOT-3K (90.98 both): mark the tie in the caption
  or leave both unbolded.
- **Runtime caption** mixes a 120-frame `Seq\_00006` memory profile with timings averaged
  over three 100-frame sequences. Honest but easy to misread; say "peak memory measured on
  one sequence".

**Standing warning:** every `corner_fusion_*` number is void (covariance + alias bugs), the
sole exception being `corner_fusion_naming` PTi 89.21 with no POT half. The draft cites none
of them; if the quad-free thread is ever mentioned, it cannot carry a number.

**Standing convention:** POT tables are on the 209-sequence valid-GT basis (V26_7 has zero
GT). `results_tables._mean` is 210-based, so the generator and the tables must be reconciled
when T5 lands.

---

## C · Open presentational / tooling items

- **T2 / P9 / L1. Nothing has ever been compiled.** No `pdflatex`, `latexmk`, `xelatex`,
  `bibtex`, `tectonic` and no `texlive` module on this cluster, so every LaTeX-level claim
  is unverified: table fit (Tabs. 4 and 5 at `\tabcolsep=4pt`, 8–9 columns), float
  placement, and above all the review-ruler patch (`0b03826`). `\if@firstcolumn` is
  evaluated inside the output routine and does not always match the column being set; check
  the first built PDF page by page, including a page whose second column starts with a
  float, and drop the patch rather than debug it — nothing depends on it. Substitute in the
  meantime: a static check each pass (environments, braces, `\cref` targets, duplicate
  labels, `\cite` keys vs `sjo-bib.bib`) — clean as of the last pass.
- **T5. Generate the tables.** `experiments/results_tables.py` already builds them from
  `eval_results.json`; a `--latex` mode writing `paper/tables/*.tex` for `\input` removes
  transcription risk and is the mechanism that would have prevented M1 and the earlier basis
  bug. Author defers until the metric protocol settles; it should exist before submission.
- **P6. No code or data release.** Neither abstract nor conclusion promises anything; the
  previous paper ends both on the release URL, and a reviewer reads its absence as a signal.
- **P7. Determinism** is stated and verified; nothing further unless the numbers move.
- **K1. UCSB mask colour is orange on an orange-brown scene** with orange fiducial markers,
  between two colours the boundary has to be read against. Maintainer's call, kept
  deliberately; teal was the most legible of the three tried. Not a correctness issue.
- **`woftsamcolor` is defined and used in no caption** — harmless, but it is the next
  `dinov2color` if a figure ever needs a second blue.
- **`\xref` fallback letters are hardcoded** (`\xref{\cref{sec:supp-naming}}{supplementary
  Sec.~B}`). Re-check them whenever a supplementary section is added, removed or reordered;
  nothing warns you, and `supplementary-standalone.tex` sets `\combinedfalse`, so those
  strings are what a standalone-PDF reviewer sees.

---

## D · Nitpicks

- `\flatsam` renders small-caps via `\method{}`; the title spells it plain "FlatSAM2".
  Decide.
- Tab. 3: `selected output` p@5 (65.5) is below the mask branch (65.9) on \pttst — true, and
  bolded correctly. A reviewer may ask why; one clause would pre-empt it.
- POIC is cited to `alismail2016robust` (the Bit-Planes paper). Confirm that is the dataset's
  origin, not merely a method evaluated on it.
- `missing_references.txt` was narrowed to missing/uncertain references only; keep it that
  way — experiment state belongs in `docs/journal.md` and the inline `\citodo` markers.
  `\citodo` is now defined but unused, i.e. no known missing citation.

---

## E · Completeness rule (keep)

A run counts as done only when the chained eval has run and `aggregate.num_sequences`
matches: 300 PTi, 210 POT, 20 POIC, 92 TMT, 96 UCSB, 573 MPOT-3K. All reported runs
currently satisfy this.
