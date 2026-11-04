// Paper-style report of the July 2026 diagnostic campaign on loma_samh.
// Build: typst compile docs/correction_findings_report.typ

#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show table: set text(size: 8.8pt)
#set table(stroke: 0.4pt + luma(150), inset: 4.5pt)
#show link: underline

#let P5 = "P@5"
#let P15 = "P@15"
#let pp = "pp"
#let takeaway(body) = block(
  fill: luma(246), stroke: 0.4pt + luma(185), inset: 9pt, radius: 3pt, width: 100%,
  body,
)

#align(center)[
  #text(size: 15.5pt, weight: "bold")[
    Correspondence--segmentation fusion in planar tracking: \
    a diagnostic study of the `loma_samh` tracker
  ]
  #v(1.5mm)
  #text(size: 10pt)[FlatSAM2 project report --- July 11, 2026]
]

#v(3mm)
#block(inset: (x: 1.1cm))[
  #text(size: 9.5pt)[
    *Abstract.* The `loma_samh` planar tracker fuses a correspondence branch
    (LoMA keypoint matching + RANSAC) with a segmentation branch (SAM3 mask
    #sym.arrow.r homography) through a hard per-frame switch. An online
    "mask correction" mechanism, predicted by an offline feasibility study to
    help on thousands of frames, was measured net-flat in deployment. We ran
    four controlled experiments to explain this and to locate the actually
    recoverable error. (1)~A GT-oracle grid decomposes the correction
    mechanism: injected masks stick, but the deployed trigger fires when
    nothing is wrong, deployable correction content is unavailable on 88% of
    the frames that need it, and even the all-oracle ceiling is only about
    +1#pp #P15 --- correction is capped by the narrow causal path from mask to
    metric. (2)~An error budget shows the dominant recoverable mass (+3.6#pp
    #P15 oracle) lies in *choosing between* the two already-computed
    homographies; two thirds of it sits on frames where LoMA passes the
    inlier gate with hundreds of inliers yet is off by tens to hundreds of
    pixels. (3)~A systematic search over simple online switching rules
    (containment, disagreement, temporal-smoothness features) recovers only
    +0.2#pp of that ceiling: threshold rules are exhausted; a genuinely new
    per-branch quality signal is required. (4)~A convergence-basin study
    validates direct mask alignment as a replacement for the segmentation
    branch's Hough-quad step (93% convergence on clean masks from far beyond
    tracking-scale perturbations, landing at #sym.approx 3 px); the resulting
    tracker prototype is implemented and running. Throughout, decisions are
    made at #P15: the benchmark's ground-truth corners carry #sym.approx 5.7
    px mean annotation error, so 5-px-level metric differences largely
    measure annotation noise.
  ]
]
#v(2mm)

= Preliminaries

== The tracker under study

`loma_samh` (decoupled variant) tracks a planar target by estimating, for
every frame $t$, a homography $H_t$ mapping the initial frame's target
corners to the current frame. Two estimators run in parallel on a shared
SAM3 segmentation stream:

- *Correspondence branch ("LoMA").* LoMA keypoints are extracted on the
  initial frame inside the (dilated) initial target mask, and on the current
  frame inside the dilated current SAM3 mask (the _densify_ / sampling mask);
  matches are fed to seeded USAC-MAGSAC. Precise and orientation-aware;
  degrades under motion blur and low texture.
- *Segmentation branch ("SAM-H").* An independent sub-tracker turns the
  current SAM3 mask into a homography: contour #sym.arrow.r Hough lines
  #sym.arrow.r top-4 line intersections #sym.arrow.r quadrilateral corner
  assignment (+ a symmetry-resolution step). Robust to blur, but
  boundary-quantized and orientation-blind. It advances every frame on its
  own state; nothing feeds back.
- *Switch rule.* The output is the LoMA homography when its RANSAC inlier
  count is #sym.gt.eq 35, else the SAM-H sub-tracker's estimate. On any
  failure the last good homography is kept.

Both candidate homographies, the inlier count, the selection, and (for
correction-enabled configurations) per-frame containment telemetry are
stored in per-sequence debug sidecars; all offline analyses below join these
sidecars with ground truth, without re-running the tracker.

== Benchmarks and metric

The target benchmark is PlanarTrack-initfix (*PTi*; 300 test sequences,
133,320 evaluated frames); POT-210 (*POT*) serves as a regression guard
(the agreed criterion: improve PTi without degrading POT). The metric is
precision at a corner-error threshold: the fraction of frames whose RMS
corner error is below 5 px (#P5) or 15 px (#P15).

PlanarTrack ground truth is imprecise: mean annotation error is 5.71 px and
about 10% of annotations exceed 15 px. All decisions in this report are
therefore made at #P15, with #P5 as a diagnostic only. This choice is
not cosmetic --- two experimental verdicts flip sign between the two
thresholds (@sec:e5-flip, @sec:basin-tau).

Reference operating point (the no-correction baseline run used as substrate
everywhere): PTi #P5 = 65.6, #P15 = 87.7; POT #P5 = 93.5, #P15 = 98.5.

= Why does online segmentation correction not help?

== Background and research question

SAM3 sometimes segments more than the tracked plane --- typically spilling
from the target face onto the whole 3D object. The tracker therefore carries
a *correction* mechanism: when the SAM3 mask looks over-segmented relative
to where LoMA says the plane is (a one-sided "excess depth" test between the
SAM mask and the LoMA-warped initial mask), the tracker injects
$"warp"("init mask", H_"LoMA")$ into SAM3 as an authoritative conditioning
frame, overwriting its appearance memory.

An offline feasibility study had estimated correction should help on 2,481
frames and hurt on 387. The deployed mechanism measured *net-flat*: POT #P5
93.5 #sym.arrow.r 93.5, PTi 65.6 #sym.arrow.r 65.4. The research question:
*why does a mechanism with an apparently large predicted benefit produce
none, and is it fixable?*

Five candidate explanations were formulated:

/ H1 (narrow causal path): The output is the LoMA homography on most frames,
  so a repaired mask changes the metric only via the minority of
  SAM-H-selected frames and via SAM3's future state. Correction could work
  perfectly and still not move the number.
/ H2 (mechanism failure): SAM3 ignores or quickly reverts injected
  corrections (its memory bank outvotes one corrected frame).
/ H3 (trigger failure): The deployed trigger fires rarely, at the wrong
  moments, or on frames where both branches are broken anyway.
/ H4 (content failure): The injected content $"warp"("init mask",
  H_"LoMA")$ is bad --- its boundary is off by LoMA's own error, it paints
  over occluders, and as _authoritative_ conditioning it can poison SAM3's
  memory.
/ H5 (wrong target): The trigger only addresses over-segmentation; it cannot
  address SAM3 jumping to a distractor object.

== Experimental design: a GT-oracle grid

Ground truth enables oracle versions of both correction ingredients: an
*oracle trigger* (fire when the SAM mask disagrees with the GT-quad mask)
and *oracle content* (inject the GT-derived mask). Crossing real and oracle
ingredients isolates each factor:

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, left, left, left),
    table.header([*cell*], [*trigger*], [*injected content*], [*what it isolates*]),
    [E0], [---], [---], [baseline substrate (no correction)],
    [E0+], [deployed, dry run], [none], [counterfactual trigger statistics on unperturbed masks],
    [E1], [oracle (vs GT mask)], [GT mask], [mechanism upper bound],
    [E2], [oracle], [LoMA-warped init mask], [content quality (deployable content, perfect timing)],
    [E3], [deployed], [GT mask], [trigger quality (perfect content, real timing)],
    [E4], [deployed], [LoMA-warped init mask], [the shipped method, fresh telemetry],
    [E5], [every 30 frames], [GT mask], [episodic vs. continuous conditioning],
  ),
  caption: [The oracle-correction grid. Oracle legs abstain on frames without
  valid GT; the LoMA-content leg abstains when LoMA is below the inlier gate.],
)

All cells ran on the full PTi test split. Because correction events are
sparse, global averages are complemented by an *event-conditional analysis*:
the corrected run and the baseline are aligned per frame, and we report the
paired error difference at horizons $k = 0 dots 50$ after each event, the
SAM-vs-GT mask IoU trajectory after each event (does the correction stick?),
re-trigger intervals, and per-sequence #P15 deltas. Paired differences also
cancel the annotation error common to both runs.

== Results

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header(
      [*cell*], [*events*], [*seqs w/ events*],
      [*#sym.Delta#P15 all*], [*#sym.Delta#P15 corrected seqs*], [*#sym.Delta#P5 all*],
    ),
    [E1 oracle trigger + content], [408], [115], [*+0.97*], [+2.71], [+0.57],
    [E2 oracle trigger, LoMA content], [298 (+2,245 abstains)], [59], [+0.12], [+1.98], [−0.11],
    [E3 deployed trigger, GT content], [278], [63], [+0.30], [+2.48], [+0.04],
    [E4 deployed (shipped method)], [159], [65], [+0.02], [+1.11], [−0.22],
    [E5 periodic-30, GT content], [4,270], [300], [*+1.64*], [+1.64], [−1.21],
  ),
  caption: [Final-metric deltas vs. the E0 baseline (per-sequence mean,
  percentage points). "Corrected seqs" averages only over sequences with at
  least one correction event.],
)

Event-conditional facts (all threshold-free, hence unaffected by the metric
choice):

- *E1:* corrected masks stick --- median mask-IoU-vs-GT 0.735 at the event,
  0.89 one frame later, still 0.94 fifty frames later; median re-trigger
  interval 8 frames. The paired error delta is −1.6 px at $k=1$,
  decaying to −0.04 px by $k=50$.
- *E3:* the deployed trigger's events sit at median mask IoU *0.975* ---
  i.e. it fires when segmentation is essentially correct --- vs. 0.735 for
  the oracle trigger; it reaches 63 sequences vs. 115. 60% of re-triggers
  come within 2 frames (bursts).
- *E2:* with perfect timing, the deployable content is *unavailable on 88%
  of trigger frames* (2,245 of 2,543): LoMA is below the inlier gate exactly
  when the mask is broken --- blur kills both branches together. When
  content is available it barely lifts the mask (IoU 0.727 #sym.arrow.r
  0.78) and the trigger re-fires at median interval 1.
- *E4:* the shipped method demonstrably *fixes masks* (IoU 0.834
  #sym.arrow.r 0.94 after events, persistent) while its error delta stays
  #sym.approx 0 and its metric delta is +0.02#pp.
- *E5:* event-local error deltas are #sym.approx 0, yet the aggregate moves
  +1.64#pp #P15 --- gains are whole-sequence regime changes (catastrophic
  loss avoided), not local nudges. The distribution is strongly bimodal: 38
  sequences improve by more than +5#pp, 18 degrade by more than
  −5#pp (extremes +51 / −59#pp).

== Attribution and takeaways <sec:e5-flip>

- *H2 is rejected.* SAM3 accepts authoritative mask conditioning and retains
  it (E1/E4 IoU trajectories rise and stay).
- *H3 is confirmed.* Replacing only the trigger with the deployed one (E1
  #sym.arrow.r E3) collapses the gain from +0.97 to +0.30#pp. The deployed
  trigger fires at IoU 0.975 and misses half the qualifying sequences.
- *H4 is confirmed, in a sharper form than hypothesized.* Replacing only the
  content (E1 #sym.arrow.r E2) collapses the gain to +0.12#pp --- but the
  dominant failure is not _bad_ content, it is _no_ content: the 88% abstain
  rate. Any correction scheme whose content requires a healthy LoMA estimate
  inherits this ceiling.
- *H1 is confirmed as the dominant effect.* Even with both ingredients
  oracle-grade (E1), the dataset-level ceiling is +0.97#pp #P15. E4 proves
  the point at event level (masks fixed, metric unmoved). The feasibility
  study's 2,481 "benefit" frames were counted per-frame in isolation; on
  most of them the output never routes through the segmentation branch
  (the strictly-static recoverable population is 142 frames at 15 px).
- *H5 remains unaddressed by design* --- no grid cell targets
  jump-to-distractor failures.
- *A metric lesson.* At #P5, E5 looks harmful (−1.21#pp) and E4
  looks mildly harmful (−0.22); at #P15, E5 is the best cell and
  E4 is flat. The #P5 "harm" is fine-alignment jitter at the annotation-noise
  floor. Continuous GT-mask coaching genuinely rescues catastrophic
  sequences --- but it is oracle-content-only, high-variance, and E2 says a
  deployable content source does not exist today.

#takeaway[
  *Takeaway.* Online mask correction is capped at about +1#pp #P15 even with
  perfect timing and perfect content, because the causal path from a better
  mask to the benchmark metric is narrow, the deployed trigger fires at the
  wrong moments, and deployable content is missing exactly when needed.
  The deployed mechanism (+0.02#pp, with occasional −19#pp
  sequences) is switched off. The controlled attribution itself --- trigger
  vs. content vs. memory dynamics --- is the reusable result.
]

= Where is the recoverable error? An error budget for selection <sec:budget>

== Research question and method

If correction is not the lever, what is? The debug sidecars of the baseline
run store *both* candidate homographies for every frame, so we can measure,
offline: how good is each branch conditional on being selected, and how much
would a per-frame *oracle selector* (pick the better stored candidate) gain?
Frames are additionally labeled by a SAM-mask state vs. the GT quad
(boundary tolerance 10 px): `ok`, `spill` (mask exceeds target), `shrink`
(target under-covered), `jump` (IoU #sym.lt 0.1).

== Results

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.header([*selected branch*], [*frames*], [*#P5*], [*#P15*], [*median err (px)*]),
    [LoMA (inliers #sym.gt.eq 35)], [99,917 (75%)], [71.7], [94.3], [3.5],
    [SAM-H], [33,403 (25%)], [47.4], [67.8], [5.5],
    [run overall], [133,320], [65.6], [87.7], [3.7],
    [*per-frame oracle selection*], [133,320], [*75.8*], [*91.2*], [3.0],
  ),
  caption: [Branch-conditional accuracy and the selection ceiling
  (baseline run, PTi).],
)

The oracle gains *+3.55#pp #P15* (and +10.3#pp #P5). Of the 16,425 #P15
misses, 4,733 have a stored candidate below 15 px:

- *3,207 (68%) are LoMA-selected frames where SAM-H was fine.* These are
  catastrophic correspondence failures _passing the inlier gate_: median
  LoMA error 30 px (quartiles 19/30/164, 90th percentile 487 px) against a
  median SAM-H error of 4.1 px, with a median of *137 RANSAC inliers*
  (quartiles 58/137/314). 97% are noise-robust (candidate gap #sym.gt.eq 5
  px), and they spread over 137 sequences --- a broad phenomenon, not a few
  bad clips.
- 1,526 (32%) are the reverse direction (SAM-H selected while LoMA was
  fine; median 23 inliers --- the gate was too pessimistic).

The inlier count cannot realize this headroom: sweeping the switch threshold
$X$ (use SAM-H when inliers $< X$) gives #P15 = 87.38, *87.68*, 87.59,
87.29, 86.95, 86.58 for $X$ = 20, 35, 50, 80, 120, 200 --- the deployed
threshold 35 is already optimal among inlier rules, and every move that
buys #P5 pays in #P15.

Two further cells matter downstream: the strictly-static population that
over-segmentation correction could ever swap (`spill` #sym.times
SAM-H-selected #sym.times LoMA-fine) is *142 frames*; and the segmentation
branch wastes *4,784 miss-frames in the `ok` mask state* --- the mask agrees
with the target, the Hough quad is still off by more than 15 px. The latter
is addressed in @sec:align.

#takeaway[
  *Takeaway.* The dominant recoverable error is a *selection* problem:
  +3.55#pp #P15 sits in already-computed candidates, two thirds of it on
  high-inlier catastrophic LoMA failures that the inlier gate cannot see.
  Any improvement must come from a quality signal other than the inlier
  count.
]

= Can simple online rules realize the selection headroom?

== Research question and method

Is there a per-frame *online* rule --- computable from signals available at
tracking time --- that approaches the oracle selector? The substrate is a
"dry-run" tracker run that behaves exactly like the baseline but records,
every frame and independently of any gating: the candidate homographies, the
inlier count, and mask-containment telemetry. From these we form:

- `n_inliers` --- LoMA RANSAC inlier count;
- `disagreement` --- mean corner distance (px) between the LoMA- and
  SAM-H-warped initial corners (two-branch consistency);
- `spill_loma`, `excess_loma`, `area_log_ratio` --- one-sided containment
  statistics between the SAM mask and the LoMA-warped initial mask;
- `loma_speed`, `samh_speed`, `speed_gap` --- per-branch temporal smoothness:
  mean corner displacement of each branch's warp between consecutive frames,
  and their difference;
- shake-normalized variants (`*_speed_exc`): each branch's speed minus its
  own lagged 15-frame rolling median (the cameras shake, so absolute speed
  is baseline-shifted).

Rule families: single-feature thresholds added on top of the baseline switch
("also switch to SAM-H when feature $> tau$"), AND-pairs of the two best
features, and a two-sided *smoothness arbitration* ("when the branches
disagree by more than $tau$ px, ignore the inlier gate and trust the
temporally smoother branch" --- the only family that can also move
SAM-H-selected frames back to LoMA). Thresholds are selected on a
sequence-level train half (150/150 split) by #P15 and evaluated on the held-out
half; feature AUCs use frames where one candidate beats the other by more
than 3 px (above the annotation-noise floor).

== Results

Feature separability (train AUC for "SAM-H is the better candidate";
oriented so that high = switch): `n_inliers` 0.726, `speed_gap` 0.703,
`spill_loma` 0.686, `loma_speed` 0.647, `disagreement` 0.601,
`excess_loma` 0.506. The shake-normalized speeds score *worse* than the raw
ones (0.567 / 0.574 / 0.500): catastrophic episodes last many frames, so a
branch's own rolling median absorbs the breakdown level and erases the
signal.

Rule frontier on the held-out half (baseline 87.92, oracle 90.93 #P15). The held-out half has 7,998
misses, of which 1,363 are rescuable by switching to SAM-H and 632 by
switching to LoMA:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*rule*], [*#P15 test*], [*#P5 test*], [*changed frames*]),
    [baseline (inlier switch)], [87.92], [66.22], [0%],
    [oracle (best stored candidate)], [90.93], [75.87], [---],
    [`speed_gap` #sym.gt 3.5 px AND `loma_speed` #sym.gt 10.4 px (best found)],
      [88.12], [66.73], [3.1%],
    [smoothness arbitration (both variants)], [#sym.lt.eq 87.88], [---], [0.1--0.2%],
  ),
  caption: [Selector rule search, held-out results. The full frontier
  (18 rules) is omitted; nothing exceeds +0.20#pp.],
)

The best rule recovers *+0.20 of the +3.0#pp* held-out ceiling. The
arbitration family self-neutered: its optimal disagreement threshold ran to
#sym.approx 700 px, i.e. "never fire" --- bidirectional switching loses more
on false switches than it gains on rescues. The structural reason simple
rules fail: the rescuable population is 2.4% of frames, the false-switch
cost is asymmetric (SAM-H's error tail is heavy), and every available signal
_ranks_ frames (AUC #sym.lt.eq 0.73) without _separating_ them.

#takeaway[
  *Takeaway.* Simple threshold rules over every currently-stored online
  signal are exhausted at +0.2#pp of a +3.0#pp ceiling. Realizing the
  selection headroom requires a genuinely new per-branch quality signal.
  Two candidates are queued: the boundary residual of the mask aligner of
  @sec:align (a direct segmentation-branch quality measure, free once the
  aligner is deployed), and LoMA introspection (spatial coverage of the
  inlier set --- motivated by the suspicion that the high-inlier failures
  are locally-consistent matches extrapolating badly to the corners; a
  visual gallery tool for these frames is in place to name the failure mode
  before designing the signal).
]

= Can direct mask alignment replace the Hough quad? <sec:align>

== Research question and method <sec:basin-tau>

The segmentation branch's weakest link is mask → homography:
4,784 #P15 misses occur with a _correct_ mask (@sec:budget --- Hough line
extraction, intersection selection, or corner assignment fails). Direct
alignment is the alternative: warp the initial target mask by a candidate
homography and optimize the homography so the warped boundary lands on the
current mask's boundary. Historically this failed when run naively; the
re-framed design is (i) *4-point parameterization* (optimize the four image
positions of the template corners, never raw 3×3 coefficients),
(ii) damped Gauss--Newton on a Huber loss over the target mask's *signed
distance field*, sampled along the warped template boundary, and (iii)
initialization from the last-frame homography, so only the residual
inter-frame motion must be recovered.

The feasibility question: *is the convergence basin wider than the
initialization error the tracker actually has?* On 2,324 PTi frames (every
60th), the GT homography was perturbed by structured, attributable families
--- translation (2--80 px), rotation (5--180#sym.degree, deliberately
reaching symmetry poses), scale, shear, perspective tilt (2--40%), iid
corner noise ($sigma$ = 1--20 px), and a realistic mix --- and the aligner
had to recover it against the *real SAM3 mask* of that frame. Outcomes:
_correct_ (final corner error #sym.lt.eq $tau$), _symmetric capture_
(correct only up to a cyclic corner relabeling --- the symmetry trap),
_diverged_.

A methodological note mirroring @sec:e5-flip: with the original $tau$ = 5
px, a flat #sym.approx 28% "failure floor" appeared at even 1--2 px
perturbations --- suspicious, because 5 px is below the combined GT
annotation noise and SAM boundary offset. Re-thresholding the stored
per-trial errors at $tau$ = 15 px (no recomputation needed) collapsed the
floor to 7.3%: the "failures" were the metric measuring annotation noise,
not divergence.

== Results ($tau$ = 15 px)

Masks split into *clean* (SAM-vs-GT IoU #sym.gt.eq 0.7; 2,219 frames, 95.5%)
and *degraded* (105 frames). On clean masks:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, left),
    table.header([*family*], [*magnitude*], [*correct %*], [*note*]),
    [any family], [small (1--10 px / #sym.lt.eq 10%)], [90--93], [median landing error 3.1--3.2 px],
    [translation], [20 / 40 / 80 px], [87 / 70 / 42], [graceful decay],
    [rotation], [20#sym.degree], [86], [],
    [rotation], [45#sym.degree], [19], [+28% symmetric captures],
    [rotation], [90#sym.degree / 180#sym.degree], [0], [68% / 92% land on a wrong cyclic pose],
    [scale / shear / tilt], [40%], [84 / 87 / 85], [],
    [corner noise], [$sigma$ = 20 px], [83], [],
    [realistic mix], [×2 / ×4], [88 / 69], [],
  ),
  caption: [Convergence on clean SAM masks, $tau$ = 15 px. Degraded masks
  (IoU #sym.lt 0.7) essentially never align (#sym.lt.eq 13% at any
  magnitude, median final error 55--80 px): the aligner fixes wrong quads,
  not wrong masks.],
)

The basin comfortably contains tracking-scale initialization error (the
LoMA estimate has median error 3.5 px; inter-frame motion at last-frame
initialization is far below 20 px). Rotations beyond 45#sym.degree land
reliably on a *wrong cyclic pose* rather than diverging --- so orientation
recovery must stay with the keypoint branch (or a multi-hypothesis scheme
over the four cyclic starts), exactly as in the current architecture.

== Prototype

The aligner is implemented as a drop-in replacement for the Hough step
behind a config switch, with the last-frame homography as initialization,
an acceptance gate (median boundary residual #sym.lt.eq 6 px + homography
validity), and Hough as rejection fallback. The integration point is the
single mask→H verb shared by all algorithm variants, so the
independent SAM-H sub-tracker inherits it automatically. Unit tests cover
translation/rotation recovery and clean failure on empty masks.

One identifiability caveat surfaced by testing: for template shapes with a
homography stabilizer (any ellipse --- a conic --- admits a 3-parameter
family of homographies mapping it onto itself), the pose is recoverable only
up to that stabilizer; mask agreement is the only guarantee. Benchmark quad
targets are fully pinned; for near-symmetric arbitrary shapes, slow gauge
drift along the stabilizer is possible and uncorrected.

#takeaway[
  *Takeaway.* Direct mask alignment is validated as the segmentation
  branch's mask→H estimator: 93% convergence on clean masks
  from far beyond tracking-scale perturbations, landing at #sym.approx 3 px
  (vs. the Hough chain's 4,784 correct-mask misses). Full PTi + POT runs of
  the prototype are in progress; expected ceiling #sym.approx +3#pp #P15 on
  PTi. Its per-frame boundary residual doubles as the missing
  segmentation-branch quality signal for the selection problem.
]

= Synthesis and outlook

#figure(
  table(
    columns: (auto, auto, auto, 1fr),
    align: (left, right, left, left),
    table.header([*lever*], [*#P15 ceiling*], [*status*], [*binding constraint*]),
    [online mask correction], [#sym.approx +1#pp], [closed (off)], [narrow causal path; no content source when needed],
    [per-frame candidate selection], [+3.6#pp], [open], [no separating online quality signal yet],
    [mask-alignment SAM-H], [#sym.approx +3#pp], [prototype running], [degraded masks out of reach by design],
    [continuous mask coaching], [+1.6#pp], [parked], [needs a trustworthy content source (possibly the aligner)],
  ),
  caption: [The recoverable-error map after the campaign (PTi; ceilings are
  oracle-grade, overlapping, not additive).],
)

The three findings compose into one picture: the tracker's losses are (i)
LoMA breaking catastrophically while *appearing* healthy --- a selection
problem awaiting a real quality signal; (ii) SAM-H wasting correct masks on
a brittle quad extractor --- addressed by the alignment prototype; (iii)
segmentation breaking under the same conditions as LoMA --- which is why
correction-style repairs stay capped. Immediate next steps: evaluate the
alignment prototype (PTi #P15 vs. 87.7, POT guard 98.5); inspect the
rendered gallery of high-inlier LoMA failures and name the failure mode
(clustered inliers vs. repeated-texture distractor vs. plausible-but-wrong
quad --- each implies a different fix); then revisit selection with the
aligner residual as a feature.

#v(4mm)
#line(length: 100%, stroke: 0.4pt + luma(150))
#text(size: 8.5pt)[
  *Reproduction map.* Oracle grid: `configs/tracker/loma_samh_correct_decoupled_*`
  driven by `experiments/run_correction_grid.sh`; event analysis
  `experiments/correction_events.py`. Error budget:
  `experiments/correction_headroom.py`. Selector search:
  `experiments/selector_study.py` (substrate: the dry-run configuration).
  Basin study: `experiments/mask_alignment_basin.py` (`report --tau 15`).
  Aligner + prototype: `flatsam2/mask_align.py`,
  `configs/tracker/loma_samh_decoupled_alignh.py`, `tests/test_mask_align.py`.
  Failure gallery: `experiments/loma_breakdown_gallery.py`. Running notes:
  `docs/correction_research.md` (plain-language summary in its §0).
]
