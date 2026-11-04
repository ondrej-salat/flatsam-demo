"""View definitions for the collect_results standalone HTML table report.

The first block of views ("story") is meant to be read in order: it walks
through what happened between the 35-inlier switch baseline and the current
quad-free tracker, and then through what the four new datasets said about both.
Each story view carries the takeaway in its note, so the table is the evidence
and the note is the claim. The older per-topic views follow unchanged.

Generate with all six datasets, otherwise the story views show empty columns::

    python collect_results.py --datasets pot pti poic tmt ucsb mpot \\
        --table-report-out report.html --views-config configs/reports/example_views.py

Metric note: this report shows **frame-weighted** ``overall_p@k`` (every
annotated frame counts once). The per-sequence means quoted in docs/journal.md
and docs/state-of-the-tracker.md weight each sequence equally instead, so the
two differ most where sequence lengths are uneven (UCSB, MPOT-3K).
"""

from flatsam2.results_table_report import ReportSpec, ViewSpec, desc


# Column groups. Dataset short labels come from flatsam2/registry.py.
_ALL_SIX = [
    "method",
    "POT p@5", "POT p@15",
    "PTi p@5", "PTi p@15",
    "POIC p@5", "POIC p@15",
    "TMT p@5", "TMT p@15",
    "UCSB p@5", "UCSB p@15",
    "MPOT p@5", "MPOT p@15",
    "config",
    "failed",
]
_POT_PTI = [
    "method", "POT p@5", "POT p@15", "PTi p@5", "PTi p@15", "config", "failed",
]

# The four configs that have been run on every dataset.
_SIX_DATASET_CONFIGS = [
    "woftsam_samfallback_nl",
    "sam_H",
    "loma_samh_decoupled_rerun",
    "corner_fusion_register_final",
]

# The publication line-up: the two previous-paper baselines (SAM2-based), the
# same mask expert on OUR segmenter, our flagship, and the paused quad-free
# branch as a teaser.
_FLAGSHIP = [
    "woftsam_samfallback_nl",
    "sam_H",
    "hough_symmetry_only",
    "woftsam3_samfallback_nl",
    "loma_samh_matchnaming",
    "corner_fusion_naming",
]

# Runs that are partial, throwaway or pre-history — never interesting in a view.
_JUNK = [
    "smoke_legacy_woftsam",
    "2025-03-15",
    "2026-03-16",
    "debug",
    "bad",
]


report = ReportSpec(
    title="FlatSAM2 evaluation report",
    subtitle=(
        "Frame-weighted overall p@k. Story views first, in reading order; "
        "per-topic views after."
    ),
    views=[
        # ------------------------------------------------------------------
        # STORY — read these in order.
        # ------------------------------------------------------------------
        ViewSpec(
            key="story-00-flagship",
            title="⓪ Headline — and is it just SAM3?",
            note=(
                "TAKEAWAY: the obvious reviewer objection is that we win by using a "
                "better segmenter (SAM3) than the published baselines (SAM2). Both "
                "baselines have a SAM3 control, and the objection does not survive them. "
                "PTi per-sequence mean p@15 -- SAM-H 80.34 (SAM2) -> hough_symmetry_only "
                "83.05 (SAM3, same mask expert, no keypoints, no naming): the segmenter "
                "is worth +2.71, and our method adds +5.13 on top to reach 88.18. "
                "WOFTSAM 77.37 (SAM2) -> woftsam3_samfallback_nl 77.15 (SAM3): the "
                "segmenter is worth NOTHING to the flow-based baseline, -0.22. So the "
                "upgrade helps only the method that consumes masks directly, and it "
                "explains at most a third of our margin. POT decomposes the same way: "
                "88.38 -> 89.37 -> 97.66, segmenter +0.99 (11%), method +8.29 (89%). "
                "OPEN: woftsam3 on POT is running (11378376) -- until it lands, the POT "
                "column of the WOFTSAM control is the one gap, and POT is where our "
                "margin over WOFTSAM is thinnest (+0.91). corner_fusion_naming is a "
                "TEASER only: PlanarTrack-only, mixed A100/V100 hardware, not quotable."
            ),
            include_configs=_FLAGSHIP,
            row_order=_FLAGSHIP,
            columns=_POT_PTI,
        ),
        ViewSpec(
            key="story-01-headline",
            title="① Where we stand — six datasets",
            note=(
                "TAKEAWAY: against published WOFTSAM2 the two current trackers win on "
                "POT (+1.0 / +0.1), PTi (+10.1 / +10.2), TMT (+5.4 / +4.0) and MPOT-3K "
                "(+1.8 / +1.8) p@15, tie on POIC, and LOSE on UCSB (-4.4 / -6.3). The "
                "mask expert alone (SAM2_H) is nowhere. Dashes = never run on that "
                "dataset; only these four configs have the full six-dataset grid, and "
                "'failed' is 0/0/0/0/0/0 for all of them."
            ),
            include_configs=_SIX_DATASET_CONFIGS,
            row_order=[
                "woftsam_samfallback_nl",
                "sam_H",
                "loma_samh_decoupled_rerun",
                "corner_fusion_register_final",
            ],
            columns=_ALL_SIX,
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-02-cross-dataset",
            title="② The four new benchmarks",
            note=(
                "TAKEAWAY: POT/PT were never enough to separate these trackers. On the "
                "new data the mask expert alone (SAM2_H) collapses — TMT 54.3, UCSB 54.3 "
                "p@15 against ~90 for anything with a flow/keypoint branch — while the "
                "switch and the fusion stay within 1-2 pp of each other."
            ),
            include_configs=_SIX_DATASET_CONFIGS,
            row_order=_SIX_DATASET_CONFIGS,
            columns=[
                "method",
                "POIC p@5", "POIC p@15",
                "TMT p@5", "TMT p@15",
                "UCSB p@5", "UCSB p@15",
                "MPOT p@5", "MPOT p@15",
                "config",
            ],
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-03-ucsb-regression",
            title="③ UCSB — the one place we went backwards",
            note=(
                "TAKEAWAY: on UCSB the 2024-era WOFTSAM2 still wins, and the quad-free "
                "tracker is the worst of the three. UCSB sequences are short close-ups "
                "(<=100 frames, median 50), so there is little for flow to drift from and "
                "the newer machinery only adds ways to be wrong. This is the open item."
            ),
            include_configs=_SIX_DATASET_CONFIGS,
            row_order=[
                "woftsam_samfallback_nl",
                "loma_samh_decoupled_rerun",
                "corner_fusion_register_final",
                "sam_H",
            ],
            columns=[
                "method", "UCSB p@5", "UCSB p@15",
                "POT p@15", "PTi p@15", "TMT p@15", "MPOT p@15", "config",
            ],
            diff_mode=True,
            baseline_config="woftsam_samfallback_nl",
        ),
        ViewSpec(
            key="story-04-two-experts",
            title="④ Two experts, and why they are combined",
            note=(
                "TAKEAWAY: neither expert is enough. Keypoints alone (LoMa_r) are strong on "
                "POT and weak on PTi; the mask expert (SAM3_H) is the reverse. Choosing "
                "between them per frame by inlier count gets most of the way to the oracle "
                "that picks the better branch with hindsight — the remaining PTi gap "
                "(87.7 vs 91.2 p@15) is the headroom the fusion work was chasing."
            ),
            include_configs=[
                "woft_replica",
                "loma_r_allobj_inflate10",
                "sam_H",
                "hough_symmetry_only",
                "woftsam_samfallback_nl",
                "loma_samh_decoupled_rerun",
                "oracle_LMr_allobj_infl10_X_SAM3H",
            ],
            row_order=[
                "woft_replica",
                "loma_r_allobj_inflate10",
                "sam_H",
                "hough_symmetry_only",
                "woftsam_samfallback_nl",
                "loma_samh_decoupled_rerun",
                "oracle_LMr_allobj_infl10_X_SAM3H",
            ],
            columns=_POT_PTI,
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-05-quadfree-path",
            title="⑤ The quad-free path — one regression, two fixes",
            note=(
                "TAKEAWAY: replacing the Hough quad with direct mask alignment first LOST "
                "10 pp PTi (alignh_pure), and calibration only recovered part of it. What "
                "actually worked was propose/refine/verify registration (+5.7 pp over "
                "alignh_calibrated), then the keypoint proposal. Read top to bottom as the "
                "chronology; the last row is the current reference config."
            ),
            include_configs=[
                "loma_samh_decoupled_rerun",
                "loma_samh_decoupled_alignh_pure",
                "loma_samh_decoupled_alignh",
                "corner_fusion_alignh_calibrated",
                "corner_fusion_alignh_redetect",
                "corner_fusion_register",
                "corner_fusion_register_kp",
                "corner_fusion_register_final",
            ],
            row_order=[
                "loma_samh_decoupled_rerun",
                "loma_samh_decoupled_alignh_pure",
                "loma_samh_decoupled_alignh",
                "corner_fusion_alignh_calibrated",
                "corner_fusion_alignh_redetect",
                "corner_fusion_register",
                "corner_fusion_register_kp",
                "corner_fusion_register_final",
            ],
            columns=_POT_PTI,
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-06-texture-names-corners",
            title="⑥ Texture is what names the corners",
            note=(
                "TAKEAWAY: turn the texture term off in the verifier and PTi p@15 drops "
                "5.5 pp (87.2 -> 81.7) while the object is still found — the loss is corner "
                "LABELLING, not detection. A square-ish planar silhouette cannot pin its own "
                "pose; appearance breaks the 4-fold tie."
            ),
            include_configs=[
                "corner_fusion_register_kp",
                "corner_fusion_register_noapp",
                "corner_fusion_register_final",
            ],
            row_order=[
                "corner_fusion_register_kp",
                "corner_fusion_register_noapp",
                "corner_fusion_register_final",
            ],
            columns=_POT_PTI,
            diff_mode=True,
            baseline_config="corner_fusion_register_kp",
        ),
        ViewSpec(
            key="story-07-mask-branch-alone",
            title="⑦ The mask branch on its own",
            note=(
                "TAKEAWAY: every way of turning a silhouette into a quad is far behind a "
                "tracker with a flow/keypoint branch, and the ranking among them is small "
                "print. Worth keeping as the reason the mask branch is a SECOND opinion, "
                "never the tracker."
            ),
            include_configs=[
                "sam_H",
                "hough_symmetry_only",
                "SAMH_ssd",
                "SAMH_motion_only",
                "mask_to_quad_rdp",
                "mask_to_quad_minarea",
            ],
            row_order=[
                "sam_H",
                "hough_symmetry_only",
                "SAMH_ssd",
                "SAMH_motion_only",
                "mask_to_quad_rdp",
                "mask_to_quad_minarea",
            ],
            columns=_POT_PTI,
            diff_mode=True,
            baseline_config="sam_H",
        ),
        ViewSpec(
            key="story-08-correction-null",
            title="⑧ Negative result — SAM3 mask correction did nothing",
            note=(
                "TAKEAWAY: a whole family of runs that detect and repair over-segmented "
                "SAM3 masks moves POT/PTi by less than +-0.5 pp, and even the ORACLE "
                "correction variants (perfect trigger / perfect choice) do not separate "
                "from the baseline. Mask correction is not where the remaining error is."
            ),
            include_configs=[
                "loma_samh_decoupled_rerun",
                "loma_samh_correct",
                "loma_samh_correct_decoupled",
                "loma_samh_correct_decoupled_inl35",
                "loma_samh_correct_decoupled_excessdepth",
                "loma_samh_correct_decoupled_relaxed",
                "loma_samh_correct_decoupled_oracleT",
                "loma_samh_correct_decoupled_oracleC",
                "loma_samh_correct_decoupled_oracleTC",
            ],
            row_order=[
                "loma_samh_decoupled_rerun",
                "loma_samh_correct",
                "loma_samh_correct_decoupled",
                "loma_samh_correct_decoupled_inl35",
                "loma_samh_correct_decoupled_excessdepth",
                "loma_samh_correct_decoupled_relaxed",
                "loma_samh_correct_decoupled_oracleT",
                "loma_samh_correct_decoupled_oracleC",
                "loma_samh_correct_decoupled_oracleTC",
            ],
            columns=_POT_PTI,
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-09-noise-floor",
            title="⑨ How big is a real difference?",
            note=(
                "TAKEAWAY: the first five rows are the SAME switch tracker under re-runs "
                "and supposedly-neutral knobs; they spread 0.1 pp on POT and 0.2 pp on "
                "PTi. The legacy re-runs give the upper bound: WOFTSAM2 re-run from "
                "flatsam2 moves 0.3 pp POT / 0.7 pp PTi p@15 (RANSAC + GPU nondeterminism), "
                "while SAM2_H is deterministic and reproduces its stored run exactly. "
                "So treat <0.7 pp on PTi and <0.3 pp on POT as noise, not as a result."
            ),
            include_configs=[
                "loma_samh_decoupled_rerun",
                "loma_samh_decoupled_rerun2",
                "loma_samh_decoupled",
                "loma_samh_decoupled_candunion",
                "loma_samh_decoupled_rerun_fullptr",
                "woftsam_samfallback_nl",
                "woftsam_samfallback_nl_rerun",
                "sam_H",
                "WACV2026-SAM-H_rerun",
            ],
            row_order=[
                "loma_samh_decoupled_rerun",
                "loma_samh_decoupled_rerun2",
                "loma_samh_decoupled",
                "loma_samh_decoupled_candunion",
                "loma_samh_decoupled_rerun_fullptr",
                "woftsam_samfallback_nl",
                "woftsam_samfallback_nl_rerun",
                "sam_H",
                "WACV2026-SAM-H_rerun",
            ],
            columns=_POT_PTI,
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-10-which-to-publish",
            title="⑩ Which tracker to publish?",
            note=(
                "TAKEAWAY: on robustness (p@15) the switch is ahead or level on all six — "
                "POT +0.9, TMT +1.4, UCSB +1.9, POIC and MPOT dead level, PTi -0.1 (noise). "
                "The quad-free tracker's real edge is PRECISION: PTi p@5 +3.4, MPOT +0.9, "
                "POIC +0.4, against POT -0.8, TMT -2.5, UCSB -0.4. So it is not 'better by "
                "0.02' — it is a different trade, and today it buys tight-threshold accuracy "
                "on PTi at the cost of TMT/POT. The quad-free story is the more publishable "
                "one (no quad, no symmetry heuristic, corners carry covariances), but on "
                "these numbers it does not yet dominate; UCSB and TMT are what it must fix. "
                "Toggle p@5/p@15 columns to see both halves."
            ),
            include_configs=_SIX_DATASET_CONFIGS,
            row_order=[
                "loma_samh_decoupled_rerun",
                "corner_fusion_register_final",
                "woftsam_samfallback_nl",
                "sam_H",
            ],
            columns=_ALL_SIX,
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="story-11-everything-six",
            title="⑪ Everything, all datasets",
            note=(
                "One big table: every run, every dataset, sorted by PTi p@15. Use the "
                "column and diff controls to dig; the story views above are the guided path."
            ),
            exclude_configs=_JUNK,
            columns=_ALL_SIX,
            sort_by=[desc("PTi p@15"), desc("POT p@15")],
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        # ------------------------------------------------------------------
        # Per-topic views (older, kept as-is).
        # ------------------------------------------------------------------
        ViewSpec(
            key="loma-sam",
            title="LoMa-SAM interaction",
            note="loma_samh (a real tracker) - the online version of LMr_allobj... (post-hoc ensemble)",
            include_configs=[
                "LMr_allobj_infl10_X_SAM3H_thr35",
                "loma_samh",
                "loma_samh_correct",
                "loma_samh_correct_decoupled",
                "loma_samh_correct_decoupled_excessdepth",
                "loma_samh_correct_decoupled_inl35",
                "loma_samh_correct_decoupled_relaxed",
                "loma_samh_decoupled",
                "loma_samh_decoupled_rerun",
                "loma_samh_decoupled_rerun2",
                "loma_samh_decoupled_candunion",
                "hough_symmetry_only",
            ],
            row_order=[
                "LMr_allobj_infl10_X_SAM3H_thr35",
                "loma_samh",
                "loma_samh_correct",
                "loma_samh_correct_decoupled",
                "loma_samh_correct_decoupled_excessdepth",
                "loma_samh_correct_decoupled_inl35",
                "loma_samh_correct_decoupled_relaxed",
                "loma_samh_decoupled",
                "loma_samh_decoupled_rerun",
                "loma_samh_decoupled_rerun2",
                "loma_samh_decoupled_candunion",
                "hough_symmetry_only",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="loma_samh_decoupled_rerun",
        ),
        ViewSpec(
            key="rebuttal mask-to-quad baselines",
            title="maks-to-quad baselines",
            note="symmetry disambiguation by minimal motion w.r.t. previous frame",
            include_configs=[
                "sam_H",
                "mask_to_quad_rdp",
                "SAMH_motion_only",
                "mask_to_quad_minarea",
            ],
            row_order=[
                "sam_H",
                "SAMH_motion_only",
                "mask_to_quad_minarea",
                "mask_to_quad_rdp",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="SAMH_motion_only",
        ),
        ViewSpec(
            key="plainSAM",
            title="WOFTSAM with plain SAM",
            note="",
            include_configs=[
                "ECCV_rebuttal_woftsam_samfallback_nl",
                "woftsam_samfallback_nl",
                "woftsam_plainSAM_samfallback_nl",
            ],
            row_order=[
                "ECCV_rebuttal_woftsam_samfallback_nl",
                "woftsam_samfallback_nl",
                "woftsam_plainSAM_samfallback_nl",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="woftsam2_samfallback_nl",
        ),
        ViewSpec(
            key="baseline-disambiguation",
            title="baseline disambiguation",
            note="",
            include_configs=[
                "sam_H",
                "SAMH_ssd",
                "SAMH_motion_only",
            ],
            row_order=[
                "sam_H",
                "SAMH_ssd",
                "SAMH_motion_only",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="sam_H",
        ),
        ViewSpec(
            key="hough ablation",
            title="Hough parameter ablations",
            note="",
            include_configs=[
                "SAMH_hough_point_to_line_dist_thr_px_25",
                "SAMH_hough_point_to_line_dist_thr_px_10",
                "SAMH_hough_voting_degree_range_20",
                "SAMH_hough_voting_degree_range_5",
                "SAMH_hough_point_to_line_angle_thr_deg_10",
                "SAMH_hough_point_to_line_angle_thr_deg_25",
                "sam_H",
                "SAMH_hough_gaussian_sigma_2",
                "SAMH_hough_gaussian_sigma_8",
            ],
            row_order=[
                "SAMH_hough_point_to_line_dist_thr_px_25",
                "SAMH_hough_point_to_line_dist_thr_px_10",
                "SAMH_hough_voting_degree_range_20",
                "SAMH_hough_voting_degree_range_5",
                "SAMH_hough_point_to_line_angle_thr_deg_10",
                "SAMH_hough_point_to_line_angle_thr_deg_25",
                "sam_H",
                "SAMH_hough_gaussian_sigma_2",
                "SAMH_hough_gaussian_sigma_8",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="sam_H",
        ),
        ViewSpec(
            key="rebuttal all",
            title="rebuttal all",
            note="All methods sorted by PTi p@15.",
            include_configs=[
                "SAMH_hough_point_to_line_dist_thr_px_25",
                "SAMH_hough_point_to_line_dist_thr_px_10",
                "SAMH_hough_voting_degree_range_20",
                "SAMH_hough_voting_degree_range_5",
                "SAMH_hough_point_to_line_angle_thr_deg_10",
                "SAMH_hough_point_to_line_angle_thr_deg_25",
                "sam_H",
                "SAMH_ssd",
                "SAMH_motion_only",
                "SAMH_hough_gaussian_sigma_2",
                "SAMH_hough_gaussian_sigma_8",
                "woftsam2_samfallback_nl",
                "woftsam_samfallback_nl",
                "woftsam_plainSAM_samfallback_nl",
                "mask_to_quad_rdp",
                "mask_to_quad_minarea",
            ],
            row_order=[
                "SAMH_hough_point_to_line_dist_thr_px_25",
                "SAMH_hough_point_to_line_dist_thr_px_10",
                "SAMH_hough_voting_degree_range_20",
                "SAMH_hough_voting_degree_range_5",
                "SAMH_hough_point_to_line_angle_thr_deg_10",
                "SAMH_hough_point_to_line_angle_thr_deg_25",
                "sam_H",
                "SAMH_ssd",
                "SAMH_motion_only",
                "SAMH_hough_gaussian_sigma_2",
                "SAMH_hough_gaussian_sigma_8",
                "woftsam2_samfallback_nl",
                "woftsam_samfallback_nl",
                "woftsam_plainSAM_samfallback_nl",
                "mask_to_quad_rdp",
                "mask_to_quad_minarea",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="sam_H",
        ),
        ViewSpec(
            key="mask_inflation",
            title="Mask inflation",
            note="Inflating mask by 10px (21px square dilation) helps.",
            include_configs=[
                "loma_r_allobj_inflate10",
                "loma_r_allobj",
            ],
            row_order=[
                "loma_r_allobj_inflate10",
                "loma_r_allobj",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "failed",
            ],
            diff_mode=True,
            baseline_config="loma_r_allobj",
        ),
        ViewSpec(
            key="inlier_threshold_noracle",
            title="Inlier threshold \"oracle\"",
            note="We can get quite close to oracle by thresholding #inliers",
            include_configs=[
                "oracle_LMr_allobj_infl10_X_SAM3H",
                "LMr_allobj_infl10_X_SAM3H_thr35",
                "hough_symmetry_only",
                "SAM_H",
                "loma_r_allobj_inflate10",
                "woftsam_samfallback_nl",
            ],
            row_order=[
                "oracle_LMr_allobj_infl10_X_SAM3H",
                "LMr_allobj_infl10_X_SAM3H_thr35",
                "hough_symmetry_only",
                "SAM_H",
                "loma_r_allobj_inflate10",
                "woftsam_samfallback_nl",
            ],
            rename_configs={
                "loma_r_allobj_inflate10": "LMr_allobj_infl10",
            },
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "failed",
            ],
            diff_mode=True,
            baseline_config="LMr_allobj_infl10_X_SAM3H_thr35",
        ),
        ViewSpec(
            key="smaller_object_budget",
            title="Reduced object budget",
            note="Having object keypoint budget 2000 instead of the full 4096 doesn't help.",
            include_configs=[
                "loma_r_allobj_inflate10",
                "loma_r_2000obj_infl10",
            ],
            row_order=[
                "loma_r_allobj_inflate10",
                "loma_r_2000obj_infl10",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "failed",
            ],
            diff_mode=True,
            baseline_config="loma_r_allobj_inflate10",
        ),
        ViewSpec(
            key="main_progress_overview",
            title="Progress overview",
            note="LoMa_r keypoints + simple failure detection + SAM3H fallback current SOTA",
            include_configs=[
                "woft_replica",
                "SAM_H",
                "woftsam_samfallback_nl",
                "oracle_woftsam_X_SAM2H",
                "hough_symmetry_only",
                "LMr_allobj_infl10_X_SAM3H_thr35",
                "oracle_LMr_allobj_infl10_X_SAM3H",
            ],
            row_order=[
                "woft_replica",
                "SAM_H",
                "woftsam_samfallback_nl",
                "oracle_woftsam_X_SAM2H",
                "hough_symmetry_only",
                "LMr_allobj_infl10_X_SAM3H_thr35",
                "oracle_LMr_allobj_infl10_X_SAM3H",
            ],
            columns=[
                "method",
                "POT p@5",
                "POT p@15",
                "PTi p@5",
                "PTi p@15",
                "config",
                "failed",
            ],
            diff_mode=True,
            baseline_config="LMr_allobj_infl10_X_SAM3H_thr35",
        ),
        ViewSpec(
            key="high_pti",
            title="High PTi performance",
            note="Configs with PTi p@15 >= 75.",
            filters=[
                lambda row: isinstance(row.get("PTi p@15"), (int, float))
                and float(row["PTi p@15"]) >= 75.0,
            ],
            sort_by=[desc("PTi p@15"), desc("POT p@15")],
        ),
        ViewSpec(
            key="focused_columns",
            title="Compact slide",
            note="Only headline metrics and config ID.",
            columns=["method", "POT p@15", "PTi p@15", "config"],
            sort_by=[desc("PTi p@15")],
        ),
        ViewSpec(
            key="everything",
            title="All the results",
            note="All methods sorted by PTi p@15.",
            sort_by=[desc("PTi p@15")],
            diff_mode=True,
            baseline_config="flow_hough_symmetry_nn",
        ),
    ],
)
