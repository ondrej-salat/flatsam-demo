"""Display aliases for tracker config directory names used in reports."""

aliases: dict[str, str] = {
    "SAM_H": "SAM2_H (pre-refactor run)",
    "hough_symmetry_only": "SAM3_H",
    "woft_replica": "WOFT",
    "woftsam_samfallback_nl": "WOFTSAM2",
    "woftsam_samfallback_nl_nn": "WOFTSAM2_failByBoundaryNN",
    "flow_hough_symmetry_nn_cascade5": "ROFTSAM3_failByBoundaryNN_c5",
    "flow_hough_symmetry_nn": "ROFTSAM3_failByBoundaryNN",
    "flow_hough_symmetry_nnpts": "ROFTSAM3_failBySimpleBoundaryNN",
    "flowsearaft_hough_symmetry_nn": "ROFTSAM3_SeaRAFT_failByBoundaryNN",
    # "flow_hough_symmetry_nn": "Flow + Hough + Sym + NN",
    # "default": "Baseline",
    # --- current trackers and their inputs -------------------------------
    "sam_H": "SAM2_H",
    "loma_r_allobj_inflate10": "LoMa_r (keypoints only)",
    "loma_samh_decoupled_rerun": "LoMa+SAM3H switch (thr35)",
    "corner_fusion_register_final": "LoMa+SAM3H fusion (quad-free)",
    "corner_fusion_register_kp": "fusion+register+kp",
    "corner_fusion_register": "fusion+register",
    "corner_fusion_register_noapp": "fusion+register+kp, no texture",
    "corner_fusion_alignh_calibrated": "fusion+alignH (calibrated)",
    "oracle_LMr_allobj_infl10_X_SAM3H": "oracle branch choice",
    # legacy trackers re-run from flatsam2 (experiments/run_legacy.py)
    "woftsam_samfallback_nl_rerun": "WOFTSAM2 (re-run here)",
    "WACV2026-SAM-H_rerun": "SAM2_H (re-run here)",
}
