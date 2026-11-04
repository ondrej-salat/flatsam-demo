"""Analysis of FlatSAM2 tracking results for failure mode investigation.

Loads eval predictions + *_debug.npz + GT to compute oracle analysis,
score calibration, failure taxonomy, selection accuracy, error trajectories,
regret heatmap, and feature scatter plots.

Usage
-----
    python analyze_results.py pti --config configs/tracker/flow_hough_symmetry_nn.py
    python analyze_results.py pti --config flow_hough_symmetry_nn --output-dir ./analysis
    python analyze_results.py pti --config flow_hough_symmetry_nn --alias-config configs/aliases/cvpr.py
"""

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

from flatsam2.config_aliases import (
    get_config_display_name,
    load_config_aliases_from_file,
)
from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_load_gt,
    pt_load_gt,
    pti_load_gt,
    warp_corners,
)
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths


# Keys as they appear in *_debug.npz (prefix, e.g. 'flow' → flow_H, flow_rmse, …)
CAND_PATHS = ("flow", "mask", "redetection_flow")
CAND_COLORS = {
    "flow": "#2196F3",  # blue
    "mask": "#FF9800",  # orange
    "redetection_flow": "#4CAF50",  # green
    "last_H": "#F44336",  # red
    "": "#9E9E9E",  # grey (unknown / missing)
}
THRESHOLD = 15.0
CLIP_ERROR = 30.0  # NN scorer training clip value (n_clip); clip actual errors to this for calibration


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _make_load_gt_fn(dataset: str, initfix_dir: Optional[Path]):
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir required for pti loader")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def _load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    pred_flat = np.loadtxt(results_dir / f"{seq_name}.txt")
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(pred_flat, "f (c xy) -> f xy c", c=4, xy=2)


def _load_debug(results_dir: Path, seq_name: str) -> Optional[dict]:
    debug_file = results_dir / f"{seq_name}_debug.npz"
    if not debug_file.exists():
        return None
    data = np.load(debug_file, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _batch_warp_corners(Hs: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply N homographies to a single set of corners.

    Args:
        Hs: (N, 3, 3)
        corners: (2, 4)

    Returns:
        (N, 2, 4)
    """
    corners_hom = np.vstack([corners, np.ones((1, 4))])  # (3, 4)
    warped_hom = np.einsum("nij,jk->nik", Hs, corners_hom)  # (N, 3, 4)
    return warped_hom[:, :2] / warped_hom[:, 2:3]  # (N, 2, 4)


def _build_sequence_data(
    seq_name: str,
    gt: np.ndarray,
    valid: np.ndarray,
    pred: np.ndarray,
    debug: Optional[dict],
) -> dict:
    """Assemble per-sequence analysis tensors.

    Returns a dict with:
        name, errors, valid_frame_idx, has_debug
        cand_errors, oracle_errors, selected, pred_rmse, pred_sigma  (only when has_debug)
    """
    eval_mask = valid.copy()
    if len(eval_mask) > 0:
        eval_mask[0] = False  # skip init frame
    eval_idx = np.where(eval_mask)[0]  # (N,) actual frame indices

    if len(eval_idx) == 0:
        return {
            "name": seq_name,
            "errors": np.array([]),
            "valid_frame_idx": eval_idx,
            "has_debug": False,
        }

    errors = alignment_error(gt[eval_idx], pred[eval_idx])  # (N,)

    if debug is None:
        return {
            "name": seq_name,
            "errors": errors,
            "valid_frame_idx": eval_idx,
            "has_debug": False,
        }

    frame_to_row = {int(fi): i for i, fi in enumerate(debug["frame_idx"])}
    init_corners = gt[0]  # (2, 4)
    N = len(eval_idx)

    cand_errors: dict[str, np.ndarray] = {}
    pred_rmse: dict[str, np.ndarray] = {}
    pred_sigma: dict[str, np.ndarray] = {}

    for p in CAND_PATHS:
        # Keys in debug NPZ: flow_H, mask_h_H, flow_maskH_H; flow_rmse, flow_sigma, …
        H_all = debug[f"{p}_H"]  # (T, 3, 3)
        rmse_all = debug[f"{p}_H_rmse"]  # (T,)
        sigma_key = f"{p}_H_sigma"
        sigma_all = debug[sigma_key] if sigma_key in debug else None  # (T,) or None

        errs_p = np.full(N, np.nan)
        rmse_p = np.full(N, np.nan)
        sigma_p = np.full(N, np.nan)

        rows = np.array([frame_to_row.get(int(t), -1) for t in eval_idx])
        has_row = rows >= 0
        valid_rows = rows[has_row]

        # Candidate actual errors via stored homographies
        H_cands = H_all[valid_rows]  # (n_valid, 3, 3)
        has_H = ~np.isnan(H_cands[:, 0, 0])
        if has_H.any():
            pos = np.where(has_row)[0][has_H]
            warped = _batch_warp_corners(H_cands[has_H], init_corners)
            errs_p[pos] = alignment_error(gt[eval_idx[pos]], warped)

        # Predicted RMSE (what the NN scorer estimated)
        rmse_cands = rmse_all[valid_rows]
        finite_r = np.isfinite(rmse_cands)
        if finite_r.any():
            pos = np.where(has_row)[0][finite_r]
            rmse_p[pos] = rmse_cands[finite_r].astype(float)

        # Predicted uncertainty (aleatoric sigma from NN scorer)
        if sigma_all is not None:
            sigma_cands = sigma_all[valid_rows]
            finite_s = np.isfinite(sigma_cands)
            if finite_s.any():
                pos = np.where(has_row)[0][finite_s]
                sigma_p[pos] = sigma_cands[finite_s].astype(float)

        cand_errors[p] = errs_p
        pred_rmse[p] = rmse_p
        pred_sigma[p] = sigma_p

    stacked = np.stack([cand_errors[p] for p in CAND_PATHS])  # (3, N)
    oracle_errors = np.nanmin(stacked, axis=0)  # NaN where all NaN

    selected = np.full(N, "", dtype="U20")
    sel_all = debug["selected"]
    for j, t in enumerate(eval_idx):
        row = frame_to_row.get(int(t), -1)
        if row >= 0:
            selected[j] = str(sel_all[row])

    return {
        "name": seq_name,
        "errors": errors,
        "valid_frame_idx": eval_idx,
        "has_debug": True,
        "cand_errors": cand_errors,
        "oracle_errors": oracle_errors,
        "selected": selected,
        "pred_rmse": pred_rmse,
        "pred_sigma": pred_sigma,
    }


def load_all_sequences(
    dataset: str,
    base_dir: Path,
    results_dir: Path,
    sequences: list,
    initfix_dir: Optional[Path],
) -> list:
    load_gt_fn = _make_load_gt_fn(dataset, initfix_dir)
    seq_data, n_fail = [], 0

    for seq_name in sequences:
        try:
            gt, valid = load_gt_fn(base_dir, seq_name)
            pred = _load_predictions(results_dir, seq_name)
            debug = _load_debug(results_dir, seq_name)
            T = min(len(gt), len(pred))
            data = _build_sequence_data(seq_name, gt[:T], valid[:T], pred[:T], debug)
            if len(data["errors"]) > 0:
                seq_data.append(data)
        except Exception as e:
            print(f"  WARNING: {seq_name}: {e}", file=sys.stderr)
            n_fail += 1

    n_debug = sum(1 for s in seq_data if s["has_debug"])
    print(
        f"Loaded {len(seq_data)} sequences ({n_fail} failed, {n_debug} with debug NPZ)"
    )
    return seq_data


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _p_at_k(errors: np.ndarray, k: float = THRESHOLD) -> float:
    if len(errors) == 0:
        return float("nan")
    return 100.0 * float(np.mean(errors <= k))


def _seq_p15(s: dict) -> float:
    return _p_at_k(s["errors"])


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (numpy-only)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")

    def rank(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a)).astype(float)

    rx, ry = rank(x), rank(y)
    cov = float(np.mean((rx - rx.mean()) * (ry - ry.mean())))
    std = float(rx.std() * ry.std())
    return cov / std if std > 0 else float("nan")


def _hex_to_rgb01(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _seq_taxonomy(s: dict, threshold: float = THRESHOLD) -> dict:
    """Return mutually-exclusive frame-category fractions for one sequence.

    Categories (sum to 1.0):
        both_good  – flow ≤ t  AND  mask_h ≤ t
        flow_only  – flow ≤ t  AND  mask_h > t  (or mask not computed)
        mask_only  – flow > t  (or not computed)  AND  mask_h ≤ t
        neither    – neither ≤ t
    """
    fe = s["cand_errors"]["flow"]
    me = s["cand_errors"]["mask"]
    fg = np.isfinite(fe) & (fe <= threshold)
    mg = np.isfinite(me) & (me <= threshold)
    n = max(1, len(fe))
    return {
        "both_good": (fg & mg).sum() / n,
        "flow_only": (fg & ~mg).sum() / n,
        "mask_only": (~fg & mg).sum() / n,
        "neither": (~fg & ~mg).sum() / n,
    }


# ---------------------------------------------------------------------------
# A1: Oracle gap decomposition
# ---------------------------------------------------------------------------


def analysis_oracle_gap(seqs: list, threshold: float = THRESHOLD) -> tuple:
    debug_seqs = [s for s in seqs if s["has_debug"]]
    if not debug_seqs:
        return None, "A1: No debug data available."

    all_actual = np.concatenate([s["errors"] for s in debug_seqs])
    all_oracle = np.concatenate([s["oracle_errors"] for s in debug_seqs])

    has_oracle = np.isfinite(all_oracle)
    actual_ev = all_actual[has_oracle]
    oracle_ev = all_oracle[has_oracle]

    actual_p15_all = _p_at_k(all_actual)
    actual_p15_ev = _p_at_k(actual_ev)
    oracle_p15 = _p_at_k(oracle_ev)

    actual_fail = actual_ev >= threshold
    oracle_fail = oracle_ev >= threshold
    wrong_sel = actual_fail & ~oracle_fail  # oracle says possible, we blew it
    irreducible = actual_fail & oracle_fail  # no stored candidate could fix it

    # For each candidate: how often is it the unique contributor to oracle ≤ threshold?
    # = frames where oracle ≤ threshold AND this candidate ≤ threshold,
    #   as % of ALL oracle-good frames
    n_oracle_good = int((oracle_ev <= threshold).sum())
    cand_achieves = {}
    for p in CAND_PATHS:
        ce = np.concatenate([s["cand_errors"][p] for s in debug_seqs])
        cand_achieves[p] = int(
            (np.isfinite(ce[has_oracle]) & (ce[has_oracle] <= threshold)).sum()
        )

    n = len(actual_ev)
    lines = [
        "A1: Oracle Gap Decomposition",
        f"  Oracle defined (≥1 candidate H available): {has_oracle.sum()} / {len(all_actual)} "
        f"({100 * has_oracle.mean():.1f}%)",
        f"  Actual P@{threshold:.0f} (all frames):        {actual_p15_all:.1f}%",
        f"  Actual P@{threshold:.0f} (oracle subset):     {actual_p15_ev:.1f}%",
        f"  Oracle P@{threshold:.0f} (best candidate):    {oracle_p15:.1f}%",
        f"  Improvement gap (oracle - actual):            {oracle_p15 - actual_p15_ev:.1f} pp",
        f"",
        f"  Breakdown of oracle-defined frames (n={n}):",
        f"    Correct (actual ≤ thresh):                {n - int(actual_fail.sum()):5d} ({100 * (1 - actual_fail.mean()):.1f}%)",
        f"    Wrong selection (oracle good, we fail):   {wrong_sel.sum():5d} ({100 * wrong_sel.mean():.1f}%)",
        f"    Irreducible failure (all candidates bad): {irreducible.sum():5d} ({100 * irreducible.mean():.1f}%)",
        f"",
        f"  Oracle-achievable frames (≤ threshold) covered by each candidate:",
        f"  (a frame can be covered by multiple candidates)",
    ]
    for p, n_cov in cand_achieves.items():
        pct = 100 * n_cov / max(1, n_oracle_good)
        lines.append(
            f"    {p:12s}: {n_cov:5d} / {n_oracle_good} oracle-good frames ({pct:.1f}%)"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("A1: Oracle Gap Analysis", fontsize=14)

    # Left: P@k bar chart – actual vs oracle
    ax = axes[0]
    ks = [5, 10, 15]
    act_ps = [_p_at_k(actual_ev, k) for k in ks]
    ora_ps = [_p_at_k(oracle_ev, k) for k in ks]
    x, w = np.arange(len(ks)), 0.35
    bars_a = ax.bar(x - w / 2, act_ps, w, label="Actual", color="steelblue")
    bars_o = ax.bar(
        x + w / 2, ora_ps, w, label="Oracle (best available candidate)", color="coral"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"P@{k}" for k in ks])
    ax.set_ylabel("Precision (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title("Actual vs Oracle Precision")
    for bar, val in zip(list(bars_a) + list(bars_o), act_ps + ora_ps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Middle: failure decomposition pie
    ax = axes[1]
    n_good = n - int(actual_fail.sum())
    n_wrong = int(wrong_sel.sum())
    n_irred = int(irreducible.sum())
    ax.pie(
        [n_good, n_wrong, n_irred],
        labels=[
            f"Correct ({n_good:,})",
            f"Wrong selection ({n_wrong:,})",
            f"Irreducible ({n_irred:,})",
        ],
        colors=["#4CAF50", "#FF9800", "#F44336"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(f"Frame outcomes (threshold = {threshold:.0f} px)")

    # Right: how many oracle-good frames each candidate can cover
    ax = axes[2]
    labels_c = list(cand_achieves.keys())
    vals_c = [100 * cand_achieves[p] / max(1, n_oracle_good) for p in labels_c]
    ax.bar(labels_c, vals_c, color=[CAND_COLORS[p] for p in labels_c])
    ax.set_ylabel("% of oracle-good frames achievable")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Candidate coverage of oracle-good frames\n"
        f"(n_oracle_good = {n_oracle_good}; overlap allowed)"
    )
    for i, v in enumerate(vals_c):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig, "\n".join(lines)


# ---------------------------------------------------------------------------
# A2: Score calibration
# ---------------------------------------------------------------------------


def analysis_score_calibration(seqs: list, clip: float = CLIP_ERROR) -> tuple:
    """Compare predicted RMSE (from NN scorer) vs actual error from stored H.

    The NN scorer was trained to clip its targets at `clip` px, so predictions
    saturate at ~`clip`.  For a fair comparison we clip the actual errors to the
    same value before computing calibration statistics.
    """
    debug_seqs = [s for s in seqs if s["has_debug"]]
    if not debug_seqs:
        return None, "A2: No debug data."

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"A2: Score Calibration — Predicted RMSE vs Actual Error (clipped at {clip:.0f} px)",
        fontsize=13,
    )
    lines = [
        f"A2: Score Calibration (actual errors clipped to {clip:.0f} px to match NN training)",
        f"  Spearman r: rank correlation between predicted RMSE and clipped actual error.",
        f"  Reliability diagram: mean clipped actual error per decile of predicted RMSE.",
        f"  Perfect calibration = bars follow the y=x diagonal.",
    ]

    for col, p in enumerate(CAND_PATHS):
        pred_r = np.concatenate([s["pred_rmse"][p] for s in debug_seqs])
        cand_e = np.concatenate([s["cand_errors"][p] for s in debug_seqs])

        mask = np.isfinite(pred_r) & np.isfinite(cand_e)
        pr, ae = pred_r[mask], np.minimum(cand_e[mask], clip)  # clip actual error
        rho = _spearman_r(pr, ae)
        lines.append(f"  {p:12s}: n={mask.sum():5d}  Spearman r={rho:.3f}")

        # Top row: scatter
        ax = axes[0, col]
        n_plot = min(len(pr), 5000)
        idx_s = np.random.default_rng(0).choice(len(pr), n_plot, replace=False)
        ax.scatter(pr[idx_s], ae[idx_s], s=2, alpha=0.3, color=CAND_COLORS[p])
        ax.plot([0, clip], [0, clip], "k--", lw=1, label="perfect calibration (y=x)")
        ax.set_xlim(0, clip)
        ax.set_ylim(0, clip)
        ax.set_xlabel("Predicted RMSE (px)")
        ax.set_ylabel("Actual error, clipped (px)")
        ax.set_title(f"{p}  ρ={rho:.3f}  n={mask.sum()}")
        ax.legend(fontsize=7)

        # Bottom row: reliability diagram
        ax = axes[1, col]
        ax.set_title(
            f"{p} reliability diagram\n"
            f"Bar height = mean actual error for frames where predicted RMSE ∈ bin.\n"
            f"Bars following the diagonal ↗ = well calibrated."
        )
        if len(pr) >= 10:
            bin_edges = np.percentile(pr, np.linspace(0, 100, 11))
            bin_edges = np.unique(bin_edges)
            bcs, bms, bss, bns = [], [], [], []
            for b0, b1 in zip(bin_edges[:-1], bin_edges[1:]):
                mask_b = (pr >= b0) & (pr < b1)
                if mask_b.sum() >= 3:
                    bcs.append((b0 + b1) / 2)
                    bms.append(ae[mask_b].mean())
                    bss.append(ae[mask_b].std())
                    bns.append(int(mask_b.sum()))
            if bcs:
                bcs, bms, bss = np.array(bcs), np.array(bms), np.array(bss)
                bw = (bcs[1] - bcs[0]) * 0.8 if len(bcs) > 1 else bcs[0] * 0.5
                ax.bar(
                    bcs,
                    bms,
                    width=bw,
                    alpha=0.7,
                    color=CAND_COLORS[p],
                    label="mean actual error per bin",
                )
                ax.errorbar(bcs, bms, yerr=bss, fmt="none", color="black", capsize=3)
                ax.plot(
                    [0, clip], [0, clip], "k--", lw=1, label="perfect calibration (y=x)"
                )
        ax.set_xlim(0, clip)
        ax.set_ylim(0, clip)
        ax.set_xlabel("Predicted RMSE bin centre (px)")
        ax.set_ylabel("Mean actual error (px)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    return fig, "\n".join(lines)


# ---------------------------------------------------------------------------
# A3: Failure taxonomy
# ---------------------------------------------------------------------------

_TAXONOMY_CATS = ("both_good", "flow_only", "mask_only", "neither")
_TAXONOMY_COLORS = {
    "both_good": "#4CAF50",  # green
    "flow_only": "#2196F3",  # blue
    "mask_only": "#FF9800",  # orange
    "neither": "#F44336",  # red
}


def analysis_failure_taxonomy(seqs: list, threshold: float = THRESHOLD) -> tuple:
    """Frame taxonomy: four mutually exclusive, exhaustive categories.

    both_good  = flow ≤ t  AND mask_h ≤ t       (sum of four = 100%)
    flow_only  = flow ≤ t  AND mask_h > t
    mask_only  = flow > t  AND mask_h ≤ t        (includes no-flow frames)
    neither    = flow > t  AND mask_h > t

    Per-sequence pixel image: 1 px per frame, max_T columns wide.
    Sequences shorter than max_T leave the remaining pixels as light grey.
    """
    debug_seqs = [s for s in seqs if s["has_debug"]]
    if not debug_seqs:
        return None, "A3: No debug data."

    # Global counts
    flow_e = np.concatenate([s["cand_errors"]["flow"] for s in debug_seqs])
    mask_e = np.concatenate([s["cand_errors"]["mask"] for s in debug_seqs])
    fmH_e = np.concatenate([s["cand_errors"]["redetection_flow"] for s in debug_seqs])

    fg = np.isfinite(flow_e) & (flow_e <= threshold)
    mg = np.isfinite(mask_e) & (mask_e <= threshold)
    fmHg = np.isfinite(fmH_e) & (fmH_e <= threshold)

    N = len(flow_e)
    cats_n = {
        "both_good": int((fg & mg).sum()),
        "flow_only": int((fg & ~mg).sum()),
        "mask_only": int((~fg & mg).sum()),
        "neither": int((~fg & ~mg).sum()),
    }
    assert sum(cats_n.values()) == N, "taxonomy categories must be exhaustive"

    rescued_fmH = (~fg & ~mg) & fmHg
    n_no_flow = int((~np.isfinite(flow_e)).sum())

    lines = [
        "A3: Failure Taxonomy (four mutually exclusive categories)",
        f"  Total eval frames: {N}",
    ]
    for cat, n_cat in cats_n.items():
        lines.append(f"    {cat:15s}: {n_cat:5d} ({100 * n_cat / N:.1f}%)")
    lines.append(
        f'  Of "neither" ({cats_n["neither"]}): {rescued_fmH.sum()} '
        f"({100 * rescued_fmH.sum() / max(1, cats_n['neither']):.1f}%) "
        f"rescued by flow_maskH (redetection path)"
    )
    lines.append(
        f"  Frames with no flow H (NaN): {n_no_flow} ({100 * n_no_flow / N:.1f}%)"
        f" — these fall in mask_only or neither"
    )

    # Per-sequence pixel image: 1 px per frame, max_T columns wide
    sorted_seqs = sorted(debug_seqs, key=_seq_p15)
    n_seqs = len(sorted_seqs)
    max_T = max(len(s["errors"]) for s in sorted_seqs)

    # Pre-compute per-category RGB arrays
    cat_rgb = {
        cat: np.array(_hex_to_rgb01(_TAXONOMY_COLORS[cat])) for cat in _TAXONOMY_CATS
    }
    bg_rgb = np.array([0.85, 0.85, 0.85])  # light grey for padding

    img = np.ones((n_seqs, max_T, 3))
    img[:] = bg_rgb  # default: padding colour

    for row, s in enumerate(sorted_seqs):
        fe = s["cand_errors"]["flow"]
        me = s["cand_errors"]["mask"]
        T = len(fe)
        fg_s = np.isfinite(fe) & (fe <= threshold)
        mg_s = np.isfinite(me) & (me <= threshold)
        # assign colour per frame
        img[row, :T] = cat_rgb["neither"]  # default within-sequence
        img[row, :T][~fg_s & mg_s] = cat_rgb["mask_only"]
        img[row, :T][fg_s & ~mg_s] = cat_rgb["flow_only"]
        img[row, :T][fg_s & mg_s] = cat_rgb["both_good"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("A3: Failure Taxonomy", fontsize=14)

    ax = axes[0]
    ax.pie(
        list(cats_n.values()),
        labels=[f"{k}\n({v:,})" for k, v in cats_n.items()],
        colors=[_TAXONOMY_COLORS[k] for k in cats_n],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Overall frame taxonomy")

    ax = axes[1]
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xlabel(
        f"Frame index (1 px per frame; max {max_T} frames; grey = sequence ended)"
    )
    ax.set_ylabel(f"Sequences (n={n_seqs}, sorted worst↑ → best↓)")
    ax.set_yticks([])
    ax.set_title("Per-sequence taxonomy (each row = one sequence, worst at top)")
    patches = [
        mpatches.Patch(color=_TAXONOMY_COLORS[c], label=c) for c in _TAXONOMY_CATS
    ]
    patches.append(mpatches.Patch(color=bg_rgb, label="padding (seq ended)"))
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig, "\n".join(lines)


# ---------------------------------------------------------------------------
# A4: Selection confusion matrix
# ---------------------------------------------------------------------------


def analysis_selection_confusion(seqs: list, threshold: float = THRESHOLD) -> tuple:
    """When flow and mask_h were both computed, did we pick the better one?

    The oracle labels each frame with which of {flow, mask_h} had the lower
    actual error.  The confusion matrix shows how often our selection matched
    the oracle choice.
    """
    debug_seqs = [s for s in seqs if s["has_debug"]]
    if not debug_seqs:
        return None, "A4: No debug data."

    all_sel = np.concatenate([s["selected"] for s in debug_seqs])
    all_flow_e = np.concatenate([s["cand_errors"]["flow"] for s in debug_seqs])
    all_mask_e = np.concatenate([s["cand_errors"]["mask"] for s in debug_seqs])
    all_actual = np.concatenate([s["errors"] for s in debug_seqs])

    # Only frames where BOTH flow and mask_h produced a homography
    both = np.isfinite(all_flow_e) & np.isfinite(all_mask_e)
    sel = all_sel[both]
    flow_e = all_flow_e[both]
    mask_e = all_mask_e[both]
    actual = all_actual[both]

    # Oracle: which of the two was better (lower actual error)?
    oracle = np.where(flow_e <= mask_e, "flow", "mask")

    # Simplify selected to {flow, mask_h, other}
    sel_s = np.where(
        np.isin(sel, ["flow", "flow_maskH"]),
        "flow",
        np.where(sel == "mask_h", "mask", "other"),
    )

    # 2×2 confusion (flow vs mask_h; ignore 'other' for the matrix)
    core = np.isin(sel_s, ["flow", "mask"])
    sel_c = sel_s[core]
    oracle_c = oracle[core]
    actual_c = actual[core]

    labels = ["flow", "mask_h"]
    conf = np.zeros((2, 2), dtype=int)
    for i, sl in enumerate(labels):
        for j, ol in enumerate(labels):
            conf[i, j] = ((sel_c == sl) & (oracle_c == ol)).sum()

    n_core = core.sum()
    n_correct = int(np.diag(conf).sum())
    n_wrong_fm = int(conf[0, 1])  # selected flow, oracle mask_h
    n_wrong_mf = int(conf[1, 0])  # selected mask_h, oracle flow

    wrong_fm2 = (sel_s[core] == "flow") & (oracle_c == "mask")
    wrong_mf2 = (sel_s[core] == "mask") & (oracle_c == "flow")

    lines = [
        "A4: Selection Accuracy (frames where both flow + mask_h homographies available)",
        f"  Total such frames: {both.sum()} (of {len(all_sel)} total eval frames)",
        f"  Frames where selection matched oracle: {n_correct} ({100 * n_correct / max(1, n_core):.1f}%)",
        f"  Wrong: selected flow but mask_h was better:  {n_wrong_fm} ({100 * n_wrong_fm / max(1, n_core):.1f}%)",
        f"  Wrong: selected mask_h but flow was better:  {n_wrong_mf} ({100 * n_wrong_mf / max(1, n_core):.1f}%)",
    ]
    if n_wrong_fm > 0:
        lines.append(
            f"    Mean actual error when sel=flow, oracle=mask_h: {actual_c[wrong_fm2].mean():.1f} px"
        )
    if n_wrong_mf > 0:
        lines.append(
            f"    Mean actual error when sel=mask_h, oracle=flow: {actual_c[wrong_mf2].mean():.1f} px"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "A4: Selection Accuracy — which of {flow, mask_h} was better?\n"
        "Oracle = whichever candidate had lower actual corner error.",
        fontsize=12,
    )

    ax = axes[0]
    im = ax.imshow(conf, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["flow\n(oracle best)", "mask_h\n(oracle best)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["flow\n(selected)", "mask_h\n(selected)"])
    ax.set_title("Selection vs oracle — frame counts\n(diagonal = correct selection)")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            pct = 100 * conf[i, j] / max(1, n_core)
            ax.text(
                j,
                i,
                f"{conf[i, j]:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if conf[i, j] > conf.max() * 0.6 else "black",
                fontsize=10,
            )

    ax = axes[1]
    bins = np.linspace(0, 60, 35)
    correct_c = ~wrong_fm2 & ~wrong_mf2
    ax.hist(
        actual_c[correct_c],
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"Correct ({correct_c.sum()})",
        color="steelblue",
    )
    if wrong_fm2.any():
        ax.hist(
            actual_c[wrong_fm2],
            bins=bins,
            alpha=0.5,
            density=True,
            label=f"Picked flow, mask_h was better ({wrong_fm2.sum()})",
            color="orange",
        )
    if wrong_mf2.any():
        ax.hist(
            actual_c[wrong_mf2],
            bins=bins,
            alpha=0.5,
            density=True,
            label=f"Picked mask_h, flow was better ({wrong_mf2.sum()})",
            color="red",
        )
    ax.axvline(
        threshold, color="black", linestyle="--", label=f"threshold={threshold:.0f} px"
    )
    ax.set_xlabel("Actual alignment error (px)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution by selection correctness")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig, "\n".join(lines)


# ---------------------------------------------------------------------------
# A5: Error trajectories – overview + per-sequence files
# ---------------------------------------------------------------------------


def _classify_trajectory(errors: np.ndarray, threshold: float = THRESHOLD) -> str:
    if len(errors) < 5:
        return "short"
    p15 = _p_at_k(errors, threshold)
    if p15 >= 90:
        return "stable"
    q = max(1, len(errors) // 4)
    if errors[-q:].mean() > max(3 * errors[:q].mean(), threshold) and p15 < 50:
        return "lost"
    trend = float(np.polyfit(np.arange(len(errors)), errors, 1)[0])
    if trend > 0.1 and p15 < 70:
        return "drift"
    return "poor" if p15 < 50 else "variable"


def _draw_trajectory(ax: plt.Axes, s: dict, threshold: float = THRESHOLD) -> None:
    """Draw one sequence trajectory on *ax* (background + error curve)."""
    errors = s["errors"]
    T = len(errors)
    ylim = max(20.0, float(np.percentile(errors, 99)) * 1.1)
    ylim = min(60.0, ylim)

    if s["has_debug"] and T > 0:
        # Background image: one pixel per frame, colored by selected candidate
        sel = s["selected"]
        bg = np.zeros((1, T, 3))
        for i in range(T):
            bg[0, i] = _hex_to_rgb01(CAND_COLORS.get(sel[i], "#DDDDDD"))
        ax.imshow(
            bg,
            extent=[-0.5, T - 0.5, 0, ylim],
            aspect="auto",
            alpha=0.35,
            zorder=1,
            interpolation="nearest",
        )

    ax.plot(errors, color="black", linewidth=0.8, zorder=5)
    ax.axhline(5, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(15, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, ylim)
    ax.set_xlim(-0.5, T - 0.5)


def analysis_error_trajectories(seqs: list, n_show: int = 12) -> tuple:
    """Overview figure: worst n_show sequences by P@15."""
    sorted_seqs = sorted(seqs, key=_seq_p15)
    worst = sorted_seqs[:n_show]

    n_cols = 4
    n_rows = max(1, (n_show + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    fig.suptitle(
        f"A5: Error Trajectories — worst {n_show} sequences by P@15", fontsize=14
    )
    axes_flat = np.array(axes).flatten()

    for idx, s in enumerate(worst):
        ax = axes_flat[idx]
        _draw_trajectory(ax, s)
        cls = _classify_trajectory(s["errors"])
        ax.set_title(f"{s['name']}\nP@15={_seq_p15(s):.1f}% [{cls}]", fontsize=8)
        ax.set_xlabel("frame", fontsize=7)
        ax.set_ylabel("error (px)", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(len(worst), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    patches = [
        mpatches.Patch(color=CAND_COLORS[p], alpha=0.5, label=p)
        for p in ["flow", "mask", "redetection_flow", "last_H"]
    ]
    fig.legend(
        handles=patches, loc="lower right", fontsize=9, title="Selected candidate (bg)"
    )

    counts = Counter(_classify_trajectory(s["errors"]) for s in seqs)
    lines = [f"A5: Trajectory classification (n={len(seqs)} sequences):"]
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:12s}: {cnt:4d} ({100 * cnt / len(seqs):.1f}%)")

    plt.tight_layout()
    return fig, "\n".join(lines)


def generate_trajectory_files(seqs: list, output_dir: Path) -> str:
    """Save one PNG per sequence to output_dir/trajectories/."""
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for s in seqs:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        _draw_trajectory(ax, s)
        cls = _classify_trajectory(s["errors"])
        ax.set_title(
            f"{s['name']} — P@5={_p_at_k(s['errors'], 5):.1f}%  "
            f"P@15={_seq_p15(s):.1f}%  [{cls}]  n={len(s['errors'])} frames",
            fontsize=9,
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("error (px)")
        patches = [
            mpatches.Patch(color=CAND_COLORS[p], alpha=0.5, label=p)
            for p in ["flow", "mask", "redetection_flow", "last_H"]
        ]
        ax.legend(
            handles=patches,
            fontsize=7,
            title="Selected candidate (bg)",
            loc="upper right",
            ncol=2,
        )
        fig.tight_layout()
        fig.savefig(traj_dir / f"{s['name']}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    return f"Saved {len(seqs)} trajectory PNGs to {traj_dir}"


# ---------------------------------------------------------------------------
# A6: Regret heatmap (pixel image)
# ---------------------------------------------------------------------------


def analysis_regret_heatmap(seqs: list) -> tuple:
    """Regret = actual_error − oracle_error (clipped to ≥0).

    Rendered as a pixel image: one pixel row per sequence, one pixel column
    per frame.  Image width = max sequence length; shorter sequences are
    padded with NaN (shown as a neutral colour via masked-array colormap).
    Bright = high selection regret.
    """
    debug_seqs = [s for s in seqs if s["has_debug"] and len(s["errors"]) > 0]
    if not debug_seqs:
        return None, "A6: No debug data."

    sorted_seqs = sorted(debug_seqs, key=_seq_p15)
    n_seqs = len(sorted_seqs)
    max_T = max(len(s["errors"]) for s in sorted_seqs)

    img = np.full((n_seqs, max_T), np.nan)

    for row, s in enumerate(sorted_seqs):
        T = len(s["errors"])
        oracle = s["oracle_errors"]
        regret = np.clip(
            s["errors"] - np.where(np.isfinite(oracle), oracle, 0.0), 0, None
        )
        img[row, :T] = regret

    # Target ~6–8 inches tall; derive width from data aspect ratio (max_T / n_seqs)
    fig_h = min(8, max(3, n_seqs * 0.025))
    fig_w = min(20, max(10, fig_h * max_T / n_seqs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.suptitle(
        "A6: Selection Regret Heatmap — 1 px per frame\n"
        "Regret = max(actual error − oracle error, 0).  "
        "Bright = wrong candidate selected.  Grey = sequence ended.",
        fontsize=11,
    )

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#cccccc")  # padding colour for NaN (sequence ended)
    im = ax.imshow(
        np.ma.masked_invalid(img),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=20,
    )
    plt.colorbar(im, ax=ax, label="Regret (px)", fraction=0.02, pad=0.01)
    ax.set_xlabel(f"Frame index (1 px per frame; max {max_T} frames)")
    ax.set_ylabel(f"Sequences (n={n_seqs}, sorted worst↑ → best↓)")
    ax.set_yticks([])

    plt.tight_layout()
    return (
        fig,
        f"A6: Regret heatmap — {n_seqs} sequences × {max_T} frames (1 px per frame).",
    )


# ---------------------------------------------------------------------------
# A7: Sequence-level feature scatter
# ---------------------------------------------------------------------------


def analysis_feature_scatter(seqs: list) -> tuple:
    debug_seqs = [s for s in seqs if s["has_debug"] and len(s["errors"]) > 0]
    if not debug_seqs:
        return None, "A7: No debug data."

    # Print actual selection distribution (helps diagnose label surprises)
    all_sel = np.concatenate([s["selected"] for s in debug_seqs])
    sel_dist = Counter(all_sel.tolist())
    sel_lines = ["  Selection label distribution across all debug frames:"]
    total_sel = len(all_sel)
    for lbl, cnt in sorted(sel_dist.items(), key=lambda x: -x[1]):
        sel_lines.append(
            f"    {repr(lbl):20s}: {cnt:6d} ({100 * cnt / total_sel:.1f}%)"
        )

    p15s = np.array([_seq_p15(s) for s in debug_seqs])

    feat_defs = {
        "frac_flow_selected": lambda s: np.isin(
            s["selected"], ["flow", "flow_maskH"]
        ).mean(),
        "frac_mask_h_selected": lambda s: (s["selected"] == "mask_h").mean(),
        "frac_last_H_fallback": lambda s: (s["selected"] == "last_H").mean(),
        "frac_flow_available": lambda s: np.isfinite(s["cand_errors"]["flow"]).mean(),
        "median_pred_rmse_flow": lambda s: float(np.nanmedian(s["pred_rmse"]["flow"])),
        "median_pred_rmse_mask_h": lambda s: float(
            np.nanmedian(s["pred_rmse"]["mask_h"])
        ),
        "sequence_length": lambda s: len(s["errors"]),
    }
    feat_arrays = {
        k: np.array([fn(s) for s in debug_seqs]) for k, fn in feat_defs.items()
    }

    n_feats = len(feat_defs)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle("A7: Sequence-Level Features vs P@15", fontsize=14)
    axes_flat = np.array(axes).flatten()

    corr_lines = ["A7: Feature correlations with P@15 (Spearman ρ):"]
    for idx, (feat_name, feat_vals) in enumerate(feat_arrays.items()):
        ax = axes_flat[idx]
        rho = _spearman_r(feat_vals, p15s)
        ax.scatter(feat_vals, p15s, s=15, alpha=0.5, color="steelblue")
        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel("P@15 (%)", fontsize=9)
        ax.set_title(f"ρ = {rho:.3f}", fontsize=10)
        corr_lines.append(f"  {feat_name:30s}: ρ = {rho:.3f}")

    for idx in range(n_feats, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    lines = ["A7: Feature scatter"] + sel_lines + [""] + corr_lines
    return fig, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    default_alias_config = Path("configs/aliases/default.py")
    default_alias_config = (
        default_alias_config if default_alias_config.exists() else None
    )

    parser = argparse.ArgumentParser(
        description="Analyze FlatSAM2 results for failure mode investigation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", choices=CLI_CHOICES, help="Dataset key (e.g. pti)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config file path or stem (e.g. flow_hough_symmetry_nn)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Top-level results directory (path_config default if unset)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Dataset base directory (path_config default if unset)",
    )
    parser.add_argument(
        "--initfix-dir",
        type=Path,
        default=None,
        help="Initfix dir for pti (path_config default if unset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis"),
        help="Directory for figures and report",
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", choices=["train", "test"], default=None)
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=default_alias_config,
        help=(
            "Python file defining config-name display aliases (expects 'aliases' "
            "mapping). Defaults to configs/aliases/default.py when present."
        ),
    )
    parser.add_argument(
        "--n-trajectory",
        type=int,
        default=12,
        help="Worst sequences shown in trajectory overview figure",
    )
    args = parser.parse_args()

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR

    config_name = args.config.stem if args.config is not None else "default"
    try:
        config_aliases = load_config_aliases_from_file(args.alias_config)
    except Exception as e:
        print(f"ERROR: Could not load alias config: {e}", file=sys.stderr)
        sys.exit(1)
    config_display_name = get_config_display_name(config_name, config_aliases)

    results_dir = compute_output_dir(args.results_dir, args.dataset, args.config)
    print(f"Results directory: {results_dir}")

    sequences = get_sequences(
        args.dataset,
        args.base_dir,
        sequences=args.sequences,
        subset=args.subset,
        split=args.split,
    )
    print(f"Sequences to analyse: {len(sequences)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seqs = load_all_sequences(
        args.dataset,
        args.base_dir,
        results_dir,
        sequences,
        args.initfix_dir,
    )
    if not seqs:
        print("ERROR: No sequences loaded.", file=sys.stderr)
        sys.exit(1)

    overall_p5 = float(np.nanmean([_p_at_k(s["errors"], 5) for s in seqs]))
    overall_p15 = float(np.nanmean([_p_at_k(s["errors"], 15) for s in seqs]))
    report_lines = [
        "FlatSAM2 Results Analysis",
        f"Dataset  : {ds_info.label}",
        f"Config   : {config_display_name} ({config_name})",
        f"Results  : {results_dir}",
        f"Sequences: {len(seqs)}",
        f"Overall P@5  (mean over seqs): {overall_p5:.1f}%",
        f"Overall P@15 (mean over seqs): {overall_p15:.1f}%",
        "",
    ]

    analyses = [
        ("oracle_gap", lambda s: analysis_oracle_gap(s)),
        ("score_calibration", lambda s: analysis_score_calibration(s)),
        ("failure_taxonomy", lambda s: analysis_failure_taxonomy(s)),
        ("selection_confusion", lambda s: analysis_selection_confusion(s)),
        (
            "error_trajectories",
            lambda s: analysis_error_trajectories(s, args.n_trajectory),
        ),
        ("regret_heatmap", lambda s: analysis_regret_heatmap(s)),
        ("feature_scatter", lambda s: analysis_feature_scatter(s)),
    ]

    for name, fn in analyses:
        print(f"Running {name}...")
        try:
            fig, text = fn(seqs)
            report_lines.append(text)
            report_lines.append("")
            if fig is not None:
                out_path = args.output_dir / f"{name}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  → {out_path}")
        except Exception as e:
            msg = f"  WARNING: {name} failed: {e}"
            print(msg, file=sys.stderr)
            traceback.print_exc()
            report_lines.append(msg)
            report_lines.append("")

    # Per-sequence trajectory files
    print("Generating per-sequence trajectory files...")
    try:
        msg = generate_trajectory_files(seqs, args.output_dir)
        print(f"  {msg}")
        report_lines.append(msg)
    except Exception as e:
        print(f"  WARNING: trajectory file generation failed: {e}", file=sys.stderr)
        traceback.print_exc()

    report_path = args.output_dir / "report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport: {report_path}")
    print("\n".join(report_lines[:20]))  # brief preview


if __name__ == "__main__":
    main()
