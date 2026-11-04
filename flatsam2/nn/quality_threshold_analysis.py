"""Evaluate decision rules for binary homography quality acceptance.

Analyzes how to use the NN's predicted RMSE and uncertainty (sigma) to make
a binary "good / bad" decision with a controlled false-positive rate.

Three publishable decision rules are compared:

  A) Simple threshold:      accept if predicted_rmse < τ
  B) UCB:                   accept if predicted_rmse + k·σ < τ_bad
  C) Probabilistic (Gauss): accept if P(error > τ_bad | pred, σ) < α
                            ≡ predicted_rmse + Φ⁻¹(1-α)·σ < τ_bad
  D) Conformal prediction:  accept if predicted_rmse + q̂·σ < τ_bad
                            where q̂ is the (1-α) empirical quantile of
                            non-conformity scores (gt-pred)/σ on a held-out
                            calibration split.  Provides a distribution-free
                            finite-sample coverage guarantee.

Usage examples::

    # Preferred mode: explicit calibration and test NPZs (sequence-disjoint splits)
    python -m flatsam2.nn.quality_threshold_analysis \\
        --cal-npz /path/eval/ckpt__split10000__details.npz \\
        --test-npz /path/eval/ckpt__split100000__details.npz \\
        --output-dir /tmp/threshold_analysis

    # Legacy single-NPZ mode: internal 50/50 split (no sequence-disjoint guarantee)
    python -m flatsam2.nn.quality_threshold_analysis \\
        --npz /path/to/details.npz \\
        --output-dir /tmp/threshold_analysis

    # Inference mode: run the model and analyse in one shot
    python -m flatsam2.nn.quality_threshold_analysis \\
        --dataset-dir /scratch/realworld_val \\
        --checkpoint checkpoints/option_e/best.pt \\
        --output-dir /tmp/threshold_analysis \\
        --subset-size 5000
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Analysis hyper-parameters exposed via CLI."""

    t_good: float = 12.0
    """GT RMSE threshold below which a homography is 'good' (positive class).
    PlanarTrack GT has mean annotation error ~5.71px; t_good must be well above
    this noise floor so that truly good homographies are not mislabelled as grey."""
    t_bad: float = 25.0
    """GT RMSE threshold above which it is 'bad' (negative class).
    10.71% of PlanarTrack annotations have GT error >15px, so t_bad=15px would
    contaminate the negative class with mislabelled positives.  25px provides
    a safe margin above that contamination zone.
    Samples in [t_good, t_bad] are the grey zone and excluded from TPR/FPR."""
    tau_bad: float = 25.0
    """The acceptance boundary used in decision rules B/C/D (pixels).
    Should match t_bad so that the decision boundary aligns with the evaluation."""
    alpha: float = 0.001
    """Target false-positive rate for rules C and D."""
    k_values: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 3.09])
    """k multipliers for rule B (UCB); also used to draw multiple ROC curves."""
    cal_fraction: float = 0.5
    """Fraction of data used for conformal calibration split."""
    seed: int = 42
    n_roc_thresholds: int = 500
    target_fprs: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05])
    calibration_n_bins: int = 20
    censor_threshold: float = 30.0
    """GT RMSE values above this are treated as censored (excluded from calibration)."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions_from_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load pre-computed predictions from a details NPZ file.

    The NPZ is expected to come from evaluate_realworld_dataset.py and
    must contain at minimum: pred_rmse, log_sigma, gt_rmse, gt_rmse_raw.

    Args:
        npz_path: Path to the NPZ file.

    Returns:
        Dict with float32 arrays keyed by field name.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        result: Dict[str, np.ndarray] = {}
        for key in data.files:
            arr = data[key]
            # Keep string arrays as-is; cast numeric arrays to float32
            if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(
                arr.dtype, np.integer
            ):
                result[key] = arr.astype(np.float32)
            else:
                result[key] = arr
    required = {"pred_rmse", "log_sigma", "gt_rmse"}
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"NPZ is missing required keys: {missing}")
    # Prefer raw (unclipped) GT RMSE when available
    if "gt_rmse_raw" in result:
        result["gt_rmse"] = result["gt_rmse_raw"].astype(np.float32)
    return result


def load_predictions_via_inference(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    subset_size: Optional[int],
    batch_size: int,
    num_workers: int,
    device_str: str,
    gt_rmse_clip: float,
) -> Dict[str, np.ndarray]:
    """Run inference on a dataset checkpoint and return predictions.

    Delegates entirely to evaluate_checkpoint() from evaluate_realworld_dataset,
    then loads the resulting NPZ.

    Args:
        dataset_dir: Collected real-world dataset directory.
        checkpoint_path: Path to model checkpoint (.pt) or run directory.
        output_dir: Where to write the intermediate NPZ.
        subset_size: Optionally limit to this many samples.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader workers.
        device_str: torch device string (e.g. 'cuda', 'cpu').
        gt_rmse_clip: GT RMSE clipping value for regression evaluation.

    Returns:
        Dict with float32 arrays (same format as load_predictions_from_npz).
    """
    import torch

    from .evaluate_realworld_dataset import (
        _resolve_checkpoint_path,
        evaluate_checkpoint,
    )

    resolved = _resolve_checkpoint_path(checkpoint_path)
    device = torch.device(device_str)
    result = evaluate_checkpoint(
        dataset_dir=dataset_dir,
        checkpoint_path=resolved,
        subset_size=subset_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        output_dir=output_dir,
        gt_rmse_clip=gt_rmse_clip,
    )
    details_npz = Path(result["details_npz"])
    return load_predictions_from_npz(details_npz)


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


def compute_standardized_residuals(
    pred_rmse: np.ndarray,
    log_sigma: np.ndarray,
    gt_rmse: np.ndarray,
    censor_threshold: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute standardized residuals z = (gt - pred) / sigma.

    Only uncensored samples (gt < censor_threshold) are included, because
    the Gaussian model was only trained on those.

    Args:
        pred_rmse: (N,) predicted RMSE in pixels.
        log_sigma: (N,) log-uncertainty from the model.
        gt_rmse: (N,) ground truth RMSE in pixels.
        censor_threshold: GT values above this are excluded.

    Returns:
        (z, mask): z is the (M,) standardized residuals array;
                   mask is the (N,) boolean mask selecting uncensored samples.
    """
    mask = gt_rmse < censor_threshold
    sigma = np.exp(np.clip(log_sigma, -5.0, 5.0))
    z = (gt_rmse[mask] - pred_rmse[mask]) / sigma[mask]
    return z, mask


def compute_calibration_curve(
    pred_rmse: np.ndarray,
    log_sigma: np.ndarray,
    gt_rmse: np.ndarray,
    tau: float = 15.0,
    n_bins: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute reliability diagram data for P(gt < tau | pred, sigma).

    Bins samples by their predicted probability p̂ = Φ((τ-pred)/σ) and
    computes the observed fraction within each bin.

    Args:
        pred_rmse: (N,) predicted RMSE.
        log_sigma: (N,) log-uncertainty.
        gt_rmse: (N,) ground truth RMSE.
        tau: Acceptance threshold (pixels).
        n_bins: Number of equal-width bins in [0, 1].

    Returns:
        Dict with keys: bin_centers (n_bins,), pred_prob (n_bins,),
        obs_frac (n_bins,), counts (n_bins,).
    """
    sigma = np.exp(np.clip(log_sigma, -5.0, 5.0))
    p_hat = scipy.stats.norm.cdf((tau - pred_rmse) / sigma)
    actually_below = (gt_rmse < tau).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pred_prob = np.zeros(n_bins, dtype=np.float64)
    obs_frac = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    bin_ids = np.digitize(p_hat, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    for b in range(n_bins):
        in_bin = bin_ids == b
        n = int(np.sum(in_bin))
        counts[b] = n
        if n > 0:
            pred_prob[b] = float(np.mean(p_hat[in_bin]))
            obs_frac[b] = float(np.mean(actually_below[in_bin]))

    return {
        "bin_centers": bin_centers,
        "pred_prob": pred_prob,
        "obs_frac": obs_frac,
        "counts": counts,
    }


def compute_ece(calibration_result: Dict[str, np.ndarray]) -> float:
    """Compute Expected Calibration Error from a reliability diagram.

    Args:
        calibration_result: Output of compute_calibration_curve().

    Returns:
        Weighted mean absolute difference between predicted and observed fractions.
    """
    pred = calibration_result["pred_prob"]
    obs = calibration_result["obs_frac"]
    counts = calibration_result["counts"]
    total = float(np.sum(counts))
    if total == 0:
        return float("nan")
    weights = counts / total
    return float(np.sum(weights * np.abs(pred - obs)))


# ---------------------------------------------------------------------------
# ROC analysis
# ---------------------------------------------------------------------------


def sweep_roc_for_score(
    acceptance_score: np.ndarray,
    gt_rmse: np.ndarray,
    t_good: float,
    t_bad: float,
    n_thresholds: int = 500,
) -> Dict[str, np.ndarray]:
    """Compute a ROC curve by thresholding a continuous acceptance score.

    Samples in the grey zone [t_good, t_bad] are excluded from TPR/FPR.
    Positives = gt_rmse < t_good, negatives = gt_rmse > t_bad.

    Args:
        acceptance_score: (N,) float — higher means more likely to accept.
        gt_rmse: (N,) ground truth RMSE.
        t_good: Upper edge of the positive class.
        t_bad: Lower edge of the negative class.
        n_thresholds: Number of threshold values to sweep.

    Returns:
        Dict with fpr (T,), tpr (T,), thresholds (T,), auc (scalar).
    """
    is_positive = gt_rmse < t_good
    is_negative = gt_rmse > t_bad
    valid = is_positive | is_negative

    score_v = acceptance_score[valid]
    pos_v = is_positive[valid]
    neg_v = is_negative[valid]

    n_pos = int(np.sum(pos_v))
    n_neg = int(np.sum(neg_v))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"No positives ({n_pos}) or negatives ({n_neg}) after grey-zone exclusion "
            f"with t_good={t_good}, t_bad={t_bad}."
        )

    thresholds = np.linspace(
        float(np.min(score_v)) - 1e-6,
        float(np.max(score_v)) + 1e-6,
        n_thresholds,
    )
    fpr = np.empty(n_thresholds, dtype=np.float64)
    tpr = np.empty(n_thresholds, dtype=np.float64)

    for i, thr in enumerate(thresholds):
        accepted = score_v >= thr
        tp = int(np.sum(accepted & pos_v))
        fp = int(np.sum(accepted & neg_v))
        tpr[i] = tp / n_pos
        fpr[i] = fp / n_neg

    auc = float(-np.trapz(tpr, fpr))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def get_operating_point(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, target_fpr: float
) -> Dict[str, float]:
    """Find the most favourable operating point with FPR ≤ target_fpr.

    Scans for the threshold that achieves FPR ≤ target_fpr while maximising TPR.

    Args:
        fpr: (T,) false positive rates.
        tpr: (T,) true positive rates.
        thresholds: (T,) corresponding acceptance score thresholds.
        target_fpr: Maximum acceptable FPR.

    Returns:
        Dict with achieved_fpr, achieved_tpr, threshold.
        If no operating point satisfies the constraint, all values are NaN.
    """
    feasible = fpr <= target_fpr
    if not np.any(feasible):
        return {"achieved_fpr": float("nan"), "achieved_tpr": float("nan"), "threshold": float("nan")}
    best_idx = int(np.argmax(tpr * feasible))
    return {
        "achieved_fpr": float(fpr[best_idx]),
        "achieved_tpr": float(tpr[best_idx]),
        "threshold": float(thresholds[best_idx]),
    }


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


def split_calibration_test(
    gt_rmse: np.ndarray,
    cal_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified calibration / test split by GT RMSE bins.

    Uses _DEFAULT_BIN_EDGES from collect_realworld_dataset to ensure both
    splits cover the full RMSE distribution proportionally.

    Args:
        gt_rmse: (N,) ground truth RMSE values.
        cal_fraction: Fraction of samples to allocate to calibration.
        seed: Random seed for reproducibility.

    Returns:
        (cal_indices, test_indices): int64 index arrays into the original array.
    """
    from .collect_realworld_dataset import _DEFAULT_BIN_EDGES, create_stratified_subset

    n_total = len(gt_rmse)
    cal_size = max(1, int(math.ceil(cal_fraction * n_total)))
    rng = np.random.default_rng(seed)
    cal_indices, _ = create_stratified_subset(
        gt_rmse.astype(np.float32),
        cal_size,
        _DEFAULT_BIN_EDGES,
        rng,
    )
    cal_set = set(cal_indices.tolist())
    test_indices = np.array(
        [i for i in range(n_total) if i not in cal_set], dtype=np.int64
    )
    return cal_indices.astype(np.int64), test_indices


def fit_conformal_quantile(
    pred_rmse_cal: np.ndarray,
    log_sigma_cal: np.ndarray,
    gt_rmse_cal: np.ndarray,
    alpha: float,
    censor_threshold: float = 30.0,
) -> Tuple[float, int]:
    """Fit the split conformal non-conformity quantile.

    Non-conformity score: r_i = (gt_i - pred_i) / σ_i.
    Returns q̂ such that P(gt ≤ pred + q̂·σ) ≥ 1-alpha, with the finite-sample
    correction q̂ = ceil((n+1)(1-alpha))-th order statistic.

    Only uncensored calibration samples (gt < censor_threshold) are used,
    matching the training regime.

    Args:
        pred_rmse_cal: (M,) predicted RMSE on calibration split.
        log_sigma_cal: (M,) log-sigma on calibration split.
        gt_rmse_cal: (M,) GT RMSE on calibration split.
        alpha: Target miscoverage rate (e.g. 0.001).
        censor_threshold: GT values above this are excluded.

    Returns:
        (q_hat, n_cal): fitted quantile and number of usable calibration samples.
    """
    z, mask = compute_standardized_residuals(
        pred_rmse_cal, log_sigma_cal, gt_rmse_cal, censor_threshold=censor_threshold
    )
    n_cal = int(np.sum(mask))
    if n_cal == 0:
        raise ValueError("No uncensored calibration samples available.")

    # Finite-sample correction: use the ceil((n+1)(1-alpha))-th order stat
    rank = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    rank = min(rank, n_cal)  # clamp to valid index
    q_hat = float(np.sort(z)[rank - 1])  # 1-based → 0-based index
    return q_hat, n_cal


def compute_single_operating_point(
    acceptance_score: np.ndarray,
    gt_rmse: np.ndarray,
    t_good: float,
    t_bad: float,
    score_threshold: float,
) -> Dict[str, float]:
    """Evaluate a fixed threshold on the acceptance score.

    Args:
        acceptance_score: (N,) float — higher means more likely to accept.
        gt_rmse: (N,) ground truth RMSE.
        t_good: Upper edge of positive class.
        t_bad: Lower edge of negative class.
        score_threshold: The fixed threshold to apply.

    Returns:
        Dict with tpr, fpr, fnr, n_accepted, n_pos, n_neg, n_grey, accept_rate_grey.
    """
    is_positive = gt_rmse < t_good
    is_negative = gt_rmse > t_bad
    is_grey = (~is_positive) & (~is_negative)
    accepted = acceptance_score >= score_threshold

    n_pos = int(np.sum(is_positive))
    n_neg = int(np.sum(is_negative))
    n_grey = int(np.sum(is_grey))

    tp = int(np.sum(accepted & is_positive))
    fp = int(np.sum(accepted & is_negative))
    tpr = tp / n_pos if n_pos > 0 else float("nan")
    fpr = fp / n_neg if n_neg > 0 else float("nan")
    fnr = 1.0 - tpr if not math.isnan(tpr) else float("nan")
    accept_grey = int(np.sum(accepted & is_grey))

    return {
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_grey": n_grey,
        "n_accepted": int(np.sum(accepted)),
        "accept_rate_grey": accept_grey / n_grey if n_grey > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _require_matplotlib():
    """Import matplotlib, raising a descriptive error if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ValueError("matplotlib is required for plotting") from exc


def plot_qq(
    z: np.ndarray,
    out_path: Path,
    ks_stat: float,
    ks_pvalue: float,
    censor_excluded: int,
    total: int,
) -> None:
    """Q-Q plot of standardized residuals against N(0,1).

    Args:
        z: (M,) standardized residuals.
        out_path: Output PNG path.
        ks_stat: Kolmogorov-Smirnov statistic.
        ks_pvalue: KS test p-value.
        censor_excluded: Number of samples excluded due to censoring.
        total: Total number of samples before exclusion.
    """
    plt = _require_matplotlib()

    n = len(z)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical = scipy.stats.norm.ppf(quantiles)
    empirical = np.sort(z)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical, empirical, s=4, alpha=0.4, color="steelblue", label="residuals")
    lo = min(float(theoretical[0]), float(empirical[0]))
    hi = max(float(theoretical[-1]), float(empirical[-1]))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y=x (perfect)")
    ax.set_xlabel("Theoretical N(0,1) quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(
        f"Q-Q plot of standardized residuals\n"
        f"KS stat={ks_stat:.3f}  p={ks_pvalue:.4f}  "
        f"n={n}  (excluded {censor_excluded}/{total} censored)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration_diagram(
    calibration_result: Dict[str, np.ndarray],
    ece: float,
    tau: float,
    out_path: Path,
) -> None:
    """Reliability diagram: predicted probability vs observed fraction.

    Args:
        calibration_result: Output of compute_calibration_curve().
        ece: Expected Calibration Error.
        tau: The threshold used (for the plot title).
        out_path: Output PNG path.
    """
    plt = _require_matplotlib()

    pred = calibration_result["pred_prob"]
    obs = calibration_result["obs_frac"]
    counts = calibration_result["counts"]
    n_total = int(np.sum(counts))

    fig, ax = plt.subplots(figsize=(5, 5))
    # Size bars by sample count
    widths = np.full(len(pred), 1.0 / len(pred))
    ax.bar(
        calibration_result["bin_centers"],
        obs,
        width=widths * 0.85,
        label="Observed fraction",
        alpha=0.75,
        color="steelblue",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5, label="Perfect calibration")
    ax.scatter(pred, obs, s=20, color="navy", zorder=5, label="Bin mean predicted prob")
    ax.set_xlabel(f"Predicted P(gt < {tau:.0f}px)")
    ax.set_ylabel(f"Observed fraction with gt < {tau:.0f}px")
    ax.set_title(f"Reliability diagram  (τ={tau:.0f}px)\nECE={ece:.4f}  n={n_total}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    roc_results: List[Dict],
    target_fprs: List[float],
    out_path: Path,
) -> None:
    """ROC curves for multiple methods with target FPR annotations.

    Args:
        roc_results: List of dicts, each with keys: name, fpr, tpr, auc.
        target_fprs: Vertical reference lines at these FPR values.
        out_path: Output PNG path.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, r in enumerate(roc_results):
        color = colors[i % len(colors)]
        ax.plot(
            r["fpr"],
            r["tpr"],
            color=color,
            linewidth=1.5,
            label=f"{r['name']} (AUC={r['auc']:.3f})",
        )

    for tgt in target_fprs:
        ax.axvline(tgt, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(tgt, 0.02, f"FPR={tgt:.3f}", fontsize=6, ha="left", va="bottom", color="gray")

    ax.set_xlabel("FPR (bad marked as good)")
    ax.set_ylabel("TPR (good correctly accepted)")
    ax.set_title("ROC curves — homography quality decision rules")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_distributions(
    scores_good: np.ndarray,
    scores_grey: np.ndarray,
    scores_bad: np.ndarray,
    rule_name: str,
    out_path: Path,
) -> None:
    """Histograms of acceptance score for good / grey / bad samples.

    Args:
        scores_good: Acceptance scores for gt < t_good samples.
        scores_grey: Acceptance scores for grey-zone samples.
        scores_bad: Acceptance scores for gt > t_bad samples.
        rule_name: Label for the x-axis.
        out_path: Output PNG path.
    """
    plt = _require_matplotlib()

    all_scores = np.concatenate([scores_good, scores_grey, scores_bad])
    bins = np.linspace(float(np.min(all_scores)), float(np.max(all_scores)), 60)

    fig, ax = plt.subplots(figsize=(7, 4))
    for scores, label, color in [
        (scores_good, f"good (n={len(scores_good)})", "green"),
        (scores_grey, f"grey zone (n={len(scores_grey)})", "orange"),
        (scores_bad, f"bad (n={len(scores_bad)})", "red"),
    ]:
        if len(scores) > 0:
            ax.hist(scores, bins=bins, density=True, alpha=0.55, label=label, color=color)
    ax.set_xlabel(rule_name)
    ax.set_ylabel("Density")
    ax.set_title("Acceptance score distributions by class")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sigma_diagnostics(
    sigma: np.ndarray,
    sigma_recal_std: np.ndarray,
    sigma_recal_q: np.ndarray,
    abs_error: np.ndarray,
    gt_rmse: np.ndarray,
    pred_rmse: np.ndarray,
    t_good: float,
    t_bad: float,
    spearman_rho: float,
    spearman_p: float,
    a_std: float,
    a_q: float,
    plots_dir: Path,
) -> None:
    """Generate sigma diagnostic plots.

    Produces:
      sigma_vs_error.png      — scatter of raw sigma vs |gt-pred|
      sigma_distribution.png  — histograms of raw and recalibrated sigmas
      sigma_per_bin.png       — per GT-RMSE bin: mean sigma vs mean |gt-pred|

    Args:
        sigma: (N,) raw sigma from model.
        sigma_recal_std: (N,) sigma recalibrated by std-based factor a_std.
        sigma_recal_q: (N,) sigma recalibrated by quantile-based factor a_q.
        abs_error: (N,) |gt_rmse - pred_rmse|.
        gt_rmse: (N,) ground truth RMSE (unclipped).
        pred_rmse: (N,) predicted RMSE.
        t_good: Upper edge of positive class.
        t_bad: Lower edge of negative class.
        spearman_rho: Pre-computed Spearman rank correlation (sigma vs abs_error).
        spearman_p: Corresponding p-value.
        a_std: Std-based scale factor (for title).
        a_q: Quantile-based scale factor (for title).
        plots_dir: Output directory for plots.
    """
    plt = _require_matplotlib()
    from .collect_realworld_dataset import _DEFAULT_BIN_EDGES

    is_positive = gt_rmse < t_good
    is_negative = gt_rmse > t_bad
    is_grey = (~is_positive) & (~is_negative)

    # --- 1. Sigma vs |gt - pred| scatter ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for mask, label, color in [
        (is_positive, f"good (<{t_good}px)", "green"),
        (is_grey, "grey zone", "orange"),
        (is_negative, f"bad (>{t_bad}px)", "red"),
    ]:
        if np.any(mask):
            ax.scatter(
                sigma[mask], abs_error[mask],
                s=4, alpha=0.3, color=color, label=label, linewidths=0,
            )
    # Reference line: sigma = abs_error (perfect calibration if a=1)
    lim = max(float(np.percentile(sigma, 99)), float(np.percentile(abs_error, 99)))
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="σ = |error|")
    ax.set_xlabel("σ (model uncertainty)")
    ax.set_ylabel("|gt_rmse - pred_rmse|")
    ax.set_title(f"Sigma vs actual error\nSpearman ρ={spearman_rho:.3f}  p={spearman_p:.2e}")
    ax.legend(fontsize=7, markerscale=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "sigma_vs_error.png", dpi=150)
    plt.close(fig)

    # --- 2. Sigma distribution: raw vs recalibrated ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    clip_pct = float(np.percentile(sigma_recal_q, 99.5))
    for val, label, color in [
        (sigma, "raw σ", "steelblue"),
        (sigma_recal_std, f"σ × {a_std:.2f} (std recal)", "darkorange"),
        (sigma_recal_q, f"σ × {a_q:.2f} (tail recal)", "green"),
    ]:
        axes[0].hist(
            np.clip(val, 0, float(np.percentile(sigma_recal_std, 99.5))),
            bins=60, density=True, alpha=0.55, label=label, color=color,
        )
        axes[1].hist(
            np.clip(val, 0, clip_pct),
            bins=60, density=True, alpha=0.55, label=label, color=color,
        )
    axes[0].set_title("Sigma distribution (std-scale view)")
    axes[1].set_title("Sigma distribution (tail-scale view)")
    for ax in axes:
        ax.set_xlabel("σ (pixels)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "sigma_distribution.png", dpi=150)
    plt.close(fig)

    # --- 3. Per GT-RMSE bin: mean sigma vs mean |gt - pred| ---
    bin_edges = _DEFAULT_BIN_EDGES
    bin_ids = np.digitize(gt_rmse, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    bin_labels, bin_sigma_mean, bin_err_mean, bin_counts = [], [], [], []
    for b in range(n_bins):
        in_bin = bin_ids == b
        if not np.any(in_bin):
            continue
        lo, hi = bin_edges[b], bin_edges[b + 1]
        label = f"[{lo:.0f},{hi:.0f})" if np.isfinite(hi) else f"[{lo:.0f},∞)"
        bin_labels.append(label)
        bin_sigma_mean.append(float(np.mean(sigma[in_bin])))
        bin_err_mean.append(float(np.mean(abs_error[in_bin])))
        bin_counts.append(int(np.sum(in_bin)))

    x = np.arange(len(bin_labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.35
    bars1 = ax.bar(x - width / 2, bin_sigma_mean, width, label="mean σ (raw)", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, bin_err_mean, width, label="mean |gt-pred|", color="tomato", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pixels")
    ax.set_title("Mean σ vs mean |gt-pred| per GT RMSE bin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    for bar, count in zip(bars1, bin_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"n={count}", ha="center", va="bottom", fontsize=6,
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "sigma_per_bin.png", dpi=150)
    plt.close(fig)


def plot_conformal_residuals(
    z_cal: np.ndarray,
    q_hat: float,
    alpha: float,
    out_path: Path,
) -> None:
    """Histogram of conformal non-conformity scores on calibration set.

    Args:
        z_cal: (M,) non-conformity scores on calibration split.
        q_hat: Fitted conformal quantile (marked with vertical line).
        alpha: Target miscoverage (for axis label).
        out_path: Output PNG path.
    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(z_cal, bins=60, density=True, alpha=0.7, color="steelblue", label="Non-conformity scores")
    ax.axvline(q_hat, color="red", linewidth=1.5, label=f"q̂={q_hat:.3f} (α={alpha})")
    # Overlay N(0,1) for reference
    x_range = np.linspace(float(np.min(z_cal)), float(np.max(z_cal)), 200)
    ax.plot(x_range, scipy.stats.norm.pdf(x_range), "k--", linewidth=1, alpha=0.6, label="N(0,1)")
    ax.set_xlabel("Non-conformity score r = (gt - pred) / σ")
    ax.set_ylabel("Density")
    ax.set_title(f"Conformal calibration non-conformity scores\nn_cal={len(z_cal)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sigma recalibration
# ---------------------------------------------------------------------------


def compute_sigma_scale_factors(
    z_cal: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """Compute two sigma recalibration factors from uncensored calibration residuals.

    Factor a_std (temperature scaling):
        a_std = std(z_cal)
        Makes the overall variance of (gt-pred)/sigma_recal ≈ 1.
        Appropriate if the distribution is roughly Gaussian but scaled.

    Factor a_q (tail-specific / quantile matching):
        a_q = empirical_(1-alpha)_quantile(z_cal) / Φ⁻¹(1-alpha)
        Directly targets the operating point we care about: after recalibration,
        the (1-alpha) percentile of standardized residuals matches the Gaussian.
        More appropriate when controlling a specific FPR threshold.

    Both are fitted on the calibration split of real-world data (not synthetic),
    so they correct for domain gap as well as the model's systematic overconfidence.

    Args:
        z_cal: (M,) standardized residuals (gt-pred)/sigma on uncensored cal samples.
        alpha: Target false-positive rate (used for a_q).

    Returns:
        (a_std, a_q): Scale factors. sigma_recal = a * sigma.
    """
    a_std = float(np.std(z_cal))
    theoretical_q = float(scipy.stats.norm.ppf(1.0 - alpha))
    empirical_q = float(np.quantile(z_cal, 1.0 - alpha))
    a_q = empirical_q / theoretical_q
    return a_std, a_q


def plot_qq_recalibrated(
    z_raw: np.ndarray,
    z_std: np.ndarray,
    z_q: np.ndarray,
    a_std: float,
    a_q: float,
    out_path: Path,
) -> None:
    """Q-Q plot comparing raw and recalibrated standardized residuals against N(0,1).

    Args:
        z_raw: (M,) raw standardized residuals (gt-pred)/sigma.
        z_std: (M,) std-recalibrated residuals z_raw / a_std.
        z_q: (M,) tail-recalibrated residuals z_raw / a_q.
        a_std: Std-based scale factor (for legend).
        a_q: Quantile-based scale factor (for legend).
        out_path: Output PNG path.
    """
    plt = _require_matplotlib()

    n = len(z_raw)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical = scipy.stats.norm.ppf(quantiles)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, z, title, color in [
        (axes[0], z_raw, "Raw σ (no recal)", "steelblue"),
        (axes[1], z_std, f"Std recal (÷{a_std:.2f})", "darkorange"),
        (axes[2], z_q, f"Tail recal (÷{a_q:.2f})", "green"),
    ]:
        empirical = np.sort(z)
        ax.scatter(theoretical, empirical, s=3, alpha=0.35, color=color)
        lo = min(float(theoretical[0]), float(empirical[0]))
        hi = max(float(theoretical[-1]), float(empirical[-1]))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ks = scipy.stats.kstest(z, "norm")
        ax.set_title(f"{title}\nKS={ks.statistic:.3f}  p={ks.pvalue:.4f}")
        ax.set_xlabel("Theoretical N(0,1)")
        ax.set_ylabel("Empirical quantile")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Q-Q plots: raw vs recalibrated residuals  (n={n})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_operating_point_table(
    methods: List[Dict],
    target_fprs: List[float],
) -> None:
    """Print a formatted operating point comparison table to stdout.

    Args:
        methods: List of dicts, each with keys: name, roc (dict with fpr/tpr/thresholds).
        target_fprs: FPR targets to evaluate at.
    """
    col_method = 22
    col_param = 10
    col_fpr_t = 10
    col_fpr_a = 10
    col_tpr = 8

    header = (
        f"{'Method':<{col_method}}"
        f"{'FPR_target':>{col_fpr_t}}"
        f"{'FPR_actual':>{col_fpr_a}}"
        f"{'TPR':>{col_tpr}}"
    )
    print()
    print(header)
    print("-" * len(header))

    for m in methods:
        roc = m["roc"]
        for tgt_fpr in target_fprs:
            op = get_operating_point(roc["fpr"], roc["tpr"], roc["thresholds"], tgt_fpr)
            tpr_str = f"{op['achieved_tpr']:.4f}" if not math.isnan(op["achieved_tpr"]) else "N/A"
            fpr_str = f"{op['achieved_fpr']:.5f}" if not math.isnan(op["achieved_fpr"]) else "N/A"
            print(
                f"{m['name']:<{col_method}}"
                f"{tgt_fpr:>{col_fpr_t}.3f}"
                f"{fpr_str:>{col_fpr_a}}"
                f"{tpr_str:>{col_tpr}}"
            )
    print()


# ---------------------------------------------------------------------------
# Main analysis orchestration
# ---------------------------------------------------------------------------


def run_full_analysis(
    pred_rmse_test: np.ndarray,
    log_sigma_test: np.ndarray,
    gt_rmse_test: np.ndarray,
    pred_rmse_cal: np.ndarray,
    log_sigma_cal: np.ndarray,
    gt_rmse_cal: np.ndarray,
    config: AnalysisConfig,
    output_dir: Path,
) -> dict:
    """Run all calibration, ROC, and conformal analyses and save outputs.

    Calibration factors (a_std, a_q, q_hat) are fitted on the calibration split.
    All diagnostic plots and ROC curves are computed on the test split.

    Args:
        pred_rmse_test: (N,) predicted RMSE on test split.
        log_sigma_test: (N,) log-sigma on test split.
        gt_rmse_test: (N,) ground truth RMSE on test split (unclipped).
        pred_rmse_cal: (M,) predicted RMSE on calibration split.
        log_sigma_cal: (M,) log-sigma on calibration split.
        gt_rmse_cal: (M,) ground truth RMSE on calibration split (unclipped).
        config: Analysis hyper-parameters.
        output_dir: Root output directory.

    Returns:
        Summary dict also written as results.json.
    """
    # Aliases for clarity — all diagnostics run on the test split
    pred_rmse = pred_rmse_test
    log_sigma = log_sigma_test
    gt_rmse = gt_rmse_test

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(gt_rmse)
    grey_zone_mask = (gt_rmse >= config.t_good) & (gt_rmse <= config.t_bad)
    n_grey = int(np.sum(grey_zone_mask))
    print(
        f"\n[Analysis] Test split N={n_total}  good (<{config.t_good}px): {int(np.sum(gt_rmse < config.t_good))}  "
        f"bad (>{config.t_bad}px): {int(np.sum(gt_rmse > config.t_bad))}  "
        f"grey zone: {n_grey} ({100*n_grey/n_total:.1f}%)"
    )
    print(
        f"[Analysis] Cal split  N={len(gt_rmse_cal)}  good: {int(np.sum(gt_rmse_cal < config.t_good))}  "
        f"bad: {int(np.sum(gt_rmse_cal > config.t_bad))}"
    )

    # ------------------------------------------------------------------
    # 1. Calibration analysis (on test split)
    # ------------------------------------------------------------------
    print("[Analysis] Computing calibration metrics on test split...")
    z, censor_mask = compute_standardized_residuals(
        pred_rmse, log_sigma, gt_rmse, censor_threshold=config.censor_threshold
    )
    n_uncensored = int(np.sum(censor_mask))
    n_censored_excluded = n_total - n_uncensored

    ks_result = scipy.stats.kstest(z, "norm")
    ks_stat = float(ks_result.statistic)
    ks_pvalue = float(ks_result.pvalue)
    print(f"  KS test: stat={ks_stat:.4f}  p={ks_pvalue:.6f}  (n_uncensored={n_uncensored})")

    plot_qq(
        z,
        plots_dir / "qq_plot.png",
        ks_stat=ks_stat,
        ks_pvalue=ks_pvalue,
        censor_excluded=n_censored_excluded,
        total=n_total,
    )

    cal_curve = compute_calibration_curve(
        pred_rmse, log_sigma, gt_rmse,
        tau=config.tau_bad,
        n_bins=config.calibration_n_bins,
    )
    ece = compute_ece(cal_curve)
    print(f"  ECE (at τ={config.tau_bad}px): {ece:.4f}")
    plot_calibration_diagram(
        cal_curve, ece, tau=config.tau_bad, out_path=plots_dir / "calibration_diagram.png"
    )

    # ------------------------------------------------------------------
    # 2. Sigma recalibration — fitted on calibration split
    # ------------------------------------------------------------------
    print("[Analysis] Fitting sigma recalibration factors on calibration split...")
    z_cal, cal_censor_mask = compute_standardized_residuals(
        pred_rmse_cal,
        log_sigma_cal,
        gt_rmse_cal,
        censor_threshold=config.censor_threshold,
    )
    n_cal_uncensored = int(np.sum(cal_censor_mask))

    a_std, a_q = compute_sigma_scale_factors(z_cal, config.alpha)
    theoretical_q = float(scipy.stats.norm.ppf(1.0 - config.alpha))
    print(f"  Calibration split: {len(gt_rmse_cal)} samples ({n_cal_uncensored} uncensored)")
    q_hat = float(np.quantile(z_cal, 1.0 - config.alpha))  # (1-alpha) quantile of z_cal
    print(f"  a_std = {a_std:.4f}  (sigma was {a_std:.1f}× too small on average, temperature scaling)")
    print(f"  empirical (1-α={config.alpha}) quantile of z_cal = {q_hat:.4f}")
    print(f"  a_q = {a_q:.4f}  (sigma was {a_q:.1f}× too small at the {(1-config.alpha)*100:.1f}th percentile, tail-specific)")

    # Apply recalibration factors to test split sigma
    sigma = np.exp(np.clip(log_sigma, -5.0, 5.0))
    sigma_recal_std = a_std * sigma
    sigma_recal_q = a_q * sigma

    # Sigma diagnostics
    abs_error = np.abs(gt_rmse - pred_rmse)
    # Spearman correlation on uncensored samples only (sigma was trained there)
    spearman_result = scipy.stats.spearmanr(sigma[censor_mask], abs_error[censor_mask])
    spearman_rho = float(spearman_result.statistic)
    spearman_p = float(spearman_result.pvalue)
    print(f"  Spearman ρ(sigma, |error|) on uncensored = {spearman_rho:.4f}  p={spearman_p:.2e}")

    plot_sigma_diagnostics(
        sigma=sigma,
        sigma_recal_std=sigma_recal_std,
        sigma_recal_q=sigma_recal_q,
        abs_error=abs_error,
        gt_rmse=gt_rmse,
        pred_rmse=pred_rmse,
        t_good=config.t_good,
        t_bad=config.t_bad,
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        a_std=a_std,
        a_q=a_q,
        plots_dir=plots_dir,
    )

    # Q-Q comparison: raw vs both recalibrations (on full uncensored set for visual)
    z_std_recal = z / a_std
    z_q_recal = z / a_q
    plot_qq_recalibrated(
        z_raw=z,
        z_std=z_std_recal,
        z_q=z_q_recal,
        a_std=a_std,
        a_q=a_q,
        out_path=plots_dir / "qq_recalibrated.png",
    )

    # Calibration curve after std recalibration (ECE check)
    log_sigma_recal_std = np.log(sigma_recal_std)
    cal_curve_recal = compute_calibration_curve(
        pred_rmse, log_sigma_recal_std, gt_rmse,
        tau=config.tau_bad,
        n_bins=config.calibration_n_bins,
    )
    ece_recal = compute_ece(cal_curve_recal)
    print(f"  ECE after std recalibration: {ece_recal:.4f}  (was {ece:.4f})")
    plot_calibration_diagram(
        cal_curve_recal, ece_recal, tau=config.tau_bad,
        out_path=plots_dir / "calibration_diagram_recal_std.png",
    )

    # Distribution of calibration residuals (diagnostic for how far sigma is off)
    plot_conformal_residuals(z_cal, q_hat, config.alpha, plots_dir / "sigma_residuals_cal.png")

    # ------------------------------------------------------------------
    # 3. ROC analysis: raw sigma + recalibrated sigma variants
    # ------------------------------------------------------------------
    print("[Analysis] Computing ROC curves...")

    methods = []

    # Rule A: score = -pred_rmse  (no sigma)
    score_a = -pred_rmse
    roc_a = sweep_roc_for_score(score_a, gt_rmse, config.t_good, config.t_bad, config.n_roc_thresholds)
    methods.append({"name": "A: -pred_rmse", "roc": roc_a})

    # Rule B: score = -(pred + k·sigma_raw) for requested k values
    for k in config.k_values:
        score_b = -(pred_rmse + k * sigma)
        roc_b = sweep_roc_for_score(score_b, gt_rmse, config.t_good, config.t_bad, config.n_roc_thresholds)
        methods.append({"name": f"B_raw: k={k:.2f}", "roc": roc_b})

    # Rule B_std: use std-recalibrated sigma, fixed k = Φ⁻¹(1-α)
    # Equivalent to: k_eff = Φ⁻¹(1-α) * a_std on raw sigma
    k_recal = theoretical_q  # theoretically correct k after recalibration
    score_b_std = -(pred_rmse + k_recal * sigma_recal_std)
    roc_b_std = sweep_roc_for_score(
        score_b_std, gt_rmse, config.t_good, config.t_bad, config.n_roc_thresholds
    )
    methods.append({"name": f"B_recal_std: k={k_recal:.2f}·σ_std", "roc": roc_b_std})

    # Rule B_q: use tail-recalibrated sigma, fixed k = Φ⁻¹(1-α)
    # Recalibration factors were fitted on cal split; evaluate on test split.
    score_b_q = -(pred_rmse + k_recal * sigma_recal_q)
    roc_b_q = sweep_roc_for_score(
        score_b_q,
        gt_rmse,
        config.t_good,
        config.t_bad,
        config.n_roc_thresholds,
    )
    methods.append({"name": f"B_recal_q: k={k_recal:.2f}·σ_q", "roc": roc_b_q})

    plot_roc_curves(
        [{"name": m["name"], "fpr": m["roc"]["fpr"], "tpr": m["roc"]["tpr"], "auc": m["roc"]["auc"]}
         for m in methods],
        target_fprs=config.target_fprs,
        out_path=plots_dir / "roc_curves.png",
    )

    # Score distribution plot for representative method (B_recal_std)
    score_rep = score_b_std
    plot_score_distributions(
        score_rep[gt_rmse < config.t_good],
        score_rep[grey_zone_mask],
        score_rep[gt_rmse > config.t_bad],
        rule_name=f"-(pred + {k_recal:.2f}·σ_std_recal)  [B_recal_std acceptance score]",
        out_path=plots_dir / "score_distributions.png",
    )

    # ------------------------------------------------------------------
    # 4. Print operating point table
    # ------------------------------------------------------------------
    print_operating_point_table(methods, config.target_fprs)

    # Fixed operating points at tau_bad for the recalibrated rules (on test split)
    op_b_std_fixed = compute_single_operating_point(
        score_b_std, gt_rmse, config.t_good, config.t_bad, score_threshold=-config.tau_bad
    )
    op_b_q_fixed = compute_single_operating_point(
        score_b_q, gt_rmse, config.t_good, config.t_bad, score_threshold=-config.tau_bad
    )

    # ------------------------------------------------------------------
    # 5. Assemble and save results
    # ------------------------------------------------------------------
    def _roc_to_json_safe(roc: dict) -> dict:
        # Downsample to 100 points for compactness
        idx = np.linspace(0, len(roc["fpr"]) - 1, 100, dtype=int)
        return {
            "fpr": roc["fpr"][idx].tolist(),
            "tpr": roc["tpr"][idx].tolist(),
            "auc": float(roc["auc"]),
            "n_pos": int(roc["n_pos"]),
            "n_neg": int(roc["n_neg"]),
        }

    methods_json = []
    for m in methods:
        operating_points = {}
        for tgt in config.target_fprs:
            op = get_operating_point(
                m["roc"]["fpr"], m["roc"]["tpr"], m["roc"]["thresholds"], tgt
            )
            operating_points[f"fpr_target_{tgt}"] = op
        methods_json.append({
            "name": m["name"],
            "roc": _roc_to_json_safe(m["roc"]),
            "operating_points": operating_points,
        })

    summary = {
        "config": {
            "t_good": config.t_good,
            "t_bad": config.t_bad,
            "tau_bad": config.tau_bad,
            "alpha": config.alpha,
            "k_values": config.k_values,
            "censor_threshold": config.censor_threshold,
        },
        "dataset": {
            "n_test": n_total,
            "n_cal": int(len(gt_rmse_cal)),
            "n_good": int(np.sum(gt_rmse < config.t_good)),
            "n_bad": int(np.sum(gt_rmse > config.t_bad)),
            "n_grey": n_grey,
            "grey_zone_fraction": float(n_grey / n_total),
            "n_uncensored_for_calibration": n_uncensored,
            "n_censored_excluded": n_censored_excluded,
        },
        "calibration": {
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "ece": ece,
            "ece_after_std_recal": ece_recal,
            "tau_for_ece": config.tau_bad,
        },
        "sigma_recalibration": {
            "a_std": float(a_std),
            "a_q": float(a_q),
            "spearman_rho_sigma_vs_error": spearman_rho,
            "spearman_p": spearman_p,
            "n_cal_uncensored": n_cal_uncensored,
            "note": (
                "a_std: temperature scaling (std of z_cal); "
                "a_q: tail-specific (empirical quantile / Gaussian quantile). "
                "Both fitted on cal split; diagnostics/ROC evaluated on test split."
            ),
        },
        "fixed_operating_points": {
            "B_recal_std_at_tau_bad": op_b_std_fixed,
            "B_recal_q_at_tau_bad": op_b_q_fixed,
        },
        "z_cal_quantile_at_alpha": float(q_hat),
        "theoretical_gaussian_quantile": float(theoretical_q),
        "methods": methods_json,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Analysis] Results saved to {output_dir}")
    print(f"  results.json : {results_path}")
    print(f"  plots/       : {plots_dir}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate binary decision rules for homography quality acceptance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input modes
    input_group = parser.add_argument_group("Input (choose one mode)")
    input_group.add_argument(
        "--cal-npz",
        type=Path,
        default=None,
        help=(
            "Calibration split NPZ (e.g. from split_10000). "
            "Recalibration factors are fitted on this split. Use with --test-npz."
        ),
    )
    input_group.add_argument(
        "--test-npz",
        type=Path,
        default=None,
        help=(
            "Test split NPZ (e.g. from split_100000). "
            "All diagnostic plots and ROC curves are evaluated on this split. Use with --cal-npz."
        ),
    )
    input_group.add_argument(
        "--npz",
        type=Path,
        default=None,
        help=(
            "Legacy single-NPZ mode: internal 50/50 split is used. "
            "Prefer --cal-npz + --test-npz for sequence-disjoint splits."
        ),
    )
    input_group.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Collected real-world dataset directory (inference mode).",
    )
    input_group.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint file or run directory (inference mode).",
    )
    input_group.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit inference to this many samples (inference mode only).",
    )
    input_group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="DataLoader batch size (inference mode only).",
    )
    input_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (inference mode only).",
    )
    input_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string, e.g. 'cuda' or 'cpu' (inference mode only).",
    )
    input_group.add_argument(
        "--gt-rmse-clip",
        type=float,
        default=30.0,
        help="GT RMSE clip for regression evaluation (inference mode only).",
    )

    # Decision rule parameters
    rule_group = parser.add_argument_group("Decision rule parameters")
    rule_group.add_argument(
        "--t-good",
        type=float,
        default=12.0,
        help=(
            "GT RMSE threshold below which a homography is 'good' (positive class). "
            "Default 12px accounts for PlanarTrack GT noise floor (~5.71px mean annotation error)."
        ),
    )
    rule_group.add_argument(
        "--t-bad",
        type=float,
        default=25.0,
        help=(
            "GT RMSE threshold above which a homography is 'bad' (negative class). "
            "Default 25px avoids contamination from the 10.71% of GT annotations with error >15px."
        ),
    )
    rule_group.add_argument(
        "--tau-bad",
        type=float,
        default=25.0,
        help="Acceptance boundary used in decision rules B/C/D (pixels). Should match --t-bad.",
    )
    rule_group.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Target false-positive rate for rules C and D.",
    )
    rule_group.add_argument(
        "--k-values",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 3.09],
        help="k multipliers for rule B (UCB).",
    )

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis parameters")
    analysis_group.add_argument(
        "--cal-fraction",
        type=float,
        default=0.5,
        help="Fraction of data for internal calibration split (legacy --npz and inference modes only).",
    )
    analysis_group.add_argument("--seed", type=int, default=42)
    analysis_group.add_argument(
        "--n-roc-thresholds",
        type=int,
        default=500,
        help="Number of threshold values to sweep for ROC curves.",
    )
    analysis_group.add_argument(
        "--target-fprs",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05],
        help="FPR values to report in the operating point table.",
    )
    analysis_group.add_argument(
        "--calibration-n-bins",
        type=int,
        default=20,
        help="Number of bins for the reliability diagram.",
    )
    analysis_group.add_argument(
        "--censor-threshold",
        type=float,
        default=30.0,
        help="GT RMSE values above this are excluded from calibration analysis.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for all output files.",
    )

    args = parser.parse_args()

    # Validate input mode
    using_cal_test = args.cal_npz is not None or args.test_npz is not None
    using_single_npz = args.npz is not None
    using_inference = args.dataset_dir is not None or args.checkpoint is not None

    n_modes = sum([using_cal_test, using_single_npz, using_inference])
    if n_modes > 1:
        parser.error(
            "Specify exactly one input mode: (--cal-npz + --test-npz), --npz, "
            "or (--dataset-dir + --checkpoint)."
        )
    if n_modes == 0:
        parser.error(
            "Specify one input mode: (--cal-npz + --test-npz), --npz, "
            "or (--dataset-dir + --checkpoint)."
        )
    if using_cal_test and (args.cal_npz is None or args.test_npz is None):
        parser.error("Both --cal-npz and --test-npz are required together.")
    if using_inference and (args.dataset_dir is None or args.checkpoint is None):
        parser.error("Both --dataset-dir and --checkpoint are required for inference mode.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AnalysisConfig(
        t_good=args.t_good,
        t_bad=args.t_bad,
        tau_bad=args.tau_bad,
        alpha=args.alpha,
        k_values=args.k_values,
        cal_fraction=args.cal_fraction,
        seed=args.seed,
        n_roc_thresholds=args.n_roc_thresholds,
        target_fprs=args.target_fprs,
        calibration_n_bins=args.calibration_n_bins,
        censor_threshold=args.censor_threshold,
    )

    # Load predictions
    if using_cal_test:
        print(f"[Loading] Cal NPZ:  {args.cal_npz}")
        data_cal = load_predictions_from_npz(args.cal_npz)
        print(f"[Loading] Test NPZ: {args.test_npz}")
        data_test = load_predictions_from_npz(args.test_npz)
        pred_rmse_cal = data_cal["pred_rmse"].astype(np.float64)
        log_sigma_cal = data_cal["log_sigma"].astype(np.float64)
        gt_rmse_cal = data_cal["gt_rmse"].astype(np.float64)
        pred_rmse_test = data_test["pred_rmse"].astype(np.float64)
        log_sigma_test = data_test["log_sigma"].astype(np.float64)
        gt_rmse_test = data_test["gt_rmse"].astype(np.float64)
    elif using_single_npz:
        print(f"[Loading] NPZ from {args.npz}  (legacy mode: internal {args.cal_fraction:.0%}/{1-args.cal_fraction:.0%} split)")
        data = load_predictions_from_npz(args.npz)
        pred_all = data["pred_rmse"].astype(np.float64)
        log_sigma_all = data["log_sigma"].astype(np.float64)
        gt_rmse_all = data["gt_rmse"].astype(np.float64)
        cal_indices, test_indices = split_calibration_test(gt_rmse_all, args.cal_fraction, args.seed)
        pred_rmse_cal = pred_all[cal_indices]
        log_sigma_cal = log_sigma_all[cal_indices]
        gt_rmse_cal = gt_rmse_all[cal_indices]
        pred_rmse_test = pred_all[test_indices]
        log_sigma_test = log_sigma_all[test_indices]
        gt_rmse_test = gt_rmse_all[test_indices]
    else:
        print(f"[Loading] Running inference: {args.checkpoint} on {args.dataset_dir}")
        inference_dir = args.output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        data = load_predictions_via_inference(
            dataset_dir=args.dataset_dir,
            checkpoint_path=args.checkpoint,
            output_dir=inference_dir,
            subset_size=args.subset_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_str=args.device,
            gt_rmse_clip=args.gt_rmse_clip,
        )
        pred_all = data["pred_rmse"].astype(np.float64)
        log_sigma_all = data["log_sigma"].astype(np.float64)
        gt_rmse_all = data["gt_rmse"].astype(np.float64)
        cal_indices, test_indices = split_calibration_test(gt_rmse_all, args.cal_fraction, args.seed)
        pred_rmse_cal = pred_all[cal_indices]
        log_sigma_cal = log_sigma_all[cal_indices]
        gt_rmse_cal = gt_rmse_all[cal_indices]
        pred_rmse_test = pred_all[test_indices]
        log_sigma_test = log_sigma_all[test_indices]
        gt_rmse_test = gt_rmse_all[test_indices]

    run_full_analysis(
        pred_rmse_test, log_sigma_test, gt_rmse_test,
        pred_rmse_cal, log_sigma_cal, gt_rmse_cal,
        config, args.output_dir,
    )


if __name__ == "__main__":
    main()
