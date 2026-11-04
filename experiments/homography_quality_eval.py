#!/usr/bin/env python3
"""Evaluate homography quality estimation against GT alignment errors.

Loads pre-computed masks and corner predictions, computes the mask-based
homography quality score per frame, and compares it to the ground-truth
alignment error.

Usage examples:
    # Evaluate on POT-210 with default config
    python experiments/homography_quality_eval.py pot

    # Evaluate specific sequences
    python experiments/homography_quality_eval.py pot --sequences seq1 seq2

    # Use a specific tracker config (for directory structure)
    python experiments/homography_quality_eval.py planartrack --config configs/my_config.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from einops import rearrange
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flatsam2.config import compute_output_dir
from flatsam2.datasets import (
    CLI_CHOICES,
    DATASETS,
    get_sequences,
    pot_load_gt,
    pt_load_gt,
    pti_load_gt,
)
from flatsam2.homography_quality import (
    HomographyExplanation,
    ReferenceContour,
    assess_homography,
    homography_from_corners,
)
from flatsam2.metrics import alignment_error
from flatsam2.path_config import paths


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _make_load_gt_fn(dataset: str, initfix_dir: Optional[Path]):
    loader = DATASETS[dataset].gt_loader
    if loader == "pot":
        return pot_load_gt
    if loader == "pt":
        return pt_load_gt
    if initfix_dir is None:
        raise ValueError("initfix_dir must be provided for the 'pti' GT loader")
    return lambda bd, sn: pti_load_gt(bd, sn, initfix_dir)


def load_predictions(results_dir: Path, seq_name: str) -> np.ndarray:
    """Load predicted corners (N, 2, 4) from a results directory.

    Args:
        results_dir: Directory that contains ``<seq_name>.txt`` files.
        seq_name: Sequence name (filename without extension).

    Returns:
        Float32 array of shape (N_frames, 2, 4).

    Raises:
        FileNotFoundError: If the prediction file does not exist.
    """
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(pred_flat, 'frames (corners xy) -> frames xy corners', corners=4, xy=2)


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """Load a grayscale mask image.  Returns None if the file does not exist.

    Args:
        mask_path: Absolute path to the mask PNG.

    Returns:
        uint8 grayscale array or None.
    """
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence_quality(
    seq_name: str,
    gt_corners: np.ndarray,
    pred_corners: np.ndarray,
    valid: np.ndarray,
    masks_dir: Path,
) -> Dict:
    """Evaluate homography quality estimation for one sequence.

    Phase 0 (ReferenceContour) is built once from the init mask and reused
    for every frame.

    Args:
        seq_name: Human-readable sequence identifier.
        gt_corners: (N, 2, 4) ground-truth corner coordinates.
        pred_corners: (N, 2, 4) predicted corner coordinates.
        valid: (N,) bool array — True for frames with valid GT.
        masks_dir: Directory containing ``00001.png``, ``00002.png``, … masks.

    Returns:
        Dict with per-frame results and aggregate statistics.

    Raises:
        FileNotFoundError: If the init mask is missing.
    """
    init_mask_path = masks_dir / f"{1:05d}.png"
    init_mask = load_mask(init_mask_path)
    if init_mask is None:
        raise FileNotFoundError(f"Init mask not found: {init_mask_path}")

    # Phase 0: build reference contour once
    ref = ReferenceContour.build(init_mask)
    init_corners = gt_corners[0]  # (2, 4)
    n_frames = len(gt_corners)
    frame_results = []

    for frame_idx in range(1, n_frames):
        if not valid[frame_idx]:
            frame_results.append(None)
            continue

        seg_mask = load_mask(masks_dir / f"{frame_idx + 1:05d}.png")
        if seg_mask is None:
            frame_results.append(None)
            continue

        H = homography_from_corners(init_corners, pred_corners[frame_idx])
        if H is None:
            frame_results.append(None)
            continue

        h, w = seg_mask.shape[:2]
        result: HomographyExplanation = assess_homography(ref, seg_mask, H, (h, w))
        gt_error = float(alignment_error(gt_corners[frame_idx], pred_corners[frame_idx]))

        frame_results.append({
            'frame_idx': frame_idx,
            'confidence': result.confidence,
            'precision': result.precision,
            'alignment_loss': result.alignment_loss,
            'frac_aligned_A': result.frac_aligned_A,
            'frac_occluded_A': result.frac_occluded_A,
            'frac_interior_A': result.frac_interior_A,
            'frac_out_of_frame_A': result.frac_out_of_frame_A,
            'frac_aligned_B': result.frac_aligned_B,
            'frac_excess_B': result.frac_excess_B,
            'frac_occl_edge_B': result.frac_occl_edge_B,
            'frac_frame_edge_B': result.frac_frame_edge_B,
            'gt_error': gt_error,
            'explanation': result.explanation,
            # Non-serialised full result kept for optional in-process use
            '_result': result,
        })

    valid_frames = [f for f in frame_results if f is not None]
    confidence_arr = np.array([f['confidence'] for f in valid_frames])
    gt_errors = np.array([f['gt_error'] for f in valid_frames])

    metrics = _compute_metrics(confidence_arr, gt_errors)

    return {
        'sequence': seq_name,
        'num_frames_evaluated': len(valid_frames),
        **metrics,
        'mean_confidence': float(np.mean(confidence_arr)) if len(confidence_arr) > 0 else float('nan'),
        'mean_gt_error': float(np.mean(gt_errors)) if len(gt_errors) > 0 else float('nan'),
        'frame_results': valid_frames,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

#: GT error threshold separating "correct" from "wrong" homography (pixels).
GT_ERROR_THRESHOLD: float = 15.0

#: Confidence threshold for binary good/bad prediction.
CONF_THRESHOLD: float = 0.5


def _compute_metrics(
    confidence_arr: np.ndarray,
    gt_errors: np.ndarray,
) -> Dict:
    """Compute the two-tier evaluation metrics.

    Tier 1 — Gross regime correctness (binary classification).
    The assessor must reliably separate "homography wrong" (GT error ≥ 15 px)
    from "homography ok" (GT error < 15 px).  We report TPR and TNR
    separately: a system that only ever predicts "good" gets TPR=1, TNR=0,
    which is useless but would score well on plain accuracy.

    Tier 2 — Fine regime ranking (within ok frames only).
    Among frames where the homography is actually correct (GT error < 15 px),
    higher confidence should mean lower GT error.  Spearman r is computed
    only on this subset so that variation among grossly-wrong frames does not
    pollute the ranking metric.

    Args:
        confidence_arr: (N,) confidence scores.
        gt_errors: (N,) GT alignment errors in pixels.

    Returns:
        Dict with all metric fields.
    """
    nan = float('nan')

    if len(confidence_arr) == 0:
        return {
            'tpr': nan, 'tnr': nan, 'balanced_accuracy': nan,
            'spearman_r_good': nan, 'spearman_p_good': nan,
            'n_good_frames': 0, 'n_bad_frames': 0,
        }

    gt_good = gt_errors < GT_ERROR_THRESHOLD   # (N,) bool
    pred_good = confidence_arr >= CONF_THRESHOLD

    n_gt_good = int(np.sum(gt_good))
    n_gt_bad = int(np.sum(~gt_good))

    # Tier 1 — TPR (sensitivity): of actually-good frames, fraction predicted good.
    # Tier 1 — TNR (specificity): of actually-bad frames, fraction predicted bad.
    tpr = float(np.sum(pred_good & gt_good) / n_gt_good) if n_gt_good > 0 else nan
    tnr = float(np.sum(~pred_good & ~gt_good) / n_gt_bad) if n_gt_bad > 0 else nan

    both_valid = not (np.isnan(tpr) or np.isnan(tnr))
    balanced_accuracy = float((tpr + tnr) / 2) if both_valid else nan

    # Tier 2 — Spearman r within the good regime only.
    good_conf = confidence_arr[gt_good]
    good_err = gt_errors[gt_good]
    if n_gt_good >= 3:
        # Negate gt_errors so that positive r = higher confidence → lower error
        spearman_r_good, spearman_p_good = stats.spearmanr(good_conf, -good_err)
    else:
        spearman_r_good = spearman_p_good = nan

    return {
        'tpr': tpr,
        'tnr': tnr,
        'balanced_accuracy': balanced_accuracy,
        'spearman_r_good': float(spearman_r_good),
        'spearman_p_good': float(spearman_p_good),
        'n_good_frames': n_gt_good,
        'n_bad_frames': n_gt_bad,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: str,
    base_dir: Path,
    base_results_dir: Path,
    sequences: List[str],
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> Dict:
    """Run homography quality evaluation on all sequences.

    Args:
        dataset: Dataset identifier (e.g. ``'pot'``).
        base_dir: Root data directory for the dataset.
        base_results_dir: Root directory containing tracker outputs.
        sequences: List of sequence names to evaluate.
        config_path: Optional tracker config path (used to compute output sub-dir).
        initfix_dir: Required for the ``'pti'`` GT loader.
        output_file: Where to write the JSON result file.

    Returns:
        Dict with ``aggregate``, ``all_results``, and ``failed`` entries.
    """
    load_gt_fn = _make_load_gt_fn(dataset, initfix_dir)
    results_dir = compute_output_dir(base_results_dir, dataset, config_path)

    if output_file is None:
        output_file = results_dir / "homography_quality_eval.json"

    print(f"Evaluating homography quality on {len(sequences)} sequences")
    print(f"Results dir: {results_dir}")

    all_results = []
    failed = []

    for seq_i, seq_name in enumerate(sequences):
        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            pred_corners = load_predictions(results_dir, seq_name)

            min_len = min(len(gt_corners), len(pred_corners))
            gt_corners = gt_corners[:min_len]
            pred_corners = pred_corners[:min_len]
            valid = valid[:min_len]

            masks_dir = results_dir / 'masks' / seq_name

            seq_result = evaluate_sequence_quality(
                seq_name, gt_corners, pred_corners, valid, masks_dir,
            )
            all_results.append(seq_result)
            print(
                f"  {seq_name}: "
                f"bal_acc={seq_result['balanced_accuracy']:.3f}  "
                f"tpr={seq_result['tpr']:.3f}  tnr={seq_result['tnr']:.3f}  "
                f"spearman_good={seq_result['spearman_r_good']:.3f}  "
                f"mean_conf={seq_result['mean_confidence']:.3f}  "
                f"mean_gt_err={seq_result['mean_gt_error']:.1f}px  "
                f"(good={seq_result['n_good_frames']}, bad={seq_result['n_bad_frames']})"
            )
        except Exception as e:
            import traceback
            print(f"  {seq_name}: FAILED — {e}")
            traceback.print_exc()
            failed.append((seq_name, str(e)))

        if seq_i > 10:
            print("Early exit for faster experimentation")
            break

    def _agg(key: str) -> float:
        vals = [r[key] for r in all_results if not np.isnan(r.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    if all_results:
        aggregate = {
            'num_sequences': len(all_results),
            'num_failed': len(failed),
            # Tier 1 — gross correctness
            'mean_balanced_accuracy': _agg('balanced_accuracy'),
            'mean_tpr': _agg('tpr'),
            'mean_tnr': _agg('tnr'),
            # Tier 2 — fine ranking within good frames
            'mean_spearman_r_good': _agg('spearman_r_good'),
            'median_spearman_r_good': float(
                np.median([r['spearman_r_good'] for r in all_results
                           if not np.isnan(r.get('spearman_r_good', float('nan')))])
            ) if any(not np.isnan(r.get('spearman_r_good', float('nan'))) for r in all_results)
            else float('nan'),
        }
    else:
        aggregate = {'num_sequences': 0, 'num_failed': len(failed)}

    print(f"\n{'='*60}")
    print(f"AGGREGATE: {aggregate['num_sequences']} sequences evaluated, "
          f"{aggregate.get('num_failed', 0)} failed")
    if 'mean_balanced_accuracy' in aggregate:
        print(f"  Balanced accuracy (TPR+TNR)/2 : {aggregate['mean_balanced_accuracy']:.3f}")
        print(f"  Mean TPR (sensitivity)        : {aggregate['mean_tpr']:.3f}")
        print(f"  Mean TNR (specificity)        : {aggregate['mean_tnr']:.3f}")
        print(f"  Mean Spearman r (good frames) : {aggregate['mean_spearman_r_good']:.3f}")
        print(f"  Median Spearman r (good frames): {aggregate['median_spearman_r_good']:.3f}")

    # Serialise (drop non-JSON-serialisable _result objects)
    serialisable = {
        'aggregate': aggregate,
        'sequences': [
            {k: v for k, v in r.items() if k != 'frame_results'}
            for r in all_results
        ],
        'per_frame': {
            r['sequence']: [
                {k: v for k, v in f.items() if k != '_result'}
                for f in r['frame_results']
            ]
            for r in all_results
        },
        'failed': [{'sequence': s, 'error': e} for s, e in failed],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"\nResults saved to: {output_file}")

    return {
        'aggregate': aggregate,
        'all_results': all_results,
        'failed': failed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate homography quality estimation against GT alignment errors"
    )
    parser.add_argument('dataset', choices=CLI_CHOICES)
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--sequences', nargs='+')
    parser.add_argument('--subset', help="Subset name from configs/subsets/")
    parser.add_argument('--split', choices=['train', 'test'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--initfix-dir', type=Path, default=None)
    parser.add_argument('--output', type=Path, help="Output JSON file")
    args = parser.parse_args(argv)

    ds_info = DATASETS[args.dataset]
    if args.base_dir is None:
        args.base_dir = getattr(paths, ds_info.base_dir_attr)
    if ds_info.gt_loader == "pti" and args.initfix_dir is None:
        args.initfix_dir = paths.PT_INITFIX_DIR
    if args.results_dir is None:
        args.results_dir = paths.OUTPUT_DIRECTORY

    try:
        sequences = get_sequences(
            args.dataset, args.base_dir,
            sequences=args.sequences,
            subset=args.subset,
            split=args.split,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    run_experiment(
        args.dataset, args.base_dir, args.results_dir, sequences,
        config_path=args.config,
        initfix_dir=args.initfix_dir,
        output_file=args.output,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
