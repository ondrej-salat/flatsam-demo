#!/usr/bin/env python3
"""Evaluate NN homography quality estimation against GT alignment errors.

Drop-in counterpart to homography_quality_eval.py that replaces the
hand-crafted assessor with a trained NNHomographyScorer checkpoint.

The two-tier metrics (TPR/TNR, Spearman rho) are identical so results
are directly comparable.

Usage examples:
    # Evaluate on POT-210 with a checkpoint
    python experiments/nn_quality_eval.py pot --checkpoint checkpoints/nn_quality/best.pt

    # Evaluate specific sequences
    python experiments/nn_quality_eval.py pot --checkpoint .../best.pt --sequences seq1 seq2

    # option_c architecture
    python experiments/nn_quality_eval.py pot --checkpoint .../best.pt --arch option_c
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from flatsam2.homography_quality import ReferenceContour, homography_from_corners
from flatsam2.metrics import alignment_error
from flatsam2.nn.inference import NNHomographyScorer
from flatsam2.path_config import paths

import cv2


# ---------------------------------------------------------------------------
# Constants (match homography_quality_eval.py)
# ---------------------------------------------------------------------------

#: GT error threshold separating "correct" from "wrong" homography (pixels).
GT_ERROR_THRESHOLD: float = 15.0


# ---------------------------------------------------------------------------
# Data loading helpers (copied from homography_quality_eval.py)
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
    """Load predicted corners (N, 2, 4) from a results directory."""
    pred_file = results_dir / f"{seq_name}.txt"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    pred_flat = np.loadtxt(pred_file)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[np.newaxis, :]
    return rearrange(pred_flat, 'frames (corners xy) -> frames xy corners', corners=4, xy=2)


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """Load a grayscale mask image. Returns None if the file does not exist."""
    if not mask_path.exists():
        return None
    return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence_nn(
    seq_name: str,
    gt_corners: np.ndarray,
    pred_corners: np.ndarray,
    valid: np.ndarray,
    masks_dir: Path,
    scorer: NNHomographyScorer,
) -> Dict:
    """Evaluate NN quality estimation for one sequence.

    Args:
        seq_name: Human-readable sequence identifier.
        gt_corners: (N, 2, 4) ground-truth corner coordinates.
        pred_corners: (N, 2, 4) predicted corner coordinates.
        valid: (N,) bool array — True for frames with valid GT.
        masks_dir: Directory containing ``00001.png``, ``00002.png``, … masks.
        scorer: Initialised NNHomographyScorer.

    Returns:
        Dict with per-frame results and aggregate statistics.
    """
    init_mask_path = masks_dir / f"{1:05d}.png"
    init_mask = load_mask(init_mask_path)
    if init_mask is None:
        raise FileNotFoundError(f"Init mask not found: {init_mask_path}")

    ref = ReferenceContour.build(init_mask)
    scorer.init_reference(ref)

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
        pred_rmse, sigma = scorer.score(seg_mask, H, (h, w))
        gt_error = float(alignment_error(gt_corners[frame_idx], pred_corners[frame_idx]))

        frame_results.append({
            'frame_idx': frame_idx,
            'predicted_rmse': pred_rmse,
            'sigma': sigma,
            'gt_error': gt_error,
        })

    valid_frames = [f for f in frame_results if f is not None]
    pred_rmse_arr = np.array([f['predicted_rmse'] for f in valid_frames])
    gt_errors = np.array([f['gt_error'] for f in valid_frames])

    metrics = _compute_metrics(pred_rmse_arr, gt_errors)

    return {
        'sequence': seq_name,
        'num_frames_evaluated': len(valid_frames),
        **metrics,
        'mean_predicted_rmse': float(np.mean(pred_rmse_arr)) if len(pred_rmse_arr) > 0 else float('nan'),
        'mean_gt_error': float(np.mean(gt_errors)) if len(gt_errors) > 0 else float('nan'),
        'frame_results': valid_frames,
    }


# ---------------------------------------------------------------------------
# Metrics (same two-tier framework as homography_quality_eval.py)
# ---------------------------------------------------------------------------

SPEARMAN_BIN_SIZE: float = 2.0  # px — bin width for binned Spearman correlation


def _compute_metrics(
    pred_rmse_arr: np.ndarray,
    gt_errors: np.ndarray,
) -> Dict:
    """Compute the two-tier evaluation metrics.

    Tier 1 — binary classification: pred_rmse < GT_ERROR_THRESHOLD → "good".
    Tier 2 — Spearman r on good frames, computed on values binned into
    SPEARMAN_BIN_SIZE-px bins so that within-bin ordering is ignored.

    Args:
        pred_rmse_arr: (N,) predicted RMSE values (lower = better).
        gt_errors: (N,) GT alignment errors in pixels.

    Returns:
        Dict with all metric fields.
    """
    nan = float('nan')

    if len(pred_rmse_arr) == 0:
        return {
            'tpr': nan, 'tnr': nan, 'balanced_accuracy': nan,
            'spearman_r_good': nan, 'spearman_p_good': nan,
            'n_good_frames': 0, 'n_bad_frames': 0,
        }

    gt_good = gt_errors < GT_ERROR_THRESHOLD
    pred_good = pred_rmse_arr < GT_ERROR_THRESHOLD

    n_gt_good = int(np.sum(gt_good))
    n_gt_bad = int(np.sum(~gt_good))

    tpr = float(np.sum(pred_good & gt_good) / n_gt_good) if n_gt_good > 0 else nan
    tnr = float(np.sum(~pred_good & ~gt_good) / n_gt_bad) if n_gt_bad > 0 else nan

    both_valid = not (np.isnan(tpr) or np.isnan(tnr))
    balanced_accuracy = float((tpr + tnr) / 2) if both_valid else nan

    good_pred = pred_rmse_arr[gt_good]
    good_err = gt_errors[gt_good]
    if n_gt_good >= 3:
        binned_pred = np.floor(good_pred / SPEARMAN_BIN_SIZE)
        binned_err  = np.floor(good_err  / SPEARMAN_BIN_SIZE)
        spearman_r_good, spearman_p_good = stats.spearmanr(-binned_pred, -binned_err)
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

def _build_aggregate(
    checkpoint: Path,
    arch: str,
    all_results: List[Dict],
    failed: List,
) -> Dict:
    def _agg(key: str) -> float:
        vals = [r[key] for r in all_results if not np.isnan(r.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    if not all_results:
        return {'checkpoint': str(checkpoint), 'arch': arch,
                'num_sequences': 0, 'num_failed': len(failed)}

    return {
        'checkpoint': str(checkpoint),
        'arch': arch,
        'num_sequences': len(all_results),
        'num_failed': len(failed),
        'mean_balanced_accuracy': _agg('balanced_accuracy'),
        'mean_tpr': _agg('tpr'),
        'mean_tnr': _agg('tnr'),
        'mean_spearman_r_good': _agg('spearman_r_good'),
        'median_spearman_r_good': float(
            np.median([r['spearman_r_good'] for r in all_results
                       if not np.isnan(r.get('spearman_r_good', float('nan')))])
        ) if any(not np.isnan(r.get('spearman_r_good', float('nan'))) for r in all_results)
        else float('nan'),
    }


def _save(
    output_file: Path,
    checkpoint: Path,
    arch: str,
    all_results: List[Dict],
    failed: List,
) -> None:
    """Atomically write the current state to output_file."""
    aggregate = _build_aggregate(checkpoint, arch, all_results, failed)
    serialisable = {
        'aggregate': aggregate,
        'sequences': [
            {k: v for k, v in r.items() if k != 'frame_results'}
            for r in all_results
        ],
        'per_frame': {
            r['sequence']: r['frame_results']
            for r in all_results
        },
        'failed': [{'sequence': s, 'error': e} for s, e in failed],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix('.tmp')
    with tmp.open('w') as fh:
        json.dump(serialisable, fh, indent=2)
    tmp.replace(output_file)  # atomic on POSIX


def run_experiment(
    dataset: str,
    base_dir: Path,
    base_results_dir: Path,
    sequences: List[str],
    checkpoint: Path,
    arch: str = "option_e",
    config_path: Optional[Path] = None,
    initfix_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    device: str = "cuda",
) -> Dict:
    """Run NN homography quality evaluation on all sequences.

    Saves results incrementally after each sequence so that an interrupted run
    can be resumed — already-computed sequences are skipped automatically.

    Args:
        dataset: Dataset identifier (e.g. ``'pot'``).
        base_dir: Root data directory for the dataset.
        base_results_dir: Root directory containing tracker outputs.
        sequences: List of sequence names to evaluate.
        checkpoint: Path to a trained checkpoint (.pt file).
        arch: Model architecture (``'option_e'`` or ``'option_c'``).
        config_path: Optional tracker config path (used to compute output sub-dir).
        initfix_dir: Required for the ``'pti'`` GT loader.
        output_file: Where to write the JSON result file.
        device: PyTorch device string.

    Returns:
        Dict with ``aggregate``, ``all_results``, and ``failed`` entries.
    """
    load_gt_fn = _make_load_gt_fn(dataset, initfix_dir)
    results_dir = compute_output_dir(base_results_dir, dataset, config_path)

    if output_file is None:
        output_file = results_dir / f"nn_quality_eval_{checkpoint.stem}.json"

    # --- Resume: load already-computed results ---
    all_results: List[Dict] = []
    failed: List = []
    done: set = set()

    if output_file.exists():
        try:
            with output_file.open() as fh:
                existing = json.load(fh)
            # Restore per-sequence summaries (frame_results not stored in 'sequences')
            for seq_summary in existing.get('sequences', []):
                seq_name = seq_summary['sequence']
                frame_results = existing.get('per_frame', {}).get(seq_name, [])
                all_results.append({**seq_summary, 'frame_results': frame_results})
                done.add(seq_name)
            for entry in existing.get('failed', []):
                failed.append((entry['sequence'], entry['error']))
                done.add(entry['sequence'])
            print(f"Resuming: {len(done)} sequences already done, "
                  f"{len(sequences) - len(done)} remaining.")
        except Exception as e:
            print(f"Warning: could not load existing results ({e}), starting fresh.")

    scorer = NNHomographyScorer(
        checkpoint_path=str(checkpoint),
        arch=arch,
        device=device,
    )
    print(f"Loaded checkpoint: {checkpoint}  arch={arch}  device={device}")
    print(f"Evaluating NN quality on {len(sequences)} sequences")
    print(f"Results dir: {results_dir}")

    for seq_name in sequences:
        if seq_name in done:
            print(f"  {seq_name}: skipped (already done)")
            continue

        try:
            gt_corners, valid = load_gt_fn(base_dir, seq_name)
            pred_corners = load_predictions(results_dir, seq_name)

            min_len = min(len(gt_corners), len(pred_corners))
            gt_corners = gt_corners[:min_len]
            pred_corners = pred_corners[:min_len]
            valid = valid[:min_len]

            masks_dir = results_dir / 'masks' / seq_name

            seq_result = evaluate_sequence_nn(
                seq_name, gt_corners, pred_corners, valid, masks_dir, scorer,
            )
            all_results.append(seq_result)
            print(
                f"  {seq_name}: "
                f"bal_acc={seq_result['balanced_accuracy']:.3f}  "
                f"tpr={seq_result['tpr']:.3f}  tnr={seq_result['tnr']:.3f}  "
                f"spearman_good={seq_result['spearman_r_good']:.3f}  "
                f"mean_pred_rmse={seq_result['mean_predicted_rmse']:.1f}px  "
                f"mean_gt_err={seq_result['mean_gt_error']:.1f}px  "
                f"(good={seq_result['n_good_frames']}, bad={seq_result['n_bad_frames']})"
            )
        except Exception as e:
            import traceback
            print(f"  {seq_name}: FAILED — {e}")
            traceback.print_exc()
            failed.append((seq_name, str(e)))

        _save(output_file, checkpoint, arch, all_results, failed)

    aggregate = _build_aggregate(checkpoint, arch, all_results, failed)

    print(f"\n{'='*60}")
    print(f"AGGREGATE ({checkpoint.name}): {aggregate['num_sequences']} sequences, "
          f"{aggregate.get('num_failed', 0)} failed")
    if 'mean_balanced_accuracy' in aggregate:
        print(f"  Balanced accuracy (TPR+TNR)/2 : {aggregate['mean_balanced_accuracy']:.3f}")
        print(f"  Mean TPR (sensitivity)        : {aggregate['mean_tpr']:.3f}")
        print(f"  Mean TNR (specificity)        : {aggregate['mean_tnr']:.3f}")
        print(f"  Mean Spearman r (good frames) : {aggregate['mean_spearman_r_good']:.3f}")
        print(f"  Median Spearman r (good frames): {aggregate['median_spearman_r_good']:.3f}")

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
        description="Evaluate NN homography quality estimation against GT alignment errors"
    )
    parser.add_argument('dataset', choices=CLI_CHOICES)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument('--arch', default="option_e", choices=["option_e", "option_c"])
    parser.add_argument('--device', default="cuda",
                        help="PyTorch device (default: cuda)")
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
        checkpoint=args.checkpoint,
        arch=args.arch,
        config_path=args.config,
        initfix_dir=args.initfix_dir,
        output_file=args.output,
        device=args.device,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
