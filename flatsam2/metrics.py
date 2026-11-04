"""Evaluation metrics for planar object tracking.

This module implements metrics for evaluating tracking quality:
- Alignment error: RMS distance across corners
- P@k metrics: Percentage of frames with error below threshold
- Best rotation error: Rotation-invariant alignment error
"""

from typing import Dict, List, Tuple
import numpy as np


def alignment_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute alignment error between ground truth and predicted corners.
    
    The alignment error is the RMS distance across all 4 corners per frame.
    
    Args:
        gt: Ground truth corners with shape (num_frames, 2, 4) or (2, 4)
        pred: Predicted corners with shape (num_frames, 2, 4) or (2, 4)
        
    Returns:
        Alignment error per frame with shape (num_frames,) or scalar
    """
    # Ensure we're working with arrays
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    
    # Compute squared differences
    diff_squared = (gt - pred) ** 2
    
    # Sum over coordinates (x, y) and corners, then take sqrt and divide by sqrt(4) = 2
    # Shape: (..., 2, 4) -> (...,)
    if gt.ndim == 2:
        # Single frame case (2, 4)
        error = np.sqrt(np.sum(diff_squared) / 4.0)
    else:
        # Multiple frames case (num_frames, 2, 4)
        error = np.sqrt(np.sum(diff_squared, axis=(1, 2)) / 4.0)
    
    return error


def best_rotation_error(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotation-invariant alignment error.
    
    Since corner ordering is ambiguous (could start from any corner), this
    function tries all 4 possible rotations and returns the minimum error.
    
    Args:
        gt: Ground truth corners with shape (num_frames, 2, 4)
        pred: Predicted corners with shape (num_frames, 2, 4)
        
    Returns:
        best_errors: Minimum alignment error per frame (num_frames,)
        best_rotations: Best rotation index per frame (num_frames,)
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    
    num_frames = gt.shape[0] if gt.ndim > 2 else 1
    if gt.ndim == 2:
        gt = gt[np.newaxis, ...]
        pred = pred[np.newaxis, ...]
    
    # Try all 4 rotations
    errors = np.zeros((num_frames, 4))
    for rot in range(4):
        # Roll the predicted corners
        pred_rotated = np.roll(pred, shift=rot, axis=2)
        errors[:, rot] = alignment_error(gt, pred_rotated)
    
    # Find best rotation per frame
    best_rotations = np.argmin(errors, axis=1)
    best_errors = errors[np.arange(num_frames), best_rotations]
    
    if num_frames == 1:
        return best_errors[0], best_rotations[0]
    
    return best_errors, best_rotations


def precision_at_threshold(errors: np.ndarray, threshold: float) -> float:
    """Compute percentage of frames with error below threshold.
    
    Args:
        errors: Alignment errors per frame
        threshold: Error threshold in pixels
        
    Returns:
        Percentage of frames with error <= threshold (0-100)
    """
    return 100.0 * np.mean(errors <= threshold)


def evaluate_sequence(
    gt: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    use_best_rotation: bool = False
) -> Dict[str, any]:
    """Evaluate tracking results for a single sequence.
    
    Args:
        gt: Ground truth corners with shape (num_frames, 2, 4)
        pred: Predicted corners with shape (num_frames, 2, 4)
        valid: Boolean mask indicating valid frames (num_frames,)
        use_best_rotation: If True, compute rotation-invariant metrics
        
    Returns:
        Dictionary containing:
            - 'errors': Per-frame alignment errors (only valid frames)
            - 'p@5': Precision at 5 pixels
            - 'p@10': Precision at 10 pixels  
            - 'p@15': Precision at 15 pixels
            - 'mean_error': Mean alignment error
            - 'median_error': Median alignment error
            If use_best_rotation=True, also includes:
            - 'best_rotation_errors': Rotation-invariant errors
            - 'best_rotations': Best rotation indices
            - 'br_p@5', 'br_p@10', 'br_p@15': Best rotation precision metrics
            - 'br_mean_error', 'br_median_error': Best rotation error statistics
    """
    # Filter to valid frames only (skip first frame which is initialization)
    valid_frames = valid.copy()
    if len(valid_frames) > 0:
        valid_frames[0] = False  # Skip initialization frame
    
    gt_valid = gt[valid_frames]
    pred_valid = pred[valid_frames]
    
    if len(gt_valid) == 0:
        empty_result = {
            'errors': np.array([]),
            'p@5': 0.0,
            'p@10': 0.0,
            'p@15': 0.0,
            'mean_error': np.nan,
            'median_error': np.nan,
            'num_valid': 0
        }
        if use_best_rotation:
            empty_result.update({
                'best_rotation_errors': np.array([]),
                'best_rotations': np.array([]),
                'br_p@5': 0.0,
                'br_p@10': 0.0,
                'br_p@15': 0.0,
                'br_mean_error': np.nan,
                'br_median_error': np.nan,
            })
        return empty_result
    
    # Compute standard alignment errors
    errors = alignment_error(gt_valid, pred_valid)
    
    result = {
        'errors': errors,
        'p@5': precision_at_threshold(errors, 5.0),
        'p@10': precision_at_threshold(errors, 10.0),
        'p@15': precision_at_threshold(errors, 15.0),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'num_valid': len(errors)
    }
    
    # Optionally compute best rotation metrics
    if use_best_rotation:
        br_errors, br_rotations = best_rotation_error(gt_valid, pred_valid)
        result.update({
            'best_rotation_errors': br_errors,
            'best_rotations': br_rotations,
            'br_p@5': precision_at_threshold(br_errors, 5.0),
            'br_p@10': precision_at_threshold(br_errors, 10.0),
            'br_p@15': precision_at_threshold(br_errors, 15.0),
            'br_mean_error': np.mean(br_errors),
            'br_median_error': np.median(br_errors),
        })
    
    return result


def aggregate_results(sequence_results: List[Dict]) -> Dict[str, float]:
    """Aggregate evaluation results across multiple sequences.
    
    Args:
        sequence_results: List of per-sequence evaluation dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    # Collect all errors
    all_errors = np.concatenate([res['errors'] for res in sequence_results if len(res['errors']) > 0])
    
    if len(all_errors) == 0:
        return {
            'mean_p@5': 0.0,
            'mean_p@10': 0.0,
            'mean_p@15': 0.0,
            'overall_mean_error': np.nan,
            'overall_median_error': np.nan,
        }
    
    # Aggregate metrics
    result = {
        'mean_p@5': np.mean([res['p@5'] for res in sequence_results]),
        'mean_p@10': np.mean([res['p@10'] for res in sequence_results]),
        'mean_p@15': np.mean([res['p@15'] for res in sequence_results]),
        'overall_p@5': precision_at_threshold(all_errors, 5.0),
        'overall_p@10': precision_at_threshold(all_errors, 10.0),
        'overall_p@15': precision_at_threshold(all_errors, 15.0),
        'overall_mean_error': np.mean(all_errors),
        'overall_median_error': np.median(all_errors),
        'num_sequences': len(sequence_results),
        'total_frames': len(all_errors),
    }
    
    # Check if best rotation metrics are available
    if 'best_rotation_errors' in sequence_results[0]:
        all_br_errors = np.concatenate([
            res['best_rotation_errors'] 
            for res in sequence_results 
            if len(res.get('best_rotation_errors', [])) > 0
        ])
        
        result.update({
            'mean_br_p@5': np.mean([res['br_p@5'] for res in sequence_results]),
            'mean_br_p@10': np.mean([res['br_p@10'] for res in sequence_results]),
            'mean_br_p@15': np.mean([res['br_p@15'] for res in sequence_results]),
            'overall_br_p@5': precision_at_threshold(all_br_errors, 5.0),
            'overall_br_p@10': precision_at_threshold(all_br_errors, 10.0),
            'overall_br_p@15': precision_at_threshold(all_br_errors, 15.0),
            'overall_br_mean_error': np.mean(all_br_errors),
            'overall_br_median_error': np.median(all_br_errors),
        })
    
    return result
