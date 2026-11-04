"""Geometry utility functions shared across the flatsam2 package."""

import numpy as np


def warp_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to a set of 2D points.

    Handles the euclidean -> projective -> euclidean conversion internally.

    Args:
        H: Homography matrix (3, 3).
        points: 2D point coordinates (N, 2).

    Returns:
        Transformed 2D points (N, 2).
    """
    n = points.shape[0]
    points_h = np.concatenate(
        [points, np.ones((n, 1), dtype=points.dtype)], axis=1
    )  # (N, 3)
    warped_h = (H @ points_h.T).T  # (N, 3)
    return warped_h[:, :2] / warped_h[:, 2:3]
