"""Detecting that a segmentation mask is an *incomplete* view of the object.

The covariance the mask branch reports says how far its corners would move if
the boundary samples were resampled. It cannot say the boundary belongs to the
wrong thing — and under occlusion that is exactly what happens: SAM returns the
visible part, whose outline includes the occluder's edge, and the registrar
fits the template to that sub-region. The edge is clean and straight, so the fit
is *precise*, the residuals are small, the covariance is tight, and the pose is
wrong. A confident wrong answer then wins the arbitration against a keypoint
branch that is honestly uncertain about the corners it cannot see.

No amount of scaling fixes that, because the error is model-form (fitting the
wrong boundary) rather than sampling noise. What can be detected is the
*signature*: compare the mask against the silhouette the tracker already
predicts for this frame, and ask two separate questions.

    precision = |mask & predicted| / |mask|        is everything SAM returned
                                                    part of the object?
    recall    = |mask & predicted| / |predicted|   did it return all of it?

High precision with low recall is "an almost perfect subset" — the object is
there, and half of it is hidden. That separates occlusion from the two things it
could be confused with: a segmenter that has locked onto something else loses
*precision*, and a tracker that has drifted loses both.

Two things the prediction must get right, both found in review. It warps the
*stored template silhouette*, not a quadrilateral through four corners — the
tracker is quad-free and the two regions differ for any non-rectangular object.
And it must be a prediction for *this* frame: comparing against the previous
pose makes fast motion look like occlusion (recall falls because the object
moved) and hides real occlusion during motion. Which pose to predict from is a
choice the offline screen measures rather than assumes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import cv2


def warp_template_silhouette(
    template_mask: np.ndarray,
    homography: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """The template's own silhouette, warped into the current frame.

    Warping the *stored silhouette* rather than rasterizing a quadrilateral
    through its four corners is the whole point of a quad-free tracker: the
    object may be an arbitrary shape, and a quad through its corners is a
    different region whenever it is not convex-rectangular. Comparing the mask
    against that quad would measure the shape assumption rather than occlusion.

    Args:
        template_mask: (H0, W0) initial-frame silhouette, non-zero is object.
        homography: (3, 3) template frame -> current frame.
        shape: (H, W) of the current frame.

    Returns:
        (H, W) bool array, the expected silhouette this frame.
    """
    warped = cv2.warpPerspective(
        (np.asarray(template_mask) > 0).astype(np.uint8),
        homography.astype(np.float64),
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    return warped > 0


def subset_evidence(
    mask: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[float, float]:
    """How much of the mask the prediction explains, and vice versa.

    Args:
        mask: (H, W) segmentation mask; non-zero is foreground.
        predicted: (H, W) expected silhouette this frame, e.g. from
            :func:`warp_template_silhouette`.

    Returns:
        ``(precision, recall)``. Precision is the share of the mask inside the
        prediction, recall the share of the prediction covered by the mask.
        Both are NaN when the corresponding region is empty.
    """
    mask_bool = np.asarray(mask) > 0
    predicted = np.asarray(predicted) > 0
    n_mask, n_pred = int(mask_bool.sum()), int(predicted.sum())
    if n_mask == 0 or n_pred == 0:
        return float("nan"), float("nan")
    inter = int(np.logical_and(mask_bool, predicted).sum())
    return inter / n_mask, inter / n_pred


def is_occluded_view(
    mask: np.ndarray,
    predicted: np.ndarray,
    min_precision: float = 0.90,
    max_recall: float = 0.80,
) -> bool:
    """Whether the mask looks like a clean partial view of the expected object.

    Deliberately a hard test rather than a continuous discount. A discount only
    makes a confidently-wrong branch lose more slowly, and needs an exponent
    chosen by hand; this asks a question the geometry can answer — *is the mask
    a subset?* — and answers yes or no.

    Args:
        mask: (H, W) segmentation mask.
        predicted: (H, W) expected silhouette this frame.
        min_precision: The mask must be this contained in the prediction, or it
            is not a clean partial view — it is a different object, or drift.
        max_recall: The prediction must be covered no more than this, or nothing
            is hidden and the mask branch has its full evidence.

    Returns:
        True when the mask is an almost-perfect subset of the expected
        silhouette, i.e. part of the object is missing from it.
    """
    precision, recall = subset_evidence(mask, predicted)
    if not (np.isfinite(precision) and np.isfinite(recall)):
        return False
    return precision >= min_precision and recall <= max_recall
