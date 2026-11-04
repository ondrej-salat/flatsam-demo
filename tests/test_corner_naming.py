"""The keypoint branch naming the mask branch's corners (``_relabel_to_reference``)."""

import numpy as np

from flatsam2.tracking.corner_fusion import _relabel_to_reference


def _square(side: float = 100.0) -> np.ndarray:
    """(2, 4) axis-aligned square, corners in order."""
    return np.array([[0.0, side, side, 0.0], [0.0, 0.0, side, side]])


def test_disabled_margin_leaves_naming_alone():
    quad = np.roll(_square(), 1, axis=1)
    kept, covs = _relabel_to_reference(quad, None, _square(), margin=0.0)
    assert np.allclose(kept, quad)
    assert covs is None


def test_recovers_a_cyclic_relabeling():
    """A rectangle rolled by 2 is un-rolled back onto the reference naming."""
    ref = np.array([[0.0, 200.0, 200.0, 0.0], [0.0, 0.0, 60.0, 60.0]])
    quad = np.roll(ref, 2, axis=1)
    fixed, _ = _relabel_to_reference(quad, None, ref, margin=2.0)
    assert np.allclose(fixed, ref)


def test_covariances_follow_the_permutation():
    ref = np.array([[0.0, 200.0, 200.0, 0.0], [0.0, 0.0, 60.0, 60.0]])
    covs = np.stack([np.eye(2) * (k + 1) for k in range(4)])
    fixed, fixed_covs = _relabel_to_reference(np.roll(ref, 1, axis=1),
                                              np.roll(covs, 1, axis=0), ref, margin=2.0)
    assert np.allclose(fixed, ref)
    assert np.allclose(fixed_covs, covs)


def test_margin_protects_against_a_badly_localized_reference():
    """A reference quad far from this one makes all four rotations near-equidistant.

    That is the case the margin exists for: on the mis-named frames the keypoint
    branch names correctly 72% of the time but is within 15 px only 45% of the
    time, so a distant reference must not be allowed to re-name anything.
    """
    quad = np.roll(_square(), 1, axis=1)
    far_reference = _square() + np.array([[5000.0], [5000.0]])
    kept, _ = _relabel_to_reference(quad, None, far_reference, margin=2.0)
    assert np.allclose(kept, quad)


def test_a_close_reference_still_re_names_through_the_margin():
    """The same relabeling is taken once the reference is genuinely nearby."""
    quad = np.roll(_square(), 1, axis=1)
    fixed, _ = _relabel_to_reference(quad, None, _square(), margin=2.0)
    assert np.allclose(fixed, _square())


def test_margin_1_takes_the_nearest_rotation():
    """With no protection, any closer rotation wins — the measured-worse setting."""
    ref = np.array([[0.0, 100.0, 104.0, 4.0], [0.0, 0.0, 100.0, 100.0]])
    quad = np.roll(ref, 3, axis=1)
    fixed, _ = _relabel_to_reference(quad, None, ref, margin=1.0)
    assert np.allclose(fixed, ref)
