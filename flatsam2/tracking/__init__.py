"""Tracking algorithms and the verbs/types shared across them.

The ``common`` module holds the homography validity/mask helpers, candidate
labels, and per-frame result/debug dataclasses used by every algorithm. The
per-algorithm modules (``flow_hough``, ``loma_samh``) are free functions taking
the :class:`~flatsam2.tracker.FlatSAMTracker` instance.
"""
