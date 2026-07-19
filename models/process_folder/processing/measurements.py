"""
measurements.py
===============

Convert a set of keypoints into the measurements of interest.

A keypoint array is expected in the shape (NUM_KEYPOINTS, 3):
    column 0 = x   (pixels)
    column 1 = y   (pixels)
    column 2 = confidence / visibility in [0, 1]

Each measurement is the SUM of the Euclidean lengths of the consecutive
segments of its keypoint chain (confirmed). For a 2-keypoint measurement this
is simply the distance between the two points; for a polyline such as a leg or
an antenna it is the length walked along the chain.
"""

from __future__ import annotations

import math

import numpy as np

from . import config
from .definitions import (
    MEASUREMENTS,
    MEASUREMENT_INDICES,
    MEASUREMENT_NAMES,
    KEYPOINT_NAMES,
)


def _segment_length(p0: np.ndarray, p1: np.ndarray) -> float:
    """Euclidean distance between two (x, y) points."""
    return float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def keypoint_visibility_threshold(keypoint_name: str) -> float:
    """Per-keypoint visibility threshold, with a global default."""
    return config.PER_KEYPOINT_VISIBILITY_THRESHOLD.get(
        keypoint_name, config.KEYPOINT_VISIBILITY_THRESHOLD
    )


def measurement_keypoint_confidences(keypoints: np.ndarray, measurement_name: str) -> list[float]:
    """Return the confidence values of the keypoints involved in a measurement."""
    idx = MEASUREMENT_INDICES[measurement_name]
    return [float(keypoints[i, 2]) for i in idx]


def measurement_below_threshold(keypoints: np.ndarray, measurement_name: str) -> bool:
    """True if ANY keypoint of the measurement is below its visibility threshold."""
    for kp_name in MEASUREMENTS[measurement_name]:
        i = KEYPOINT_NAMES.index(kp_name)
        if keypoints[i, 2] < keypoint_visibility_threshold(kp_name):
            return True
    return False


def compute_measurement(keypoints: np.ndarray, measurement_name: str) -> float:
    """Compute a single measurement (in pixels) as the sum of its segments.

    Returns NaN if the measurement is dropped because of low-visibility
    keypoints (only when DROP_MEASUREMENT_IF_KP_BELOW_THRESHOLD is True).
    """
    if (config.DROP_MEASUREMENT_IF_KP_BELOW_THRESHOLD
            and measurement_below_threshold(keypoints, measurement_name)):
        return math.nan

    idx = MEASUREMENT_INDICES[measurement_name]
    total = 0.0
    for a, b in zip(idx[:-1], idx[1:]):
        total += _segment_length(keypoints[a, :2], keypoints[b, :2])
    return total


def compute_measurements(keypoints: np.ndarray) -> dict[str, float]:
    """Compute every measurement (in pixels) for one keypoint array.

    Parameters
    ----------
    keypoints : (NUM_KEYPOINTS, 3) array of (x, y, confidence).

    Returns
    -------
    dict {measurement_name: pixel_value}
    """
    return {name: compute_measurement(keypoints, name) for name in MEASUREMENT_NAMES}
