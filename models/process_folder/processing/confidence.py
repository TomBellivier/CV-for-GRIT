"""
confidence.py
=============

ALL confidence computations live here, so the methodology is in one readable
place and easy to swap once you have looked at your validation split.

A confidence is only useful if it *predicts the real error*: high confidence
should mean small error and low confidence should mean large error. The raw
scores below are reasonable, monotone signals to start from; the recommended
next step (documented in the project README) is to CALIBRATE them on the val
split so that a number like 0.9 maps to a known error level.

Contents
--------
POSE (per-measurement), two interchangeable signals -- pick one in config:
    1. measurement_confidence_keypoint(...)   -> aggregated keypoint score
    2. measurement_confidence_tta(...)        -> test-time-augmentation stability
POSE (whole insect):
    overall_pose_confidence(...)
SCALE:
    scale_bar_confidence(...)                 -> bar_conf * text_conf * ocr
    ruler_confidence(...)                     -> magnitude separation of groups
CONVERSION (pixels -> mm):
    converted_measurement_confidence(...)     -> combine measurement & scale
"""

from __future__ import annotations

import math

import numpy as np

from . import config


# ========================================================================== #
# 1. POSE -- per-measurement confidence, signal A: KEYPOINT confidence
# ========================================================================== #
def aggregate_keypoint_confidence(kp_confidences, method: str = "min") -> float:
    """Aggregate the keypoint confidences of ONE measurement into a scalar.

    Rationale for the choices
    -------------------------
    - "min": the weakest link. A distance is ruined as soon as one endpoint is
      badly placed, so the smallest keypoint confidence should drive the value.
      This is the safest, most interpretable default for 2-point measurements.
    - "geometric_mean": still strongly penalises one weak point but less
      brutally than the min; convenient for polylines (antenna, leg) where only
      one of several segments is affected.
    - "mean": the most lenient; provided for completeness / experimentation.

    Parameters
    ----------
    kp_confidences : iterable of per-keypoint confidences in [0, 1].
    method         : "min" | "geometric_mean" | "mean".
    """
    c = np.asarray(list(kp_confidences), dtype=float)
    if c.size == 0 or np.any(np.isnan(c)):
        return math.nan

    if method == "min":
        return float(np.min(c))
    if method == "mean":
        return float(np.mean(c))
    if method == "geometric_mean":
        # Guard against a zero collapsing the whole product.
        c = np.clip(c, 1e-9, 1.0)
        return float(np.exp(np.mean(np.log(c))))
    raise ValueError(f"Unknown keypoint aggregation method: {method!r}")


def measurement_confidence_keypoint(kp_confidences, method: str | None = None) -> float:
    """Signal A -- confidence of one measurement from its keypoint scores.

    Thin wrapper over aggregate_keypoint_confidence that reads the default
    aggregation method from the config (KEYPOINT_AGGREGATION) so the choice is
    made in one place and stays easy to tune by hand.
    """
    method = method or config.KEYPOINT_AGGREGATION
    return aggregate_keypoint_confidence(kp_confidences, method)


# ========================================================================== #
# 1'. POSE -- per-measurement confidence, signal B: TTA stability
# ========================================================================== #
def measurement_confidence_tta(values, beta: float | None = None) -> float:
    """Signal B -- confidence of one measurement from test-time augmentation.

    Idea: run inference several times on slightly perturbed copies of the image
    and look at how much the measurement moves. A stable measurement (small
    dispersion) is trustworthy; a jumpy one is not.

    We use the coefficient of variation  cv = std / mean  (scale-free, so it is
    comparable across measurements of very different sizes) and map it to a
    confidence with:

        confidence = exp(-beta * cv)          clamped to [0, 1]

    'beta' (config.TTA_CV_BETA) controls strictness: larger beta means a small
    dispersion already pulls the confidence down. exp() is used so the result
    stays in (0, 1] and degrades smoothly.

    Parameters
    ----------
    values : iterable of the SAME measurement obtained on each augmented pass
             (already mapped back to the original image frame). NaNs are ignored.
    """
    beta = config.TTA_CV_BETA if beta is None else beta
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]

    # Need at least two passes to estimate a dispersion.
    if v.size < 2:
        return math.nan
    mean = float(np.mean(v))
    if mean <= 0:
        return math.nan

    cv = float(np.std(v)) / mean
    return float(min(1.0, max(0.0, math.exp(-beta * cv))))


# ========================================================================== #
# 2. POSE -- overall confidence for the whole insect
# ========================================================================== #
def overall_pose_confidence(detection_conf: float, keypoint_confidences,
                            method: str | None = None) -> float:
    """Global pose confidence combining detection score and keypoint scores.

    - detection_conf : the YOLO box score ("this really is an insect").
    - keypoint scores : how well the individual points are placed
                        ("and it is well articulated"), summarised by their mean.

    method:
        "det_x_kp" -> detection_conf * mean(keypoint_conf)   (default)
        "min"      -> min(detection_conf, mean(keypoint_conf))
    """
    method = method or config.OVERALL_POSE_METHOD
    kp = np.asarray(list(keypoint_confidences), dtype=float)
    if kp.size == 0 or np.any(np.isnan(kp)):
        return math.nan
    mean_kp = float(np.mean(kp))

    if method == "det_x_kp":
        return float(detection_conf * mean_kp)
    if method == "min":
        return float(min(detection_conf, mean_kp))
    raise ValueError(f"Unknown overall-pose method: {method!r}")


# ========================================================================== #
# 3. SCALE -- scale-bar confidence
# ========================================================================== #
def scale_bar_confidence(bar_box_conf: float,
                         text_box_conf: float,
                         ocr_reliability: float) -> float:
    """Confidence of a scale read from a scale bar.

    As requested: the PRODUCT of
        - the detection confidence of the bar box,
        - the detection confidence of the text box,
        - the reliability of the OCR read.

    Using a product means the result collapses toward 0 if ANY of the three
    steps is weak (a mislocated bar, a missed text box, or an unreadable
    number), which is the desired conservative behaviour. Each factor is clamped
    to [0, 1] first.
    """
    factors = [bar_box_conf, text_box_conf, ocr_reliability]
    clamped = [float(min(1.0, max(0.0, f))) for f in factors]
    result = 1.0
    for f in clamped:
        result *= f
    return result


# ========================================================================== #
# 3'. SCALE -- ruler confidence (Fourier groups)
# ========================================================================== #
def ruler_confidence(group_magnitudes) -> float:
    """Confidence of a scale read from a ruler via the Fourier analysis.

    The ruler detector groups image rows by their dominant spatial frequency.
    `group_magnitudes` is the list of the groups' mean FFT magnitudes, ordered
    so that index 0 is the MAIN group (the one used to compute px/mm) and the
    following ones are the (up to 4) secondary groups.

    Rule (as requested)
    -------------------
    - A single group detected  -> confidence = 1
      (nothing competes with the ruler frequency, so it is unambiguous).
    - Several groups detected   -> mean relative magnitude gap between the main
      group and the n secondary groups:

            confidence = (1/n) * sum_i  |M0 - Mi| / M0        for i = 1 .. n

      A confidence close to 1 means the main group dominates (its magnitude is
      far above the competitors), i.e. the detected frequency is clearly the
      ruler. A confidence close to 0 means a secondary group is almost as strong
      as the main one, i.e. the reading is ambiguous.

    The result is clamped to [0, 1] because a secondary group could, in rare
    cases, have a higher mean magnitude than the (most-populated) main group,
    which would otherwise push a term above 1.
    """
    m = np.asarray(list(group_magnitudes), dtype=float)
    if m.size == 0:
        return math.nan
    if m.size == 1:
        return 1.0

    m0 = m[0]
    if m0 <= 0:
        return math.nan

    secondaries = m[1:]
    n = secondaries.size
    rel_gaps = np.abs(m0 - secondaries) / m0
    conf = float(np.sum(rel_gaps) / n)
    return float(min(1.0, max(0.0, conf)))


# ========================================================================== #
# 4. CONVERSION -- confidence of a millimetre measurement
# ========================================================================== #
def converted_measurement_confidence(measurement_conf: float,
                                     scale_conf: float,
                                     method: str | None = None) -> float:
    """Confidence of a measurement AFTER conversion to millimetres.

    A millimetre value is only as good as the weaker of its two ingredients:
    the pixel measurement and the scale. Two simple combinations are offered:

        "min"     -> min(measurement_conf, scale_conf)   (conservative default)
        "product" -> measurement_conf * scale_conf

    (A more rigorous option, once both confidences are calibrated into relative
    errors, is to combine the errors in quadrature:
        err_mm ~ sqrt(err_pixel^2 + err_scale^2)
    and convert back to a confidence. That upgrade belongs in the calibrated
    version and is described in the README.)
    """
    method = method or config.CONVERTED_CONF_METHOD
    if measurement_conf is None or scale_conf is None:
        return math.nan
    if math.isnan(measurement_conf) or math.isnan(scale_conf):
        return math.nan

    if method == "min":
        return float(min(measurement_conf, scale_conf))
    if method == "product":
        return float(measurement_conf * scale_conf)
    raise ValueError(f"Unknown converted-confidence method: {method!r}")
