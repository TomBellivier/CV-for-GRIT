"""Derived features for measurement-validity classification.

The raw pose outputs (per-keypoint confidence, per-keypoint pixel position,
per-measurement pixel length) are weak predictors on their own: they are
expressed in image units, so they encode the acquisition protocol as much as the
quality of the pose. The functions below build features that are invariant to
image scale and that make the usual failure modes explicit:

* **scale normalisation** - every length divided by a robust reference length,
  so a measurement becomes comparable across images and magnifications;
* **anatomical plausibility** - deviation of a normalised measurement from the
  median of its taxonomic group, which is the most direct signal of a
  foreshortened or misplaced keypoint;
* **confidence aggregates** - min / mean / product / spread of the confidences
  of the keypoints involved, plus their rank within the image, which a tree
  ensemble cannot easily recover from 42 independent columns;
* **geometry** - orientation of the measured segment relative to the body axis,
  which is where out-of-plane rotation (the main cause of an unmeasurable
  distance) becomes visible;
* **framing** - apparent size of the animal and proximity of the keypoints to
  the image border, i.e. effective resolution and truncation.

Group-conditional statistics (the medians used for the plausibility features)
are *learnt*, so they are fitted inside the cross-validation pipeline via
:class:`GroupReferenceStats`, never on the full dataset.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dataset import ColumnIndex
from insect_anatomy import (
    BILATERAL_PAIRS,
    BODY_AXIS_CANDIDATES,
    INSECT_GROUPS,
    MEAS_TO_KP,
    MEASUREMENTS,
    POINTS,
    SCALE_REFERENCE_MEASURES,
)

LOGGER = logging.getLogger(__name__)

FEATURE_PREFIX = "feat_"
EPS = 1e-9


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio returning NaN where the denominator is ~zero."""
    denominator = denominator.where(denominator.abs() > EPS)
    return numerator / denominator


def scale_reference(frame: pd.DataFrame, columns: ColumnIndex) -> pd.Series:
    """Robust per-image reference length used to make lengths dimensionless.

    Prefers an anatomical total length; falls back to the diagonal of the
    keypoint bounding box, which is always available.
    """
    for measure in SCALE_REFERENCE_MEASURES:
        column = columns.measure.get(measure)
        if column is None:
            continue
        values = frame[column].astype(float)
        if values.gt(EPS).mean() > 0.9:
            return values.where(values > EPS)

    LOGGER.warning("No anatomical scale reference available; using keypoint bbox diagonal")
    return keypoint_bbox_diagonal(frame, columns)


def keypoint_bbox_diagonal(frame: pd.DataFrame, columns: ColumnIndex) -> pd.Series:
    """Diagonal of the axis-aligned bounding box of all detected keypoints."""
    points = columns.complete_points
    if not points:
        return pd.Series(np.nan, index=frame.index)
    xs = frame[[columns.x[p] for p in points]].astype(float)
    ys = frame[[columns.y[p] for p in points]].astype(float)
    width = xs.max(axis=1) - xs.min(axis=1)
    height = ys.max(axis=1) - ys.min(axis=1)
    return np.sqrt(width**2 + height**2)


def _body_axis(frame: pd.DataFrame, columns: ColumnIndex) -> Optional[tuple]:
    """Vector components of the antero-posterior body axis, if computable."""
    for head, tail in BODY_AXIS_CANDIDATES:
        if all(p in columns.x and p in columns.y for p in (head, tail)):
            dx = frame[columns.x[tail]].astype(float) - frame[columns.x[head]].astype(float)
            dy = frame[columns.y[tail]].astype(float) - frame[columns.y[head]].astype(float)
            return dx, dy
    return None


def confidence_features(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    measure: str,
) -> pd.DataFrame:
    """Aggregates over the confidences of the keypoints defining ``measure``.

    Also includes image-level confidence statistics, which act as a proxy for
    overall image quality, and the rank of the target keypoints within the
    image, which separates "this keypoint is bad" from "this whole image is
    bad".
    """
    out = pd.DataFrame(index=frame.index)
    available = [p for p in POINTS if p in columns.conf]
    if not available:
        return out

    all_conf = frame[[columns.conf[p] for p in available]].astype(float)
    out[f"{FEATURE_PREFIX}conf_image_mean"] = all_conf.mean(axis=1)
    out[f"{FEATURE_PREFIX}conf_image_min"] = all_conf.min(axis=1)
    out[f"{FEATURE_PREFIX}conf_image_std"] = all_conf.std(axis=1)
    for threshold in (0.3, 0.5, 0.7):
        out[f"{FEATURE_PREFIX}conf_image_frac_below_{threshold}"] = (
            all_conf.lt(threshold).mean(axis=1)
        )

    target_points = [p for p in MEAS_TO_KP.get(measure, []) if p in columns.conf]
    if not target_points:
        return out

    target_conf = frame[[columns.conf[p] for p in target_points]].astype(float)
    out[f"{FEATURE_PREFIX}conf_target_min"] = target_conf.min(axis=1)
    out[f"{FEATURE_PREFIX}conf_target_mean"] = target_conf.mean(axis=1)
    out[f"{FEATURE_PREFIX}conf_target_max"] = target_conf.max(axis=1)
    out[f"{FEATURE_PREFIX}conf_target_spread"] = target_conf.max(axis=1) - target_conf.min(axis=1)
    out[f"{FEATURE_PREFIX}conf_target_prod"] = target_conf.prod(axis=1)
    out[f"{FEATURE_PREFIX}conf_target_vs_image"] = (
        target_conf.mean(axis=1) - all_conf.mean(axis=1)
    )
    # Rank of the worst target keypoint among all keypoints of the image.
    ranks = all_conf.rank(axis=1, pct=True)
    target_rank_cols = [columns.conf[p] for p in target_points]
    out[f"{FEATURE_PREFIX}conf_target_rank_min"] = ranks[target_rank_cols].min(axis=1)
    return out


def geometry_features(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    measure: str,
) -> pd.DataFrame:
    """Scale-invariant geometry: relative size, orientation, framing, spread."""
    out = pd.DataFrame(index=frame.index)
    reference = scale_reference(frame, columns)
    diagonal = keypoint_bbox_diagonal(frame, columns)

    out[f"{FEATURE_PREFIX}kp_bbox_diagonal"] = diagonal
    out[f"{FEATURE_PREFIX}scale_reference"] = reference
    out[f"{FEATURE_PREFIX}bbox_over_reference"] = _safe_ratio(diagonal, reference)

    points = columns.complete_points
    if points:
        xs = frame[[columns.x[p] for p in points]].astype(float)
        ys = frame[[columns.y[p] for p in points]].astype(float)
        out[f"{FEATURE_PREFIX}kp_bbox_aspect"] = _safe_ratio(
            xs.max(axis=1) - xs.min(axis=1), ys.max(axis=1) - ys.min(axis=1)
        )
        # Dispersion of the keypoint cloud, normalised: a collapsed cloud means
        # the pose model gave up and stacked every keypoint at one location.
        out[f"{FEATURE_PREFIX}kp_spread"] = _safe_ratio(
            np.sqrt(xs.var(axis=1) + ys.var(axis=1)), diagonal
        )
        # Distance of the closest keypoint to the image border, in units of the
        # animal size. Requires the image dimensions when available.
        for width_col, height_col in (("image_width", "image_height"), ("width", "height")):
            if width_col in frame.columns and height_col in frame.columns:
                img_w = frame[width_col].astype(float)
                img_h = frame[height_col].astype(float)
                margin = pd.concat(
                    [
                        xs.min(axis=1),
                        ys.min(axis=1),
                        img_w - xs.max(axis=1),
                        img_h - ys.max(axis=1),
                    ],
                    axis=1,
                ).min(axis=1)
                out[f"{FEATURE_PREFIX}border_margin_norm"] = _safe_ratio(margin, diagonal)
                out[f"{FEATURE_PREFIX}animal_image_coverage"] = _safe_ratio(
                    diagonal, np.sqrt(img_w**2 + img_h**2)
                )
                break

    axis = _body_axis(frame, columns)
    target_points = [p for p in MEAS_TO_KP.get(measure, []) if p in columns.x and p in columns.y]
    if axis is not None and len(target_points) >= 2:
        axis_dx, axis_dy = axis
        axis_angle = np.arctan2(axis_dy, axis_dx)
        head, tail = target_points[0], target_points[-1]
        seg_dx = frame[columns.x[tail]].astype(float) - frame[columns.x[head]].astype(float)
        seg_dy = frame[columns.y[tail]].astype(float) - frame[columns.y[head]].astype(float)
        seg_angle = np.arctan2(seg_dy, seg_dx)
        delta = np.abs(((seg_angle - axis_angle + np.pi) % (2 * np.pi)) - np.pi)
        # Undirected angle in [0, pi/2]: a segment and its reverse are the same.
        delta = np.minimum(delta, np.pi - delta)
        out[f"{FEATURE_PREFIX}target_angle_to_body_axis"] = delta
        out[f"{FEATURE_PREFIX}target_angle_cos"] = np.cos(delta)
        out[f"{FEATURE_PREFIX}body_axis_length_norm"] = _safe_ratio(
            np.sqrt(axis_dx**2 + axis_dy**2), diagonal
        )

    return out


def measurement_features(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    measure: str,
) -> pd.DataFrame:
    """Normalised lengths and left/right asymmetries.

    Ratios to the scale reference remove the dependency on magnification;
    asymmetries capture oblique poses and unilateral occlusions.
    """
    out = pd.DataFrame(index=frame.index)
    reference = scale_reference(frame, columns)

    for name, column in columns.measure.items():
        values = frame[column].astype(float)
        key = name.replace(" ", "_")
        out[f"{FEATURE_PREFIX}norm_{key}"] = _safe_ratio(values, reference)

    for left, right in BILATERAL_PAIRS:
        if left in columns.measure and right in columns.measure:
            left_values = frame[columns.measure[left]].astype(float)
            right_values = frame[columns.measure[right]].astype(float)
            key = left.replace("left ", "").replace(" ", "_")
            total = left_values.abs() + right_values.abs()
            out[f"{FEATURE_PREFIX}asym_{key}"] = _safe_ratio(
                (left_values - right_values).abs(), total
            )

    target_column = columns.measure.get(measure)
    if target_column is not None:
        out[f"{FEATURE_PREFIX}target_norm_length"] = _safe_ratio(
            frame[target_column].astype(float), reference
        )
    return out


def build_derived_features(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    measure: str,
) -> pd.DataFrame:
    """Concatenate every stateless derived feature block.

    Everything here is computed row by row and therefore leak-free; the
    group-conditional statistics live in :class:`GroupReferenceStats`.
    """
    blocks = [
        confidence_features(frame, columns, measure),
        geometry_features(frame, columns, measure),
        measurement_features(frame, columns, measure),
    ]
    derived = pd.concat(blocks, axis=1)
    derived = derived.replace([np.inf, -np.inf], np.nan)
    derived = derived.loc[:, ~derived.columns.duplicated()]
    LOGGER.info("Built %d derived features for %r", derived.shape[1], measure)
    return derived


class GroupReferenceStats(BaseEstimator, TransformerMixin):
    """Add deviation-from-group-median features for normalised measurements.

    Fitted on the training fold only. For every normalised measurement, the
    transformer stores the median and the median absolute deviation *per
    taxonomic group* and emits a robust z-score. A large absolute z-score means
    the measured length is anatomically implausible for that group, which is the
    signature of a foreshortened or mislocated keypoint.

    Parameters
    ----------
    group_column:
        Name of the column holding the taxonomic group. It is used for the
        conditioning but is not itself emitted as a feature.
    columns:
        Columns to standardise. Defaults to every ``feat_norm_*`` column.
    """

    def __init__(self, group_column: str = "group", columns: Optional[Sequence[str]] = None):
        self.group_column = group_column
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "GroupReferenceStats":
        self.columns_ = list(
            self.columns
            if self.columns is not None
            else [c for c in X.columns if c.startswith(f"{FEATURE_PREFIX}norm_")]
        )
        self.global_median_ = X[self.columns_].median()
        self.global_scale_ = (X[self.columns_] - self.global_median_).abs().median() + EPS

        self.medians_: Dict[str, pd.Series] = {}
        self.scales_: Dict[str, pd.Series] = {}
        if self.group_column in X.columns:
            for group, subset in X.groupby(self.group_column, observed=True):
                if len(subset) < 10:
                    continue
                median = subset[self.columns_].median()
                self.medians_[group] = median
                self.scales_[group] = (subset[self.columns_] - median).abs().median() + EPS
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if not self.columns_:
            return out

        medians = pd.DataFrame(index=X.index, columns=self.columns_, dtype=float)
        scales = pd.DataFrame(index=X.index, columns=self.columns_, dtype=float)
        for column in self.columns_:
            medians[column] = self.global_median_[column]
            scales[column] = self.global_scale_[column]

        if self.group_column in X.columns:
            for group, median in self.medians_.items():
                mask = X[self.group_column] == group
                if not mask.any():
                    continue
                medians.loc[mask, self.columns_] = median[self.columns_].to_numpy()
                scales.loc[mask, self.columns_] = self.scales_[group][self.columns_].to_numpy()

        z_scores = (X[self.columns_] - medians) / scales
        z_scores.columns = [
            c.replace(f"{FEATURE_PREFIX}norm_", f"{FEATURE_PREFIX}z_") for c in self.columns_
        ]
        return pd.concat([out, z_scores], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)