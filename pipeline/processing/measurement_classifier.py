"""
measurement_classifier.py
==========================

Scores each measurement of a processed image with its pre-trained validity
classifier ("is this automatic measurement trustworthy?"). Inference only:
the classifiers are trained offline, once, by
    ../modules/meas_classifier/train_measure_validity.py
(the "rf_related" approach: a random forest per measurement, fed the keypoint
COORDINATES of its anatomical neighbourhood -- expressed inside the bounding box
of the instance's keypoints -- plus the insect's taxonomic group)
and saved as one joblib file per measurement, next to the metrics.csv holding
their decision thresholds, under
    ../retained_models/measurement_validity/rf_related_<measure>.joblib
Nothing is trained here, and no new place in this pipeline trains a model.

The anatomical neighbourhood ("related" keypoints) and the taxonomic groups
come from kp_infos.yaml, like at training time (see definitions.py).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .definitions import INSECT_GROUPS, KEYPOINT_INDEX, MEASUREMENT_NAMES, related_entities

NA_FILL = -1.0                 # same sentinel used to train the random forests
DEFAULT_THRESHOLD = 0.5        # fallback when no tuned threshold is available

# Feature space: the GEOMETRY of the keypoints, not the pose model's confidences.
# Each point is expressed inside the bounding box of the instance's keypoints,
#     x_rel = (x - cx) / w        y_rel = (y - cy) / h
# which makes the features independent of image resolution and framing.
#
# /!\ This mirrors add_relative_coordinates() in
# ../modules/meas_classifier/measure_validity_lib.py, which is the source of
# truth. The two MUST stay identical: a model fed features that no longer mean
# what they meant at training time returns confident nonsense. The guard in
# load_measurement_classifiers() catches a mismatch of feature NAMES; it cannot
# catch a mismatch of their definition.
REL_X_SUFFIX = " kp_x_rel"
REL_Y_SUFFIX = " kp_y_rel"

VALID_PROBA_SUFFIX = " [valid_proba]"
VALID_PRED_SUFFIX = " [valid_pred]"


def relative_keypoints(keypoints) -> dict[str, float]:
    """Keypoint coordinates inside their own bounding box, as named features.

    A keypoint that is missing (non-finite) or sits exactly at (0, 0) -- what an
    undetected point looks like -- is left out of the box and comes back NaN, so
    the forest sees the absence sentinel rather than a fake position at the
    origin. Returns {} when fewer than two usable points remain, or when the box
    is flat: there is then no frame to express anything in.
    """
    xy = np.asarray(keypoints, dtype=float)[:, :2]
    usable = np.isfinite(xy).all(axis=1) & ~((xy[:, 0] == 0) & (xy[:, 1] == 0))
    if usable.sum() < 2:
        return {}

    x_min, y_min = xy[usable].min(axis=0)
    x_max, y_max = xy[usable].max(axis=0)
    width, height = x_max - x_min, y_max - y_min
    if not (width > 0 and height > 0):
        return {}
    cx, cy = x_min + width / 2.0, y_min + height / 2.0

    features: dict[str, float] = {}
    for name, index in KEYPOINT_INDEX.items():
        if index >= len(xy) or not usable[index]:
            continue
        features[f"{name}{REL_X_SUFFIX}"] = (xy[index, 0] - cx) / width
        features[f"{name}{REL_Y_SUFFIX}"] = (xy[index, 1] - cy) / height
    return features


def _load_thresholds(metrics_csv: Path) -> dict[str, float]:
    """Per-measurement decision threshold (median over CV folds, rf_related)."""
    if not metrics_csv.is_file():
        print(f"[measure-clf] no metrics CSV at {metrics_csv} -> using the default "
              f"threshold ({DEFAULT_THRESHOLD}) for every measurement.")
        return {}
    df = pd.read_csv(metrics_csv)
    sub = df[df["model"] == "rf_related"]
    return dict(zip(sub["measure"], sub["threshold_median"]))


class MeasurementClassifiers:
    """Loaded once, shared read-only across worker threads.

    Unlike the YOLO models (which need one instance per thread, see worker.py),
    a fitted RandomForestClassifier's predict_proba() does not mutate the
    estimator, so the same models can be scored concurrently from every thread.
    """

    def __init__(self, models: dict[str, object], thresholds: dict[str, float]):
        self.models = models
        self.thresholds = thresholds
        # Precomputed once: which keypoints feed each measurement's model.
        self._related_points = {name: related_entities(name)[0] for name in models}

    def score(self, keypoints, group: str | None) -> dict[str, dict[str, float]]:
        """{'<measure>': {'proba': float, 'pred': 0/1}} for every scored measure.

        'proba' is the probability the measurement IS trustworthy (measurable);
        'pred' is that same call thresholded at the measure's tuned cutoff.
        """
        out: dict[str, dict[str, float]] = {}
        if keypoints is None or not self.models:
            return out

        group_features = {f"{g}_one_hot": (1.0 if g == group else 0.0) for g in INSECT_GROUPS}
        # One frame per image: every measurement of this instance is expressed in
        # the same box, exactly as at training time.
        relative = relative_keypoints(keypoints)
        for name, model in self.models.items():
            features = {
                column: relative[column]
                for p in self._related_points[name] if p in KEYPOINT_INDEX
                for column in (f"{p}{REL_X_SUFFIX}", f"{p}{REL_Y_SUFFIX}")
                if column in relative
            }
            features.update(group_features)

            row = pd.DataFrame(
                [[features.get(col, np.nan) for col in model.feature_names_in_]],
                columns=model.feature_names_in_,
            ).fillna(NA_FILL)

            proba_unmeasurable = float(model.predict_proba(row)[0, 1])
            threshold = self.thresholds.get(name, DEFAULT_THRESHOLD)
            out[name] = {
                "proba": 1.0 - proba_unmeasurable,
                "pred": 0 if proba_unmeasurable >= threshold else 1,
            }
        return out


def load_measurement_classifiers(models_dir=None, metrics_csv=None) -> MeasurementClassifiers:
    models_dir = Path(models_dir or config.MEASUREMENT_CLASSIFIER_DIR)
    thresholds = _load_thresholds(Path(metrics_csv or config.MEASUREMENT_CLASSIFIER_METRICS_CSV))

    models: dict[str, object] = {}
    if not models_dir.is_dir():
        print(f"[measure-clf] models dir not found: {models_dir} -> "
              f"measurement-validity columns will be empty.")
        return MeasurementClassifiers(models, thresholds)

    buildable = {f"{p}{suffix}" for p in KEYPOINT_INDEX for suffix in (REL_X_SUFFIX, REL_Y_SUFFIX)}
    buildable |= {f"{g}_one_hot" for g in INSECT_GROUPS}
    stale: list[str] = []

    for name in MEASUREMENT_NAMES:
        path = models_dir / f"rf_related_{name}.joblib"
        if path.is_file():
            model = joblib.load(path)
            # A model whose features this pipeline cannot build would be scored on
            # a row of NaN -> NA_FILL: a confident answer to a question it was
            # never asked. Models trained on keypoint confidences (the feature set
            # used before the switch to geometry) land here.
            unknown = [c for c in getattr(model, "feature_names_in_", []) if c not in buildable]
            if unknown:
                stale.append(name)
                del model
                continue
            # Saved with n_jobs=-1 (useful for batch training, not for scoring
            # one row at a time): left as-is, every predict_proba() call spins
            # up its own internal thread pool across cores purely for
            # overhead, and 16 pipeline workers x 26 measures makes that
            # contention real. Forcing n_jobs=1 is ~4x faster per call.
            model.n_jobs = 1
            models[name] = model

    if stale:
        print(f"[measure-clf] /!\\ {len(stale)} classifier(s) in {models_dir} expect features "
              f"this pipeline no longer produces (e.g. keypoint confidences): they are "
              f"IGNORED, and their CSV columns will be empty. Retrain them with "
              f"modules/meas_classifier/train_measure_validity.py. First: {stale[:3]}")
    print(f"[measure-clf] loaded {len(models)}/{len(MEASUREMENT_NAMES)} "
          f"measurement-validity classifiers from {models_dir}")
    return MeasurementClassifiers(models, thresholds)
