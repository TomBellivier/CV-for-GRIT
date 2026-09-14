"""
measurement_classifier.py
==========================

Scores each measurement of a processed image with its pre-trained validity
classifier ("is this automatic measurement trustworthy?"). Inference only:
the classifiers are trained offline, once, in
    ../conf_classifier/measure_validity_classifiers.ipynb
(the "rf_related" approach: a random forest per measurement, fed the keypoint
confidences of its anatomical neighbourhood plus the insect's taxonomic group)
and saved as one joblib file per measurement under
    trained_models/measurement_classification_models/rf_related_<measure>.joblib
Nothing is trained here, and no new place in this pipeline trains a model.

The anatomical neighbourhood ("related" keypoints) and the taxonomic groups
are defined once in the sibling conf_classifier project and imported from
there rather than duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .definitions import KEYPOINT_INDEX, MEASUREMENT_NAMES

_CONF_CLASSIFIER_DIR = config.PROJECT_ROOT.parent / "conf_classifier"
if str(_CONF_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONF_CLASSIFIER_DIR))
from .insect_anatomy import INSECT_GROUPS, related_entities  # noqa: E402

NA_FILL = -1.0                 # same sentinel used to train the random forests
DEFAULT_THRESHOLD = 0.5        # fallback when no tuned threshold is available

VALID_PROBA_SUFFIX = " [valid_proba]"
VALID_PRED_SUFFIX = " [valid_pred]"


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
        for name, model in self.models.items():
            features = {
                f"{p} kp_conf": float(keypoints[KEYPOINT_INDEX[p], 2])
                for p in self._related_points[name] if p in KEYPOINT_INDEX
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

    for name in MEASUREMENT_NAMES:
        path = models_dir / f"rf_related_{name}.joblib"
        if path.is_file():
            model = joblib.load(path)
            # Saved with n_jobs=-1 (useful for batch training, not for scoring
            # one row at a time): left as-is, every predict_proba() call spins
            # up its own internal thread pool across cores purely for
            # overhead, and 16 pipeline workers x 26 measures makes that
            # contention real. Forcing n_jobs=1 is ~4x faster per call.
            model.n_jobs = 1
            models[name] = model
    print(f"[measure-clf] loaded {len(models)}/{len(MEASUREMENT_NAMES)} "
          f"measurement-validity classifiers from {models_dir}")
    return MeasurementClassifiers(models, thresholds)
