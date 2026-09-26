"""Functions reused by ``train_measure_validity.py`` (production) and
``compare_measure_validity_approaches.py`` (comparison, research).

Extracted from ``measure_validity_classifiers.ipynb`` (parts 3 to 5: feature sets,
approaches, score/threshold/cross-validation, metrics and figures), without changing
the behaviour. Does not duplicate ``evaluation.py`` / ``features.py``, which belong to
the old comparison script ``conf_classifier.py`` and are not used by this notebook.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import joblib

from insect_anatomy import MEAS_TO_KP, POINTS, related_entities


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --------------------------------------------------------------------------- #
# 3. Feature sets and approaches
# --------------------------------------------------------------------------- #
# The production models work on the GEOMETRY of the keypoints, not on the
# confidences of the pose model: two stacked points or a point sent outside the body
# show in the coordinates, not in a score.
#
# Raw coordinates would be unusable as is (a 278x679 database and a 4000x3000 one
# would have nothing in common): each point is therefore brought back into the
# bounding box of the keypoints of ITS image,
#     x_rel = (x - cx) / w        y_rel = (y - cy) / h
# which makes the features invariant to the resolution as well as to the framing. The
# values then live around [-0.5, 0.5].
#
# /!\ This transformation is replayed identically at inference, in
# pipeline/processing/measurement_classifier.py (relative_keypoints). Any change here
# MUST be carried over there, otherwise the models receive features that no longer
# mean what they meant at training time.
REL_X_SUFFIX = " kp_x_rel"
REL_Y_SUFFIX = " kp_y_rel"


def add_relative_coordinates(frame: pd.DataFrame, columns) -> dict:
    """Add to ``frame`` the relative coordinates of each resolved keypoint.

    Returns {point: (x_column, y_column)}. A missing keypoint (NaN) or one placed
    exactly at (0, 0) -- the output of an undetected point -- does not enter the box
    and comes out as NaN, which the forest handles as the absence sentinel.
    """
    points = [p for p in POINTS if p in columns.x and p in columns.y]
    if not points:
        raise KeyError(
            "No coordinate column (kp_x / kp_y) in the pose outputs: "
            "check that the pipeline ran with config.EXPORT_KEYPOINTS = True."
        )

    xs = frame[[columns.x[p] for p in points]].to_numpy(dtype=float)
    ys = frame[[columns.y[p] for p in points]].to_numpy(dtype=float)

    missing = (~np.isfinite(xs)) | (~np.isfinite(ys)) | ((xs == 0) & (ys == 0))
    xs = np.where(missing, np.nan, xs)
    ys = np.where(missing, np.nan, ys)

    with np.errstate(invalid="ignore"):
        # all-NaN on a row (no keypoint): nanmin warns, NaN is wanted.
        empty = np.isnan(xs).all(axis=1)
        x_min, x_max = np.nanmin(np.where(empty[:, None], 0.0, xs), axis=1), \
            np.nanmax(np.where(empty[:, None], 0.0, xs), axis=1)
        y_min, y_max = np.nanmin(np.where(empty[:, None], 0.0, ys), axis=1), \
            np.nanmax(np.where(empty[:, None], 0.0, ys), axis=1)

    width, height = x_max - x_min, y_max - y_min
    degenerate = empty | ~(width > 0) | ~(height > 0)   # 0 or 1 point, or all aligned
    width = np.where(degenerate, np.nan, width)
    height = np.where(degenerate, np.nan, height)
    cx, cy = x_min + width / 2.0, y_min + height / 2.0

    coords = {}
    for i, point in enumerate(points):
        x_col, y_col = f"{point}{REL_X_SUFFIX}", f"{point}{REL_Y_SUFFIX}"
        frame[x_col] = (xs[:, i] - cx) / width
        frame[y_col] = (ys[:, i] - cy) / height
        coords[point] = (x_col, y_col)

    n_degenerate = int(degenerate.sum())
    if n_degenerate:
        print(f"  /!\\ {n_degenerate} image(s) without a usable keypoint box: "
              f"geometric features set to NaN")
    return coords


def make_coord_columns(coords: dict, points) -> list:
    """Relative coordinate columns (x then y) for a list of keypoints."""
    return [column for p in points if p in coords for column in coords[p]]


def make_conf_columns(columns, points) -> list:
    """Existing confidence columns for a list of keypoints."""
    return [columns.conf[p] for p in points if p in columns.conf]


def make_feature_sets(columns, coords: dict, all_coords: list, measure: str) -> dict:
    """Feature sets of a measurement: its keypoints, its neighbourhood, or all of them.

    ``*_conf`` stays available for the "rule" approaches (threshold on the
    confidence), which only make sense on confidences.
    """
    return {
        "direct": make_coord_columns(coords, MEAS_TO_KP[measure]),
        "related": make_coord_columns(coords, related_entities(measure)[0]),
        "all": all_coords,
        "direct_conf": make_conf_columns(columns, MEAS_TO_KP[measure]),
        "related_conf": make_conf_columns(columns, related_entities(measure)[0]),
    }


def make_target(frame: pd.DataFrame, status_suffix: str, measure: str) -> np.ndarray:
    """1 = non measurable (positive, minority class)."""
    return 1 - frame[f"{measure}{status_suffix}"].astype(int).to_numpy()


def make_xgb(y_train: np.ndarray, random_state: int) -> XGBClassifier:
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=n_neg / n_pos,          # imbalance
        eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=random_state,
    )


def make_rf(y_train: np.ndarray, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        class_weight="balanced",                  # imbalance
        n_jobs=-1, random_state=random_state,
    )


# (key, type, feature set, model factory)
#
# PRODUCTION_APPROACHES: only "rf_related" -- the set of models saved and consumed by
# pipeline/processing/measurement_classifier.py. Used by train_measure_validity.py.
PRODUCTION_APPROACHES = [
    ("rf_related", "model", "related", make_rf),
]

# ALL_APPROACHES: the 8 approaches compared in measure_validity_classifiers.ipynb
# (thresholds, XGBoost and Random Forest on 3 feature sets). Research / comparison
# only -- used by compare_measure_validity_approaches.py, produces no production model.
ALL_APPROACHES = [
    ("rule_conf_mean", "rule", "direct", None),
    ("rule_conf_min", "rule", "direct", None),
    ("xgb_direct", "model", "direct", make_xgb),
    ("rf_direct", "model", "direct", make_rf),
    ("xgb_related", "model", "related", make_xgb),
    ("rf_related", "model", "related", make_rf),
    ("xgb_all", "model", "all", make_xgb),
    ("rf_all", "model", "all", make_rf),
]


def names_of(approaches: list) -> list:
    return [a[0] for a in approaches]


# --------------------------------------------------------------------------- #
# 4. Score, threshold and cross-validation
# --------------------------------------------------------------------------- #
def rule_score(values: pd.DataFrame, how: str) -> np.ndarray:
    data = np.nan_to_num(values.to_numpy(dtype=float), nan=0.0)  # missing kp -> conf 0
    agg = data.mean(axis=1) if how == "mean" else data.min(axis=1)
    return -agg   # low confidence -> high score -> non measurable


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising the MCC, searched on data not seen during training."""
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 201)))
    best_t, best_m = candidates[0], -np.inf
    for t in candidates:
        m = matthews_corrcoef(y, (score >= t).astype(int))
        if m > best_m:
            best_t, best_m = t, m
    return float(best_t)


def evaluate_measure(
    frame: pd.DataFrame,
    columns,
    coords: dict,
    all_coords: list,
    group_cols: list,
    measure: str,
    status_suffix: str,
    n_folds: int,
    random_state: int,
    na_fill: float,
    approaches: list,
    models_dir: Path | None = None,
) -> dict:
    """Out-of-fold predictions of the ``approaches`` for one measurement.

    If ``models_dir`` is given, saves the "rf_related" model (if it is part of
    ``approaches``) trained on the inner partition (``inner_fit``) of EACH outer fold
    as ``models_dir/rf_related_<measure>.joblib``: the final file is therefore that of
    the last outer fold processed (same behaviour as the notebook).
    """
    names = names_of(approaches)
    y = make_target(frame, status_suffix, measure)
    sets = make_feature_sets(columns, coords, all_coords, measure)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scores = {name: np.zeros(len(y)) for name in names}
    preds = {name: np.zeros(len(y), dtype=int) for name in names}
    thresholds = {name: [] for name in names}

    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        # inner partition: only used to set the decision threshold
        inner_fit, inner_val = train_test_split(
            train_idx, test_size=0.25, stratify=y[train_idx], random_state=random_state
        )
        for name, kind, which, factory in approaches:
            # A threshold applies to a confidence, not to a coordinate.
            cols = sets[f"{which}_conf"] if kind == "rule" else sets[which]
            if not cols:
                continue
            if kind == "rule":
                how = "mean" if name.endswith("mean") else "min"
                val_score = rule_score(frame.iloc[inner_val][cols], how)
                test_score = rule_score(frame.iloc[test_idx][cols], how)
            else:
                data = frame[cols + group_cols]
                if name.startswith("rf"):
                    data = data.fillna(na_fill)   # the RF does not handle NaN
                model = factory(y[inner_fit], random_state)
                model.fit(data.iloc[inner_fit], y[inner_fit])
                if name == "rf_related" and models_dir is not None:
                    joblib.dump(model, models_dir / f"rf_related_{measure}.joblib")
                val_score = model.predict_proba(data.iloc[inner_val])[:, 1]
                del model
                gc.collect()
                model = factory(y[train_idx], random_state)
                model.fit(data.iloc[train_idx], y[train_idx])
                test_score = model.predict_proba(data.iloc[test_idx])[:, 1]
                del model
                gc.collect()

            threshold = best_threshold(y[inner_val], val_score)
            scores[name][test_idx] = test_score
            preds[name][test_idx] = (test_score >= threshold).astype(int)
            thresholds[name].append(threshold)

    return {"y": y, "scores": scores, "preds": preds, "thresholds": thresholds}


# --------------------------------------------------------------------------- #
# 5. Metrics and figures
# --------------------------------------------------------------------------- #
def metric_rows(measure: str, result: dict, approaches: list) -> list:
    y = result["y"]
    prev = float(y.mean())
    rows = []
    for name in names_of(approaches):
        score, pred = result["scores"][name], result["preds"][name]
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        ap = float(average_precision_score(y, score))
        rows.append({
            "measure": measure,
            "model": name,
            "n": len(y),
            "prevalence_unmeasurable": prev,
            "mcc": float(matthews_corrcoef(y, pred)),
            "average_precision": ap,
            "average_precision_norm": (ap - prev) / (1 - prev) if prev < 1 else np.nan,
            "accuracy": float((tp + tn) / len(y)),
            "accuracy_unmeasurable": float(tp / (tp + fn)) if (tp + fn) else np.nan,
            "accuracy_measurable": float(tn / (tn + fp)) if (tn + fp) else np.nan,
            "balanced_accuracy": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
            "precision_unmeasurable": float(tp / (tp + fp)) if (tp + fp) else np.nan,
            "threshold_median": float(np.median(result["thresholds"][name])),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return rows


def plot_pr(out_dir: Path, measure: str, result: dict, approaches: list) -> None:
    y = result["y"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in names_of(approaches):
        score = result["scores"][name]
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(recall, precision, lw=1.6, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y.mean(), color="grey", ls="--", lw=1, label=f"chance ({y.mean():.3f})")
    ax.set_xlabel("Recall (non measurable)")
    ax.set_ylabel("Precision (non measurable)")
    ax.set_title(f"Precision-recall curve: {measure}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves" / f"{slug(measure)}.png", dpi=140)
    plt.close(fig)


def plot_confusion(out_dir: Path, measure: str, result: dict, approaches: list) -> None:
    y = result["y"]
    names = names_of(approaches)
    labels = ["measurable", "non meas."]
    n_cols = min(len(names), 4) or 1
    n_rows = -(-len(names) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.25 * n_cols, 3.5 * n_rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        matrix = confusion_matrix(y, result["preds"][name], labels=[0, 1])
        normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}\n{normalised[i, j]:.0%}",
                        ha="center", va="center", fontsize=9,
                        color="white" if normalised[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{name}\nMCC={matthews_corrcoef(y, result['preds'][name]):.3f}", fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices (out-of-fold): {measure}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion" / f"{slug(measure)}.png", dpi=140)
    plt.close(fig)
