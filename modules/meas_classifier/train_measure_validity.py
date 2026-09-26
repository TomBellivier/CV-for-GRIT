"""Production training of the measurement-validity classifiers.

For each measurement (27), trains a RandomForestClassifier ("rf_related": coordinates of
the keypoints of the anatomical neighbourhood of the measurement, brought back into
their bounding box, + taxonomic group one-hot) that predicts whether the automatic
measurement is measurable or not, and
saves one ``.joblib`` per measurement in
``retained_models/measurement_validity/``, with the ``metrics.csv`` that carries their
decision thresholds -- these are the models loaded as is by
``pipeline/processing/measurement_classifier.py`` (see
``pipeline/processing/config.py``: ``MEASUREMENT_CLASSIFIER_DIR`` /
``MEASUREMENT_CLASSIFIER_METRICS_CSV``). The research outputs (PR curves, confusion
matrices, importances) go to ``results/meas_classifier/training/`` at the repository
root.

Convention: the positive class is "non measurable" (minority class).
Protocol: stratified 5-fold, out-of-fold predictions; the decision threshold
(``threshold_median`` in ``metrics.csv``) is chosen on an inner partition of the train
set (MCC maximisation), never on the test set.

This is the *production* version: a single approach (rf_related). To compare the 8
approaches evaluated originally (thresholds, XGBoost, Random Forest on different
feature sets), see ``compare_measure_validity_approaches.py`` (research only, saves no
production model).

Usage
-----
::

    python train_measure_validity.py
    python train_measure_validity.py --min-per-class 30 --n-folds 5

Does not duplicate ``dataset.py`` / ``insect_anatomy.py`` (direct import) nor
``evaluation.py`` / ``features.py``, which belong to the old comparison script
``conf_classifier.py`` and are not used here.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no window: PNG files are written
import matplotlib.pyplot as plt

from insect_anatomy import INSECT_GROUPS, MEASUREMENTS, POINTS
from dataset import STATUS_SUFFIX, load_annotation_data

from measure_validity_lib import (
    PRODUCTION_APPROACHES,
    REL_X_SUFFIX,
    REL_Y_SUFFIX,
    add_relative_coordinates,
    evaluate_measure,
    make_coord_columns,
    make_feature_sets,
    make_target,
    metric_rows,
    names_of,
    plot_confusion,
    plot_pr,
    slug,
)

warnings.filterwarnings("ignore")

APPROACHES = PRODUCTION_APPROACHES         # a single approach: rf_related
NAMES = names_of(APPROACHES)
MODEL_NAME, _, MODEL_FEATURES, MODEL_FACTORY = APPROACHES[0]

# --- paths (to adapt) ---------------------------------------------------------
# Anchored on the repository root: the script runs from any folder.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Single annotation table: pose + scale + measurement validity, one row per image
# (see annotation_tools/build_annotation_data.py). It is the ONLY input.
ANNOTATION_DATA = REPO_ROOT / "annotation_data" / "annotation_data.csv"
# Metrics and figures: in the shared results/ folder of the repository (results/README.md).
OUT_DIR = REPO_ROOT / "results" / "meas_classifier" / "training"
# Production models: written directly where pipeline/ reads them, with the metrics.csv
# that carries their decision thresholds (see retained_models/README.md).
MODELS_DIR = REPO_ROOT / "retained_models" / "measurement_validity"

# --- protocol -----------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 0
MIN_MINORITY = 20   # safeguard: measurement skipped below this number
NA_FILL = -1.0       # imputation sentinel for the random forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the production measurement-validity classifiers (rf_related).")
    parser.add_argument("--annotation-data", default=str(ANNOTATION_DATA),
                        help="Annotation table (default: annotation_data/annotation_data.csv).")
    parser.add_argument("--min-per-class", type=int, default=MIN_MINORITY,
                        help="A measurement gets a classifier only if each class (measurable "
                             f"/ non measurable) has at least this many images "
                             f"(default: {MIN_MINORITY}).")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS,
                        help=f"Stratified folds of the out-of-fold evaluation (default: {N_FOLDS}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_data = Path(args.annotation_data)
    min_minority = int(args.min_per_class)
    n_folds = int(args.n_folds)

    for sub in ("pr_curves", "confusion", "importance"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data loading -------------------------------------------------------------
    if not annotation_data.is_file():
        raise SystemExit(
            f"Annotation table missing: {annotation_data}\n"
            "Build it with: python annotation_tools/build_annotation_data.py"
        )
    frame, columns = load_annotation_data(annotation_data)
    group_cols = [f"{g}_one_hot" for g in INSECT_GROUPS]
    print(f"{len(frame)} images | {columns.summary()}")

    # The out-of-fold predictions are stacked measurement by measurement: every
    # measurement must cover the same rows. An image with a single unannotated
    # measurement is therefore left out, rather than counted as non measurable.
    status_columns = list(columns.status.values())
    incomplete = frame[status_columns].isna().any(axis=1)
    if incomplete.any():
        print(f"{int(incomplete.sum())} image(s) with an incomplete annotation left out "
              f"({len(frame) - int(incomplete.sum())} kept)")
        frame = frame.loc[~incomplete].reset_index(drop=True)

    # Prevalence per measurement: used by the safeguard and to read the accuracies.
    prevalence = []
    for measure in MEASUREMENTS:
        status = f"{measure}{STATUS_SUFFIX}"
        if status not in frame.columns:
            continue
        y = 1 - frame[status].astype(int).to_numpy()   # 1 = non measurable
        prevalence.append({
            "measure": measure,
            "n": len(y),
            "n_unmeasurable": int(y.sum()),
            "prevalence_unmeasurable": float(y.mean()),
            "kept": bool(min(y.sum(), len(y) - y.sum()) >= min_minority),
        })
    prevalence = pd.DataFrame(prevalence)
    prevalence.to_csv(OUT_DIR / "prevalence.csv", index=False)

    # A measurement gets its classifier as soon as EACH class (measurable / non
    # measurable) counts at least --min-per-class images: it appears by itself at the next
    # training, without touching anything else (pipeline/ loads every .joblib present).
    kept = prevalence.loc[prevalence["kept"], "measure"].tolist()
    skipped = prevalence.loc[~prevalence["kept"]]
    print(f"{len(kept)} measurements kept, {len(skipped)} skipped (< {min_minority} minority examples)")
    for row in skipped.itertuples(index=False):
        minority = min(row.n_unmeasurable, row.n - row.n_unmeasurable)
        print(f"  skipped: {row.measure} ({minority}/{min_minority} examples of the minority class)")

    # A measurement that falls back below the threshold loses its old model: otherwise
    # pipeline/ would keep loading it, without a decision threshold in the new metrics.csv.
    for measure in MEASUREMENTS:
        stale = MODELS_DIR / f"rf_related_{measure}.joblib"
        if measure not in kept and stale.is_file():
            stale.unlink()
            print(f"  stale model removed: {stale.name}")

    # Features: the keypoint geometry, brought back into its bounding box (see
    # measure_validity_lib.add_relative_coordinates). The pose-model confidences are now
    # only used by the "rule" approaches of the comparison.
    coords = add_relative_coordinates(frame, columns)
    all_coords = make_coord_columns(coords, POINTS)
    print(f"{len(coords)}/{len(POINTS)} keypoints in relative coordinates "
          f"({len(all_coords)} geometric features)")

    # --- Main loop: training + saving + metrics -------------------------------------
    rows, oof = [], {}
    for i, measure in enumerate(kept, start=1):
        print(f"[{i:2d}/{len(kept)}] {measure}", flush=True)
        result = evaluate_measure(
            frame, columns, coords, all_coords, group_cols, measure, STATUS_SUFFIX,
            n_folds, RANDOM_STATE, NA_FILL, APPROACHES, MODELS_DIR,
        )
        rows.extend(metric_rows(measure, result, APPROACHES))
        plot_pr(OUT_DIR, measure, result, APPROACHES)
        plot_confusion(OUT_DIR, measure, result, APPROACHES)
        oof[measure] = {"y": result["y"], MODEL_NAME: result["preds"][MODEL_NAME]}
        del result
        gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\n{len(metrics)} rows written to {OUT_DIR / 'metrics.csv'}")
    # The decision threshold is part of the model: it goes with the .joblib files,
    # otherwise pipeline/ falls back to 0.5 for every measurement.
    metrics.to_csv(MODELS_DIR / "metrics.csv", index=False)
    print(f"Production models and thresholds written to {MODELS_DIR}")

    # --- Performance summary (a single model: no ranking) ----------------------------
    summary = (
        metrics
        .rename(columns={
            "precision_unmeasurable": "precision",
            "accuracy_unmeasurable": "recall",
        })
        [["mcc", "precision", "recall"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.to_csv(OUT_DIR / "summary_mcc_precision_recall.csv")
    print(summary)

    # --- Practical consequence: usable measurements per image ------------------------
    y_true = np.column_stack([oof[m]["y"] for m in kept])        # 1 = non measurable
    valid = y_true == 0                                          # measurement actually usable

    practical_rows, per_image = [], {
        "image_name": frame["image_name"], "group": frame["group"].astype(str),
    }

    def summarise(label, rejected):
        keep = ~rejected
        keep_valid = keep & valid
        per_image[f"n_kept_{label}"] = keep.sum(axis=1)
        per_image[f"n_valid_kept_{label}"] = keep_valid.sum(axis=1)
        practical_rows.append({
            "filtering": label,
            "kept_measures_per_image": keep.sum(axis=1).mean(),
            "of_which_actually_valid": keep_valid.sum(axis=1).mean(),
            "valid_lost_per_image": (valid & rejected).sum(axis=1).mean(),
            "valid_retention": keep_valid.sum() / valid.sum(),
            "kept_contamination": (keep & ~valid).sum() / max(keep.sum(), 1),
        })

    summarise("no_filtering", np.zeros_like(y_true, dtype=bool))
    predicted = np.column_stack([oof[m][MODEL_NAME] for m in kept])
    summarise(MODEL_NAME, predicted == 1)

    practical = pd.DataFrame(practical_rows).set_index("filtering").round(3)
    practical.to_csv(OUT_DIR / "filtering_impact.csv")
    pd.DataFrame(per_image).to_csv(OUT_DIR / "measures_per_image.csv", index=False)
    print(f"\n{len(kept)} measurements evaluated, {valid.mean():.1%} actually usable\n")

    # --- Label distribution per measurement (stacked bars) ----------------------------
    order = prevalence.sort_values("prevalence_unmeasurable")
    n_ok = (order["n"] - order["n_unmeasurable"]).to_numpy()
    n_ko = order["n_unmeasurable"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(order) + 2))
    ax.barh(order["measure"], n_ok, color="#4c9f70", label="measurable")
    ax.barh(order["measure"], n_ko, left=n_ok, color="#c0504d", label="non measurable")
    for i, (a, b) in enumerate(zip(n_ok, n_ko)):
        ax.text(a + b + order["n"].max() * 0.01, i, f"{b / (a + b):.0%}", va="center", fontsize=7)
    ax.set_xlim(0, order["n"].max() * 1.09)
    ax.set_xlabel("number of images")
    ax.set_title("Label repartition per measure (sorted by non-measurable ratio)")
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "labels_per_measure.png", dpi=140)
    plt.close(fig)
    print("written:", OUT_DIR / "labels_per_measure.png")

    # --- Feature importance -----------------------------------------------------------
    # The "rf_related" model is retrained on the whole data, measurement by
    # measurement. These importances are descriptive: they show which keypoints carry
    # the signal, they do not measure a performance.
    def pretty(column: str) -> str:
        """'head-top kp_x_rel' -> 'head-top x', 'coleoptera_one_hot' -> 'coleoptera'."""
        for suffix, axis in ((REL_X_SUFFIX, " x"), (REL_Y_SUFFIX, " y")):
            if column.endswith(suffix):
                return column[: -len(suffix)] + axis
        return column.replace("_one_hot", "")

    series = []
    for measure in kept:
        cols = make_feature_sets(columns, coords, all_coords, measure)[MODEL_FEATURES]
        if not cols:
            continue
        y = make_target(frame, STATUS_SUFFIX, measure)
        data = frame[cols + group_cols].fillna(NA_FILL)

        model = MODEL_FACTORY(y, RANDOM_STATE)
        model.fit(data, y)
        values = pd.Series(model.feature_importances_, index=[pretty(c) for c in data.columns])
        del model
        gc.collect()
        series.append(values.rename(measure))

        top = values.sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top) + 1.6))
        ax.barh(top.index, top.to_numpy(), color="steelblue")
        ax.set_xlabel("importance")
        ax.set_title(f"{MODEL_NAME}: {measure}", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "importance" / f"{slug(measure)}.png", dpi=140)
        plt.close(fig)

    importance = pd.concat(series, axis=1).T
    importance.to_csv(OUT_DIR / "feature_importance.csv")
    print(f"{len(importance)} measurements, {importance.shape[1]} variables")

    mean_importance = importance.mean().sort_values().tail(20)
    coverage = importance.notna().sum()

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(mean_importance) + 2))
    ax.barh(mean_importance.index, mean_importance.to_numpy(), color="#4472c4")
    for i, name in enumerate(mean_importance.index):
        ax.text(mean_importance[name], i, f"  {coverage[name]}/{len(importance)}",
                va="center", fontsize=7, color="grey")
    ax.set_xlim(0, float(mean_importance.max()) * 1.12)
    ax.set_xlabel("mean importance (grey number: number of measures where the variable exists)")
    ax.set_title(f"Most used variables ({MODEL_NAME})")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_importance.png", dpi=140)
    plt.close(fig)
    print("written:", OUT_DIR / "mean_importance.png")

    # --- Measurements whose label is almost determined by the taxonomic group ---------
    suspects = []
    for measure in kept:
        y = pd.Series(make_target(frame, STATUS_SUFFIX, measure))
        by_group = y.groupby(frame["group"].astype(str).to_numpy()).mean()
        if ((by_group < 0.05) | (by_group > 0.95)).all():
            suspects.append({"measure": measure, **by_group.round(2).to_dict()})

    suspects = pd.DataFrame(suspects)
    if not suspects.empty:
        suspects.to_csv(OUT_DIR / "group_determined_measures.csv", index=False)
        print("Non-measurable rate per group (trivial measurements):")
        print(suspects)


if __name__ == "__main__":
    main()
