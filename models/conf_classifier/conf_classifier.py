"""Measurement-validity classification from insect pose-estimation outputs.

For a given morphometric measurement, the model predicts whether the automatic
measurement is trustworthy ("measurable") or not, from the pose model's own
outputs: keypoint confidences, keypoint pixel positions and the measured
distances.

Usage
-----
::

    python conf_classifier.py --measure "head length"
    python conf_classifier.py --measure "head length" --n-repeats 4 --no-shap
    python conf_classifier.py --list-measures

Outputs, written to ``results/<measure>/``:

``folds.csv``
    One row per configuration, fold and evaluation scope. This is the file to
    keep: every aggregate and every statistical comparison is recomputed from
    it, and paired tests are only possible with the raw per-fold values.
``summary.csv``
    Mean and standard deviation of every metric, plus the uplift over the
    majority-class baseline computed on the same folds.
``paired_vs_<reference>.csv``
    Fold-wise comparison of every configuration against the reference one.
``labels.csv``
    Class prevalence per taxonomic group, with the baseline accuracy.
``*.png``
    ROC envelope, accuracy against baseline, SHAP beeswarm.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from dataset import STATUS_SUFFIX, ColumnIndex, build_dataset, label_report
from evaluation import (
    GROUP_COLUMN,
    RunConfig,
    build_pipeline,
    evaluate_config,
    paired_comparison,
    select_feature_columns,
    summarise,
)
from features import build_derived_features
from insect_anatomy import INSECT_GROUPS, MEASUREMENTS

LOGGER = logging.getLogger("conf_classifier")

DATA_DIR = Path("./data")
DATABASE_DIR = Path("../../databases/full databases")
PROCESS_FOLDER_RESULTS = Path("../process_folder/results_4.csv")
RESULTS_DIR = Path("./results")

DEFAULT_MEASURE = "head length"
REFERENCE_RUN = "raw_conf_meas_pos"


def build_run_configs() -> List[RunConfig]:
    """The configuration grid.

    Kept deliberately factorial so that each effect can be isolated: the raw
    families alone, the effect of the anatomical restriction, the effect of the
    taxonomic one-hot, the effect of PCA at a *constant* number of components,
    and the effect of the engineered features. In the previous grid the number
    of components varied between runs (5 or 10), which confounded the PCA effect
    with its dimensionality.
    """
    configs: List[RunConfig] = []

    for family in ("conf", "meas", "pos"):
        configs.append(RunConfig(name=f"raw_{family}", families=(family,)))
        configs.append(
            RunConfig(name=f"related_{family}", families=(family,), related_only=True)
        )
        configs.append(
            RunConfig(name=f"one_hot_{family}", families=(family,), one_hot=True)
        )

    all_families = ("conf", "meas", "pos")
    configs += [
        RunConfig(name="raw_conf_meas_pos", families=all_families),
        RunConfig(name="related_conf_meas_pos", families=all_families, related_only=True),
        RunConfig(name="one_hot_conf_meas_pos", families=all_families, one_hot=True),
        RunConfig(
            name="related_one_hot_conf_meas_pos",
            families=all_families,
            related_only=True,
            one_hot=True,
        ),
        RunConfig(name="pca10_conf_meas_pos", families=all_families, n_pca=10),
        RunConfig(
            name="pca10_one_hot_conf_meas_pos",
            families=all_families,
            one_hot=True,
            n_pca=10,
        ),
        RunConfig(
            name="pca10_related_conf_meas_pos",
            families=all_families,
            related_only=True,
            n_pca=10,
        ),
        # Engineered features: added on their own, then on top of the raw block.
        RunConfig(name="derived_only", derived=True),
        RunConfig(name="derived_group_stats", derived=True, group_stats=True),
        RunConfig(
            name="derived_group_stats_one_hot",
            derived=True,
            group_stats=True,
            one_hot=True,
        ),
        RunConfig(
            name="full", families=all_families, derived=True, group_stats=True, one_hot=True
        ),
        RunConfig(
            name="full_balanced",
            families=all_families,
            derived=True,
            group_stats=True,
            one_hot=True,
            balance=True,
        ),
        RunConfig(
            name="full_related",
            families=all_families,
            derived=True,
            group_stats=True,
            one_hot=True,
            related_only=True,
        ),
    ]
    return configs


def plot_roc_envelope(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    config: RunConfig,
    measure: str,
    output_path: Path,
    n_curves: int = 20,
    seed: int = 42,
) -> None:
    """Mean ROC with a +/- 1 sd band, on a subsample of folds.

    Plotting a hundred overlapping curves hides the dispersion; the mean curve
    interpolated on a common FPR grid with its band is readable and comparable
    between configurations.
    """
    from sklearn.model_selection import StratifiedKFold

    feature_columns = select_feature_columns(frame, columns, config, measure)
    X = frame[feature_columns].copy()
    X[GROUP_COLUMN] = frame[GROUP_COLUMN].astype(str).to_numpy()
    y = frame[f"{measure}{STATUS_SUFFIX}"].to_numpy()

    grid = np.linspace(0.0, 1.0, 101)
    curves = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        if fold_id >= n_curves:
            break
        pipeline = build_pipeline(config)
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[test_idx], proba)
        curves.append(np.interp(grid, fpr, tpr))

    curves = np.vstack(curves)
    mean_tpr, std_tpr = curves.mean(axis=0), curves.std(axis=0)

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.plot(grid, mean_tpr, color="tab:blue", label="Mean ROC")
    axes.fill_between(
        grid,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color="tab:blue",
        alpha=0.2,
        label="+/- 1 sd",
    )
    axes.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(f"{config.name} - {measure}")
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_accuracy_against_baseline(
    summary: pd.DataFrame,
    run_name: str,
    measure: str,
    output_path: Path,
) -> None:
    """Per-group accuracy with the majority-class baseline overlaid.

    Without the baseline, a 96 % accuracy on a group that is 96.7 % positive
    reads as a success rather than as a model that has learnt nothing.
    """
    subset = summary[summary["run_name"] == run_name].set_index("scope")
    scopes = [s for s in ["ALL"] + INSECT_GROUPS if s in subset.index]
    accuracy = subset.loc[scopes, "accuracy_mean"].to_numpy()
    errors = subset.loc[scopes, "accuracy_std"].to_numpy()
    baseline = subset.loc[scopes, "accuracy_baseline_mean"].to_numpy()
    balanced = subset.loc[scopes, "balanced_accuracy_mean"].to_numpy()

    positions = np.arange(len(scopes))
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.bar(positions - 0.2, accuracy, width=0.4, yerr=errors, label="Accuracy", capsize=3)
    axes.bar(positions + 0.2, balanced, width=0.4, label="Balanced accuracy")
    axes.plot(
        positions,
        baseline,
        marker="D",
        linestyle="none",
        color="black",
        label="Majority-class baseline",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(scopes, rotation=20, ha="right")
    axes.set_ylim(0.0, 1.05)
    axes.set_ylabel("Score")
    axes.set_title(f"{run_name} - {measure}")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_shap_summary(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    config: RunConfig,
    measure: str,
    output_path: Path,
    seed: int = 42,
) -> None:
    """SHAP beeswarm computed on held-out data.

    Explaining the training fold shows what the model memorised; explaining a
    held-out fold shows what it generalised.
    """
    try:
        import shap
    except ImportError:
        LOGGER.warning("shap is not installed, skipping explanation plots")
        return

    from sklearn.model_selection import train_test_split

    feature_columns = select_feature_columns(frame, columns, config, measure)
    X = frame[feature_columns].copy()
    X[GROUP_COLUMN] = frame[GROUP_COLUMN].astype(str).to_numpy()
    y = frame[f"{measure}{STATUS_SUFFIX}"].to_numpy()

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    transformed = X_test
    for name, step in pipeline.steps[:-1]:
        transformed = step.transform(transformed)

    explainer = shap.TreeExplainer(pipeline["model"])
    shap_values = explainer(transformed)
    shap.plots.beeswarm(shap_values, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close("all")


def run(
    measure: str,
    data_dir: Path,
    database_dir: Path,
    results_path: Path,
    output_root: Path,
    n_splits: int,
    n_repeats: int,
    seed: int,
    tune_threshold: bool,
    make_shap: bool,
    only: Optional[Sequence[str]] = None,
) -> None:
    """Full experiment for one measurement."""
    frame, columns = build_dataset(data_dir, database_dir, results_path)

    status_column = f"{measure}{STATUS_SUFFIX}"
    if status_column not in frame.columns:
        raise KeyError(
            f"No annotation column {status_column!r}. Available measures: "
            f"{sorted(c[: -len(STATUS_SUFFIX)] for c in frame.columns if c.endswith(STATUS_SUFFIX))}"
        )

    derived = build_derived_features(frame, columns, measure)
    frame = pd.concat([frame, derived], axis=1)

    output_dir = output_root / measure.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = label_report(frame, measure)
    labels.to_csv(output_dir / "labels.csv", index=False)
    LOGGER.info("Class prevalence:\n%s", labels.to_string(index=False))

    configs = build_run_configs()
    if only:
        wanted = set(only)
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            raise ValueError(f"No configuration matches {sorted(wanted)}")

    all_folds: List[pd.DataFrame] = []
    for config in configs:
        LOGGER.info("=== %s ===", config.name)
        fold_results = evaluate_config(
            frame,
            columns,
            config,
            measure,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            tune_decision_threshold=tune_threshold,
        )
        all_folds.append(fold_results)

        overall = fold_results[fold_results["scope"] == "ALL"]
        LOGGER.info(
            "AUC %.4f  PR-AUC(bad) %.4f  bal.acc %.4f  acc %.4f (baseline %.4f)",
            overall["roc_auc"].mean(),
            overall["pr_auc_bad"].mean(),
            overall["balanced_accuracy"].mean(),
            overall["accuracy"].mean(),
            overall["accuracy_baseline"].mean(),
        )

        plot_roc_envelope(
            frame, columns, config, measure, output_dir / f"{config.name}_roc.png", seed=seed
        )
        if make_shap and config.n_pca == 0:
            plot_shap_summary(
                frame, columns, config, measure, output_dir / f"{config.name}_shap.png", seed=seed
            )

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(output_dir / "folds.csv", index=False)

    summary = summarise(folds)
    summary.to_csv(output_dir / "summary.csv", index=False)

    reference = REFERENCE_RUN if REFERENCE_RUN in set(folds["run_name"]) else configs[0].name
    for metric in ("roc_auc", "pr_auc_bad", "balanced_accuracy"):
        comparison = paired_comparison(folds, reference=reference, metric=metric)
        comparison.to_csv(output_dir / f"paired_{metric}_vs_{reference}.csv", index=False)

    for config in configs:
        plot_accuracy_against_baseline(
            summary, config.name, measure, output_dir / f"{config.name}_accuracy.png"
        )

    LOGGER.info("Results written to %s", output_dir)
    overall = summary[summary["scope"] == "ALL"].sort_values("roc_auc_mean", ascending=False)
    LOGGER.info(
        "Ranking by AUC:\n%s",
        overall[
            ["run_name", "roc_auc_mean", "pr_auc_bad_mean", "balanced_accuracy_mean", "accuracy_uplift"]
        ]
        .round(4)
        .to_string(index=False),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", default=DEFAULT_MEASURE, help="Target measurement")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--database-dir", type=Path, default=DATABASE_DIR)
    parser.add_argument("--results", type=Path, default=PROCESS_FOLDER_RESULTS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--n-repeats", type=int, default=20, help="5 splits x 20 repeats = 100 folds"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-threshold-tuning",
        dest="tune_threshold",
        action="store_false",
        help="Decide at 0.5 instead of the in-fold balanced-accuracy optimum",
    )
    parser.add_argument("--no-shap", dest="make_shap", action="store_false")
    parser.add_argument("--only", nargs="*", help="Restrict to these configuration names")
    parser.add_argument("--list-measures", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_measures:
        print("\n".join(MEASUREMENTS))
        return 0

    run(
        measure=args.measure,
        data_dir=args.data_dir,
        database_dir=args.database_dir,
        results_path=args.results,
        output_root=args.output,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        tune_threshold=args.tune_threshold,
        make_shap=args.make_shap,
        only=args.only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())