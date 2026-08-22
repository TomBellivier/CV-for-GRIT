"""Leak-free model building and evaluation for measurement-validity models.

Two methodological points drive the design of this module.

**Paired resampling.** Every configuration is evaluated on the *same* folds,
generated once from a fixed seed by a repeated stratified k-fold. Comparing
configurations then reduces to a paired test on the per-fold differences, which
is far more sensitive than comparing two means whose standard deviations
overlap. Stratification is done on the interaction group x label so that no fold
ends up with a degenerate class distribution inside a taxonomic group.

**Everything that learns lives in the pipeline.** Group-conditional statistics,
imputation, scaling and PCA are fitted on the training fold only. Fitting them
on the full table, as a preprocessing step, leaks test information and inflates
the scores.

The metrics reported are chosen for a strongly imbalanced problem: raw accuracy
is dominated by the majority class and is therefore always reported next to the
accuracy of the constant majority classifier computed on the same fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from dataset import STATUS_SUFFIX, ColumnIndex
from features import FEATURE_PREFIX, GroupReferenceStats
from insect_anatomy import INSECT_GROUPS, related_entities

LOGGER = logging.getLogger(__name__)

GROUP_COLUMN = "group"
ONE_HOT_SUFFIX = "_one_hot"

#: Feature families and the token identifying their raw columns.
FEATURE_FAMILY_TOKENS: Dict[str, Tuple[str, ...]] = {
    "conf": ("kp_conf",),
    "meas": (" px",),
    "pos": ("kp_x", "kp_y"),
}


@dataclass
class RunConfig:
    """One feature-space configuration to evaluate.

    Attributes
    ----------
    name:
        Identifier used in the result tables and figure filenames.
    families:
        Raw feature families to include among ``conf``, ``meas``, ``pos``.
    derived:
        Include the engineered features (:mod:`features`).
    group_stats:
        Add the group-conditional robust z-scores. Requires ``derived``.
    one_hot:
        Expose the taxonomic group to the model as one-hot columns.
    related_only:
        Restrict the raw columns to the anatomical neighbourhood of the target
        measurement.
    n_pca:
        Number of principal components, or 0 to disable. PCA is applied to the
        continuous block only; one-hot columns are passed through unchanged.
    balance:
        Set ``scale_pos_weight`` from the training fold class ratio.
    """

    name: str
    families: Tuple[str, ...] = ()
    derived: bool = False
    group_stats: bool = False
    one_hot: bool = False
    related_only: bool = False
    n_pca: int = 0
    balance: bool = False


class GroupEncoder(BaseEstimator, TransformerMixin):
    """Drop the helper group column, optionally replacing it with one-hots.

    The group column is needed by :class:`GroupReferenceStats` upstream but must
    never reach the estimator as a raw categorical.
    """

    def __init__(self, one_hot: bool = False, group_column: str = GROUP_COLUMN):
        self.one_hot = one_hot
        self.group_column = group_column

    def fit(self, X: pd.DataFrame, y=None) -> "GroupEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.group_column not in out.columns:
            return out
        groups = out.pop(self.group_column)
        if self.one_hot:
            for group in INSECT_GROUPS:
                out[f"{group}{ONE_HOT_SUFFIX}"] = (groups == group).astype(int)
        return out


class PartialPCA(BaseEstimator, TransformerMixin):
    """Impute, standardise and compress the continuous block only.

    Two fixes over a plain ``PCA`` on the raw table: features are standardised
    first (otherwise pixel coordinates, whose variance is orders of magnitude
    larger than a confidence in [0, 1], monopolise every component), and binary
    indicators are excluded from the rotation and passed through, so that the
    taxonomic one-hots survive the reduction.
    """

    def __init__(self, n_components: int = 0, passthrough_suffix: str = ONE_HOT_SUFFIX):
        self.n_components = n_components
        self.passthrough_suffix = passthrough_suffix

    def fit(self, X: pd.DataFrame, y=None) -> "PartialPCA":
        self.passthrough_ = [c for c in X.columns if c.endswith(self.passthrough_suffix)]
        self.continuous_ = [c for c in X.columns if c not in self.passthrough_]
        n_components = min(self.n_components, len(self.continuous_), len(X))
        self.pipeline_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=0)),
            ]
        )
        self.pipeline_.fit(X[self.continuous_])
        self.n_components_ = n_components
        self.explained_variance_ratio_ = self.pipeline_["pca"].explained_variance_ratio_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        components = self.pipeline_.transform(X[self.continuous_])
        out = pd.DataFrame(
            components,
            index=X.index,
            columns=[f"PC{i:02d}" for i in range(self.n_components_)],
        )
        if self.passthrough_:
            out = pd.concat([out, X[self.passthrough_]], axis=1)
        return out


def select_feature_columns(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    config: RunConfig,
    measure: str,
) -> List[str]:
    """Resolve the list of columns feeding one configuration.

    ``related_only`` filters the *raw* anatomical columns; derived features and
    one-hot indicators are never filtered out by it. In the original code the
    specificity filter was applied to every column, which silently removed the
    taxonomic one-hots from every ``related_*`` run.
    """
    selected: List[str] = []

    raw_candidates: List[str] = []
    for family in config.families:
        tokens = FEATURE_FAMILY_TOKENS[family]
        if family == "conf":
            pool = {p: columns.conf[p] for p in columns.conf}
        elif family == "pos":
            pool = {p: columns.x[p] for p in columns.x}
            pool.update({f"{p}#y": columns.y[p] for p in columns.y})
        else:
            pool = dict(columns.measure)
        raw_candidates.extend(pool.items())

    if config.related_only:
        related_points, related_measures = related_entities(measure)
        allowed = set(related_points) | set(related_measures)
        raw_candidates = [
            (entity, column)
            for entity, column in raw_candidates
            if entity.split("#")[0] in allowed
        ]

    selected.extend(column for _, column in raw_candidates)

    if config.derived:
        selected.extend(c for c in frame.columns if c.startswith(FEATURE_PREFIX))

    # Deduplicate while preserving order, and never let the label leak in.
    seen = set()
    ordered = []
    target_status = f"{measure}{STATUS_SUFFIX}"
    for column in selected:
        if column in seen or column == target_status or column.endswith(STATUS_SUFFIX):
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def build_pipeline(config: RunConfig, scale_pos_weight: float = 1.0) -> Pipeline:
    """Assemble the full leak-free pipeline for one configuration."""
    steps: List[Tuple[str, object]] = []
    if config.group_stats:
        steps.append(("group_stats", GroupReferenceStats(group_column=GROUP_COLUMN)))
    steps.append(("encode", GroupEncoder(one_hot=config.one_hot)))
    if config.n_pca > 0:
        steps.append(("reduce", PartialPCA(n_components=config.n_pca)))
    steps.append(
        (
            "model",
            XGBClassifier(
                random_state=42,
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=4,
            ),
        )
    )
    return Pipeline(steps)


def tune_threshold(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
) -> float:
    """Pick the decision threshold maximising balanced accuracy, in-fold.

    Uses an inner cross-validation on the training fold: choosing the threshold
    on the test fold, or on training-set predictions, would be optimistic.
    """
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in inner.split(X, y):
        clone = build_pipeline_like(pipeline)
        clone.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = clone.predict_proba(X.iloc[valid_idx])[:, 1]

    candidates = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 99)))
    scores = [balanced_accuracy_score(y, (oof >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def build_pipeline_like(pipeline: Pipeline) -> Pipeline:
    """Clone a pipeline without fitted state (thin wrapper for readability)."""
    from sklearn.base import clone

    return clone(pipeline)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC returning NaN when a single class is present."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


MIN_MINORITY_COUNT = 3


def fold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    majority_class: int,
) -> Dict[str, float]:
    """Metric bundle for one fold and one evaluation scope.

    ``y_true == 1`` means *measurable*. The class of interest is the minority
    one, *unmeasurable*, so precision / recall / average precision are computed
    on the inverted problem and suffixed ``_bad``.

    Every rank- or recall-based metric is returned as NaN when the scope holds
    fewer than :data:`MIN_MINORITY_COUNT` minority examples. On a group such as
    coleoptera, a test fold can contain one or two unmeasurable cases; the
    resulting AUC or average precision is then either 0 or 1 and its mean over
    folds is meaningless. Reporting NaN and counting how many folds were usable
    is more honest than averaging noise.
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_bad = 1 - y_true
    proba_bad = 1.0 - y_proba
    baseline_pred = np.full_like(y_true, majority_class)

    n_bad = int(y_bad.sum())
    n_good = int(y_true.sum())
    usable = min(n_bad, n_good) >= MIN_MINORITY_COUNT

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_baseline": accuracy_score(y_true, baseline_pred),
        "brier": brier_score_loss(y_true, y_proba, pos_label=1),
        "prevalence_bad": float(y_bad.mean()),
        "n": float(len(y_true)),
        "n_bad": float(n_bad),
        "usable": float(usable),
    }
    if not usable:
        metrics.update(
            {
                "roc_auc": np.nan,
                "pr_auc_bad": np.nan,
                "balanced_accuracy": np.nan,
                "mcc": np.nan,
                "recall_bad": np.nan,
                "precision_bad": np.nan,
            }
        )
        return metrics

    metrics.update(
        {
            "roc_auc": _safe_auc(y_true, y_proba),
            "pr_auc_bad": average_precision_score(y_bad, proba_bad),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "recall_bad": recall_score(y_bad, 1 - y_pred, zero_division=0),
            "precision_bad": precision_score(y_bad, 1 - y_pred, zero_division=0),
        }
    )
    return metrics


def evaluate_config(
    frame: pd.DataFrame,
    columns: ColumnIndex,
    config: RunConfig,
    measure: str,
    n_splits: int = 5,
    n_repeats: int = 20,
    seed: int = 42,
    tune_decision_threshold: bool = True,
) -> pd.DataFrame:
    """Run the paired repeated stratified CV for one configuration.

    Returns one row per fold and per evaluation scope (``ALL`` plus each
    taxonomic group), so that paired statistical comparisons between
    configurations can be done afterwards on the raw values.
    """
    feature_columns = select_feature_columns(frame, columns, config, measure)
    if not feature_columns:
        raise ValueError(f"Configuration {config.name!r} selected no feature column")

    X = frame[feature_columns].copy()
    X[GROUP_COLUMN] = frame[GROUP_COLUMN].astype(str).to_numpy()
    y = frame[f"{measure}{STATUS_SUFFIX}"].to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()

    # Stratify on the group x label interaction so that every fold keeps the
    # per-group prevalence, which is what the per-group metrics depend on.
    strata = pd.Series(groups).str.cat(pd.Series(y).astype(str), sep="|").to_numpy()

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    records: List[Dict[str, object]] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, strata)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scale_pos_weight = 1.0
        if config.balance:
            n_pos = max(int(y_train.sum()), 1)
            n_neg = max(len(y_train) - n_pos, 1)
            scale_pos_weight = n_neg / n_pos

        pipeline = build_pipeline(config, scale_pos_weight=scale_pos_weight)
        threshold = 0.5
        if tune_decision_threshold:
            threshold = tune_threshold(pipeline, X_train, y_train, seed=seed + fold_id)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]

        majority_class = int(round(float(y_train.mean())))
        scopes = [("ALL", np.ones(len(y_test), dtype=bool))]
        test_groups = groups[test_idx]
        for group in INSECT_GROUPS:
            scopes.append((group, test_groups == group))

        for scope, mask in scopes:
            if mask.sum() == 0:
                continue
            if scope == "ALL":
                majority = majority_class
            else:
                group_train = y_train[groups[train_idx] == scope]
                majority = (
                    int(round(float(group_train.mean()))) if len(group_train) else majority_class
                )
            metrics = fold_metrics(y_test[mask], proba[mask], threshold, majority)
            records.append(
                {
                    "run_name": config.name,
                    "measure": measure,
                    "fold": fold_id,
                    "scope": scope,
                    "threshold": threshold,
                    "n_features": X_train.shape[1] - 1,
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def summarise(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results into mean / std / uplift-over-baseline."""
    metric_columns = [
        "roc_auc",
        "pr_auc_bad",
        "accuracy",
        "accuracy_baseline",
        "balanced_accuracy",
        "mcc",
        "recall_bad",
        "precision_bad",
        "brier",
        "prevalence_bad",
    ]
    grouped = fold_results.groupby(["run_name", "scope"], observed=True)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["accuracy_uplift"] = summary["accuracy_mean"] - summary["accuracy_baseline_mean"]
    summary["n_folds"] = grouped.size()
    summary["n_folds_usable"] = grouped["usable"].sum()
    summary["mean_n_bad_per_fold"] = grouped["n_bad"].mean()
    return summary.reset_index()


def paired_comparison(
    fold_results: pd.DataFrame,
    reference: str,
    metric: str = "roc_auc",
    scope: str = "ALL",
) -> pd.DataFrame:
    """Paired per-fold comparison of every configuration against a reference.

    Because all configurations share the same folds, the fold-wise difference
    removes the split-to-split variance, which dominates the raw standard
    deviation. The reported interval is a 95 % interval on the mean difference;
    a configuration whose interval contains zero is not distinguishable from the
    reference.
    """
    from scipy import stats

    subset = fold_results[fold_results["scope"] == scope]
    pivot = subset.pivot_table(index="fold", columns="run_name", values=metric)
    if reference not in pivot.columns:
        raise KeyError(f"Reference run {reference!r} not found")

    rows = []
    for run_name in pivot.columns:
        deltas = (pivot[run_name] - pivot[reference]).dropna()
        if run_name == reference or deltas.empty:
            continue
        mean = float(deltas.mean())
        sem = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
        try:
            _, p_value = stats.wilcoxon(deltas)
        except ValueError:
            p_value = np.nan
        rows.append(
            {
                "run_name": run_name,
                "reference": reference,
                "metric": metric,
                "scope": scope,
                "delta_mean": mean,
                "delta_ci_low": mean - 1.96 * sem,
                "delta_ci_high": mean + 1.96 * sem,
                "win_rate": float((deltas > 0).mean()),
                "wilcoxon_p": p_value,
            }
        )
    result = pd.DataFrame(rows).sort_values("delta_mean", ascending=False)
    return result.reset_index(drop=True)