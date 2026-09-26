"""Aggregation of every run (CONVENTIONS.md §8.4).

The only path to a results table. A run without a manifest is ignored: it cannot be
reproduced, hence cannot be quoted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from insectpose.paths import ProjectPaths
from insectpose.utils.io import read_json, read_parquet, write_parquet
from insectpose.utils.logging import get_logger

log = get_logger("aggregate")

_MANIFEST_FIELDS = (
    "approach", "data_scope", "split_id", "tag", "mode", "seed", "content_hash",
    "variant_hash",
    "eval_version", "primary_metric", "duration_s",
    # Run-level costs: an approach fills them through ctx.extra (§7.2).
    "model_params", "train_time_s", "peak_vram_mb", "n_qualitative_figures",
)


def collect_runs(paths: ProjectPaths) -> pd.DataFrame:
    """Scan runs/ and assemble metrics + manifest metadata."""
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for run_dir in sorted(p for p in paths.runs.glob("*") if p.is_dir()):
        if run_dir.name == "optuna":
            continue
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.parquet"
        if not manifest_path.exists() or not metrics_path.exists():
            skipped.append(run_dir.name)
            continue
        manifest = read_json(manifest_path)
        metrics = read_parquet(metrics_path)
        for field in _MANIFEST_FIELDS:
            metrics[field] = manifest.get(field)
        # An HPO run is NOT a result: it served to choose hyperparameters, on an inner
        # split. Aggregating it with the final runs would bias the report.
        metrics["role_in_protocol"] = manifest.get("role_in_protocol", "final")
        # An inner split is called <split_id>__outer<k>: the `fold` is then an INNER
        # fold, and the outer one is carried by the name of the split.
        split_id = str(manifest.get("split_id", ""))
        if "__outer" in split_id:
            metrics["outer_fold"] = int(split_id.rsplit("__outer", 1)[-1])
            metrics["inner_fold"] = metrics["fold"]
        else:
            metrics["outer_fold"] = metrics["fold"]
            metrics["inner_fold"] = None
        metrics["trial_number"] = manifest.get("trial_number")
        metrics["optuna_study"] = manifest.get("optuna_study")
        metrics["git_commit"] = (manifest.get("git") or {}).get("commit")
        device = ((manifest.get("environment") or {}).get("device") or {})
        devices = device.get("devices") or []
        metrics["device"] = devices[0]["name"] if devices else device.get("resolved")
        metrics["amp"] = manifest.get("amp")
        frames.append(metrics)

    if skipped:
        log.warning("%d run(s) ignored (missing manifest or metrics): %s",
                    len(skipped), skipped[:5])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def model_label(frame: pd.DataFrame) -> pd.Series:
    """Model label: approach · tag, completed by the hash if two variants coexist.

    It is what keeps two distinct models carrying the same tag (two starting weights,
    for instance) from being averaged together as if they were two folds.
    """
    approach = frame["approach"].astype(str)
    tag = frame["tag"].astype(str) if "tag" in frame.columns else ""
    base = approach + " · " + tag
    if "variant_hash" not in frame.columns:
        return base
    variants = frame.groupby(base.rename("base"))["variant_hash"].transform("nunique")
    suffix = frame["variant_hash"].astype(str).str[:6]
    return base.where(variants <= 1, base + " · " + suffix)


def final_runs(master: pd.DataFrame) -> pd.DataFrame:
    """Quotable runs: excluding the HPO trials (§6.3, §8.4)."""
    if "role_in_protocol" not in master.columns:
        return master
    return master[master["role_in_protocol"].fillna("final") == "final"]


def write_master(paths: ProjectPaths) -> Path:
    """Write results/master.parquet. Side effect: writes this file."""
    table = collect_runs(paths)
    if table.empty:
        raise FileNotFoundError(
            f"No complete run in {paths.runs}. Run at least one 'train' before 'report'."
        )
    trials = len(table) - len(final_runs(table))
    if trials:
        log.info("%d row(s) of HPO trials kept in master.parquet for auditing, "
                 "but excluded from the results tables.", trials)
    _warn_on_incomparable(final_runs(table))
    return write_parquet(paths.master_results(), table)


def _warn_on_incomparable(table: pd.DataFrame) -> None:
    """Warn if aggregated runs are not comparable with each other (§6.2, §7.2)."""
    for column, message in (
        ("split_id", "different folds: the approaches are not comparable"),
        ("content_hash", "different annotations between runs"),
        ("eval_version", "different evaluation configuration versions"),
        ("device", "different hardware: the costs (latency, VRAM) are not comparable"),
        ("primary_metric", "different optimisation objectives between runs"),
    ):
        values = table[column].dropna().unique()
        if len(values) > 1:
            log.warning("Warning - %s (%s: %s).", message, column, list(values)[:4])


def fold_table(master: pd.DataFrame, metric: str, scope: str = "overall",
               split: str = "test", include_trials: bool = False) -> pd.DataFrame:
    """Approach x fold table for a metric: basis of the paired tests (§8.4)."""
    master = master if include_trials else final_runs(master)
    sel = master[
        (master["metric"] == metric) & (master["scope"] == scope) & (master["split"] == split)
    ].copy()
    sel["model"] = model_label(sel)
    return sel.pivot_table(index="model", columns="fold", values="value", aggfunc="mean")


def summary_table(master: pd.DataFrame, metric: str, scope: str = "overall",
                  split: str = "test", include_trials: bool = False) -> pd.DataFrame:
    """Mean, standard deviation and n across folds per MODEL (§6.2).

    The grouping is done by variant, not by approach: two different models carrying the
    same tag must not be averaged as if they were two folds.
    """
    master = master if include_trials else final_runs(master)
    sel = master[
        (master["metric"] == metric) & (master["scope"] == scope) & (master["split"] == split)
    ].copy()
    sel["model"] = model_label(sel)
    out = (
        sel.groupby("model")["value"]
        .agg(mean="mean", std="std", n_folds="size")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    counts = sel.groupby("model")["n"].sum().rename("n_instances").reset_index()
    identity = sel.groupby("model")[["approach", "tag"]].first().reset_index()
    return out.merge(counts, on="model", how="left").merge(identity, on="model", how="left")