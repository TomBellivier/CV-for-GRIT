"""Agregation de tous les runs (CONVENTIONS.md §8.3).

Unique chemin vers un tableau de resultats. Un run sans manifeste est ignore :
il n'est pas reproductible, donc pas citable.
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
    "eval_version", "primary_metric", "duration_s",
    # Couts au niveau run : une approche les renseigne via ctx.extra (§7.2).
    "model_params", "train_time_s", "peak_vram_mb", "n_qualitative_figures",
)


def collect_runs(paths: ProjectPaths) -> pd.DataFrame:
    """Scanne runs/ et assemble metriques + metadonnees de manifeste."""
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
        metrics["trial_number"] = manifest.get("trial_number")
        metrics["optuna_study"] = manifest.get("optuna_study")
        metrics["git_commit"] = (manifest.get("git") or {}).get("commit")
        device = ((manifest.get("environment") or {}).get("device") or {})
        devices = device.get("devices") or []
        metrics["device"] = devices[0]["name"] if devices else device.get("resolved")
        metrics["amp"] = manifest.get("amp")
        frames.append(metrics)

    if skipped:
        log.warning("%d run(s) ignore(s) (manifeste ou metriques manquants) : %s",
                    len(skipped), skipped[:5])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_master(paths: ProjectPaths) -> Path:
    """Ecrit results/master.parquet. Effet de bord : ecrit ce fichier."""
    table = collect_runs(paths)
    if table.empty:
        raise FileNotFoundError(
            f"Aucun run complet dans {paths.runs}. Lancer au moins un 'train' avant 'report'."
        )
    _warn_on_incomparable(table)
    return write_parquet(paths.master_results(), table)


def _warn_on_incomparable(table: pd.DataFrame) -> None:
    """Alerte si des runs agreges ne sont pas comparables entre eux (§6.2, §7.2)."""
    for column, message in (
        ("split_id", "folds differents : les approches ne sont pas comparables"),
        ("content_hash", "annotations differentes entre runs"),
        ("eval_version", "versions de configuration d'evaluation differentes"),
        ("device", "materiels differents : les couts (latence, VRAM) ne sont pas comparables"),
        ("primary_metric", "objectifs d'optimisation differents entre runs"),
    ):
        values = table[column].dropna().unique()
        if len(values) > 1:
            log.warning("Attention - %s (%s : %s).", message, column, list(values)[:4])


def fold_table(master: pd.DataFrame, metric: str, scope: str = "overall",
               split: str = "test") -> pd.DataFrame:
    """Tableau approche x fold pour une metrique : base des tests apparies (§8.3)."""
    sel = master[
        (master["metric"] == metric) & (master["scope"] == scope) & (master["split"] == split)
    ]
    return sel.pivot_table(index="approach", columns="fold", values="value", aggfunc="mean")


def summary_table(master: pd.DataFrame, metric: str, scope: str = "overall",
                  split: str = "test") -> pd.DataFrame:
    """Moyenne, ecart-type et n inter-folds par approche (§6.2)."""
    sel = master[
        (master["metric"] == metric) & (master["scope"] == scope) & (master["split"] == split)
    ]
    out = (
        sel.groupby("approach")["value"]
        .agg(mean="mean", std="std", n_folds="size")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    counts = sel.groupby("approach")["n"].sum().rename("n_instances").reset_index()
    return out.merge(counts, on="approach", how="left")
