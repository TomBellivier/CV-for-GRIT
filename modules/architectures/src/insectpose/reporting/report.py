"""Tableaux de resultats et tests apparies (CONVENTIONS.md §8.3).

Toute figure ou tableau du rapport vient d'ici. Un chiffre copie depuis une console
n'est pas un resultat : il n'est ni tracable ni reproductible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.evaluation.aggregate import fold_table, summary_table
from insectpose.paths import ProjectPaths
from insectpose.utils.io import read_parquet, write_json, write_parquet
from insectpose.utils.logging import get_logger

log = get_logger("report")


def paired_tests(master: pd.DataFrame, metric: str, scope: str = "overall",
                 split: str = "test") -> pd.DataFrame:
    """Wilcoxon apparie par fold entre chaque paire d'approches, avec correction Holm.

    Comparer des moyennes sans test apparie sur les MEMES folds sur-interprete
    systematiquement les ecarts (§8.3).
    """
    from itertools import combinations

    from scipy import stats

    table = fold_table(master, metric, scope, split).dropna(axis=1, how="any")
    approaches = list(table.index)
    rows: list[dict[str, Any]] = []
    for a, b in combinations(approaches, 2):
        x, y = table.loc[a].to_numpy(float), table.loc[b].to_numpy(float)
        degenerate = len(x) < 3 or np.allclose(x, y)
        p = float("nan") if degenerate else float(stats.wilcoxon(x, y).pvalue)
        rows.append(
            {"metric": metric, "scope": scope, "approach_a": a, "approach_b": b,
             "mean_a": float(x.mean()), "mean_b": float(y.mean()),
             "delta": float(x.mean() - y.mean()), "n_folds": int(len(x)), "p_value": p}
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Correction de Holm pour comparaisons multiples
    order = out["p_value"].rank(method="first")
    m = out["p_value"].notna().sum()
    out["p_holm"] = np.minimum(1.0, out["p_value"] * (m - order + 1))
    return out.sort_values("p_value")


def cost_performance(master: pd.DataFrame, metric: str, split: str = "test") -> pd.DataFrame:
    """Croisement performance / latence : base du choix d'un modele deployable (§7.2)."""
    perf = summary_table(master, metric, "overall", split)[["approach", "mean"]]
    cost = master[
        (master["metric"] == "latency_ms_per_instance") & (master["split"] == split)
    ].groupby("approach")["value"].mean().rename("latency_ms").reset_index()
    return perf.merge(cost, on="approach", how="left")


def write_report(paths: ProjectPaths, cfg: Any) -> Path:
    """Ecrit les tableaux de synthese. Effet de bord : results/*.parquet et *.json.

    Retourne le chemin du tableau principal.
    """
    from insectpose.evaluation.aggregate import final_runs

    master = read_parquet(paths.master_results())
    metric = str(cfg.eval.primary_metric)
    citable = final_runs(master)

    main_table = summary_table(master, metric)
    write_parquet(paths.results / "summary_primary.parquet", main_table)
    write_parquet(paths.results / "folds_primary.parquet",
                  fold_table(master, metric).reset_index())

    per_dataset = []
    for scope in sorted(s for s in master["scope"].unique() if str(s).startswith("dataset:")):
        sub = summary_table(master, metric, scope)
        sub["scope"] = scope
        per_dataset.append(sub)
    if per_dataset:
        write_parquet(paths.results / "summary_per_dataset.parquet",
                      pd.concat(per_dataset, ignore_index=True))

    tests = paired_tests(master, metric)
    if not tests.empty:
        write_parquet(paths.results / "paired_tests.parquet", tests)

    write_json(
        paths.results / "report_meta.json",
        {
            "primary_metric": metric,
            "n_runs": int(citable["run_id"].nunique()),
            "n_hpo_trial_runs_excluded": int(
                master["run_id"].nunique() - citable["run_id"].nunique()
            ),
            "approaches": sorted(citable["approach"].dropna().unique().tolist()),
            "split_ids": sorted(master["split_id"].dropna().unique().tolist()),
            "eval_versions": sorted(master["eval_version"].dropna().unique().tolist()),
            "warning": (
                "Rows with bbox_source=gt are diagnostic only and are not comparable "
                "with end-to-end approaches."
            ),
        },
    )
    if bool(cfg.report.figures):
        from insectpose.reporting.figures import write_figures, write_per_run_figures

        write_figures(paths, cfg, master)
        if bool(cfg.report.get("per_run_figures", True)):
            write_per_run_figures(paths, cfg, master)

    log.info("Rapport : %d approche(s), %d run(s) citables (%d trials d'HPO exclus).",
             citable["approach"].nunique(), citable["run_id"].nunique(),
             master["run_id"].nunique() - citable["run_id"].nunique())
    return paths.results / "summary_primary.parquet"