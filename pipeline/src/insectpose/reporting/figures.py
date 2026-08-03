"""Figures du rapport (CONVENTIONS.md §8.3).

Toutes les figures sont produites a partir des ARTEFACTS : `results/master.parquet`,
les rapports de couverture, les predictions et les journaux d'entrainement. Aucune ne
recalcule une metrique — sans quoi deux chiffres du meme rapport pourraient diverger.

Les runs d'HPO sont exclus par `final_runs` : ce sont des essais, pas des resultats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")   # backend sans affichage : le rapport tourne aussi sans X
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from insectpose.data.keypoints import KeypointSchema, load_schema  # noqa: E402
from insectpose.data.measurements import load_measurements, measure_all  # noqa: E402
from insectpose.evaluation.aggregate import final_runs  # noqa: E402
from insectpose.paths import ProjectPaths  # noqa: E402
from insectpose.utils.io import read_parquet  # noqa: E402
from insectpose.utils.logging import get_logger  # noqa: E402

log = get_logger("figures")

# Groupes anatomiques utilises pour le code couleur. L'ordre compte : la premiere
# regle qui correspond gagne (les regles specifiques avant les generiques).
_GROUP_RULES: tuple[tuple[str, str], ...] = (
    ("eye", "eyes"),
    ("antenna", "antennae"),
    ("forewing", "forewings"),
    ("hindwing", "hindwings"),
    ("leg", "legs"),
    ("thorax", "thorax"),
    ("body", "abdomen"),
    ("head", "head"),
    ("neck", "head"),
)

_GROUP_COLORS = {
    "head": "#d62728", "eyes": "#7f7f7f", "antennae": "#17becf",
    "thorax": "#ff7f0e", "abdomen": "#2ca02c", "forewings": "#e377c2",
    "hindwings": "#9467bd", "legs": "#1f77b4", "other": "#8c564b",
}


def keypoint_group(name: str, with_side: bool = True) -> str:
    """Groupe anatomique d'un keypoint, ex. 'right hindwings' ou 'head'.

    Les points de l'axe median n'ont pas de cote ; les autres le portent, car une
    asymetrie gauche/droite est en soi une information de diagnostic.
    """
    lowered = name.lower()
    base = next((group for token, group in _GROUP_RULES if token in lowered), "other")
    if not with_side:
        return base
    if lowered.startswith("left-"):
        return f"left {base}"
    if lowered.startswith("right-"):
        return f"right {base}"
    return base


def group_color(group: str) -> str:
    """Couleur stable d'un groupe anatomique, cote inclus ou non."""
    base = group.replace("left ", "").replace("right ", "")
    return _GROUP_COLORS.get(base, _GROUP_COLORS["other"])


# --- helpers ----------------------------------------------------------------
def _save(fig: Any, path: Path, dpi: int = 150) -> Path:
    """Ecrit une figure et la ferme. Effet de bord : cree `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _select(master: pd.DataFrame, metric: str, split: str = "test",
            scope_prefix: str | None = None, scope: str | None = None) -> pd.DataFrame:
    """Sous-ensemble citable de master.parquet pour une metrique."""
    data = final_runs(master)
    data = data[(data["metric"] == metric) & (data["split"] == split)]
    if scope is not None:
        data = data[data["scope"] == scope]
    if scope_prefix is not None:
        data = data[data["scope"].str.startswith(scope_prefix)]
    return data


def _dataset_of(scope: str) -> str:
    """Nom du dataset porte par un scope 'dataset:x' ou 'keypoint:x:nom'."""
    parts = str(scope).split(":")
    return parts[1] if len(parts) > 1 else "overall"


def available_metrics(master: pd.DataFrame, split: str = "test") -> list[str]:
    """Metriques scalaires disponibles au perimetre 'overall' ou par dataset."""
    data = final_runs(master)
    data = data[(data["split"] == split) & (
        (data["scope"] == "overall") | data["scope"].str.startswith("dataset:"))]
    return sorted(data["metric"].unique())


# --- 1. une figure par metrique : barres par dataset -------------------------
def fig_metric_by_dataset(master: pd.DataFrame, metric: str, out_dir: Path,
                          split: str = "test", dpi: int = 150) -> Path | None:
    """Barres par dataset, groupees par approche, avec ecart-type inter-folds.

    L'ecart-type n'est pas decoratif : sans lui, un ecart de quelques points entre
    deux approches n'est pas interpretable (§8.3).
    """
    data = _select(master, metric, split, scope_prefix="dataset:")
    overall = _select(master, metric, split, scope="overall")
    data = pd.concat([data, overall], ignore_index=True)
    if data.empty:
        return None
    data = data.assign(dataset=data["scope"].map(_dataset_of))

    stats = data.groupby(["dataset", "approach"])["value"].agg(["mean", "std", "size"])
    stats = stats.reset_index()
    datasets = sorted(stats["dataset"].unique(), key=lambda d: (d != "overall", d))
    approaches = sorted(stats["approach"].unique())

    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) * max(len(approaches), 1) + 3, 4.5))
    width = 0.8 / max(len(approaches), 1)
    positions = np.arange(len(datasets), dtype=float)
    for i, approach in enumerate(approaches):
        sub = stats[stats["approach"] == approach].set_index("dataset")
        means = [sub["mean"].get(d, np.nan) for d in datasets]
        errors = [sub["std"].get(d, np.nan) for d in datasets]
        ax.bar(positions + i * width, means, width, yerr=errors, capsize=3, label=approach)

    ax.set_xticks(positions + width * (len(approaches) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(metric)
    n_folds = int(stats["size"].max())
    ax.set_title(f"{metric} by dataset ({split} split, mean ± std over "
                 f"{n_folds} fold(s))")
    ax.grid(axis="y", alpha=0.3)
    if len(approaches) > 1:
        ax.legend()
    return _save(fig, out_dir / f"metric_{metric.replace('@', '').replace('/', '_')}.png", dpi)


# --- 2. confiance vs erreur par keypoint, un panneau par dataset -------------
def fig_confidence_vs_error(master: pd.DataFrame, out_dir: Path, split: str = "test",
                            dpi: int = 150) -> Path | None:
    """Confiance predite vs erreur normalisee, par keypoint, code couleur anatomique.

    Un nuage en L (confiance haute et erreur basse) signifie que la confiance est
    exploitable comme filtre en production ; un nuage sans structure signifie
    l'inverse, et c'est une information de premier plan pour l'usage reel.
    """
    conf = _select(master, "kpt_conf_mean", split, scope_prefix="keypoint:")
    err = _select(master, "nme", split, scope_prefix="keypoint:")
    if conf.empty or err.empty:
        return None

    merged = (
        conf.groupby("scope")["value"].mean().rename("conf").to_frame()
        .join(err.groupby("scope")["value"].mean().rename("error"), how="inner")
        .join(err.groupby("scope")["n"].sum().rename("n"), how="inner")
        .reset_index()
    )
    merged["dataset"] = merged["scope"].map(_dataset_of)
    merged["keypoint"] = merged["scope"].map(lambda s: str(s).split(":")[-1])
    merged["group"] = merged["keypoint"].map(keypoint_group)

    datasets = sorted(merged["dataset"].unique())
    cols = 2 if len(datasets) > 1 else 1
    rows = int(np.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for ax, dataset in zip(axes.flat, datasets, strict=False):
        sub = merged[merged["dataset"] == dataset]
        for group, part in sub.groupby("group"):
            ax.scatter(part["conf"], part["error"], s=30, alpha=0.85,
                       color=group_color(str(group)), label=str(group),
                       edgecolors="black", linewidths=0.3)
        ax.set_title(dataset)
        ax.set_xlabel("mean keypoint confidence")
        ax.set_ylabel("normalised error (NME)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Predicted confidence vs error, per keypoint")
    return _save(fig, out_dir / "keypoint_confidence_vs_error.png", dpi)


# --- 3. courbes d'apprentissage ---------------------------------------------
def fig_training_curves(paths: ProjectPaths, master: pd.DataFrame, out_dir: Path,
                        dpi: int = 150) -> Path | None:
    """Courbes d'apprentissage par epoque, une ligne par run (si le framework en produit).

    Ces courbes servent au diagnostic (convergence, surapprentissage) et JAMAIS a la
    comparaison entre approches : les metriques citables viennent de l'evaluateur (§7.1).
    """
    curves: list[pd.DataFrame] = []
    for run_id in sorted(final_runs(master)["run_id"].dropna().unique()):
        csv = paths.run_dir(str(run_id)) / "logs" / "train" / "results.csv"
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        frame.columns = [c.strip() for c in frame.columns]
        frame["run_id"] = run_id
        meta = final_runs(master)
        meta = meta[meta["run_id"] == run_id]
        frame["label"] = f"{meta['approach'].iloc[0]} fold{int(meta['fold'].iloc[0])}"
        curves.append(frame)

    if not curves:
        log.info("Aucune courbe d'apprentissage : le framework n'en produit pas.")
        return None

    data = pd.concat(curves, ignore_index=True)
    columns = [c for c in data.columns
               if c.startswith(("train/", "val/", "metrics/")) and data[c].notna().any()]
    if not columns:
        return None

    cols = 3
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    epoch_col = "epoch" if "epoch" in data.columns else None
    for ax, column in zip(axes.flat, columns, strict=False):
        for label, part in data.groupby("label"):
            x = part[epoch_col] if epoch_col else np.arange(len(part))
            ax.plot(x, part[column], lw=1.2, label=str(label))
        ax.set_title(column, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(columns):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles and len(labels) <= 12:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training curves (diagnostic only - not comparable across approaches)")
    return _save(fig, out_dir / "training_curves.png", dpi)


# --- 4. dispersion inter-folds ----------------------------------------------
def fig_fold_boxplot(master: pd.DataFrame, metric: str, out_dir: Path,
                     split: str = "test", dpi: int = 150) -> Path | None:
    """Boxplot de la metrique par approche, un point par fold.

    C'est la figure qui accompagne les tests apparies : elle montre si un ecart de
    moyenne survit a la variabilite entre folds.
    """
    data = _select(master, metric, split, scope="overall")
    if data.empty:
        return None
    groups = data.groupby("approach")["value"]
    labels = list(groups.groups)
    values = [groups.get_group(name).to_numpy() for name in labels]

    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 4.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    for i, series in enumerate(values, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, len(series))
        ax.scatter(np.full(len(series), i) + jitter, series, s=25, alpha=0.8, zorder=3)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per fold ({split} split)")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir / f"folds_{metric.replace('@', '')}.png", dpi)


# --- 5. courbe PCK vs alpha --------------------------------------------------
def fig_pck_curve(master: pd.DataFrame, out_dir: Path, split: str = "test",
                  dpi: int = 150) -> Path | None:
    """PCK en fonction du seuil alpha, une ligne par (approche, dataset).

    Un seuil unique cache la forme de la distribution d'erreur : deux modeles au meme
    PCK@0.25 peuvent differer nettement aux seuils serres.
    """
    data = final_runs(master)
    data = data[data["metric"].str.startswith("pck@") & (data["split"] == split)
                & (data["scope"].str.startswith("dataset:") | (data["scope"] == "overall"))]
    if data.empty:
        return None
    data = data.assign(
        alpha=data["metric"].str.extract(r"pck@([0-9.]+)_")[0].astype(float),
        dataset=data["scope"].map(_dataset_of),
    ).dropna(subset=["alpha"])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for (approach, dataset), part in data.groupby(["approach", "dataset"]):
        curve = part.groupby("alpha")["value"].mean().sort_index()
        style = "--" if dataset == "overall" else "-"
        ax.plot(curve.index, curve.to_numpy(), style, marker="o", ms=3,
                lw=2 if dataset == "overall" else 1.2,
                label=f"{approach} · {dataset}")
    ax.set_xlabel("alpha (fraction of thorax width)")
    ax.set_ylabel("PCK")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("PCK curve")
    return _save(fig, out_dir / "pck_curve.png", dpi)


# --- 6. PCK par keypoint vs couverture d'annotation -------------------------
def _keypoint_pck(master: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """PCK moyen par (dataset, keypoint), au seuil de reference."""
    data = final_runs(master)
    data = data[data["scope"].str.startswith("keypoint:") & (data["split"] == split)
                & data["metric"].str.startswith("pck@")]
    if data.empty:
        return data
    frame = data.groupby(["approach", "scope"])["value"].mean().reset_index()
    frame["dataset"] = frame["scope"].map(_dataset_of)
    frame["keypoint"] = frame["scope"].map(lambda s: str(s).split(":")[-1])
    return frame


def fig_pck_vs_coverage(master: pd.DataFrame, coverage: pd.DataFrame, out_dir: Path,
                        split: str = "test", dpi: int = 150) -> Path | None:
    """PCK par keypoint vs taux d'annotation de ce keypoint dans le dataset.

    Evite la conclusion erronee "ce point est mal predit" alors qu'il est simplement
    rarement annote (ADR-0016).
    """
    pck = _keypoint_pck(master, split)
    if pck.empty or coverage.empty:
        return None
    merged = pck.merge(coverage[["dataset", "keypoint", "rate", "n_annotated"]],
                       on=["dataset", "keypoint"], how="inner")
    if merged.empty:
        return None
    merged["group"] = merged["keypoint"].map(keypoint_group)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for group, part in merged.groupby("group"):
        ax.scatter(part["rate"], part["value"], s=28, alpha=0.85,
                   color=group_color(str(group)), label=str(group),
                   edgecolors="black", linewidths=0.3)
    ax.set_xlabel("keypoint annotation rate")
    ax.set_ylabel("keypoint PCK")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Keypoint PCK vs annotation coverage")
    return _save(fig, out_dir / "pck_vs_coverage.png", dpi)


# --- 7. PCK par keypoint vs difficulte experte ------------------------------
def fig_pck_vs_difficulty(master: pd.DataFrame, schema: KeypointSchema, out_dir: Path,
                          split: str = "test", dpi: int = 150) -> Path | None:
    """PCK par keypoint vs difficulte declaree par l'expert.

    Cette figure valide (ou non) l'echelle de difficulte qui fonde les sigmas OKS
    (ADR-0007) : une correlation faible signifierait que la metrique primaire elle-meme
    est mal calibree.
    """
    pck = _keypoint_pck(master, split)
    if pck.empty:
        return None
    difficulty = dict(zip(schema.names, schema.difficulty, strict=True))
    pck = pck.assign(difficulty=pck["keypoint"].map(difficulty)).dropna(subset=["difficulty"])
    if pck.empty:
        return None
    pck["group"] = pck["keypoint"].map(keypoint_group)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for group, part in pck.groupby("group"):
        jitter = np.random.default_rng(1).normal(0, 0.5, len(part))
        ax.scatter(part["difficulty"] + jitter, part["value"], s=28, alpha=0.85,
                   color=group_color(str(group)), label=str(group),
                   edgecolors="black", linewidths=0.3)
    if len(pck) > 2:
        correlation = float(np.corrcoef(pck["difficulty"], pck["value"])[0, 1])
        ax.set_title(f"Keypoint PCK vs expert difficulty (r = {correlation:.2f})")
    else:
        ax.set_title("Keypoint PCK vs expert difficulty")
    ax.set_xlabel("declared difficulty (10 = easy, 40 = very hard)")
    ax.set_ylabel("keypoint PCK")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    return _save(fig, out_dir / "pck_vs_difficulty.png", dpi)


# --- 8. symetrie gauche/droite des mesures ----------------------------------
def fig_symmetry_scatter(paths: ProjectPaths, master: pd.DataFrame, cfg: Any,
                         out_dir: Path, split: str = "test", dpi: int = 150
                         ) -> Path | None:
    """Mesure gauche vs mesure droite predite, un panneau par paire symetrique.

    Un modele coherent aligne les points sur la diagonale. L'ecart a la diagonale se
    lit sans verite terrain : c'est un controle qualite utilisable en production.
    """
    spec = load_measurements(Path(str(cfg.eval.measurements.file)))
    if not spec.symmetric_pairs:
        return None
    schema = load_schema(spec.keypoint_schema, paths.configs)
    index = spec.indices(schema)

    frames: list[pd.DataFrame] = []
    for run_id in sorted(final_runs(master)["run_id"].dropna().unique()):
        for file in sorted((paths.run_dir(str(run_id)) / "predictions").glob(f"{split}_*.parquet")):
            frames.append(read_parquet(file))
    if not frames:
        return None
    predictions = pd.concat(frames, ignore_index=True)
    kpts = np.stack(
        predictions["kpts_xy"].map(lambda v: np.asarray(v, float)).to_numpy()
    ).reshape(len(predictions), -1, 2)
    values = measure_all(kpts, index)

    pairs = [(a, b) for a, b in spec.symmetric_pairs if a in values and b in values]
    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    datasets = predictions["dataset"].to_numpy()

    for ax, (left, right) in zip(axes.flat, pairs, strict=False):
        x, y = values[left], values[right]
        usable = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        for dataset in sorted(set(datasets)):
            mask = usable & (datasets == dataset)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=12, alpha=0.6, label=dataset)
        if usable.any():
            limit = float(np.percentile(np.concatenate([x[usable], y[usable]]), 99.5))
            ax.plot([0, limit], [0, limit], "k--", lw=1)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            gap = np.abs(x[usable] - y[usable]) / ((x[usable] + y[usable]) / 2)
            ax.set_title(f"{left.replace('left ', '')}\nmedian gap {np.median(gap):.1%}",
                         fontsize=9)
        ax.set_xlabel("left (px)", fontsize=8)
        ax.set_ylabel("right (px)", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Symmetry of predicted measurements (left vs right)")
    return _save(fig, out_dir / "symmetry_pairs.png", dpi)


# --- point d'entree ----------------------------------------------------------
def write_figures(paths: ProjectPaths, cfg: Any, master: pd.DataFrame) -> list[Path]:
    """Produit toutes les figures du rapport. Effet de bord : ecrit results/figures/."""
    out_dir = paths.results / "figures"
    dpi = int(cfg.report.dpi)
    split = str(cfg.report.split)
    written: list[Path] = []

    for metric in available_metrics(master, split):
        written.append(fig_metric_by_dataset(master, metric, out_dir, split, dpi))
        written.append(fig_fold_boxplot(master, metric, out_dir, split, dpi))

    written.append(fig_confidence_vs_error(master, out_dir, split, dpi))
    written.append(fig_pck_curve(master, out_dir, split, dpi))
    written.append(fig_training_curves(paths, master, out_dir, dpi))

    coverage_file = paths.processed / "coverage_keypoints.parquet"
    if coverage_file.exists():
        written.append(fig_pck_vs_coverage(master, read_parquet(coverage_file), out_dir,
                                           split, dpi))

    schema_name = cfg.data.get("keypoint_schema")
    if schema_name:
        written.append(fig_pck_vs_difficulty(
            master, load_schema(str(schema_name), paths.configs), out_dir, split, dpi))

    if bool(cfg.eval.measurements.enabled):
        written.append(fig_symmetry_scatter(paths, master, cfg, out_dir, split, dpi))

    produced = [p for p in written if p is not None]
    log.info("%d figure(s) ecrite(s) dans %s", len(produced), out_dir)
    return produced