"""Comparaison des approches de classification de validite de mesure (recherche).

Conversion 1:1 de ``measure_validity_classifiers.ipynb`` : pour chaque mesure
(27), compare 8 approches (seuils sur la confiance, XGBoost et Random Forest
sur 3 jeux de features -- direct / related / all) et produit les figures et
tableaux de comparaison (classement MCC, heatmap, importance des variables,
etc).

**Recherche uniquement : ne sauvegarde aucun modele de production.** Les
modeles utilises par ``pipeline/`` sont entraines par
``train_measure_validity.py`` (une seule approche, "rf_related"), qui est le
script a lancer pour (re)produire les ``.joblib`` consommes par
``pipeline/processing/measurement_classifier.py``. Desactive par defaut : ce
script n'est pas appele par un README ou une CLI de production, il se lance a
la main quand on veut reevaluer si "rf_related" reste le meilleur choix.

Toutes les sorties vont dans ``results/meas_classifier/comparison/`` a la racine
du depot (pas dans ``retained_models/``, qui porte les seuils lus par
``pipeline/``), pour ne jamais ecraser les seuils de production.

Usage
-----
::

    python compare_measure_validity_approaches.py

Ne duplique pas ``dataset.py`` / ``insect_anatomy.py`` (import direct) ni
``evaluation.py`` / ``features.py``, qui appartiennent a l'ancien script de
comparaison ``conf_classifier.py`` et ne sont pas utilises ici.
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # pas de fenetre : on ecrit des PNG
import matplotlib.pyplot as plt

from insect_anatomy import INSECT_GROUPS, MEASUREMENTS, POINTS
from dataset import STATUS_SUFFIX, load_annotation_data

from measure_validity_lib import (
    ALL_APPROACHES,
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

APPROACHES = ALL_APPROACHES                # les 8 approches comparees
NAMES = names_of(APPROACHES)

# --- chemins ----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# Meme entree unique que l'entrainement de production : la table d'annotation du
# depot (annotation_tools/build_annotation_data.py).
ANNOTATION_DATA = REPO_ROOT / "annotation_data" / "annotation_data.csv"
# Dans results/ commun du depot, a cote (jamais a la place) des sorties de production.
OUT_DIR = REPO_ROOT / "results" / "meas_classifier" / "comparison"

# --- protocole ---------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 0
MIN_MINORITY = 20   # garde-fou : mesure ignoree en dessous de ce nombre
NA_FILL = -1.0       # sentinelle d'imputation pour la random forest


def main() -> None:
    for sub in ("pr_curves", "confusion", "importance"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # --- Chargement des donnees -----------------------------------------------
    if not ANNOTATION_DATA.is_file():
        raise SystemExit(
            f"Table d'annotation absente : {ANNOTATION_DATA}\n"
            "La construire avec : python annotation_tools/build_annotation_data.py"
        )
    frame, columns = load_annotation_data(ANNOTATION_DATA)
    group_cols = [f"{g}_one_hot" for g in INSECT_GROUPS]
    print(f"{len(frame)} images | {columns.summary()}")

    # Prevalence par mesure : sert au garde-fou et a la lecture des accuracies.
    prevalence = []
    for measure in MEASUREMENTS:
        status = f"{measure}{STATUS_SUFFIX}"
        if status not in frame.columns:
            continue
        y = 1 - frame[status].astype(int).to_numpy()   # 1 = non mesurable
        prevalence.append({
            "measure": measure,
            "n": len(y),
            "n_unmeasurable": int(y.sum()),
            "prevalence_unmeasurable": float(y.mean()),
            "kept": bool(min(y.sum(), len(y) - y.sum()) >= MIN_MINORITY),
        })
    prevalence = pd.DataFrame(prevalence)
    prevalence.to_csv(OUT_DIR / "prevalence.csv", index=False)

    kept = prevalence.loc[prevalence["kept"], "measure"].tolist()
    skipped = prevalence.loc[~prevalence["kept"], "measure"].tolist()
    print(f"{len(kept)} mesures retenues, {len(skipped)} ignorees (< {MIN_MINORITY} exemples minoritaires)")
    print("ignorees :", skipped)

    # Memes features que la production : geometrie des keypoints ramenee dans leur
    # boite. Les approches "regle" gardent les confiances, un seuil n'ayant de sens
    # que sur un score (voir measure_validity_lib.make_feature_sets).
    coords = add_relative_coordinates(frame, columns)
    all_coords = make_coord_columns(coords, POINTS)

    # --- Boucle principale (pas de sauvegarde de modele : models_dir=None) ----
    rows, oof = [], {}
    for i, measure in enumerate(kept, start=1):
        print(f"[{i:2d}/{len(kept)}] {measure}", flush=True)
        result = evaluate_measure(
            frame, columns, coords, all_coords, group_cols, measure, STATUS_SUFFIX,
            N_FOLDS, RANDOM_STATE, NA_FILL, APPROACHES, models_dir=None,
        )
        rows.extend(metric_rows(measure, result, APPROACHES))
        plot_pr(OUT_DIR, measure, result, APPROACHES)
        plot_confusion(OUT_DIR, measure, result, APPROACHES)
        oof[measure] = {"y": result["y"], **{n: result["preds"][n] for n in NAMES}}
        del result
        gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\n{len(metrics)} lignes ecrites dans {OUT_DIR / 'metrics.csv'}")

    # --- Comparaison des approches -----------------------------------------
    mcc = metrics.pivot(index="measure", columns="model", values="mcc")[NAMES]
    ranks = mcc.rank(axis=1, ascending=False)

    ranking = pd.DataFrame({
        "mean_rank": ranks.mean(),
        "median_mcc": mcc.median(),
        "mean_mcc": mcc.mean(),
        "mean_ap_norm": metrics.pivot(index="measure", columns="model",
                                      values="average_precision_norm")[NAMES].mean(),
        "mean_accuracy_unmeasurable": metrics.pivot(index="measure", columns="model",
                                                    values="accuracy_unmeasurable")[NAMES].mean(),
        "wins": mcc.idxmax(axis=1).value_counts().reindex(NAMES).fillna(0).astype(int),
    }).sort_values("mean_rank")
    ranking.to_csv(OUT_DIR / "ranking.csv")

    # Heatmap mesure x approche (MCC)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(mcc) + 2.5))
    image = ax.imshow(mcc.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(NAMES)))
    ax.set_yticks(range(len(mcc)))
    ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mcc.index, fontsize=8)
    for i in range(mcc.shape[0]):
        for j in range(mcc.shape[1]):
            value = mcc.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if value < 0.6 else "black")
    fig.colorbar(image, ax=ax, label="MCC")
    ax.set_title("MCC out-of-fold par mesure et par approche")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_mcc.png", dpi=140)
    plt.close(fig)

    # Barplot du rang moyen
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ranking.index[::-1], ranking["mean_rank"][::-1], color="steelblue")
    for name, value in zip(ranking.index[::-1], ranking["mean_rank"][::-1]):
        ax.text(value + 0.05, name, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlabel(f"MCC mean rank on {len(mcc)} measures (1 = best)")
    ax.set_title("Approach ranking")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranking.png", dpi=140)
    plt.close(fig)
    print("Figures ecrites :", len(list(OUT_DIR.rglob("*.png"))))

    # --- Tableau recapitulatif : MCC, precision, rappel (classe "non mesurable")
    summary = (
        metrics
        .rename(columns={
            "precision_unmeasurable": "precision",
            "accuracy_unmeasurable": "recall",
        })
        .groupby("model")[["mcc", "precision", "recall"]]
        .agg(["mean", "std"])
        .reindex(NAMES)
        .sort_values(("mcc", "mean"), ascending=False)
        .round(3)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_mesures", metrics.groupby("model")["measure"].nunique().reindex(summary.index))
    summary.to_csv(OUT_DIR / "summary_mcc_precision_recall.csv")

    # --- Consequence pratique : mesures exploitables par image -----------------
    baseline = "seuil_conf_min"          # "avant" : seuil sur la confiance minimale
    best = ranking.index[1] if len(ranking.index) > 1 else ranking.index[0]
    print(f"avant = {baseline} | apres = {best}")

    y_true = np.column_stack([oof[m]["y"] for m in kept])        # 1 = non mesurable
    valid = y_true == 0                                          # mesure reellement exploitable

    practical_rows, per_image = [], {
        "image_name": frame["image_name"], "group": frame["group"].astype(str),
    }

    def summarise(label, rejected):
        keep = ~rejected
        keep_valid = keep & valid
        per_image[f"n_conservees_{label}"] = keep.sum(axis=1)
        per_image[f"n_valides_conservees_{label}"] = keep_valid.sum(axis=1)
        practical_rows.append({
            "filtrage": label,
            "mesures_conservees_par_image": keep.sum(axis=1).mean(),
            "dont_reellement_valides": keep_valid.sum(axis=1).mean(),
            "valides_perdues_par_image": (valid & rejected).sum(axis=1).mean(),
            "retention_des_valides": keep_valid.sum() / valid.sum(),
            "contamination_des_conservees": (keep & ~valid).sum() / max(keep.sum(), 1),
        })

    summarise("sans_filtrage", np.zeros_like(y_true, dtype=bool))
    for name in (baseline, best):
        if name not in NAMES:
            continue
        predicted = np.column_stack([oof[m][name] for m in kept])
        summarise(name, predicted == 1)

    practical = pd.DataFrame(practical_rows).set_index("filtrage").round(3)
    practical.to_csv(OUT_DIR / "impact_filtrage.csv")
    pd.DataFrame(per_image).to_csv(OUT_DIR / "mesures_par_image.csv", index=False)
    print(f"\n{len(kept)} mesures evaluees, {valid.mean():.1%} reellement exploitables\n")

    # --- Nombre de variables d'entree par approche ------------------------------
    counts = []
    for measure in kept:
        sets = make_feature_sets(columns, coords, all_coords, measure)
        for name, kind, which, _ in APPROACHES:
            used = sets[f"{which}_conf"] if kind == "rule" else sets[which]
            counts.append({
                "measure": measure,
                "model": name,
                "n_features": len(used) + (len(group_cols) if kind == "model" else 0),
            })
    n_features = (
        pd.DataFrame(counts)
        .groupby("model")["n_features"]
        .agg(mean_n_features="mean", min_n_features="min", max_n_features="max")
        .reindex(NAMES)
        .round(2)
    )
    n_features["mean_mcc"] = metrics.groupby("model")["mcc"].mean().round(3)
    n_features = n_features.sort_values("mean_n_features")
    n_features.to_csv(OUT_DIR / "n_features.csv")

    # --- Figures complementaires --------------------------------------------
    # Repartition des labels par mesure (barres empilees)
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
    fig.savefig(OUT_DIR / "labels_par_mesure.png", dpi=140)
    plt.close(fig)
    print("ecrit :", OUT_DIR / "labels_par_mesure.png")

    # MCC par mesure : xgb_related vs rf_related
    pair = [n for n in ("xgb_related", "rf_related") if n in NAMES]
    if len(pair) == 2:
        pair_mcc = mcc[pair].sort_values(pair[0])
        positions = np.arange(len(pair_mcc))
        height = 0.38

        fig, ax = plt.subplots(figsize=(9, 0.42 * len(pair_mcc) + 2))
        ax.barh(positions + height / 2, pair_mcc[pair[0]], height=height,
                color="#4472c4", label=pair[0])
        ax.barh(positions - height / 2, pair_mcc[pair[1]], height=height,
                color="#ed7d31", label=pair[1])
        ax.set_yticks(positions)
        ax.set_yticklabels(pair_mcc.index, fontsize=8)
        ax.set_xlim(min(0.0, float(pair_mcc.min().min()) - 0.05), 1.0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("MCC (out-of-fold)")
        ax.set_title("MCC per measure : models trained on all keypoints and measures")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "mcc_xgb_related_vs_rf_related.png", dpi=140)
        plt.close(fig)
        print("ecrit :", OUT_DIR / "mcc_xgb_related_vs_rf_related.png")

    # --- Importance des variables ------------------------------------------------
    # Le modele le mieux classe (hors regles de seuil, qui n'ont pas de
    # variables) est reentraine sur l'integralite des donnees, mesure par
    # mesure. Ces importances sont descriptives : elles servent a voir quels
    # keypoints portent le signal, pas a mesurer une performance.
    model_approaches = {a[0]: (a[2], a[3]) for a in APPROACHES if a[1] == "model"}
    best_model = next((name for name in ranking.index if name in model_approaches), None)
    if best_model is not None:
        which, factory = model_approaches[best_model]
        print(f"Meilleur modele : {best_model} (features : {which})")

        def pretty(column: str) -> str:
            """'head-top kp_x_rel' -> 'head-top x', 'coleoptera_one_hot' -> 'coleoptera'."""
            for suffix, axis in ((REL_X_SUFFIX, " x"), (REL_Y_SUFFIX, " y")):
                if column.endswith(suffix):
                    return column[: -len(suffix)] + axis
            return column.replace("_one_hot", "")

        series = []
        for measure in kept:
            cols = make_feature_sets(columns, coords, all_coords, measure)[which]
            if not cols:
                continue
            y = make_target(frame, STATUS_SUFFIX, measure)
            data = frame[cols + group_cols]
            if best_model.startswith("rf"):
                data = data.fillna(NA_FILL)

            model = factory(y, RANDOM_STATE)
            model.fit(data, y)
            values = pd.Series(model.feature_importances_, index=[pretty(c) for c in data.columns])
            del model
            gc.collect()
            series.append(values.rename(measure))

            top = values.sort_values().tail(15)
            fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top) + 1.6))
            ax.barh(top.index, top.to_numpy(), color="steelblue")
            ax.set_xlabel("importance")
            ax.set_title(f"{best_model} : {measure}", fontsize=10)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "importance" / f"{slug(measure)}.png", dpi=140)
            plt.close(fig)

        importance = pd.concat(series, axis=1).T
        importance.to_csv(OUT_DIR / "feature_importance.csv")
        print(f"{len(importance)} mesures, {importance.shape[1]} variables")

        # Importance moyenne sur l'ensemble des mesures
        mean_importance = importance.mean().sort_values().tail(20)
        coverage = importance.notna().sum()

        fig, ax = plt.subplots(figsize=(7, 0.32 * len(mean_importance) + 2))
        ax.barh(mean_importance.index, mean_importance.to_numpy(), color="#4472c4")
        for i, name in enumerate(mean_importance.index):
            ax.text(mean_importance[name], i, f"  {coverage[name]}/{len(importance)}",
                    va="center", fontsize=7, color="grey")
        ax.set_xlim(0, float(mean_importance.max()) * 1.12)
        ax.set_xlabel("mean importance (grey number : number of measure where variable exists)")
        ax.set_title(f"Most used variables ({best_model})")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "importance_moyenne.png", dpi=140)
        plt.close(fig)
        print("ecrit :", OUT_DIR / "importance_moyenne.png")

    # --- Mesures dont le label est quasi determine par le groupe taxonomique ---
    suspects = []
    for measure in kept:
        y = pd.Series(make_target(frame, STATUS_SUFFIX, measure))
        by_group = y.groupby(frame["group"].astype(str).to_numpy()).mean()
        if ((by_group < 0.05) | (by_group > 0.95)).all():
            suspects.append({"measure": measure, **by_group.round(2).to_dict()})

    suspects = pd.DataFrame(suspects)
    if not suspects.empty:
        suspects.to_csv(OUT_DIR / "group_determined_measures.csv", index=False)
        print("Taux de non-mesurable par groupe (mesures triviales) :")
        print(suspects)


if __name__ == "__main__":
    main()
