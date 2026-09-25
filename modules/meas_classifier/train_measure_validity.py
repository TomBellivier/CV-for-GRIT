"""Entrainement de production des classifieurs de validite de mesure.

Pour chaque mesure (27), entraine un RandomForestClassifier ("rf_related" :
coordonnees des keypoints du voisinage anatomique de la mesure, ramenees dans
leur boite englobante, + groupe taxonomique en one-hot) qui predit si la mesure
automatique est mesurable ou non, et
sauvegarde un ``.joblib`` par mesure dans
``retained_models/measurement_validity/``, avec le ``metrics.csv`` qui porte leurs
seuils de decision -- ce sont les modeles charges tels quels par
``pipeline/processing/measurement_classifier.py`` (voir
``pipeline/processing/config.py`` : ``MEASUREMENT_CLASSIFIER_DIR`` /
``MEASUREMENT_CLASSIFIER_METRICS_CSV``). Les sorties de recherche (courbes PR,
matrices de confusion, importances) vont dans ``results/meas_classifier/training/``
a la racine du depot.

Convention : la classe positive est "non mesurable" (classe minoritaire).
Protocole : 5-fold stratifie, predictions out-of-fold ; le seuil de decision
(``threshold_median`` dans ``metrics.csv``) est choisi sur une
partition interne du train (maximisation du MCC), jamais sur le test.

C'est la version *production* : une seule approche (rf_related). Pour
comparer les 8 approches evaluees a l'origine (seuils, XGBoost, Random Forest
sur differents jeux de features), voir
``compare_measure_validity_approaches.py`` (recherche uniquement, ne
sauvegarde aucun modele de production).

Usage
-----
::

    python train_measure_validity.py

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

APPROACHES = PRODUCTION_APPROACHES         # une seule approche : rf_related
NAMES = names_of(APPROACHES)
MODEL_NAME, _, MODEL_FEATURES, MODEL_FACTORY = APPROACHES[0]

# --- chemins (a adapter) ----------------------------------------------------
# Ancres sur la racine du depot : le script tourne depuis n'importe quel dossier.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Table d'annotation unique : pose + scale + validite des mesures, une ligne par
# image (voir annotation_tools/build_annotation_data.py). C'est la SEULE entree.
ANNOTATION_DATA = REPO_ROOT / "annotation_data" / "annotation_data.csv"
# Metriques et figures : dans le dossier results/ commun du depot (results/README.md).
OUT_DIR = REPO_ROOT / "results" / "meas_classifier" / "training"
# Modeles de production : ecrits directement la ou pipeline/ les lit, avec le
# metrics.csv qui porte leurs seuils de decision (voir retained_models/README.md).
MODELS_DIR = REPO_ROOT / "retained_models" / "measurement_validity"

# --- protocole ---------------------------------------------------------------
N_FOLDS = 5
RANDOM_STATE = 0
MIN_MINORITY = 20   # garde-fou : mesure ignoree en dessous de ce nombre
NA_FILL = -1.0       # sentinelle d'imputation pour la random forest


def main() -> None:
    for sub in ("pr_curves", "confusion", "importance"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Chargement des donnees -----------------------------------------------
    if not ANNOTATION_DATA.is_file():
        raise SystemExit(
            f"Table d'annotation absente : {ANNOTATION_DATA}\n"
            "La construire avec : python annotation_tools/build_annotation_data.py"
        )
    frame, columns = load_annotation_data(ANNOTATION_DATA)
    group_cols = [f"{g}_one_hot" for g in INSECT_GROUPS]
    print(f"{len(frame)} images | {columns.summary()}")

    # Les predictions out-of-fold sont empilees mesure par mesure : toutes les
    # mesures doivent porter sur les memes lignes. Une image dont une seule mesure
    # n'est pas annotee est donc ecartee, plutot que comptee comme non mesurable.
    status_columns = list(columns.status.values())
    incomplete = frame[status_columns].isna().any(axis=1)
    if incomplete.any():
        print(f"{int(incomplete.sum())} image(s) a l'annotation incomplete ecartee(s) "
              f"({len(frame) - int(incomplete.sum())} conservees)")
        frame = frame.loc[~incomplete].reset_index(drop=True)

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

    # Features : la geometrie des keypoints, ramenee dans leur boite englobante
    # (voir measure_validity_lib.add_relative_coordinates). Les confiances du
    # modele de pose ne servent plus qu'aux approches "regle" de la comparaison.
    coords = add_relative_coordinates(frame, columns)
    all_coords = make_coord_columns(coords, POINTS)
    print(f"{len(coords)}/{len(POINTS)} keypoints en coordonnees relatives "
          f"({len(all_coords)} features geometriques)")

    # --- Boucle principale : entrainement + sauvegarde + metriques ------------
    rows, oof = [], {}
    for i, measure in enumerate(kept, start=1):
        print(f"[{i:2d}/{len(kept)}] {measure}", flush=True)
        result = evaluate_measure(
            frame, columns, coords, all_coords, group_cols, measure, STATUS_SUFFIX,
            N_FOLDS, RANDOM_STATE, NA_FILL, APPROACHES, MODELS_DIR,
        )
        rows.extend(metric_rows(measure, result, APPROACHES))
        plot_pr(OUT_DIR, measure, result, APPROACHES)
        plot_confusion(OUT_DIR, measure, result, APPROACHES)
        oof[measure] = {"y": result["y"], MODEL_NAME: result["preds"][MODEL_NAME]}
        del result
        gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\n{len(metrics)} lignes ecrites dans {OUT_DIR / 'metrics.csv'}")
    # Le seuil de decision fait partie du modele : il part avec les .joblib, sans
    # quoi pipeline/ retombe sur 0.5 pour toutes les mesures.
    metrics.to_csv(MODELS_DIR / "metrics.csv", index=False)
    print(f"Modeles de production et seuils ecrits dans {MODELS_DIR}")

    # --- Resume des performances (un seul modele : pas de classement) ---------
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

    # --- Consequence pratique : mesures exploitables par image -----------------
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
    predicted = np.column_stack([oof[m][MODEL_NAME] for m in kept])
    summarise(MODEL_NAME, predicted == 1)

    practical = pd.DataFrame(practical_rows).set_index("filtrage").round(3)
    practical.to_csv(OUT_DIR / "impact_filtrage.csv")
    pd.DataFrame(per_image).to_csv(OUT_DIR / "mesures_par_image.csv", index=False)
    print(f"\n{len(kept)} mesures evaluees, {valid.mean():.1%} reellement exploitables\n")

    # --- Repartition des labels par mesure (barres empilees) -------------------
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

    # --- Importance des variables -----------------------------------------------
    # Le modele "rf_related" est reentraine sur l'integralite des donnees,
    # mesure par mesure. Ces importances sont descriptives : elles servent a
    # voir quels keypoints portent le signal, pas a mesurer une performance.
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
        ax.set_title(f"{MODEL_NAME} : {measure}", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "importance" / f"{slug(measure)}.png", dpi=140)
        plt.close(fig)

    importance = pd.concat(series, axis=1).T
    importance.to_csv(OUT_DIR / "feature_importance.csv")
    print(f"{len(importance)} mesures, {importance.shape[1]} variables")

    mean_importance = importance.mean().sort_values().tail(20)
    coverage = importance.notna().sum()

    fig, ax = plt.subplots(figsize=(7, 0.32 * len(mean_importance) + 2))
    ax.barh(mean_importance.index, mean_importance.to_numpy(), color="#4472c4")
    for i, name in enumerate(mean_importance.index):
        ax.text(mean_importance[name], i, f"  {coverage[name]}/{len(importance)}",
                va="center", fontsize=7, color="grey")
    ax.set_xlim(0, float(mean_importance.max()) * 1.12)
    ax.set_xlabel("mean importance (grey number : number of measure where variable exists)")
    ax.set_title(f"Most used variables ({MODEL_NAME})")
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
