#!/usr/bin/env python
"""Figures de diagnostic des etudes Optuna (ADR-0031).

Elles ne mesurent pas la performance d'un modele — c'est le role de `report` — mais la
qualite de la RECHERCHE : le budget a-t-il servi a quelque chose, quels
hyperparametres comptent, et la recherche avait-elle converge au moment de l'arret ?

C'est ce qui justifie le budget depense, et ce qui oriente le prochain espace de
recherche. Sans ces figures, on ne sait pas si 20 trials suffisaient ou si l'on s'est
arrete trop tot.

Usage :
    python plot_optuna.py                          # toutes les etudes
    python plot_optuna.py --studies yolo_pooled     # filtre par sous-chaine
    python plot_optuna.py --out-dir results/optuna --dpi 200

Sortie : un dossier par etude, sous `results/optuna/<nom_etude>/`.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def charger_etudes(racine: Path, filtre: str | None = None) -> list[optuna.Study]:
    """Toutes les etudes des bases SQLite de `runs/optuna/`."""
    etudes: list[optuna.Study] = []
    for base in sorted(glob.glob(str(racine / "*.db"))):
        stockage = f"sqlite:///{base}"
        for nom in optuna.get_all_study_names(stockage):
            if filtre and filtre not in nom:
                continue
            etudes.append(optuna.load_study(study_name=nom, storage=stockage))
    return etudes


def trials_termines(etude: optuna.Study) -> int:
    """Trials REELLEMENT termines : les elagues et les interrompus ne comptent pas."""
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in etude.trials)


def tracer_historique(etude: optuna.Study, out_dir: Path, dpi: int) -> Path | None:
    """Valeur de chaque trial et meilleure valeur cumulee.

    Une courbe de meilleure valeur encore croissante a la fin signale une recherche
    coupee trop tot ; un plateau precoce signale au contraire un budget suffisant, ou
    un espace de recherche trop etroit.
    """
    complets = [t for t in etude.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complets) < 2:
        return None

    numeros = [t.number for t in complets]
    valeurs = [t.value for t in complets]
    maximiser = etude.direction == optuna.study.StudyDirection.MAXIMIZE
    cumule, courant = [], (-float("inf") if maximiser else float("inf"))
    for valeur in valeurs:
        courant = max(courant, valeur) if maximiser else min(courant, valeur)
        cumule.append(courant)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(numeros, valeurs, s=28, alpha=0.75, label="trial", zorder=3)
    ax.plot(numeros, cumule, color="crimson", lw=1.6, label="best so far", zorder=2)

    elagues = [t.number for t in etude.trials
               if t.state == optuna.trial.TrialState.PRUNED]
    if elagues:
        # Les trials elagues n'ont pas de valeur : on les marque sur l'axe, car leur
        # nombre dit si le pruner a fait son travail (ADR-0031).
        ax.scatter(elagues, [min(valeurs)] * len(elagues), marker="x", s=30,
                   color="grey", label=f"pruned ({len(elagues)})", zorder=3)

    ax.set_xlabel("trial")
    ax.set_ylabel(etude.metric_names[0] if etude.metric_names else "objective")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Optimisation history — {trials_termines(etude)} completed trials")
    return _sauver(fig, out_dir / "optuna_history.png", dpi)


def tracer_importances(etude: optuna.Study, out_dir: Path, dpi: int) -> Path | None:
    """Importance relative de chaque hyperparametre (fANOVA).

    Un parametre a importance quasi nulle occupe une dimension du budget pour rien :
    c'est le premier candidat au retrait de l'espace de recherche. A l'inverse, un
    parametre dominant merite peut-etre des bornes elargies.
    """
    if trials_termines(etude) < 4:
        return None
    try:
        importances = optuna.importance.get_param_importances(etude)
    except Exception as exc:  # noqa: BLE001 - depend du nombre de trials et du sampler
        print(f"    importance indisponible : {exc}")
        return None
    if not importances:
        return None

    noms = list(importances)[::-1]
    valeurs = [importances[n] for n in noms]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(noms) + 2.2))
    ax.barh(noms, valeurs, color="#4c78a8", edgecolor="black", linewidth=0.4)
    for i, valeur in enumerate(valeurs):
        ax.text(valeur, i, f" {valeur:.3f}", va="center", fontsize=8)
    ax.set_xlabel("relative importance")
    ax.set_xlim(0, max(valeurs) * 1.18)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"Hyperparameter importance — {trials_termines(etude)} trials")
    return _sauver(fig, out_dir / "optuna_importance.png", dpi)


def tracer_contours(etude: optuna.Study, out_dir: Path, dpi: int,
                    max_params: int = 4) -> Path | None:
    """Contours des paires d'hyperparametres les plus importantes.

    Montre OU se situe l'optimum dans l'espace : colle a une borne, il faut elargir la
    plage ; au centre d'un plateau, la recherche a converge.

    Limite aux `max_params` parametres les plus influents : au-dela, la grille devient
    illisible et le nombre de trials ne suffit plus a estimer une surface.
    """
    if trials_termines(etude) < 6:
        return None
    try:
        importances = optuna.importance.get_param_importances(etude)
        params = list(importances)[:max_params]
    except Exception:  # noqa: BLE001
        params = list(etude.best_params)[:max_params]
    if len(params) < 2:
        return None

    try:
        axes = optuna.visualization.matplotlib.plot_contour(etude, params=params)
    except Exception as exc:  # noqa: BLE001
        print(f"    contours indisponibles : {exc}")
        return None

    fig = axes.figure if hasattr(axes, "figure") else axes[0][0].figure
    taille = max(3.0 * len(params), 8)
    fig.set_size_inches(taille, taille)
    fig.suptitle(f"Contour plots — {trials_termines(etude)} trials", y=1.0)
    return _sauver(fig, out_dir / "optuna_contour.png", dpi)


def tracer_coordonnees(etude: optuna.Study, out_dir: Path, dpi: int) -> Path | None:
    """Coordonnees paralleles : chaque ligne est un trial, coloree par sa valeur.

    Complement des contours : elle montre d'un coup d'oeil quelles COMBINAISONS de
    valeurs donnent de bons resultats, la ou les contours ne prennent les parametres
    que deux a deux.
    """
    if trials_termines(etude) < 4:
        return None
    try:
        axe = optuna.visualization.matplotlib.plot_parallel_coordinate(etude)
    except Exception as exc:  # noqa: BLE001
        print(f"    coordonnees paralleles indisponibles : {exc}")
        return None
    fig = axe.figure
    fig.set_size_inches(11, 5.5)
    return _sauver(fig, out_dir / "optuna_parallel.png", dpi)


def ecrire_table(etude: optuna.Study, out_dir: Path) -> Path:
    """Table des trials, pour audit. Effet de bord : ecrit optuna_trials.csv."""
    table = etude.trials_dataframe(attrs=("number", "value", "state", "params",
                                          "datetime_start", "duration"))
    chemin = out_dir / "optuna_trials.csv"
    table.to_csv(chemin, index=False)
    return chemin


def _sauver(fig: Any, chemin: Path, dpi: int) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return chemin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="Racine du projet.")
    parser.add_argument("--studies", default=None,
                        help="Ne traiter que les etudes dont le nom contient cette chaine.")
    parser.add_argument("--out-dir", default="results/optuna")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    racine = Path(args.root) / "runs" / "optuna"
    if not racine.exists():
        raise SystemExit(f"{racine} absent : aucune etude Optuna a tracer.")

    etudes = charger_etudes(racine, args.studies)
    if not etudes:
        raise SystemExit("Aucune etude ne correspond au filtre.")

    resume = []
    for etude in etudes:
        complets = trials_termines(etude)
        print(f"\n{etude.study_name}  ({complets} trials termines)")
        if complets == 0:
            print("    aucun trial termine : rien a tracer.")
            continue

        out_dir = Path(args.root) / args.out_dir / etude.study_name
        produites = [
            tracer_historique(etude, out_dir, args.dpi),
            tracer_importances(etude, out_dir, args.dpi),
            tracer_contours(etude, out_dir, args.dpi),
            tracer_coordonnees(etude, out_dir, args.dpi),
        ]
        ecrire_table(etude, out_dir)
        for chemin in [p for p in produites if p is not None]:
            print(f"    {chemin}")

        resume.append({
            "study": etude.study_name,
            "trials": complets,
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in etude.trials),
            "best_value": round(etude.best_value, 4),
            "best_trial": etude.best_trial.number,
        })

    if resume:
        print("\n=== Resume ===")
        print(pd.DataFrame(resume).to_string(index=False))
        print("\nLecture :")
        print("  - 'best so far' encore croissant a la fin => recherche coupee trop tot ;")
        print("  - un parametre a importance quasi nulle occupe une dimension pour rien ;")
        print("  - un optimum colle a une borne => plage a elargir (nouvel ADR).")


if __name__ == "__main__":
    main()