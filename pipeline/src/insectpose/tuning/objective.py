"""Objectif Optuna generique (CONVENTIONS.md §6.3).

Un trial = un run complet, avec son propre run_id et son manifeste : les trials sont
donc auditables et re-evaluables comme n'importe quel run. L'objectif est TOUJOURS la
metrique primaire calculee par l'evaluateur partage, jamais une loss de framework.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf

from insectpose.paths import ProjectPaths
from insectpose.registry import APPROACHES
from insectpose.tuning.search_spaces import to_hydra_overrides
from insectpose.utils.hashing import short_hash, stable_hash
from insectpose.utils.io import write_json
from insectpose.utils.logging import get_logger

log = get_logger("tuning")

RunFn = Callable[[DictConfig, dict[str, Any]], float]


def build_study(cfg: DictConfig, paths: ProjectPaths, suffix: str = "") -> optuna.Study:
    """Cree ou reprend une etude Optuna. Effet de bord : cree runs/optuna/<study>.db."""
    study_name = study_name_for(cfg, suffix)
    sampler = (
        optuna.samplers.TPESampler(
            seed=int(cfg.tuning.seed),
            n_startup_trials=int(cfg.tuning.get("n_startup_trials", 10)),
        )
        if str(cfg.tuning.sampler) == "tpe"
        else optuna.samplers.RandomSampler(seed=int(cfg.tuning.seed))
    )
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=int(cfg.tuning.pruner_warmup_steps))
        if str(cfg.tuning.pruner) == "median"
        else optuna.pruners.NopPruner()
    )
    storage = None
    if str(cfg.tuning.storage) == "sqlite":
        db = paths.optuna_storage(study_name)
        db.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db}"
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=str(cfg.eval.primary_direction),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(cfg.tuning.load_if_exists),
    )


def protocol_hash(cfg: DictConfig) -> str:
    """Empreinte de l'espace de recherche ET des reglages de tuning (ADR-0031).

    Elle entre dans le nom de l'etude : modifier l'espace ou le budget cree donc
    AUTOMATIQUEMENT une nouvelle etude, et l'ancienne reste intacte a cote. Sans cela,
    une reprise apres modification melangerait des trials evalues sous deux protocoles
    differents — le TPE construirait ses densites sur du bruit, et `best_trial` pourrait
    retenir un trial dont les parametres ne sont meme plus cherches.
    """
    payload = {
        "search_space": OmegaConf.to_container(
            cfg.approach.get("search_space", {}), resolve=True),
        "tuning": {
            key: OmegaConf.to_container(cfg.tuning[key], resolve=True)
            if hasattr(cfg.tuning[key], "keys") else cfg.tuning[key]
            for key in ("mode", "n_trials", "inner_folds", "n_startup_trials",
                        "sampler", "pruner", "pruner_warmup_steps")
            if key in cfg.tuning
        },
        "epochs": int(cfg.train.epochs),
    }
    return short_hash(stable_hash(payload), 6)


def study_name_for(cfg: DictConfig, suffix: str = "") -> str:
    """Nom canonique : <approche>__<split_id>__<metrique>__sp<hash>[__<suffixe>]."""
    name = (f"{cfg.approach.name}__{cfg.split_id}__{cfg.eval.primary_metric}"
            f"__sp{protocol_hash(cfg)}")
    return f"{name}__{suffix}" if suffix else name


def make_objective(cfg: DictConfig, run_fn: RunFn) -> Callable[[optuna.Trial], float]:
    """Fabrique l'objectif : echantillonne l'espace, execute les folds, retourne la moyenne.

    `run_fn(cfg, overrides) -> valeur de la metrique primaire` est injectee par le
    pipeline : le module de tuning n'appelle jamais directement une approche.
    """
    approach_cls = APPROACHES.get(str(cfg.approach.name))

    def objective(trial: optuna.Trial) -> float:
        overrides = approach_cls.search_space(trial, cfg)
        folds = list(range(int(cfg.tuning.inner_folds)))
        values: list[float] = []
        for step, fold in enumerate(folds):
            value = run_fn(cfg, {**overrides, "fold": fold, "trial_number": trial.number,
                                 "optuna_study": study_name_for(cfg)})
            values.append(value)
            trial.report(float(np.mean(values)), step=step)
            if trial.should_prune():
                log.info("Trial %d elague apres %d fold(s).", trial.number, step + 1)
                raise optuna.TrialPruned
        trial.set_user_attr("overrides", to_hydra_overrides(overrides))
        trial.set_user_attr("per_fold", values)
        return float(np.mean(values))

    return objective


def completed_trials(study: optuna.Study) -> int:
    """Nombre de trials REELLEMENT termines dans l'etude."""
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def remaining_trials(study: optuna.Study, budget: int) -> int:
    """Trials restants pour atteindre le budget TOTAL de l'etude (§6.3).

    `study.optimize(n_trials=N)` ajoute N trials A CHAQUE APPEL. Sur une etude reprise
    apres interruption, cela gonflerait le budget d'un fold sans toucher aux autres, et
    la comparaison entre folds — puis entre approches — mesurerait le budget autant que
    la methode. On vise donc un total, pas un increment.
    """
    done = completed_trials(study)
    return max(0, int(budget) - done)


def save_best(study: optuna.Study, cfg: DictConfig, paths: ProjectPaths) -> dict[str, Any]:
    """Serialise le meilleur trial et le budget effectivement consomme (§6.3).

    Effet de bord : ecrit runs/optuna/<study>_best.json.
    """
    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "approach": str(cfg.approach.name),
        "split_id": str(cfg.split_id),
        "primary_metric": str(cfg.eval.primary_metric),
        "direction": str(cfg.eval.primary_direction),
        "mode": str(cfg.tuning.mode),
        "inner_folds": int(cfg.tuning.inner_folds),
        "n_trials_requested": int(cfg.tuning.n_trials),
        "n_startup_trials": int(cfg.tuning.get("n_startup_trials", 10)),
        "protocol_hash": protocol_hash(cfg),
        "epochs": int(cfg.train.epochs),
        "n_trials_completed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": best.params,
        "best_overrides": best.user_attrs.get("overrides", []),
        "per_fold": best.user_attrs.get("per_fold", []),
    }
    write_json(paths.runs / "optuna" / f"{study.study_name}_best.json", payload)
    return payload