"""Generic Optuna objective (CONVENTIONS.md §6.3).

A trial = a complete run, with its own run_id and manifest: the trials can therefore be
audited and re-evaluated like any run. The objective is ALWAYS the primary metric
computed by the shared evaluator, never a framework loss.
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
    """Create or resume an Optuna study. Side effect: creates runs/optuna/<study>.db."""
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
    """Fingerprint of the search space AND of the tuning settings (ADR-0031).

    It enters the name of the study: changing the space or the budget therefore
    AUTOMATICALLY creates a new study, and the old one stays intact next to it. Without
    it, a resume after a change would mix trials evaluated under two different
    protocols — the TPE would build its densities on noise, and `best_trial` could retain
    a trial whose parameters are not even searched any more.
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
    """Canonical name: <approach>__<split_id>__<metric>__sp<hash>[__<suffix>]."""
    name = (f"{cfg.approach.name}__{cfg.split_id}__{cfg.eval.primary_metric}"
            f"__sp{protocol_hash(cfg)}")
    return f"{name}__{suffix}" if suffix else name


def make_objective(cfg: DictConfig, run_fn: RunFn) -> Callable[[optuna.Trial], float]:
    """Build the objective: samples the space, runs the folds, returns the mean.

    `run_fn(cfg, overrides) -> value of the primary metric` is injected by the pipeline:
    the tuning module never calls an approach directly.
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
                log.info("Trial %d pruned after %d fold(s).", trial.number, step + 1)
                raise optuna.TrialPruned
        trial.set_user_attr("overrides", to_hydra_overrides(overrides))
        trial.set_user_attr("per_fold", values)
        return float(np.mean(values))

    return objective


def completed_trials(study: optuna.Study) -> int:
    """Number of trials ACTUALLY finished in the study."""
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def remaining_trials(study: optuna.Study, budget: int) -> int:
    """Trials left to reach the TOTAL budget of the study (§6.3).

    `study.optimize(n_trials=N)` adds N trials AT EVERY CALL. On a study resumed after an
    interruption, it would inflate the budget of one fold without touching the others,
    and the comparison between folds — then between approaches — would measure the
    budget as much as the method. A total is therefore targeted, not an increment.
    """
    done = completed_trials(study)
    return max(0, int(budget) - done)


def save_best(study: optuna.Study, cfg: DictConfig, paths: ProjectPaths) -> dict[str, Any]:
    """Serialise the best trial and the budget actually consumed (§6.3).

    Side effect: writes runs/optuna/<study>_best.json.
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