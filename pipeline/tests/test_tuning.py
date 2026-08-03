"""Tests du tuning : l'objectif doit venir de l'evaluateur, jamais d'un framework (§6.3)."""

from __future__ import annotations

import optuna
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.tuning.objective import study_name_for
from insectpose.tuning.search_spaces import suggest_from_spec, to_hydra_overrides


def test_search_space_translation() -> None:
    spec = {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "depth": {"type": "int", "low": 1, "high": 4},
        "head": {"type": "categorical", "choices": ["a", "b"]},
    }
    study = optuna.create_study()
    trial = study.ask()
    values = suggest_from_spec(trial, spec, prefix="approach")
    assert set(values) == {"approach.lr", "approach.depth", "approach.head"}
    assert all("=" in o for o in to_hydra_overrides(values))


def test_unknown_space_type_is_rejected() -> None:
    study = optuna.create_study()
    with pytest.raises(ValueError, match="inconnu"):
        suggest_from_spec(study.ask(), {"x": {"type": "gaussian"}})


def test_study_name_is_canonical(cfg) -> None:
    OmegaConf.update(cfg, "split_id", "kfold5_grouped_seed42_pooled", force_add=True)
    assert study_name_for(cfg).startswith("mean_pose__kfold5_grouped_seed42_pooled__")
    assert study_name_for(cfg, "outer2").endswith("__outer2")


@pytest.mark.smoke
def test_nested_tuning_never_searches_on_outer_test(cfg, project) -> None:
    """Coeur du protocole (ADR-0012) : aucun trial ne voit une image de test externe."""

    from insectpose.data.splits import inner_split_id, make_split_id
    from insectpose.utils.io import read_json, read_parquet

    OmegaConf.update(cfg, "tuning.n_trials", 2)
    OmegaConf.update(cfg, "tuning.inner_folds", 2)
    OmegaConf.update(cfg, "tuning.mode", "nested")
    pipeline.cmd_split(cfg)

    outer_id = make_split_id(cfg)
    outer = read_parquet(project.split_file(outer_id))
    for outer_fold in sorted(outer["fold"].unique()):
        inner = read_parquet(project.split_file(inner_split_id(outer_id, int(outer_fold))))
        outer_test = set(outer.loc[(outer["fold"] == outer_fold) & (outer["role"] == "test"),
                                   "image_id"])
        assert not (set(inner["image_id"]) & outer_test), (
            f"fuite : le decoupage interne du fold {outer_fold} contient du test externe"
        )

    results = pipeline.cmd_tune(cfg)
    assert results["mode"] == "nested"
    assert len(results["final_runs"]) == int(cfg.cv.n_folds)
    # Chaque fold externe a ses propres hyperparametres, issus de sa propre recherche
    for outer_fold, run_id in results["final_runs"].items():
        manifest = read_json(project.manifest(run_id))
        assert manifest["hpo_source_fold"] == outer_fold
        assert manifest["split_id"] == outer_id
    assert (project.runs / "optuna").exists()


@pytest.mark.smoke
def test_tune_once_reuses_one_search_for_all_folds(cfg, project) -> None:
    from insectpose.utils.io import read_json

    OmegaConf.update(cfg, "tuning.n_trials", 2)
    OmegaConf.update(cfg, "tuning.inner_folds", 2)
    OmegaConf.update(cfg, "tuning.mode", "tune_once")
    OmegaConf.update(cfg, "tuning.tuning_outer_fold", 1)
    pipeline.cmd_split(cfg)
    results = pipeline.cmd_tune(cfg)
    assert set(results["outer"]) == {1}
    for run_id in results["final_runs"].values():
        assert read_json(project.manifest(run_id))["hpo_source_fold"] == 1


@pytest.mark.smoke
def test_hpo_trials_are_excluded_from_results(cfg, project) -> None:
    """Un run d'HPO a servi a CHOISIR des hyperparametres : ce n'est pas un resultat.

    Il tourne sur un decoupage interne ; l'agreger avec les runs finaux melangerait
    exploration et evaluation dans le meme tableau.
    """
    from insectpose.evaluation.aggregate import final_runs, summary_table, write_master
    from insectpose.utils.io import read_parquet

    OmegaConf.update(cfg, "tuning.n_trials", 2)
    OmegaConf.update(cfg, "tuning.inner_folds", 2)
    OmegaConf.update(cfg, "tuning.mode", "tune_once")
    pipeline.cmd_split(cfg)
    pipeline.cmd_tune(cfg)

    master = read_parquet(write_master(project))
    assert set(master["role_in_protocol"]) == {"final", "hpo_trial"}
    citable = final_runs(master)
    assert citable["run_id"].nunique() < master["run_id"].nunique()
    # Les runs citables sont ceux du decoupage EXTERNE
    assert citable["split_id"].nunique() == 1
    assert "__outer" not in citable["split_id"].iloc[0]

    summary = summary_table(master, str(cfg.eval.primary_metric))
    assert summary["n_folds"].iloc[0] == int(cfg.cv.n_folds)