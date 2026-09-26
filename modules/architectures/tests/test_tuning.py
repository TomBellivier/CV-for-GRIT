"""Tuning tests: the objective must come from the evaluator, never from a framework (§6.3)."""

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
    with pytest.raises(ValueError, match="Unknown"):
        suggest_from_spec(study.ask(), {"x": {"type": "gaussian"}})


def test_study_name_is_canonical(cfg) -> None:
    OmegaConf.update(cfg, "split_id", "kfold5_grouped_seed42_pooled", force_add=True)
    assert study_name_for(cfg).startswith("mean_pose__kfold5_grouped_seed42_pooled__")
    assert study_name_for(cfg, "outer2").endswith("__outer2")


@pytest.mark.smoke
def test_nested_tuning_never_searches_on_outer_test(cfg, project) -> None:
    """Core of the protocol (ADR-0012): no trial sees an outer test image."""

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
            f"leak: the inner split of fold {outer_fold} contains outer test images"
        )

    results = pipeline.cmd_tune(cfg)
    assert results["mode"] == "nested"
    assert len(results["final_runs"]) == int(cfg.cv.n_folds)
    # Each outer fold has its own hyperparameters, from its own search
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
    """An HPO run was used to CHOOSE hyperparameters: it is not a result.

    It runs on an inner split; aggregating it with the final runs would mix exploration
    and evaluation in the same table.
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
    # The quotable runs are those of the OUTER split
    assert citable["split_id"].nunique() == 1
    assert "__outer" not in citable["split_id"].iloc[0]

    summary = summary_table(master, str(cfg.eval.primary_metric))
    assert summary["n_folds"].iloc[0] == int(cfg.cv.n_folds)


def test_budget_targets_a_total_not_an_increment() -> None:
    """Regression: a resumed study inflated the budget of one fold without touching the others."""
    import optuna

    from insectpose.tuning.objective import completed_trials, remaining_trials

    study = optuna.create_study()
    assert remaining_trials(study, 40) == 40

    for value in range(12):
        study.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=float(value)))
    assert completed_trials(study) == 12
    assert remaining_trials(study, 40) == 28
    assert remaining_trials(study, 10) == 0      # budget already exceeded: nothing to add


@pytest.mark.smoke
def test_relaunching_tune_does_not_inflate_the_budget(cfg, project) -> None:
    """Relaunching `tune` completes the budget instead of adding to it (§6.3)."""
    import optuna

    from insectpose.tuning.objective import completed_trials

    OmegaConf.update(cfg, "tuning.n_trials", 2)
    OmegaConf.update(cfg, "tuning.inner_folds", 2)
    OmegaConf.update(cfg, "tuning.mode", "tune_once")
    pipeline.cmd_split(cfg)
    pipeline.cmd_tune(cfg)
    pipeline.cmd_tune(cfg)          # second call: must add nothing

    db = next((project.runs / "optuna").glob("*.db"))
    storage = f"sqlite:///{db}"
    name = optuna.get_all_study_names(storage)[0]
    assert completed_trials(optuna.load_study(study_name=name, storage=storage)) == 2


# ===========================================================================
# Frozen HPO protocol (ADR-0031, ADR-0033)
# ===========================================================================
@pytest.mark.parametrize("approach", ["yolo_pooled", "yolo_per_dataset", "detect_then_pose",
                                      "lora", "group_bn", "yolo_pooled_reduced"])
def test_all_approaches_share_the_same_hpo_budget(config_factory, approach) -> None:
    """§6.3: comparing approaches with different budgets would measure the budget."""
    cfg = config_factory([f"approach={approach}"])
    assert int(cfg.tuning.n_trials) == 20
    assert int(cfg.tuning.n_startup_trials) == 5
    assert int(cfg.tuning.inner_folds) == 3
    assert str(cfg.tuning.mode) == "tune_once"
    assert int(cfg.tuning.pruner_warmup_steps) == 1
    # Four dimensions: beyond that, 20 trials are not enough to explore the space
    assert len(dict(cfg.approach.search_space)) == 4


def test_default_epoch_budget_is_frozen() -> None:
    """The training duration is part of the protocol (ADR-0031).

    Read from the shipped YAML, because the test fixture deliberately reduces it.
    """
    from pathlib import Path

    import yaml

    # Path derived from THIS file: a `tests` package installed in the environment would
    # shadow an absolute `tests.conftest` import.
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / "configs" / "config.yaml").read_text())
    assert int(config["train"]["epochs"]) == 100


@pytest.mark.parametrize("approach", ["yolo_pooled", "yolo_per_dataset", "lora",
                                      "group_bn", "yolo_pooled_reduced"])
def test_base_weights_are_frozen_to_the_same_model(config_factory, approach) -> None:
    """ADR-0033: a different model size would make the comparison be about it."""
    cfg = config_factory([f"approach={approach}"])
    assert str(cfg.approach.weights) == "yolo26n-pose.pt"


def test_detect_then_pose_uses_the_same_base_family(config_factory) -> None:
    cfg = config_factory(["approach=detect_then_pose"])
    assert str(cfg.approach.detector.weights).startswith("yolo26n")
    assert str(cfg.approach.pose.weights).startswith("yolo26n")


def test_augmentation_is_not_searched(config_factory) -> None:
    """ADR-0032: the augmentation is fixed, not searched."""
    cfg = config_factory(["approach=yolo_pooled"])
    searched = set(dict(cfg.approach.search_space))
    for fixed in ("degrees", "scale", "mosaic", "fliplr", "translate", "lrf"):
        assert fixed not in searched


def test_lora_alpha_is_derived_from_rank(config_factory) -> None:
    """ADR-0031: searching alpha on top of r would explore a redundancy."""
    from omegaconf import OmegaConf

    from insectpose.approaches.lora import LoraApproach

    cfg = config_factory(["approach=lora"])
    assert "lora.alpha" not in dict(cfg.approach.search_space)

    approach = LoraApproach(cfg)
    OmegaConf.update(cfg, "approach.lora.r", 16)
    assert approach._alpha() == pytest.approx(32.0)      # 2 x r
    OmegaConf.update(cfg, "approach.lora.alpha", 8.0)
    assert approach._alpha() == pytest.approx(8.0)       # explicit value takes precedence


# ===========================================================================
# Protocol hash: prevents mixing two search spaces
# ===========================================================================
def test_changing_the_search_space_creates_a_new_study(config_factory) -> None:
    """Regression: resuming after a modification mixed two protocols."""
    from omegaconf import OmegaConf

    from insectpose.tuning.objective import study_name_for

    cfg = config_factory(["approach=yolo_pooled"])
    OmegaConf.update(cfg, "split_id", "s1", force_add=True)
    before = study_name_for(cfg, "outer0")

    OmegaConf.update(cfg, "approach.search_space.lr0.high", 0.5)
    assert study_name_for(cfg, "outer0") != before


@pytest.mark.parametrize("key,value", [("tuning.n_trials", 40), ("tuning.inner_folds", 2),
                                       ("tuning.mode", "nested"), ("train.epochs", 500)])
def test_changing_the_budget_creates_a_new_study(config_factory, key, value) -> None:
    """The budget and the duration are part of the protocol: changing them isolates the study."""
    from omegaconf import OmegaConf

    from insectpose.tuning.objective import study_name_for

    cfg = config_factory(["approach=yolo_pooled"])
    OmegaConf.update(cfg, "split_id", "s1", force_add=True)
    before = study_name_for(cfg, "outer0")
    OmegaConf.update(cfg, key, value)
    assert study_name_for(cfg, "outer0") != before


def test_study_name_is_stable_for_an_unchanged_protocol(config_factory) -> None:
    from omegaconf import OmegaConf

    from insectpose.tuning.objective import study_name_for

    cfg = config_factory(["approach=yolo_pooled"])
    OmegaConf.update(cfg, "split_id", "s1", force_add=True)
    assert study_name_for(cfg, "outer0") == study_name_for(cfg.copy(), "outer0")
    assert "__sp" in study_name_for(cfg, "outer0")