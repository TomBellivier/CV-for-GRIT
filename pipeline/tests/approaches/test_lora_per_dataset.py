"""Tests de l'approche H (lora_per_dataset).

La logique risquee est la repartition du budget d'epoques et le gel de la phase 2 :
toutes deux sont ecrites en fonctions pures et testees ici, sans torch.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.registry import APPROACHES
from insectpose.training.patching import freeze_patterns_for
from insectpose.utils.io import read_json, read_parquet

DATASETS = ("coleoptera", "diptera")


@pytest.fixture()
def cfg_h(config_factory):
    return config_factory(["approach=lora_per_dataset", "train.device=cpu"])


def test_approach_is_registered() -> None:
    assert "lora_per_dataset" in APPROACHES.available()


# --- repartition du budget d'epoques ----------------------------------------
@pytest.mark.parametrize(
    ("total", "split", "attendu"),
    [(100, 0.6, (60, 40)), (100, 0.3, (30, 70)), (10, 0.5, (5, 5)), (3, 0.6, (2, 1))],
)
def test_epoch_budget_is_split_not_added(config_factory, total, split, attendu) -> None:
    """§6.3 : le budget est REPARTI. Sinon H aurait plus de calcul que les autres."""
    from insectpose.approaches.lora_per_dataset import LoraPerDatasetApproach

    cfg = config_factory(["approach=lora_per_dataset"])
    OmegaConf.update(cfg, "train.epochs", total)
    OmegaConf.update(cfg, "approach.epoch_split", split)
    stage1, stage2 = LoraPerDatasetApproach(cfg)._epoch_budget()
    assert (stage1, stage2) == attendu
    assert stage1 + stage2 == total or total < 3   # arrondi sur les tres petits budgets


@pytest.mark.parametrize("split", [0.0, 1.0, -0.2, 1.5])
def test_invalid_epoch_split_is_refused(config_factory, split) -> None:
    from insectpose.approaches.lora_per_dataset import LoraPerDatasetApproach

    cfg = config_factory(["approach=lora_per_dataset"])
    OmegaConf.update(cfg, "approach.epoch_split", split)
    with pytest.raises(ValueError, match="epoch_split"):
        LoraPerDatasetApproach(cfg)._epoch_budget()


# --- gel de la phase 2 -------------------------------------------------------
def test_phase_two_freezes_everything_but_adapters() -> None:
    """Les tetes doivent etre GELEES en phase 2 : sinon chaque groupe aurait un modele
    presque entier, et H basculerait dans la categorie de B."""
    parameters = [
        "model.0.conv.weight",
        "model.20.conv.base_layer.weight",
        "model.20.conv.lora_A.default.weight",
        "model.20.conv.lora_B.default.weight",
        "model.23.one2one_cv4.sigma.2.weight",     # tete
    ]
    frozen = freeze_patterns_for(parameters, [r"lora_[AB]"])
    assert "model.23.one2one_cv4.sigma.2.weight" in frozen   # tete gelee
    assert "model.0.conv.weight" in frozen
    assert not any("lora_" in n for n in frozen)
    assert len(frozen) == 3


# --- protocole ---------------------------------------------------------------
def test_same_base_weights_as_other_approaches(config_factory) -> None:
    socle = config_factory(["approach=yolo_pooled"])
    h = config_factory(["approach=lora_per_dataset"])
    assert str(h.approach.weights) == str(socle.approach.weights)


def test_search_space_has_four_dimensions(cfg_h) -> None:
    """ADR-0031 : budget identique aux autres approches."""
    space = dict(cfg_h.approach.search_space)
    assert len(space) == 4
    assert "epoch_split" in space          # l'arbitrage tronc/adaptateurs est cherche
    assert "lora.alpha" not in space       # alpha est lie au rang


@pytest.mark.smoke
def test_adapters_are_injected_in_phase_two_only(
    fake_ultralytics, cfg_h, project  # noqa: ARG001
) -> None:
    """Le tronc sauvegarde a ses adaptateurs FUSIONNES : la phase 2 doit en injecter
    de nouveaux, pas esperer les retrouver (ADR-0036)."""
    from insectpose.approaches.lora_per_dataset import LoraPerDatasetApproach

    approach = LoraPerDatasetApproach(cfg_h)

    class _Sans:
        def named_parameters(self):
            return [("model.0.conv.weight", object())]

    with pytest.raises(RuntimeError, match="Aucune couche LoRA"):
        approach._freeze_all_but_adapters(_Sans())


@pytest.mark.smoke
def test_two_phases_produce_one_model_per_group(
    fake_ultralytics, cfg_h, project  # noqa: ARG001
) -> None:
    pipeline.cmd_split(cfg_h)
    ctx = pipeline.cmd_train(cfg_h)

    # Un tronc commun, puis un jeu d'adaptateurs par groupe
    assert (project.run_dir(ctx.run_id) / "weights" / "trunk" / "best.pt").exists()
    for dataset in DATASETS:
        assert (project.run_dir(ctx.run_id) / "weights" / dataset / "best.pt").exists()

    trainings = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert len(trainings) == 1 + len(DATASETS)


@pytest.mark.smoke
def test_epoch_budget_is_recorded_and_split(
    fake_ultralytics, cfg_h, project  # noqa: ARG001
) -> None:
    OmegaConf.update(cfg_h, "train.epochs", 10)
    OmegaConf.update(cfg_h, "approach.epoch_split", 0.6)
    pipeline.cmd_split(cfg_h)
    ctx = pipeline.cmd_train(cfg_h)

    manifest = read_json(project.manifest(ctx.run_id))
    assert manifest["stage1_epochs"] == 6
    assert manifest["stage2_epochs_per_group"] == 4
    assert manifest["n_adapter_sets"] == len(DATASETS)

    trainings = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert trainings[0]["epochs"] == 6                     # tronc
    assert all(c["epochs"] == 4 for c in trainings[1:])    # adaptateurs


@pytest.mark.smoke
def test_predictions_are_routed_by_dataset(
    fake_ultralytics, cfg_h, project  # noqa: ARG001
) -> None:
    from insectpose.data.datamodule import load_annotations

    pipeline.cmd_split(cfg_h)
    ctx = pipeline.cmd_train(cfg_h)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    expected = load_annotations(list(DATASETS), project).set_index("image_id")["dataset"]
    assert (predictions["dataset"].to_numpy()
            == predictions["image_id"].map(expected).to_numpy()).all()
    assert len(predictions["kpts_xy"].iloc[0]) == 84