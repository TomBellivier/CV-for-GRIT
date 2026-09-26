"""Tests of approach H (lora_per_dataset).

The risky logic is the split of the epoch budget and the freezing of phase 2: both are
written as pure functions and tested here, without torch.
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


# --- split of the epoch budget ----------------------------------------------
@pytest.mark.parametrize(
    ("total", "split", "expected"),
    [(100, 0.6, (60, 40)), (100, 0.3, (30, 70)), (10, 0.5, (5, 5)), (3, 0.6, (2, 1))],
)
def test_epoch_budget_is_split_not_added(config_factory, total, split, expected) -> None:
    """§6.3: the budget is SPLIT. Otherwise H would get more compute than the others."""
    from insectpose.approaches.lora_per_dataset import LoraPerDatasetApproach

    cfg = config_factory(["approach=lora_per_dataset"])
    OmegaConf.update(cfg, "train.epochs", total)
    OmegaConf.update(cfg, "approach.epoch_split", split)
    stage1, stage2 = LoraPerDatasetApproach(cfg)._epoch_budget()
    assert (stage1, stage2) == expected
    assert stage1 + stage2 == total or total < 3   # rounding on very small budgets


@pytest.mark.parametrize("split", [0.0, 1.0, -0.2, 1.5])
def test_invalid_epoch_split_is_refused(config_factory, split) -> None:
    from insectpose.approaches.lora_per_dataset import LoraPerDatasetApproach

    cfg = config_factory(["approach=lora_per_dataset"])
    OmegaConf.update(cfg, "approach.epoch_split", split)
    with pytest.raises(ValueError, match="epoch_split"):
        LoraPerDatasetApproach(cfg)._epoch_budget()


# --- phase 2 freezing --------------------------------------------------------
def test_phase_two_freezes_everything_but_adapters() -> None:
    """The heads must be FROZEN in phase 2: otherwise each group would have an almost
    complete model, and H would fall into the category of B."""
    parameters = [
        "model.0.conv.weight",
        "model.20.conv.base_layer.weight",
        "model.20.conv.lora_A.default.weight",
        "model.20.conv.lora_B.default.weight",
        "model.23.one2one_cv4.sigma.2.weight",     # head
    ]
    frozen = freeze_patterns_for(parameters, [r"lora_[AB]"])
    assert "model.23.one2one_cv4.sigma.2.weight" in frozen   # frozen head
    assert "model.0.conv.weight" in frozen
    assert not any("lora_" in n for n in frozen)
    assert len(frozen) == 3


# --- protocol ----------------------------------------------------------------
def test_same_base_weights_as_other_approaches(config_factory) -> None:
    baseline = config_factory(["approach=yolo_pooled"])
    h = config_factory(["approach=lora_per_dataset"])
    assert str(h.approach.weights) == str(baseline.approach.weights)


def test_search_space_has_four_dimensions(cfg_h) -> None:
    """ADR-0031: same budget as the other approaches."""
    space = dict(cfg_h.approach.search_space)
    assert len(space) == 4
    assert "epoch_split" in space          # the trunk/adapters trade-off is searched
    assert "lora.alpha" not in space       # alpha is tied to the rank


@pytest.mark.smoke
def test_adapters_are_injected_in_phase_two_only(
    fake_ultralytics, cfg_h, project  # noqa: ARG001
) -> None:
    """The saved trunk has its adapters MERGED: phase 2 must inject new ones, not hope
    to find them again (ADR-0036)."""
    from insectpose.approaches.lora_per_dataset import LoraPerDatasetApproach

    approach = LoraPerDatasetApproach(cfg_h)

    class _WithoutLora:
        def named_parameters(self):
            return [("model.0.conv.weight", object())]

    with pytest.raises(RuntimeError, match="No LoRA layer"):
        approach._freeze_all_but_adapters(_WithoutLora())


@pytest.mark.smoke
def test_two_phases_produce_one_model_per_group(
    fake_ultralytics, cfg_h, project  # noqa: ARG001
) -> None:
    pipeline.cmd_split(cfg_h)
    ctx = pipeline.cmd_train(cfg_h)

    # A shared trunk, then one adapter set per group
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
    assert trainings[0]["epochs"] == 6                     # trunk
    assert all(c["epochs"] == 4 for c in trainings[1:])    # adapters


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