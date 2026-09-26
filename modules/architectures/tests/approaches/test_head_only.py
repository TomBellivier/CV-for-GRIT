"""Tests of approach G (head_only), control of approach D.

As for D and E, the torch patch cannot be exercised without a GPU. The decision logic
— which blocks stay trainable — is written as pure functions and tested here.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.registry import APPROACHES
from insectpose.training.patching import freeze_patterns_for, head_index
from insectpose.utils.io import read_json

# Typical structure of a YOLO-pose: Sequential of blocks, the head last.
PARAMETERS = [
    "model.0.conv.weight",
    "model.9.conv.weight",
    "model.20.conv.weight",
    "model.22.cv1.conv.weight",
    "model.23.cv4.0.0.conv.weight",
    "model.23.one2one_cv4.sigma.2.weight",
]
MODULES = [n.rsplit(".", 1)[0] for n in PARAMETERS]


def test_approach_is_registered() -> None:
    assert "head_only" in APPROACHES.available()


def test_only_the_last_block_stays_trainable() -> None:
    """blocks=1: the exact control of D with train_head=true."""
    last = head_index(MODULES)
    assert last == 23
    frozen = freeze_patterns_for(PARAMETERS, [rf"^model\.({last})\."])
    assert all(not n.startswith("model.23.") for n in frozen)
    assert "model.0.conv.weight" in frozen
    assert "model.22.cv1.conv.weight" in frozen
    assert len(frozen) == 4


def test_more_blocks_unfreeze_the_end_of_the_neck() -> None:
    last = head_index(MODULES)
    blocks = "|".join(str(i) for i in range(last - 3 + 1, last + 1))   # 21, 22, 23
    frozen = freeze_patterns_for(PARAMETERS, [rf"^model\.({blocks})\."])
    assert "model.22.cv1.conv.weight" not in frozen
    assert "model.20.conv.weight" in frozen


def test_no_adapter_is_injected(config_factory) -> None:
    """G has no adapter: that is what makes it the control of D."""
    cfg = config_factory(["approach=head_only"])
    assert "lora" not in cfg.approach
    assert int(cfg.approach.head.blocks) >= 1


def test_search_space_matches_the_baseline(config_factory) -> None:
    """Same hyperparameters as A/B/E/F: the comparison with D must isolate the adapters."""
    baseline = config_factory(["approach=yolo_pooled"])
    head = config_factory(["approach=head_only"])
    assert set(dict(head.approach.search_space)) == set(dict(baseline.approach.search_space))
    assert str(head.approach.weights) == str(baseline.approach.weights)


@pytest.mark.smoke
def test_trainable_ratio_is_recorded(fake_ultralytics, config_factory, project) -> None:  # noqa: ARG001
    """The trainable ratio is THE measure that distinguishes G from A: it must be in the manifest."""
    cfg = config_factory(["approach=head_only", "train.device=cpu"])
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    manifest = read_json(project.manifest(ctx.run_id))
    assert manifest["head_blocks"] == 1
    assert "head_patterns" in manifest


@pytest.mark.smoke
def test_predictions_follow_the_contract(fake_ultralytics, config_factory, project) -> None:  # noqa: ARG001
    from insectpose.utils.io import read_parquet

    cfg = config_factory(["approach=head_only", "train.device=cpu"])
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    assert len(predictions["kpts_xy"].iloc[0]) == 84
    assert predictions["bbox_source"].unique().tolist() == ["predicted"]