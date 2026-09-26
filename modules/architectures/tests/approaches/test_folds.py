"""Tests of the fold selection of `train` and `tune` (ADR-0039).

`folds` says which outer folds a command runs. It must neither change the identity of a
run (each fold stays a run of its own) nor let an unknown fold train nothing silently,
and the folds of one `train` must form ONE ensemble for the pipeline.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.context import make_run_id, variant_hash
from insectpose.utils.io import read_json


@pytest.mark.parametrize(
    ("folds", "expected"),
    [(None, [1]), ([2, 0, 2], [0, 2]), ("all", [0, 1, 2]), (2, [2])],
)
def test_train_fold_selection(cfg, folds, expected) -> None:
    """Without `folds`, `train` keeps its historical behaviour: `fold` alone."""
    OmegaConf.update(cfg, "fold", 1)
    OmegaConf.update(cfg, "folds", folds)
    assert pipeline.selected_folds(cfg, default_all=False) == expected


def test_tune_runs_every_outer_fold_by_default(cfg) -> None:
    assert pipeline.selected_folds(cfg, default_all=True) == [0, 1, 2]   # cv.n_folds=3 here


@pytest.mark.parametrize("folds", [[3], [-1], [], "some"])
def test_invalid_folds_are_refused(cfg, folds) -> None:
    """An unknown fold would train nothing without saying so."""
    OmegaConf.update(cfg, "folds", folds)
    with pytest.raises(ValueError):
        pipeline.selected_folds(cfg, default_all=False)


def test_folds_do_not_change_the_identity_of_a_run(cfg) -> None:
    """Otherwise `train folds=[0,1]` would retrain a fold already trained alone."""
    before = make_run_id(cfg, "content"), variant_hash(cfg)
    OmegaConf.update(cfg, "folds", [0, 1])
    assert (make_run_id(cfg, "content"), variant_hash(cfg)) == before


@pytest.mark.smoke
def test_several_folds_form_one_ensemble(fake_ultralytics, config_factory, tmp_path) -> None:  # noqa: ARG001
    cfg = config_factory(["approach=yolo_pooled", "train.device=cpu"])
    OmegaConf.update(cfg, "mode", "full")   # a smoke run is never exported
    OmegaConf.update(cfg, "paths.retained", str(tmp_path / "retained"))
    OmegaConf.update(cfg, "folds", [0, 1])
    pipeline.cmd_split(cfg)
    contexts = pipeline.cmd_train_folds(cfg)

    pose = tmp_path / "retained" / "pose"
    members = sorted(p.name for p in pose.iterdir() if p.is_dir())
    assert members == sorted(ctx.run_id for ctx in contexts)
    card = read_json(pose / "ensemble.json")
    assert card["source"] == "train"
    assert card["n_members"] == 2
    assert card["folds"] == [0, 1]
    assert card["cv_estimate"]["n_folds"] == 2
