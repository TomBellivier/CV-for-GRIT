"""End-to-end smoke test (§10.3).

An approach that does not pass this test is not considered implemented. Loops over ALL
the registered approaches: adding an approach includes it automatically, without
touching this file.
"""

from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.evaluation.aggregate import summary_table, write_master
from insectpose.evaluation.evaluator import primary_value
from insectpose.registry import APPROACHES
from insectpose.utils.io import read_json, read_parquet


@pytest.mark.smoke
@pytest.mark.parametrize("approach_name", sorted(APPROACHES.available()))
def test_full_pipeline_per_approach(config_factory, project, approach_name) -> None:
    available, reason = APPROACHES.get(approach_name).availability()
    if not available:
        pytest.skip(f"{approach_name} unavailable in this environment: {reason}")
    # Recomposition of the Hydra group: patching `approach.name` would leave the keys of
    # the previous approach (real defect fixed afterwards).
    cfg = config_factory([f"approach={approach_name}"])
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)

    # Mandatory artefacts of the run (§8.2)
    assert project.manifest(ctx.run_id).exists(), "missing manifest: incomplete run"
    assert project.metrics(ctx.run_id).exists()
    assert (project.run_dir(ctx.run_id) / "config.yaml").exists()
    assert project.predictions(ctx.run_id, "test", ctx.fold).exists()

    metrics = read_parquet(project.metrics(ctx.run_id))
    assert primary_value(metrics, cfg.eval) >= 0.0
    assert {"overall"}.issubset(set(metrics["scope"]))


@pytest.mark.smoke
def test_run_id_is_deterministic_and_idempotent(cfg, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(cfg)
    first = pipeline.cmd_train(cfg)
    second = pipeline.cmd_train(cfg)   # must be skipped, not replayed
    assert first.run_id == second.run_id


@pytest.mark.smoke
def test_reevaluation_without_retraining(cfg, project) -> None:
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    before = read_parquet(project.metrics(ctx.run_id))
    pipeline.cmd_evaluate(cfg, ctx.run_id)
    after = read_parquet(project.metrics(ctx.run_id))
    pd.testing.assert_frame_equal(
        before.sort_values(["scope", "metric"]).reset_index(drop=True),
        after.sort_values(["scope", "metric"]).reset_index(drop=True),
    )


@pytest.mark.smoke
def test_aggregation_over_multiple_folds(cfg, project) -> None:
    pipeline.cmd_split(cfg)
    for fold in range(2):
        OmegaConf.update(cfg, "fold", fold)
        pipeline.cmd_train(cfg)
    master = read_parquet(write_master(project))
    summary = summary_table(master, str(cfg.eval.primary_metric))
    assert summary["n_folds"].max() >= 2
    assert set(master["scope"]).issuperset({"overall"})


@pytest.mark.smoke
def test_manifest_records_reproducibility_fields(cfg, project) -> None:
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    manifest = read_json(project.manifest(ctx.run_id))
    for field in ("run_id", "approach", "split_id", "content_hash", "seed", "config",
                  "git", "environment", "eval_version", "primary_metric"):
        assert field in manifest, f"field '{field}' missing from the manifest"


@pytest.mark.smoke
def test_fit_never_sees_test_data(cfg, project, monkeypatch) -> None:  # noqa: ARG001
    """Anti-leakage safeguard: `fit` must never read data.test (§4.2)."""
    from insectpose.data.datamodule import ImageSet

    pipeline.cmd_split(cfg)
    seen: list[str] = []
    original = ImageSet.instances_array

    def spy(self):  # noqa: ANN001, ANN202
        seen.append(self.name)
        return original(self)

    monkeypatch.setattr(ImageSet, "instances_array", spy)
    ctx, data, approach = pipeline._prepare_run(cfg)
    ctx.setup()
    approach.fit(data, ctx)
    assert "test" not in seen


@pytest.mark.smoke
def test_qualitative_export_is_produced(cfg, project) -> None:
    """Each run exports pred vs GT figures, including the worst cases (§8.5)."""
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    figures = sorted((project.run_dir(ctx.run_id) / "figures").glob("*.png"))
    assert len(figures) == int(cfg.eval.qualitative.n_examples)
    index = read_json(project.run_dir(ctx.run_id) / "figures" / "qualitative_index.json")
    reasons = [e["reason"] for e in index["examples"]]
    assert reasons.count("worst") == int(cfg.eval.qualitative.n_worst)

    scores = {r: [e["oks"] for e in index["examples"] if e["reason"] == r]
              for r in ("worst", "best", "random")}
    # The worst cases are indeed the worst, the best ones indeed the best
    assert not scores["random"] or max(scores["worst"]) <= min(scores["random"])
    assert not scores["random"] or min(scores["best"]) >= max(scores["random"])

    # One best case per dataset (§8.5): the global best would always come from the
    # easiest dataset.
    best_datasets = [e["dataset"] for e in index["examples"] if e["reason"] == "best"]
    assert len(best_datasets) == len(set(best_datasets))
    assert set(best_datasets) == set(
        e["dataset"] for e in index["examples"]) & set(best_datasets)


@pytest.mark.smoke
def test_missing_images_are_refused_by_default(cfg, project) -> None:
    """A silently empty qualitative export would hide a broken image path."""
    import pytest as _pytest

    pipeline.cmd_split(cfg)
    for image in (project.raw / "coleoptera" / "images").glob("*.png"):
        image.unlink()
    with _pytest.raises(FileNotFoundError, match="qualitative export"):
        pipeline.cmd_train(cfg)


@pytest.mark.parametrize("approach_name", sorted(APPROACHES.available()))
def test_approach_config_matches_its_name(config_factory, approach_name) -> None:
    """Safeguard: the loaded config group must be THE ONE of the approach.

    Without this check, a test that patches `approach.name` without recomposing the Hydra
    group leaves the keys of the previous approach and fails further on, with an
    incomprehensible 'Missing key'.
    """
    cfg = config_factory([f"approach={approach_name}"])
    assert str(cfg.approach.name) == approach_name
    assert str(cfg.approach._target_).rsplit(".", 1)[0].endswith(
        APPROACHES.get(approach_name).__module__.rsplit(".", 1)[-1]
    )


@pytest.mark.smoke
def test_best_examples_are_one_per_dataset() -> None:
    """The best cases are chosen PER dataset, not globally."""
    from insectpose.reporting.qualitative import select_examples

    scores = pd.DataFrame({
        "image_id": [f"i{i}" for i in range(8)],
        "dataset": ["coleoptera"] * 4 + ["diptera"] * 4,
        "gt_row": range(8), "pred_row": range(8),
        # Coleoptera is better overall: without a per-dataset selection, both "best"
        # would come from it and diptera would have no reference.
        "oks": [0.90, 0.92, 0.94, 0.96, 0.30, 0.40, 0.50, 0.60],
    })
    selection = select_examples(scores, n_examples=6, n_worst=2, seed=0,
                                n_best_per_dataset=1)
    best = selection[selection["reason"] == "best"]
    assert set(best["dataset"]) == {"coleoptera", "diptera"}
    assert best.loc[best["dataset"] == "coleoptera", "oks"].iloc[0] == pytest.approx(0.96)
    assert best.loc[best["dataset"] == "diptera", "oks"].iloc[0] == pytest.approx(0.60)
    assert len(selection) == 6


@pytest.mark.smoke
def test_best_selection_can_be_disabled() -> None:
    from insectpose.reporting.qualitative import select_examples

    scores = pd.DataFrame({
        "image_id": [f"i{i}" for i in range(6)], "dataset": ["coleoptera"] * 6,
        "gt_row": range(6), "pred_row": range(6),
        "oks": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    selection = select_examples(scores, n_examples=4, n_worst=2, seed=0,
                                n_best_per_dataset=0)
    assert set(selection["reason"]) == {"worst", "random"}
    assert len(selection) == 4


@pytest.mark.smoke
def test_a_model_that_detects_nothing_yields_zero_metrics(cfg, project, monkeypatch) -> None:
    """Zero prediction is a measurable RESULT, not a pipeline error.

    An under-trained or badly tuned model detects nothing. The evaluation must then
    publish zero metrics: interrupting the pipeline would suggest a broken run while the
    model is simply bad.
    """
    from insectpose.registry import APPROACHES
    from insectpose.utils.io import read_parquet

    # Patch the CONCRETE class: each approach overrides `predict_instances`, so patching
    # BaseApproach would have no effect.
    approach_cls = APPROACHES.get(str(cfg.approach.name))
    monkeypatch.setattr(
        approach_cls, "predict_instances",
        lambda _self, _images, _ctx: pd.DataFrame(), raising=False)

    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)

    # The run stays COMPLETE: manifest written, hence aggregatable and auditable
    assert project.manifest(ctx.run_id).exists()

    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    assert predictions.empty

    metrics = read_parquet(project.metrics(ctx.run_id))
    primary = metrics[(metrics["metric"] == str(cfg.eval.primary_metric))
                      & (metrics["scope"] == "overall") & (metrics["split"] == "test")]
    assert len(primary) == 1
    assert primary["value"].iloc[0] == 0.0
    assert primary["n"].iloc[0] > 0        # the denominator stays that of the GT