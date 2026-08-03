"""Smoke test de bout en bout (§10.3).

Une approche qui ne passe pas ce test n'est pas consideree comme implementee.
Boucle sur TOUTES les approches enregistrees : ajouter une approche l'inclut
automatiquement, sans toucher a ce fichier.
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
        pytest.skip(f"{approach_name} indisponible dans cet environnement : {reason}")
    # Recomposition du groupe Hydra : patcher `approach.name` laisserait les cles de
    # l'approche precedente (defaut reel corrige apres coup).
    cfg = config_factory([f"approach={approach_name}"])
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)

    # Artefacts obligatoires du run (§8.2)
    assert project.manifest(ctx.run_id).exists(), "manifeste manquant : run incomplet"
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
    second = pipeline.cmd_train(cfg)   # doit etre saute, pas rejoue
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
        assert field in manifest, f"champ '{field}' absent du manifeste"


@pytest.mark.smoke
def test_fit_never_sees_test_data(cfg, project, monkeypatch) -> None:  # noqa: ARG001
    """Garde-fou anti-fuite : `fit` ne doit jamais lire data.test (§4.2)."""
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
    """Chaque run exporte des figures pred vs GT, dont les pires cas (§8.4)."""
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    figures = sorted((project.run_dir(ctx.run_id) / "figures").glob("*.png"))
    assert len(figures) == int(cfg.eval.qualitative.n_examples)
    index = read_json(project.run_dir(ctx.run_id) / "figures" / "qualitative_index.json")
    reasons = [e["reason"] for e in index["examples"]]
    assert reasons.count("worst") == int(cfg.eval.qualitative.n_worst)
    # Les pires cas sont bien les moins bons OKS
    worst = [e["oks"] for e in index["examples"] if e["reason"] == "worst"]
    others = [e["oks"] for e in index["examples"] if e["reason"] == "random"]
    assert not others or max(worst) <= min(others)


@pytest.mark.smoke
def test_missing_images_are_refused_by_default(cfg, project) -> None:
    """Un export qualitatif silencieusement vide masquerait un chemin d'image casse."""
    import pytest as _pytest

    pipeline.cmd_split(cfg)
    for image in (project.raw / "coleoptera" / "images").glob("*.png"):
        image.unlink()
    with _pytest.raises(FileNotFoundError, match="export qualitatif"):
        pipeline.cmd_train(cfg)


@pytest.mark.parametrize("approach_name", sorted(APPROACHES.available()))
def test_approach_config_matches_its_name(config_factory, approach_name) -> None:
    """Garde-fou : le groupe de config charge doit etre CELUI de l'approche.

    Sans cette verification, un test qui patche `approach.name` sans recomposer le
    groupe Hydra laisse les cles de l'approche precedente et echoue plus loin, sur un
    'Missing key' incomprehensible.
    """
    cfg = config_factory([f"approach={approach_name}"])
    assert str(cfg.approach.name) == approach_name
    assert str(cfg.approach._target_).rsplit(".", 1)[0].endswith(
        APPROACHES.get(approach_name).__module__.rsplit(".", 1)[-1]
    )