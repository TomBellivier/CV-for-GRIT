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

    scores = {r: [e["oks"] for e in index["examples"] if e["reason"] == r]
              for r in ("worst", "best", "random")}
    # Les pires cas sont bien les moins bons, les meilleurs bien les meilleurs
    assert not scores["random"] or max(scores["worst"]) <= min(scores["random"])
    assert not scores["random"] or min(scores["best"]) >= max(scores["random"])

    # Un meilleur cas par dataset (§8.5) : le meilleur global viendrait toujours du
    # dataset le plus facile.
    best_datasets = [e["dataset"] for e in index["examples"] if e["reason"] == "best"]
    assert len(best_datasets) == len(set(best_datasets))
    assert set(best_datasets) == set(
        e["dataset"] for e in index["examples"]) & set(best_datasets)


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


@pytest.mark.smoke
def test_best_examples_are_one_per_dataset() -> None:
    """Les meilleurs cas sont choisis PAR dataset, pas globalement."""
    from insectpose.reporting.qualitative import select_examples

    scores = pd.DataFrame({
        "image_id": [f"i{i}" for i in range(8)],
        "dataset": ["coleoptera"] * 4 + ["diptera"] * 4,
        "gt_row": range(8), "pred_row": range(8),
        # Coleoptera est globalement meilleur : sans selection par dataset, les deux
        # "best" viendraient de lui et diptera n'aurait aucune reference.
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
    """Zero prediction est un RESULTAT mesurable, pas une erreur de pipeline.

    Un modele sous-entraine ou mal regle ne detecte rien. L'evaluation doit alors
    publier des metriques nulles : interrompre le pipeline laisserait croire a un run
    casse alors que le modele est simplement mauvais.
    """
    from insectpose.registry import APPROACHES
    from insectpose.utils.io import read_parquet

    # Patcher la classe CONCRETE : chaque approche surcharge `predict_instances`,
    # donc patcher BaseApproach serait sans effet.
    approach_cls = APPROACHES.get(str(cfg.approach.name))
    monkeypatch.setattr(
        approach_cls, "predict_instances",
        lambda _self, _images, _ctx: pd.DataFrame(), raising=False)

    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)

    # Le run reste COMPLET : manifeste ecrit, donc agregeable et auditable
    assert project.manifest(ctx.run_id).exists()

    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    assert predictions.empty

    metrics = read_parquet(project.metrics(ctx.run_id))
    primary = metrics[(metrics["metric"] == str(cfg.eval.primary_metric))
                      & (metrics["scope"] == "overall") & (metrics["split"] == "test")]
    assert len(primary) == 1
    assert primary["value"].iloc[0] == 0.0
    assert primary["n"].iloc[0] > 0        # le denominateur reste celui des GT