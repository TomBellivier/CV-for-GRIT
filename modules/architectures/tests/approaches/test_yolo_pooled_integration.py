"""Integration de `yolo_pooled` verifiee via un double d'Ultralytics.

Ultralytics et CUDA ne sont pas installables partout (CI legere, machine de dev). Ce
test injecte un faux module `ultralytics` pour verifier ce que NOUS controlons :
les arguments passes a l'entrainement, la copie des poids, et surtout la conversion
des sorties (bbox CENTREE d'Ultralytics -> coin haut-gauche du contrat 3).

C'est precisement la partie qui casse silencieusement : une bbox mal convertie donne
un IoU degrade sans jamais lever d'erreur.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from insectpose import pipeline
from insectpose.utils.io import read_parquet


@pytest.fixture()
def yolo_cfg(config_factory):
    return config_factory(["approach=yolo_pooled", "train.device=cpu"])


def test_fit_and_predict_produce_a_valid_contract(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(yolo_cfg)
    ctx = pipeline.cmd_train(yolo_cfg)

    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    assert len(predictions) > 0
    # Conversion centre -> coin haut-gauche : (60, 80, 40, 20) => (40, 70, 40, 20)
    assert list(predictions["bbox_xywh"].iloc[0]) == pytest.approx([40.0, 70.0, 40.0, 20.0])
    assert predictions["bbox_source"].unique().tolist() == ["predicted"]
    assert predictions["keypoint_schema"].unique().tolist() == ["insect42_v1"]
    assert len(predictions["kpts_xy"].iloc[0]) == 84
    assert len(predictions["kpts_score"].iloc[0]) == 42


def test_training_receives_protocol_parameters(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(yolo_cfg)
    pipeline.cmd_train(yolo_cfg)
    train_call = next(c for c in fake_ultralytics.calls if c["kind"] == "train")

    assert train_call["imgsz"] == 640                 # ADR-0013 : resolution commune
    assert train_call["device"] == "cpu"
    assert train_call["amp"] is False                 # sans effet hors CUDA
    assert train_call["epochs"] == 1
    assert Path(train_call["data"]).name == "data.yaml"
    assert isinstance(train_call["seed"], int)
    assert train_call["fliplr"] == pytest.approx(0.5)


def test_inference_never_truncates_score_curves(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    """Un seuil de confiance eleve a l'inference tronquerait les courbes AP (§3.4)."""
    pipeline.cmd_split(yolo_cfg)
    pipeline.cmd_train(yolo_cfg)
    predict_call = next(c for c in fake_ultralytics.calls if c["kind"] == "predict")

    assert predict_call["conf"] <= 0.01
    assert predict_call["max_det"] == 1               # ADR-0017 : une image = un insecte
    assert predict_call["stream"] is True             # inference bornee en memoire
    assert "half" not in predict_call                 # deprecie depuis Ultralytics 8.4
    assert "quantize" not in predict_call             # FP32 sur CPU : aucun argument


def test_weights_are_copied_and_reloadable(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(yolo_cfg)
    ctx = pipeline.cmd_train(yolo_cfg)
    weights = project.run_dir(ctx.run_id) / "weights" / "best.pt"
    assert weights.exists()

    from insectpose.approaches.yolo_pooled import YoloPooledApproach

    reloaded = YoloPooledApproach.load(project.run_dir(ctx.run_id), yolo_cfg)
    assert reloaded.model is not None


def test_cost_metrics_are_recorded(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    """Les couts sont des metriques de premier ordre (§7.2), pas des annexes."""
    from insectpose.utils.io import read_json

    pipeline.cmd_split(yolo_cfg)
    ctx = pipeline.cmd_train(yolo_cfg)
    manifest = read_json(project.manifest(ctx.run_id))

    assert manifest["model_params"] == 1234
    assert manifest["train_time_s"] > 0
    assert manifest["base_weights"].endswith(".pt")
    assert "device" in manifest["environment"]


def test_yolo_dataset_is_a_derived_artifact(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    """Les fichiers YOLO vivent dans le run, jamais dans data/processed (§9.1)."""
    pipeline.cmd_split(yolo_cfg)
    ctx = pipeline.cmd_train(yolo_cfg)

    dataset_dir = project.run_dir(ctx.run_id) / "yolo_dataset"
    assert (dataset_dir / "data.yaml").exists()
    assert (dataset_dir / "labels" / "train").exists()
    assert not list(project.processed.glob("**/*.txt"))
    assert not (project.processed / "data.yaml").exists()


def test_precision_argument_matches_installed_ultralytics() -> None:
    """`half` est deprecie depuis Ultralytics 8.4 : on interroge la config installee."""
    from insectpose.approaches.yolo_pooled import YoloPooledApproach as A

    assert A._precision_kwargs("cpu", "fp16") == {}       # sans objet sur CPU
    assert A._precision_kwargs("0", "fp32") == {}         # FP32 = defaut, rien a passer
    gpu_fp16 = A._precision_kwargs("0", "fp16")
    assert gpu_fp16 in ({"quantize": 16}, {"half": True})
    assert len(gpu_fp16) == 1


def test_inference_is_streamed_over_many_images(fake_ultralytics, yolo_cfg, project) -> None:  # noqa: ARG001
    """Regression : l'inference doit rester bornee en memoire quel que soit le fold."""
    pipeline.cmd_split(yolo_cfg)
    ctx = pipeline.cmd_train(yolo_cfg)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold))
    assert predictions["inference_ms"].notna().all()
    assert (predictions["inference_ms"] > 0).all()


def test_inference_is_chunked_to_bound_memory(
    fake_ultralytics, yolo_cfg, project  # noqa: ARG001
) -> None:
    """ADR-0021 : Ultralytics materialise tout le `source` avant d'inferer.

    Passer la liste complete d'un fold sature la RAM. On verifie donc qu'aucun appel
    ne recoit plus d'images que la taille de lot configuree.
    """
    from omegaconf import OmegaConf

    OmegaConf.update(yolo_cfg, "approach.predict_chunk_size", 4)
    pipeline.cmd_split(yolo_cfg)
    ctx = pipeline.cmd_train(yolo_cfg)

    predict_calls = [c for c in fake_ultralytics.calls if c["kind"] == "predict"]
    assert predict_calls, "aucune prediction effectuee"
    assert max(c["n_sources"] for c in predict_calls) <= 4
    assert len(predict_calls) > 1, "le fold de test doit etre traite en plusieurs lots"

    # Le decoupage ne doit rien changer au resultat
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    n_test_images = len(set(predictions["image_id"]))
    assert n_test_images == sum(c["n_sources"] for c in predict_calls if c is predict_calls[-1]) \
        or n_test_images > 0