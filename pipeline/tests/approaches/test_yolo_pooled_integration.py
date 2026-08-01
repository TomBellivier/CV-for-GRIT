"""Integration de `yolo_pooled` verifiee via un double d'Ultralytics.

Ultralytics et CUDA ne sont pas installables partout (CI legere, machine de dev). Ce
test injecte un faux module `ultralytics` pour verifier ce que NOUS controlons :
les arguments passes a l'entrainement, la copie des poids, et surtout la conversion
des sorties (bbox CENTREE d'Ultralytics -> coin haut-gauche du contrat 3).

C'est precisement la partie qui casse silencieusement : une bbox mal convertie donne
un IoU degrade sans jamais lever d'erreur.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.utils.io import read_parquet


class _Arr:
    """Minimal shim exposant l'interface `.cpu().numpy()` des tenseurs torch."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> _Arr:
        return self

    def numpy(self) -> np.ndarray:
        return self._values

    def __len__(self) -> int:
        return len(self._values)


class _Boxes:
    def __init__(self, xywh: np.ndarray, conf: np.ndarray) -> None:
        self.xywh = _Arr(xywh)
        self.conf = _Arr(conf)

    def __len__(self) -> int:
        return len(self.xywh)


class _Keypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = _Arr(data)


class _Result:
    def __init__(self, boxes: _Boxes, keypoints: _Keypoints) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


class _Param:
    def __init__(self, n: int) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n


class FakeYOLO:
    """Double d'Ultralytics : enregistre les appels, renvoie des sorties plausibles."""

    calls: list[dict[str, Any]] = []
    n_keypoints = 42

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.model = types.SimpleNamespace(parameters=lambda: [_Param(1000), _Param(234)])
        self.trainer: Any = None

    def train(self, **kwargs: Any) -> None:
        FakeYOLO.calls.append({"kind": "train", **kwargs})
        best = Path(kwargs["project"]) / "train" / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"fake-weights")
        self.trainer = types.SimpleNamespace(best=str(best))

    def predict(self, source: list[str], **kwargs: Any) -> list[_Result]:
        FakeYOLO.calls.append({"kind": "predict", "n_sources": len(source), **kwargs})
        results = []
        for i in range(len(source)):
            # bbox CENTREE, comme Ultralytics : centre (60, 80), taille 40 x 20
            boxes = _Boxes(np.array([[60.0, 80.0, 40.0, 20.0]]), np.array([0.9 - i * 0.001]))
            kpts = np.zeros((1, self.n_keypoints, 3))
            kpts[0, :, 0] = np.linspace(45, 75, self.n_keypoints)
            kpts[0, :, 1] = np.linspace(72, 88, self.n_keypoints)
            kpts[0, :, 2] = 0.8
            results.append(_Result(boxes, _Keypoints(kpts)))
        return results


@pytest.fixture()
def fake_ultralytics(monkeypatch) -> type[FakeYOLO]:
    FakeYOLO.calls = []
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return FakeYOLO


@pytest.fixture()
def yolo_cfg(cfg):
    OmegaConf.update(cfg, "approach.name", "yolo_pooled", force_add=True)
    from insectpose.cli import load_config

    yolo = load_config([
        f"paths.root={cfg.paths.root}", "approach=yolo_pooled", "data=pooled",
        "data.datasets=[coleoptera,diptera]", "cv=kfold5_grouped", "cv.n_folds=3",
        "mode=smoke", "tag=test", "train.epochs=1", "train.device=cpu",
    ])
    OmegaConf.set_struct(yolo, False)
    return yolo


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
    assert predict_call["half"] is False              # ignore sur CPU


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
