"""Double d'Ultralytics partage par les tests des approches YOLO.

Ultralytics et CUDA ne sont pas installables partout (CI legere, machine de dev). Ce
double permet de verifier ce que NOUS controlons — arguments passes a l'entrainement,
conversion des coordonnees, routage entre modeles — sans GPU ni poids a telecharger.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


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
        assert kwargs.get("stream") is True, (
            "predict() doit etre streame : sinon Ultralytics garde un Results par image "
            "(image d'origine comprise) et le processus se fait tuer par l'OOM killer."
        )
        assert "half" not in kwargs, "'half' est deprecie : utiliser 'quantize'."
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


# --- A SUBSTITUER a la fixture `fake_ultralytics` existante dans
#     tests/approaches/conftest.py -------------------------------------------

class FakePoseTrainer:
    """Trainer minimal : les approches a patch en derivent via make_patched_trainer.

    Sans lui, `pose_trainer_class()` tenterait d'importer le vrai
    `ultralytics.models.yolo.pose` et echouerait sur un module double plat.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = types.SimpleNamespace(**kwargs)
        self.model: Any = None

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
        return self.model

    def get_validator(self) -> Any:
        return types.SimpleNamespace(preprocess=lambda batch: batch)

    def preprocess_batch(self, batch: Any) -> Any:
        return batch

    def _setup_train(self) -> None:
        return None

    def _model_train(self) -> None:
        return None

    def _build_train_pipeline(self) -> None:
        return None

    def final_eval(self) -> None:
        return None


@pytest.fixture()
def fake_ultralytics(monkeypatch) -> type[FakeYOLO]:
    """Injecte un faux `ultralytics`, sous-modules compris.

    Le double doit exposer `ultralytics.models.yolo.pose.PoseTrainer` : les approches
    a patch (head_only, lora, group_bn) en derivent leur trainer personnalise.
    """
    FakeYOLO.calls = []

    root = types.ModuleType("ultralytics")
    root.YOLO = FakeYOLO  # type: ignore[attr-defined]

    cfg = types.ModuleType("ultralytics.cfg")
    cfg.DEFAULT_CFG_DICT = {"quantize": None}  # type: ignore[attr-defined]

    modules = {"ultralytics": root, "ultralytics.cfg": cfg}
    for name in ("ultralytics.models", "ultralytics.models.yolo",
                 "ultralytics.models.yolo.pose"):
        modules[name] = types.ModuleType(name)
    modules["ultralytics.models.yolo.pose"].PoseTrainer = FakePoseTrainer  # type: ignore[attr-defined]
    modules["ultralytics.models.yolo"].pose = modules["ultralytics.models.yolo.pose"]  # type: ignore[attr-defined]
    modules["ultralytics.models"].yolo = modules["ultralytics.models.yolo"]  # type: ignore[attr-defined]
    root.models = modules["ultralytics.models"]  # type: ignore[attr-defined]

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeYOLO