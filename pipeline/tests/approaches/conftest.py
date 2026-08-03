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




@pytest.fixture()
def fake_ultralytics(monkeypatch) -> type[FakeYOLO]:
    """Injecte le faux module `ultralytics` pour la duree du test."""
    FakeYOLO.calls = []
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return FakeYOLO