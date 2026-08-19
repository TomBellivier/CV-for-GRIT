"""Double d'Ultralytics partage par les tests des approches YOLO.

Ultralytics et CUDA ne sont pas installables partout (CI legere, machine de dev). Ce
double permet de verifier ce que NOUS controlons — arguments passes a l'entrainement,
conversion des coordonnees, routage entre modeles, gel des parametres — sans GPU ni
poids a telecharger.

Il expose aussi `ultralytics.models.yolo.pose` et `ultralytics.cfg` : les approches qui
passent un trainer personnalise (lora, group_bn, head_only) importent `PoseTrainer`, et
remplacer `sys.modules["ultralytics"]` par un module plat ferait echouer ces imports
avec "ultralytics.models is not a package".
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
    def __init__(self, n: int, name: str = "") -> None:
        self._n = n
        self.name = name
        self.requires_grad = True

    def numel(self) -> int:
        return self._n


class FakePoseTrainer:
    """Trainer minimal : les approches en derivent via `make_patched_trainer`.

    Il n'entraine rien, mais expose les points d'accroche que le patch surcharge, ce qui
    permet de verifier que la chaine se construit sans erreur.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = types.SimpleNamespace(freeze=None)
        self.model: Any = None
        self.validator: Any = None

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
        return self.model

    def _setup_train(self) -> None:
        return None

    def _model_train(self) -> None:
        return None

    def _build_train_pipeline(self) -> None:
        return None

    def preprocess_batch(self, batch: Any) -> Any:
        return batch

    def get_validator(self) -> Any:
        return self.validator

    def final_eval(self) -> None:
        return None


class _FakeCheckpointModule:
    """Module minimal picklable, imitant l'interface attendue d'un checkpoint YOLO."""

    def __init__(self) -> None:
        self.children: dict[str, Any] = {}

    def named_children(self) -> list[tuple[str, Any]]:
        return list(self.children.items())


def _write_fake_checkpoint(path: Path) -> None:
    """Ecrit un checkpoint lisible par torch.load, ou pickle en repli.

    Effet de bord : cree `path`.
    """
    payload = {"model": _FakeCheckpointModule(), "ema": None, "epoch": -1}
    try:
        import torch

        torch.save(payload, path)
    except ImportError:
        import pickle

        with path.open("wb") as handle:
            pickle.dump(payload, handle)


class FakeYOLO:
    """Double d'Ultralytics : enregistre les appels, renvoie des sorties plausibles."""

    calls: list[dict[str, Any]] = []
    n_keypoints = 42

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.model = types.SimpleNamespace(
            parameters=lambda: [_Param(1000), _Param(234)],
            named_parameters=lambda: [("model.0.conv.weight", _Param(1000)),
                                      ("model.23.cv4.0.0.conv.weight", _Param(234))],
            named_modules=lambda: [("model.0.conv", None), ("model.23.cv4.0.0.conv", None)],
        )
        self.trainer: Any = None

    def train(self, **kwargs: Any) -> None:
        FakeYOLO.calls.append({"kind": "train", **kwargs})
        best = Path(kwargs["project"]) / "train" / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        # Un checkpoint PICKLABLE : les approches qui fusionnent leurs adaptateurs
        # (lora, lora_per_dataset) le rechargent avec torch.load avant de le reecrire.
        # Un simple b"fake-weights" ferait echouer le depickling.
        _write_fake_checkpoint(best)
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


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installe `ultralytics` ET ses sous-modules dans sys.modules.

    Un module plat ferait echouer `from ultralytics.models.yolo.pose import PoseTrainer`
    avec "ultralytics.models is not a package" : Python resout ces imports par
    sys.modules, pas par attribut.
    """
    root = types.ModuleType("ultralytics")
    root.YOLO = FakeYOLO  # type: ignore[attr-defined]
    root.__path__ = []  # type: ignore[attr-defined]

    modules: dict[str, types.ModuleType] = {"ultralytics": root}
    for name in ("ultralytics.models", "ultralytics.models.yolo",
                 "ultralytics.models.yolo.pose"):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
    modules["ultralytics.models.yolo.pose"].PoseTrainer = FakePoseTrainer  # type: ignore[attr-defined]

    cfg = types.ModuleType("ultralytics.cfg")
    # `quantize` present : le code doit produire {"quantize": 16}, pas {"half": True}
    cfg.DEFAULT_CFG_DICT = {"quantize": None, "half": False}  # type: ignore[attr-defined]
    modules["ultralytics.cfg"] = cfg

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture()
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    """Injecte le faux module `ultralytics` pour la duree du test."""
    FakeYOLO.calls = []
    _install_fake_ultralytics(monkeypatch)
    return FakeYOLO