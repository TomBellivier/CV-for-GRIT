"""Ultralytics test double shared by the tests of the YOLO approaches.

Ultralytics and CUDA cannot be installed everywhere (light CI, dev machine). This double
checks what WE control — arguments passed to the training, coordinate conversion,
routing between models, parameter freezing — without a GPU or weights to download.

It also exposes `ultralytics.models.yolo.pose` and `ultralytics.cfg`: the approaches that
pass a custom trainer (lora, group_bn, head_only) import `PoseTrainer`, and replacing
`sys.modules["ultralytics"]` with a flat module would make these imports fail with
"ultralytics.models is not a package".
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class _Arr:
    """Minimal shim exposing the `.cpu().numpy()` interface of torch tensors."""

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
    """Minimal trainer: the approaches derive from it through `make_patched_trainer`.

    It trains nothing, but exposes the hooks that the patch overrides, which checks that
    the chain builds without error.
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


class FakeLoraLayer:
    """Minimal LoRA layer, mergeable and picklable.

    Approaches D and H merge their adapters into the base weights before saving
    (ADR-0025), and refuse a checkpoint that contains none — a legitimate safeguard of
    the real code. The double must therefore produce some.
    """

    def __init__(self, name: str = "conv") -> None:
        self.base = _FakeBaseLayer(name)
        self.merged = False

    def named_children(self) -> list[tuple[str, Any]]:
        return [("base_layer", self.base)]

    def merge(self) -> None:
        self.merged = True

    def get_base_layer(self) -> Any:
        return self.base


class _FakeBaseLayer:
    def __init__(self, name: str) -> None:
        self.name = name

    def named_children(self) -> list[tuple[str, Any]]:
        return []


class _FakeCheckpointModule:
    """Picklable module mimicking the tree of a patched YOLO checkpoint."""

    def __init__(self) -> None:
        self.conv = FakeLoraLayer("model.20.conv")

    def named_children(self) -> list[tuple[str, Any]]:
        return [("conv", self.conv)]


def _write_fake_checkpoint(path: Path) -> None:
    """Write a checkpoint readable by torch.load, containing a LoRA layer.

    Side effect: creates `path`.
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
    """Ultralytics double: records the calls, returns plausible outputs."""

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
        # A PICKLABLE checkpoint: the approaches that merge their adapters
        # (lora, lora_per_dataset) reload it with torch.load before rewriting it.
        # A plain b"fake-weights" would make the unpickling fail.
        _write_fake_checkpoint(best)
        self.trainer = types.SimpleNamespace(best=str(best))

    def predict(self, source: list[str], **kwargs: Any) -> list[_Result]:
        FakeYOLO.calls.append({"kind": "predict", "n_sources": len(source), **kwargs})
        assert kwargs.get("stream") is True, (
            "predict() must be streamed: otherwise Ultralytics keeps one Results per image "
            "(original image included) and the process gets killed by the OOM killer."
        )
        assert "half" not in kwargs, "'half' is deprecated: use 'quantize'."
        results = []
        for i in range(len(source)):
            # CENTRED bbox, like Ultralytics: centre (60, 80), size 40 x 20
            boxes = _Boxes(np.array([[60.0, 80.0, 40.0, 20.0]]), np.array([0.9 - i * 0.001]))
            kpts = np.zeros((1, self.n_keypoints, 3))
            kpts[0, :, 0] = np.linspace(45, 75, self.n_keypoints)
            kpts[0, :, 1] = np.linspace(72, 88, self.n_keypoints)
            kpts[0, :, 2] = 0.8
            results.append(_Result(boxes, _Keypoints(kpts)))
        return results


class _FakeLoraConfig:
    """Minimal LoRA config: the double only reads the logged fields."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _fake_inject(config: Any, model: Any, **kwargs: Any) -> Any:
    """Simulated injection: makes a few 'lora_' parameters visible on the model.

    Enough to check that the freezing spares the adapters and freezes the rest.
    """
    existing = list(model.named_parameters()) if hasattr(model, "named_parameters") else []
    adapters = [("model.20.conv.lora_A.default.weight", _Param(64)),
                ("model.20.conv.lora_B.default.weight", _Param(64))]
    model.named_parameters = lambda: [*existing, *adapters]  # type: ignore[attr-defined]
    return model


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install `ultralytics` AND its submodules in sys.modules.

    A flat module would make `from ultralytics.models.yolo.pose import PoseTrainer` fail
    with "ultralytics.models is not a package": Python resolves these imports through
    sys.modules, not through attributes.
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
    # `quantize` present: the code must produce {"quantize": 16}, not {"half": True}
    cfg.DEFAULT_CFG_DICT = {"quantize": None, "half": False}  # type: ignore[attr-defined]
    modules["ultralytics.cfg"] = cfg

    # `peft`: approaches D and H import it to inject then merge the adapters. The fake
    # `LoraLayer` must be the class FakeLoraLayer inherits from, otherwise
    # `merge_lora_weights` — which tests isinstance — would recognise nothing and
    # refuse the checkpoint.
    peft_layer = types.ModuleType("peft.tuners.lora.layer")
    peft_layer.LoraLayer = FakeLoraLayer  # type: ignore[attr-defined]
    for name in ("peft", "peft.tuners", "peft.tuners.lora"):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
    modules["peft"].LoraConfig = _FakeLoraConfig  # type: ignore[attr-defined]
    modules["peft"].inject_adapter_in_model = _fake_inject  # type: ignore[attr-defined]
    modules["peft.tuners.lora"].LoraLayer = FakeLoraLayer  # type: ignore[attr-defined]
    modules["peft.tuners.lora.layer"] = peft_layer

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture()
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYOLO]:
    """Inject the fake `ultralytics` module for the duration of the test."""
    FakeYOLO.calls = []
    _install_fake_ultralytics(monkeypatch)
    return FakeYOLO