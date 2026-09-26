"""Approach A: YOLO-pose trained on all the datasets (CONVENTIONS.md §9.1).

A single "insect" class, a single model, the 42 keypoints of the common schema
(ADR-0006). Since the schema is shared by the 4 orders, no union -> local reprojection
is needed: the predictions already come out in the expected schema.

The keypoints absent from a dataset (ADR-0016) are written `vis = 0` in the labels:
Ultralytics masks them in the pose loss, it does not learn them as zeros.

This module is a THIN LAYER on top of Ultralytics. All the risky logic (coordinate
conversion, label format) lives in `data/yolo_export.py`, which is testable without a
GPU by a round trip.

It is the approach whose runs are retained for the pipeline (retain.approaches).
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import pandas as pd

from insectpose.approaches.base import BaseApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.data.keypoints import KeypointSchema
from insectpose.data.yolo_export import export_fold, export_split, flat_name, write_data_yaml
from insectpose.registry import register_approach
from insectpose.utils.device import (
    amp_enabled,
    device_info,
    peak_vram_mb,
    reset_peak_vram,
    resolve_device,
    resolve_num_workers,
)

_TRAIN_KEYS = (
    "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "box", "pose", "kobj", "cls", "dfl", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
    "fliplr", "mosaic", "mixup", "copy_paste", "erasing", "close_mosaic", "patience",
    "workers", "cos_lr", "dropout", "freeze",
)


def precision_kwargs(device: str, precision: str = "fp16") -> dict[str, Any]:
    """Inference precision argument, compatible across Ultralytics versions.

    `half` is deprecated since Ultralytics 8.4 in favour of `quantize` (16 = FP16,
    None = FP32). The installed default config is queried rather than comparing version
    numbers, and NOTHING is passed in FP32: it is already the default, and it avoids a
    deprecation warning at every call.
    """
    if str(precision) != "fp16" or device == "cpu":
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError:
        return {"half": True}
    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


def release_model(model: Any) -> None:
    """Release an Ultralytics model and the memory of its trainer.

    Without it, inference starts with several GB already taken by the dataloaders,
    workers and augmentation buffers of the training (ADR-0019).
    """
    trainer = getattr(model, "trainer", None)
    for attribute in ("train_loader", "test_loader", "validator", "ema", "optimizer"):
        if trainer is not None and hasattr(trainer, attribute):
            setattr(trainer, attribute, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@register_approach("yolo_pooled")
class YoloPooledApproach(BaseApproach):
    """A single YOLO-pose, trained on the 4 datasets pooled."""

    #: Keys the approach reads in its config. No default value hidden in the code
    #: (CONVENTIONS.md §5.2): the config must declare them, but their absence must produce
    #: an actionable message rather than a raw OmegaConf error.
    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
    )
    REQUIRED_TRAIN_KEYS = (
        "epochs", "batch_size", "image_size", "num_workers", "early_stopping_patience",
        "device", "amp", "cache", "plots",
    )

    def __init__(self, cfg: Any, namespace: str = "") -> None:
        super().__init__(cfg)
        self.model: Any = None
        self.schema: KeypointSchema | None = None
        # A non-empty `namespace` isolates the artefacts of this model in the run, which
        # lets a composite approach (§9.2) host several of them without collision.
        self.namespace = namespace
        self._check_config()

    def _artifact_dir(self, ctx: RunContext, kind: str) -> Path:
        """Sub-folder of the run for this model ('weights', 'yolo_dataset'...)."""
        return ctx.subdir(f"{kind}/{self.namespace}" if self.namespace else kind)

    def _check_config(self) -> None:
        """Check the presence of the expected keys and name the missing ones."""
        missing = [f"approach.{k}" for k in self.REQUIRED_APPROACH_KEYS
                   if k not in self.cfg.approach]
        missing += [f"train.{k}" for k in self.REQUIRED_TRAIN_KEYS if k not in self.cfg.train]
        if missing:
            raise KeyError(
                f"[{self.name}] missing configuration keys: {missing}. "
                "Your configs/approach/yolo_pooled.yaml or configs/config.yaml is older "
                "than the code. Compare with the version of the repository."
            )

    # --- availability -----------------------------------------------------------
    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """Ultralytics and torch are first-rank dependencies (ADR-0019).

        The mechanism stays in place: it lets a machine without a GPU or a light CI
        cleanly skip the approach instead of failing.
        """
        try:
            import torch  # noqa: F401
            import ultralytics  # noqa: F401
        except ImportError as exc:
            return False, f"{exc.name} missing: pip install -e \".[dev]\""
        return True, ""

    # --- training ---------------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Export the fold in the YOLO format then train. Never reads data.test.

        Side effect: writes runs/<run_id>/yolo_dataset/ and runs/<run_id>/weights/.
        """
        from ultralytics import YOLO

        data = self._prepare_data(data, ctx)
        self.schema = self._schema(data)
        dataset_dir = self._artifact_dir(ctx, "yolo_dataset")
        data_yaml = export_fold(data, self.schema, dataset_dir, splits=("train", "val"))
        self._check_augmentation()

        device = self._device()
        amp = amp_enabled(bool(self.cfg.train.amp), str(self.cfg.mode), device)
        reset_peak_vram()
        ctx.logger.info("YOLO training on '%s' (AMP=%s) | %s",
                        device, amp, device_info(device).get("devices", "cpu"))

        started = time.perf_counter()
        self.model = YOLO(str(self.cfg.approach.weights))
        trainer = self._trainer_class(ctx)
        self.model.train(
            data=str(data_yaml),
            **({"trainer": trainer} if trainer is not None else {}),
            project=str(self._artifact_dir(ctx, "logs")),
            name="train",
            exist_ok=True,
            seed=ctx.seed("train"),
            deterministic=str(self.cfg.mode) == "debug",
            device=device,
            amp=amp,
            cache=self.cfg.train.cache,
            plots=bool(self.cfg.train.plots),
            verbose=False,
            **self._train_kwargs(),
        )
        best = Path(self.model.trainer.best)
        target = self._artifact_dir(ctx, "weights") / "best.pt"
        self._write_checkpoint(best, target)

        # The trainer holds dataloaders, workers and augmentation buffers: without an
        # explicit release, the prediction starts with several GB already taken.
        self._release_trainer()
        self.model = YOLO(str(target))
        self._prepare_inference_model(self.model)

        prefix = f"{self.namespace}_" if self.namespace else ""
        ctx.extra.update({
            f"{prefix}train_time_s": time.perf_counter() - started,
            f"{prefix}model_params": int(sum(p.numel() for p in self.model.model.parameters())),
            f"{prefix}base_weights": str(self.cfg.approach.weights),
            f"{prefix}peak_vram_mb": peak_vram_mb(),
            f"{prefix}amp": amp,
            f"{prefix}device": device_info(device),
        })

    # --- extension points for the derived approaches ------------------------------
    def _prepare_data(self, data: FoldData, ctx: RunContext) -> FoldData:  # noqa: ARG002
        """Transformation of the fold before the export. By default: none."""
        return data

    def _trainer_class(self, ctx: RunContext) -> Any:  # noqa: ARG002
        """Custom Ultralytics trainer, or None for the standard trainer."""
        return None

    def _write_checkpoint(self, best: Path, target: Path) -> None:
        """Write the weights of the run. By default: raw copy of the best checkpoint."""
        target.write_bytes(best.read_bytes())

    def _prepare_inference_model(self, model: Any) -> None:
        """Adjustments of the model right after loading it for inference. None by default."""

    def _release_trainer(self) -> None:
        """Release the trainer and its memory between training and inference."""
        release_model(self.model)
        self.model = None

    # --- inference --------------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:  # noqa: ARG002
        """Predict then put everything back into the frame of the original image (§3.4).

        Ultralytics already returns absolute coordinates in the source image, but as a
        CENTRED bbox: the conversion to the top-left corner of contract 3 happens here.
        No strong score threshold is applied: thresholding is an evaluation operation
        (§3.4).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded: call fit() or load() first.")
        schema = self.schema or self._schema(images)
        approach_cfg = self.cfg.approach

        table = images.images.set_index("image_id")
        paths = [str(images.absolute_path(row.image_path)) for row in table.itertuples()]
        image_ids = list(table.index)

        device = self._device()
        started = time.perf_counter()
        # SPLITTING INTO CHUNKS IS MANDATORY (ADR-0021). Passing the whole list of images
        # to predict() makes Ultralytics build a loader that materialises ALL the images
        # at once (`self.im0 = [...]`, `bs = len(im0)`). `stream=True` changes nothing:
        # the accumulation happens when the loader is built, before any inference. On a
        # fold of several thousand images, the RAM saturates and the process is killed by
        # the OOM killer.
        chunk_size = max(1, int(approach_cfg.predict_chunk_size))
        precision = self._precision_kwargs(device, str(approach_cfg.inference_precision))

        rows: list[dict[str, Any]] = []
        for start in range(0, len(paths), chunk_size):
            chunk_paths = paths[start:start + chunk_size]
            chunk_ids = image_ids[start:start + chunk_size]
            results = self.model.predict(
                source=chunk_paths,
                imgsz=self._imgsz(),
                conf=float(approach_cfg.conf),
                iou=float(approach_cfg.iou),
                max_det=int(approach_cfg.max_det),
                device=device,
                verbose=False,
                stream=True,
                **precision,
            )
            rows.extend(self._rows_from_results(chunk_ids, results, table, schema))
            del results
            gc.collect()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["inference_ms"] = elapsed_ms / max(len(image_ids), 1)
        return frame

    @staticmethod
    def _rows_from_results(image_ids: list[str], results: Any, table: pd.DataFrame,
                           schema: KeypointSchema) -> list[dict[str, Any]]:
        """Convert a chunk of Results into rows of contract 3.

        Isolated from the chunk loop so that no Results object outlives its chunk: each
        one carries the original image, and it is this retention that saturated the RAM.
        """
        rows: list[dict[str, Any]] = []
        for image_id, result in zip(image_ids, results, strict=True):
            dataset = str(table.loc[image_id, "dataset"])
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xywh = boxes.xywh.cpu().numpy()          # centre + size, source image pixels
            scores = boxes.conf.cpu().numpy()
            kpts = result.keypoints.data.cpu().numpy()   # (n, K, 3): x, y, score
            for i in range(len(boxes)):
                cx, cy, w, h = xywh[i]
                rows.append({
                    "image_id": image_id,
                    "dataset": dataset,
                    "bbox_xywh": [float(cx - w / 2), float(cy - h / 2), float(w), float(h)],
                    "bbox_score": float(scores[i]),
                    "kpts_xy": [float(v) for v in kpts[i, :, :2].reshape(-1)],
                    "kpts_score": [float(v) for v in kpts[i, :, 2]],
                    "keypoint_schema": schema.name,
                    "bbox_source": "predicted",
                })
        return rows

    # --- reloading --------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any, namespace: str = "") -> YoloPooledApproach:
        """Reload the weights of the run, without retraining."""
        from ultralytics import YOLO

        weights = Path(run_dir) / "weights" / namespace / "best.pt" if namespace \
            else Path(run_dir) / "weights" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        obj = cls(cfg, namespace=namespace)
        obj.model = YOLO(str(weights))
        obj._prepare_inference_model(obj.model)
        return obj

    # --- internal utilities -----------------------------------------------------
    @staticmethod
    def _schema(source: Any) -> KeypointSchema:
        """Common schema of the current scope; refuses a heterogeneous scope."""
        schemas = source.schemas
        dataset_schemas = {
            name: schema for name, schema in schemas.items() if schema.kind == "dataset_schema"
        }
        if len(dataset_schemas) != 1:
            raise ValueError(
                f"yolo_pooled requires a single keypoint schema, found "
                f"{sorted(dataset_schemas)}. With diverging schemas, it would have to go "
                "through the union space and reproject when writing (§3.1)."
            )
        return next(iter(dataset_schemas.values()))

    def _check_augmentation(self) -> None:
        """A mirror without a symmetry table learns a wrong anatomy (§3.1)."""
        if float(self.cfg.approach.get("fliplr", 0.0)) > 0 and self.schema is not None:
            identity = list(self.schema.flip_index) == list(range(self.schema.n_keypoints))
            if identity:
                raise ValueError(
                    "fliplr > 0 whereas the schema has no symmetry pair: the mirror "
                    "augmentation would swap left and right without permuting the labels."
                )

    def _imgsz(self) -> int:
        """Common resolution of the protocol (ADR-0013). Ultralytics wants an integer."""
        size = self.cfg.train.image_size
        values = [int(size), int(size)] if isinstance(size, int) else [int(v) for v in size]
        if values[0] != values[1]:
            raise ValueError(f"Ultralytics requires a square resolution, got {values}.")
        return values[0]

    @staticmethod
    def _precision_kwargs(device: str, precision: str = "fp16") -> dict[str, Any]:
        """Delegate to the module helper, reused by the other YOLO approaches."""
        return precision_kwargs(device, precision)

    def _device(self) -> str:
        """Resolved device (ADR-0019): 'auto' -> GPU 0 if CUDA, else the Apple GPU, else 'cpu'."""
        return resolve_device(self.cfg.train.device)

    def _train_kwargs(self) -> dict[str, Any]:
        """Hyperparameters passed to Ultralytics, all coming from the config."""
        approach_cfg = self.cfg.approach
        kwargs: dict[str, Any] = {
            "epochs": int(self.cfg.train.epochs),
            "batch": int(self.cfg.train.batch_size),
            "imgsz": self._imgsz(),
            "workers": resolve_num_workers(self.cfg.train.num_workers),
            "patience": int(self.cfg.train.early_stopping_patience),
        }
        for key in _TRAIN_KEYS:
            if key in approach_cfg and key not in kwargs:
                kwargs[key] = approach_cfg[key]
        return kwargs


def export_prediction_set(images: ImageSet, schema: KeypointSchema, root: Path,
                          split: str = "test") -> Path:
    """Export an ImageSet in the YOLO format (diagnostic: manual inspection of a fold).

    Side effect: writes under `root`. Not used by the pipeline.
    """
    export_split(images, schema, root, split)
    return write_data_yaml(root, schema, {split: split})


__all__ = ["YoloPooledApproach", "export_prediction_set", "flat_name"]