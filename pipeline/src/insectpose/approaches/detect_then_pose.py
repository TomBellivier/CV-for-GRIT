"""Approche C : detection puis estimation de pose sur crop (CONVENTIONS.md §9.3).

Deux modeles enchaines dans un seul run :
  1. un detecteur YOLO **poule** (une classe "insecte"), entraine sur les images entieres ;
  2. un modele YOLO-pose entraine sur des crops normalises.

Les trois pieges de cette architecture, tous traites ici :
- le modele de pose s'entraine sur des bboxes GT **bruitees** (§9.3), faute de quoi il
  ne verrait jamais les cadrages imparfaits que produit un detecteur ;
- toute prediction est **retro-projetee** dans le repere de l'image d'origine avant
  ecriture, sans quoi les coordonnees resteraient dans celui du crop (contrat 3) ;
- l'evaluation bout-en-bout utilise les bboxes PREDITES. La variante a bboxes GT est un
  diagnostic (`pose_on_gt_boxes: true`) et ne doit jamais figurer dans le meme tableau.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.approaches.base import BaseApproach
from insectpose.approaches.yolo_pooled import YoloPooledApproach, precision_kwargs, release_model
from insectpose.context import RunContext
from insectpose.data.crop_export import crops_from_boxes, export_crop_fold
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.data.keypoints import KeypointSchema
from insectpose.data.yolo_export import export_fold
from insectpose.registry import register_approach
from insectpose.utils.device import amp_enabled, device_info, peak_vram_mb, reset_peak_vram
from insectpose.utils.geometry import apply_affine, invert_affine

_DETECTOR_KEYS = (
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr", "mosaic",
    "close_mosaic", "cos_lr",
)
_POSE_KEYS = (*_DETECTOR_KEYS, "pose", "kobj")


@register_approach("detect_then_pose")
class DetectThenPoseApproach(BaseApproach):
    """Detecteur poule + modele de pose sur crops normalises."""

    REQUIRED_APPROACH_KEYS = (
        "detector", "pose", "crop", "conf", "iou", "max_det", "inference_precision",
        "predict_chunk_size",
    )

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.detector: Any = None
        self.pose_model: Any = None
        self.schema: KeypointSchema | None = None
        missing = [f"approach.{k}" for k in self.REQUIRED_APPROACH_KEYS
                   if k not in cfg.approach]
        if missing:
            raise KeyError(
                f"[{self.name}] cles de configuration absentes : {missing}. "
                "Comparer configs/approach/detect_then_pose.yaml avec la version du depot."
            )

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        return YoloPooledApproach.availability()

    # --- entrainement ------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Entraine le detecteur puis le modele de pose. Ne lit jamais data.test.

        Effet de bord : ecrit runs/<run_id>/{weights,yolo_dataset,crops,logs}/.
        """
        from ultralytics import YOLO

        self.schema = self._schema(data)
        device = self._device()
        amp = amp_enabled(bool(self.cfg.train.amp), str(self.cfg.mode), device)
        reset_peak_vram()
        started = time.perf_counter()

        # 1. detecteur poule : une classe, images entieres, labels sans keypoints
        detector_data = export_fold(
            data, self.schema, ctx.subdir("yolo_dataset/detector"),
            splits=("train", "val"), with_keypoints=False,
        )
        detector = YOLO(str(self.cfg.approach.detector.weights))
        detector.train(
            data=str(detector_data), project=str(ctx.subdir("logs/detector")), name="train",
            exist_ok=True, seed=ctx.seed("detector"), device=device, amp=amp,
            deterministic=str(self.cfg.mode) == "debug", verbose=False,
            **self._train_kwargs(self.cfg.approach.detector, _DETECTOR_KEYS),
        )
        detector_weights = ctx.subdir("weights/detector") / "best.pt"
        detector_weights.write_bytes(Path(detector.trainer.best).read_bytes())
        detector_time = time.perf_counter() - started
        release_model(detector)

        # 2. modele de pose sur crops issus de bboxes GT BRUITEES (§9.3)
        crop = self.cfg.approach.crop
        pose_data = export_crop_fold(
            data, self.schema, ctx.subdir("crops"), out_size=self._crop_size(),
            padding=float(crop.padding), jitter_scale=float(crop.jitter_scale),
            jitter_shift=float(crop.jitter_shift), seed=ctx.seed("crops"),
        )
        pose_started = time.perf_counter()
        pose_model = YOLO(str(self.cfg.approach.pose.weights))
        pose_model.train(
            data=str(pose_data), project=str(ctx.subdir("logs/pose")), name="train",
            exist_ok=True, seed=ctx.seed("pose"), device=device, amp=amp,
            deterministic=str(self.cfg.mode) == "debug", verbose=False,
            **self._train_kwargs(self.cfg.approach.pose, _POSE_KEYS),
        )
        pose_weights = ctx.subdir("weights/pose") / "best.pt"
        pose_weights.write_bytes(Path(pose_model.trainer.best).read_bytes())
        release_model(pose_model)

        self.detector = YOLO(str(detector_weights))
        self.pose_model = YOLO(str(pose_weights))
        ctx.extra.update({
            "train_time_s": time.perf_counter() - started,
            "detector_train_time_s": detector_time,
            "pose_train_time_s": time.perf_counter() - pose_started,
            "model_params": int(sum(p.numel() for p in self.detector.model.parameters())
                                + sum(p.numel() for p in self.pose_model.model.parameters())),
            "peak_vram_mb": peak_vram_mb(),
            "amp": amp,
            "device": device_info(device),
            "crop_size": list(self._crop_size()),
        })

    # --- inference ---------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:  # noqa: ARG002
        """Detecte, recadre, estime la pose, puis retro-projette (§3.4).

        L'inference est decoupee en lots : Ultralytics materialise tout son `source`
        avant d'inferer, et un fold entier saturerait la RAM (ADR-0021).
        """
        from PIL import Image

        if self.detector is None or self.pose_model is None:
            raise RuntimeError("Modeles non charges : appeler fit() ou load() d'abord.")
        schema = self.schema or self._schema(images)
        approach_cfg = self.cfg.approach
        use_gt_boxes = bool(approach_cfg.get("pose_on_gt_boxes", False))

        table = images.images.set_index("image_id")
        image_ids = list(table.index)
        chunk = max(1, int(approach_cfg.predict_chunk_size))
        device = self._device()
        precision = precision_kwargs(device, str(approach_cfg.inference_precision))
        gt_boxes = self._gt_boxes(images) if use_gt_boxes else {}

        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for start in range(0, len(image_ids), chunk):
            batch_ids = image_ids[start:start + chunk]
            paths = [str(images.absolute_path(table.loc[i, "image_path"])) for i in batch_ids]

            if use_gt_boxes:
                detections = {i: (gt_boxes.get(i, np.zeros((0, 4))),
                                  np.ones(len(gt_boxes.get(i, [])))) for i in batch_ids}
            else:
                detections = self._detect(paths, batch_ids, device, precision)

            crops, matrices, owners = [], [], []
            for image_id, path in zip(batch_ids, paths, strict=True):
                boxes, _ = detections[image_id]
                if len(boxes) == 0:
                    continue
                with Image.open(path) as handle:
                    image = handle.convert("RGB")
                    batch_crops, batch_matrices = crops_from_boxes(
                        image, boxes, self._crop_size(), float(approach_cfg.crop.padding))
                crops.extend(batch_crops)
                matrices.extend(batch_matrices)
                owners.extend([(image_id, j) for j in range(len(boxes))])

            if crops:
                rows.extend(self._pose_on_crops(crops, matrices, owners, detections, table,
                                                schema, device, precision, use_gt_boxes))
            del crops, matrices

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["inference_ms"] = (
                (time.perf_counter() - started) * 1000.0 / max(len(image_ids), 1)
            )
        return frame

    # --- etapes internes ---------------------------------------------------
    def _detect(self, paths: list[str], image_ids: list[str], device: str,
                precision: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Detection sur images entieres, en coordonnees absolues xywh (coin haut-gauche)."""
        results = self.detector.predict(
            source=paths, imgsz=self._imgsz(), conf=float(self.cfg.approach.conf),
            iou=float(self.cfg.approach.iou), max_det=int(self.cfg.approach.max_det),
            device=device, verbose=False, stream=True, **precision,
        )
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for image_id, result in zip(image_ids, results, strict=True):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                out[image_id] = (np.zeros((0, 4)), np.zeros(0))
                continue
            xywh = boxes.xywh.cpu().numpy()
            corner = np.column_stack(
                [xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]]
            )
            out[image_id] = (corner, boxes.conf.cpu().numpy())
        return out

    def _pose_on_crops(self, crops: list[Any], matrices: list[np.ndarray],
                       owners: list[tuple[str, int]],
                       detections: dict[str, tuple[np.ndarray, np.ndarray]],
                       table: pd.DataFrame, schema: KeypointSchema, device: str,
                       precision: dict[str, Any], use_gt_boxes: bool) -> list[dict[str, Any]]:
        """Pose sur les crops, puis retro-projection vers l'image d'origine."""
        results = self.pose_model.predict(
            source=crops, imgsz=self._crop_size()[0], conf=0.0, max_det=1,
            device=device, verbose=False, stream=True, **precision,
        )
        rows: list[dict[str, Any]] = []
        for (image_id, index), matrix, result in zip(owners, matrices, results, strict=True):
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue
            kpts = result.keypoints.data.cpu().numpy()[0]        # (K, 3) repere du crop
            points = apply_affine(invert_affine(matrix), kpts[:, :2])   # -> image d'origine
            boxes, scores = detections[image_id]
            rows.append({
                "image_id": image_id,
                "dataset": str(table.loc[image_id, "dataset"]),
                "bbox_xywh": [float(v) for v in boxes[index]],
                "bbox_score": float(scores[index]),
                "kpts_xy": [float(v) for v in points.reshape(-1)],
                "kpts_score": [float(v) for v in kpts[:, 2]],
                "keypoint_schema": schema.name,
                # Diagnostic si les bboxes viennent de la verite terrain (§9.3).
                "bbox_source": "gt" if use_gt_boxes else "predicted",
            })
        return rows

    @staticmethod
    def _gt_boxes(images: ImageSet) -> dict[str, np.ndarray]:
        """Bboxes GT par image, pour le mode diagnostic uniquement."""
        return {
            str(image_id): np.stack(
                group["bbox_xywh"].map(lambda v: np.asarray(v, float)).to_numpy())
            for image_id, group in images.annotations.groupby("image_id")
        }

    # --- rechargement ------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> DetectThenPoseApproach:
        """Recharge les deux modeles du run, sans reentrainement."""
        from ultralytics import YOLO

        obj = cls(cfg)
        for attribute, name in (("detector", "detector"), ("pose_model", "pose")):
            weights = Path(run_dir) / "weights" / name / "best.pt"
            if not weights.exists():
                raise FileNotFoundError(f"Poids introuvables : {weights}")
            setattr(obj, attribute, YOLO(str(weights)))
        return obj

    # --- utilitaires -------------------------------------------------------
    @staticmethod
    def _schema(source: Any) -> KeypointSchema:
        return YoloPooledApproach._schema(source)

    def _crop_size(self) -> tuple[int, int]:
        """Taille des crops. Egale a la resolution du protocole (ADR-0024)."""
        size = self.cfg.approach.crop.size
        values = [int(size), int(size)] if isinstance(size, int) else [int(v) for v in size]
        return values[0], values[1]

    def _imgsz(self) -> int:
        size = self.cfg.train.image_size
        values = [int(size), int(size)] if isinstance(size, int) else [int(v) for v in size]
        return values[0]

    def _device(self) -> str:
        from insectpose.utils.device import resolve_device

        return resolve_device(self.cfg.train.device)

    def _train_kwargs(self, section: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        """Hyperparametres d'un des deux modeles, tous issus de la config."""
        kwargs: dict[str, Any] = {
            "epochs": int(section.get("epochs", self.cfg.train.epochs)),
            "batch": int(section.get("batch", self.cfg.train.batch_size)),
            "imgsz": self._crop_size()[0] if section is self.cfg.approach.pose else self._imgsz(),
            "workers": int(self.cfg.train.num_workers),
            "patience": int(self.cfg.train.early_stopping_patience),
            "cache": self.cfg.train.cache,
            "plots": bool(self.cfg.train.plots),
        }
        for key in keys:
            if key in section:
                kwargs[key] = section[key]
        return kwargs