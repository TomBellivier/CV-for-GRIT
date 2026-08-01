"""Approche A : YOLO-pose entraine sur l'ensemble des datasets (CONVENTIONS.md §9.1).

Une seule classe "insecte", un seul modele, les 42 keypoints du schema commun
(ADR-0006). Comme le schema est partage par les 4 ordres, aucune reprojection
union -> local n'est necessaire : les predictions sortent deja dans le schema attendu.

Les keypoints absents d'un dataset (ADR-0016) sont ecrits `vis = 0` dans les labels :
Ultralytics les masque dans la loss de pose, il ne les apprend pas comme des zeros.

Ce module est une COUCHE MINCE au-dessus d'Ultralytics. Toute la logique risquee
(conversion de coordonnees, format des labels) vit dans `data/yolo_export.py`, qui est
testable sans GPU par aller-retour.
"""

from __future__ import annotations

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
)

_TRAIN_KEYS = (
    "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "box", "pose", "kobj", "cls", "dfl", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
    "fliplr", "mosaic", "mixup", "copy_paste", "erasing", "close_mosaic", "patience",
    "workers", "cos_lr", "dropout", "freeze",
)


@register_approach("yolo_pooled")
class YoloPooledApproach(BaseApproach):
    """YOLO-pose unique, entraine sur les 4 datasets confondus."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.model: Any = None
        self.schema: KeypointSchema | None = None

    # --- disponibilite -----------------------------------------------------
    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """Ultralytics et torch sont des dependances de premier rang (ADR-0019).

        Le mecanisme reste en place : il permet a une machine sans GPU ou a une CI
        legere d'ignorer proprement l'approche au lieu d'echouer.
        """
        try:
            import torch  # noqa: F401
            import ultralytics  # noqa: F401
        except ImportError as exc:
            return False, f"{exc.name} absent : pip install -e \".[dev]\""
        return True, ""

    # --- entrainement ------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Exporte le fold au format YOLO puis entraine. Ne lit jamais data.test.

        Effet de bord : ecrit runs/<run_id>/yolo_dataset/ et runs/<run_id>/weights/.
        """
        from ultralytics import YOLO

        self.schema = self._schema(data)
        dataset_dir = ctx.subdir("yolo_dataset")
        data_yaml = export_fold(data, self.schema, dataset_dir, splits=("train", "val"))
        self._check_augmentation()

        device = self._device()
        amp = amp_enabled(bool(self.cfg.train.amp), str(self.cfg.mode), device)
        reset_peak_vram()
        ctx.logger.info("Entrainement YOLO sur '%s' (AMP=%s) | %s",
                        device, amp, device_info(device).get("devices", "cpu"))

        started = time.perf_counter()
        self.model = YOLO(str(self.cfg.approach.weights))
        self.model.train(
            data=str(data_yaml),
            project=str(ctx.subdir("logs")),
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
        target = ctx.subdir("weights") / "best.pt"
        target.write_bytes(best.read_bytes())
        self.model = YOLO(str(target))

        ctx.extra.update({
            "train_time_s": time.perf_counter() - started,
            "model_params": int(sum(p.numel() for p in self.model.model.parameters())),
            "base_weights": str(self.cfg.approach.weights),
            "peak_vram_mb": peak_vram_mb(),
            "amp": amp,
            "device": device_info(device),
        })

    # --- inference ---------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:  # noqa: ARG002
        """Predit puis remet tout dans le repere de l'image d'origine (§3.4).

        Ultralytics renvoie deja des coordonnees absolues dans l'image source, mais en
        bbox CENTREE : la conversion vers le coin haut-gauche du contrat 3 se fait ici.
        Aucun seuil de score fort n'est applique : le seuillage est une operation
        d'evaluation (§3.4).
        """
        if self.model is None:
            raise RuntimeError("Modele non charge : appeler fit() ou load() d'abord.")
        schema = self.schema or self._schema(images)
        approach_cfg = self.cfg.approach

        table = images.images.set_index("image_id")
        paths = [str(images.absolute_path(row.image_path)) for row in table.itertuples()]
        image_ids = list(table.index)

        started = time.perf_counter()
        device = self._device()
        results = self.model.predict(
            source=paths,
            imgsz=self._imgsz(),
            conf=float(approach_cfg.conf),
            iou=float(approach_cfg.iou),
            max_det=int(approach_cfg.max_det),
            device=device,
            half=bool(approach_cfg.half) and device != "cpu",
            batch=int(self.cfg.train.batch_size),
            verbose=False,
            stream=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        rows: list[dict[str, Any]] = []
        for image_id, result in zip(image_ids, results, strict=True):
            dataset = str(table.loc[image_id, "dataset"])
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xywh = boxes.xywh.cpu().numpy()          # centre + taille, pixels image source
            scores = boxes.conf.cpu().numpy()
            kpts = result.keypoints.data.cpu().numpy()   # (n, K, 3) : x, y, score
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
                    "inference_ms": elapsed_ms / max(len(image_ids), 1),
                })
        return pd.DataFrame(rows)

    # --- rechargement ------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> YoloPooledApproach:
        """Recharge les poids du run, sans reentrainement."""
        from ultralytics import YOLO

        weights = Path(run_dir) / "weights" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Poids introuvables : {weights}")
        obj = cls(cfg)
        obj.model = YOLO(str(weights))
        return obj

    # --- utilitaires internes ---------------------------------------------
    @staticmethod
    def _schema(source: Any) -> KeypointSchema:
        """Schema commun du perimetre courant ; refuse un perimetre heterogene."""
        schemas = source.schemas
        dataset_schemas = {
            name: schema for name, schema in schemas.items() if schema.kind == "dataset_schema"
        }
        if len(dataset_schemas) != 1:
            raise ValueError(
                f"yolo_pooled exige un schema de keypoints unique, trouve "
                f"{sorted(dataset_schemas)}. Avec des schemas divergents, il faudrait passer "
                "par l'espace union et reprojeter a l'ecriture (§3.1)."
            )
        return next(iter(dataset_schemas.values()))

    def _check_augmentation(self) -> None:
        """Un miroir sans table de symetrie apprend une anatomie fausse (§3.1)."""
        if float(self.cfg.approach.get("fliplr", 0.0)) > 0 and self.schema is not None:
            identity = list(self.schema.flip_index) == list(range(self.schema.n_keypoints))
            if identity:
                raise ValueError(
                    "fliplr > 0 alors que le schema n'a aucune paire de symetrie : "
                    "l'augmentation par miroir echangerait gauche et droite sans permuter "
                    "les labels."
                )

    def _imgsz(self) -> int:
        """Resolution commune du protocole (ADR-0013). Ultralytics veut un entier."""
        size = self.cfg.train.image_size
        values = [int(size), int(size)] if isinstance(size, int) else [int(v) for v in size]
        if values[0] != values[1]:
            raise ValueError(f"Ultralytics exige une resolution carree, recu {values}.")
        return values[0]

    def _device(self) -> str:
        """Peripherique resolu (ADR-0019) : 'auto' -> GPU 0 si CUDA, sinon 'cpu'."""
        return resolve_device(self.cfg.train.device)

    def _train_kwargs(self) -> dict[str, Any]:
        """Hyperparametres passes a Ultralytics, tous issus de la config."""
        approach_cfg = self.cfg.approach
        kwargs: dict[str, Any] = {
            "epochs": int(self.cfg.train.epochs),
            "batch": int(self.cfg.train.batch_size),
            "imgsz": self._imgsz(),
            "workers": int(self.cfg.train.num_workers),
            "patience": int(self.cfg.train.early_stopping_patience),
        }
        for key in _TRAIN_KEYS:
            if key in approach_cfg and key not in kwargs:
                kwargs[key] = approach_cfg[key]
        return kwargs


def export_prediction_set(images: ImageSet, schema: KeypointSchema, root: Path,
                          split: str = "test") -> Path:
    """Exporte un ImageSet au format YOLO (diagnostic : inspection manuelle d'un fold).

    Effet de bord : ecrit sous `root`. N'est pas utilise par le pipeline.
    """
    export_split(images, schema, root, split)
    return write_data_yaml(root, schema, {split: split})


__all__ = ["YoloPooledApproach", "export_prediction_set", "flat_name"]
