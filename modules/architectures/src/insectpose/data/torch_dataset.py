"""Dataset torch generique produisant le SUPERSET de champs du §4.3.

Une approche ignore les champs qui ne la concernent pas ; aucune ne doit avoir a
modifier le datamodule pour obtenir un champ manquant.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from insectpose.data.datamodule import ImageSet, dataset_index
from insectpose.utils.geometry import crop_affine, invert_affine, jitter_bbox
from insectpose.utils.optional import require


def build_instance_records(image_set: ImageSet) -> list[dict[str, Any]]:
    """Aplatit un ImageSet en enregistrements par instance (sans charger les images)."""
    records: list[dict[str, Any]] = []
    for row in image_set.annotations.itertuples(index=False):
        records.append(
            {
                "image_id": row.image_id,
                "instance_id": row.instance_id,
                "image_path": str(image_set.absolute_path(row.image_path)),
                "dataset": row.dataset,
                "dataset_index": dataset_index(row.dataset),
                "group_id": row.group_id,
                "orig_size": (int(row.image_width), int(row.image_height)),
                "bbox_xywh": np.asarray(row.bbox_xywh, dtype=float),
                "kpts_xy": np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2),
                "kpts_vis": np.asarray(row.kpts_vis, dtype=int),
                "keypoint_schema": row.keypoint_schema,
            }
        )
    return records


class InstanceCropDataset:  # pragma: no cover - necessite torch + images reelles
    """Dataset d'instances recadrees, avec matrice de transformation conservee.

    La `transform_matrix` retournee permet la retro-projection obligatoire des
    predictions vers le repere de l'image d'origine (§3.4, §9.3).
    """

    def __init__(self, image_set: ImageSet, out_size: tuple[int, int], train: bool = False,
                 jitter_scale: float = 0.15, jitter_shift: float = 0.10, seed: int = 0) -> None:
        self.records = build_instance_records(image_set)
        self.out_size = out_size
        self.train = train
        self.jitter_scale = jitter_scale
        self.jitter_shift = jitter_shift
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        rec = self.records[idx]
        bbox = rec["bbox_xywh"]
        if self.train:
            # Crops issus de bboxes GT BRUITEES : sinon decalage train/test (§9.3).
            bbox = jitter_bbox(bbox, self.rng, self.jitter_scale, self.jitter_shift)
        matrix = crop_affine(bbox, self.out_size)
        image = Image.open(rec["image_path"]).convert("RGB")
        crop = image.transform(
            self.out_size, Image.AFFINE, data=tuple(invert_affine(matrix).reshape(-1)),
            resample=Image.BILINEAR,
        )
        kpts = rec["kpts_xy"] @ matrix[:, :2].T + matrix[:, 2]
        return {
            "image": np.asarray(crop, dtype=np.float32) / 255.0,
            "keypoints": kpts.astype(np.float32),
            "visibility": rec["kpts_vis"],
            "bbox": bbox.astype(np.float32),
            "meta": {
                "image_id": rec["image_id"],
                "instance_id": rec["instance_id"],
                "dataset": rec["dataset"],
                "dataset_index": rec["dataset_index"],
                "group_id": rec["group_id"],
                "orig_size": rec["orig_size"],
                "transform_matrix": matrix.astype(np.float32),
                "keypoint_schema": rec["keypoint_schema"],
            },
        }

    def as_torch(self) -> Any:
        """Enveloppe torch.utils.data.Dataset (import differe)."""
        torch_utils = require("torch.utils.data", "torch")
        parent = self

        class _Wrapped(torch_utils.Dataset):  # type: ignore[misc, name-defined]
            def __len__(self) -> int:
                return len(parent)

            def __getitem__(self, i: int) -> dict[str, Any]:
                return parent[i]

        return _Wrapped()
