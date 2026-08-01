"""Adaptateur COCO-keypoints generique (format le plus courant en sortie d'annotation).

Enregistre sous le nom 'coco'. Toute particularite d'un dataset se declare en config
(`data.adapter_options`), jamais par un `if dataset == ...` dans le code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.adapters.base import BaseAdapter
from insectpose.registry import register_adapter
from insectpose.utils.geometry import bbox_from_keypoints


@register_adapter("coco")
class CocoKeypointsAdapter(BaseAdapter):
    """Lit un ou plusieurs JSON COCO-keypoints et produit le format canonique."""

    def read(self) -> pd.DataFrame:
        pattern = str(self.options.get("annotations_glob", "*.json"))
        images_subdir = str(self.options.get("images_subdir", "images"))
        group_field = self.options.get("group_id_field")
        files = sorted(self.source_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"Aucun fichier '{pattern}' dans {self.source_dir}. "
                "Verifier paths.raw et data.raw_subdir."
            )

        rows: list[dict[str, Any]] = []
        for file in files:
            payload = json.loads(Path(file).read_text(encoding="utf-8"))
            images = {img["id"]: img for img in payload.get("images", [])}
            counters: dict[str, int] = {}
            for ann in payload.get("annotations", []):
                img = images.get(ann["image_id"])
                if img is None:
                    raise ValueError(f"[{file.name}] annotation sans image : {ann.get('id')}")
                stem = Path(img["file_name"]).stem
                image_id = f"{self.dataset}/{stem}"
                n = counters.get(image_id, 0)
                counters[image_id] = n + 1

                kpts = np.asarray(ann.get("keypoints", []), dtype=float).reshape(-1, 3)
                kpts_xy = kpts[:, :2].reshape(-1)
                kpts_vis = kpts[:, 2].astype(int)
                bbox = ann.get("bbox")
                if bbox is None:
                    bbox = bbox_from_keypoints(
                        kpts[:, :2], kpts_vis, image_wh=(img["width"], img["height"])
                    ).tolist()

                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": str(
                            Path(self.source_dir.name) / images_subdir / img["file_name"]
                        ),
                        "image_width": int(img["width"]),
                        "image_height": int(img["height"]),
                        "instance_id": f"{image_id}#{n}",
                        "group_id": str(img.get(group_field)) if group_field else None,
                        "bbox_xywh": [float(v) for v in bbox],
                        "kpts_xy": [float(v) for v in kpts_xy],
                        "kpts_vis": [int(v) for v in kpts_vis],
                        "area": float(ann.get("area", bbox[2] * bbox[3])),
                        "keypoint_schema": str(self.options.get("keypoint_schema", self.dataset)),
                        "split_source": str(self.options.get("split_source", "unknown")),
                    }
                )
        df = pd.DataFrame(rows)
        # image_path doit etre relatif a paths.data (contrat 1) : on prefixe par 'raw/'.
        df["image_path"] = "raw/" + df["image_path"].astype(str)
        return df
