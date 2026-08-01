"""Adaptateur raw YOLO-pose -> format canonique (contrat 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from insectpose.data.adapters.base import BaseAdapter
from insectpose.registry import register_adapter

_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@register_adapter("yolo")
class YoloPoseAdapter(BaseAdapter):
    """Lit images/<split>/ + labels/<split>/ au format YOLO-pose normalisé."""

    def read(self) -> pd.DataFrame:
        splits = [str(s) for s in self.options.get("splits", ["train", "val", "test"])]
        schema_name = str(self.options.get("keypoint_schema", "insect42_v1"))
        rows: list[dict[str, Any]] = []

        for split in splits:
            label_dir = self.source_dir / "labels" / split
            image_dir = self.source_dir / "images" / split
            if not label_dir.exists():
                continue
            for label_file in sorted(label_dir.glob("*.txt")):
                image_path = next(
                    (image_dir / f"{label_file.stem}{ext}" for ext in _EXTENSIONS
                     if (image_dir / f"{label_file.stem}{ext}").exists()), None
                )
                if image_path is None:
                    raise FileNotFoundError(
                        f"Aucune image pour {label_file} dans {image_dir}"
                    )
                with Image.open(image_path) as img:
                    width, height = img.size
                image_id = f"{self.dataset}/{label_file.stem}"

                for n, line in enumerate(label_file.read_text().split("\n")):
                    if not line.strip():
                        continue
                    values = [float(v) for v in line.split()]
                    cx, cy, bw, bh = values[1:5]
                    kpts = np.asarray(values[5:], dtype=float)
                    stride = 3 if kpts.size % 3 == 0 else 2
                    kpts = kpts.reshape(-1, stride)
                    xy = kpts[:, :2] * np.array([width, height])
                    vis = (
                        kpts[:, 2].astype(int) if stride == 3
                        else np.where((kpts[:, :2] == 0).all(axis=1), 0, 2)
                    )
                    x = (cx - bw / 2) * width
                    y = (cy - bh / 2) * height
                    rows.append({
                        "image_id": image_id,
                        "image_path": (
                            f"raw/{self.source_dir.name}/"
                            f"{image_path.relative_to(self.source_dir)}"
                        ),
                        "image_width": width,
                        "image_height": height,
                        "instance_id": f"{image_id}#{n}",
                        "group_id": None,       # ADR-0011 : une image = un spécimen
                        "bbox_xywh": [x, y, bw * width, bh * height],
                        "kpts_xy": [float(v) for v in xy.reshape(-1)],
                        "kpts_vis": [int(v) for v in vis],
                        "area": float(bw * width * bh * height),
                        "keypoint_schema": schema_name,
                        "split_source": split,
                    })
        return pd.DataFrame(rows)