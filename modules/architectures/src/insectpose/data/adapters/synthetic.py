"""Adaptateur synthetique : genere un mini-corpus reproductible.

Sert aux tests de contrat et au smoke test de bout en bout (§10). Ne doit jamais
alimenter un resultat de rapport : les images sont artificielles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.adapters.base import BaseAdapter
from insectpose.registry import register_adapter

# Gabarit anatomique approximatif du schema insect42_v1, en unites de demi-longueur
# de corps (x vers la droite, y vers le bas). Sert uniquement aux fixtures de test :
# il rend la largeur de thorax et les mesures morphometriques realistes, donc les
# metriques du pipeline interpretables sur des donnees jouets.
_TEMPLATE_42 = [
    (0.00, -1.00), (-0.18, -0.85), (0.18, -0.85), (-0.10, -0.88), (0.10, -0.88),
    (0.00, -0.72), (-0.22, -0.45), (0.22, -0.45), (0.00, -0.20), (-0.18, 0.15),
    (0.18, 0.15), (0.00, 0.60),
    (-0.08, -1.02), (-0.25, -1.25), (-0.45, -1.45),
    (0.08, -1.02), (0.25, -1.25), (0.45, -1.45),
    (-0.22, -0.40), (-1.00, 0.10), (-0.62, -0.30), (-0.55, 0.25),
    (0.22, -0.40), (1.00, 0.10), (0.62, -0.30), (0.55, 0.25),
    (-0.20, -0.25), (-0.75, 0.35), (-0.45, -0.05), (-0.40, 0.45),
    (0.20, -0.25), (0.75, 0.35), (0.45, -0.05), (0.40, 0.45),
    (-0.20, -0.30), (-0.45, -0.05), (-0.60, 0.30), (-0.70, 0.55),
    (0.20, -0.30), (0.45, -0.05), (0.60, 0.30), (0.70, 0.55),
]

@register_adapter("synthetic")
class SyntheticAdapter(BaseAdapter):
    """Genere des annotations plausibles (et les images si `write_images=True`)."""

    def read(self) -> pd.DataFrame:
        opts: dict[str, Any] = self.options
        n_images = int(opts.get("n_images", 12))
        n_keypoints = int(opts.get("n_keypoints", 12))
        groups = int(opts.get("n_groups", max(2, n_images // 2)))
        size = int(opts.get("image_size", 128))
        rng = np.random.default_rng(int(opts.get("seed", 0)))
        absent = [int(k) for k in opts.get("absent_keypoints", [])]

        rows = []
        for i in range(n_images):
            image_id = f"{self.dataset}/img{i:04d}"
            cx, cy = rng.uniform(0.35, 0.65, size=2) * size
            scale = rng.uniform(0.16, 0.26) * size
            if n_keypoints == len(_TEMPLATE_42):
                base = np.asarray(_TEMPLATE_42, dtype=float)
            else:
                angles = np.linspace(0, 2 * np.pi, n_keypoints, endpoint=False)
                base = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            theta = rng.uniform(-0.25, 0.25)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            kpts = (base @ rot.T) * scale + np.array([cx, cy])
            kpts += rng.normal(0, 0.01 * size, kpts.shape)
            kpts = np.clip(kpts, 1.0, size - 2.0)
            vis = np.full(n_keypoints, 2, dtype=int)
            vis[rng.random(n_keypoints) < 0.1] = 1
            # ADR-0016 : certains keypoints n'existent pas chez tous les ordres.
            for k in absent:
                vis[k] = 0
            x0, y0 = kpts.min(axis=0)
            x1, y1 = kpts.max(axis=0)
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": f"raw/{self.dataset}/images/img{i:04d}.png",
                    "image_width": size,
                    "image_height": size,
                    "instance_id": f"{image_id}#0",
                    "group_id": f"{self.dataset}/grp{i % groups:03d}",
                    "bbox_xywh": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "kpts_xy": [float(v) for v in kpts.reshape(-1)],
                    "kpts_vis": [int(v) for v in vis],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "keypoint_schema": str(opts.get("keypoint_schema", self.dataset)),
                    "split_source": "unknown",
                }
            )
        df = pd.DataFrame(rows)
        if bool(opts.get("write_images", False)):
            self._write_images(df, Path(opts["images_root"]), size)
        return df

    @staticmethod
    def _write_images(df: pd.DataFrame, root: Path, size: int) -> None:
        """Effet de bord : ecrit des PNG gris sous `root` (fixtures de test uniquement)."""
        from PIL import Image

        for path in df["image_path"]:
            out = root / str(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (size, size), (32, 32, 32)).save(out)
