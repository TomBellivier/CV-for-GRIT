"""Adaptateur annotation_data.csv -> format canonique (contrat 1, CONVENTIONS.md §3.2).

Lit la table d'annotation unique du depot,
``annotation_data/annotation_data.csv`` (voir
``annotation_tools/build_annotation_data.py``) : une ligne par image, les
keypoints en pixels absolus dans des colonnes ``<point>_x`` / ``<point>_y`` /
``<point>_v``, et la colonne ``group`` pour l'ordre d'insecte.

Les colonnes de scale et de validite des mesures de cette table ne concernent
pas la pose : elles sont ignorees ici, sans etre modifiees.

Comme tout adaptateur : lit, convertit, ne filtre pas et ne decide rien. Les
lignes sans aucun keypoint sont les seules ecartees -- ce ne sont pas des
annotations de pose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.adapters.base import BaseAdapter
from insectpose.data.keypoints import load_schema
from insectpose.registry import register_adapter

_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


@register_adapter("annotation_csv")
class AnnotationCsvAdapter(BaseAdapter):
    """Table d'annotation du depot -> annotations canoniques d'un dataset.

    Options (``data.adapter_options``) :
        csv_path        chemin du CSV, absolu ou relatif a ``source_dir``
                        (defaut : ``annotation_data.csv``)
        images_subdir   dossier des images, relatif a ``paths.data``, ou les
                        noms de fichiers du CSV sont cherches
                        (defaut : ``raw/<dataset>/images``)
        group_column    colonne portant l'ordre d'insecte (defaut : ``group``)
        margin          marge, en pixels, autour des keypoints pour la bbox
                        quand le CSV n'en porte pas (defaut : 10)
        keypoint_schema nom du schema (defaut : ``insect42_v1``)
    """

    def _schema_points(self) -> list[str]:
        """Ordre des keypoints du schema (§3.1) : il fixe l'ordre des colonnes lues.

        L'ordre vient du schema, jamais de l'ordre des colonnes du CSV : un index
        de keypoint est encode dans tous les artefacts produits ensuite.
        """
        configs_dir = self.options.get("configs_dir")
        if not configs_dir:
            raise KeyError(
                "adapter_options.configs_dir manquant : ajouter "
                "'configs_dir: ${paths.configs}' dans le fichier configs/data/ du dataset."
            )
        name = str(self.options.get("keypoint_schema", "insect42_v1"))
        return list(load_schema(name, Path(configs_dir)).names)

    def read(self) -> pd.DataFrame:
        csv_path = Path(self.options.get("csv_path", "annotation_data.csv"))
        if not csv_path.is_absolute():
            # Relatif a la racine du projet (injectee par cmd_prepare), pas au
            # repertoire courant : la commande doit marcher d'ou qu'on la lance.
            csv_path = (Path(self.options.get("project_root", self.source_dir)) / csv_path).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Table d'annotation absente : {csv_path}. La construire avec "
                "python annotation_tools/build_annotation_data.py"
            )

        frame = pd.read_csv(csv_path)
        points = self._schema_points()
        missing = [p for p in points if f"{p}_x" not in frame.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} keypoint(s) du schema absent(s) de {csv_path.name} "
                f"(ex. {missing[:3]}) : la table ne suit pas le meme squelette."
            )

        group_column = str(self.options.get("group_column", "group"))
        if group_column in frame.columns:
            frame = frame[frame[group_column].astype(str).str.lower() == self.dataset]
        images_subdir = str(self.options.get("images_subdir", f"raw/{self.dataset}/images"))
        margin = float(self.options.get("margin", 10))
        schema_name = str(self.options.get("keypoint_schema", "insect42_v1"))

        rows: list[dict[str, Any]] = []
        for _, record in frame.iterrows():
            xy = np.array([[record[f"{p}_x"], record[f"{p}_y"]] for p in points], dtype=float)
            vis = np.array([record.get(f"{p}_v", 0) for p in points], dtype=float)
            vis = np.nan_to_num(vis, nan=0.0).astype(int)
            # Un point non annote sort en (0, 0, 0) : c'est la convention du
            # contrat, et elle evite qu'une coordonnee manquante deplace la bbox.
            absent = (vis == 0) | ~np.isfinite(xy).all(axis=1)
            xy[absent] = 0.0
            if not (~absent).any():
                continue

            width, height = record.get("width"), record.get("height")
            if not (np.isfinite(width) and np.isfinite(height)):
                raise ValueError(
                    f"Taille d'image absente pour {record['image_name']} : "
                    "la colonne width/height du CSV est vide."
                )
            width, height = int(width), int(height)

            box = self._bbox(record, xy[~absent], margin, width, height)
            stem = Path(str(record["image_name"])).stem
            image_id = f"{self.dataset}/{stem}"
            rows.append({
                "dataset": self.dataset,
                "image_id": image_id,
                "image_path": f"{images_subdir}/{record['image_name']}",
                "image_width": width,
                "image_height": height,
                "instance_id": f"{image_id}#0",     # ADR-0017 : une image = un insecte
                "group_id": image_id,
                "bbox_xywh": [float(v) for v in box],
                "kpts_xy": [float(v) for v in xy.reshape(-1)],
                "kpts_vis": [int(v) for v in vis],
                "area": float(box[2] * box[3]),
                "keypoint_schema": schema_name,
                "split_source": "unknown",
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _bbox(record: pd.Series, visible_xy: np.ndarray, margin: float,
              width: int, height: int) -> tuple[float, float, float, float]:
        """bbox du CSV si elle y est, sinon l'enveloppe des keypoints + marge."""
        values = [record.get(c) for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]
        if all(v is not None and np.isfinite(v) for v in values) and values[2] > 0:
            return tuple(float(v) for v in values)

        x_min, y_min = visible_xy.min(axis=0)
        x_max, y_max = visible_xy.max(axis=0)
        x = max(0.0, x_min - margin)
        y = max(0.0, y_min - margin)
        return (x, y, min(width, x_max + margin) - x, min(height, y_max + margin) - y)
