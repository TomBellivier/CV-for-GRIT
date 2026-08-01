"""Acces aux donnees d'un fold (CONVENTIONS.md §4.3).

`FoldData` est ce que recoit `Approach.fit`. Il expose le SUPERSET des champs utiles
a toutes les approches (dataset_index pour BatchNorm-par-groupe, transform_matrix pour
la retro-projection...) : les ajouter au coup par coup casserait la modularite.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.contracts import DATASETS, ContractError
from insectpose.data.keypoints import KeypointSchema
from insectpose.data.splits import FoldAssignment
from insectpose.paths import ProjectPaths
from insectpose.utils.io import read_parquet


@dataclass(frozen=True)
class ImageSet:
    """Sous-ensemble d'images + leurs annotations, dans un role donne (train/val/test)."""

    name: str
    annotations: pd.DataFrame
    paths: ProjectPaths
    schemas: dict[str, KeypointSchema]

    @cached_property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(pd.unique(self.annotations["image_id"]))

    @cached_property
    def images(self) -> pd.DataFrame:
        """Table au niveau image (une ligne par image)."""
        return (
            self.annotations.groupby("image_id", as_index=False)
            .agg(dataset=("dataset", "first"), image_path=("image_path", "first"),
                 image_width=("image_width", "first"), image_height=("image_height", "first"),
                 keypoint_schema=("keypoint_schema", "first"), n_instances=("instance_id", "size"))
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def n_instances(self) -> int:
        return len(self.annotations)

    def absolute_path(self, image_path: str) -> Path:
        """Chemin absolu d'une image (les artefacts ne stockent que du relatif)."""
        return self.paths.data / image_path

    def schema_for(self, schema_name: str) -> KeypointSchema:
        """Schema de keypoints par son nom (colonne `keypoint_schema` des artefacts)."""
        if schema_name not in self.schemas:
            raise ContractError(
                f"Schema '{schema_name}' non charge. Charges : {sorted(self.schemas)}."
            )
        return self.schemas[schema_name]

    def instances_array(self) -> dict[str, np.ndarray]:
        """Vue tableau des instances : kpts (N,K,2), vis (N,K), bbox (N,4).

        Valable uniquement si toutes les instances partagent le meme schema.
        """
        schemas = set(self.annotations["keypoint_schema"])
        if len(schemas) > 1:
            raise ContractError(
                f"instances_array() exige un schema unique, trouve {sorted(schemas)}. "
                "Passer par l'espace union pour un traitement multi-datasets."
            )
        def stack(column: str, dtype: type) -> np.ndarray:
            values = self.annotations[column].map(lambda v: np.asarray(v, dtype))
            return np.stack(values.to_numpy())

        return {
            "kpts": stack("kpts_xy", float).reshape(len(self.annotations), -1, 2),
            "vis": stack("kpts_vis", int),
            "bbox": stack("bbox_xywh", float),
        }


@dataclass(frozen=True)
class FoldData:
    """Les trois roles d'un fold. `fit` NE DOIT JAMAIS toucher `test` (§4.2)."""

    split_id: str
    fold: int
    train: ImageSet
    val: ImageSet
    test: ImageSet
    schemas: dict[str, KeypointSchema]

    def role(self, name: str) -> ImageSet:
        """Acces par nom de role."""
        if name not in ("train", "val", "test"):
            raise ContractError(f"Role inconnu : {name}")
        return getattr(self, name)  # type: ignore[no-any-return]

    def summary(self) -> dict[str, Any]:
        """Comptages, a journaliser au demarrage de chaque run."""
        return {
            r: {"images": len(self.role(r)), "instances": self.role(r).n_instances}
            for r in ("train", "val", "test")
        }


def dataset_index(dataset: str) -> int:
    """Index stable d'un dataset. Utilise par les approches a BatchNorm par groupe (§9.5)."""
    return DATASETS.index(dataset)


def load_annotations(datasets: list[str], paths: ProjectPaths) -> pd.DataFrame:
    """Charge et concatene les annotations canoniques (contrat 1) de plusieurs datasets."""
    frames = []
    for name in datasets:
        path = paths.annotations(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Annotations canoniques absentes pour '{name}' ({path}). "
                "Lancer d'abord : python -m insectpose.cli prepare data=" + name
            )
        frames.append(read_parquet(path, artifact="annotations", validate=True))
    return pd.concat(frames, ignore_index=True)


def build_fold_data(annotations: pd.DataFrame, assignment: FoldAssignment,
                    schemas: dict[str, KeypointSchema], paths: ProjectPaths) -> FoldData:
    """Assemble un FoldData a partir des annotations et d'une assignation de fold."""

    def subset(name: str, ids: tuple[str, ...]) -> ImageSet:
        sub = annotations[annotations["image_id"].isin(ids)].reset_index(drop=True)
        return ImageSet(name=name, annotations=sub, paths=paths, schemas=schemas)

    return FoldData(
        split_id=assignment.split_id,
        fold=assignment.fold,
        train=subset("train", assignment.train),
        val=subset("val", assignment.val),
        test=subset("test", assignment.test),
        schemas=schemas,
    )
