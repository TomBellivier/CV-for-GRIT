"""Access to the data of a fold (CONVENTIONS.md §4.3).

`FoldData` is what `Approach.fit` receives. It exposes the SUPERSET of the fields
useful to every approach (dataset_index for BatchNorm-per-group, transform_matrix for
the back-projection...): adding them one at a time would break modularity.
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
    """Subset of images + their annotations, in a given role (train/val/test)."""

    name: str
    annotations: pd.DataFrame
    paths: ProjectPaths
    schemas: dict[str, KeypointSchema]

    @cached_property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(pd.unique(self.annotations["image_id"]))

    @cached_property
    def images(self) -> pd.DataFrame:
        """Image-level table (one row per image)."""
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
        """Absolute path of an image (the artefacts only store relative paths)."""
        return self.paths.data / image_path

    def schema_for(self, schema_name: str) -> KeypointSchema:
        """Keypoint schema by its name (`keypoint_schema` column of the artefacts)."""
        if schema_name not in self.schemas:
            raise ContractError(
                f"Schema '{schema_name}' not loaded. Loaded: {sorted(self.schemas)}."
            )
        return self.schemas[schema_name]

    def filter_dataset(self, dataset: str) -> ImageSet:
        """Subset restricted to one dataset, same schemas and same paths.

        Used by the per-dataset approaches (§9.2): they reuse the SAME folds as the
        pooled approaches, simply restricted (§6.2).
        """
        sub = self.annotations[self.annotations["dataset"] == dataset]
        return ImageSet(name=self.name, annotations=sub.reset_index(drop=True),
                        paths=self.paths, schemas=self.schemas)

    def instances_array(self) -> dict[str, np.ndarray]:
        """Array view of the instances: kpts (N,K,2), vis (N,K), bbox (N,4).

        Only valid if every instance shares the same schema.
        """
        schemas = set(self.annotations["keypoint_schema"])
        if len(schemas) > 1:
            raise ContractError(
                f"instances_array() requires a single schema, found {sorted(schemas)}. "
                "Go through the union space for a multi-dataset processing."
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
    """The three roles of a fold. `fit` MUST NEVER touch `test` (§4.2)."""

    split_id: str
    fold: int
    train: ImageSet
    val: ImageSet
    test: ImageSet
    schemas: dict[str, KeypointSchema]

    def role(self, name: str) -> ImageSet:
        """Access by role name."""
        if name not in ("train", "val", "test"):
            raise ContractError(f"Unknown role: {name}")
        return getattr(self, name)  # type: ignore[no-any-return]

    def filter_dataset(self, dataset: str) -> FoldData:
        """Fold restricted to one dataset, without regenerating any split (§6.2)."""
        return FoldData(
            split_id=self.split_id, fold=self.fold,
            train=self.train.filter_dataset(dataset),
            val=self.val.filter_dataset(dataset),
            test=self.test.filter_dataset(dataset),
            schemas=self.schemas,
        )

    def summary(self) -> dict[str, Any]:
        """Counts, to be logged at the start of every run."""
        return {
            r: {"images": len(self.role(r)), "instances": self.role(r).n_instances}
            for r in ("train", "val", "test")
        }


def dataset_index(dataset: str) -> int:
    """Stable index of a dataset. Used by the per-group BatchNorm approaches (§9.5)."""
    return DATASETS.index(dataset)


def load_annotations(datasets: list[str], paths: ProjectPaths) -> pd.DataFrame:
    """Load and concatenate the canonical annotations (contract 1) of several datasets."""
    frames = []
    for name in datasets:
        path = paths.annotations(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical annotations missing for '{name}' ({path}). "
                "Run first: python -m insectpose.cli prepare data=" + name
            )
        frames.append(read_parquet(path, artifact="annotations", validate=True))
    return pd.concat(frames, ignore_index=True)


def build_fold_data(annotations: pd.DataFrame, assignment: FoldAssignment,
                    schemas: dict[str, KeypointSchema], paths: ProjectPaths) -> FoldData:
    """Assemble a FoldData from the annotations and a fold assignment."""

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