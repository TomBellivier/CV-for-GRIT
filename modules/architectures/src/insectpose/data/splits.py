"""Generation and reading of the shared folds (CONVENTIONS.md §3.3, §6.1, §6.2).

The folds are generated ONCE and used by every approach. An approach that builds its
own split makes any comparison invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from insectpose.contracts import SPLIT_SCHEMA_VERSION, ContractError
from insectpose.paths import ProjectPaths
from insectpose.utils.hashing import content_hash_annotations
from insectpose.utils.io import read_json, read_parquet, write_json, write_parquet
from insectpose.utils.logging import get_logger

log = get_logger("splits")


def make_split_id(cfg: Any) -> str:
    """Canonical identifier of a split, derived from the CV config."""
    if cfg.get("split_id"):
        return str(cfg.split_id)
    cv = cfg.cv
    return f"{cv.name}_seed{int(cv.seed)}_{cfg.data.scope}"


def _image_level(annotations: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Image-level table: one image = one unit of the split."""
    if group_by not in annotations.columns:
        raise ContractError(f"Grouping column '{group_by}' missing from the annotations.")
    per_image = (
        annotations.groupby("image_id")
        .agg(dataset=("dataset", "first"), group_id=(group_by, "first"),
             n_groups=(group_by, "nunique"), n_instances=("instance_id", "size"))
        .reset_index()
    )
    ambiguous = per_image.loc[per_image["n_groups"] > 1, "image_id"]
    if len(ambiguous):
        raise ContractError(
            f"{len(ambiguous)} images have several '{group_by}' (e.g. {ambiguous.iloc[0]}). "
            "The group must be constant per image, otherwise the anti-leakage does not hold."
        )
    return per_image.drop(columns=["n_groups"])


def _carve_val(train_images: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Carve a val subset out of the train, by group and stratified by dataset."""
    if val_fraction <= 0:
        return train_images.assign(role="train")
    n_splits = max(2, int(round(1.0 / val_fraction)))
    n_splits = min(n_splits, train_images["group_id"].nunique())
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(
        splitter.split(train_images, train_images["dataset"], train_images["group_id"])
    )
    roles = np.array(["train"] * len(train_images), dtype=object)
    roles[val_idx] = "val"
    return train_images.assign(role=roles)


def build_splits(annotations: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the fold table + its metadata. No side effect.

    Returns (table complying with contract 2, metadata to serialise).
    """
    cv = cfg.cv
    split_id = make_split_id(cfg)
    images = _image_level(annotations, str(cv.group_by))

    strategy = str(cv.strategy)
    if strategy == "stratified_group_kfold":
        n_folds = int(cv.n_folds)
    elif strategy == "stratified_group_holdout":
        n_folds = 1
    else:
        raise ContractError(f"Unknown split strategy: {strategy}")

    n_groups = images["group_id"].nunique()
    k = int(cv.n_folds) if strategy == "stratified_group_kfold" else max(
        2, int(round(1.0 / float(cv.test_fraction)))
    )
    if n_groups < k:
        raise ContractError(
            f"{n_groups} groups for {k} folds: impossible to split without leakage. "
            "Reduce cv.n_folds or review group_id (DECISION OPEN-04)."
        )

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=int(cv.seed))
    folds = list(splitter.split(images, images["dataset"], images["group_id"]))

    rows: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(folds[:n_folds]):
        test = images.iloc[test_idx].assign(role="test")
        train = _carve_val(
            images.iloc[train_idx].reset_index(drop=True),
            float(cv.val_fraction),
            int(cv.seed) + fold,
        )
        part = pd.concat([train, test], ignore_index=True)
        part["fold"] = fold
        rows.append(part)

    table = pd.concat(rows, ignore_index=True)
    table["split_id"] = split_id
    table["schema_version"] = SPLIT_SCHEMA_VERSION
    table = table[["schema_version", "split_id", "image_id", "dataset", "group_id", "fold", "role"]]

    meta = {
        "split_id": split_id,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": strategy,
        "n_folds": n_folds,
        "group_by": str(cv.group_by),
        "stratify_by": str(cv.stratify_by),
        "val_fraction": float(cv.val_fraction),
        "seed": int(cv.seed),
        "content_hash": content_hash_annotations(annotations),
        "n_images": int(len(images)),
        "n_groups": int(n_groups),
        "n_instances": int(len(annotations)),
        "group_is_image_id": bool((images["group_id"] == images["image_id"]).all()),
        "counts": (
            table.groupby(["fold", "role", "dataset"]).size().rename("n").reset_index()
            .to_dict(orient="records")
        ),
    }
    if meta["group_is_image_id"]:
        log.info(
            "group_id == image_id: one image = one specimen (ADR-0011). If a dataset one "
            "day brings several views per specimen, fill "
            "data.adapter_options.group_id_field, otherwise there will be leakage."
        )
    return table, meta


def write_splits(table: pd.DataFrame, meta: dict[str, Any], paths: ProjectPaths) -> Path:
    """Write contract 2 + metadata. Side effect: data/splits/<split_id>.{parquet,json}."""
    split_id = meta["split_id"]
    out = write_parquet(paths.split_file(split_id), table, artifact="splits")
    write_json(paths.split_meta(split_id), meta)
    return out


def load_splits(split_id: str, paths: ProjectPaths, annotations: pd.DataFrame | None = None
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a split and REFUSE to serve it if the annotations have changed (§3.3)."""
    table = read_parquet(paths.split_file(split_id), artifact="splits", validate=True)
    meta = read_json(paths.split_meta(split_id))
    if annotations is not None:
        current = content_hash_annotations(annotations)
        if current != meta.get("content_hash"):
            raise ContractError(
                f"The split '{split_id}' was generated on different annotations "
                f"(hash {meta.get('content_hash')} != {current}). Regenerate the splits, or "
                "the results will not be comparable."
            )
    return table, meta


def inner_split_id(split_id: str, outer_fold: int) -> str:
    """Identifier of the INNER split attached to an outer fold."""
    return f"{split_id}__outer{outer_fold}"


def build_inner_splits(annotations: pd.DataFrame, outer_table: pd.DataFrame, outer_fold: int,
                       cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inner split of an outer fold, for the nested HPO (ADR-0012).

    Built only from the train+val images of the outer fold: the outer test NEVER enters
    the hyperparameter search. No side effect.
    """
    from omegaconf import OmegaConf

    outer = outer_table[outer_table["fold"] == outer_fold]
    if outer.empty:
        raise ContractError(f"Outer fold {outer_fold} missing from the parent split.")
    inner_images = set(outer.loc[outer["role"].isin(["train", "val"]), "image_id"])
    subset = annotations[annotations["image_id"].isin(inner_images)]
    if subset.empty:
        raise ContractError(f"No training image in outer fold {outer_fold}.")

    inner_cfg = cfg.copy()
    OmegaConf.update(inner_cfg, "cv.n_folds", int(cfg.tuning.inner_folds))
    OmegaConf.update(inner_cfg, "cv.seed", int(cfg.cv.seed) + 1000 + outer_fold)

    table, meta = build_splits(subset, inner_cfg)
    parent = str(outer_table["split_id"].iloc[0])
    identifier = inner_split_id(parent, outer_fold)
    table["split_id"] = identifier
    meta.update({
        "split_id": identifier,
        "parent_split_id": parent,
        "outer_fold": outer_fold,
        "role_in_protocol": "inner",
        # The hash covers the WHOLE annotations: any change of the data also
        # invalidates the inner splits.
        "content_hash": content_hash_annotations(annotations),
        "subset_content_hash": content_hash_annotations(subset),
    })
    return table, meta


def full_split_id(split_id: str) -> str:
    """Identifier of the 'all images' split derived from a parent split."""
    return f"{split_id}__full"


def build_full_split(annotations: pd.DataFrame, cfg: Any, val_fraction: float | None = None
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split of the full-data model: all the images, no test. No side effect.

    A model trained this way sees all the available data; a fold model only sees 4/5 of
    it. This split therefore has no 'test' role: the model coming out of it IS NOT
    EVALUATED (ADR-0012 -- its metrics would be about images seen at training time). The
    performance estimate remains that of the outer folds.

    Only the `val` share is removed from the train: it is only used to choose the
    checkpoint (early stopping), not to measure anything.
    """
    fraction = float(cfg.cv.val_fraction if val_fraction is None else val_fraction)
    if not 0.0 < fraction < 1.0:
        raise ContractError(
            f"val_fraction={fraction}: the final model needs a non-empty val share to "
            "choose its checkpoint, and a non-empty rest."
        )

    images = _image_level(annotations, str(cfg.cv.group_by))
    table = _carve_val(images.reset_index(drop=True), fraction, int(cfg.cv.seed))
    table["fold"] = 0

    parent = make_split_id(cfg)
    identifier = full_split_id(parent)
    table["split_id"] = identifier
    table["schema_version"] = SPLIT_SCHEMA_VERSION
    table = table[["schema_version", "split_id", "image_id", "dataset", "group_id", "fold", "role"]]

    counts = table["role"].value_counts()
    meta = {
        "split_id": identifier,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "strategy": "full",
        "parent_split_id": parent,
        "role_in_protocol": "final_full",
        "n_folds": 1,
        "group_by": str(cfg.cv.group_by),
        "stratify_by": str(cfg.cv.stratify_by),
        "val_fraction": fraction,
        "seed": int(cfg.cv.seed),
        "content_hash": content_hash_annotations(annotations),
        "n_images": int(len(images)),
        "n_groups": int(images["group_id"].nunique()),
        "n_instances": int(len(annotations)),
        "group_is_image_id": bool((images["group_id"] == images["image_id"]).all()),
        "counts": (
            table.groupby(["fold", "role", "dataset"]).size().rename("n").reset_index()
            .to_dict(orient="records")
        ),
    }
    log.info("Final split '%s': %d images (train %d, val %d), no test.",
             identifier, len(table), int(counts.get("train", 0)), int(counts.get("val", 0)))
    return table, meta


@dataclass(frozen=True)
class FoldAssignment:
    """Image identifiers of a fold, per role."""

    split_id: str
    fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def check_disjoint(self) -> None:
        """Check that no image appears in two roles (tested invariant)."""
        s = [set(self.train), set(self.val), set(self.test)]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            overlap = s[i] & s[j]
            if overlap:
                raise ContractError(
                    f"Leakage detected in {self.split_id} fold {self.fold}: "
                    f"{len(overlap)} shared images (e.g. {sorted(overlap)[:2]})."
                )


def fold_assignment(table: pd.DataFrame, fold: int) -> FoldAssignment:
    """Extract the image lists of a fold and check that they are disjoint."""
    sub = table[table["fold"] == fold]
    if sub.empty:
        known = sorted(table["fold"].unique())
        raise ContractError(f"Fold {fold} missing from the split (available folds: {known}).")
    assignment = FoldAssignment(
        split_id=str(sub["split_id"].iloc[0]),
        fold=fold,
        train=tuple(sub.loc[sub["role"] == "train", "image_id"]),
        val=tuple(sub.loc[sub["role"] == "val", "image_id"]),
        test=tuple(sub.loc[sub["role"] == "test", "image_id"]),
    )
    assignment.check_disjoint()
    return assignment
