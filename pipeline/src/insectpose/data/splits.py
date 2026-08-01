"""Generation et lecture des folds partages (CONVENTIONS.md §3.3, §6.1, §6.2).

Les folds sont generes UNE FOIS et utilises par toutes les approches. Une approche
qui fabrique son propre decoupage rend toute comparaison invalide.
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
    """Identifiant canonique d'un decoupage, derive de la config CV."""
    if cfg.get("split_id"):
        return str(cfg.split_id)
    cv = cfg.cv
    return f"{cv.name}_seed{int(cv.seed)}_{cfg.data.scope}"


def _image_level(annotations: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Table au niveau image : une image = une unite de decoupage."""
    if group_by not in annotations.columns:
        raise ContractError(f"Colonne de groupement '{group_by}' absente des annotations.")
    per_image = (
        annotations.groupby("image_id")
        .agg(dataset=("dataset", "first"), group_id=(group_by, "first"),
             n_groups=(group_by, "nunique"), n_instances=("instance_id", "size"))
        .reset_index()
    )
    ambiguous = per_image.loc[per_image["n_groups"] > 1, "image_id"]
    if len(ambiguous):
        raise ContractError(
            f"{len(ambiguous)} images ont plusieurs '{group_by}' (ex. {ambiguous.iloc[0]}). "
            "Le groupe doit etre constant par image, sinon l'anti-fuite ne tient pas."
        )
    return per_image.drop(columns=["n_groups"])


def _carve_val(train_images: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    """Extrait un sous-ensemble val du train, par groupe et stratifie par dataset."""
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
    """Construit la table des folds + ses metadonnees. Aucun effet de bord.

    Retourne (table conforme au contrat 2, metadonnees a serialiser).
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
        raise ContractError(f"Strategie de decoupage inconnue : {strategy}")

    n_groups = images["group_id"].nunique()
    k = int(cv.n_folds) if strategy == "stratified_group_kfold" else max(
        2, int(round(1.0 / float(cv.test_fraction)))
    )
    if n_groups < k:
        raise ContractError(
            f"{n_groups} groupes pour {k} folds : impossible de decouper sans fuite. "
            "Reduire cv.n_folds ou revoir group_id (DECISION OPEN-04)."
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
            "group_id == image_id : une image = un specimen (ADR-0011). Si un dataset "
            "apporte un jour plusieurs vues par specimen, renseigner "
            "data.adapter_options.group_id_field, sinon il y aura fuite."
        )
    return table, meta


def write_splits(table: pd.DataFrame, meta: dict[str, Any], paths: ProjectPaths) -> Path:
    """Ecrit contrat 2 + metadonnees. Effet de bord : data/splits/<split_id>.{parquet,json}."""
    split_id = meta["split_id"]
    out = write_parquet(paths.split_file(split_id), table, artifact="splits")
    write_json(paths.split_meta(split_id), meta)
    return out


def load_splits(split_id: str, paths: ProjectPaths, annotations: pd.DataFrame | None = None
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge un decoupage et REFUSE de le servir si les annotations ont change (§3.3)."""
    table = read_parquet(paths.split_file(split_id), artifact="splits", validate=True)
    meta = read_json(paths.split_meta(split_id))
    if annotations is not None:
        current = content_hash_annotations(annotations)
        if current != meta.get("content_hash"):
            raise ContractError(
                f"Le decoupage '{split_id}' a ete genere sur des annotations differentes "
                f"(hash {meta.get('content_hash')} != {current}). Regenerer les splits, ou "
                "les resultats ne seront pas comparables."
            )
    return table, meta


def inner_split_id(split_id: str, outer_fold: int) -> str:
    """Identifiant du decoupage INTERNE associe a un fold externe."""
    return f"{split_id}__outer{outer_fold}"


def build_inner_splits(annotations: pd.DataFrame, outer_table: pd.DataFrame, outer_fold: int,
                       cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decoupage interne d'un fold externe, pour l'HPO niche (ADR-0012).

    Construit uniquement a partir des images train+val du fold externe : le test
    externe n'entre JAMAIS dans la recherche d'hyperparametres. Aucun effet de bord.
    """
    from omegaconf import OmegaConf

    outer = outer_table[outer_table["fold"] == outer_fold]
    if outer.empty:
        raise ContractError(f"Fold externe {outer_fold} absent du decoupage parent.")
    inner_images = set(outer.loc[outer["role"].isin(["train", "val"]), "image_id"])
    subset = annotations[annotations["image_id"].isin(inner_images)]
    if subset.empty:
        raise ContractError(f"Aucune image d'entrainement dans le fold externe {outer_fold}.")

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
        # Le hash porte sur les annotations COMPLETES : toute modification des donnees
        # invalide aussi les decoupages internes.
        "content_hash": content_hash_annotations(annotations),
        "subset_content_hash": content_hash_annotations(subset),
    })
    return table, meta


@dataclass(frozen=True)
class FoldAssignment:
    """Identifiants d'images d'un fold, par role."""

    split_id: str
    fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def check_disjoint(self) -> None:
        """Verifie qu'aucune image n'apparait dans deux roles (invariant teste)."""
        s = [set(self.train), set(self.val), set(self.test)]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            overlap = s[i] & s[j]
            if overlap:
                raise ContractError(
                    f"Fuite detectee dans {self.split_id} fold {self.fold} : "
                    f"{len(overlap)} images partagees (ex. {sorted(overlap)[:2]})."
                )


def fold_assignment(table: pd.DataFrame, fold: int) -> FoldAssignment:
    """Extrait les listes d'images d'un fold et verifie leur disjonction."""
    sub = table[table["fold"] == fold]
    if sub.empty:
        known = sorted(table["fold"].unique())
        raise ContractError(f"Fold {fold} absent du decoupage (folds disponibles : {known}).")
    assignment = FoldAssignment(
        split_id=str(sub["split_id"].iloc[0]),
        fold=fold,
        train=tuple(sub.loc[sub["role"] == "train", "image_id"]),
        val=tuple(sub.loc[sub["role"] == "val", "image_id"]),
        test=tuple(sub.loc[sub["role"] == "test", "image_id"]),
    )
    assignment.check_disjoint()
    return assignment
