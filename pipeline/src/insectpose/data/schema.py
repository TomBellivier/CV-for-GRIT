"""Validation des contrats de donnees (CONVENTIONS.md §3, §10).

Echoue tot, bruyamment, avec un message actionnable. Aucun filtrage, aucune
correction automatique : valider n'est pas nettoyer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from insectpose.contracts import (
    BBOX_SOURCES,
    DATASETS,
    ROLES,
    SCHEMA_VERSIONS,
    SCHEMAS,
    ContractError,
    required_columns,
)

_LIST_KINDS = {"list_float", "list_int"}


def validate_frame(df: pd.DataFrame, artifact: str) -> None:
    """Valide un DataFrame contre un contrat ('annotations', 'splits', ...).

    Leve ContractError au premier probleme structurel.
    """
    if artifact not in SCHEMAS:
        raise ContractError(f"Artefact inconnu : {artifact}. Connus : {sorted(SCHEMAS)}")

    missing = [c for c in required_columns(artifact) if c not in df.columns]
    if missing:
        raise ContractError(
            f"[{artifact}] colonnes obligatoires manquantes : {missing}. "
            f"Attendu : {required_columns(artifact)}"
        )
    if df.empty:
        return

    expected_version = SCHEMA_VERSIONS[artifact]
    versions = set(pd.unique(df["schema_version"]))
    if versions != {expected_version}:
        raise ContractError(
            f"[{artifact}] schema_version={versions}, attendu {expected_version}. "
            "Un artefact d'une autre version doit passer par un lecteur dedie."
        )

    for spec in SCHEMAS[artifact]:
        if spec.name not in df.columns:
            continue
        col = df[spec.name]
        if col.isna().any() and spec.required:
            raise ContractError(f"[{artifact}] valeurs nulles interdites dans '{spec.name}'.")
        if spec.kind in _LIST_KINDS:
            _check_list_column(artifact, spec.name, col)

    _validate_vocabulary(df, artifact)
    _validate_geometry(df, artifact)


def _check_list_column(artifact: str, name: str, col: pd.Series) -> None:
    """Verifie qu'une colonne de listes contient bien des sequences numeriques."""
    sample = col.iloc[0]
    if not isinstance(sample, (list, tuple, np.ndarray)):
        raise ContractError(
            f"[{artifact}] '{name}' doit contenir des listes, trouve {type(sample).__name__}."
        )


def _validate_vocabulary(df: pd.DataFrame, artifact: str) -> None:
    """Verifie les colonnes a vocabulaire ferme."""
    if "dataset" in df.columns:
        unknown = set(df["dataset"].unique()) - set(DATASETS)
        if unknown:
            raise ContractError(
                f"[{artifact}] datasets inconnus : {sorted(unknown)}. "
                f"Vocabulaire ferme : {list(DATASETS)} (cf. contracts.DATASETS)."
            )
    if artifact == "splits":
        unknown_roles = set(df["role"].unique()) - set(ROLES)
        if unknown_roles:
            raise ContractError(f"[splits] roles inconnus : {sorted(unknown_roles)}")
    if artifact == "predictions":
        unknown_src = set(df["bbox_source"].unique()) - set(BBOX_SOURCES)
        if unknown_src:
            raise ContractError(f"[predictions] bbox_source inconnus : {sorted(unknown_src)}")
        if df["pred_id"].duplicated().any():
            dup = df.loc[df["pred_id"].duplicated(), "pred_id"].head(3).tolist()
            raise ContractError(f"[predictions] pred_id non uniques, ex. {dup}")
    if artifact == "annotations" and df["instance_id"].duplicated().any():
        dup = df.loc[df["instance_id"].duplicated(), "instance_id"].head(3).tolist()
        raise ContractError(f"[annotations] instance_id non uniques, ex. {dup}")


def _validate_geometry(df: pd.DataFrame, artifact: str) -> None:
    """Verifie la coherence bbox / keypoints / visibilite."""
    if artifact not in ("annotations", "predictions"):
        return
    bad_bbox = df["bbox_xywh"].map(lambda b: len(b) != 4)
    if bad_bbox.any():
        raise ContractError(f"[{artifact}] bbox_xywh doit contenir 4 valeurs (xywh).")

    n_kpts = df["kpts_xy"].map(len)
    if (n_kpts % 2 != 0).any():
        raise ContractError(f"[{artifact}] kpts_xy doit contenir 2K valeurs.")
    k = (n_kpts // 2).astype(int)

    second = "kpts_vis" if artifact == "annotations" else "kpts_score"
    n_second = df[second].map(len).astype(int)
    if not (n_second == k).all():
        raise ContractError(
            f"[{artifact}] longueur de '{second}' incoherente avec kpts_xy "
            f"(K deduit={sorted(set(k))[:3]}, trouve={sorted(set(n_second))[:3]})."
        )

    per_schema = df.groupby("keypoint_schema")["kpts_xy"].apply(lambda s: {len(v) for v in s})
    for schema_name, sizes in per_schema.items():
        if len(sizes) > 1:
            raise ContractError(
                f"[{artifact}] le schema '{schema_name}' apparait avec plusieurs tailles "
                f"de keypoints : {sorted(sizes)}. L'ordre et le nombre de points sont figes."
            )


def validate_single_instance(df: pd.DataFrame) -> None:
    """Verifie l'hypothese "une image = un insecte" (ADR-0017).

    Si elle est violee, la detection top-1 des approches devient fausse en silence :
    l'echec doit donc etre bloquant, au moment de la preparation des donnees.
    """
    counts = df.groupby("image_id").size()
    offenders = counts[counts > 1]
    if len(offenders):
        raise ContractError(
            f"{len(offenders)} image(s) contiennent plusieurs instances "
            f"(ex. {offenders.index[0]} : {int(offenders.iloc[0])}), alors que "
            "data.single_instance_per_image=true (ADR-0017). Corriger les annotations "
            "ou passer ce drapeau a false et revoir les approches a detection top-1."
        )


def validate_coordinates_in_image(df: pd.DataFrame, tolerance: float = 0.05) -> pd.Series:
    """Signale (sans supprimer) les instances dont les coordonnees sortent de l'image.

    Retourne une Series de drapeaux ; le filtrage reste une decision de config (§3.2).
    """
    flags = []
    for row in df.itertuples(index=False):
        w, h = float(row.image_width), float(row.image_height)
        pts = np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2)
        vis = np.asarray(row.kpts_vis) > 0
        issues: list[str] = []
        if vis.any():
            p = pts[vis]
            if (p[:, 0] < -tolerance * w).any() or (p[:, 0] > (1 + tolerance) * w).any():
                issues.append("kpt_x_out_of_image")
            if (p[:, 1] < -tolerance * h).any() or (p[:, 1] > (1 + tolerance) * h).any():
                issues.append("kpt_y_out_of_image")
        else:
            issues.append("no_visible_keypoint")
        bw, bh = float(row.bbox_xywh[2]), float(row.bbox_xywh[3])
        if bw <= 0 or bh <= 0:
            issues.append("degenerate_bbox")
        flags.append(";".join(issues))
    return pd.Series(flags, index=df.index, dtype="object")


def ensure_columns(df: pd.DataFrame, artifact: str, extra: dict[str, Any] | None = None
                   ) -> pd.DataFrame:
    """Complete les colonnes optionnelles manquantes et ordonne selon le contrat."""
    out = df.copy()
    if extra:
        for key, value in extra.items():
            out[key] = value
    out["schema_version"] = SCHEMA_VERSIONS[artifact]
    for spec in SCHEMAS[artifact]:
        if spec.name not in out.columns and not spec.required:
            out[spec.name] = "" if spec.kind == "str" else np.nan
    ordered: Sequence[str] = [c.name for c in SCHEMAS[artifact] if c.name in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[[*ordered, *rest]]
