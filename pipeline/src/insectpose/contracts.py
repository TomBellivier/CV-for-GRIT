"""Contrats de donnees figes (CONVENTIONS.md §3).

Ce module est l'API du projet : approches, evaluateur et reporting ne se parlent
qu'a travers ces schemas. Toute modification passe par un increment de
`*_SCHEMA_VERSION` et un lecteur retrocompatible, jamais par une edition en place.

Aucun effet de bord : ce module ne lit ni n'ecrit aucun fichier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- versions de schema -----------------------------------------------------
ANNOTATION_SCHEMA_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

# --- vocabulaire ferme ------------------------------------------------------
DATASETS: tuple[str, ...] = ("coleoptera", "diptera", "hymenoptera", "lepidoptera")
ROLES: tuple[str, ...] = ("train", "val", "test")
BBOX_SOURCES: tuple[str, ...] = ("predicted", "gt", "derived")

VIS_ABSENT = 0
VIS_OCCLUDED = 1
VIS_VISIBLE = 2

ColumnKind = Literal["str", "int", "float", "bool", "list_float", "list_int"]


class ContractError(ValueError):
    """Violation d'un contrat de donnees. Toujours bloquante, jamais rattrapee."""


@dataclass(frozen=True)
class ColumnSpec:
    """Description d'une colonne d'un artefact parquet."""

    name: str
    kind: ColumnKind
    required: bool = True
    description: str = ""


# --- Contrat 1 : annotations canoniques (§3.2) ------------------------------
ANNOTATION_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version du contrat 1"),
    ColumnSpec("dataset", "str", True, "coleoptera | diptera | hymenoptera | lepidoptera"),
    ColumnSpec("image_id", "str", True, "identifiant global : <dataset>/<nom_sans_ext>"),
    ColumnSpec("image_path", "str", True, "chemin RELATIF a paths.data, jamais absolu"),
    ColumnSpec("image_width", "int", True, "pixels, image d'origine"),
    ColumnSpec("image_height", "int", True, "pixels, image d'origine"),
    ColumnSpec("instance_id", "str", True, "<image_id>#<n>"),
    ColumnSpec("group_id", "str", True, "cle anti-fuite : specimen / planche / session"),
    ColumnSpec("bbox_xywh", "list_float", True, "4 valeurs, pixels absolus, image d'origine"),
    ColumnSpec("kpts_xy", "list_float", True, "2K valeurs, pixels absolus, image d'origine"),
    ColumnSpec("kpts_vis", "list_int", True, "K valeurs : 0 absent / 1 occulte / 2 visible"),
    ColumnSpec("area", "float", True, "aire de reference de l'instance"),
    ColumnSpec("keypoint_schema", "str", True, "nom du schema de keypoints (§3.1)"),
    ColumnSpec("split_source", "str", False, "train | test_officiel | unknown"),
    ColumnSpec("qc_flags", "str", False, "anomalies detectees ; jamais un filtre"),
)

# --- Contrat 2 : splits (§3.3) ----------------------------------------------
SPLIT_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version du contrat 2"),
    ColumnSpec("split_id", "str", True, "decoupage partage par TOUTES les approches"),
    ColumnSpec("image_id", "str", True, ""),
    ColumnSpec("dataset", "str", True, ""),
    ColumnSpec("group_id", "str", True, "unite de decoupage effective"),
    ColumnSpec("fold", "int", True, "index du fold externe"),
    ColumnSpec("role", "str", True, "train | val | test"),
)

# --- Contrat 3 : predictions (§3.4) -----------------------------------------
PREDICTION_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version du contrat 3"),
    ColumnSpec("run_id", "str", True, ""),
    ColumnSpec("fold", "int", True, ""),
    ColumnSpec("split", "str", True, "train | val | test"),
    ColumnSpec("dataset", "str", True, ""),
    ColumnSpec("image_id", "str", True, ""),
    ColumnSpec("pred_id", "str", True, "unique dans le fichier"),
    ColumnSpec("bbox_xywh", "list_float", True, "repere image d'origine, pixels absolus"),
    ColumnSpec("bbox_score", "float", True, "1.0 si non applicable"),
    ColumnSpec("kpts_xy", "list_float", True, "2K, repere image d'origine, SCHEMA LOCAL"),
    ColumnSpec("kpts_score", "list_float", True, "K"),
    ColumnSpec("keypoint_schema", "str", True, "doit correspondre au dataset de l'image"),
    ColumnSpec("bbox_source", "str", True, "predicted | gt | derived"),
    ColumnSpec("inference_ms", "float", False, "temps par instance"),
)

# --- Contrat 4 : metriques (§3.5) -------------------------------------------
METRIC_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version du contrat 4"),
    ColumnSpec("run_id", "str", True, ""),
    ColumnSpec("approach", "str", True, ""),
    ColumnSpec("fold", "int", True, ""),
    ColumnSpec("split", "str", True, ""),
    ColumnSpec("scope", "str", True, "overall | dataset:<nom> | keypoint:<dataset>:<nom>"),
    ColumnSpec("metric", "str", True, "nom canonique, ex. pck@0.05_bboxdiag"),
    ColumnSpec("value", "float", True, ""),
    ColumnSpec("n", "int", True, "taille de l'echantillon sous-jacent (§7.4)"),
)

SCHEMAS: dict[str, tuple[ColumnSpec, ...]] = {
    "annotations": ANNOTATION_COLUMNS,
    "splits": SPLIT_COLUMNS,
    "predictions": PREDICTION_COLUMNS,
    "metrics": METRIC_COLUMNS,
}

SCHEMA_VERSIONS: dict[str, int] = {
    "annotations": ANNOTATION_SCHEMA_VERSION,
    "splits": SPLIT_SCHEMA_VERSION,
    "predictions": PREDICTION_SCHEMA_VERSION,
    "metrics": METRIC_SCHEMA_VERSION,
}


def required_columns(artifact: str) -> list[str]:
    """Noms des colonnes obligatoires d'un artefact ('annotations', 'splits', ...)."""
    return [c.name for c in SCHEMAS[artifact] if c.required]


def all_columns(artifact: str) -> list[str]:
    """Noms de toutes les colonnes (obligatoires et optionnelles) d'un artefact."""
    return [c.name for c in SCHEMAS[artifact]]
