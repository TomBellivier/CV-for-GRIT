"""Frozen data contracts (CONVENTIONS.md §3).

This module is the API of the project: approaches, evaluator and reporting only talk
to each other through these schemas. Any change goes through an increment of
`*_SCHEMA_VERSION` and a backward-compatible reader, never through an in-place edit.

No side effect: this module reads and writes no file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --- schema versions -----------------------------------------------------------
ANNOTATION_SCHEMA_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

# --- closed vocabulary ---------------------------------------------------------
DATASETS: tuple[str, ...] = ("coleoptera", "diptera", "hymenoptera", "lepidoptera")
ROLES: tuple[str, ...] = ("train", "val", "test")
BBOX_SOURCES: tuple[str, ...] = ("predicted", "gt", "derived")

VIS_ABSENT = 0
VIS_OCCLUDED = 1
VIS_VISIBLE = 2

ColumnKind = Literal["str", "int", "float", "bool", "list_float", "list_int"]


class ContractError(ValueError):
    """Violation of a data contract. Always blocking, never caught."""


@dataclass(frozen=True)
class ColumnSpec:
    """Description of a column of a parquet artefact."""

    name: str
    kind: ColumnKind
    required: bool = True
    description: str = ""


# --- Contract 1: canonical annotations (§3.2) -------------------------------------
ANNOTATION_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version of contract 1"),
    ColumnSpec("dataset", "str", True, "coleoptera | diptera | hymenoptera | lepidoptera"),
    ColumnSpec("image_id", "str", True, "global identifier: <dataset>/<name_without_ext>"),
    ColumnSpec("image_path", "str", True, "path RELATIVE to paths.data, never absolute"),
    ColumnSpec("image_width", "int", True, "pixels, original image"),
    ColumnSpec("image_height", "int", True, "pixels, original image"),
    ColumnSpec("instance_id", "str", True, "<image_id>#<n>"),
    ColumnSpec("group_id", "str", True, "anti-leakage key: specimen / plate / session"),
    ColumnSpec("bbox_xywh", "list_float", True, "4 values, absolute pixels, original image"),
    ColumnSpec("kpts_xy", "list_float", True, "2K values, absolute pixels, original image"),
    ColumnSpec("kpts_vis", "list_int", True, "K values: 0 absent / 1 occluded / 2 visible"),
    ColumnSpec("area", "float", True, "reference area of the instance"),
    ColumnSpec("keypoint_schema", "str", True, "name of the keypoint schema (§3.1)"),
    ColumnSpec("split_source", "str", False, "train | official_test | unknown"),
    ColumnSpec("qc_flags", "str", False, "anomalies detected; never a filter"),
)

# --- Contract 2: splits (§3.3) -----------------------------------------------------
SPLIT_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version of contract 2"),
    ColumnSpec("split_id", "str", True, "split shared by ALL the approaches"),
    ColumnSpec("image_id", "str", True, ""),
    ColumnSpec("dataset", "str", True, ""),
    ColumnSpec("group_id", "str", True, "actual unit of the split"),
    ColumnSpec("fold", "int", True, "index of the outer fold"),
    ColumnSpec("role", "str", True, "train | val | test"),
)

# --- Contract 3: predictions (§3.4) ------------------------------------------------
PREDICTION_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version of contract 3"),
    ColumnSpec("run_id", "str", True, ""),
    ColumnSpec("fold", "int", True, ""),
    ColumnSpec("split", "str", True, "train | val | test"),
    ColumnSpec("dataset", "str", True, ""),
    ColumnSpec("image_id", "str", True, ""),
    ColumnSpec("pred_id", "str", True, "unique in the file"),
    ColumnSpec("bbox_xywh", "list_float", True, "original image frame, absolute pixels"),
    ColumnSpec("bbox_score", "float", True, "1.0 if not applicable"),
    ColumnSpec("kpts_xy", "list_float", True, "2K, original image frame, LOCAL SCHEMA"),
    ColumnSpec("kpts_score", "list_float", True, "K"),
    ColumnSpec("keypoint_schema", "str", True, "must match the dataset of the image"),
    ColumnSpec("bbox_source", "str", True, "predicted | gt | derived"),
    ColumnSpec("inference_ms", "float", False, "time per instance"),
)

# --- Contract 4: metrics (§3.5) ----------------------------------------------------
METRIC_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("schema_version", "int", True, "version of contract 4"),
    ColumnSpec("run_id", "str", True, ""),
    ColumnSpec("approach", "str", True, ""),
    ColumnSpec("fold", "int", True, ""),
    ColumnSpec("split", "str", True, ""),
    ColumnSpec("scope", "str", True, "overall | dataset:<name> | keypoint:<dataset>:<name>"),
    ColumnSpec("metric", "str", True, "canonical name, e.g. pck@0.05_bboxdiag"),
    ColumnSpec("value", "float", True, ""),
    ColumnSpec("n", "int", True, "size of the underlying sample (§7.4)"),
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
    """Names of the mandatory columns of an artefact ('annotations', 'splits', ...)."""
    return [c.name for c in SCHEMAS[artifact] if c.required]


def all_columns(artifact: str) -> list[str]:
    """Names of every column (mandatory and optional) of an artefact."""
    return [c.name for c in SCHEMAS[artifact]]
