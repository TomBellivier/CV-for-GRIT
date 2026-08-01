"""Tests de contrat : un artefact non conforme ne doit jamais s'ecrire (§10.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from insectpose.contracts import ContractError, required_columns
from insectpose.data.schema import ensure_columns, validate_frame
from insectpose.utils.io import read_parquet


def test_annotations_respect_contract(project) -> None:
    df = read_parquet(project.annotations("coleoptera"), artifact="annotations", validate=True)
    assert set(required_columns("annotations")).issubset(df.columns)
    assert df["instance_id"].is_unique


def test_missing_column_is_rejected() -> None:
    df = pd.DataFrame({"schema_version": [1], "dataset": ["coleoptera"]})
    with pytest.raises(ContractError, match="colonnes obligatoires manquantes"):
        validate_frame(df, "annotations")


def test_keypoint_length_mismatch_is_rejected(project) -> None:
    df = read_parquet(project.annotations("coleoptera"))
    df.at[0, "kpts_vis"] = [1, 1, 1]   # incoherent avec kpts_xy
    with pytest.raises(ContractError, match="incoherente"):
        validate_frame(df, "annotations")


def test_unknown_dataset_is_rejected(project) -> None:
    df = read_parquet(project.annotations("coleoptera"))
    df["dataset"] = "orthoptera"
    with pytest.raises(ContractError, match="datasets inconnus"):
        validate_frame(df, "annotations")


def test_schema_version_mismatch_is_rejected(project) -> None:
    df = read_parquet(project.annotations("coleoptera"))
    df["schema_version"] = 99
    with pytest.raises(ContractError, match="schema_version"):
        validate_frame(df, "annotations")


def test_ensure_columns_fills_optional_fields() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["r"], "fold": [0], "split": ["test"], "dataset": ["diptera"],
            "image_id": ["diptera/x"], "pred_id": ["p0"], "bbox_xywh": [[0.0, 0.0, 1.0, 1.0]],
            "bbox_score": [1.0], "kpts_xy": [[0.0, 0.0]], "kpts_score": [[1.0]],
            "keypoint_schema": ["diptera"], "bbox_source": ["gt"],
        }
    )
    out = ensure_columns(df, "predictions")
    validate_frame(out, "predictions")
    assert "inference_ms" in out.columns


def test_multiple_instances_per_image_are_refused(project) -> None:
    """ADR-0017 : une image = un insecte. Sinon la detection top-1 ment en silence."""
    from insectpose.data.schema import validate_single_instance

    df = read_parquet(project.annotations("coleoptera"))
    duplicated = pd.concat([df, df.head(1).assign(instance_id="doublon")], ignore_index=True)
    with pytest.raises(ContractError, match="plusieurs instances"):
        validate_single_instance(duplicated)


def test_single_instance_dataset_passes(project) -> None:
    from insectpose.data.schema import validate_single_instance

    validate_single_instance(read_parquet(project.annotations("coleoptera")))
