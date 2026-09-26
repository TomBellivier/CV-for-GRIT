"""Contract tests: a non-compliant artefact must never be written (§10.1)."""

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
    with pytest.raises(ContractError, match="missing mandatory columns"):
        validate_frame(df, "annotations")


def test_keypoint_length_mismatch_is_rejected(project) -> None:
    df = read_parquet(project.annotations("coleoptera"))
    df.at[0, "kpts_vis"] = [1, 1, 1]   # inconsistent with kpts_xy
    with pytest.raises(ContractError, match="inconsistent"):
        validate_frame(df, "annotations")


def test_unknown_dataset_is_rejected(project) -> None:
    df = read_parquet(project.annotations("coleoptera"))
    df["dataset"] = "orthoptera"
    with pytest.raises(ContractError, match="unknown datasets"):
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
    """ADR-0017: one image = one insect. Otherwise the top-1 detection lies silently."""
    from insectpose.data.schema import validate_single_instance

    df = read_parquet(project.annotations("coleoptera"))
    duplicated = pd.concat([df, df.head(1).assign(instance_id="duplicate")], ignore_index=True)
    with pytest.raises(ContractError, match="several instances"):
        validate_single_instance(duplicated)


def test_single_instance_dataset_passes(project) -> None:
    from insectpose.data.schema import validate_single_instance

    validate_single_instance(read_parquet(project.annotations("coleoptera")))


def test_keypoint_count_mismatch_names_the_offending_files(project) -> None:
    """A message that does not name the culprits forces a manual investigation."""
    df = read_parquet(project.annotations("coleoptera"))
    df.at[0, "kpts_xy"] = [0.0] * 256          # 128 keypoints instead of 42
    with pytest.raises(ContractError) as excinfo:
        validate_frame(df, "annotations")

    message = str(excinfo.value)
    assert "42 points" in message
    assert "1 diverging instance(s)" in message
    assert str(df.at[0, "image_path"]) in message
    assert "128 keypoints" in message