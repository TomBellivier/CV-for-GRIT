"""Tests of the keypoint coverage (ADR-0016).

The schema is shared by the 4 orders, but not every point exists everywhere. This
artefact makes the information visible BEFORE training, rather than afterwards while
reading an incomprehensible per-keypoint PCK.
"""

from __future__ import annotations

import pytest

from insectpose.data.coverage import keypoint_coverage, measurement_coverage, summarize
from insectpose.data.datamodule import load_annotations
from insectpose.data.keypoints import load_schema
from insectpose.data.measurements import load_measurements

SCHEMA = "insect42_v1"


@pytest.fixture()
def coverage(project):
    annotations = load_annotations(["coleoptera", "diptera"], project)
    schemas = {SCHEMA: load_schema(SCHEMA, project.configs)}
    return annotations, schemas, keypoint_coverage(annotations, schemas)


def test_absent_keypoints_are_detected_per_dataset(coverage) -> None:
    _, _, table = coverage
    absent = table[(table["dataset"] == "diptera") & (table["status"] == "absent")]
    assert set(absent["keypoint_index"]) == set(range(26, 34))
    assert absent["keypoint"].str.contains("hindwing").all()
    # The same point is present in the other dataset: it is indeed a LOCAL absence
    other = table[(table["dataset"] == "coleoptera") & (table["keypoint_index"] == 26)]
    assert other["status"].iloc[0] == "present"


def test_summary_separates_local_and_global_absence(coverage) -> None:
    _, _, table = coverage
    summary = summarize(table)
    assert summary["absent_everywhere"] == []   # no point missing from BOTH datasets
    assert len(summary["absent_by_dataset"]["diptera"]) == 8
    assert "thorax-left" in summary["present_everywhere"]


def test_rates_are_consistent(coverage) -> None:
    _, _, table = coverage
    assert ((table["rate"] >= 0) & (table["rate"] <= 1)).all()
    assert (table["rate_visible"] <= table["rate"] + 1e-9).all()
    assert (table["n_annotated"] <= table["n_instances"]).all()


def test_measurement_coverage_flags_unusable_measurements(coverage, project) -> None:
    annotations, schemas, _ = coverage
    spec = load_measurements()
    table = measurement_coverage(annotations, schemas, spec)
    diptera = table[table["dataset"] == "diptera"].set_index("measurement")
    # The hind wing measurements cannot be computed for lack of annotated points
    assert not diptera.loc["left hind wing length", "usable"]
    assert diptera.loc["thorax width", "usable"]


def test_prepare_writes_coverage_artifacts(cfg, project, raw_coco) -> None:  # noqa: ARG001
    from omegaconf import OmegaConf

    from insectpose import pipeline
    from insectpose.utils.io import read_json, read_parquet

    # The fixture writes COCO: the adapter is forced, because the local repository may
    # have switched `data.adapter` to another format (yolo, for instance).
    OmegaConf.update(cfg, "data.adapter", "coco")
    OmegaConf.update(cfg, "data.adapter_options.annotations_glob", "*.json")
    pipeline.cmd_prepare(cfg)
    table = read_parquet(project.processed / "coverage_keypoints.parquet")
    assert len(table) == 42 * 2
    summary = read_json(project.processed / "coverage_summary.json")
    assert "absent_by_dataset" in summary
    assert (project.processed / "coverage_measurements.parquet").exists()


def test_absent_keypoints_are_excluded_from_metrics(cfg, project) -> None:
    """An unannotated point must never count as a zero error."""
    from insectpose import pipeline
    from insectpose.utils.io import read_parquet

    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    metrics = read_parquet(project.metrics(ctx.run_id))
    scopes = set(metrics["scope"])
    assert "keypoint:diptera:left-hindwing-tip" not in scopes
    assert "keypoint:coleoptera:left-hindwing-tip" in scopes