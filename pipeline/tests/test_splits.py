"""Tests des folds : anti-fuite, reproductibilite, invalidation (§3.3, §6.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from insectpose.contracts import ContractError
from insectpose.data.datamodule import load_annotations
from insectpose.data.splits import build_splits, fold_assignment, load_splits, write_splits


def _annotations(project):
    return load_annotations(["coleoptera", "diptera"], project)


def test_no_group_leaks_between_roles(cfg, project) -> None:
    ann = _annotations(project)
    table, _ = build_splits(ann, cfg)
    for fold in table["fold"].unique():
        sub = table[table["fold"] == fold]
        roles = sub["role"].unique()
        groups = {r: set(sub.loc[sub["role"] == r, "group_id"]) for r in roles}
        assert not (groups.get("train", set()) & groups.get("test", set()))
        assert not (groups.get("val", set()) & groups.get("test", set()))
        assert not (groups.get("train", set()) & groups.get("val", set()))


def test_every_image_appears_once_per_fold(cfg, project) -> None:
    table, _ = build_splits(_annotations(project), cfg)
    counts = table.groupby(["fold", "image_id"]).size()
    assert (counts == 1).all()


def test_splits_are_deterministic(cfg, project) -> None:
    ann = _annotations(project)
    a, _ = build_splits(ann, cfg)
    b, _ = build_splits(ann, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_modified_annotations_invalidate_splits(cfg, project) -> None:
    ann = _annotations(project)
    table, meta = build_splits(ann, cfg)
    write_splits(table, meta, project)
    tampered = ann.iloc[:-1]
    with pytest.raises(ContractError, match="annotations differentes"):
        load_splits(meta["split_id"], project, tampered)


def test_fold_assignment_checks_disjunction(cfg, project) -> None:
    table, _ = build_splits(_annotations(project), cfg)
    assignment = fold_assignment(table, 0)
    assert len(assignment.test) > 0
    assignment.check_disjoint()
