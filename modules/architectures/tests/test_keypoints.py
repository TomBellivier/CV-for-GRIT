"""Tests of the insect42_v1 keypoint schema (§3.1, ADR-0006/0007)."""

from __future__ import annotations

import numpy as np
import pytest

from insectpose.contracts import ContractError
from insectpose.data.keypoints import (
    build_union_mapping,
    load_schema,
    load_schemas,
    local_to_union,
    union_mask,
    union_to_local,
)

SCHEMA = "insect42_v1"


def test_schema_shape_and_status(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    assert schema.n_keypoints == 42
    assert not schema.is_placeholder
    assert schema.sigma_source == "difficulty"


def test_sigmas_derive_from_difficulty(project) -> None:
    """sigma = difficulty * scale (ADR-0007): 10 -> 0.025, 40 -> 0.100."""
    schema = load_schema(SCHEMA, project.configs)
    assert np.allclose(schema.sigmas, schema.difficulty * 0.0025)
    assert schema.sigmas.min() == pytest.approx(0.025)
    assert schema.sigmas.max() == pytest.approx(0.100)
    # A hard point must have a LARGER tolerance than an easy point
    assert schema.sigmas[schema.index("left-forewing-front")] > schema.sigmas[schema.index("neck")]


def test_flip_pairs_are_symmetric(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    for i, j in enumerate(schema.flip_index):
        assert schema.flip_index[j] == i
    assert schema.names[schema.flip_index[schema.index("left-eye")]] == "right-eye"
    # The points of the median axis are their own mirror
    for axial in ("head-top", "neck", "thorax-bottom", "body-tip"):
        assert schema.flip_index[schema.index(axial)] == schema.index(axial)


def test_skeleton_indices_are_in_range(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    for a, b in schema.skeleton:
        assert 0 <= a < schema.n_keypoints and 0 <= b < schema.n_keypoints


def test_union_mapping_is_identity(project) -> None:
    """ADR-0006: the 4 datasets share this schema, the union mapping is the identity."""
    schema = load_schema(SCHEMA, project.configs)
    mapping = build_union_mapping(schema, schema)
    assert np.array_equal(mapping.local_to_union, np.arange(schema.n_keypoints))
    assert mapping.masked_local == []
    assert union_mask(mapping).all()

    values = np.arange(schema.n_keypoints * 2, dtype=float).reshape(-1, 2)
    assert np.allclose(union_to_local(local_to_union(values, mapping, fill=0.0), mapping), values)


def test_strict_mode_accepts_validated_schema(project) -> None:
    assert load_schemas([SCHEMA], project.configs, strict=True)[SCHEMA].n_keypoints == 42


def test_strict_mode_refuses_placeholder_schema(project) -> None:
    """The safeguard must stay active for any future non-validated schema."""
    (project.configs / "keypoints" / "draft.yaml").write_text(
        "schema_version: 1\nname: draft\nstatus: PLACEHOLDER\n"
        "keypoints:\n  - {name: a, union: null, sigma: 0.05, flip: null}\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="PLACEHOLDER"):
        load_schemas(["draft"], project.configs, strict=True)


def test_keypoint_without_tolerance_is_refused(project) -> None:
    """An OKS without sigma or difficulty makes no sense: explicit failure."""
    (project.configs / "keypoints" / "no_sigma.yaml").write_text(
        "schema_version: 1\nname: no_sigma\nkeypoints:\n  - {name: a, union: null}\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="difficulty"):
        load_schemas(["no_sigma"], project.configs)
