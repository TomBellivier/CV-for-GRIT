"""Tests de l'approche yolo_pooled (CONVENTIONS.md §9.1, §11).

La couche Ultralytics n'est pas testable sans GPU ni dependance lourde. En revanche
TOUTE la logique risquee — conversion de coordonnees, format des labels, symetrie —
l'est par aller-retour, et c'est elle qui casse silencieusement en production.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from insectpose.contracts import ContractError
from insectpose.data.datamodule import ImageSet, load_annotations
from insectpose.data.keypoints import load_schema
from insectpose.data.yolo_export import (
    export_split,
    flat_name,
    parse_label_line,
    to_label_lines,
    write_data_yaml,
)
from insectpose.registry import APPROACHES

SCHEMA = "insect42_v1"


@pytest.fixture()
def image_set(project) -> ImageSet:
    annotations = load_annotations(["coleoptera", "diptera"], project)
    schema = load_schema(SCHEMA, project.configs)
    return ImageSet(name="train", annotations=annotations, paths=project, schemas={SCHEMA: schema})


# --- aller-retour de coordonnees --------------------------------------------
def test_label_roundtrip_preserves_geometry(image_set, project) -> None:
    """Le bug classique du format YOLO : bbox CENTREE vs coin haut-gauche."""
    schema = load_schema(SCHEMA, project.configs)
    row = image_set.annotations.iloc[[0]]
    width = int(row["image_width"].iloc[0])
    height = int(row["image_height"].iloc[0])

    lines, clipped = to_label_lines(row, width, height, schema.n_keypoints)
    assert clipped == 0
    parsed = parse_label_line(lines[0], width, height)

    assert np.allclose(parsed["bbox_xywh"], row["bbox_xywh"].iloc[0], atol=1e-3)
    assert np.allclose(parsed["kpts_xy"], row["kpts_xy"].iloc[0], atol=1e-3)
    assert parsed["kpts_vis"] == list(row["kpts_vis"].iloc[0])
    assert parsed["class"] == 0   # une seule classe : insecte


def test_unannotated_keypoints_are_written_as_masked(project) -> None:
    """Un point absent doit sortir en vis=0, jamais en zero appris."""
    import pandas as pd

    schema = load_schema(SCHEMA, project.configs)
    k = schema.n_keypoints
    kpts = [float(v) for v in np.linspace(10, 90, 2 * k)]
    vis = [2] * k
    vis[7] = 0
    row = pd.DataFrame([{
        "instance_id": "x#0", "bbox_xywh": [10.0, 10.0, 80.0, 80.0],
        "kpts_xy": kpts, "kpts_vis": vis,
    }])
    parsed = parse_label_line(to_label_lines(row, 100, 100, k)[0][0], 100, 100)
    assert parsed["kpts_vis"][7] == 0
    assert parsed["kpts_xy"][14:16] == [0.0, 0.0]
    assert parsed["kpts_vis"][6] == 2


def test_out_of_image_coordinates_are_clipped_and_counted(project) -> None:
    import pandas as pd

    schema = load_schema(SCHEMA, project.configs)
    k = schema.n_keypoints
    kpts = [150.0] * (2 * k)   # tous les points hors image
    row = pd.DataFrame([{
        "instance_id": "x#0", "bbox_xywh": [0.0, 0.0, 100.0, 100.0],
        "kpts_xy": kpts, "kpts_vis": [2] * k,
    }])
    _, clipped = to_label_lines(row, 100, 100, k)
    assert clipped > 0


def test_wrong_keypoint_count_is_refused(project) -> None:  # noqa: ARG001
    import pandas as pd

    row = pd.DataFrame([{
        "instance_id": "x#0", "bbox_xywh": [0.0, 0.0, 10.0, 10.0],
        "kpts_xy": [1.0, 2.0], "kpts_vis": [2],
    }])
    with pytest.raises(ContractError, match="keypoints pour un schema"):
        to_label_lines(row, 100, 100, 42)


# --- arborescence exportee ---------------------------------------------------
def test_export_creates_one_label_per_image(image_set, project, tmp_path) -> None:
    schema = load_schema(SCHEMA, project.configs)
    root = tmp_path / "yolo"
    n = export_split(image_set, schema, root, "train")
    labels = sorted((root / "labels" / "train").glob("*.txt"))
    images = sorted((root / "images" / "train").glob("*"))
    assert len(labels) == len(images) == n
    assert all(f.read_text().strip() for f in labels)


def test_filenames_do_not_collide_across_datasets(image_set) -> None:
    """Deux datasets ont des `img000.png` : sans aplatissement, ils s'ecrasent."""
    ids = list(image_set.images["image_id"])
    assert len({flat_name(i) for i in ids}) == len(ids)
    assert flat_name("coleoptera/img000") == "coleoptera__img000"


def test_data_yaml_carries_flip_index(project, tmp_path) -> None:
    """Sans flip_idx, l'augmentation par miroir apprend une anatomie fausse."""
    schema = load_schema(SCHEMA, project.configs)
    path = write_data_yaml(tmp_path, schema, {"train": "train", "val": "val"})
    payload = yaml.safe_load(path.read_text())
    assert payload["kpt_shape"] == [42, 3]
    assert payload["flip_idx"] == list(schema.flip_index)
    assert payload["names"] == {0: "insect"}
    left = schema.index("left-eye")
    assert payload["flip_idx"][left] == schema.index("right-eye")


# --- approche ----------------------------------------------------------------
def test_approach_is_registered_and_declares_availability() -> None:
    cls = APPROACHES.get("yolo_pooled")
    available, reason = cls.availability()
    assert isinstance(available, bool)
    assert available or any(dep in reason for dep in ("torch", "ultralytics"))


def test_search_space_is_declared(cfg) -> None:
    """L'espace de recherche doit etre lisible en config, pas enterre dans le code."""
    import optuna
    from omegaconf import OmegaConf

    OmegaConf.update(cfg, "approach.name", "yolo_pooled", force_add=True)
    from insectpose.cli import load_config

    yolo_cfg = load_config([f"paths.root={cfg.paths.root}", "approach=yolo_pooled"])
    overrides = APPROACHES.get("yolo_pooled").search_space(optuna.create_study().ask(), yolo_cfg)
    assert "approach.lr0" in overrides
    assert 1e-5 <= overrides["approach.lr0"] <= 1e-1


def test_single_detection_per_image_is_configured(project) -> None:
    """ADR-0017 : une image = un insecte, donc max_det = 1."""
    from insectpose.cli import load_config

    yolo_cfg = load_config([f"paths.root={project.root}", "approach=yolo_pooled"])
    assert int(yolo_cfg.approach.max_det) == 1
    assert float(yolo_cfg.approach.conf) <= 0.01   # pas de troncature des courbes AP


def test_heterogeneous_schemas_are_refused(project) -> None:
    """yolo_pooled suppose un schema unique : sinon il faut passer par l'espace union."""
    from types import SimpleNamespace

    from insectpose.approaches.yolo_pooled import YoloPooledApproach

    schema = load_schema(SCHEMA, project.configs)
    other = load_schema(SCHEMA, project.configs)
    source = SimpleNamespace(schemas={"a": schema, "b": other})
    with pytest.raises(ValueError, match="schema de keypoints unique"):
        YoloPooledApproach._schema(source)
