"""Tests des approches B (yolo_per_dataset) et C (detect_then_pose).

Comme pour l'approche A, la couche Ultralytics est remplacee par un double : on
verifie ce que NOUS controlons — routage entre modeles, isolement des artefacts,
recadrage, retro-projection — sans GPU ni poids a telecharger.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.data.crop_export import crop_label_line, expand_bbox
from insectpose.data.datamodule import load_annotations
from insectpose.data.keypoints import load_schema
from insectpose.utils.geometry import apply_affine, crop_affine, invert_affine
from insectpose.utils.io import read_json, read_parquet

SCHEMA = "insect42_v1"
DATASETS = ("coleoptera", "diptera")


# ===========================================================================
# Approche B : un modele par dataset
# ===========================================================================
@pytest.fixture()
def cfg_b(config_factory):
    return config_factory(["approach=yolo_per_dataset", "train.device=cpu"])


def test_each_dataset_gets_its_own_model(fake_ultralytics, cfg_b, project) -> None:  # noqa: ARG001
    """B entraine N modeles, isoles les uns des autres dans le meme run."""
    pipeline.cmd_split(cfg_b)
    ctx = pipeline.cmd_train(cfg_b)

    trainings = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert len(trainings) == len(DATASETS)
    for dataset in DATASETS:
        assert (project.run_dir(ctx.run_id) / "weights" / dataset / "best.pt").exists()
        assert (project.run_dir(ctx.run_id) / "yolo_dataset" / dataset / "data.yaml").exists()


def test_base_weights_not_pooled_model(fake_ultralytics, cfg_b, project) -> None:  # noqa: ARG001
    """ADR-0023 : chaque modele repart des poids de base, jamais du modele poule."""
    pipeline.cmd_split(cfg_b)
    ctx = pipeline.cmd_train(cfg_b)
    manifest = read_json(project.manifest(ctx.run_id))
    for dataset in DATASETS:
        assert manifest[f"{dataset}_base_weights"] == str(cfg_b.approach.weights)


def test_predictions_are_routed_by_dataset(fake_ultralytics, cfg_b, project) -> None:  # noqa: ARG001
    """Chaque image doit etre predite par le modele de SON dataset."""
    pipeline.cmd_split(cfg_b)
    ctx = pipeline.cmd_train(cfg_b)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    annotations = load_annotations(list(DATASETS), project)
    expected = annotations.set_index("image_id")["dataset"]
    assert (predictions["dataset"].to_numpy()
            == predictions["image_id"].map(expected).to_numpy()).all()
    assert set(predictions["dataset"]) <= set(DATASETS)


def test_shared_search_space_keeps_budget_equal(cfg_b) -> None:
    """ADR-0023 : un trial entraine les N modeles, donc le budget egale celui de A."""
    import optuna

    from insectpose.registry import APPROACHES

    overrides = APPROACHES.get("yolo_per_dataset").search_space(
        optuna.create_study().ask(), cfg_b)
    assert overrides, "espace de recherche vide"
    # Aucun hyperparametre n'est declare par dataset : ils sont partages
    assert not any(dataset in key for key in overrides for dataset in DATASETS)


def test_folds_are_reused_not_regenerated(cfg_b, project) -> None:
    """§6.2 : B utilise les MEMES folds que A, simplement restreints."""
    from insectpose.data.splits import fold_assignment, make_split_id

    pipeline.cmd_split(cfg_b)
    table = read_parquet(project.split_file(make_split_id(cfg_b)))
    assignment = fold_assignment(table, 0)

    ctx, data, _ = pipeline._prepare_run(cfg_b)
    subset = data.filter_dataset("coleoptera")
    assert set(subset.test.image_ids) <= set(assignment.test)
    assert all(i.startswith("coleoptera/") for i in subset.test.image_ids)
    assert len(subset.train) + len(subset.val) + len(subset.test) < len(data.train) + \
        len(data.val) + len(data.test)


# ===========================================================================
# Approche C : detection puis pose sur crop
# ===========================================================================
@pytest.fixture()
def cfg_c(config_factory):
    return config_factory([
        "approach=detect_then_pose", "train.device=cpu", "approach.crop.size=[128,128]",
    ])


def test_two_models_are_trained(fake_ultralytics, cfg_c, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(cfg_c)
    ctx = pipeline.cmd_train(cfg_c)

    trainings = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert len(trainings) == 2
    assert (project.run_dir(ctx.run_id) / "weights" / "detector" / "best.pt").exists()
    assert (project.run_dir(ctx.run_id) / "weights" / "pose" / "best.pt").exists()


def test_detector_labels_carry_no_keypoints(fake_ultralytics, cfg_c, project) -> None:  # noqa: ARG001
    """Le detecteur n'a que faire des keypoints : ses labels sont 'classe cx cy w h'."""
    import yaml

    pipeline.cmd_split(cfg_c)
    ctx = pipeline.cmd_train(cfg_c)
    root = project.run_dir(ctx.run_id) / "yolo_dataset" / "detector"
    label = sorted((root / "labels" / "train").glob("*.txt"))[0]
    assert len(label.read_text().split()) == 5

    payload = yaml.safe_load((root / "data.yaml").read_text())
    assert "kpt_shape" not in payload and "flip_idx" not in payload


def test_pose_crops_use_jittered_boxes_in_train_only(fake_ultralytics, cfg_c, project) -> None:  # noqa: ARG001
    """§9.3 : cadrages bruites a l'entrainement, nets en validation."""
    from PIL import Image

    pipeline.cmd_split(cfg_c)
    ctx = pipeline.cmd_train(cfg_c)
    crops = project.run_dir(ctx.run_id) / "crops"
    assert (crops / "data.yaml").exists()
    for split in ("train", "val"):
        images = sorted((crops / "images" / split).glob("*.jpg"))
        labels = sorted((crops / "labels" / split).glob("*.txt"))
        assert images and len(images) == len(labels)
        with Image.open(images[0]) as handle:
            assert handle.size == (128, 128)
    # Un label de crop contient une seule instance, avec ses 42 keypoints
    line = sorted((crops / "labels" / "train").glob("*.txt"))[0].read_text().split()
    assert len(line) == 5 + 42 * 3


def test_predictions_are_back_projected_to_the_source_image(
    fake_ultralytics, cfg_c, project  # noqa: ARG001
) -> None:
    """Le contrat 3 impose le repere de l'image d'origine, pas celui du crop (§3.4)."""
    pipeline.cmd_split(cfg_c)
    ctx = pipeline.cmd_train(cfg_c)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    assert len(predictions) > 0
    assert predictions["bbox_source"].unique().tolist() == ["predicted"]

    sizes = load_annotations(list(DATASETS), project).set_index("image_id")
    for row in predictions.itertuples(index=False):
        width = int(sizes.loc[row.image_id, "image_width"])
        height = int(sizes.loc[row.image_id, "image_height"])
        points = np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2)
        # Les crops font 128 px : sans retro-projection, tous les points seraient < 128
        assert points[:, 0].max() <= 1.5 * width
        assert points[:, 1].max() <= 1.5 * height
        assert len(points) == 42


def test_gt_box_mode_is_flagged_as_diagnostic(fake_ultralytics, cfg_c, project) -> None:  # noqa: ARG001
    """§9.3 : la pose sur bboxes GT est un diagnostic, jamais un resultat bout-en-bout."""
    OmegaConf.update(cfg_c, "approach.pose_on_gt_boxes", True)
    OmegaConf.update(cfg_c, "tag", "diagnostic")
    pipeline.cmd_split(cfg_c)
    ctx = pipeline.cmd_train(cfg_c)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold))
    assert predictions["bbox_source"].unique().tolist() == ["gt"]


# ===========================================================================
# Geometrie du recadrage (testable sans Ultralytics)
# ===========================================================================
def test_expand_bbox_keeps_centre() -> None:
    box = expand_bbox(np.array([10.0, 20.0, 40.0, 60.0]), 0.5)
    assert box[0] + box[2] / 2 == pytest.approx(30.0)
    assert box[1] + box[3] / 2 == pytest.approx(50.0)
    assert box[2] == pytest.approx(60.0)


def test_crop_label_roundtrip(project) -> None:
    """Les keypoints du label doivent revenir a leur position d'origine."""
    schema = load_schema(SCHEMA, project.configs)
    rng = np.random.default_rng(0)
    kpts = rng.uniform(40, 160, size=(schema.n_keypoints, 2))
    vis = np.full(schema.n_keypoints, 2)
    matrix = crop_affine(np.array([30.0, 30.0, 150.0, 150.0]), (128, 128))

    line = crop_label_line(kpts, vis, matrix, (128, 128))
    values = np.asarray([float(v) for v in line.split()[5:]], dtype=float).reshape(-1, 3)
    back = apply_affine(invert_affine(matrix), values[:, :2] * np.array([128, 128]))
    inside = values[:, 2] > 0
    assert inside.any()
    assert np.allclose(back[inside], kpts[inside], atol=0.5)


def test_keypoints_outside_the_crop_are_masked(project) -> None:
    """Un point hors cadre est marque non supervise : ni appris a zero, ni compte faux."""
    schema = load_schema(SCHEMA, project.configs)
    kpts = np.full((schema.n_keypoints, 2), 60.0)
    kpts[0] = [5000.0, 5000.0]          # tres loin du crop
    vis = np.full(schema.n_keypoints, 2)
    matrix = crop_affine(np.array([40.0, 40.0, 40.0, 40.0]), (128, 128))

    values = np.asarray(
        [float(v) for v in crop_label_line(kpts, vis, matrix, (128, 128)).split()[5:]]
    ).reshape(-1, 3)
    assert values[0, 2] == 0
    assert values[0, 0] == 0.0 and values[0, 1] == 0.0
    assert values[1, 2] == 2