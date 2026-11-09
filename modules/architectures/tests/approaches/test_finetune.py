"""`finetune` verb, end to end through the Ultralytics double.

Checks what the verb adds on top of `train`: finding the base model by variant_hash,
reading the Label Studio exports of finetune-images/, the 10 folds, the before/after
runs, and that the main annotations are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from insectpose.finetune import cmd_finetune, find_base_model
from insectpose.contracts import ContractError
from insectpose.data.keypoints import load_schema

N_IMAGES = 12
SIZE = 192


def _write_card(folder: Path, variant: str, fold: int) -> None:
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"weights")
    (folder / "model_card.json").write_text(
        json.dumps({"run_id": folder.name, "variant_hash": variant, "fold": fold}),
        encoding="utf-8")


def _finetune_images(root: Path, names: list[str]) -> Path:
    """finetune-images/coleoptera/{images/, export.json} in the Label Studio format."""
    from PIL import Image

    order_dir = root / "finetune-images" / "coleoptera"
    (order_dir / "images").mkdir(parents=True)
    tasks = []
    for i in range(N_IMAGES):
        name = f"ft_{i:02d}.png"
        Image.new("RGB", (SIZE, SIZE), (i * 10, 80, 120)).save(order_dir / "images" / name)
        results = [
            {"type": "keypointlabels", "original_width": SIZE, "original_height": SIZE,
             "value": {"keypointlabels": [point], "x": 20 + k * 1.2, "y": 30 + (k % 7) * 5}}
            for k, point in enumerate(names)
        ]
        tasks.append({"id": i, "data": {"img": f"/data/local-files/?d=images/{name}"},
                      "annotations": [{"id": i, "result": results}]})
    (order_dir / "export.json").write_text(json.dumps(tasks), encoding="utf-8")
    return root / "finetune-images"


@pytest.fixture()
def retained(tmp_path: Path) -> Path:
    pose = tmp_path / "retained" / "pose"
    _write_card(pose / "run_fold0", "abc123", 0)
    _write_card(pose / "run_fold1", "abc123", 1)
    _write_card(pose / "lonely", "solo99", 3)
    return pose


def test_base_model_is_fold0_or_the_only_member(retained: Path) -> None:
    weights, card = find_base_model(retained, "abc123")
    assert card["fold"] == 0 and weights.parent.name == "run_fold0"
    _, card = find_base_model(retained, "solo99")
    assert card["fold"] == 3
    with pytest.raises(ContractError):
        find_base_model(retained, "unknown")


def test_finetune_runs_10_folds_and_compares(fake_ultralytics, config_factory, project,
                                             retained, tmp_path) -> None:
    names = list(load_schema("insect42_v1", project.configs).names)
    images_dir = _finetune_images(tmp_path, names)
    main_annotations = project.annotations("coleoptera").read_bytes()

    cfg = config_factory([
        "approach=yolo_pooled", "train.device=cpu",
        f"paths.retained={retained.parent}",
        "finetune.base=abc123", "finetune.epochs=1",
        f"finetune.images_dir={images_dir.as_posix()}",
    ])
    out_dir = cmd_finetune(cfg)

    per_fold = pd.read_csv(out_dir / "comparison_per_fold.csv")
    assert sorted(per_fold["fold"].unique()) == list(range(10))
    assert {"before", "after", "delta"} <= set(per_fold.columns)
    summary = pd.read_csv(out_dir / "comparison.csv")
    assert {"detection_rate", "n_images_detected"} <= set(summary["metric"])
    # Point metrics only on the images detected before AND after (none with the double,
    # which never matches the GT); every metric on every image in comparison_all_images.
    scopes = per_fold.groupby("metric")["evaluated_on"].first()
    assert scopes["detection_rate"] == "all_test_images"
    points = per_fold[per_fold["evaluated_on"] == "detected_before_and_after"]
    detected = per_fold[per_fold["metric"] == "n_images_detected"]
    assert points.empty == (detected[["before", "after"]].min(axis=1) == 0).all()
    raw = pd.read_csv(out_dir / "comparison_all_images.csv")
    assert "oks_ap" in set(raw["metric"])
    # One before | after figure per annotated image (each is in exactly one test fold).
    assert len(list((out_dir / "figures").glob("fold*__*.png"))) == N_IMAGES
    # One fine-tuning per fold, all from the base model, for the epochs asked.
    trains = [c for c in fake_ultralytics.calls if c["kind"] == "train"]
    assert len(trains) == 10 and all(c["epochs"] == 1 for c in trains)
    # The main datasets are never overwritten.
    assert project.annotations("coleoptera").read_bytes() == main_annotations
