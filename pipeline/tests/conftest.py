"""Fixtures partagees : mini-corpus synthetique et projet jetable.

Le smoke test tourne de bout en bout sur ces fixtures en quelques secondes (§10.3).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from insectpose.data.adapters.synthetic import SyntheticAdapter
from insectpose.paths import ProjectPaths
from insectpose.registry import load_all_plugins

# Les plugins sont charges A L'IMPORT : la parametrisation de tests/test_smoke.py
# lit le registre au moment de la COLLECTE pytest, avant toute fixture.
load_all_plugins()

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["coleoptera", "diptera"]
SCHEMA = "insect42_v1"   # ADR-0006 : schema commun aux 4 datasets
N_KPTS = 42
# ADR-0016 : le corpus de test reproduit l'absence de certains points selon l'ordre
# (ici : pas d'ailes posterieures annotees chez les "diptera" du corpus jouet).
ABSENT_KEYPOINTS = {"coleoptera": [], "diptera": list(range(26, 34))}


@pytest.fixture()
def project(tmp_path: Path) -> ProjectPaths:
    """Projet jetable : configs reelles copiees, donnees synthetiques."""
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    paths = ProjectPaths.default(tmp_path)
    paths.ensure_writable_dirs()
    for dataset in DATASETS:
        adapter = SyntheticAdapter(
            dataset=dataset,
            source_dir=paths.raw_dir(dataset),
            options={
                "n_images": 24, "n_keypoints": N_KPTS, "n_groups": 8, "seed": 7,
                "keypoint_schema": SCHEMA, "image_size": 192,
                "absent_keypoints": ABSENT_KEYPOINTS[dataset],
                # Images reelles : l'export qualitatif (§8.4) fait partie du smoke test.
                "write_images": True, "images_root": str(paths.data),
            },
        )
        adapter.run(paths)
    return paths


@pytest.fixture()
def raw_coco(project: ProjectPaths) -> ProjectPaths:
    """Ajoute des annotations COCO brutes, pour tester la chaine `prepare` complete."""
    import json

    import numpy as np

    from insectpose.utils.io import read_parquet

    for dataset in DATASETS:
        annotations = read_parquet(project.annotations(dataset))
        images, anns = [], []
        for i, row in enumerate(annotations.itertuples(index=False)):
            images.append({"id": i, "file_name": f"{Path(row.image_path).stem}.png",
                           "width": int(row.image_width), "height": int(row.image_height)})
            kpts = np.asarray(row.kpts_xy, float).reshape(-1, 2)
            vis = np.asarray(row.kpts_vis, int).reshape(-1, 1)
            anns.append({"id": i, "image_id": i,
                         "keypoints": np.hstack([kpts, vis]).reshape(-1).tolist(),
                         "bbox": [float(v) for v in row.bbox_xywh], "area": float(row.area)})
        (project.raw_dir(dataset) / "annotations.json").write_text(
            json.dumps({"images": images, "annotations": anns}), encoding="utf-8"
        )
    return project


@pytest.fixture()
def cfg(project: ProjectPaths) -> DictConfig:
    """Config complete pointant vers le projet jetable."""
    from insectpose.cli import load_config

    overrides = [
        f"paths.root={project.root}",
        "data=pooled",
        f"data.datasets=[{','.join(DATASETS)}]",
        "cv=kfold5_grouped",
        "cv.n_folds=3",
        "mode=smoke",
        "tag=test",
        "train.epochs=1",
    ]
    config = load_config(overrides, config_dir=project.configs)
    OmegaConf.set_struct(config, False)
    return config
