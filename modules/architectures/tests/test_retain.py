"""Export vers retained_models/pose/ : un ensemble par commande, jamais un melange."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from insectpose.paths import ProjectPaths
from insectpose.retain import (
    clear_retained,
    retain_context,
    retain_existing_run,
    write_ensemble_card,
)
from insectpose.utils.io import read_json, write_json


def _paths(tmp_path: Path) -> ProjectPaths:
    paths = dataclasses.replace(ProjectPaths.default(tmp_path), retained=tmp_path / "retained")
    paths.ensure_writable_dirs()
    return paths


def _ctx(paths: ProjectPaths, run_id: str, approach: str = "yolo_pooled",
         mode: str = "full", extra: dict | None = None) -> SimpleNamespace:
    """Run complet minimal : un manifeste et un best.pt."""
    run_dir = paths.run_dir(run_id)
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"poids")
    write_json(paths.manifest(run_id), {"approach": approach, "mode": mode, "config": {}})
    cfg = OmegaConf.create({
        "mode": mode, "approach": {"name": approach},
        "retain": {"enabled": True, "approaches": ["yolo_pooled"], "name": None,
                   "hpo_trials": False},
        "eval": {"primary_metric": "oks_ap"},
    })
    return SimpleNamespace(run_id=run_id, cfg=cfg, paths=paths, extra=extra or {})


def _members(paths: ProjectPaths) -> list[str]:
    return sorted(p.name for p in (paths.retained / "pose").iterdir() if p.is_dir())


def test_train_replaces_the_ensemble(tmp_path) -> None:
    paths = _paths(tmp_path)
    retain_context(_ctx(paths, "run_a"))
    retain_context(_ctx(paths, "run_b"))
    assert _members(paths) == ["run_b"]


def test_tune_folds_accumulate(tmp_path) -> None:
    paths = _paths(tmp_path)
    retain_context(_ctx(paths, "ancien"))
    for fold in range(3):
        retain_context(_ctx(paths, f"fold{fold}"), replace=fold == 0)
    assert _members(paths) == ["fold0", "fold1", "fold2"]
    card = read_json(write_ensemble_card(paths, {"source": "tune"}))
    assert card["n_members"] == 3 and card["members"] == ["fold0", "fold1", "fold2"]


def test_excluded_runs_leave_the_ensemble_intact(tmp_path) -> None:
    """Un run non exportable ne doit pas vider l'ensemble en place."""
    paths = _paths(tmp_path)
    retain_context(_ctx(paths, "livre"))
    assert retain_context(_ctx(paths, "lora_run", approach="lora")) is None
    assert retain_context(_ctx(paths, "smoke_run", mode="smoke")) is None
    assert retain_context(_ctx(paths, "trial", extra={"role_in_protocol": "hpo_trial"})) is None
    assert _members(paths) == ["livre"]


def test_evaluate_replaces_the_ensemble_with_the_same_guards(tmp_path) -> None:
    """`evaluate run_id=...` lit l'approche et le mode dans le manifeste du run."""
    paths = _paths(tmp_path)
    retain_context(_ctx(paths, "livre"))
    cfg = _ctx(paths, "yolo_ancien").cfg
    _ctx(paths, "lora_ancien", approach="lora")
    _ctx(paths, "smoke_ancien", mode="smoke")
    assert retain_existing_run("lora_ancien", paths, cfg) is None
    assert retain_existing_run("smoke_ancien", paths, cfg) is None
    assert _members(paths) == ["livre"]
    assert retain_existing_run("yolo_ancien", paths, cfg) is not None
    assert _members(paths) == ["yolo_ancien"]


def test_clear_keeps_hidden_files(tmp_path) -> None:
    paths = _paths(tmp_path)
    pose = paths.retained / "pose"
    (pose / "vieux").mkdir(parents=True)
    (pose / ".gitkeep").write_text("")
    clear_retained(paths)
    assert [p.name for p in pose.iterdir()] == [".gitkeep"]
