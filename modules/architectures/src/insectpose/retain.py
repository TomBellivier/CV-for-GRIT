"""Export of a run to `retained_models/` (CONVENTIONS.md §1.5, §2, §5.2, §8.2).

A complete run lives in `runs/<run_id>/` and never leaves it: it is the write area of
the approaches. This module does the only thing that takes something out of it, and
to a single place: it COPIES the weights of the run to `retained_models/pose/<name>/`,
with a model card, so that `pipeline/` loads them without knowing anything about this
module (see retained_models/README.md).

`retained_models/pose/` holds AN ENSEMBLE of models, which the pipeline runs and
averages point by point: a single model after `train`, one per outer fold after
`tune`. Every command replaces the previous ensemble (`clear_retained`): two trainings
never mix in it. Only the approaches of `retain.approaches` (a YOLO-pose, a `best.pt`,
the full schema) are exported, since the pipeline loads them all the same way.

Nothing is moved or deleted in `runs/`: the run stays the source of truth, the export
is reproducible (exporting again overwrites the previous copy).

§8.2: only a COMPLETE run (manifest present) can be exported; a run without a manifest
is a broken run, exporting it would spread a model that cannot be audited.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from insectpose.paths import ProjectPaths
from insectpose.utils.io import read_json, write_json
from insectpose.utils.logging import get_logger

log = get_logger("retain")

CARD_NAME = "model_card.json"
ENSEMBLE_NAME = "ensemble.json"


def clear_retained(paths: ProjectPaths, kind: str = "pose") -> None:
    """Empty `retained_models/<kind>/` before writing a new ensemble to it.

    The pipeline runs EVERY model of the folder: a model left by a previous command
    would enter the mean without anything saying so. Hidden files (.gitkeep) are kept.
    """
    folder = paths.retained / kind
    if not folder.is_dir():
        return
    removed = 0
    for child in folder.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    if removed:
        log.info("Previous ensemble removed from %s (%d item(s)).", folder, removed)


def write_ensemble_card(paths: ProjectPaths, card: dict[str, Any], kind: str = "pose") -> Path:
    """Describe the retained ensemble: its members and, after `tune`, the CV estimate."""
    folder = paths.retained / kind
    members = sorted(p.parent.name for p in folder.glob(f"*/{CARD_NAME}"))
    target = folder / ENSEMBLE_NAME
    write_json(target, {"members": members, "n_members": len(members),
                        "written_at": time.time(), **card})
    return target


def _primary_metric(run_id: str, paths: ProjectPaths, cfg: Any) -> dict[str, Any]:
    """Primary metric of the run, for the model card. Never blocking."""
    path = paths.metrics(run_id)
    if not path.exists():
        return {}
    try:
        from insectpose.evaluation.evaluator import primary_value
        from insectpose.utils.io import read_parquet

        return {
            "primary_metric": str(cfg.eval.primary_metric),
            "primary_value": primary_value(read_parquet(path), cfg.eval),
        }
    except Exception as exc:  # noqa: BLE001 - an incomplete card is better than a lost export
        log.warning("Primary metric unreadable for %s: %s", run_id, exc)
        return {}


def _keypoint_names(paths: ProjectPaths, schema_name: str | None) -> list[str]:
    """Order of the keypoints of the schema, so that `pipeline/` can check it."""
    if not schema_name:
        return []
    try:
        from insectpose.data.keypoints import load_schema

        return list(load_schema(str(schema_name), paths.configs).names)
    except Exception as exc:  # noqa: BLE001
        log.warning("Keypoint schema '%s' unreadable: %s", schema_name, exc)
        return []


def _warn_on_overwrite(target: Path, run_id: str) -> None:
    """Warn if the export overwrites the model of ANOTHER run.

    Real case: `tune` retrains one outer fold per fold, hence several runs. With a fixed
    `retain.name`, they all target the same folder and the last fold silently wins.
    Exporting the same run again, on the other hand, is a normal operation.
    """
    card = target / CARD_NAME
    if not card.exists():
        return
    try:
        previous = read_json(card).get("run_id")
    except Exception:  # noqa: BLE001 - an unreadable card must not block the export
        return
    if previous and previous != run_id:
        log.warning(
            "%s already held run %s: it is overwritten by %s. A fixed `retain.name` over "
            "several folds (tune) only keeps the last one; leave retain.name=null for one "
            "folder per run.", target, previous, run_id,
        )


def retain_run(run_id: str, paths: ProjectPaths, cfg: Any, name: str | None = None,
               extra_card: dict[str, Any] | None = None) -> Path | None:
    """Copy the weights of a complete run to `retained_models/pose/<name>/`.

    `extra_card` adds fields to the model card: this is how the final model receives
    the performance estimate of its outer folds, which it cannot measure itself.

    Returns the folder written, or None if the run cannot be exported (incomplete run,
    or approach without weights: `mean_pose` produces none).
    Side effect: writes retained_models/pose/<name>/.
    """
    manifest_path = paths.manifest(run_id)
    if not manifest_path.exists():
        log.warning("Incomplete run (no manifest), not exported: %s", run_id)
        return None

    weights_dir = paths.run_dir(run_id) / "weights"
    if not weights_dir.is_dir() or not any(weights_dir.rglob("*")):
        log.info("Run %s: no weights to export (approach without a trained model).", run_id)
        return None

    target = paths.retained_model(name or str(cfg.retain.name or run_id))
    _warn_on_overwrite(target, run_id)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(weights_dir, target, dirs_exist_ok=True)

    manifest = read_json(manifest_path)
    config = manifest.get("config") or {}
    data_cfg = config.get("data") or {}
    schema_name = data_cfg.get("keypoint_schema")

    exported = sorted(str(p.relative_to(target)).replace("\\", "/")
                      for p in target.rglob("*") if p.is_file() and p.name != CARD_NAME)
    write_json(target / CARD_NAME, {
        "run_id": run_id,
        "approach": manifest.get("approach"),
        "data_scope": manifest.get("data_scope"),
        "datasets": list(data_cfg.get("datasets") or []),
        "split_id": manifest.get("split_id"),
        "fold": manifest.get("fold"),
        "tag": manifest.get("tag"),
        "variant_hash": manifest.get("variant_hash"),
        "content_hash": manifest.get("content_hash"),
        "git": manifest.get("git"),
        "image_size": list((config.get("protocol") or {}).get("image_size") or []),
        "keypoint_schema": schema_name,
        "keypoint_names": _keypoint_names(paths, schema_name),
        "weights": exported,
        "source_run": str(paths.run_dir(run_id)),
        "exported_at": time.time(),
        **_primary_metric(run_id, paths, cfg),
        **(extra_card or {}),
    })

    log.info("Retained model: %s (%d file(s) from %s)", target, len(exported), weights_dir)
    return target


def retainable_approach(cfg: Any, approach: str | None = None) -> bool:
    """True if the approach (default: that of `cfg`) produces a model the pipeline can
    run as a member of the ensemble."""
    allowed = [str(a) for a in (cfg.retain.get("approaches") or [])]
    return str(approach or cfg.approach.name) in allowed


def retain_existing_run(run_id: str, paths: ProjectPaths, cfg: Any) -> Path | None:
    """`evaluate run_id=...`: the ensemble becomes this single, already-trained run.

    Same guards as `retain_context`, read in the manifest of the run (its approach and
    its mode, not those of the `evaluate` command): a smoke run, an HPO trial or an
    approach outside `retain.approaches` leaves the ensemble intact.
    """
    if not bool(cfg.retain.enabled) or not paths.manifest(run_id).exists():
        return None
    manifest = read_json(paths.manifest(run_id))
    if str(manifest.get("mode")) == "smoke":
        log.info("Smoke run not exported: %s", run_id)
        return None
    if manifest.get("role_in_protocol") == "hpo_trial" and not bool(cfg.retain.hpo_trials):
        return None
    if not retainable_approach(cfg, manifest.get("approach")):
        log.info("Approach '%s' outside retain.approaches: run not exported (%s).",
                 manifest.get("approach"), run_id)
        return None
    clear_retained(paths)
    return retain_run(run_id, paths, cfg)


def retain_context(ctx: Any, extra_card: dict[str, Any] | None = None,
                   replace: bool = True) -> Path | None:
    """Export the run of a `RunContext`, applying the `retain` config.

    `replace=True` first empties `retained_models/pose/`: the run alone becomes the
    ensemble of the pipeline (the `train` case). `tune` passes `replace=True` for the
    first outer fold then `False` for the next ones, so that the ensemble gathers its
    folds.

    Three complete runs are nevertheless not models to deliver:
      - `mode=smoke` (§1.7): 2 epochs on a toy corpus, it is a wiring test; exporting it
        would hand a toy model to the pipeline (and make the test suite write outside
        its tmp_path);
      - an HPO trial: auditable, but a single search would fill `retained_models/` with
        dozens of models (`retain.hpo_trials=true` to export them anyway);
      - an approach outside `retain.approaches`: the pipeline cannot load it as a
        member of the ensemble.
    None of these cases empties the folder: the ensemble in place stays intact.
    """
    cfg = ctx.cfg.retain
    if not bool(cfg.enabled):
        return None
    if str(ctx.cfg.mode) == "smoke":
        log.info("Smoke run not exported: %s", ctx.run_id)
        return None
    if ctx.extra.get("role_in_protocol") == "hpo_trial" and not bool(cfg.hpo_trials):
        return None
    if not retainable_approach(ctx.cfg):
        log.info("Approach '%s' outside retain.approaches: run not exported (%s).",
                 ctx.cfg.approach.name, ctx.run_id)
        return None
    if replace:
        clear_retained(ctx.paths)
    return retain_run(ctx.run_id, ctx.paths, ctx.cfg, extra_card=extra_card)
