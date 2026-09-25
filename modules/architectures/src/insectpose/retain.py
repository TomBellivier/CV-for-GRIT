"""Export d'un run vers `retained_models/` (CONVENTIONS.md §1.5, §2, §5.2, §8.2).

Un run complet vit dans `runs/<run_id>/` et n'en sort jamais : c'est la zone
d'ecriture des approches. Ce module fait la seule chose qui en sort quelque
chose, et vers un seul endroit : il COPIE les poids du run dans
`retained_models/pose/<name>/`, avec une carte de modele, pour que `pipeline/`
les charge sans rien savoir de ce module (voir retained_models/README.md).

`retained_models/pose/` contient UN ENSEMBLE de modeles, que le pipeline infere
tous et moyenne point par point : un seul modele apres `train`, un par fold
externe apres `tune`. Chaque commande remplace l'ensemble precedent
(`clear_retained`) : deux entrainements ne s'y melangent jamais. Seules les
approches de `retain.approaches` (un YOLO-pose, un `best.pt`, le schema complet)
sont exportees, car le pipeline les charge toutes de la meme facon.

Rien n'est deplace ni supprime dans `runs/` : le run reste la source de verite,
l'export est reproductible (reexporter ecrase la copie precedente).

§8.2 : seul un run COMPLET (manifeste present) est exportable ; un run sans
manifeste est un run casse, l'exporter propagerait un modele non auditable.
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
    """Vide `retained_models/<kind>/` avant d'y ecrire un nouvel ensemble.

    Le pipeline infere TOUS les modeles du dossier : un modele laisse par une
    commande precedente entrerait dans la moyenne sans que rien ne le signale.
    Les fichiers caches (.gitkeep) sont conserves.
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
        log.info("Ensemble precedent retire de %s (%d element(s)).", folder, removed)


def write_ensemble_card(paths: ProjectPaths, card: dict[str, Any], kind: str = "pose") -> Path:
    """Decrit l'ensemble retenu : ses membres et, apres `tune`, l'estimation CV."""
    folder = paths.retained / kind
    members = sorted(p.parent.name for p in folder.glob(f"*/{CARD_NAME}"))
    target = folder / ENSEMBLE_NAME
    write_json(target, {"members": members, "n_members": len(members),
                        "written_at": time.time(), **card})
    return target


def _primary_metric(run_id: str, paths: ProjectPaths, cfg: Any) -> dict[str, Any]:
    """Metrique primaire du run, pour la carte de modele. Jamais bloquant."""
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
    except Exception as exc:  # noqa: BLE001 - une carte incomplete vaut mieux qu'un export perdu
        log.warning("Metrique primaire illisible pour %s : %s", run_id, exc)
        return {}


def _keypoint_names(paths: ProjectPaths, schema_name: str | None) -> list[str]:
    """Ordre des keypoints du schema, pour que `pipeline/` puisse le verifier."""
    if not schema_name:
        return []
    try:
        from insectpose.data.keypoints import load_schema

        return list(load_schema(str(schema_name), paths.configs).names)
    except Exception as exc:  # noqa: BLE001
        log.warning("Schema de keypoints '%s' illisible : %s", schema_name, exc)
        return []


def _warn_on_overwrite(target: Path, run_id: str) -> None:
    """Alerte si l'export ecrase le modele d'un AUTRE run.

    Cas reel : `tune` reentraine un fold externe par fold, donc plusieurs runs. Avec
    un `retain.name` fixe, ils visent tous le meme dossier et le dernier fold gagne
    en silence. Reexporter le meme run, lui, est une operation normale.
    """
    card = target / CARD_NAME
    if not card.exists():
        return
    try:
        previous = read_json(card).get("run_id")
    except Exception:  # noqa: BLE001 - une carte illisible ne doit pas bloquer l'export
        return
    if previous and previous != run_id:
        log.warning(
            "%s contenait deja le run %s : il est ecrase par %s. Un `retain.name` fixe "
            "sur plusieurs folds (tune) ne garde que le dernier ; laisser retain.name=null "
            "pour un dossier par run.", target, previous, run_id,
        )


def retain_run(run_id: str, paths: ProjectPaths, cfg: Any, name: str | None = None,
               extra_card: dict[str, Any] | None = None) -> Path | None:
    """Copie les poids d'un run complet dans `retained_models/pose/<name>/`.

    `extra_card` ajoute des champs a la carte du modele : c'est par la que le
    modele final recoit l'estimation de performance de ses folds externes, qu'il ne
    peut pas mesurer lui-meme.

    Retourne le dossier ecrit, ou None si le run n'est pas exportable (run
    incomplet, ou approche sans poids : `mean_pose` n'en produit pas).
    Effet de bord : ecrit retained_models/pose/<name>/.
    """
    manifest_path = paths.manifest(run_id)
    if not manifest_path.exists():
        log.warning("Run incomplet (pas de manifeste), non exporte : %s", run_id)
        return None

    weights_dir = paths.run_dir(run_id) / "weights"
    if not weights_dir.is_dir() or not any(weights_dir.rglob("*")):
        log.info("Run %s : aucun poids a exporter (approche sans modele entraine).", run_id)
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

    log.info("Modele retenu : %s (%d fichier(s) depuis %s)", target, len(exported), weights_dir)
    return target


def retainable_approach(cfg: Any, approach: str | None = None) -> bool:
    """True si l'approche (defaut : celle de `cfg`) produit un modele que le pipeline
    sait inferer en ensemble."""
    allowed = [str(a) for a in (cfg.retain.get("approaches") or [])]
    return str(approach or cfg.approach.name) in allowed


def retain_existing_run(run_id: str, paths: ProjectPaths, cfg: Any) -> Path | None:
    """`evaluate run_id=...` : l'ensemble devient ce seul run deja entraine.

    Memes garde-fous que `retain_context`, lus dans le manifeste du run (son
    approche et son mode, pas ceux de la commande `evaluate`) : un run smoke, un
    trial d'HPO ou une approche hors de `retain.approaches` laisse l'ensemble intact.
    """
    if not bool(cfg.retain.enabled) or not paths.manifest(run_id).exists():
        return None
    manifest = read_json(paths.manifest(run_id))
    if str(manifest.get("mode")) == "smoke":
        log.info("Run smoke non exporte : %s", run_id)
        return None
    if manifest.get("role_in_protocol") == "hpo_trial" and not bool(cfg.retain.hpo_trials):
        return None
    if not retainable_approach(cfg, manifest.get("approach")):
        log.info("Approche '%s' hors de retain.approaches : run non exporte (%s).",
                 manifest.get("approach"), run_id)
        return None
    clear_retained(paths)
    return retain_run(run_id, paths, cfg)


def retain_context(ctx: Any, extra_card: dict[str, Any] | None = None,
                   replace: bool = True) -> Path | None:
    """Exporte le run d'un `RunContext`, en appliquant la config `retain`.

    `replace=True` vide d'abord `retained_models/pose/` : le run devient a lui seul
    l'ensemble du pipeline (cas de `train`). `tune` passe `replace=True` au premier
    fold externe puis `False` aux suivants, pour que l'ensemble reunisse ses folds.

    Trois runs complets ne sont pourtant pas des modeles a livrer :
      - `mode=smoke` (§1.7) : 2 epochs sur un corpus jouet, c'est un test de
        branchement ; l'exporter mettrait un modele jouet a la disposition du
        pipeline (et ferait ecrire la suite de tests hors de son tmp_path) ;
      - un trial d'HPO : auditable, mais une seule recherche remplirait
        `retained_models/` de dizaines de modeles (`retain.hpo_trials=true`
        pour les exporter quand meme) ;
      - une approche hors de `retain.approaches` : le pipeline ne sait pas la
        charger comme un membre d'ensemble.
    Aucun de ces cas ne vide le dossier : l'ensemble en place reste intact.
    """
    cfg = ctx.cfg.retain
    if not bool(cfg.enabled):
        return None
    if str(ctx.cfg.mode) == "smoke":
        log.info("Run smoke non exporte : %s", ctx.run_id)
        return None
    if ctx.extra.get("role_in_protocol") == "hpo_trial" and not bool(cfg.hpo_trials):
        return None
    if not retainable_approach(ctx.cfg):
        log.info("Approche '%s' hors de retain.approaches : run non exporte (%s).",
                 ctx.cfg.approach.name, ctx.run_id)
        return None
    if replace:
        clear_retained(ctx.paths)
    return retain_run(ctx.run_id, ctx.paths, ctx.cfg, extra_card=extra_card)
