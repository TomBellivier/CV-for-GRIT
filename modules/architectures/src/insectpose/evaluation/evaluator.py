"""Evaluateur unique du projet (CONVENTIONS.md §7.1).

Entrees : un fichier de predictions (contrat 3), les annotations canoniques
(contrat 1) et configs/eval/*.yaml. Rien d'autre. Il ne charge aucun modele et
n'importe aucun module d'approche : si c'etait necessaire, le design serait casse.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from insectpose.contracts import METRIC_SCHEMA_VERSION, ContractError
from insectpose.data.keypoints import KeypointSchema
from insectpose.evaluation.bundle import EvalBundle
from insectpose.evaluation.matching import build_pairs
from insectpose.paths import ProjectPaths
from insectpose.registry import METRICS
from insectpose.utils.io import read_json, read_parquet, write_parquet
from insectpose.utils.logging import get_logger

log = get_logger("evaluator")


def evaluate_predictions(predictions: pd.DataFrame, annotations: pd.DataFrame,
                         schemas: dict[str, KeypointSchema], eval_cfg: Any) -> pd.DataFrame:
    """Calcule toutes les metriques configurees. Aucun effet de bord.

    Les predictions sont filtrees au seuil bas des courbes ; le seuillage fort
    n'intervient que dans les metriques ponctuelles (§3.4).
    """
    gt = annotations[annotations["image_id"].isin(set(predictions["image_id"]))
                     | annotations["image_id"].isin(set(annotations["image_id"]))]
    gt = gt.reset_index(drop=True)
    pred = predictions[
        predictions["bbox_score"] >= float(eval_cfg.score_threshold_curves)
    ].reset_index(drop=True)

    _check_schema_consistency(gt, pred)

    pairs = build_pairs(gt, pred, schemas, area_source=str(eval_cfg.oks.area_source))
    bundle = EvalBundle(gt=gt, pred=pred, pairs=pairs, schemas=schemas, cfg=eval_cfg)

    rows: list[dict[str, Any]] = []
    for name in list(eval_cfg.metrics):
        fn = METRICS.get(name)
        produced = fn(bundle)
        if not produced:
            log.info("Metrique '%s' non applicable a ce run (aucune ligne produite).", name)
        rows.extend(produced)
    if not rows:
        if pred.empty:
            # Un modele qui ne detecte rien obtient des metriques NULLES, pas une
            # absence de metriques : c'est un resultat mesurable, et le masquer
            # laisserait croire a un run casse alors que le modele est simplement mauvais.
            from insectpose.evaluation.bundle import record

            n_gt = int(len(gt))
            rows = [record("overall", str(eval_cfg.primary_metric), 0.0, n_gt),
                    record("overall", "kpt_coverage", 0.0, n_gt)]
            for dataset in sorted(gt["dataset"].unique()):
                subset = int((gt["dataset"] == dataset).sum())
                rows.append(record(f"dataset:{dataset}", str(eval_cfg.primary_metric),
                                   0.0, subset))
            log.warning("Aucune prediction : metriques nulles publiees sur %d instance(s).",
                        n_gt)
            return pd.DataFrame(rows)
        raise ContractError(
            "Predictions presentes mais aucune metrique produite : verifier que "
            "eval.metrics n'est pas vide et que les scopes sont actives."
        )
    return pd.DataFrame(rows)


def _check_schema_consistency(gt: pd.DataFrame, pred: pd.DataFrame) -> None:
    """Refuse une prediction dont le schema de keypoints ne suit pas celui du dataset."""
    if pred.empty:
        return
    ref = gt.drop_duplicates("image_id").set_index("image_id")["keypoint_schema"]
    merged = pred[["image_id", "keypoint_schema"]].join(ref, on="image_id", rsuffix="_gt")
    bad = merged[merged["keypoint_schema"] != merged["keypoint_schema_gt"]]
    if len(bad):
        raise ContractError(
            f"{len(bad)} predictions dans un schema different de celui du dataset "
            f"(ex. image {bad['image_id'].iloc[0]}). Un modele multi-datasets doit "
            "reprojeter vers le schema LOCAL avant ecriture (§3.1)."
        )
    unknown = set(pred["image_id"]) - set(gt["image_id"])
    if unknown:
        raise ContractError(
            f"{len(unknown)} images predites hors du perimetre evalue "
            f"(ex. {sorted(unknown)[:2]}). Une prediction de test ne doit couvrir que "
            "les images du fold."
        )


def parse_prediction_filename(path: Path) -> tuple[str, int]:
    """(split, fold) d'un fichier `<split>_fold<k>.parquet`."""
    match = re.match(r"(?P<split>[a-z]+)_fold(?P<fold>\d+)$", Path(path).stem)
    if not match:
        raise ContractError(
            f"Nom de fichier de predictions inattendu : {Path(path).name}. "
            "Format attendu : <split>_fold<k>.parquet"
        )
    return match.group("split"), int(match.group("fold"))


def _fold_images(paths: ProjectPaths, split_id: str,
                 annotations: pd.DataFrame) -> dict[tuple[str, int], set[str]]:
    """Images de chaque (split, fold) d'apres le decoupage du run.

    Sert de perimetre d'evaluation quand un fichier de predictions est vide.
    """
    file = paths.split_file(split_id) if split_id else None
    if file is None or not file.exists():
        return {}
    table = read_parquet(file)
    known = set(annotations["image_id"])
    return {
        (str(role), int(fold)): set(group["image_id"]) & known
        for (fold, role), group in table.groupby(["fold", "role"])
    }


def evaluate_run(run_id: str, paths: ProjectPaths, annotations: pd.DataFrame,
                 schemas: dict[str, KeypointSchema], eval_cfg: Any,
                 splits: list[str] | None = None, approach: str | None = None,
                 split_id: str | None = None) -> Path:
    """Evalue tous les fichiers de predictions d'un run et ecrit `metrics.parquet`.

    `approach` est passe explicitement pendant un entrainement : le manifeste s'ecrit
    EN DERNIER (§8.2) et n'est donc pas encore lisible a ce moment.

    Effet de bord : ecrit runs/<run_id>/metrics.parquet (contrat 4).
    """
    manifest_path = paths.manifest(run_id)
    meta = read_json(manifest_path) if manifest_path.exists() else {}
    approach_name = approach or meta.get("approach")
    if not approach_name:
        raise ContractError(
            f"Nom d'approche inconnu pour le run '{run_id}' : passer approach=... ou "
            "evaluer un run dont le manifeste existe."
        )
    pred_dir = paths.run_dir(run_id) / "predictions"
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Aucune prediction dans {pred_dir}. Lancer 'predict' d'abord.")

    frames: list[pd.DataFrame] = []
    fold_table = _fold_images(paths, split_id or str(meta.get("split_id", "")), annotations)
    for file in files:
        pred = read_parquet(file, artifact="predictions", validate=True)
        # Le split et le fold viennent du NOM du fichier : un modele qui ne detecte rien
        # produit un fichier vide, dont aucune ligne ne pourrait les porter.
        split, fold = parse_prediction_filename(file)
        if splits is not None and split not in splits:
            continue
        if pred.empty:
            # Le perimetre d'evaluation est alors celui du decoupage, pas celui des
            # predictions : sinon le denominateur serait nul et l'echec invisible.
            images = fold_table.get((split, fold), set())
            subset = annotations[annotations["image_id"].isin(images)]
        else:
            subset = annotations[annotations["image_id"].isin(set(pred["image_id"]))]
        metrics = evaluate_predictions(pred, subset, schemas, eval_cfg)
        metrics["run_id"] = run_id
        metrics["approach"] = approach_name
        metrics["fold"] = fold
        metrics["split"] = split
        metrics["schema_version"] = METRIC_SCHEMA_VERSION
        frames.append(metrics)

    if not frames:
        raise ContractError(f"Aucun split evaluable pour {run_id} (filtre : {splits}).")
    table = pd.concat(frames, ignore_index=True)
    return write_parquet(paths.metrics(run_id), table, artifact="metrics")


def primary_value(metrics: pd.DataFrame, eval_cfg: Any, split: str = "test",
                  scope: str = "overall") -> float:
    """Valeur de la metrique primaire, celle qu'Optuna optimise (§6.3).

    Echoue si elle est absente : renvoyer une valeur de repli masquerait un run casse.
    """
    name = str(eval_cfg.primary_metric)
    sel = metrics[
        (metrics["metric"] == name) & (metrics["scope"] == scope) & (metrics["split"] == split)
    ]
    if sel.empty:
        raise ContractError(
            f"Metrique primaire '{name}' absente (scope={scope}, split={split}). "
            f"Disponibles : {sorted(metrics['metric'].unique())[:8]}"
        )
    return float(sel["value"].mean())