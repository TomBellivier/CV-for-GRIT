"""Evaluateur unique du projet (CONVENTIONS.md §7.1).

Entrees : un fichier de predictions (contrat 3), les annotations canoniques
(contrat 1) et configs/eval/*.yaml. Rien d'autre. Il ne charge aucun modele et
n'importe aucun module d'approche : si c'etait necessaire, le design serait casse.
"""

from __future__ import annotations

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
        raise ContractError(
            "Aucune metrique produite : verifier que les predictions couvrent bien les "
            "images de test et que eval.metrics n'est pas vide."
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


def evaluate_run(run_id: str, paths: ProjectPaths, annotations: pd.DataFrame,
                 schemas: dict[str, KeypointSchema], eval_cfg: Any,
                 splits: list[str] | None = None, approach: str | None = None) -> Path:
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
    for file in files:
        pred = read_parquet(file, artifact="predictions", validate=True)
        split = str(pred["split"].iloc[0])
        if splits is not None and split not in splits:
            continue
        fold = int(pred["fold"].iloc[0])
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
