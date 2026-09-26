"""Single evaluator of the project (CONVENTIONS.md §7.1).

Inputs: a predictions file (contract 3), the canonical annotations (contract 1) and
configs/eval/*.yaml. Nothing else. It loads no model and imports no approach module:
if that were necessary, the design would be broken.
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
    """Compute every configured metric. No side effect.

    The predictions are filtered at the low threshold of the curves; the strong
    thresholding only happens in the point metrics (§3.4).
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
            log.info("Metric '%s' not applicable to this run (no row produced).", name)
        rows.extend(produced)
    if not rows:
        if pred.empty:
            # A model that detects nothing gets ZERO metrics, not an absence of metrics:
            # it is a measurable result, and hiding it would suggest a broken run when the
            # model is simply bad.
            from insectpose.evaluation.bundle import record

            n_gt = int(len(gt))
            rows = [record("overall", str(eval_cfg.primary_metric), 0.0, n_gt),
                    record("overall", "kpt_coverage", 0.0, n_gt)]
            for dataset in sorted(gt["dataset"].unique()):
                subset = int((gt["dataset"] == dataset).sum())
                rows.append(record(f"dataset:{dataset}", str(eval_cfg.primary_metric),
                                   0.0, subset))
            log.warning("No prediction: zero metrics published over %d instance(s).",
                        n_gt)
            return pd.DataFrame(rows)
        raise ContractError(
            "Predictions present but no metric produced: check that eval.metrics is not "
            "empty and that the scopes are enabled."
        )
    return pd.DataFrame(rows)


def _check_schema_consistency(gt: pd.DataFrame, pred: pd.DataFrame) -> None:
    """Refuse a prediction whose keypoint schema does not follow that of the dataset."""
    if pred.empty:
        return
    ref = gt.drop_duplicates("image_id").set_index("image_id")["keypoint_schema"]
    merged = pred[["image_id", "keypoint_schema"]].join(ref, on="image_id", rsuffix="_gt")
    bad = merged[merged["keypoint_schema"] != merged["keypoint_schema_gt"]]
    if len(bad):
        raise ContractError(
            f"{len(bad)} predictions in a schema different from that of the dataset "
            f"(e.g. image {bad['image_id'].iloc[0]}). A multi-dataset model must "
            "reproject to the LOCAL schema before writing (§3.1)."
        )
    unknown = set(pred["image_id"]) - set(gt["image_id"])
    if unknown:
        raise ContractError(
            f"{len(unknown)} predicted images outside the evaluated scope "
            f"(e.g. {sorted(unknown)[:2]}). A test prediction must only cover the "
            "images of the fold."
        )


def parse_prediction_filename(path: Path) -> tuple[str, int]:
    """(split, fold) of a `<split>_fold<k>.parquet` file."""
    match = re.match(r"(?P<split>[a-z]+)_fold(?P<fold>\d+)$", Path(path).stem)
    if not match:
        raise ContractError(
            f"Unexpected predictions file name: {Path(path).name}. "
            "Expected format: <split>_fold<k>.parquet"
        )
    return match.group("split"), int(match.group("fold"))


def _fold_images(paths: ProjectPaths, split_id: str,
                 annotations: pd.DataFrame) -> dict[tuple[str, int], set[str]]:
    """Images of each (split, fold) according to the split of the run.

    Serves as the evaluation scope when a predictions file is empty.
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
    """Evaluate every predictions file of a run and write `metrics.parquet`.

    `approach` is passed explicitly during a training: the manifest is written LAST
    (§8.2) and so cannot be read yet at that time.

    Side effect: writes runs/<run_id>/metrics.parquet (contract 4).
    """
    manifest_path = paths.manifest(run_id)
    meta = read_json(manifest_path) if manifest_path.exists() else {}
    approach_name = approach or meta.get("approach")
    if not approach_name:
        raise ContractError(
            f"Unknown approach name for run '{run_id}': pass approach=... or evaluate a "
            "run whose manifest exists."
        )
    pred_dir = paths.run_dir(run_id) / "predictions"
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction in {pred_dir}. Run 'predict' first.")

    frames: list[pd.DataFrame] = []
    fold_table = _fold_images(paths, split_id or str(meta.get("split_id", "")), annotations)
    for file in files:
        pred = read_parquet(file, artifact="predictions", validate=True)
        # The split and the fold come from the NAME of the file: a model that detects
        # nothing produces an empty file, where no row could carry them.
        split, fold = parse_prediction_filename(file)
        if splits is not None and split not in splits:
            continue
        if pred.empty:
            # The evaluation scope is then that of the split, not that of the
            # predictions: otherwise the denominator would be zero and the failure invisible.
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
        raise ContractError(f"No split to evaluate for {run_id} (filter: {splits}).")
    table = pd.concat(frames, ignore_index=True)
    return write_parquet(paths.metrics(run_id), table, artifact="metrics")


def primary_value(metrics: pd.DataFrame, eval_cfg: Any, split: str = "test",
                  scope: str = "overall") -> float:
    """Value of the primary metric, the one Optuna optimises (§6.3).

    Fails if it is missing: returning a fallback value would hide a broken run.
    """
    name = str(eval_cfg.primary_metric)
    sel = metrics[
        (metrics["metric"] == name) & (metrics["scope"] == scope) & (metrics["split"] == split)
    ]
    if sel.empty:
        raise ContractError(
            f"Primary metric '{name}' missing (scope={scope}, split={split}). "
            f"Available: {sorted(metrics['metric'].unique())[:8]}"
        )
    return float(sel["value"].mean())