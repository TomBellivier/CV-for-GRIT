"""`finetune` verb: fine-tune an existing pose model on a small set of new images.

Protocol (pose only):
  1. the base model is the fold-0 member of a retained ensemble, found by its
     `variant_hash` in retained_models/pose/*/model_card.json (or the only member
     carrying that hash);
  2. the images of annotated_images/finetune-images/<order>/images/, annotated by the
     Label Studio JSON exports of <order>/, are split into `finetune.n_folds` folds;
  3. per fold: the base model is evaluated as it is ("before"), then fine-tuned on the
     train of the fold for the number of epochs it was trained with ("after"); both are
     evaluated on the same untouched test images;
  4. the before/after comparison is written to results/pose/finetune/<variant_hash>/:
     detection metrics on every test image, point metrics only on the images detected
     both before AND after (see `write_comparison`).

Everything this verb writes (canonical annotations, split, runs) lives under its own
folders: the main datasets, runs and report are never touched, and the fine-tuned
models are not exported to retained_models/.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from insectpose.contracts import DATASETS, ContractError
from insectpose.data.splits import build_splits, write_splits
from insectpose.evaluation.evaluator import evaluate_predictions, evaluate_run
from insectpose.paths import ProjectPaths
from insectpose.registry import ADAPTERS
from insectpose.utils.io import read_json, read_parquet
from insectpose.utils.logging import get_logger

log = get_logger("finetune")

CARD_NAME = "model_card.json"
# Metrics printed in the console summary (all of them are in the CSV files).
SUMMARY_METRICS = ("detection_rate", "oks_ap", "pck@0.25_thorax_width", "nme",
                   "measurement_mape_median")


# --- base model ------------------------------------------------------------------
def find_base_model(retained_pose_dir: Path, variant: str) -> tuple[Path, dict[str, Any]]:
    """Weights + model card of the fold-0 member of the ensemble `variant`.

    If a single model carries this variant_hash, it is taken whatever its fold.
    """
    cards = []
    for card_path in sorted(retained_pose_dir.glob(f"*/{CARD_NAME}")):
        card = read_json(card_path)
        if str(card.get("variant_hash")) == variant:
            cards.append((card_path.parent, card))
    if not cards:
        known = sorted({str(read_json(p).get("variant_hash"))
                        for p in retained_pose_dir.glob(f"*/{CARD_NAME}")})
        raise ContractError(
            f"No retained model with variant_hash '{variant}' in {retained_pose_dir}. "
            f"Available: {known}"
        )
    if len(cards) > 1:
        cards = [(d, c) for d, c in cards if int(c.get("fold", -1)) == 0]
        if len(cards) != 1:
            raise ContractError(
                f"Several models carry variant_hash '{variant}' but not exactly one fold 0."
            )
    folder, card = cards[0]
    weights = folder / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Weights missing for the base model: {weights}")
    return weights, card


def base_epochs(weights: Path) -> int:
    """Number of epochs the base model was trained with (Ultralytics train_args)."""
    from ultralytics import YOLO

    train_args = (getattr(YOLO(str(weights)), "ckpt", None) or {}).get("train_args") or {}
    if "epochs" not in train_args:
        raise ContractError(f"No 'epochs' in the train_args of {weights}: pass train.epochs=N.")
    return int(train_args["epochs"])


# --- data ------------------------------------------------------------------------
def _labelstudio_rows(json_path: Path, group: str) -> list[dict[str, Any]]:
    """Rows of a Label Studio export, with the converter of annotation_tools/."""
    script = Path(__file__).resolve().parents[4] / "annotation_tools" / "labelstudio_to_csv.py"
    spec = importlib.util.spec_from_file_location("labelstudio_to_csv", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.rows_from_export(json_path, group))


def prepare_images(cfg: DictConfig, paths: ProjectPaths, images_dir: Path) -> list[str]:
    """<order>/*.json + <order>/images/ -> canonical annotations of each order present.

    Side effect: <processed>/<order>/annotations.parquet and <processed>/<order>.csv.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Fine-tuning folder missing: {images_dir}")
    orders = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
    unknown = [o for o in orders if o not in DATASETS]
    if unknown or not orders:
        raise ContractError(
            f"{images_dir} must hold one sub-folder per insect order among {list(DATASETS)}, "
            f"each with the Label Studio JSON export(s) and an images/ folder. "
            f"Found: {orders}"
        )
    paths.processed.mkdir(parents=True, exist_ok=True)
    for order in orders:
        exports = sorted((images_dir / order).glob("*.json"))
        if not exports:
            raise FileNotFoundError(f"No Label Studio JSON export in {images_dir / order}")
        rows = [row for export in exports for row in _labelstudio_rows(export, order)]
        csv_path = paths.processed / f"{order}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        options = {
            "csv_path": str(csv_path),
            "group_column": "group",
            # Relative to paths.data (contract 1 forbids absolute image paths).
            "images_subdir": Path(os.path.relpath(images_dir / order / "images",
                                                  paths.data)).as_posix(),
            "margin": float(cfg.data.adapter_options.get("margin", 10)),
            "keypoint_schema": str(cfg.data.keypoint_schema),
            "skip_missing_images": True,
            "project_root": str(paths.root),
            "configs_dir": str(paths.configs),
            "data_dir": str(paths.data),
        }
        adapter = ADAPTERS.get("annotation_csv")(dataset=order, source_dir=images_dir / order,
                                                 options=options)
        log.info("[%s] %d annotation(s) -> %s", order, len(rows), adapter.run(paths))
    return orders


# --- comparison ------------------------------------------------------------------
# Metrics that measure the DETECTION itself: they keep every test image of the fold.
# All the others (point placement) are computed only on the images where the insect
# is detected both before AND after fine-tuning, otherwise the images the base model
# misses count as point failures and bias the "before" downwards.
DETECTION_METRICS = ("det_ap@0.5", "det_ap@[.5:.95]", "kpt_coverage", "oks_ar")


def _test_metrics(paths: ProjectPaths, run_id: str) -> pd.DataFrame:
    metrics = read_parquet(paths.metrics(run_id))
    return metrics[(metrics["split"] == "test") & (metrics["scope"] == "overall")]


def _test_predictions(paths: ProjectPaths, run_id: str, fold: int) -> pd.DataFrame:
    file = paths.run_dir(run_id) / "predictions" / f"test_fold{fold}.parquet"
    return read_parquet(file, artifact="predictions", validate=True)


def _prepared(pred: pd.DataFrame, gt: pd.DataFrame, eval_cfg: Any
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GT and predictions filtered exactly as `evaluate_predictions` does."""
    pred = pred[pred["bbox_score"] >= float(eval_cfg.score_threshold_curves)]
    return gt.reset_index(drop=True), pred.reset_index(drop=True)


def detected_images(pred: pd.DataFrame, gt: pd.DataFrame, schemas: dict[str, Any],
                    eval_cfg: Any) -> set[str]:
    """Images where at least one GT insect is matched by a prediction.

    Same matching as the point metrics (score_threshold_pointwise + match_oks_threshold):
    an image counted here is exactly an image whose points are evaluated.
    """
    from insectpose.evaluation.bundle import EvalBundle
    from insectpose.evaluation.matching import build_pairs

    if pred.empty or gt.empty:
        return set()
    gt, pred = _prepared(pred, gt, eval_cfg)
    pairs = build_pairs(gt, pred, schemas, area_source=str(eval_cfg.oks.area_source))
    bundle = EvalBundle(gt=gt, pred=pred, pairs=pairs, schemas=schemas, cfg=eval_cfg)
    return set(bundle.matched_instances()["image_id"])


def _metrics_on(pred: pd.DataFrame, gt: pd.DataFrame, images: set[str],
                schemas: dict[str, Any], eval_cfg: Any) -> pd.Series:
    """Overall metrics of `pred` restricted to `images` (empty if no image)."""
    if not images:
        return pd.Series(dtype=float)
    metrics = evaluate_predictions(pred[pred["image_id"].isin(images)],
                                   gt[gt["image_id"].isin(images)], schemas, eval_cfg)
    return metrics[metrics["scope"] == "overall"].set_index("metric")["value"]


def _summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    per_fold["delta"] = per_fold["after"] - per_fold["before"]
    summary = (per_fold.groupby("metric")[["before", "after", "delta"]]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    return summary


def write_comparison(paths: ProjectPaths, runs: dict[int, dict[str, str]],
                     test_images: dict[int, list[str]], out_dir: Path,
                     annotations: pd.DataFrame, schemas: dict[str, Any],
                     eval_cfg: Any) -> pd.DataFrame:
    """Per-fold and mean before/after table, from the saved predictions only.

    - detection metrics (DETECTION_METRICS + n/rate of detected images): every test image;
    - every other metric: only the images detected before AND after (`evaluated_on`).

    Side effects in `out_dir`: comparison.csv / comparison_per_fold.csv (corrected) and
    comparison_all_images*.csv (every metric on every test image, for traceability).
    """
    rows, raw_rows = [], []
    for fold, pair in runs.items():
        fold_images = set(test_images[fold])
        gt = annotations[annotations["image_id"].isin(fold_images)]
        preds = {when: _test_predictions(paths, pair[when], fold) for when in ("before", "after")}
        detected = {when: detected_images(preds[when], gt, schemas, eval_cfg)
                    for when in preds}
        common = detected["before"] & detected["after"]

        full = {when: _test_metrics(paths, pair[when]).set_index("metric")["value"]
                for when in preds}
        points = {when: _metrics_on(preds[when], gt, common, schemas, eval_cfg)
                  for when in preds}
        base = {"fold": fold, "test_images": " ".join(test_images[fold])}

        for metric in sorted(set(full["before"].index) | set(full["after"].index)):
            raw_rows.append({**base, "metric": metric, "before": full["before"].get(metric),
                             "after": full["after"].get(metric)})

        n = len(fold_images)
        for metric, b, a in [
            ("n_images_detected", len(detected["before"]), len(detected["after"])),
            ("detection_rate", len(detected["before"]) / n if n else float("nan"),
             len(detected["after"]) / n if n else float("nan")),
        ]:
            rows.append({**base, "evaluated_on": "all_test_images", "n_images": n,
                         "metric": metric, "before": b, "after": a})
        for metric in DETECTION_METRICS:
            if metric in full["before"].index or metric in full["after"].index:
                rows.append({**base, "evaluated_on": "all_test_images", "n_images": n,
                             "metric": metric, "before": full["before"].get(metric),
                             "after": full["after"].get(metric)})
        for metric in sorted(set(points["before"].index) | set(points["after"].index)):
            if metric in DETECTION_METRICS:
                continue
            rows.append({**base, "evaluated_on": "detected_before_and_after",
                         "n_images": len(common), "metric": metric,
                         "before": points["before"].get(metric),
                         "after": points["after"].get(metric)})
        log.info("Fold %d: %d test image(s), detected before %d / after %d / both %d.",
                 fold, n, len(detected["before"]), len(detected["after"]), len(common))

    per_fold = pd.DataFrame(rows)
    summary = _summarise(per_fold)
    raw = pd.DataFrame(raw_rows)
    raw_summary = _summarise(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_dir / "comparison_per_fold.csv", index=False)
    summary.to_csv(out_dir / "comparison.csv", index=False)
    raw.to_csv(out_dir / "comparison_all_images_per_fold.csv", index=False)
    raw_summary.to_csv(out_dir / "comparison_all_images.csv", index=False)
    return summary


# --- verb ------------------------------------------------------------------------
def cmd_finetune(cfg: DictConfig) -> Path:
    """Fine-tune the base model on `finetune.images_dir`, fold by fold, and compare."""
    from ultralytics import YOLO

    from insectpose.pipeline import _load_context_data, _prepare_run, cmd_train

    variant = cfg.finetune.get("base")
    if not variant:
        raise SystemExit("finetune requires base=<variant_hash> (see model_card.json).")
    variant = str(variant)

    main_paths = ProjectPaths.from_config(cfg)
    weights, card = find_base_model(main_paths.retained / "pose", variant)
    epochs = int(cfg.finetune.epochs) if cfg.finetune.get("epochs") else base_epochs(weights)
    log.info("Base model: %s (run %s, fold %s) | %d epochs", weights, card.get("run_id"),
             card.get("fold"), epochs)

    # Own folders: the main annotations, splits, runs and report stay untouched.
    cfg = cfg.copy()
    work = main_paths.data / "finetune" / variant
    images_dir = (main_paths.root / str(cfg.finetune.images_dir)).resolve()
    for key, value in {"paths.processed": work / "processed", "paths.splits": work / "splits",
                       "paths.runs": main_paths.runs / "finetune" / variant}.items():
        OmegaConf.update(cfg, key, str(value), force_add=True)
    paths = ProjectPaths.from_config(cfg)
    paths.ensure_writable_dirs()

    orders = prepare_images(cfg, paths, images_dir)
    n_folds = int(cfg.finetune.n_folds)
    overrides = {
        "data.scope": "finetune", "data.datasets": orders,
        "cv.name": f"kfold{n_folds}_grouped", "cv.n_folds": n_folds,
        "approach.weights": str(weights), "train.epochs": epochs,
        "retain.enabled": False, "folds": None,
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)

    annotations, schemas = _load_context_data(cfg, paths)
    table, meta = build_splits(annotations, cfg)
    write_splits(table, meta, paths)
    log.info("Split '%s': %d images, %d folds.", meta["split_id"], meta["n_images"], n_folds)

    runs: dict[int, dict[str, str]] = {}
    test_images: dict[int, list[str]] = {}
    for fold in range(n_folds):
        fold_cfg = cfg.copy()
        OmegaConf.update(fold_cfg, "fold", fold)

        # BEFORE: the base model as it is, predicted on the test of the fold.
        before_cfg = fold_cfg.copy()
        OmegaConf.update(before_cfg, "tag", f"ft-{variant}-before", force_add=True)
        ctx, data, approach = _prepare_run(before_cfg, {"role_in_protocol": "finetune_before",
                                                        "base_model": str(weights)})
        test_images[fold] = sorted(data.role("test").images["image_id"])
        if not ctx.is_complete() or bool(cfg.force):
            ctx.setup()
            approach.model = YOLO(str(weights))
            approach.predict(data.role("test"), ctx, "test")
            evaluate_run(ctx.run_id, paths, annotations, schemas, cfg.eval,
                         approach=str(cfg.approach.name), split_id=ctx.split_id)
            ctx.write_manifest()

        # AFTER: fine-tuned on the train of the fold, evaluated on the same test.
        after_cfg = fold_cfg.copy()
        OmegaConf.update(after_cfg, "tag", f"ft-{variant}", force_add=True)
        after = cmd_train(after_cfg, extra={"role_in_protocol": "finetune",
                                            "base_model": str(weights)})
        runs[fold] = {"before": ctx.run_id, "after": after.run_id}

    out_dir = main_paths.results / "finetune" / variant
    summary = write_comparison(paths, runs, test_images, out_dir, annotations, schemas,
                               cfg.eval)
    shown = summary[summary["metric"].isin(SUMMARY_METRICS)]
    log.info("Before / after fine-tuning, mean over %d folds (test):\n%s",
             n_folds, shown.to_string(index=False))
    log.info("Comparison written to %s", out_dir)
    return out_dir
