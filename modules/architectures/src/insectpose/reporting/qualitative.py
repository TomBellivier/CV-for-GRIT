"""Mandatory qualitative export of each run (CONVENTIONS.md §8.5).

A model is never validated on numbers alone. Each run exports annotated test images
pred vs GT, including the worst cases by per-instance OKS.

This module only reads artefacts (contracts 1 and 3): like the evaluator, it does not
know which approach produced the predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.keypoints import KeypointSchema
from insectpose.evaluation.matching import assign_greedy, build_pairs
from insectpose.utils.io import write_json
from insectpose.utils.logging import get_logger

log = get_logger("qualitative")

# Fixed colours: GT in green, prediction in orange, error link in red.
_GT_COLOR = (60, 200, 90)
_PRED_COLOR = (245, 150, 40)
_ERROR_COLOR = (220, 60, 60)


def instance_scores(gt: pd.DataFrame, pred: pd.DataFrame, schemas: dict[str, KeypointSchema],
                    eval_cfg: Any) -> pd.DataFrame:
    """OKS of each GT instance and index of the matched prediction.

    An unmatched instance gets oks=0.0 and pred_row=-1: it is therefore a priority
    candidate for the visual inspection, which is exactly the goal (§7.2).
    """
    pairs = build_pairs(gt, pred, schemas, area_source=str(eval_cfg.oks.area_source))
    threshold = float(eval_cfg.match_oks_threshold)
    rows: list[dict[str, Any]] = []
    for p in pairs:
        matched_gt, matched_sim = assign_greedy(p.oks, p.scores, threshold)
        best = {int(g): (int(p.pred_rows[i]), float(matched_sim[i]))
                for i, g in enumerate(matched_gt) if g >= 0}
        for local_idx, gt_row in enumerate(p.gt_rows):
            pred_row, oks = best.get(local_idx, (-1, 0.0))
            rows.append({"image_id": p.image_id, "dataset": p.dataset, "gt_row": int(gt_row),
                         "pred_row": pred_row, "oks": oks})
    return pd.DataFrame(rows)


def select_examples(scores: pd.DataFrame, n_examples: int, n_worst: int, seed: int,
                    n_best_per_dataset: int = 1) -> pd.DataFrame:
    """Select the worst cases, the best per dataset, then a random draw.

    The three categories answer three different questions:
    - `worst`: where the model fails, and how;
    - `best`: what it can do at best. Without this reference, one cannot tell whether the
      failures reflect a ceiling of the model or accidental cases. The selection is made
      **per dataset**, since the global best case would always come from the easiest
      order;
    - `random`: what an ordinary case looks like, the only unbiased sample.

    The random draw absorbs the variation: `n_examples` stays the total.
    """
    if scores.empty:
        return scores
    ordered = scores.sort_values("oks", ascending=True)
    worst = ordered.head(min(n_worst, len(ordered)))
    remaining = ordered.drop(worst.index)

    budget = max(0, n_examples - len(worst))
    best = remaining.head(0)
    if n_best_per_dataset > 0 and budget > 0 and not remaining.empty:
        best = (
            remaining.sort_values("oks", ascending=False)
            .groupby("dataset", sort=True)
            .head(n_best_per_dataset)
            .head(budget)
        )
        remaining = remaining.drop(best.index)

    n_random = max(0, min(n_examples - len(worst) - len(best), len(remaining)))
    sample = (
        remaining.sample(n=n_random, random_state=seed) if n_random else remaining.head(0)
    )
    selection = pd.concat([
        worst.assign(reason="worst"),
        best.assign(reason="best"),
        sample.assign(reason="random"),
    ])
    return selection.reset_index(drop=True)


def _draw_instance(draw: Any, kpts: np.ndarray, color: tuple[int, int, int],
                   skeleton: tuple[tuple[int, int], ...], radius: float,
                   mask: np.ndarray | None = None) -> None:
    """Draw the skeleton and the points of an instance on an ImageDraw."""
    keep = np.ones(len(kpts), dtype=bool) if mask is None else mask
    for a, b in skeleton:
        if a < len(kpts) and b < len(kpts) and keep[a] and keep[b]:
            draw.line([tuple(kpts[a]), tuple(kpts[b])], fill=color, width=2)
    for i, (x, y) in enumerate(kpts):
        if not keep[i]:
            continue
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def export_qualitative(run_dir: Path, gt: pd.DataFrame, pred: pd.DataFrame,
                       schemas: dict[str, KeypointSchema], eval_cfg: Any, data_root: Path,
                       seed: int = 0) -> list[Path]:
    """Write the pred vs GT figures of the run.

    Side effect: writes runs/<run_id>/figures/*.png and figures/qualitative_index.json.
    Returns the list of the figures produced.
    """
    from PIL import Image, ImageDraw

    cfg = eval_cfg.qualitative
    if pred.empty:
        log.warning("No prediction: qualitative export skipped for this run.")
        return []
    scores = instance_scores(gt, pred, schemas, eval_cfg)
    selection = select_examples(
        scores, int(cfg.n_examples), int(cfg.n_worst), seed,
        n_best_per_dataset=int(cfg.get("n_best_per_dataset", 1)),
    )
    if selection.empty:
        log.warning("No instance to export: check the predictions of the run.")
        return []

    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index: list[dict[str, Any]] = []

    for rank, row in enumerate(selection.itertuples(index=False)):
        gt_row = gt.loc[row.gt_row]
        image_path = data_root / str(gt_row.image_path)
        if not image_path.exists():
            if not bool(cfg.allow_missing_images):
                raise FileNotFoundError(
                    f"Image missing for the qualitative export: {image_path}. "
                    "Fix image_path (relative to paths.data) or set "
                    "eval.qualitative.allow_missing_images=true."
                )
            log.warning("Image missing, example skipped: %s", image_path)
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        schema = schemas[str(gt_row.keypoint_schema)]
        radius = max(2.0, 0.004 * max(image.size))

        gt_kpts = np.asarray(gt_row.kpts_xy, dtype=float).reshape(-1, 2)
        gt_vis = np.asarray(gt_row.kpts_vis, dtype=int) > 0
        _draw_instance(draw, gt_kpts, _GT_COLOR, schema.skeleton, radius, gt_vis)

        if row.pred_row >= 0:
            pred_row = pred.loc[row.pred_row]
            pred_kpts = np.asarray(pred_row.kpts_xy, dtype=float).reshape(-1, 2)
            _draw_instance(draw, pred_kpts, _PRED_COLOR, schema.skeleton, radius)
            for i in np.where(gt_vis)[0]:
                draw.line([tuple(gt_kpts[i]), tuple(pred_kpts[i])], fill=_ERROR_COLOR, width=1)

        detected = "" if row.pred_row >= 0 else " | NOT DETECTED"
        label = f"{row.reason} | OKS={row.oks:.3f}{detected}"
        draw.text((4, 4), label, fill=(255, 255, 255))

        name = f"{rank:02d}_{row.reason}_{str(gt_row.instance_id).replace('/', '_')}.png"
        path = out_dir / name
        image.save(path)
        written.append(path)
        index.append({"file": name, "instance_id": str(gt_row.instance_id),
                      "dataset": str(row.dataset), "oks": float(row.oks),
                      "reason": str(row.reason), "detected": bool(row.pred_row >= 0)})

    write_json(out_dir / "qualitative_index.json",
               {"n_examples": len(index), "legend": {"gt": "green", "prediction": "orange",
                                                     "error": "red"}, "examples": index})
    log.info("Qualitative export: %d figure(s) in %s", len(written), out_dir)
    return written