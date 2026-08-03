"""Export qualitatif obligatoire de chaque run (CONVENTIONS.md §8.4).

Un modele n'est jamais valide sur des chiffres seuls. Chaque run exporte des images
de test annotees pred vs GT, dont les pires cas selon l'OKS par instance.

Ce module ne lit que des artefacts (contrats 1 et 3) : comme l'evaluateur, il ignore
quelle approche a produit les predictions.
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

# Couleurs fixes : GT en vert, prediction en orange, lien d'erreur en rouge.
_GT_COLOR = (60, 200, 90)
_PRED_COLOR = (245, 150, 40)
_ERROR_COLOR = (220, 60, 60)


def instance_scores(gt: pd.DataFrame, pred: pd.DataFrame, schemas: dict[str, KeypointSchema],
                    eval_cfg: Any) -> pd.DataFrame:
    """OKS de chaque instance GT et index de la prediction appariee.

    Une instance non appariee recoit oks=0.0 et pred_row=-1 : elle est donc candidate
    prioritaire a l'inspection visuelle, ce qui est exactement le but (§7.2).
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


def select_examples(scores: pd.DataFrame, n_examples: int, n_worst: int,
                    seed: int) -> pd.DataFrame:
    """Selectionne les pires cas puis un echantillon aleatoire reproductible."""
    if scores.empty:
        return scores
    ordered = scores.sort_values("oks", ascending=True)
    worst = ordered.head(min(n_worst, len(ordered)))
    remaining = ordered.drop(worst.index)
    n_random = max(0, min(n_examples - len(worst), len(remaining)))
    sample = (
        remaining.sample(n=n_random, random_state=seed) if n_random else remaining.head(0)
    )
    selection = pd.concat([worst.assign(reason="worst"), sample.assign(reason="random")])
    return selection.reset_index(drop=True)


def _draw_instance(draw: Any, kpts: np.ndarray, color: tuple[int, int, int],
                   skeleton: tuple[tuple[int, int], ...], radius: float,
                   mask: np.ndarray | None = None) -> None:
    """Trace squelette et points d'une instance sur un ImageDraw."""
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
    """Ecrit les figures pred vs GT du run.

    Effet de bord : ecrit runs/<run_id>/figures/*.png et figures/qualitative_index.json.
    Retourne la liste des figures produites.
    """
    from PIL import Image, ImageDraw

    cfg = eval_cfg.qualitative
    scores = instance_scores(gt, pred, schemas, eval_cfg)
    selection = select_examples(
        scores, int(cfg.n_examples), int(cfg.n_worst), seed
    )
    if selection.empty:
        log.warning("Aucune instance a exporter : verifier les predictions du run.")
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
                    f"Image absente pour l'export qualitatif : {image_path}. "
                    "Corriger image_path (relatif a paths.data) ou passer "
                    "eval.qualitative.allow_missing_images=true."
                )
            log.warning("Image absente, exemple ignore : %s", image_path)
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
    log.info("Export qualitatif : %d figure(s) dans %s", len(written), out_dir)
    return written