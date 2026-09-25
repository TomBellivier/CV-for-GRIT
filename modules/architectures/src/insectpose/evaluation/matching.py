"""Appariement predictions <-> verite terrain (CONVENTIONS.md §7.3).

Implemente UNE SEULE FOIS ici. Aucune metrique ne reimplemente son propre
appariement, sans quoi deux metriques peuvent se contredire sur le meme run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insectpose.data.keypoints import KeypointSchema
from insectpose.utils.geometry import bbox_area, bbox_iou, oks_matrix


@dataclass
class ImagePairs:
    """Similarites pred x gt pour une image."""

    image_id: str
    dataset: str
    gt_rows: np.ndarray      # indices de lignes dans le DataFrame GT
    pred_rows: np.ndarray    # indices de lignes dans le DataFrame predictions
    scores: np.ndarray       # (P,) score de detection
    oks: np.ndarray          # (P, G)
    iou: np.ndarray          # (P, G)

    @property
    def n_gt(self) -> int:
        return int(self.gt_rows.size)

    @property
    def n_pred(self) -> int:
        return int(self.pred_rows.size)


def _stack(series: pd.Series, dtype: type = float) -> np.ndarray:
    if len(series) == 0:
        return np.zeros((0, 0), dtype=dtype)
    return np.stack(series.map(lambda v: np.asarray(v, dtype=dtype)).to_numpy())


def build_pairs(gt: pd.DataFrame, pred: pd.DataFrame, schemas: dict[str, KeypointSchema],
                area_source: str = "bbox") -> list[ImagePairs]:
    """Calcule OKS et IoU par image. Les images sans prediction sont conservees.

    Une image GT sans prediction produit un ImagePairs a P=0 : indispensable pour que
    la non-detection soit penalisee (§7.2) plutot qu'ignoree.
    """
    pairs: list[ImagePairs] = []
    pred_by_image = dict(list(pred.groupby("image_id"))) if len(pred) else {}

    for image_id, gt_group in gt.groupby("image_id"):
        dataset = str(gt_group["dataset"].iloc[0])
        schema = schemas[str(gt_group["keypoint_schema"].iloc[0])]
        gt_kpts = _stack(gt_group["kpts_xy"]).reshape(len(gt_group), -1, 2)
        gt_vis = _stack(gt_group["kpts_vis"], int)
        gt_bbox = _stack(gt_group["bbox_xywh"])
        areas = (
            bbox_area(gt_bbox) if area_source == "bbox"
            else gt_group["area"].to_numpy(dtype=float)
        )

        p_group = pred_by_image.get(image_id)
        if p_group is None or len(p_group) == 0:
            pairs.append(
                ImagePairs(image_id=str(image_id), dataset=dataset,
                           gt_rows=gt_group.index.to_numpy(), pred_rows=np.zeros(0, dtype=int),
                           scores=np.zeros(0), oks=np.zeros((0, len(gt_group))),
                           iou=np.zeros((0, len(gt_group))))
            )
            continue

        p_kpts = _stack(p_group["kpts_xy"]).reshape(len(p_group), -1, 2)
        p_bbox = _stack(p_group["bbox_xywh"])
        pairs.append(
            ImagePairs(
                image_id=str(image_id),
                dataset=dataset,
                gt_rows=gt_group.index.to_numpy(),
                pred_rows=p_group.index.to_numpy(),
                scores=p_group["bbox_score"].to_numpy(dtype=float),
                oks=oks_matrix(gt_kpts, gt_vis, p_kpts, schema.sigmas, areas),
                iou=bbox_iou(p_bbox, gt_bbox),
            )
        )
    return pairs


def assign_greedy(similarity: np.ndarray, scores: np.ndarray, threshold: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Appariement glouton par score decroissant, a un seuil de similarite donne.

    Retourne (gt_index_par_prediction, similarite_retenue) ; -1 = prediction non appariee.
    """
    n_pred, n_gt = similarity.shape
    matched_gt = np.full(n_pred, -1, dtype=int)
    matched_sim = np.zeros(n_pred, dtype=float)
    if n_pred == 0 or n_gt == 0:
        return matched_gt, matched_sim
    used = np.zeros(n_gt, dtype=bool)
    for p in np.argsort(-scores, kind="stable"):
        candidates = np.where(~used & (similarity[p] >= threshold))[0]
        if candidates.size == 0:
            continue
        best = candidates[int(np.argmax(similarity[p, candidates]))]
        used[best] = True
        matched_gt[p] = best
        matched_sim[p] = similarity[p, best]
    return matched_gt, matched_sim


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int,
                      n_points: int = 101) -> float:
    """AP par interpolation en 101 points (convention COCO). 0.0 si aucun GT."""
    if n_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp_sorted = tp[order].astype(float)
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1.0 - tp_sorted)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # precision monotone decroissante
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, n_points)
    idx = np.searchsorted(recall, grid, side="left")
    interp = np.where(idx < precision.size, precision[np.clip(idx, 0, precision.size - 1)], 0.0)
    return float(interp.mean())
