"""Approche de REFERENCE : gabarit, smoke test et baseline plancher.

Predit la pose moyenne du train, replacee dans la bbox GT de chaque instance.
Elle utilise donc les bboxes GT (`bbox_source='gt'`) : c'est un DIAGNOSTIC, jamais
une ligne comparable aux approches bout-en-bout (CONVENTIONS.md §9.3).

Ce fichier est le modele a copier pour implementer une vraie approche (§11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.approaches.base import BaseApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.registry import register_approach


@register_approach("mean_pose")
class MeanPoseApproach(BaseApproach):
    """Pose moyenne normalisee par dataset, replacee dans chaque bbox."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.priors: dict[str, np.ndarray] = {}

    # --- entrainement ------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Calcule la pose moyenne en coordonnees relatives a la bbox.

        Effet de bord : ecrit runs/<run_id>/weights/priors.json.
        """
        train = data.train.annotations  # data.test n'est jamais lu (§4.2)
        per_dataset = bool(self.cfg.approach.per_dataset_prior)
        key = "dataset" if per_dataset else "keypoint_schema"

        for group_key, group in train.groupby(key):
            rel = []
            for row in group.itertuples(index=False):
                pts = np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2)
                vis = np.asarray(row.kpts_vis) > 0
                x, y, w, h = np.asarray(row.bbox_xywh, dtype=float)
                if w <= 0 or h <= 0:
                    continue
                norm = (pts - np.array([x, y])) / np.array([max(w, 1e-9), max(h, 1e-9)])
                norm[~vis] = np.nan
                rel.append(norm)
            if rel:
                stacked = np.stack(rel)
                # ADR-0016 : un keypoint jamais annote dans ce dataset n'a pas de moyenne.
                # On le place au centre de la bbox, ce qui est explicite et sans effet sur
                # les metriques (il est exclu de l'evaluation, faute d'annotation).
                observed = np.isfinite(stacked).any(axis=0)
                prior = np.full(stacked.shape[1:], 0.5, dtype=float)
                if observed.any():
                    prior[observed] = np.nanmean(stacked[:, observed], axis=0)
                self.priors[str(group_key)] = prior

        shrink = float(self.cfg.approach.shrinkage)
        if shrink > 0:
            for k, v in self.priors.items():
                self.priors[k] = (1 - shrink) * v + shrink * 0.5

        weights = ctx.subdir("weights") / "priors.json"
        weights.write_text(
            json.dumps({k: v.tolist() for k, v in self.priors.items()}), encoding="utf-8"
        )
        ctx.logger.info("mean_pose : %d prior(s) estime(s) sur %d instances.",
                        len(self.priors), len(train))

    # --- inference ---------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:  # noqa: ARG002
        """Replace le prior dans chaque bbox GT (repere image d'origine)."""
        per_dataset = bool(self.cfg.approach.per_dataset_prior)
        rows = []
        for row in images.annotations.itertuples(index=False):
            key = row.dataset if per_dataset else row.keypoint_schema
            prior = self.priors.get(str(key))
            if prior is None:
                prior = np.full((len(row.kpts_vis), 2), 0.5)
            x, y, w, h = np.asarray(row.bbox_xywh, dtype=float)
            pts = prior * np.array([w, h]) + np.array([x, y])
            rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": row.dataset,
                    "bbox_xywh": [float(v) for v in row.bbox_xywh],
                    "bbox_score": 1.0,
                    "kpts_xy": [float(v) for v in pts.reshape(-1)],
                    "kpts_score": [1.0] * len(prior),
                    "keypoint_schema": row.keypoint_schema,
                    "bbox_source": "gt",  # DIAGNOSTIC : non comparable au bout-en-bout
                }
            )
        return pd.DataFrame(rows)

    # --- rechargement ------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> MeanPoseApproach:
        """Recharge les priors sans reentrainement."""
        obj = cls(cfg)
        payload = json.loads((run_dir / "weights" / "priors.json").read_text(encoding="utf-8"))
        obj.priors = {k: np.asarray(v, dtype=float) for k, v in payload.items()}
        return obj
