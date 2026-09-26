"""REFERENCE approach: template, smoke test and floor baseline.

Predicts the mean pose of the train set, placed back into the GT bbox of each instance.
It therefore uses the GT bboxes (`bbox_source='gt'`): it is a DIAGNOSTIC, never a row
comparable with the end-to-end approaches (CONVENTIONS.md §9.3).

This file is the model to copy to implement a real approach (§11).
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
    """Mean normalised pose per dataset, placed back into each bbox."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.priors: dict[str, np.ndarray] = {}

    # --- training ---------------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Compute the mean pose in coordinates relative to the bbox.

        Side effect: writes runs/<run_id>/weights/priors.json.
        """
        train = data.train.annotations  # data.test is never read (§4.2)
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
                # ADR-0016: a keypoint never annotated in this dataset has no mean. It is
                # placed at the centre of the bbox, which is explicit and has no effect on
                # the metrics (it is excluded from the evaluation, for lack of annotation).
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
        ctx.logger.info("mean_pose: %d prior(s) estimated on %d instances.",
                        len(self.priors), len(train))

    # --- inference --------------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:  # noqa: ARG002
        """Place the prior back into each GT bbox (frame of the original image)."""
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
                    "bbox_source": "gt",  # DIAGNOSTIC: not comparable with end-to-end
                }
            )
        return pd.DataFrame(rows)

    # --- reloading --------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> MeanPoseApproach:
        """Reload the priors without retraining."""
        obj = cls(cfg)
        payload = json.loads((run_dir / "weights" / "priors.json").read_text(encoding="utf-8"))
        obj.priors = {k: np.asarray(v, dtype=float) for k, v in payload.items()}
        return obj
