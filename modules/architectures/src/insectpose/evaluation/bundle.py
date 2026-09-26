"""Evaluation context shared by every metric.

The evaluator knows neither the approach nor the framework that produced the
predictions: it only sees DataFrames complying with contracts 1 and 3 (§7.1).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.keypoints import KeypointSchema
from insectpose.evaluation.matching import ImagePairs


@dataclass
class EvalBundle:
    """Everything a metric needs, and nothing more."""

    gt: pd.DataFrame
    pred: pd.DataFrame
    pairs: list[ImagePairs]
    schemas: dict[str, KeypointSchema]
    cfg: Any
    records: list[dict[str, Any]] = field(default_factory=list)

    def scopes(self) -> Iterator[tuple[str, list[ImagePairs]]]:
        """Aggregation scopes: overall then dataset:<name> (§7.4)."""
        if bool(self.cfg.scopes.overall):
            yield "overall", self.pairs
        if bool(self.cfg.scopes.per_dataset):
            datasets = sorted({p.dataset for p in self.pairs})
            for dataset in datasets:
                yield f"dataset:{dataset}", [p for p in self.pairs if p.dataset == dataset]

    def n_gt(self, pairs: list[ImagePairs]) -> int:
        """Number of GT instances of a scope (denominator of the metrics)."""
        return int(sum(p.n_gt for p in pairs))

    def matched_instances(self, pairs: list[ImagePairs] | None = None) -> pd.DataFrame:
        """Matched (GT instance, prediction) pairs, at the configured matching threshold.

        Single entry point of the metrics that need the pairs: they never redo their own
        matching (§7.3).
        """
        from insectpose.evaluation.matching import assign_greedy

        selected = self.pairs if pairs is None else pairs
        threshold = float(self.cfg.match_oks_threshold)
        score_threshold = float(self.cfg.score_threshold_pointwise)
        rows: list[dict[str, Any]] = []
        for p in selected:
            if p.n_gt == 0 or p.n_pred == 0:
                continue
            keep = p.scores >= score_threshold
            matched, sim = assign_greedy(np.where(keep[:, None], p.oks, -1.0), p.scores, threshold)
            for i, g in enumerate(matched):
                if g < 0:
                    continue
                gt_row = int(p.gt_rows[g])
                rows.append({
                    "image_id": p.image_id, "dataset": p.dataset, "gt_row": gt_row,
                    "pred_row": int(p.pred_rows[i]), "oks": float(sim[i]),
                    "keypoint_schema": str(self.gt.loc[gt_row, "keypoint_schema"]),
                })
        return pd.DataFrame(
            rows, columns=["image_id", "dataset", "gt_row", "pred_row", "oks", "keypoint_schema"]
        )

    def bbox_sources(self) -> set[str]:
        """Origins of the bboxes present in the predictions."""
        return set(self.pred["bbox_source"].unique()) if len(self.pred) else set()

    def gt_array(self, key: str, rows: np.ndarray) -> np.ndarray:
        """Extract a list column of the GT as a stacked array."""
        return np.stack(self.gt.loc[rows, key].map(lambda v: np.asarray(v, float)).to_numpy())

    def pred_array(self, key: str, rows: np.ndarray) -> np.ndarray:
        """Extract a list column of the predictions as a stacked array."""
        return np.stack(self.pred.loc[rows, key].map(lambda v: np.asarray(v, float)).to_numpy())


def record(scope: str, metric: str, value: float, n: int) -> dict[str, Any]:
    """Create a row of contract 4 (long format, never wide)."""
    return {"scope": scope, "metric": metric, "value": float(value), "n": int(n)}
