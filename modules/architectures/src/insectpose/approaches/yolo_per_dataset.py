"""Approach B: one YOLO-pose model per dataset (CONVENTIONS.md §9.2).

A SINGLE approach from the point of view of the pipeline: it wraps N models and routes
by `meta.dataset`. The pipeline does not see the difference, which guarantees that B
and A are evaluated exactly the same way.

Protocol choices (ADR-0023):
- each model starts again from the base weights (COCO), not from the pooled model: A
  and B remain independent, and the question asked is indeed "is a specialist worth a
  generalist?";
- the hyperparameters are SHARED by the 4 models, one Optuna trial training them all:
  the HPO budget thus stays strictly equal to A's (§6.3);
- same number of epochs for every dataset, whatever its size.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from insectpose.approaches.base import BaseApproach
from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.registry import register_approach


@register_approach("yolo_per_dataset")
class YoloPerDatasetApproach(BaseApproach):
    """N YOLO-pose models, one per insect order, routed by dataset."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.datasets = [str(d) for d in cfg.data.datasets]
        # Each sub-model has its own namespace in the run: weights, YOLO export and logs
        # are stored under weights/<dataset>/, yolo_dataset/<dataset>/...
        self.models: dict[str, YoloPooledApproach] = {
            dataset: YoloPooledApproach(cfg, namespace=dataset) for dataset in self.datasets
        }

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """Same dependencies as the pooled approach."""
        return YoloPooledApproach.availability()

    # --- training ---------------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Train one model per dataset, on the SAME folds, simply restricted.

        No split is regenerated here (§6.2): it is what makes A and B comparable.
        Side effect: writes runs/<run_id>/{weights,yolo_dataset,logs}/<dataset>/.
        """
        started = time.perf_counter()
        for dataset in self.datasets:
            subset = data.filter_dataset(dataset)
            if len(subset.train) == 0:
                raise ValueError(
                    f"[{self.name}] no training image for '{dataset}' in "
                    f"fold {data.fold}. Check the data.datasets scope."
                )
            ctx.logger.info("[%s] %s", dataset, subset.summary())
            self.models[dataset].fit(subset, ctx)

        # Cost of the approach = sum of the costs of the models; the per-dataset detail
        # stays available in the manifest under <dataset>_train_time_s, etc.
        ctx.extra["train_time_s"] = time.perf_counter() - started
        ctx.extra["model_params"] = sum(
            int(ctx.extra.get(f"{dataset}_model_params", 0)) for dataset in self.datasets
        )
        ctx.extra["n_models"] = len(self.datasets)

    # --- inference --------------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Route each image to the model of its dataset, then concatenate."""
        frames: list[pd.DataFrame] = []
        for dataset in self.datasets:
            subset = images.filter_dataset(dataset)
            if len(subset) == 0:
                continue
            frames.append(self.models[dataset].predict_instances(subset, ctx))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # --- reloading --------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> YoloPerDatasetApproach:
        """Reload the N models from their respective namespaces."""
        obj = cls(cfg)
        obj.models = {
            dataset: YoloPooledApproach.load(Path(run_dir), cfg, namespace=dataset)
            for dataset in obj.datasets
        }
        return obj