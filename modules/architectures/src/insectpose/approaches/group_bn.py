"""Approach E: BatchNorm conditioned on the insect group (ADR-0026).

Full training from the COCO weights, but each BatchNorm is duplicated into N copies —
one per dataset, statistics AND affine parameters. The convolution weights stay shared:
the hypothesis tested is that the difference between insect orders largely lies in
activation statistics, not in different filters.

Batches are mixed; the group of each image is derived from the exported file name
(`<dataset>__<stem>`). At inference, the dataset is always known (ADR-0014) and an
unknown group raises an explicit error rather than a guessed fallback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import ImageSet
from insectpose.models.group_norm import (
    CONTEXT,
    active_group,
    dataset_indices_from_paths,
    default_datasets,
    replace_batchnorm,
)
from insectpose.registry import register_approach
from insectpose.training.patching import (
    disable_fuse,
    make_patched_trainer,
    pose_trainer_class,
)
from insectpose.utils.logging import get_logger

log = get_logger("group_bn")


@register_approach("group_bn")
class GroupBatchNormApproach(YoloPooledApproach):
    """Pooled YOLO-pose whose normalisations are conditioned by dataset."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "group_norm",
    )

    def __init__(self, cfg: Any, namespace: str = "") -> None:
        super().__init__(cfg, namespace)
        self.datasets = default_datasets(cfg)

    # --- patch of the model -----------------------------------------------------
    def _patch(self, model: Any) -> None:
        """Replace the BatchNorm2d. Side effect: modifies `model`."""
        replaced = replace_batchnorm(model, len(self.datasets))
        if replaced == 0:
            raise RuntimeError(
                "No BatchNorm2d found: the approach would have no effect. Check the "
                "architecture of the starting model."
            )
        self._n_replaced = replaced

    def _on_batch(self, trainer: Any, batch: Any) -> None:  # noqa: ARG002
        """Fill the group of each image of the batch before the forward pass."""
        files = batch.get("im_file") if isinstance(batch, dict) else None
        if not files:
            raise RuntimeError(
                "The batch carries no 'im_file': impossible to determine the dataset of "
                "each image. The per-group normalisation cannot work."
            )
        CONTEXT.set(dataset_indices_from_paths(list(files), self.datasets))

    def _trainer_class(self, ctx: RunContext) -> Any:  # noqa: ARG002
        return make_patched_trainer(
            pose_trainer_class(), patch=self._patch, on_batch=self._on_batch,
            # The final evaluation reloads and fuses the model: impossible with a
            # conditional normalisation, and without a group context anyway.
            skip_final_eval=True,
        )

    def _prepare_inference_model(self, model: Any) -> None:
        """Neutralise the conv+BN fusion, incompatible with N sets of statistics."""
        disable_fuse(model)

    # --- training ---------------------------------------------------------------
    def fit(self, data: Any, ctx: RunContext) -> None:
        super().fit(data, ctx)
        ctx.extra["group_norm_groups"] = self.datasets
        ctx.extra["group_norm_layers"] = int(getattr(self, "_n_replaced", 0))
        CONTEXT.clear()

    # --- inference --------------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Predict dataset by dataset, each group setting its normalisation.

        The grouping is not an optimisation: it is the only way to declare the active
        group, since the information does not exist at the layer level.
        """
        frames: list[pd.DataFrame] = []
        for index, dataset in enumerate(self.datasets):
            subset = images.filter_dataset(dataset)
            if len(subset) == 0:
                continue
            with active_group(index):
                frames.append(super().predict_instances(subset, ctx))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)