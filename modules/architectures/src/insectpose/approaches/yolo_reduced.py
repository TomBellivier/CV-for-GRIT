"""Approach F: pooled YOLO-pose deprived of some keypoints (ADR-0027).

Variant of approach A where the legs and hind wings are **removed from the training
labels** (`vis = 0`). Question asked: does the capacity of the network, freed from the
hardest and most mobile points, improve the placement of the others?

Essential reading precaution: the ground truth still contains these points, and the
evaluation counts them. The `overall` metrics of F are therefore **mechanically worse**
than A's and are not comparable. The valid comparison is on the `keypoint:*` scopes of
the KEPT points:

    python scripts/compare_models.py --exclude-keypoints leg hindwing
"""

from __future__ import annotations

from typing import Any

from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData
from insectpose.data.keypoints import KeypointSchema
from insectpose.registry import register_approach
from insectpose.utils.logging import get_logger

log = get_logger("yolo_reduced")


def dropped_indices(schema: KeypointSchema, patterns: list[str]) -> list[int]:
    """Indices of the keypoints whose name contains one of the patterns. Pure function."""
    return [i for i, name in enumerate(schema.names)
            if any(str(p).lower() in name.lower() for p in patterns)]


def mask_keypoints(annotations: Any, indices: list[int]) -> Any:
    """Copy of the annotations with `vis = 0` on the given indices.

    The coordinates are kept as they are: the visibility drives the supervision, and a
    point with vis=0 is masked in the loss, never learnt as zero.
    """
    if not indices:
        return annotations
    frame = annotations.copy()
    frame["kpts_vis"] = frame["kpts_vis"].map(
        lambda v: [0 if i in set(indices) else int(x) for i, x in enumerate(v)]
    )
    return frame


@register_approach("yolo_pooled_reduced")
class YoloPooledReducedApproach(YoloPooledApproach):
    """Pooled YOLO-pose trained without supervision on a subset of the keypoints."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "drop_keypoints",
    )

    def _prepare_data(self, data: FoldData, ctx: RunContext) -> FoldData:
        """Mask the excluded keypoints in train and val, never in test.

        The test stays intact: it is the reference ground truth, common to every
        approach. Masking the test as well would amount to changing the metric.
        """
        schema = self._schema(data)
        patterns = [str(p) for p in self.cfg.approach.drop_keypoints]
        indices = dropped_indices(schema, patterns)
        if not indices:
            raise ValueError(
                f"No keypoint matches {patterns} in the schema "
                f"'{schema.name}': the approach would be identical to yolo_pooled."
            )
        names = [schema.names[i] for i in indices]
        log.info("%d keypoint(s) removed from the training labels: %s",
                 len(indices), ", ".join(names))
        ctx.extra["dropped_keypoints"] = names
        ctx.extra["n_supervised_keypoints"] = schema.n_keypoints - len(indices)

        from dataclasses import replace as dataclass_replace

        return dataclass_replace(
            data,
            train=dataclass_replace(data.train,
                                    annotations=mask_keypoints(data.train.annotations, indices)),
            val=dataclass_replace(data.val,
                                  annotations=mask_keypoints(data.val.annotations, indices)),
        )


__all__ = ["YoloPooledReducedApproach", "dropped_indices", "mask_keypoints"]