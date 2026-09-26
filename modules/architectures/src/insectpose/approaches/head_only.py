"""Approach G: training the heads only (ADR-0035).

Backbone and neck entirely frozen, only the detection/pose heads are trained. No
adapter.

This approach is the **essential control of LoRA** (approach D). On YOLO26, the heads
represent about two thirds of the parameters at training time: a LoRA variant that
keeps them trainable therefore trains ~67 % of the network, its adapters weighing only
~1.3 %. Without this control, one cannot tell whether the observed gain comes from the
adapters or simply from retraining the heads.

Three readings become possible when comparing D and G:
- G close to D  -> the adapters bring nothing, only the head retraining counts;
- D clearly above G -> the adapters do bring something;
- G close to A (full training) -> the COCO backbone transfers well, and freezing most of
  the network is enough.
"""

from __future__ import annotations

from typing import Any

from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.registry import register_approach
from insectpose.training.patching import (
    freeze_patterns_for,
    head_index,
    make_patched_trainer,
    parameter_report,
    pose_trainer_class,
)
from insectpose.utils.logging import get_logger

log = get_logger("head_only")


@register_approach("head_only")
class HeadOnlyApproach(YoloPooledApproach):
    """YOLO-pose whose heads only are trained."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "head",
    )

    def _trainable_patterns(self, model: Any) -> list[str]:
        """Patterns of the parameters to leave trainable.

        The number of blocks is computed from the STRUCTURE of the model: the index of the
        head varies with the network size (n/s/m/l) and with the YOLO version.
        """
        names = [name for name, _ in model.named_modules()]
        last = head_index(names)
        depth = int(self.cfg.approach.head.blocks)
        blocks = "|".join(str(i) for i in range(max(last - depth + 1, 0), last + 1))
        return [rf"^model\.({blocks})\."]

    def _freeze(self, model: Any) -> None:
        """Freeze everything but the last blocks.

        Re-applied AFTER the Ultralytics unfreeze loop (ADR-0028), which would otherwise
        re-enable `requires_grad` on the frozen parameters.
        """
        patterns = self._trainable_patterns(model)
        parameters = dict(model.named_parameters())
        for name in freeze_patterns_for(parameters, patterns):
            parameters[name].requires_grad_(False)
        self._patterns = patterns

    def _trainer_class(self, ctx: RunContext) -> Any:
        """Trainer applying the freeze at the right moment of the Ultralytics cycle."""
        report: dict[str, Any] = {}
        ctx.extra["head_report"] = report
        return make_patched_trainer(
            pose_trainer_class(), freeze=self._freeze, report=report,
            # The final evaluation reloads and fuses the checkpoint: useless here, and its
            # metrics are only used for monitoring anyway (§7.1).
            skip_final_eval=True,
        )

    def fit(self, data: Any, ctx: RunContext) -> None:
        """Train then record the share actually trained (§7.2)."""
        super().fit(data, ctx)
        report = ctx.extra.pop("head_report", {})
        ctx.extra.update({f"head_{k}": v for k, v in report.items()})
        ctx.extra["head_blocks"] = int(self.cfg.approach.head.blocks)
        ctx.extra["head_patterns"] = list(getattr(self, "_patterns", []))
        if self.model is not None:
            ctx.extra["head_final_report"] = parameter_report(self.model.model)