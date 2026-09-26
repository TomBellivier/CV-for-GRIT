"""Approach D: LoRA adapters on a pre-trained YOLO-pose (ADR-0025).

The network starts from the COCO weights. Backbone and neck are **frozen**; LoRA
adapters are injected on the convolutions right before the head, and the
detection/pose heads stay trainable.

What this approach tests: can the performance of a full training be reached while
training only a fraction of the parameters? The manifest therefore records the **number
of trainable parameters**: without it, "LoRA" means nothing, since the same label covers
very different configurations depending on what stays unfrozen next to the adapters.

Ultralytics constraint: the model is rebuilt at the start of `train()` and the manual
freeze is undone there. Everything therefore goes through `training/patching.py`, the
only place that depends on the internals of the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.models.group_norm import replace_modules
from insectpose.registry import register_approach
from insectpose.training.patching import (
    freeze_patterns_for,
    head_index,
    make_patched_trainer,
    match_conv_targets,
    parameter_report,
    pose_trainer_class,
)
from insectpose.utils.logging import get_logger

log = get_logger("lora")


def _lora_layer_class() -> Any:
    """LoRA layer class of the installed peft version."""
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError:  # different layout depending on the versions
        from peft.tuners.lora import LoraLayer
    return LoraLayer


def merge_lora_weights(model: Any) -> int:
    """Merge the adapters into the base weights and remove the wrappers.

    Without this merge, the checkpoint contains `peft.tuners.lora.Conv2d` that do not
    expose the attributes of a convolution (`out_channels`...). Ultralytics then fails as
    soon as it fuses conv+BN, i.e. when loading for inference.

    After the merge, the checkpoint is a perfectly standard YOLO: reloadable, fusable, and
    usable without peft. Returns the number of layers merged.
    """
    lora_cls = _lora_layer_class()

    def _merge(layer: Any) -> Any:
        layer.merge()
        return layer.get_base_layer()

    return replace_modules(
        model,
        is_target=lambda m: isinstance(m, lora_cls),
        make_replacement=_merge,
        # A merged layer becomes a plain convolution again: nothing to skip.
        is_replacement=lambda _m: False,
    )


@register_approach("lora")
class LoraApproach(YoloPooledApproach):
    """Pooled YOLO-pose whose adapters and heads only are trained."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "lora",
    )

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        available, reason = YoloPooledApproach.availability()
        if not available:
            return available, reason
        try:
            import peft  # noqa: F401
        except ImportError:
            return False, "peft missing: pip install -e \".[dev]\""
        return True, ""

    # --- patch of the model -----------------------------------------------------
    def _target_modules(self, model: Any) -> list[str]:
        """Convolutions receiving the adapters: the last block of the neck by default.

        The pattern is computed from the STRUCTURE of the model, not hard-coded: a change
        of network size (n/s/m/l) shifts the block indices.
        """
        import torch

        names = [name for name, _ in model.named_modules()]
        last = head_index(names)
        depth = int(self.cfg.approach.lora.neck_blocks)
        blocks = "|".join(str(i) for i in range(max(last - depth, 0), last))
        pattern = rf"^model\.({blocks})\..*\bconv$"

        convolutions = [
            (name, int(getattr(module, "groups", 1)))
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        targets, skipped = match_conv_targets(convolutions, [pattern])
        if skipped:
            log.info("%d grouped (depthwise) convolution(s) skipped: peft requires a "
                     "rank divisible by `groups`, for no gain.", len(skipped))
        if not targets:
            raise RuntimeError(
                f"No convolution LoRA can adapt (pattern '{pattern}', "
                f"{len(skipped)} depthwise skipped). Increase "
                "approach.lora.neck_blocks to go up to blocks containing standard "
                "convolutions."
            )
        self._lora_skipped = skipped
        return targets

    def _apply_lora(self, model: Any) -> None:
        """Inject the adapters in place. Side effect: modifies `model`."""
        from peft import LoraConfig, inject_adapter_in_model

        targets = self._target_modules(model)
        lora = self.cfg.approach.lora
        alpha = self._alpha()
        config = LoraConfig(
            r=int(lora.r), lora_alpha=alpha, lora_dropout=float(lora.dropout),
            target_modules=targets, bias="none",
        )
        inject_adapter_in_model(config, model)
        log.info("LoRA injected on %d convolution(s), rank %d.", len(targets), int(lora.r))
        self._lora_targets = targets

    def _alpha(self) -> float:
        """Scale factor of the adapters.

        Derived from the rank (`alpha = alpha_ratio x r`) unless `alpha` is set
        explicitly. peft scales the contribution by alpha/r: at a constant ratio, the
        optimal learning rate stays almost independent of the rank, which avoids spending
        a search dimension on a redundancy (ADR-0031).
        """
        lora = self.cfg.approach.lora
        explicit = lora.get("alpha")
        if explicit is not None:
            return float(explicit)
        return float(lora.get("alpha_ratio", 2.0)) * int(lora.r)

    def _freeze(self, model: Any) -> None:
        """Freeze everything but the adapters and the head.

        Re-applied AFTER the Ultralytics freeze loop, which would otherwise re-enable
        `requires_grad` on the frozen parameters (see training/patching.py).
        """
        names = [name for name, _ in model.named_parameters()]
        last = head_index([n for n, _ in model.named_modules()])
        trainable = [r"lora_[AB]"]
        if bool(self.cfg.approach.lora.train_head):
            trainable.append(rf"^model\.{last}\.")
        for name in freeze_patterns_for(names, trainable):
            dict(model.named_parameters())[name].requires_grad = False

    def _trainer_class(self, ctx: RunContext) -> Any:
        """Trainer applying the injection then the freeze, at the right moments."""
        report: dict[str, Any] = {}
        ctx.extra["lora_report"] = report
        return make_patched_trainer(
            pose_trainer_class(), patch=self._apply_lora, freeze=self._freeze, report=report,
            # The checkpoint still contains the LoRA wrappers at this stage: the final
            # evaluation of Ultralytics would reload and fuse it, which fails.
            skip_final_eval=True,
        )

    def _write_checkpoint(self, best: Path, target: Path) -> None:
        """Merge the adapters then write a standard YOLO checkpoint.

        Side effect: writes `target`. The saved model no longer depends on peft.
        """
        import torch

        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        merged = 0
        for key in ("model", "ema"):
            module = checkpoint.get(key)
            if module is not None and hasattr(module, "named_children"):
                merged += merge_lora_weights(module)
        if merged == 0:
            raise RuntimeError(
                "No LoRA layer found in the checkpoint: the injection did not take "
                "place, or the trainer rebuilt the model after the patch."
            )
        log.info("%d LoRA layer(s) merged into the base weights.", merged)
        torch.save(checkpoint, target)

    # --- training ---------------------------------------------------------------
    def fit(self, data: Any, ctx: RunContext) -> None:
        """Train then record the share actually trained (§7.2)."""
        super().fit(data, ctx)
        report = ctx.extra.pop("lora_report", {})
        ctx.extra.update({f"lora_{k}": v for k, v in report.items()})
        ctx.extra["lora_rank"] = int(self.cfg.approach.lora.r)
        ctx.extra["lora_alpha"] = self._alpha()
        ctx.extra["lora_targets"] = getattr(self, "_lora_targets", [])
        ctx.extra["lora_skipped_grouped"] = len(getattr(self, "_lora_skipped", []))
        if self.model is not None:
            ctx.extra["lora_final_report"] = parameter_report(self.model.model)