"""Patch of the Ultralytics model before training (ADR-0025, ADR-0026, ADR-0028).

Ultralytics foresees neither LoRA adapters nor conditional normalisation. Both
approaches must therefore modify the `nn.Module` built by the trainer. This module
isolates everything that depends on the Ultralytics internals, so that an update of the
library only breaks one place.

Three internals are used, checked on the sources of the installed version:

1. **Callbacks do not fit.** `on_pretrain_routine_start` fires BEFORE the model is
   built; `on_pretrain_routine_end` AFTER the optimiser and the EMA are created. A patch
   applied at these moments would either be lost or missing from the optimiser. A custom
   trainer is therefore passed (`train(trainer=...)`), and the patch is applied in
   `get_model`, at the very moment of the construction.

2. **Ultralytics unfreezes what we freeze.** Its `freeze` loop sets `requires_grad=True`
   again on every frozen parameter whose name does not match `args.freeze`, emitting
   "setting 'requires_grad=True' for frozen layer '...'". A plain
   `requires_grad=False` is therefore silently undone.

   **The actual order matters**: this loop lives in `_setup_train`, which THEN calls
   `_build_train_pipeline` to build the optimiser. Re-applying the freeze in
   `_build_train_pipeline` — as an earlier version of this module did — therefore came
   BEFORE the unfreeze, and was undone: the training updated the whole network while
   presenting itself as LoRA. The freeze is now applied at the EXIT of `_setup_train`,
   after the unfreeze.

3. **The pipeline can be rebuilt during training** (batch size change, resume), which
   would run the unfreeze loop again. The freeze is therefore re-applied at every epoch
   through `_model_train`.

The count of trainable parameters is logged and recorded in the manifest: it is the only
reliable signal that a patch took. A ratio close to 1 on a LoRA approach immediately
signals that the freeze did not work.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from insectpose.utils.logging import get_logger

log = get_logger("patching")

PatchFn = Callable[[Any], None]


# --- module selection: pure, testable without torch ----------------------------
def match_module_names(names: Iterable[str], patterns: Iterable[str]) -> list[str]:
    """Module names matching at least one regular expression.

    Pure function: it is what decides WHERE the adapters go, and so it is what must be
    tested. The rest of the patch is only torch plumbing.
    """
    compiled = [re.compile(p) for p in patterns]
    return [name for name in names if any(c.search(name) for c in compiled)]


def match_conv_targets(convolutions: Iterable[tuple[str, int]],
                       patterns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Convolutions that LoRA can adapt among those matching the patterns.

    Returns (kept, skipped). A GROUPED convolution (depthwise, `groups > 1`) is skipped:
    peft then requires a rank divisible by `groups`, which would impose a rank of several
    dozens for no gain — a depthwise convolution carries only a handful of parameters.
    The YOLO architectures have some in the neck, hence the need for this filter.
    """
    compiled = [re.compile(p) for p in patterns]
    kept: list[str] = []
    skipped: list[str] = []
    for name, groups in convolutions:
        if not any(c.search(name) for c in compiled):
            continue
        (kept if int(groups) == 1 else skipped).append(name)
    return kept, skipped


def head_index(names: Iterable[str]) -> int:
    """Index of the last `model.<i>` block: the detection/pose head.

    Ultralytics convention: the network is a `Sequential` whose last element is the
    head. Everything before it is backbone + neck.
    """
    indices = {int(m.group(1)) for name in names if (m := re.match(r"model\.(\d+)\.", name))}
    if not indices:
        raise ValueError("No 'model.<i>.' module found: unexpected structure.")
    return max(indices)


def freeze_patterns_for(names: Iterable[str], trainable: Iterable[str]) -> list[str]:
    """Names of the parameters to freeze: all but those matching `trainable`."""
    keep = [re.compile(p) for p in trainable]
    return [name for name in names if not any(c.search(name) for c in keep)]


def parameter_report(model: Any) -> dict[str, Any]:
    """Count of the trainable parameters. To be recorded in the manifest.

    An unexpected ratio is the only reliable signal that a patch did not take: without
    it, a "LoRA" training where everything is trainable would go unnoticed.
    """
    total = trainable = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": round(trainable / max(total, 1), 6),
        "n_trainable_tensors": len(trainable_names),
        "trainable_sample": trainable_names[:8],
    }


# --- Ultralytics integration --------------------------------------------------
def pose_trainer_class() -> Any:
    """YOLO-pose trainer class of the installed version."""
    from ultralytics.models.yolo.pose import PoseTrainer

    return PoseTrainer


def disable_fuse(model: Any) -> None:
    """Neutralise the Ultralytics conv+BN fusion on a patched model.

    The fusion assumes a classic BatchNorm per convolution. It is therefore impossible
    with a conditional normalisation (it would crush N sets of statistics into one) and
    wrong with unmerged adapters (it would only use the base weights). Neutralised, it
    costs a little inference speed and changes no result.
    """
    import types

    target = getattr(model, "model", model)
    target.fuse = types.MethodType(lambda self, verbose=True: self, target)  # noqa: ARG005


def make_patched_trainer(base_cls: Any, patch: PatchFn | None = None,
                         freeze: PatchFn | None = None,
                         on_batch: Callable[[Any, Any], Any] | None = None,
                         report: dict[str, Any] | None = None,
                         skip_final_eval: bool = False) -> Any:
    """Derived trainer applying a patch to the model, then a freeze after the unfreeze.

    - `patch` runs when the model is built (`get_model`);
    - `freeze` runs at the EXIT of `_setup_train`, hence after the Ultralytics unfreeze
      loop, then at every epoch through `_model_train` — the pipeline can be rebuilt
      along the way;
    - `on_batch` runs at every preprocessed batch, **on the training AND on the
      validation side**: the validator has its own `preprocess` and does not go through
      the trainer's;
    - `skip_final_eval` disables the final evaluation, which reloads the best checkpoint
      and FUSES it: on a patched model, the fusion fails or biases the result. Its
      metrics are only used for monitoring anyway (§7.1).
    """

    class _PatchedTrainer(base_cls):  # type: ignore[misc, valid-type]
        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if patch is not None:
                patch(model)
                log.info("Patch applied to the model when it was built.")
            return model

        def _setup_train(self) -> Any:
            result = super()._setup_train()
            # The Ultralytics unfreeze loop has just run: our freeze must come AFTER it,
            # otherwise it is undone without any effect.
            self._apply_insectpose_freeze()
            return result

        def _apply_insectpose_freeze(self) -> None:
            """Apply the freeze and log the share that is actually trainable."""
            if freeze is None:
                return
            freeze(self.model)
            summary = parameter_report(self.model)
            log.info("Trainable parameters: %d / %d (%.2f %%)",
                     summary["trainable_params"], summary["total_params"],
                     100 * summary["trainable_ratio"])
            if report is not None:
                report.update(summary)
            if summary["trainable_params"] == 0:
                raise RuntimeError(
                    "No trainable parameter after the patch: check the selection "
                    "patterns (the training would do nothing)."
                )

        def _model_train(self) -> Any:
            # Re-applied at every epoch: Ultralytics can rebuild the pipeline during
            # training, which would run the unfreeze loop again.
            result = super()._model_train()
            if freeze is not None:
                freeze(self.model)
            return result

        def preprocess_batch(self, batch: Any) -> Any:
            processed = super().preprocess_batch(batch)
            if on_batch is not None:
                on_batch(self, processed)
            return processed

        def get_validator(self) -> Any:
            validator = super().get_validator()
            if on_batch is None:
                return validator
            # The validator has its OWN preprocess: without this relay, the context would
            # keep the indices of the last training batch while facing a validation
            # batch of a different size.
            original = validator.preprocess
            trainer = self

            def _preprocess(batch: Any) -> Any:
                processed = original(batch)
                on_batch(trainer, processed)
                return processed

            validator.preprocess = _preprocess
            return validator

        def final_eval(self) -> Any:
            if skip_final_eval:
                log.info("Ultralytics final evaluation skipped (patched model): "
                         "its metrics are only used for monitoring (§7.1).")
                return None
            return super().final_eval()

    _PatchedTrainer.__name__ = f"Patched{base_cls.__name__}"
    return _PatchedTrainer