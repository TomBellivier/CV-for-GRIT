"""Approach H: LoRA adapters per insect group (ADR-0036).

"Partial weights per group" category: a trunk common to every order, plus one set of
adapters per order. It is the canonical use of LoRA, and the economical counterpart of
approach B (a whole model per group) — four sets of adapters weigh less than 1 % of the
network where four complete models quadruple it.

Training in TWO PHASES, within a single run:

1. **Common trunk** — the whole model (backbone, neck, heads) is trained on the WHOLE
   train of the fold, WITHOUT adapters. Budget: `epoch_split` of the epochs. This phase
   produces a standard YOLO, which serves as the common starting point.
2. **Adapters per group** — the trunk is reloaded, NEW adapters are injected into it,
   then everything but them is frozen. For each insect order, they are trained on the
   WHOLE train of that order. Budget: the remaining epochs, per group.

Decisive implementation point: the adapters are **injected in phase 2, never in phase
1**. Injecting them from the start would be useless, since saving merges them into the
base weights (ADR-0025): the reloaded trunk would then no longer contain any LoRA layer
to specialise, and the phase 2 freeze would find nothing.

Three protocol points, all deliberate:

- **No data is set aside.** A split specific to this approach (half for phase 1, half
  for phase 2) would break §6.2 and measure the data volume rather than the method. The
  adapters therefore see again images the trunk has already seen — it is the real regime
  of a deployment, where a general model is specialised with the data at hand.
- **The heads are trained in phase 1, frozen in phase 2.** On YOLO26 they weigh ~67 % of
  the parameters: unfreezing them per group would give four almost complete models, and
  the approach would switch to the category of B instead of staying a light
  specialisation.
- **The total epoch budget equals the other approaches'.** `epoch_split=0.6` puts 60 %
  on the trunk and 40 % on the adapters; without this constraint, H would gain compute
  time and not method (§6.3).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from insectpose.approaches.lora import LoraApproach
from insectpose.approaches.yolo_pooled import YoloPooledApproach, release_model
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.data.yolo_export import export_fold
from insectpose.registry import register_approach
from insectpose.training.patching import (
    freeze_patterns_for,
    make_patched_trainer,
    parameter_report,
    pose_trainer_class,
)
from insectpose.utils.device import (
    amp_enabled,
    device_info,
    peak_vram_mb,
    reset_peak_vram,
    resolve_num_workers,
)
from insectpose.utils.logging import get_logger

log = get_logger("lora_per_dataset")

_TRAIN_KEYS = (
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "box",
    "pose", "kobj", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
    "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "close_mosaic",
    "mixup", "erasing", "cos_lr", "dropout",
)


@register_approach("lora_per_dataset")
class LoraPerDatasetApproach(LoraApproach):
    """Shared common trunk, LoRA adapters specialised per insect order."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "lora", "epoch_split",
    )

    def __init__(self, cfg: Any, namespace: str = "") -> None:
        super().__init__(cfg)
        self.namespace = namespace
        self.datasets = [str(d) for d in cfg.data.datasets]
        # One model per group at inference: same trunk, different adapters.
        self.models: dict[str, Any] = {}

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        return LoraApproach.availability()

    # --- split of the epoch budget ----------------------------------------------
    def _epoch_budget(self) -> tuple[int, int]:
        """(trunk epochs, epochs per group). The total equals `train.epochs`.

        The budget is split, never added: otherwise H would have more compute than the
        other approaches and would win through it, not through the method (§6.3).
        """
        total = int(self.cfg.train.epochs)
        split = float(self.cfg.approach.epoch_split)
        if not 0.0 < split < 1.0:
            raise ValueError(
                f"approach.epoch_split must be in ]0, 1[, got {split}. "
                "It splits the epoch budget between the trunk and the adapters."
            )
        stage1 = max(1, round(total * split))
        stage2 = max(1, total - stage1)
        return stage1, stage2

    # --- phase 2 freeze ----------------------------------------------------------
    def _freeze_all_but_adapters(self, model: Any) -> None:
        """Freeze everything but the LoRA adapters — heads included.

        Re-applied AFTER the Ultralytics unfreeze loop (ADR-0028), which would otherwise
        re-enable `requires_grad` on the frozen parameters.
        """
        parameters = dict(model.named_parameters())
        frozen = freeze_patterns_for(parameters, [r"lora_[AB]"])
        if len(frozen) == len(parameters):
            raise RuntimeError(
                "No LoRA layer in the phase 2 model: the injection did not take place. "
                "The saved trunk has its adapters MERGED (ADR-0025), so phase 2 must "
                "INJECT new ones, not hope to find them again."
            )
        for name in frozen:
            parameters[name].requires_grad_(False)

    # --- training ----------------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Phase 1 (common trunk) then phase 2 (adapters per group).

        Never reads data.test. Side effect: writes runs/<run_id>/weights/{trunk,
        <dataset>}/ and runs/<run_id>/yolo_dataset/.
        """
        from ultralytics import YOLO

        self.schema = self._schema(data)
        device = self._device()
        amp = amp_enabled(bool(self.cfg.train.amp), str(self.cfg.mode), device)
        stage1_epochs, stage2_epochs = self._epoch_budget()
        reset_peak_vram()
        started = time.perf_counter()

        # --- phase 1: common trunk, on the WHOLE train of the fold ---
        # No adapter here: saving would merge them into the base weights, and the
        # reloaded trunk would have nothing left to specialise in phase 2.
        trunk_data = export_fold(data, self.schema, ctx.subdir("yolo_dataset/trunk"),
                                 splits=("train", "val"))
        ctx.logger.info("Phase 1: common trunk, %d epoch(s) on %d image(s).",
                        stage1_epochs, len(data.train))
        model = YOLO(str(self.cfg.approach.weights))
        model.train(
            data=str(trunk_data), project=str(ctx.subdir("logs/trunk")), name="train",
            exist_ok=True, seed=ctx.seed("trunk"), device=device, amp=amp,
            deterministic=str(self.cfg.mode) == "debug", verbose=False,
            epochs=stage1_epochs, **self._train_kwargs(),
        )
        trunk_weights = ctx.subdir("weights/trunk") / "best.pt"
        trunk_weights.write_bytes(Path(model.trainer.best).read_bytes())
        stage1_time = time.perf_counter() - started
        release_model(model)

        # --- phase 2: specialised adapters, one set per group ---
        per_dataset_time: dict[str, float] = {}
        for dataset in self.datasets:
            subset = data.filter_dataset(dataset)
            if len(subset.train) == 0:
                raise ValueError(
                    f"[{self.name}] no training image for '{dataset}' in "
                    f"fold {data.fold}."
                )
            group_started = time.perf_counter()
            group_data = export_fold(subset, self.schema,
                                     ctx.subdir(f"yolo_dataset/{dataset}"),
                                     splits=("train", "val"))
            ctx.logger.info("Phase 2 [%s]: adapters only, %d epoch(s) on %d image(s).",
                            dataset, stage2_epochs, len(subset.train))

            group_model = YOLO(str(trunk_weights))
            group_report: dict[str, Any] = {}
            group_model.train(
                data=str(group_data), project=str(ctx.subdir(f"logs/{dataset}")),
                name="train", exist_ok=True, seed=ctx.seed(f"adapters_{dataset}"),
                device=device, amp=amp, deterministic=str(self.cfg.mode) == "debug",
                verbose=False, epochs=stage2_epochs,
                # Each group starts again from NEW adapters injected into the trunk:
                # otherwise the processing order of the groups would influence the result.
                trainer=make_patched_trainer(
                    pose_trainer_class(), patch=self._apply_lora,
                    freeze=self._freeze_all_but_adapters, report=group_report,
                    skip_final_eval=True,
                ),
                **self._train_kwargs(),
            )
            group_weights = ctx.subdir(f"weights/{dataset}") / "best.pt"
            self._write_checkpoint(Path(group_model.trainer.best), group_weights)
            release_model(group_model)
            per_dataset_time[dataset] = time.perf_counter() - group_started
            ctx.extra[f"{dataset}_adapter_report"] = dict(group_report)

        # Reloading for inference: one model per group, identical trunk
        self.models = {d: YOLO(str(ctx.subdir(f"weights/{d}") / "best.pt"))
                       for d in self.datasets}
        for group_model in self.models.values():
            self._prepare_inference_model(group_model)

        first = next(iter(self.models.values()))
        ctx.extra.update({
            "train_time_s": time.perf_counter() - started,
            "trunk_train_time_s": stage1_time,
            "adapter_train_time_s": per_dataset_time,
            "stage1_epochs": stage1_epochs,
            "stage2_epochs_per_group": stage2_epochs,
            "epoch_split": float(self.cfg.approach.epoch_split),
            "n_adapter_sets": len(self.datasets),
            "model_params": int(sum(p.numel() for p in first.model.parameters())),
            "peak_vram_mb": peak_vram_mb(),
            "amp": amp,
            "device": device_info(device),
            "lora_rank": int(self.cfg.approach.lora.r),
            "lora_alpha": self._alpha(),
            "lora_final_report": parameter_report(first.model),
        })

    # --- inference ---------------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Route each image to the model carrying the adapters of its group."""
        if not self.models:
            raise RuntimeError("Models not loaded: call fit() or load() first.")
        frames: list[pd.DataFrame] = []
        for dataset in self.datasets:
            subset = images.filter_dataset(dataset)
            if len(subset) == 0:
                continue
            self.model = self.models[dataset]
            frames.append(YoloPooledApproach.predict_instances(self, subset, ctx))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # --- reloading ---------------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any, namespace: str = "") -> LoraPerDatasetApproach:
        """Reload the N specialised models, without retraining."""
        from ultralytics import YOLO

        obj = cls(cfg, namespace=namespace)
        for dataset in obj.datasets:
            weights = Path(run_dir) / "weights" / dataset / "best.pt"
            if not weights.exists():
                raise FileNotFoundError(f"Weights not found: {weights}")
            obj.models[dataset] = YOLO(str(weights))
            obj._prepare_inference_model(obj.models[dataset])
        return obj

    # --- utilities ---------------------------------------------------------------
    def _train_kwargs(self) -> dict[str, Any]:
        """Hyperparameters common to both phases, all coming from the config.

        `epochs` is excluded: it is split between the phases by `_epoch_budget`.
        """
        approach_cfg = self.cfg.approach
        kwargs: dict[str, Any] = {
            "batch": int(self.cfg.train.batch_size),
            "imgsz": self._imgsz(),
            "workers": resolve_num_workers(self.cfg.train.num_workers),
            "patience": int(self.cfg.train.early_stopping_patience),
            "cache": self.cfg.train.cache,
            "plots": bool(self.cfg.train.plots),
        }
        for key in _TRAIN_KEYS:
            if key in approach_cfg:
                kwargs[key] = approach_cfg[key]
        return kwargs