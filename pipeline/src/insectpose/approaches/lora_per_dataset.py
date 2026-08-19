"""Approche H : adaptateurs LoRA par groupe d'insecte (ADR-0036).

Categorie "poids partiels par groupe" : un tronc commun a tous les ordres, plus un jeu
d'adaptateurs par ordre. C'est l'usage canonique de LoRA, et le pendant economique de
l'approche B (un modele entier par groupe) — quatre jeux d'adaptateurs pesent moins de
1 % du reseau la ou quatre modeles complets le quadruplent.

Entrainement en DEUX PHASES, dans un seul run :

1. **Tronc commun** — le modele entier (backbone, cou, tetes) est entraine sur TOUT le
   train du fold, adaptateurs inclus. Budget : `epoch_split` des epoques.
2. **Adaptateurs par groupe** — le tronc et les tetes sont geles ; pour chaque ordre
   d'insecte, un jeu d'adaptateurs neuf est entraine sur TOUT le train de cet ordre.
   Budget : le reste des epoques, par groupe.

Trois points de protocole, tous deliberes :

- **Aucune donnee n'est mise de cote.** Un decoupage propre a cette approche (moitie
  pour la phase 1, moitie pour la phase 2) violerait §6.2 et ferait mesurer le volume
  de donnees plutot que la methode. Les adaptateurs revoient donc des images que le
  tronc a deja vues — c'est le regime reel d'un deploiement, ou l'on specialise un
  modele general avec les donnees dont on dispose.
- **Les tetes sont entrainees en phase 1, gelees en phase 2.** Sur YOLO26 elles pesent
  ~67 % des parametres : les degeler par groupe donnerait quatre modeles presque
  entiers, et l'approche basculerait vers la categorie de B au lieu de rester une
  specialisation legere.
- **Le budget d'epoques total egale celui des autres approches.** `epoch_split=0.6`
  repartit 60 % sur le tronc et 40 % sur les adaptateurs ; sans cette contrainte, H
  gagnerait du temps de calcul et non de la methode (§6.3).
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
from insectpose.utils.device import amp_enabled, device_info, peak_vram_mb, reset_peak_vram
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
    """Tronc commun partage, adaptateurs LoRA specialises par ordre d'insecte."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "lora", "epoch_split",
    )

    def __init__(self, cfg: Any, namespace: str = "") -> None:
        super().__init__(cfg)
        self.namespace = namespace
        self.datasets = [str(d) for d in cfg.data.datasets]
        # Un modele par groupe a l'inference : meme tronc, adaptateurs differents.
        self.models: dict[str, Any] = {}

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        return LoraApproach.availability()

    # --- repartition du budget d'epoques -----------------------------------
    def _epoch_budget(self) -> tuple[int, int]:
        """(epoques du tronc, epoques par groupe). Le total egale `train.epochs`.

        Le budget est reparti, jamais ajoute : sinon H disposerait de plus de calcul
        que les autres approches et gagnerait par la, pas par la methode (§6.3).
        """
        total = int(self.cfg.train.epochs)
        split = float(self.cfg.approach.epoch_split)
        if not 0.0 < split < 1.0:
            raise ValueError(
                f"approach.epoch_split doit etre dans ]0, 1[, recu {split}. "
                "Il repartit le budget d'epoques entre tronc et adaptateurs."
            )
        stage1 = max(1, round(total * split))
        stage2 = max(1, total - stage1)
        return stage1, stage2

    # --- gel de la phase 2 --------------------------------------------------
    def _freeze_all_but_adapters(self, model: Any) -> None:
        """Gele tout sauf les adaptateurs LoRA — tetes comprises.

        Reapplique APRES la boucle de degel d'Ultralytics (ADR-0028), qui reactiverait
        sinon `requires_grad` sur les parametres geles.
        """
        parameters = dict(model.named_parameters())
        for name in freeze_patterns_for(parameters, [r"lora_[AB]"]):
            parameters[name].requires_grad_(False)

    def _reset_adapters(self, model: Any) -> int:
        """Reinitialise les adaptateurs avant de specialiser un nouveau groupe.

        Sans cela, chaque groupe partirait des adaptateurs du groupe precedent et
        l'ordre de traitement influencerait les resultats.
        """
        count = 0
        for module in model.modules():
            if hasattr(module, "reset_lora_parameters"):
                for adapter in getattr(module, "lora_A", {}):
                    module.reset_lora_parameters(adapter, init_lora_weights=True)
                    count += 1
        return count

    # --- entrainement -------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Phase 1 (tronc commun) puis phase 2 (adaptateurs par groupe).

        Ne lit jamais data.test. Effet de bord : ecrit runs/<run_id>/weights/{trunk,
        <dataset>}/ et runs/<run_id>/yolo_dataset/.
        """
        from ultralytics import YOLO

        self.schema = self._schema(data)
        device = self._device()
        amp = amp_enabled(bool(self.cfg.train.amp), str(self.cfg.mode), device)
        stage1_epochs, stage2_epochs = self._epoch_budget()
        reset_peak_vram()
        started = time.perf_counter()

        # --- phase 1 : tronc commun, sur TOUT le train du fold ---
        trunk_data = export_fold(data, self.schema, ctx.subdir("yolo_dataset/trunk"),
                                 splits=("train", "val"))
        ctx.logger.info("Phase 1 : tronc commun, %d epoque(s) sur %d image(s).",
                        stage1_epochs, len(data.train))
        model = YOLO(str(self.cfg.approach.weights))
        report: dict[str, Any] = {}
        model.train(
            data=str(trunk_data), project=str(ctx.subdir("logs/trunk")), name="train",
            exist_ok=True, seed=ctx.seed("trunk"), device=device, amp=amp,
            deterministic=str(self.cfg.mode) == "debug", verbose=False,
            epochs=stage1_epochs,
            # Phase 1 : les adaptateurs sont injectes mais RIEN n'est gele — le tronc
            # et les tetes doivent apprendre le domaine avant qu'on les fige.
            trainer=make_patched_trainer(
                pose_trainer_class(), patch=self._apply_lora, report=report,
                skip_final_eval=True,
            ),
            **self._train_kwargs(),
        )
        trunk_weights = ctx.subdir("weights/trunk") / "best.pt"
        trunk_weights.write_bytes(Path(model.trainer.best).read_bytes())
        stage1_time = time.perf_counter() - started
        release_model(model)

        # --- phase 2 : adaptateurs specialises, un jeu par groupe ---
        per_dataset_time: dict[str, float] = {}
        for dataset in self.datasets:
            subset = data.filter_dataset(dataset)
            if len(subset.train) == 0:
                raise ValueError(
                    f"[{self.name}] aucune image d'entrainement pour '{dataset}' dans le "
                    f"fold {data.fold}."
                )
            group_started = time.perf_counter()
            group_data = export_fold(subset, self.schema,
                                     ctx.subdir(f"yolo_dataset/{dataset}"),
                                     splits=("train", "val"))
            ctx.logger.info("Phase 2 [%s] : adaptateurs seuls, %d epoque(s) sur %d image(s).",
                            dataset, stage2_epochs, len(subset.train))

            group_model = YOLO(str(trunk_weights))
            group_report: dict[str, Any] = {}
            group_model.train(
                data=str(group_data), project=str(ctx.subdir(f"logs/{dataset}")),
                name="train", exist_ok=True, seed=ctx.seed(f"adapters_{dataset}"),
                device=device, amp=amp, deterministic=str(self.cfg.mode) == "debug",
                verbose=False, epochs=stage2_epochs,
                trainer=make_patched_trainer(
                    pose_trainer_class(), patch=self._reset_adapters,
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

        # Rechargement pour l'inference : un modele par groupe, tronc identique
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
            "trunk_report": dict(report),
            "lora_final_report": parameter_report(first.model),
        })

    # --- inference ----------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Route chaque image vers le modele portant les adaptateurs de son groupe."""
        if not self.models:
            raise RuntimeError("Modeles non charges : appeler fit() ou load() d'abord.")
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

    # --- rechargement -------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any, namespace: str = "") -> LoraPerDatasetApproach:
        """Recharge les N modeles specialises, sans reentrainement."""
        from ultralytics import YOLO

        obj = cls(cfg, namespace=namespace)
        for dataset in obj.datasets:
            weights = Path(run_dir) / "weights" / dataset / "best.pt"
            if not weights.exists():
                raise FileNotFoundError(f"Poids introuvables : {weights}")
            obj.models[dataset] = YOLO(str(weights))
            obj._prepare_inference_model(obj.models[dataset])
        return obj

    # --- utilitaires --------------------------------------------------------
    def _train_kwargs(self) -> dict[str, Any]:
        """Hyperparametres communs aux deux phases, tous issus de la config.

        `epochs` est exclu : il est reparti entre les phases par `_epoch_budget`.
        """
        approach_cfg = self.cfg.approach
        kwargs: dict[str, Any] = {
            "batch": int(self.cfg.train.batch_size),
            "imgsz": self._imgsz(),
            "workers": int(self.cfg.train.num_workers),
            "patience": int(self.cfg.train.early_stopping_patience),
            "cache": self.cfg.train.cache,
            "plots": bool(self.cfg.train.plots),
        }
        for key in _TRAIN_KEYS:
            if key in approach_cfg:
                kwargs[key] = approach_cfg[key]
        return kwargs