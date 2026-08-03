"""Approche D : adaptateurs LoRA sur un YOLO-pose pre-entraine (ADR-0025).

Le reseau part des poids COCO. Backbone et cou sont **geles** ; des adaptateurs LoRA
sont injectes sur les convolutions situees juste apres le cou, et les tetes de
detection/pose restent entrainables.

Ce que cette approche teste : peut-on atteindre la performance d'un entrainement
complet en n'entrainant qu'une fraction des parametres ? Le manifeste enregistre donc
le **nombre de parametres entrainables** : sans lui, "LoRA" ne veut rien dire, puisque
la meme etiquette recouvre des configurations tres differentes selon ce qui reste
degele a cote des adaptateurs.

Contrainte Ultralytics : le modele est reconstruit au debut de `train()` et le gel
manuel y est annule. Tout passe donc par `training/patching.py`, seul endroit qui
depend des internes de la bibliotheque.
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
    match_module_names,
    parameter_report,
    pose_trainer_class,
)
from insectpose.utils.logging import get_logger

log = get_logger("lora")


def _lora_layer_class() -> Any:
    """Classe de couche LoRA de la version de peft installee."""
    try:
        from peft.tuners.lora.layer import LoraLayer
    except ImportError:  # organisation differente selon les versions
        from peft.tuners.lora import LoraLayer
    return LoraLayer


def merge_lora_weights(model: Any) -> int:
    """Fusionne les adaptateurs dans les poids de base et retire les enveloppes.

    Sans cette fusion, le point de sauvegarde contient des `peft.tuners.lora.Conv2d`
    qui n'exposent pas les attributs d'une convolution (`out_channels`...). Ultralytics
    echoue alors des qu'il fusionne conv+BN, c'est-a-dire au chargement pour inference.

    Apres fusion, le checkpoint est un YOLO parfaitement standard : rechargeable,
    fusionnable, et exploitable sans peft. Retourne le nombre de couches fusionnees.
    """
    lora_cls = _lora_layer_class()

    def _merge(layer: Any) -> Any:
        layer.merge()
        return layer.get_base_layer()

    return replace_modules(
        model,
        is_target=lambda m: isinstance(m, lora_cls),
        make_replacement=_merge,
        # Une couche fusionnee redevient une convolution simple : rien a ignorer.
        is_replacement=lambda _m: False,
    )


@register_approach("lora")
class LoraApproach(YoloPooledApproach):
    """YOLO-pose poule dont seuls les adaptateurs et les tetes sont entraines."""

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
            return False, "peft absent : pip install -e \".[dev]\""
        return True, ""

    # --- patch du modele ---------------------------------------------------
    def _target_modules(self, model: Any) -> list[str]:
        """Convolutions recevant les adaptateurs : le dernier bloc du cou par defaut.

        Le motif est calcule depuis la STRUCTURE du modele, pas ecrit en dur : un
        changement de taille de reseau (n/s/m/l) decale les index de blocs.
        """
        names = [name for name, _ in model.named_modules()]
        last = head_index(names)
        depth = int(self.cfg.approach.lora.neck_blocks)
        blocks = "|".join(str(i) for i in range(max(last - depth, 0), last))
        pattern = rf"^model\.({blocks})\..*\bconv$"
        targets = match_module_names(names, [pattern])
        if not targets:
            raise RuntimeError(
                f"Aucune convolution cible pour LoRA (motif '{pattern}'). Verifier "
                "approach.lora.neck_blocks ou la structure du modele."
            )
        return targets

    def _apply_lora(self, model: Any) -> None:
        """Injecte les adaptateurs en place. Effet de bord : modifie `model`."""
        from peft import LoraConfig, inject_adapter_in_model

        targets = self._target_modules(model)
        lora = self.cfg.approach.lora
        config = LoraConfig(
            r=int(lora.r), lora_alpha=float(lora.alpha), lora_dropout=float(lora.dropout),
            target_modules=targets, bias="none",
        )
        inject_adapter_in_model(config, model)
        log.info("LoRA injecte sur %d convolution(s), rang %d.", len(targets), int(lora.r))
        self._lora_targets = targets

    def _freeze(self, model: Any) -> None:
        """Gele tout sauf les adaptateurs et la tete.

        Reapplique APRES la boucle de gel d'Ultralytics, qui reactiverait sinon
        `requires_grad` sur les parametres geles (cf. training/patching.py).
        """
        names = [name for name, _ in model.named_parameters()]
        last = head_index([n for n, _ in model.named_modules()])
        trainable = [r"lora_[AB]"]
        if bool(self.cfg.approach.lora.train_head):
            trainable.append(rf"^model\.{last}\.")
        for name in freeze_patterns_for(names, trainable):
            dict(model.named_parameters())[name].requires_grad = False

    def _trainer_class(self, ctx: RunContext) -> Any:
        """Trainer applicant l'injection puis le gel, aux bons moments."""
        report: dict[str, Any] = {}
        ctx.extra["lora_report"] = report
        return make_patched_trainer(
            pose_trainer_class(), patch=self._apply_lora, freeze=self._freeze, report=report,
            # Le checkpoint contient encore les enveloppes LoRA a ce stade : l'evaluation
            # finale d'Ultralytics le rechargerait et le fusionnerait, ce qui echoue.
            skip_final_eval=True,
        )

    def _write_checkpoint(self, best: Path, target: Path) -> None:
        """Fusionne les adaptateurs puis ecrit un checkpoint YOLO standard.

        Effet de bord : ecrit `target`. Le modele sauvegarde ne depend plus de peft.
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
                "Aucune couche LoRA trouvee dans le checkpoint : l'injection n'a pas eu "
                "lieu, ou le trainer a reconstruit le modele apres le patch."
            )
        log.info("%d couche(s) LoRA fusionnee(s) dans les poids de base.", merged)
        torch.save(checkpoint, target)

    # --- entrainement ------------------------------------------------------
    def fit(self, data: Any, ctx: RunContext) -> None:
        """Entraine puis enregistre la part reellement entrainee (§7.2)."""
        super().fit(data, ctx)
        report = ctx.extra.pop("lora_report", {})
        ctx.extra.update({f"lora_{k}": v for k, v in report.items()})
        ctx.extra["lora_rank"] = int(self.cfg.approach.lora.r)
        ctx.extra["lora_targets"] = getattr(self, "_lora_targets", [])
        if self.model is not None:
            ctx.extra["lora_final_report"] = parameter_report(self.model.model)