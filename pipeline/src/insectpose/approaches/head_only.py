"""Approche G : entrainement des tetes seules (ADR-0035).

Backbone et cou entierement geles, seules les tetes de detection/pose sont entrainees.
Aucun adaptateur.

Cette approche est le **temoin indispensable de LoRA** (approche D). Sur YOLO26, les
tetes representent environ les deux tiers des parametres a l'entrainement : une variante
LoRA qui les laisse entrainables entraine donc ~67 % du reseau, et ses adaptateurs n'en
pesent que ~1,3 %. Sans ce temoin, on ne peut pas savoir si le gain observe vient des
adaptateurs ou simplement du reentrainement des tetes.

Trois lectures deviennent possibles en comparant D et G :
- G proche de D  -> les adaptateurs n'apportent rien, seul le reentrainement des tetes compte ;
- D nettement au-dessus de G -> les adaptateurs apportent bien quelque chose ;
- G proche de A (entrainement complet) -> le backbone COCO transfere bien, et geler
  l'essentiel du reseau suffit.
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
    """YOLO-pose dont seules les tetes sont entrainees."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "head",
    )

    def _trainable_patterns(self, model: Any) -> list[str]:
        """Motifs des parametres a laisser entrainables.

        Le nombre de blocs est calcule depuis la STRUCTURE du modele : l'index de la
        tete varie avec la taille du reseau (n/s/m/l) et avec la version de YOLO.
        """
        names = [name for name, _ in model.named_modules()]
        last = head_index(names)
        depth = int(self.cfg.approach.head.blocks)
        blocks = "|".join(str(i) for i in range(max(last - depth + 1, 0), last + 1))
        return [rf"^model\.({blocks})\."]

    def _freeze(self, model: Any) -> None:
        """Gele tout sauf les derniers blocs.

        Reapplique APRES la boucle de degel d'Ultralytics (ADR-0028), qui reactiverait
        sinon `requires_grad` sur les parametres geles.
        """
        patterns = self._trainable_patterns(model)
        parameters = dict(model.named_parameters())
        for name in freeze_patterns_for(parameters, patterns):
            parameters[name].requires_grad_(False)
        self._patterns = patterns

    def _trainer_class(self, ctx: RunContext) -> Any:
        """Trainer applicant le gel au bon moment du cycle Ultralytics."""
        report: dict[str, Any] = {}
        ctx.extra["head_report"] = report
        return make_patched_trainer(
            pose_trainer_class(), freeze=self._freeze, report=report,
            # L'evaluation finale recharge et fusionne le checkpoint : inutile ici, et
            # ses metriques ne servent de toute facon qu'au monitoring (§7.1).
            skip_final_eval=True,
        )

    def fit(self, data: Any, ctx: RunContext) -> None:
        """Entraine puis enregistre la part reellement entrainee (§7.2)."""
        super().fit(data, ctx)
        report = ctx.extra.pop("head_report", {})
        ctx.extra.update({f"head_{k}": v for k, v in report.items()})
        ctx.extra["head_blocks"] = int(self.cfg.approach.head.blocks)
        ctx.extra["head_patterns"] = list(getattr(self, "_patterns", []))
        if self.model is not None:
            ctx.extra["head_final_report"] = parameter_report(self.model.model)