"""Approche E : BatchNorm conditionnee par groupe d'insecte (ADR-0026).

Entrainement complet depuis les poids COCO, mais chaque BatchNorm est dupliquee en N
copies — une par dataset, statistiques ET parametres affines. Les poids convolutifs
restent partages : l'hypothese testee est que la difference entre ordres d'insectes
tient largement a des statistiques d'activation, pas a des filtres differents.

Les lots sont mixtes ; le groupe de chaque image est deduit du nom de fichier exporte
(`<dataset>__<stem>`). A l'inference, le dataset est toujours connu (ADR-0014) et un
groupe inconnu leve une erreur explicite plutot qu'un repli devine.
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
    """YOLO-pose poule dont les normalisations sont conditionnees par dataset."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "group_norm",
    )

    def __init__(self, cfg: Any, namespace: str = "") -> None:
        super().__init__(cfg, namespace)
        self.datasets = default_datasets(cfg)

    # --- patch du modele ---------------------------------------------------
    def _patch(self, model: Any) -> None:
        """Remplace les BatchNorm2d. Effet de bord : modifie `model`."""
        replaced = replace_batchnorm(model, len(self.datasets))
        if replaced == 0:
            raise RuntimeError(
                "Aucune BatchNorm2d trouvee : l'approche serait sans effet. Verifier "
                "l'architecture du modele de depart."
            )
        self._n_replaced = replaced

    def _on_batch(self, trainer: Any, batch: Any) -> None:  # noqa: ARG002
        """Renseigne le groupe de chaque image du lot avant le forward."""
        files = batch.get("im_file") if isinstance(batch, dict) else None
        if not files:
            raise RuntimeError(
                "Le lot ne porte pas 'im_file' : impossible de determiner le dataset de "
                "chaque image. La normalisation par groupe ne peut pas fonctionner."
            )
        CONTEXT.set(dataset_indices_from_paths(list(files), self.datasets))

    def _trainer_class(self, ctx: RunContext) -> Any:  # noqa: ARG002
        return make_patched_trainer(
            pose_trainer_class(), patch=self._patch, on_batch=self._on_batch,
            # L'evaluation finale recharge et fusionne le modele : impossible avec une
            # normalisation conditionnelle, et sans contexte de groupe de toute facon.
            skip_final_eval=True,
        )

    def _prepare_inference_model(self, model: Any) -> None:
        """Neutralise la fusion conv+BN, incompatible avec N jeux de statistiques."""
        disable_fuse(model)

    # --- entrainement ------------------------------------------------------
    def fit(self, data: Any, ctx: RunContext) -> None:
        super().fit(data, ctx)
        ctx.extra["group_norm_groups"] = self.datasets
        ctx.extra["group_norm_layers"] = int(getattr(self, "_n_replaced", 0))
        CONTEXT.clear()

    # --- inference ---------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Predit dataset par dataset, chaque groupe fixant sa normalisation.

        Le regroupement n'est pas une optimisation : c'est la seule facon de declarer
        le groupe actif, puisque l'information n'existe pas au niveau des couches.
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