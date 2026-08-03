"""Approche B : un modele YOLO-pose par dataset (CONVENTIONS.md §9.2).

Une SEULE approche du point de vue du pipeline : elle encapsule N modeles et route
selon `meta.dataset`. Le pipeline ne voit pas la difference, ce qui garantit que B et
A sont evaluees exactement de la meme facon.

Choix de protocole (ADR-0023) :
- chaque modele repart des poids de base (COCO), pas du modele poule : A et B restent
  independantes, et la question posee est bien "un specialiste vaut-il un generaliste ?" ;
- les hyperparametres sont PARTAGES par les 4 modeles, un trial d'Optuna les entrainant
  tous : le budget d'HPO reste ainsi strictement egal a celui de A (§6.3) ;
- meme nombre d'epoques pour tous les datasets, quel que soit leur effectif.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from insectpose.approaches.base import BaseApproach
from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.registry import register_approach


@register_approach("yolo_per_dataset")
class YoloPerDatasetApproach(BaseApproach):
    """N modeles YOLO-pose, un par ordre d'insecte, routes par dataset."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.datasets = [str(d) for d in cfg.data.datasets]
        # Chaque sous-modele a son propre espace de noms dans le run : poids, export
        # YOLO et journaux sont ranges sous weights/<dataset>/, yolo_dataset/<dataset>/...
        self.models: dict[str, YoloPooledApproach] = {
            dataset: YoloPooledApproach(cfg, namespace=dataset) for dataset in self.datasets
        }

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """Memes dependances que l'approche poulee."""
        return YoloPooledApproach.availability()

    # --- entrainement ------------------------------------------------------
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Entraine un modele par dataset, sur les MEMES folds simplement restreints.

        Aucun decoupage n'est regenere ici (§6.2) : c'est ce qui rend A et B comparables.
        Effet de bord : ecrit runs/<run_id>/{weights,yolo_dataset,logs}/<dataset>/.
        """
        started = time.perf_counter()
        for dataset in self.datasets:
            subset = data.filter_dataset(dataset)
            if len(subset.train) == 0:
                raise ValueError(
                    f"[{self.name}] aucune image d'entrainement pour '{dataset}' dans le "
                    f"fold {data.fold}. Verifier le perimetre data.datasets."
                )
            ctx.logger.info("[%s] %s", dataset, subset.summary())
            self.models[dataset].fit(subset, ctx)

        # Cout de l'approche = somme des couts des modeles ; le detail par dataset
        # reste disponible dans le manifeste sous <dataset>_train_time_s, etc.
        ctx.extra["train_time_s"] = time.perf_counter() - started
        ctx.extra["model_params"] = sum(
            int(ctx.extra.get(f"{dataset}_model_params", 0)) for dataset in self.datasets
        )
        ctx.extra["n_models"] = len(self.datasets)

    # --- inference ---------------------------------------------------------
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Route chaque image vers le modele de son dataset, puis concatene."""
        frames: list[pd.DataFrame] = []
        for dataset in self.datasets:
            subset = images.filter_dataset(dataset)
            if len(subset) == 0:
                continue
            frames.append(self.models[dataset].predict_instances(subset, ctx))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # --- rechargement ------------------------------------------------------
    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> YoloPerDatasetApproach:
        """Recharge les N modeles depuis leurs espaces de noms respectifs."""
        obj = cls(cfg)
        obj.models = {
            dataset: YoloPooledApproach.load(Path(run_dir), cfg, namespace=dataset)
            for dataset in obj.datasets
        }
        return obj