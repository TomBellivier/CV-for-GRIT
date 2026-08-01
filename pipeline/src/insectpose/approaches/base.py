"""Protocole des approches (CONVENTIONS.md §4.2).

Une approche : `fit` (entraine), `predict` (ecrit un contrat 3), `load` (recharge),
`search_space` (declare ses hyperparametres a Optuna). Elle ne calcule JAMAIS de
metrique et n'ecrit jamais hors de `ctx.run_dir`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from insectpose.context import RunContext
from insectpose.contracts import PREDICTION_SCHEMA_VERSION, ContractError
from insectpose.data.datamodule import FoldData, ImageSet
from insectpose.data.schema import ensure_columns
from insectpose.utils.io import write_parquet


@runtime_checkable
class Approach(Protocol):
    """Interface vue par le pipeline. Aucun autre point de contact n'est autorise."""

    name: str

    def fit(self, data: FoldData, ctx: RunContext) -> None: ...

    def predict(self, images: ImageSet, ctx: RunContext, split: str) -> Path: ...

    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> Approach: ...

    @classmethod
    def search_space(cls, trial: Any, cfg: Any) -> dict[str, Any]: ...


class BaseApproach(ABC):
    """Base commune : gestion du nom, des artefacts et de l'ecriture des predictions."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.name = str(cfg.approach.name)

    # --- a implementer -----------------------------------------------------
    @abstractmethod
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Entraine sur data.train, valide sur data.val. NE DOIT PAS lire data.test."""

    @abstractmethod
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Retourne les predictions brutes, DEJA dans le repere de l'image d'origine.

        Colonnes attendues : image_id, bbox_xywh, bbox_score, kpts_xy, kpts_score,
        keypoint_schema, bbox_source, inference_ms (optionnel).
        """

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """(disponible, raison). Permet a une approche de declarer une dependance lourde.

        Le smoke test ignore proprement une approche indisponible au lieu d'echouer :
        l'absence d'un GPU ou d'un extra pip n'est pas un defaut du socle.
        """
        return True, ""

    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> BaseApproach:
        """Reconstruit un predicteur depuis les artefacts, sans reentrainement."""
        raise NotImplementedError(
            f"{cls.__name__}.load n'est pas implemente : le run ne sera pas rejouable."
        )

    @classmethod
    def search_space(cls, trial: Any, cfg: Any) -> dict[str, Any]:
        """Surcharges proposees a Optuna. Par defaut : lit `approach.search_space` du YAML."""
        from insectpose.tuning.search_spaces import suggest_from_spec

        spec = cfg.approach.get("search_space", {})
        return suggest_from_spec(trial, spec, prefix="approach")

    # --- fourni par la base ------------------------------------------------
    def predict(self, images: ImageSet, ctx: RunContext, split: str) -> Path:
        """Enveloppe `predict_instances`, complete le contrat 3 et ecrit le parquet.

        Effet de bord : ecrit runs/<run_id>/predictions/<split>_fold<k>.parquet.
        """
        started = time.perf_counter()
        raw = self.predict_instances(images, ctx)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if raw.empty:
            raise ContractError(
                f"[{self.name}] aucune prediction produite sur '{split}'. Une approche qui "
                "ne detecte rien doit ecrire un fichier vide explicite, pas planter en aval."
            )
        required = {"image_id", "bbox_xywh", "kpts_xy", "kpts_score", "keypoint_schema"}
        missing = required - set(raw.columns)
        if missing:
            raise ContractError(f"[{self.name}] colonnes manquantes en sortie : {sorted(missing)}")

        df = raw.copy()
        df["run_id"] = ctx.run_id
        df["fold"] = ctx.fold
        df["split"] = split
        df["schema_version"] = PREDICTION_SCHEMA_VERSION
        if "dataset" not in df.columns:
            lookup = images.images.set_index("image_id")["dataset"]
            df["dataset"] = df["image_id"].map(lookup)
        if "bbox_score" not in df.columns:
            df["bbox_score"] = 1.0
        if "bbox_source" not in df.columns:
            df["bbox_source"] = "derived"
        if "inference_ms" not in df.columns:
            df["inference_ms"] = elapsed_ms / max(len(df), 1)
        df["pred_id"] = [f"{ctx.run_id}|{split}|{i}" for i in range(len(df))]

        self._check_in_image(df, images)
        df = ensure_columns(df, "predictions")
        out = ctx.paths.predictions(ctx.run_id, split, ctx.fold)
        return write_parquet(out, df, artifact="predictions")

    @staticmethod
    def _check_in_image(df: pd.DataFrame, images: ImageSet) -> None:
        """Garde-fou anti-oubli de retro-projection (§3.4, §9.3).

        Des keypoints massivement hors image signalent presque toujours des coordonnees
        laissees dans le repere du crop ou en normalise.
        """
        sizes = images.images.set_index("image_id")[["image_width", "image_height"]]
        sample = df.head(200)
        offenders = 0
        for row in sample.itertuples(index=False):
            if row.image_id not in sizes.index:
                continue
            w, h = sizes.loc[row.image_id]
            pts = np.asarray(row.kpts_xy, dtype=float).reshape(-1, 2)
            if pts.size == 0:
                continue
            out_of_bounds = (
                (pts[:, 0] < -0.5 * w) | (pts[:, 0] > 1.5 * w)
                | (pts[:, 1] < -0.5 * h) | (pts[:, 1] > 1.5 * h)
            )
            if out_of_bounds.mean() > 0.5:
                offenders += 1
        if offenders > 0.5 * max(len(sample), 1):
            raise ContractError(
                "Plus de la moitie des predictions sortent largement de l'image. "
                "Cause quasi certaine : coordonnees laissees dans le repere du crop ou "
                "normalisees. Le contrat 3 impose le repere de l'image d'origine."
            )
