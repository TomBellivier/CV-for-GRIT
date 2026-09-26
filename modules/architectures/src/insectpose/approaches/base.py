"""Protocol of the approaches (CONVENTIONS.md §4.2).

An approach: `fit` (trains), `predict` (writes a contract 3), `load` (reloads),
`search_space` (declares its hyperparameters to Optuna). It NEVER computes a metric and
never writes outside `ctx.run_dir`.
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
    """Interface seen by the pipeline. No other contact point is allowed."""

    name: str

    def fit(self, data: FoldData, ctx: RunContext) -> None: ...

    def predict(self, images: ImageSet, ctx: RunContext, split: str) -> Path: ...

    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> Approach: ...

    @classmethod
    def search_space(cls, trial: Any, cfg: Any) -> dict[str, Any]: ...


class BaseApproach(ABC):
    """Common base: handling of the name, the artefacts and the writing of the predictions."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.name = str(cfg.approach.name)

    # --- to implement ---------------------------------------------------------
    @abstractmethod
    def fit(self, data: FoldData, ctx: RunContext) -> None:
        """Train on data.train, validate on data.val. MUST NOT read data.test."""

    @abstractmethod
    def predict_instances(self, images: ImageSet, ctx: RunContext) -> pd.DataFrame:
        """Return the raw predictions, ALREADY in the frame of the original image.

        Expected columns: image_id, bbox_xywh, bbox_score, kpts_xy, kpts_score,
        keypoint_schema, bbox_source, inference_ms (optional).
        """

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """(available, reason). Lets an approach declare a heavy dependency.

        The smoke test cleanly skips an unavailable approach instead of failing: the
        absence of a GPU or of a pip extra is not a defect of the framework.
        """
        return True, ""

    @classmethod
    def load(cls, run_dir: Path, cfg: Any) -> BaseApproach:
        """Rebuild a predictor from the artefacts, without retraining."""
        raise NotImplementedError(
            f"{cls.__name__}.load is not implemented: the run will not be replayable."
        )

    @classmethod
    def search_space(cls, trial: Any, cfg: Any) -> dict[str, Any]:
        """Overrides proposed to Optuna. By default: reads `approach.search_space` from the YAML."""
        from insectpose.tuning.search_spaces import suggest_from_spec

        spec = cfg.approach.get("search_space", {})
        return suggest_from_spec(trial, spec, prefix="approach")

    # --- provided by the base -------------------------------------------------
    def predict(self, images: ImageSet, ctx: RunContext, split: str) -> Path:
        """Wrap `predict_instances`, complete contract 3 and write the parquet.

        Side effect: writes runs/<run_id>/predictions/<split>_fold<k>.parquet.
        """
        started = time.perf_counter()
        raw = self.predict_instances(images, ctx)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if raw.empty:
            # Zero prediction is a RESULT, not an error: an under-trained or badly tuned
            # model detects nothing, and the evaluation must measure it (zero OKS, zero
            # coverage) rather than interrupt the pipeline. A compliant but empty file is
            # therefore written, and loudly logged.
            ctx.logger.warning(
                "[%s] NO prediction on '%s' (%d image(s)). The metrics of this split "
                "will be zero. Usual causes: under-trained model, confidence threshold "
                "too high, or malformed labels.", self.name, split, len(images),
            )
            return self._write_empty(ctx, split)

        required = {"image_id", "bbox_xywh", "kpts_xy", "kpts_score", "keypoint_schema"}
        missing = required - set(raw.columns)
        if missing:
            raise ContractError(f"[{self.name}] missing output columns: {sorted(missing)}")

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

    def _write_empty(self, ctx: RunContext, split: str) -> Path:
        """Write a predictions file that is empty but complies with contract 3.

        Side effect: writes runs/<run_id>/predictions/<split>_fold<k>.parquet.
        """
        from insectpose.contracts import all_columns

        empty = pd.DataFrame({name: pd.Series(dtype="object")
                              for name in all_columns("predictions")})
        out = ctx.paths.predictions(ctx.run_id, split, ctx.fold)
        return write_parquet(out, empty, artifact="predictions")

    @staticmethod
    def _check_in_image(df: pd.DataFrame, images: ImageSet) -> None:
        """Guard against a forgotten back-projection (§3.4, §9.3).

        Keypoints massively outside the image almost always signal coordinates left in
        the frame of the crop or normalised.
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
                "More than half of the predictions fall far outside the image. "
                "Almost certain cause: coordinates left in the frame of the crop or "
                "normalised. Contract 3 imposes the frame of the original image."
            )