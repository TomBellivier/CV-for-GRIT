"""Contract of the data adapters (CONVENTIONS.md §3.2).

An adapter: reads -> converts -> validates -> writes. It does not filter, does not
augment and takes no methodological decision. Doubtful instances are kept with a
`qc_flags`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from insectpose.contracts import ANNOTATION_SCHEMA_VERSION
from insectpose.data.schema import ensure_columns, validate_coordinates_in_image, validate_frame
from insectpose.paths import ProjectPaths
from insectpose.utils.io import write_parquet
from insectpose.utils.logging import get_logger

log = get_logger("adapter")


class BaseAdapter(ABC):
    """Skeleton common to every adapter."""

    def __init__(self, dataset: str, source_dir: Path, options: dict[str, Any]) -> None:
        self.dataset = dataset
        self.source_dir = Path(source_dir)
        self.options = options

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read the source and return a DataFrame with the columns of contract 1 (not final)."""

    def convert(self) -> pd.DataFrame:
        """Read, complete, quality-check and validate. No side effect."""
        df = self.read()
        if df.empty:
            raise ValueError(f"[{self.dataset}] no annotation read from {self.source_dir}")
        df = ensure_columns(df, "annotations", extra={"dataset": self.dataset})
        df["schema_version"] = ANNOTATION_SCHEMA_VERSION
        if "group_id" not in df.columns or df["group_id"].isna().all():
            # DECISION OPEN-04: degraded default, without an actual grouping.
            df["group_id"] = df["image_id"]
            log.info(
                "[%s] group_id = image_id (ADR-0011: one image = one specimen).",
                self.dataset,
            )
        df["qc_flags"] = validate_coordinates_in_image(df)
        flagged = int((df["qc_flags"] != "").sum())
        if flagged:
            log.warning(
                "[%s] %d instances flagged in qc_flags (kept, not filtered).",
                self.dataset, flagged,
            )
        validate_frame(df, "annotations")
        return df

    def write(self, df: pd.DataFrame, paths: ProjectPaths) -> Path:
        """Write contract 1. Side effect: data/processed/<dataset>/annotations.parquet."""
        return write_parquet(paths.annotations(self.dataset), df, artifact="annotations")

    def run(self, paths: ProjectPaths) -> Path:
        """convert + write."""
        return self.write(self.convert(), paths)
