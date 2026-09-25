"""Contrat des adaptateurs de donnees (CONVENTIONS.md §3.2).

Un adaptateur : lit -> convertit -> valide -> ecrit. Il ne filtre pas, n'augmente pas
et ne prend aucune decision methodologique. Les instances douteuses sont conservees
avec un `qc_flags`.
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
    """Squelette commun a tous les adaptateurs."""

    def __init__(self, dataset: str, source_dir: Path, options: dict[str, Any]) -> None:
        self.dataset = dataset
        self.source_dir = Path(source_dir)
        self.options = options

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Lit la source et retourne un DataFrame aux colonnes du contrat 1 (non finalise)."""

    def convert(self) -> pd.DataFrame:
        """Lit, complete, controle qualite et valide. Aucun effet de bord."""
        df = self.read()
        if df.empty:
            raise ValueError(f"[{self.dataset}] aucune annotation lue depuis {self.source_dir}")
        df = ensure_columns(df, "annotations", extra={"dataset": self.dataset})
        df["schema_version"] = ANNOTATION_SCHEMA_VERSION
        if "group_id" not in df.columns or df["group_id"].isna().all():
            # DECISION OPEN-04 : defaut degrade, sans groupement effectif.
            df["group_id"] = df["image_id"]
            log.info(
                "[%s] group_id = image_id (ADR-0011 : une image = un specimen).",
                self.dataset,
            )
        df["qc_flags"] = validate_coordinates_in_image(df)
        flagged = int((df["qc_flags"] != "").sum())
        if flagged:
            log.warning(
                "[%s] %d instances marquees qc_flags (conservees, non filtrees).",
                self.dataset, flagged,
            )
        validate_frame(df, "annotations")
        return df

    def write(self, df: pd.DataFrame, paths: ProjectPaths) -> Path:
        """Ecrit le contrat 1. Effet de bord : data/processed/<dataset>/annotations.parquet."""
        return write_parquet(paths.annotations(self.dataset), df, artifact="annotations")

    def run(self, paths: ProjectPaths) -> Path:
        """convert + write."""
        return self.write(self.convert(), paths)
