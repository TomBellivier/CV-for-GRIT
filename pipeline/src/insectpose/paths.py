"""Construction des chemins du projet (CONVENTIONS.md §2).

SEUL module autorise a fabriquer des chemins. Un `.py` qui concatene un chemin en
dur est un bug. Aucun effet de bord a l'import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    """Racines du projet, resolues en absolu."""

    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    splits: Path
    runs: Path
    results: Path
    reports: Path
    configs: Path

    @classmethod
    def from_config(cls, cfg: Any) -> ProjectPaths:
        """Construit les chemins depuis la section `paths` d'une config Hydra."""
        p = cfg.paths if hasattr(cfg, "paths") else cfg
        root = Path(str(p.root)).resolve()

        def sub(key: str, default: str) -> Path:
            value = getattr(p, key, None)
            return Path(str(value)).resolve() if value is not None else root / default

        return cls(
            root=root,
            data=sub("data", "data"),
            raw=sub("raw", "data/raw"),
            interim=sub("interim", "data/interim"),
            processed=sub("processed", "data/processed"),
            splits=sub("splits", "data/splits"),
            runs=sub("runs", "runs"),
            results=sub("results", "results"),
            reports=sub("reports", "reports"),
            configs=sub("configs", "configs"),
        )

    @classmethod
    def default(cls, root: str | Path = ".") -> ProjectPaths:
        """Chemins standards relatifs a une racine donnee."""
        r = Path(root).resolve()
        return cls(
            root=r, data=r / "data", raw=r / "data/raw", interim=r / "data/interim",
            processed=r / "data/processed", splits=r / "data/splits", runs=r / "runs",
            results=r / "results", reports=r / "reports", configs=r / "configs",
        )

    # --- artefacts ---------------------------------------------------------
    def annotations(self, dataset: str) -> Path:
        """Contrat 1 : annotations canoniques d'un dataset."""
        return self.processed / dataset / "annotations.parquet"

    def raw_dir(self, dataset: str, subdir: str | None = None) -> Path:
        """Repertoire source IMMUABLE d'un dataset."""
        return self.raw / (subdir or dataset)

    def split_file(self, split_id: str) -> Path:
        """Contrat 2 : table des folds."""
        return self.splits / f"{split_id}.parquet"

    def split_meta(self, split_id: str) -> Path:
        """Metadonnees du decoupage (seed, strategie, content_hash)."""
        return self.splits / f"{split_id}.json"

    def run_dir(self, run_id: str) -> Path:
        """Racine des artefacts d'un run. SEULE zone d'ecriture d'une approche."""
        return self.runs / run_id

    def manifest(self, run_id: str) -> Path:
        """Contrat 5. Sa presence signale un run complet (§8.2)."""
        return self.run_dir(run_id) / "manifest.json"

    def predictions(self, run_id: str, split: str, fold: int) -> Path:
        """Contrat 3."""
        return self.run_dir(run_id) / "predictions" / f"{split}_fold{fold}.parquet"

    def metrics(self, run_id: str) -> Path:
        """Contrat 4."""
        return self.run_dir(run_id) / "metrics.parquet"

    def master_results(self) -> Path:
        """Agregat de tous les runs : unique source des tableaux du rapport (§8.3)."""
        return self.results / "master.parquet"

    def optuna_storage(self, study_name: str) -> Path:
        """Base Optuna d'une etude (reprise possible)."""
        return self.runs / "optuna" / f"{study_name}.db"

    def keypoint_schema(self, name: str) -> Path:
        """Fichier de schema de keypoints (§3.1)."""
        return self.configs / "keypoints" / f"{name}.yaml"

    def ensure_writable_dirs(self) -> None:
        """Cree les repertoires d'ecriture autorises. Ne touche jamais a `raw`."""
        for d in (self.interim, self.processed, self.splits, self.runs, self.results, self.reports):
            d.mkdir(parents=True, exist_ok=True)
