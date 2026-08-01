"""Definitions des mesures morphometriques (ADR-0008).

Une mesure = longueur de la polyligne joignant une suite de keypoints. C'est la
grandeur reellement utilisee en aval du projet : l'erreur sur les mesures est donc
une metrique de premier plan, au meme titre que l'OKS.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from insectpose.contracts import ContractError
from insectpose.data.keypoints import KeypointSchema


@dataclass(frozen=True)
class MeasurementSet:
    """Mesures et paires de symetrie associees a un schema de keypoints."""

    name: str
    schema_version: int
    keypoint_schema: str
    definitions: dict[str, tuple[str, ...]]
    symmetric_pairs: tuple[tuple[str, str], ...]

    def indices(self, schema: KeypointSchema) -> dict[str, np.ndarray]:
        """Traduit les noms de points en indices du schema. Echoue si un point manque."""
        out: dict[str, np.ndarray] = {}
        for measure, points in self.definitions.items():
            missing = [p for p in points if p not in schema.names]
            if missing:
                raise ContractError(
                    f"Mesure '{measure}' : points absents du schema '{schema.name}' : {missing}."
                )
            out[measure] = np.asarray([schema.index(p) for p in points], dtype=int)
        return out


@lru_cache(maxsize=8)
def load_measurements(path: Path) -> MeasurementSet:
    """Charge un fichier de mesures. Aucun effet de bord."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Definitions de mesures introuvables : {file}. "
            "Renseigner eval.measurements.file ou desactiver eval.measurements.enabled."
        )
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    return MeasurementSet(
        name=str(raw["name"]),
        schema_version=int(raw["schema_version"]),
        keypoint_schema=str(raw["keypoint_schema"]),
        definitions={str(k): tuple(v) for k, v in raw["measurements"].items()},
        symmetric_pairs=tuple((str(a), str(b)) for a, b in raw.get("symmetric_pairs", [])),
    )


def polyline_length(points: np.ndarray) -> np.ndarray:
    """Longueur d'une polyligne (..., P, 2) : somme des segments consecutifs."""
    p = np.asarray(points, dtype=float)
    return np.linalg.norm(np.diff(p, axis=-2), axis=-1).sum(axis=-1)


def measure_all(kpts: np.ndarray, index: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calcule toutes les mesures pour un lot d'instances (N, K, 2), en pixels."""
    arr = np.asarray(kpts, dtype=float)
    return {name: polyline_length(arr[:, idx, :]) for name, idx in index.items()}
