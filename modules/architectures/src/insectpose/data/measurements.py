"""Definitions of the morphometric measurements (ADR-0008).

A measurement = length of the polyline joining a sequence of keypoints. It is the
quantity actually used downstream of the project: the error on the measurements is
therefore a first-class metric, like the OKS.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from insectpose.contracts import ContractError
from insectpose.data.keypoints import KeypointSchema
from insectpose.paths import KP_INFOS_PATH


@dataclass(frozen=True)
class MeasurementSet:
    """Measurements and symmetry pairs attached to a keypoint schema."""

    name: str
    schema_version: int
    keypoint_schema: str
    definitions: dict[str, tuple[str, ...]]
    symmetric_pairs: tuple[tuple[str, str], ...]

    def indices(self, schema: KeypointSchema) -> dict[str, np.ndarray]:
        """Translate the point names into indices of the schema. Fails if a point is missing."""
        out: dict[str, np.ndarray] = {}
        for measure, points in self.definitions.items():
            missing = [p for p in points if p not in schema.names]
            if missing:
                raise ContractError(
                    f"Measurement '{measure}': points missing from the schema '{schema.name}': {missing}."
                )
            out[measure] = np.asarray([schema.index(p) for p in points], dtype=int)
        return out


def measurements_file(value: object = None) -> Path:
    """Measurements file: `value`, or kp_infos.yaml (definition of the repository) if empty."""
    if value is None or str(value) in ("", "None", "null"):
        return KP_INFOS_PATH
    return Path(str(value))


@lru_cache(maxsize=8)
def load_measurements(path: Path | None = None) -> MeasurementSet:
    """Load a measurements file (default: kp_infos.yaml). No side effect."""
    file = measurements_file(path)
    if not file.exists():
        raise FileNotFoundError(
            f"Measurement definitions not found: {file}. "
            "Set eval.measurements.file or disable eval.measurements.enabled."
        )
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    return MeasurementSet(
        name=str(raw["name"]),
        schema_version=int(raw["schema_version"]),
        # kp_infos.yaml declares the schema and its measurements in the same file.
        keypoint_schema=str(raw.get("keypoint_schema", raw["name"])),
        definitions={str(k): tuple(v) for k, v in raw["measurements"].items()},
        symmetric_pairs=tuple((str(a), str(b)) for a, b in raw.get("symmetric_pairs", [])),
    )


def polyline_length(points: np.ndarray) -> np.ndarray:
    """Length of a polyline (..., P, 2): sum of the consecutive segments."""
    p = np.asarray(points, dtype=float)
    return np.linalg.norm(np.diff(p, axis=-2), axis=-1).sum(axis=-1)


def measure_all(kpts: np.ndarray, index: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute every measurement for a batch of instances (N, K, 2), in pixels."""
    arr = np.asarray(kpts, dtype=float)
    return {name: polyline_length(arr[:, idx, :]) for name, idx in index.items()}
