"""Loader of kp_infos.yaml, the single definition of keypoints and measurements.

Import this module rather than reading the YAML by hand: it exposes the file's
content and every mapping derived from it, built once at import time.

    from kp_infos import KEYPOINT_NAMES, MEASUREMENTS, FLIP_INDEX, related_entities

A module that lives in a sub-folder puts the repository root on `sys.path` first
(see `pipeline/processing/definitions.py`). The `KP_INFOS` environment variable
points at another YAML file, e.g. for a pipeline copied outside the repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

KP_INFOS_PATH = Path(os.environ.get("KP_INFOS", Path(__file__).resolve().parent / "kp_infos.yaml"))

with KP_INFOS_PATH.open(encoding="utf-8") as _f:
    RAW: dict = yaml.safe_load(_f)

SCHEMA_NAME: str = RAW["name"]
SCHEMA_VERSION: int = int(RAW["schema_version"])
INSECT_GROUPS: List[str] = list(RAW["insect_groups"])

# --- keypoints (the order is the one of the model output) --------------------
KEYPOINT_NAMES: List[str] = [kp["name"] for kp in RAW["keypoints"]]
POINTS = KEYPOINT_NAMES
NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)
KEYPOINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
KEYPOINT_DIFFICULTY: List[float] = [float(kp["difficulty"]) for kp in RAW["keypoints"]]
KEYPOINT_SIGMAS: List[float] = [d * float(RAW["sigma_from_difficulty"]["scale"])
                                for d in KEYPOINT_DIFFICULTY]
KEYPOINT_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    kp["name"]: tuple(kp["color"]) for kp in RAW["keypoints"]}
KEYPOINT_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    name: (r, g, b) for name, (b, g, r) in KEYPOINT_COLORS_BGR.items()}
KEYPOINT_LAYOUT: Dict[str, Tuple[float, float]] = {
    kp["name"]: tuple(kp["layout"]) for kp in RAW["keypoints"] if kp.get("layout")}

# FLIP_INDEX[i] = index of the keypoint that keypoint i becomes after a horizontal flip.
FLIP_INDEX: List[int] = [KEYPOINT_INDEX[kp["flip"] or kp["name"]] for kp in RAW["keypoints"]]

SKELETON: List[Tuple[int, int]] = [tuple(edge) for edge in RAW["skeleton"]]
SKELETON_NAMES: List[Tuple[str, str]] = [(KEYPOINT_NAMES[a], KEYPOINT_NAMES[b]) for a, b in SKELETON]

# --- measurements ----------------------------------------------------------------
MEAS_TO_KP: Dict[str, List[str]] = {name: list(chain) for name, chain in RAW["measurements"].items()}
MEASUREMENT_NAMES: List[str] = list(MEAS_TO_KP)
MEASUREMENTS = MEASUREMENT_NAMES
MEASUREMENT_INDICES: Dict[str, List[int]] = {
    name: [KEYPOINT_INDEX[kp] for kp in chain] for name, chain in MEAS_TO_KP.items()}
SYMMETRIC_MEASUREMENT_PAIRS: List[Tuple[str, str]] = [tuple(p) for p in RAW["symmetric_pairs"]]
BILATERAL_PAIRS = SYMMETRIC_MEASUREMENT_PAIRS

# --- anatomy -----------------------------------------------------------------------
PART_TO_KP: Dict[str, List[str]] = {part: list(kps) for part, kps in RAW["parts"].items()}
BODY_AXIS_CANDIDATES: List[Tuple[str, str]] = [tuple(p) for p in RAW["body_axis_candidates"]]
SCALE_REFERENCE_MEASURES: List[str] = list(RAW["scale_reference_measures"])


def _build_reverse_index(mapping: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Invert a one-to-many mapping, preserving insertion order."""
    reverse: Dict[str, List[str]] = {}
    for key, values in mapping.items():
        for value in values:
            reverse.setdefault(value, [])
            if key not in reverse[value]:
                reverse[value].append(key)
    return reverse


KP_TO_MEAS: Dict[str, List[str]] = {point: [] for point in KEYPOINT_NAMES}
KP_TO_MEAS.update(_build_reverse_index(MEAS_TO_KP))

KP_TO_PART: Dict[str, List[str]] = {point: [] for point in KEYPOINT_NAMES}
KP_TO_PART.update(_build_reverse_index(PART_TO_KP))


def expand(values: Sequence[str], mapping: Dict[str, Sequence[str]]) -> List[str]:
    """Expand a list of names through a one-to-many mapping, deduplicated."""
    expanded: List[str] = []
    for value in values:
        for related in mapping.get(value, []):
            if related not in expanded:
                expanded.append(related)
    return expanded


def related_entities(measure: str) -> tuple:
    """Return the keypoints and measurements anatomically related to ``measure``.

    The neighbourhood is defined by walking measurement -> keypoints ->
    anatomical parts -> all keypoints of those parts -> all measurements
    touching those keypoints.
    """
    if not measure:
        return [], []
    direct_points = MEAS_TO_KP.get(measure, [])
    parts = expand(direct_points, KP_TO_PART)
    points = expand(parts, PART_TO_KP)
    measures = expand(points, KP_TO_MEAS)
    return points, measures


def _check() -> None:
    """Refuse an inconsistent file at import rather than a silent wrong measurement."""
    if len(set(KEYPOINT_NAMES)) != NUM_KEYPOINTS:
        raise ValueError(f"{KP_INFOS_PATH}: duplicated keypoint names")
    known = set(KEYPOINT_NAMES)
    for section, mapping in (("measurements", MEAS_TO_KP), ("parts", PART_TO_KP)):
        for key, kps in mapping.items():
            unknown = [kp for kp in kps if kp not in known]
            if unknown:
                raise ValueError(f"{KP_INFOS_PATH}: {section}[{key!r}] uses unknown keypoints {unknown}")
    for a, b in SYMMETRIC_MEASUREMENT_PAIRS:
        if a not in MEAS_TO_KP or b not in MEAS_TO_KP:
            raise ValueError(f"{KP_INFOS_PATH}: symmetric pair ({a!r}, {b!r}) is not a measurement")
    if any(FLIP_INDEX[j] != i for i, j in enumerate(FLIP_INDEX)):
        raise ValueError(f"{KP_INFOS_PATH}: `flip` is not a symmetric left/right mapping")


_check()
