"""Static anatomical ontology shared by the pose model and the classifiers.

Defines the keypoint vocabulary, the measurement definitions (which keypoints a
measurement is computed from), and the anatomical part each keypoint belongs to.
All derived mappings (reverse indexes) are built once at import time.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

INSECT_GROUPS: List[str] = [
    "coleoptera",
    "diptera",
    "hymenoptera",
    "lepidoptera",
]

POINTS: List[str] = [
    "head-top",
    "head-left",
    "head-right",
    "left-eye",
    "right-eye",
    "neck",
    "thorax-left",
    "thorax-right",
    "thorax-bottom",
    "body-left",
    "body-right",
    "body-tip",
    "left-antenna-0",
    "left-antenna-1",
    "left-antenna-2",
    "right-antenna-0",
    "right-antenna-1",
    "right-antenna-2",
    "left-forewing-base",
    "left-forewing-tip",
    "left-forewing-front",
    "left-forewing-rear",
    "right-forewing-base",
    "right-forewing-tip",
    "right-forewing-front",
    "right-forewing-rear",
    "left-hindwing-base",
    "left-hindwing-tip",
    "left-hindwing-front",
    "left-hindwing-rear",
    "right-hindwing-base",
    "right-hindwing-tip",
    "right-hindwing-front",
    "right-hindwing-rear",
    "left-leg-0",
    "left-leg-1",
    "left-leg-2",
    "left-leg-3",
    "right-leg-0",
    "right-leg-1",
    "right-leg-2",
    "right-leg-3",
]

MEAS_TO_KP: Dict[str, List[str]] = {
    "total length": ["head-top", "neck", "thorax-bottom", "body-tip"],
    "head width": ["head-left", "head-right"],
    "head length": ["head-top", "neck"],
    "inter ocular distance": ["right-eye", "left-eye"],
    "right antenna length": ["right-antenna-0", "right-antenna-1", "right-antenna-2"],
    "left antenna length": ["left-antenna-0", "left-antenna-1", "left-antenna-2"],
    "thorax width": ["thorax-left", "thorax-right"],
    "thorax length": ["neck", "thorax-bottom"],
    "abdomen width": ["body-left", "body-right"],
    "abdomen length": ["thorax-bottom", "body-tip"],
    "intertegular distance": ["left-forewing-base", "right-forewing-base"],
    "left hind wing length": ["left-hindwing-base", "left-hindwing-tip"],
    "right hind wing length": ["right-hindwing-base", "right-hindwing-tip"],
    "left hind wing width": ["left-hindwing-front", "left-hindwing-rear"],
    "right hind wing width": ["right-hindwing-front", "right-hindwing-rear"],
    "left fore wing length": ["left-forewing-base", "left-forewing-tip"],
    "right fore wing length": ["right-forewing-base", "right-forewing-tip"],
    "left fore wing width": ["left-forewing-front", "left-forewing-rear"],
    "right fore wing width": ["right-forewing-front", "right-forewing-rear"],
    "left hind leg length": ["left-leg-0", "left-leg-1", "left-leg-2", "left-leg-3"],
    "left hind leg femur length": ["left-leg-0", "left-leg-1"],
    "left hind leg tibia length": ["left-leg-1", "left-leg-2"],
    "left hind leg tarsus length": ["left-leg-2", "left-leg-3"],
    "right hind leg length": ["right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3"],
    "right hind leg femur length": ["right-leg-0", "right-leg-1"],
    "right hind leg tibia length": ["right-leg-1", "right-leg-2"],
    "right hind leg tarsus length": ["right-leg-2", "right-leg-3"],
}

PART_TO_KP: Dict[str, List[str]] = {
    "head": [
        "head-top",
        "head-left",
        "head-right",
        "neck",
        "left-eye",
        "right-eye",
        "right-antenna-0",
        "left-antenna-0",
    ],
    "thorax": [
        "neck",
        "thorax-left",
        "thorax-right",
        "thorax-bottom",
        "left-forewing-base",
        "right-forewing-base",
        "right-hindwing-base",
        "left-hindwing-base",
    ],
    "body": [
        "thorax-bottom",
        "body-left",
        "body-right",
        "body-tip",
        "left-leg-0",
        "right-leg-0",
    ],
    "right-antenna": ["right-antenna-0", "right-antenna-1", "right-antenna-2"],
    "left-antenna": ["left-antenna-0", "left-antenna-1", "left-antenna-2"],
    "left-forewing": [
        "left-forewing-base",
        "left-forewing-tip",
        "left-forewing-front",
        "left-forewing-rear",
    ],
    "right-forewing": [
        "right-forewing-base",
        "right-forewing-tip",
        "right-forewing-front",
        "right-forewing-rear",
    ],
    "left-hindwing": [
        "left-hindwing-base",
        "left-hindwing-tip",
        "left-hindwing-front",
        "left-hindwing-rear",
    ],
    "right-hindwing": [
        "right-hindwing-base",
        "right-hindwing-tip",
        "right-hindwing-front",
        "right-hindwing-rear",
    ],
    "right-leg": ["right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3"],
    "left-leg": ["left-leg-0", "left-leg-1", "left-leg-2", "left-leg-3"],
}

#: Bilateral measurement pairs, used to build left/right asymmetry features.
BILATERAL_PAIRS: List[tuple] = [
    ("left antenna length", "right antenna length"),
    ("left hind wing length", "right hind wing length"),
    ("left hind wing width", "right hind wing width"),
    ("left fore wing length", "right fore wing length"),
    ("left fore wing width", "right fore wing width"),
    ("left hind leg length", "right hind leg length"),
    ("left hind leg femur length", "right hind leg femur length"),
    ("left hind leg tibia length", "right hind leg tibia length"),
    ("left hind leg tarsus length", "right hind leg tarsus length"),
]

#: Keypoint pair defining the antero-posterior body axis, with fallbacks.
BODY_AXIS_CANDIDATES: List[tuple] = [
    ("head-top", "body-tip"),
    ("neck", "body-tip"),
    ("neck", "thorax-bottom"),
]

#: Measurements used as a scale reference, in order of preference.
SCALE_REFERENCE_MEASURES: List[str] = [
    "total length",
    "thorax length",
    "inter ocular distance",
]


def _build_reverse_index(mapping: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Invert a one-to-many mapping, preserving insertion order."""
    reverse: Dict[str, List[str]] = {}
    for key, values in mapping.items():
        for value in values:
            reverse.setdefault(value, [])
            if key not in reverse[value]:
                reverse[value].append(key)
    return reverse


KP_TO_MEAS: Dict[str, List[str]] = {point: [] for point in POINTS}
KP_TO_MEAS.update(_build_reverse_index(MEAS_TO_KP))

KP_TO_PART: Dict[str, List[str]] = {point: [] for point in POINTS}
KP_TO_PART.update(_build_reverse_index(PART_TO_KP))

MEASUREMENTS: List[str] = list(MEAS_TO_KP)


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