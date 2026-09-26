"""Internal data models: an image annotation = keypoints (absolute px, read as is,
never modified) + the measurable/non-measurable status of its measurements.

The status is stored per edge (segment between two consecutive kp) and not per
measurement: several measurements can share the same edge (e.g. "femur" and
"leg length"), which makes the greying of an edge cascade to every measurement that
depends on it.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

MEASURABLE = "measurable"
NON_MEASURABLE = "non_measurable"


def edge_key(a, b):
    return tuple(sorted((a, b)))


@dataclass
class ImageAnnotation:
    image_name: str
    image_path: str  # path to the image, as written in the input CSV
    width: int
    height: int
    annotation_id: str = ""
    # display rotation in degrees (0/90/180/270): only affects the view, never the
    # keypoint coordinates, which stay in the frame of the original image.
    image_rotation: int = 0
    keypoints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # sorted (kpA, kpB) -> MEASURABLE | NON_MEASURABLE (manual override of an edge)
    edge_overrides: Dict[Tuple[str, str], str] = field(default_factory=dict)
    done: bool = False  # classification validated by the user

    def key(self):
        """Stable identifier, used to save/restore the classification state
        (a same image can appear twice if it was annotated twice)."""
        return f"{self.image_name}::{self.annotation_id}"

    def has_kp(self, name):
        return name in self.keypoints

    def edge_status(self, key):
        a, b = key
        if not (self.has_kp(a) and self.has_kp(b)):
            return NON_MEASURABLE
        return self.edge_overrides.get(key, MEASURABLE)

    def measurement_status(self, edge_keys):
        """Return MEASURABLE only if every edge of the measurement is."""
        if not edge_keys:
            return NON_MEASURABLE
        return MEASURABLE if all(self.edge_status(k) == MEASURABLE for k in edge_keys) else NON_MEASURABLE


@dataclass
class Preset:
    name: str
    # "kpA::kpB" (sorted) -> forced status
    overrides: Dict[str, str] = field(default_factory=dict)
