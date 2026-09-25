"""
definitions.py
==============

Keypoints, measurements and anatomy of the pipeline, read from the single
definition of the project: `kp_infos.yaml` at the repository root (loaded by
`kp_infos.py`, next to it). Nothing is declared here any more.

    * KEYPOINT_NAMES      -> the canonical keypoint order. THIS ORDER MUST MATCH the
                             order the model was trained with, because YOLO returns
                             keypoints as an array indexed by this order.
    * KEYPOINT_COLORS     -> RGB colour of each keypoint (overlays).
    * MEASUREMENTS        -> for each measurement, the ordered list of keypoints
                             whose consecutive segments are summed.
    * FLIP_INDEX          -> permutation that swaps left/right keypoints, used by
                             the horizontal-flip test-time augmentation.

A pipeline copied outside the repository takes `kp_infos.py` + `kp_infos.yaml`
with it (in the folder above `pipeline/`), or sets the `KP_INFOS` environment
variable to the YAML file.
"""

import sys

from . import config

if str(config.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(config.REPO_ROOT))

from kp_infos import (  # noqa: E402,F401  (re-exported)
    FLIP_INDEX,
    INSECT_GROUPS,
    KEYPOINT_COLORS_RGB as KEYPOINT_COLORS,
    KEYPOINT_INDEX,
    KEYPOINT_NAMES,
    MEAS_TO_KP as MEASUREMENTS,
    MEASUREMENT_INDICES,
    MEASUREMENT_NAMES,
    NUM_KEYPOINTS,
    SYMMETRIC_MEASUREMENT_PAIRS,
    related_entities,
)
