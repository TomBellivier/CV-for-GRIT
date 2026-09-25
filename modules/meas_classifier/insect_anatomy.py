"""Anatomical ontology of the classifiers, read from the root kp_infos.yaml.

Keypoint vocabulary, measurement definitions and anatomical parts are declared
once for the whole project, in `kp_infos.yaml` at the repository root; this
module only re-exports them under the names the classifiers use.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kp_infos import (  # noqa: E402,F401  (re-exported)
    BILATERAL_PAIRS,
    BODY_AXIS_CANDIDATES,
    INSECT_GROUPS,
    KP_TO_MEAS,
    KP_TO_PART,
    MEAS_TO_KP,
    MEASUREMENTS,
    PART_TO_KP,
    POINTS,
    SCALE_REFERENCE_MEASURES,
    expand,
    related_entities,
)
