"""
pose_inference.py
=================

Thin wrapper around the trained YOLO-pose model:

    * load exactly one pose model (the one named in config.py),
    * run it on one image,
    * pick the instance to measure when several insects are present,
    * return that instance's keypoints as a clean (NUM_KEYPOINTS, 3) array.

No training happens here -- the model is only loaded and queried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from . import config
from .definitions import NUM_KEYPOINTS


@dataclass
class PoseResult:
    """Everything the rest of the pipeline needs about one detected insect."""
    keypoints: np.ndarray        # (NUM_KEYPOINTS, 3) -> x, y, confidence
    detection_conf: float        # YOLO box score of this instance
    box_xyxy: np.ndarray         # (4,) bounding box
    n_instances: int             # how many instances were detected on the image


def load_pose_model(model_path=None) -> YOLO:
    """Load the single pose model declared in the config.

    The class / keypoint names embedded in the weights are printed so you can
    sanity-check that the model order matches definitions.KEYPOINT_NAMES.
    """
    path = str(model_path or config.POSE_MODEL_PATH)
    print(f"[pose] loading pose model: {path}")
    model = YOLO(path)
    # Ultralytics stores the detection class names in model.names.
    print(f"[pose] model classes: {getattr(model, 'names', 'unknown')}")
    return model


def _select_instance(boxes, selection: str) -> int:
    """Return the index of the instance to measure among all detections."""
    confs = boxes.conf.cpu().numpy()
    if selection == "largest_box":
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return int(np.argmax(areas))
    # default: highest detection confidence
    return int(np.argmax(confs))


def extract_pose(results) -> PoseResult | None:
    """Extract the chosen instance from a single-image YOLO result list.

    Returns None if no instance / no keypoints were found.
    """
    if not results:
        return None
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None or result.keypoints.xy is None:
        return None

    n_instances = len(result.boxes)
    inst = _select_instance(result.boxes, config.INSTANCE_SELECTION)

    xy = result.keypoints.xy.cpu().numpy()          # (n_inst, n_kp, 2)
    # Keypoint confidence is available when the model was trained with
    # visibility flags (the standard YOLOv8/YOLO11-pose case).
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf.cpu().numpy()  # (n_inst, n_kp)
    else:
        # Fallback: no per-keypoint score -> assume fully visible.
        conf = np.ones(xy.shape[:2], dtype=float)

    kp_xy = xy[inst]                                 # (n_kp, 2)
    kp_conf = conf[inst]                             # (n_kp,)

    keypoints = np.concatenate([kp_xy, kp_conf[:, None]], axis=1)  # (n_kp, 3)

    # Safety check: the number of keypoints must match the skeleton definition.
    if keypoints.shape[0] != NUM_KEYPOINTS:
        raise ValueError(
            f"Model returned {keypoints.shape[0]} keypoints but the skeleton "
            f"defines {NUM_KEYPOINTS}. Check that KEYPOINT_NAMES matches the "
            f"model's training order."
        )

    det_conf = float(result.boxes.conf.cpu().numpy()[inst])
    box = result.boxes.xyxy.cpu().numpy()[inst]

    return PoseResult(
        keypoints=keypoints,
        detection_conf=det_conf,
        box_xyxy=box,
        n_instances=n_instances,
    )


def run_pose_on_path(model: YOLO, img_path: str) -> PoseResult | None:
    """Run the pose model on an image path and return the measured instance."""
    results = model.predict(source=img_path, conf=config.POSE_CONF_THRESHOLD, verbose=False)
    return extract_pose(results)


def run_pose_on_array(model: YOLO, img_rgb_or_bgr: np.ndarray) -> PoseResult | None:
    """Run the pose model on an in-memory image (used by TTA)."""
    results = model.predict(source=img_rgb_or_bgr, conf=config.POSE_CONF_THRESHOLD, verbose=False)
    return extract_pose(results)
