"""
pose_inference.py
=================

Ensemble inference with the retained YOLO-pose models:

    * load EVERY pose model found under retained_models/pose/ (one after a
      `train`, one per outer fold after a `tune`, see retained_models/README.md),
    * run each of them on the image,
    * pick the instance to measure, and the matching instance in every model,
    * return the per-keypoint MEAN over the models, plus its standard deviation.

Everything downstream (measurements, confidences, measurement-validity
classifiers) works on the mean keypoints. The standard deviation is the spread
of the ensemble: 0 when a single model is retained.

No training happens here -- the models are only loaded and queried.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from . import config
from .definitions import NUM_KEYPOINTS
from .hardware import precision_kwargs


@dataclass
class PoseResult:
    """Everything the rest of the pipeline needs about one detected insect."""
    keypoints: np.ndarray        # (NUM_KEYPOINTS, 3) -> x, y, confidence, MEAN over the models
    keypoints_std: np.ndarray    # (NUM_KEYPOINTS, 3) -> std of the same values (0 for 1 model)
    detection_conf: float        # YOLO box score of this instance, mean over the models
    box_xyxy: np.ndarray         # (4,) bounding box, mean over the models
    n_instances: int             # how many instances the reference model detected
    n_models: int                # how many models contributed to the mean


@dataclass
class _Instances:
    """All the instances one model detected on one image."""
    xy: np.ndarray               # (n_inst, NUM_KEYPOINTS, 2)
    conf: np.ndarray             # (n_inst, NUM_KEYPOINTS)
    det_conf: np.ndarray         # (n_inst,)
    boxes: np.ndarray            # (n_inst, 4) xyxy


def pose_model_paths(models_dir=None) -> list[Path]:
    """Every model of the ensemble: all the *.pt files under the pose folder.

    `models_dir` may also be a single .pt file (an ensemble of one).
    """
    root = Path(models_dir or config.POSE_MODELS_DIR)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.pt")) if root.is_dir() else []


def load_pose_models(models_dir=None) -> list[YOLO]:
    """Load every model of the ensemble.

    The keypoint names embedded in each model are printed so you can check
    that their order matches definitions.KEYPOINT_NAMES.
    """
    paths = pose_model_paths(models_dir)
    if not paths:
        raise FileNotFoundError(f"No pose model (*.pt) under {models_dir or config.POSE_MODELS_DIR}")
    models = []
    for path in paths:
        print(f"[pose] loading pose model: {path}")
        model = YOLO(str(path))
        print(f"[pose] model classes: {getattr(model, 'names', 'unknown')}")
        models.append(model)
    return models


def _instances(results) -> _Instances | None:
    """Every instance of a single-image YOLO result, or None if there is none."""
    if not results:
        return None
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None or result.keypoints.xy is None:
        return None

    xy = result.keypoints.xy.cpu().numpy()          # (n_inst, n_kp, 2)
    # Keypoint confidence is available when the model was trained with
    # visibility flags (the standard YOLOv8/YOLO11-pose case).
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf.cpu().numpy()  # (n_inst, n_kp)
    else:
        # Fallback: no per-keypoint score -> assume fully visible.
        conf = np.ones(xy.shape[:2], dtype=float)

    # Safety check: the number of keypoints must match the skeleton definition.
    if xy.shape[1] != NUM_KEYPOINTS:
        raise ValueError(
            f"Model returned {xy.shape[1]} keypoints but the skeleton "
            f"defines {NUM_KEYPOINTS}. Check that KEYPOINT_NAMES (kp_infos.yaml) "
            f"matches the model's training order."
        )
    return _Instances(xy=xy, conf=conf,
                      det_conf=result.boxes.conf.cpu().numpy(),
                      boxes=result.boxes.xyxy.cpu().numpy())


def _select_instance(inst: _Instances, selection: str) -> int:
    """Return the index of the instance to measure among all detections."""
    if selection == "largest_box":
        b = inst.boxes
        return int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    # default: highest detection confidence
    return int(np.argmax(inst.det_conf))


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an (n, 4) array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    union = area(box) + area(boxes) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def ensemble_pose(per_model: list[_Instances | None]) -> PoseResult | None:
    """Average the same insect over the models of the ensemble.

    The reference is the instance picked by INSTANCE_SELECTION in the model that
    is the most confident about it. Every other model contributes the instance
    that overlaps it best, and only if that overlap reaches ENSEMBLE_MIN_IOU:
    a model that found another insect (or none) is left out of the mean rather
    than blending two specimens.
    """
    detected = [inst for inst in per_model if inst is not None]
    if not detected:
        return None

    picks = [(inst, _select_instance(inst, config.INSTANCE_SELECTION)) for inst in detected]
    ref_inst, ref_idx = max(picks, key=lambda p: p[0].det_conf[p[1]])
    ref_box = ref_inst.boxes[ref_idx]

    xy, conf, det, boxes = [], [], [], []
    for inst in detected:
        overlaps = _iou(ref_box, inst.boxes)
        j = int(np.argmax(overlaps))
        if inst is not ref_inst and overlaps[j] < config.ENSEMBLE_MIN_IOU:
            continue
        if inst is ref_inst:
            j = ref_idx
        xy.append(inst.xy[j])
        conf.append(inst.conf[j])
        det.append(inst.det_conf[j])
        boxes.append(inst.boxes[j])

    stacked = np.concatenate([np.stack(xy), np.stack(conf)[..., None]], axis=2)  # (n_models, K, 3)
    return PoseResult(
        keypoints=stacked.mean(axis=0),
        # Population std: exactly 0 with a single model, as documented in the CSV.
        keypoints_std=stacked.std(axis=0),
        detection_conf=float(np.mean(det)),
        box_xyxy=np.mean(boxes, axis=0),
        n_instances=len(ref_inst.det_conf),
        n_models=len(xy),
    )


def run_pose_ensemble(models: list[YOLO], img, device: str | None = None) -> PoseResult | None:
    """Run every model of the ensemble on an image (path or in-memory array).

    `device` is the worker's device (hardware.next_device); FP16 on a CUDA GPU.
    """
    kwargs = {"device": device, **precision_kwargs(device)} if device else {}
    return ensemble_pose([
        _instances(model.predict(source=img, conf=config.POSE_CONF_THRESHOLD,
                                 verbose=False, **kwargs))
        for model in models
    ])

