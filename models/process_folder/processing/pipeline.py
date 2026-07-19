"""
pipeline.py
===========

The per-image pipeline, factored out of process_folder.py so it can be reused
by every worker thread and by both input sources (local folder and Hugging
Face). It operates on an IN-MEMORY image (a BGR numpy array) and returns the
record dict consumed by csv_writer.

Why an array and not a path?
    Hugging Face images are downloaded straight into RAM and never written to
    disk. Working on the decoded array means the exact same code path serves
    local files (read with cv2.imread) and streamed HF images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from . import config
from . import confidence
from . import tta
from .definitions import MEASUREMENT_NAMES
from .measurements import compute_measurements, measurement_keypoint_confidences
from .pose_inference import run_pose_on_array
from .scale import detect_scale


@dataclass
class Models:
    """The models a worker needs. Built once per thread (see worker.py)."""
    pose_model: YOLO
    scale_bar_model: YOLO | None


def _measurement_confidences_keypoint(keypoints) -> dict[str, float]:
    """Keypoint-based confidence for every measurement (cheap signal)."""
    return {
        name: confidence.measurement_confidence_keypoint(
            measurement_keypoint_confidences(keypoints, name))
        for name in MEASUREMENT_NAMES
    }


def _measurement_confidences_tta(pose_model, img_bgr) -> dict[str, float]:
    """TTA-based confidence for every measurement (stability signal)."""
    per_measure = tta.collect_tta_measurements(pose_model, img_bgr)
    return {
        name: confidence.measurement_confidence_tta(per_measure.get(name, []))
        for name in MEASUREMENT_NAMES
    }


def process_image(img_bgr: np.ndarray, image_name: str,
                  models: Models, membership) -> dict:
    """Run the whole pipeline on one decoded image and return a CSV record."""
    record: dict = {
        "image_name": image_name,
        "in_train": membership.in_train(image_name),
        "in_val": membership.in_val(image_name),
        "pixels": {},
        "mm": {},
        "conf": {},
        "overall_pose_confidence": math.nan,
        "scale_px_per_mm": None,
        "scale_confidence": 0.0,
    }

    # ---- 1. scale (independent of the pose) ---------------------------------
    scale = detect_scale(img_bgr, models.scale_bar_model)
    record["scale_px_per_mm"] = scale.px_per_mm
    record["scale_confidence"] = scale.scale_conf

    # ---- 2. pose inference ---------------------------------------------------
    pose = run_pose_on_array(models.pose_model, img_bgr)
    if pose is None:
        _fill_optional_columns(record, img_bgr, pose, scale)
        return record

    keypoints = pose.keypoints

    # ---- 3. pixel measurements ----------------------------------------------
    pixels = compute_measurements(keypoints)
    record["pixels"] = pixels

    # ---- 4. measurement confidences (selected signal) -----------------------
    if config.MEASUREMENT_CONFIDENCE_SIGNAL == "tta":
        record["conf"] = _measurement_confidences_tta(models.pose_model, img_bgr)
    else:  # "keypoint" (default)
        record["conf"] = _measurement_confidences_keypoint(keypoints)

    # ---- 5. overall pose confidence -----------------------------------------
    record["overall_pose_confidence"] = confidence.overall_pose_confidence(
        detection_conf=pose.detection_conf,
        keypoint_confidences=keypoints[:, 2],
    )

    # ---- 6. convert to millimetres ------------------------------------------
    px_per_mm = scale.px_per_mm
    for name in MEASUREMENT_NAMES:
        px_val = pixels.get(name, math.nan)
        if px_per_mm and px_per_mm > 0 and not math.isnan(px_val):
            record["mm"][name] = px_val / px_per_mm
        else:
            record["mm"][name] = math.nan

    # To report the COMBINED (measurement + scale) confidence for the mm values
    # instead of the raw measurement confidence, uncomment:
    # for name in MEASUREMENT_NAMES:
    #     record["conf"][name] = confidence.converted_measurement_confidence(
    #         record["conf"][name], scale.scale_conf)

    _fill_optional_columns(record, img_bgr, pose, scale)
    return record


def _fill_optional_columns(record, img_bgr, pose, scale):
    """Populate the optional columns that are enabled in the config."""
    opt = config.OPTIONAL_COLUMNS
    if opt.get("scale_method"):
        record["scale_method"] = scale.method
    if opt.get("n_instances"):
        record["n_instances"] = pose.n_instances if pose is not None else 0
    if opt.get("detection_confidence"):
        record["detection_confidence"] = pose.detection_conf if pose is not None else math.nan
    if opt.get("image_width") or opt.get("image_height"):
        h, w = (img_bgr.shape[0], img_bgr.shape[1]) if img_bgr is not None else (None, None)
        if opt.get("image_width"):
            record["image_width"] = w
        if opt.get("image_height"):
            record["image_height"] = h
    if opt.get("needs_review"):
        pose_conf = record.get("overall_pose_confidence", math.nan)
        scale_conf = record.get("scale_confidence", 0.0)
        thr = config.NEEDS_REVIEW_THRESHOLD
        flagged = (
            (isinstance(pose_conf, float) and not math.isnan(pose_conf) and pose_conf < thr)
            or (scale_conf < thr)
        )
        record["needs_review"] = bool(flagged)
