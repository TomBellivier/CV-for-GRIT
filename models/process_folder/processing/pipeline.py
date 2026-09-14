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
import time
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
                  models: Models, membership,
                  measurement_classifiers=None, group_index=None) -> dict:
    """Run the whole pipeline on one decoded image and return a CSV record."""
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

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
        "group_one_hot": {},
        "measure_valid": {},
    }

    # ---- 0. insect group (independent of pose/scale) ------------------------
    if group_index is not None:
        record["group_one_hot"] = group_index.one_hot(image_name)

    # ---- 1. scale (independent of the pose) ---------------------------------
    t0 = time.perf_counter()
    scale = detect_scale(img_bgr, models.scale_bar_model)
    timings["scale"] = time.perf_counter() - t0
    record["scale_px_per_mm"] = scale.px_per_mm
    record["scale_confidence"] = scale.scale_conf

    # ---- 2. pose inference ---------------------------------------------------
    t0 = time.perf_counter()
    pose = run_pose_on_array(models.pose_model, img_bgr)
    timings["pose"] = time.perf_counter() - t0
    if pose is None:
        timings["total"] = time.perf_counter() - t_start
        record["timings"] = timings
        _fill_optional_columns(record, img_bgr, pose, scale)
        return record

    keypoints = pose.keypoints
    record["keypoints"] = keypoints           # exported as raw kp columns (x,y,conf)

    # ---- 3. pixel measurements ----------------------------------------------
    t0 = time.perf_counter()
    pixels = compute_measurements(keypoints)
    timings["measurements"] = time.perf_counter() - t0
    record["pixels"] = pixels

    # ---- 4. measurement confidences (selected signal) -----------------------
    t0 = time.perf_counter()
    if config.MEASUREMENT_CONFIDENCE_SIGNAL == "tta":
        record["conf"] = _measurement_confidences_tta(models.pose_model, img_bgr)
    else:  # "keypoint" (default)
        record["conf"] = _measurement_confidences_keypoint(keypoints)

    # ---- 5. overall pose confidence -----------------------------------------
    record["overall_pose_confidence"] = confidence.overall_pose_confidence(
        detection_conf=pose.detection_conf,
        keypoint_confidences=keypoints[:, 2],
    )
    timings["confidence"] = time.perf_counter() - t0

    # ---- 6. convert to millimetres ------------------------------------------
    t0 = time.perf_counter()
    px_per_mm = scale.px_per_mm
    # A scale implying an unrealistic photographed extent (too zoomed in OR
    # out, relative to the image's own resolution -- see config for the
    # rationale) is a scale-detection glitch, not a real close-up/wide shot.
    # scale_px_per_mm/scale_confidence are left untouched (still diagnostic);
    # only the mm conversion is skipped.
    if px_per_mm is not None and img_bgr is not None:
        image_size = max(img_bgr.shape[0], img_bgr.shape[1])
        lo = config.MIN_SCALE_PX_PER_MM_FRACTION * image_size
        hi = config.MAX_SCALE_PX_PER_MM_FRACTION * image_size
        if not (lo <= px_per_mm <= hi):
            px_per_mm = None
    for name in MEASUREMENT_NAMES:
        px_val = pixels.get(name, math.nan)
        if px_per_mm and px_per_mm > 0 and not math.isnan(px_val):
            record["mm"][name] = px_val / px_per_mm
        else:
            record["mm"][name] = math.nan
    timings["mm_conversion"] = time.perf_counter() - t0

    # To report the COMBINED (measurement + scale) confidence for the mm values
    # instead of the raw measurement confidence, uncomment:
    # for name in MEASUREMENT_NAMES:
    #     record["conf"][name] = confidence.converted_measurement_confidence(
    #         record["conf"][name], scale.scale_conf)

    # ---- 7. measurement-validity classifier (pre-trained, inference only) ---
    t0 = time.perf_counter()
    if config.RUN_MEASUREMENT_CLASSIFIER and measurement_classifiers is not None:
        group = group_index.group_of(image_name) if group_index is not None else None
        record["measure_valid"] = measurement_classifiers.score(keypoints, group)
    timings["measurement_classifier"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start
    record["timings"] = timings

    _fill_optional_columns(record, img_bgr, pose, scale)
    return record


def _format_box(box) -> str | None:
    """(x1, y1, x2, y2) -> 'x1,y1,x2,y2', or None if no box."""
    return ",".join(str(int(v)) for v in box) if box is not None else None


def _fill_optional_columns(record, img_bgr, pose, scale):
    """Populate the optional columns that are enabled in the config."""
    opt = config.OPTIONAL_COLUMNS
    if opt.get("scale_method"):
        record["scale_method"] = scale.method
    if opt.get("n_instances"):
        record["n_instances"] = pose.n_instances if pose is not None else 0
    if opt.get("detection_confidence"):
        record["detection_confidence"] = pose.detection_conf if pose is not None else math.nan
    if opt.get("scale_bar_confidence"):
        record["scale_bar_confidence"] = scale.scale_bar_conf if scale is not None else math.nan
    if opt.get("ruler_confidence"):
        record["ruler_confidence"] = scale.ruler_conf if scale is not None else math.nan
    if opt.get("scale_bar_box"):
        record["scale_bar_box"] = _format_box(scale.bar_box) if scale is not None else None
    if opt.get("scale_text_box"):
        record["scale_text_box"] = _format_box(scale.text_box) if scale is not None else None
    if opt.get("scale_ocr_text"):
        record["scale_ocr_text"] = scale.ocr_text if scale is not None else None
    if opt.get("ruler_line"):
        record["ruler_line"] = scale.ruler_line if scale is not None else math.nan
    if opt.get("ruler_orientation"):
        record["ruler_orientation"] = scale.ruler_orientation if scale is not None else None
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
