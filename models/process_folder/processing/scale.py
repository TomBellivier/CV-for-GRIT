"""
scale.py
========

Decide the image scale (pixels per millimetre) and its confidence.

Strategy (mirrors your pipeline description):
    1. Try the scale-bar detector. If it yields a px/mm value, use it.
    2. Otherwise fall back to the ruler detector.
    3. If neither works, the scale is unknown (confidence 0, no mm conversion).

The scale confidence is computed by the dedicated functions in confidence.py:
    - scale bar -> product of the two box scores and the OCR reliability,
    - ruler     -> magnitude separation of the Fourier groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from . import config
from . import confidence
from .ruler_detection import detect_ruler_from_rgb
from .scale_bar_detection_utils import detect_scale_bar


@dataclass
class ScaleResult:
    """Resolved scale for one image."""
    px_per_mm: float | None      # None -> scale unknown
    method: str                  # "scale_bar" | "ruler" | "none"
    scale_conf: float            # confidence of the scale (0 when unknown)
    info: str                    # human-readable log line


def load_scale_bar_model(model_path=None) -> YOLO:
    """Load the YOLO scale-bar detector (loaded once, reused for all images)."""
    path = str(model_path or config.SCALE_BAR_MODEL_PATH)
    print(f"[scale] loading scale-bar model: {path}")
    model = YOLO(path)
    # Print the class map so the SCALE_BAR_*_CLASS_ID values can be verified.
    print(f"[scale] scale-bar model classes: {getattr(model, 'names', 'unknown')}")
    return model


def detect_scale(img_bgr: np.ndarray, scale_bar_model: YOLO | None) -> ScaleResult:
    """Resolve the scale for a single in-memory image (BGR array).

    Taking the decoded array (instead of a path) lets the exact same code run
    on local files and on Hugging Face images that are never written to disk.
    """

    # ---- 1. scale bar --------------------------------------------------------
    if config.USE_SCALE_BAR and scale_bar_model is not None and img_bgr is not None:
        det = detect_scale_bar(
            scale_bar_model=scale_bar_model,
            img_bgr=img_bgr,
            conf=config.SCALE_BAR_CONF_THRESHOLD,
            padding=config.SCALE_BAR_PADDING,
            bar_class_id=config.SCALE_BAR_BAR_CLASS_ID,
            text_class_id=config.SCALE_BAR_TEXT_CLASS_ID,
            missing_box_conf=config.SCALE_BAR_MISSING_BOX_CONF,
        )
        if det.px_per_mm is not None:
            conf = confidence.scale_bar_confidence(
                bar_box_conf=det.bar_box_conf if det.bar_box_conf is not None else 0.0,
                text_box_conf=det.text_box_conf if det.text_box_conf is not None else 0.0,
                ocr_reliability=det.ocr_reliability if det.ocr_reliability is not None else 0.0,
            )
            return ScaleResult(det.px_per_mm, "scale_bar", conf, det.info)

    # ---- 2. ruler fallback ---------------------------------------------------
    if config.USE_RULER_FALLBACK and img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        px_per_mm, line, conf = detect_ruler_from_rgb(
            img_rgb, ratio=config.RULER_RATIO
        )
        if px_per_mm is not None:
            info = (f"Ruler: {px_per_mm:.2f} px/mm at line {line}  "
                    f"[conf={conf:.3f}]")
            return ScaleResult(px_per_mm, "ruler", conf, info)

    # ---- 3. nothing worked ---------------------------------------------------
    return ScaleResult(None, "none", 0.0, "No scale found (scale bar and ruler failed).")
