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
    scale_bar_conf:float
    ruler_conf:float
    scale_conf: float            # confidence of the scale (0 when unknown)
    info: str                    # human-readable log line
    # Method-specific geometry, populated when that method actually ran
    # (regardless of which one "won"), for a visual/QC record of the read:
    bar_box: tuple[int, int, int, int] | None = None    # scale-bar box (x1,y1,x2,y2)
    text_box: tuple[int, int, int, int] | None = None   # scale-bar text box (x1,y1,x2,y2)
    ocr_text: str | None = None                          # scale-bar raw OCR read
    ruler_line: float | None = None                      # ruler: row/col index of the read
    ruler_orientation: str | None = None                 # ruler: "horizontal" | "vertical"


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

    selected_scale_method = "none"
    scale_bar_conf = 0.0
    px_per_mm, line, ruler_conf, info = None, None, 0.0, ""
    ruler_orientation = None
    det = None

    # ---- 1. scale bar --------------------------------------------------------
    if config.USE_SCALE_BAR and scale_bar_model is not None and img_bgr is not None:
        det = detect_scale_bar(
            scale_bar_model=scale_bar_model,
            img_bgr=img_bgr,
            #conf=config.SCALE_BAR_CONF_THRESHOLD,
            padding=config.SCALE_BAR_PADDING,
            bar_class_id=config.SCALE_BAR_BAR_CLASS_ID,
            text_class_id=config.SCALE_BAR_TEXT_CLASS_ID,
            missing_box_conf=config.SCALE_BAR_MISSING_BOX_CONF,
        )
        scale_bar_conf = confidence.scale_bar_confidence(
            bar_box_conf=det.bar_box_conf if det.bar_box_conf is not None else 0.0,
            text_box_conf=det.text_box_conf if det.text_box_conf is not None else 0.0,
            ocr_reliability=det.ocr_reliability if det.ocr_reliability is not None else 0.0,
        )
        if det.px_per_mm is not None and scale_bar_conf > config.SCALE_BAR_CONF_THRESHOLD:
            selected_scale_method = "scale_bar"

    # ---- 2. ruler fallback ---------------------------------------------------
    # The ruler detector runs the (costly) Fourier analysis twice, horizontal
    # and vertical. Skip it once the scale bar has already won -- unless
    # ONLY_SCALE_ANNOTATED is set, in which case both raw confidences are kept
    # on every image for the scale-method evaluation figures.
    run_ruler = (config.USE_RULER_FALLBACK and img_bgr is not None
                and (selected_scale_method != "scale_bar" or config.ONLY_SCALE_ANNOTATED))
    if run_ruler:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        px_per_mm_h, line_h, ruler_conf_h = detect_ruler_from_rgb(img_rgb, ratio=config.RULER_RATIO)

        if not config.HORIZONTAL_RULER_ONLY:
            rotated_img = np.rot90(img_rgb)
            px_per_mm_v, line_v, ruler_conf_v = detect_ruler_from_rgb(rotated_img, ratio=config.RULER_RATIO)
        else:
            px_per_mm_v, line_v, ruler_conf_v = None, None, 0.0

        if ruler_conf_v is None or (ruler_conf_h is not None and ruler_conf_h > ruler_conf_v):
            px_per_mm, line, ruler_conf = px_per_mm_h, line_h, ruler_conf_h
            ruler_orientation = "horizontal" if px_per_mm_h is not None else None
        elif ruler_conf_h is None or (ruler_conf_v is not None and ruler_conf_v > ruler_conf_h):
            px_per_mm, line, ruler_conf = px_per_mm_v, line_v, ruler_conf_v
            ruler_orientation = "vertical" if px_per_mm_v is not None else None
        else:
            px_per_mm, line, ruler_conf = None, None, 0.0

        if px_per_mm is not None and ruler_conf >= config.RULER_CONF_THRESHOLD:
            info = (f"Ruler: {px_per_mm:.2f} px/mm at line {line}  "
                    f"[conf={ruler_conf:.3f}]")
            if selected_scale_method == "none": # priority given to scale_bar detection
                selected_scale_method = "ruler"

    # Geometry of whichever method(s) actually ran, regardless of which won
    # (both can run under ONLY_SCALE_ANNOTATED) -- exported for a visual/QC
    # record of the read (see config.OPTIONAL_COLUMNS).
    bar_box = det.bar_box if det is not None else None
    text_box = det.text_box if det is not None else None
    ocr_text = det.ocr_text if det is not None else None
    ruler_line = line if run_ruler else None

    if selected_scale_method == "scale_bar":
        return ScaleResult(det.px_per_mm, "scale_bar", scale_bar_conf, ruler_conf, scale_bar_conf,
                           det.info, bar_box=bar_box, text_box=text_box, ocr_text=ocr_text,
                           ruler_line=ruler_line, ruler_orientation=ruler_orientation)
    elif selected_scale_method == "ruler":
        return ScaleResult(px_per_mm, "ruler", scale_bar_conf, ruler_conf, ruler_conf, info,
                           bar_box=bar_box, text_box=text_box, ocr_text=ocr_text,
                           ruler_line=ruler_line, ruler_orientation=ruler_orientation)

    # ---- 3. nothing worked ---------------------------------------------------
    return ScaleResult(None, "none", scale_bar_conf, ruler_conf, 0.0,
                       "No scale found (scale bar and ruler failed).",
                       bar_box=bar_box, text_box=text_box, ocr_text=ocr_text,
                       ruler_line=ruler_line, ruler_orientation=ruler_orientation)
