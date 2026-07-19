"""
scale_bar_detection_utils.py
============================

Adapted from your original file. What changed and why:

  1. detect_scale_bar now returns a small dataclass (ScaleBarDetection) instead
     of a bare (px_per_mm, info) tuple, so it can also carry the confidence
     ingredients requested for the scale-bar confidence:
        * bar_box_conf   -> detection score of the BAR box,
        * text_box_conf  -> detection score of the TEXT box,
        * ocr_reliability-> reliability of the OCR read.
     (The final confidence itself is computed in confidence.scale_bar_confidence
     so that all confidence logic stays in one module.)

  2. The detector now separates the BAR box from the TEXT box using the class
     ids given in the config. If your model has a single class, set
     SCALE_BAR_TEXT_CLASS_ID = None: the text confidence then falls back to a
     neutral value and the OCR is read from the padded bar crop (old behaviour).

  3. Bug fix: `_ensure_ocr_reader()` was called with no argument although it
     required one. It is replaced by a module-level cached reader.

  4. OCR reliability is derived from EasyOCR's own per-fragment confidence
     scores (the third element of each readtext result).

Only the model is loaded/queried here -- nothing is trained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

import easyocr


# --------------------------------------------------------------------------- #
# Unit handling
# --------------------------------------------------------------------------- #
UNIT_TO_MM: dict[str, float] = {
    "nm":     1e-6,
    "um":     0.001, "µm": 0.001, "μm": 0.001,
    "micron": 0.001, "microns": 0.001,
    "mm":     1.0,
    "cm":     10.0,
    "m":      1000.0,
}
OCR_LANGUAGES = ["en"]

# Cached EasyOCR reader (weights ~100 MB are downloaded on the first call).
_OCR_READER = None


def _ensure_ocr_reader():
    """Lazy-load and cache the EasyOCR reader (bug-fixed: no argument needed)."""
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
    return _OCR_READER


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ScaleBarDetection:
    """Outcome of a scale-bar detection on one image."""
    px_per_mm: float | None         # None if detection / OCR failed
    info: str                       # human-readable log line
    bar_box_conf: float | None      # detection score of the bar box
    text_box_conf: float | None     # detection score of the text box
    ocr_reliability: float | None   # EasyOCR reliability of the parsed read


# --------------------------------------------------------------------------- #
# OCR text parsing
# --------------------------------------------------------------------------- #
def parse_scale_text(text: str) -> tuple[float, str] | None:
    """Extract a (value, unit) pair from raw OCR text.

    Handles formats like '500 µm', '1mm', '0.5 cm'.
    Fallbacks: number but no unit -> assume µm; no number -> None.
    """
    text = text.replace("μ", "µ")  # normalise Unicode mu variants
    pattern_full = r"(\d+(?:[.,]\d+)?)\s*(nm|µm|μm|um|microns?|mm|cm|m)\b"
    m = re.search(pattern_full, text, re.IGNORECASE)
    if m:
        value_str = m.group(1).replace(",", ".")
        unit = m.group(2).lower().rstrip("s")   # 'microns' -> 'micron'
        return float(value_str), unit

    pattern_num = r"(\d+(?:[.,]\d+)?)"
    m = re.search(pattern_num, text)
    if m:
        value_str = m.group(1).replace(",", ".")
        return float(value_str), "µm"           # default unit
    return None


def _preprocess_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    """Upscale tiny crops and threshold them to help OCR. Returns RGB."""
    h, w = crop_rgb.shape[:2]
    if h < 64:
        scale = 64 / h
        crop_rgb = cv2.resize(crop_rgb, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def _ocr_reliability(raw_ocr) -> tuple[str, float]:
    """Join OCR fragments and estimate the reliability of the read.

    EasyOCR (detail=1) returns tuples (bbox, text, confidence). We prefer the
    fragments that actually contain a digit (those are the ones carrying the
    scale value) and average their confidence. If none contain a digit we fall
    back to the mean confidence of all fragments.
    """
    full_text = " ".join(r[1].strip() for r in raw_ocr)

    digit_confs = [float(r[2]) for r in raw_ocr if any(ch.isdigit() for ch in r[1])]
    all_confs = [float(r[2]) for r in raw_ocr]

    if digit_confs:
        reliability = float(np.mean(digit_confs))
    elif all_confs:
        reliability = float(np.mean(all_confs))
    else:
        reliability = 0.0
    return full_text, reliability


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #
def _collect_detections(results) -> list[dict]:
    """Flatten YOLO results into a list of {x1,y1,x2,y2,conf,cls} dicts."""
    dets = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "confidence": conf_score, "cls": cls_id})
    return dets


def _best_of_class(dets: list[dict], cls_id) -> dict | None:
    """Most confident detection of a given class (or overall if cls_id is None)."""
    if cls_id is None:
        pool = dets
    else:
        pool = [d for d in dets if d["cls"] == cls_id]
    if not pool:
        return None
    return max(pool, key=lambda d: d["confidence"])


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def detect_scale_bar(
    scale_bar_model: YOLO,
    img_bgr: np.ndarray,
    img_path: str | None = None,
    conf: float = 0.25,
    padding: int = 20,
    bar_class_id: int | None = 0,
    text_class_id: int | None = 1,
    missing_box_conf: float = 1.0,
) -> ScaleBarDetection:
    """Run scale-bar YOLO detection + EasyOCR on a single image.

    Parameters
    ----------
    scale_bar_model : loaded YOLO model.
    img_bgr         : the image in BGR (used for detection, cropping and OCR).
    img_path        : kept for logging only; detection now runs on img_bgr so
                      the same code path works for in-memory (Hugging Face)
                      images that never touch the disk.
    conf            : YOLO detection confidence threshold.
    padding         : extra pixels around a crop before OCR.
    bar_class_id    : class id of the BAR box (px width is taken from it).
    text_class_id   : class id of the TEXT box (OCR crop + text confidence);
                      set to None if the model has no dedicated text class.
    missing_box_conf: neutral confidence used when a box class is absent.

    Returns
    -------
    ScaleBarDetection
    """
    if scale_bar_model is None:
        return ScaleBarDetection(None, "Scale bar model not loaded.", None, None, None)

    # Detection runs on the in-memory array (Ultralytics accepts a BGR ndarray),
    # so the scale bar works identically for local files and HF images.
    results = scale_bar_model.predict(source=img_bgr, conf=conf, verbose=False)
    dets = _collect_detections(results)
    if not dets:
        return ScaleBarDetection(None, "No scale bar detected.", None, None, None)

    # ---- pick the BAR box (drives the pixel length) --------------------------
    bar = _best_of_class(dets, bar_class_id)
    if bar is None:
        # The configured bar class was not found; fall back to the single most
        # confident detection so the pipeline still works on 1-class models.
        bar = max(dets, key=lambda d: d["confidence"])
    bar_box_conf = bar["confidence"]

    bar_px = float(bar["x2"] - bar["x1"])
    if bar_px <= 0:
        return ScaleBarDetection(None, "Scale bar box has zero width.",
                                 bar_box_conf, None, None)

    # ---- pick the TEXT box (drives the OCR crop) -----------------------------
    text = _best_of_class(dets, text_class_id) if text_class_id is not None else None
    if text is not None:
        text_box_conf = text["confidence"]
        crop_src = text
    else:
        # No dedicated text box: OCR the padded bar crop and use a neutral score.
        text_box_conf = missing_box_conf
        crop_src = bar

    # ---- crop and run OCR ----------------------------------------------------
    h_img, w_img = img_bgr.shape[:2]
    cx1 = max(0, crop_src["x1"] - padding)
    cy1 = max(0, crop_src["y1"] - padding)
    cx2 = min(w_img, crop_src["x2"] + padding)
    cy2 = min(h_img, crop_src["y2"] + padding)
    crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    processed = _preprocess_for_ocr(crop_rgb)

    try:
        reader = _ensure_ocr_reader()
        raw_ocr = reader.readtext(processed, detail=1, paragraph=False)
    except Exception as exc:  # noqa: BLE001 - report any OCR failure in the log
        return ScaleBarDetection(None, f"OCR failed: {exc}",
                                 bar_box_conf, text_box_conf, 0.0)

    full_text, ocr_reliability = _ocr_reliability(raw_ocr)
    parsed = parse_scale_text(full_text)

    if parsed is None:
        return ScaleBarDetection(
            None,
            f"Scale bar detected (bar_conf={bar_box_conf:.3f}) "
            f"but OCR text not parsed: '{full_text}'",
            bar_box_conf, text_box_conf, ocr_reliability,
        )

    scale_value, unit = parsed
    mm_value = scale_value * UNIT_TO_MM.get(unit, 1.0)
    if mm_value <= 0:
        return ScaleBarDetection(
            None, f"Parsed scale value is zero or negative: {scale_value} {unit}",
            bar_box_conf, text_box_conf, ocr_reliability,
        )

    px_per_mm = bar_px / mm_value
    info = (f"Scale bar: {scale_value} {unit}  "
            f"({bar_px:.0f} px -> {mm_value:.4f} mm)  "
            f"-> {px_per_mm:.2f} px/mm  "
            f"[bar_conf={bar_box_conf:.3f}, text_conf={text_box_conf:.3f}, "
            f"ocr={ocr_reliability:.3f}]")
    return ScaleBarDetection(px_per_mm, info, bar_box_conf, text_box_conf, ocr_reliability)
