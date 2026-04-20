import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import re

SCALE_BAR_MODEL_PATH = "models/scale_bar_detection/best.pt"

UNIT_TO_MM: dict[str, float] = {
    "nm":      1e-6,
    "um":      0.001, "µm": 0.001, "μm": 0.001,
    "micron":  0.001, "microns": 0.001,
    "mm":      1.0,
    "cm":      10.0,
    "m":       1000.0,
}
OCR_LANGUAGES          = ["en"]  

def _ensure_ocr_reader(reader) -> object:
    """Lazy-load EasyOCR reader (downloads weights ~100 MB on first call)."""
    if reader is None:
        reader = easyocr.Reader(OCR_LANGUAGES, gpu=False, verbose=False)
    return reader

def parse_scale_text(text: str) -> tuple[float, str] | None:
    """
    Extract a (value, unit) pair from raw OCR text.
    Handles formats like '500 µm', '1mm', '0.5 cm'.

    Fallback rules
    --------------
    - Number found but no unit  → assume µm (most common microscopy unit).
    - No number found at all    → return None (nothing can be done).
    """
    # normalise Unicode mu variants
    text = text.replace("μ", "µ")
    # try full match: number + recognised unit
    pattern_full = r"(\d+(?:[.,]\d+)?)\s*(nm|µm|μm|um|microns?|mm|cm|m)\b"
    m = re.search(pattern_full, text, re.IGNORECASE)
    if m:
        value_str = m.group(1).replace(",", ".")
        unit      = m.group(2).lower().rstrip("s")   # 'microns' → 'micron'
        return float(value_str), unit
    # fallback: number only → assume µm
    pattern_num = r"(\d+(?:[.,]\d+)?)"
    m = re.search(pattern_num, text)
    if m:
        value_str = m.group(1).replace(",", ".")
        return float(value_str), "µm"   # default unit
    # no number at all
    return None

def _preprocess_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Upscale very small crops and apply adaptive thresholding to improve OCR.
    Returns a 3-channel image (RGB) suitable for EasyOCR.
    """
    h, w = crop_rgb.shape[:2]
    if h < 64:
        scale = 64 / h
        crop_rgb = cv2.resize(crop_rgb, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def detect_scale_bar(
    img_bgr: np.ndarray,
    img_path: str,
    conf: float = 0.25,
    padding: int = 20,
) -> tuple[float | None, str]:
    """
    Run scale bar YOLO detection + EasyOCR on a single image.

    Returns
    -------
    px_per_mm : float | None
        Pixels per millimetre derived from the detected scale bar, or None
        if detection/OCR failed.
    info      : str
        Human-readable description of what was found (for the log).
    annotated : np.ndarray
        Copy of img_bgr with the scale bar bounding box drawn on it (BGR).
    """

    scale_bar_model = YOLO(SCALE_BAR_MODEL_PATH)

    if scale_bar_model is None:
        return None, "Scale bar model not loaded."

    results    = scale_bar_model.predict(source=img_path, conf=conf, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_score = float(box.conf[0].cpu().numpy())
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "confidence": conf_score})

    if not detections:
        return None, "No scale bar detected."

    # Pick the most-confident detection
    det = max(detections, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

    # Pixel width of the scale bar line/box
    bar_px = float(x2 - x1)
    if bar_px <= 0:
        return None, "Scale bar box has zero width."

    # Crop the detected region (with padding) and run OCR
    h_img, w_img = img_bgr.shape[:2]
    cx1 = max(0, x1 - padding)
    cy1 = max(0, y1 - padding)
    cx2 = min(w_img, x2 + padding)
    cy2 = min(h_img, y2 + padding)
    crop_bgr  = img_bgr[cy1:cy2, cx1:cx2]
    crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    processed = _preprocess_for_ocr(crop_rgb)

    try:
        reader  = _ensure_ocr_reader()
        raw_ocr = reader.readtext(processed, detail=1, paragraph=False)
    except Exception as exc:
        return None, f"OCR failed: {exc}"

    full_text = " ".join(r[1].strip() for r in raw_ocr)
    parsed    = parse_scale_text(full_text)

    if parsed is None:
        return (None,
                f"Scale bar detected (conf={det['confidence']:.3f}) "
                f"but OCR text not parsed: '{full_text}'")

    scale_value, unit = parsed
    mm_value          = scale_value * UNIT_TO_MM.get(unit, 1.0)

    if mm_value <= 0:
        return None, f"Parsed scale value is zero or negative: {scale_value} {unit}"

    px_per_mm = bar_px / mm_value
    info = (f"Scale bar: {scale_value} {unit}  "
            f"({bar_px:.0f} px → {mm_value:.4f} mm)  "
            f"→ {px_per_mm:.2f} px/mm")
    return px_per_mm, info