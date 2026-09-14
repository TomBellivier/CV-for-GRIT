"""
config.py
=========

Single place for every tunable parameter of the pipeline. Nothing here runs
any heavy code; it only declares values that the other modules read. Edit this
file first when you adapt the pipeline to a new machine / dataset / model.

All paths are resolved relative to the PROJECT ROOT (the folder that contains
`process_folder.py`), so the pipeline works regardless of where you launch it
from.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# PROJECT_ROOT = folder that contains process_folder.py (one level above /processing)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root of the YOLO datasets. Expected layout (standard YOLO):
#     datasets/<dataset_name>/images/<split>/<image files>
#     datasets/<dataset_name>/labels/<split>/<label files>
DATASETS_ROOT = PROJECT_ROOT.parent / "datasets"

# Folder that holds the trained pose models (files named "XXX.pt").
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

# Exactly ONE pose model is used per run. Put its file name here.
# (Only this model is loaded; see load_pose_model in pose_inference.py.)
POSE_MODEL_NAME = "yolo_pooled_tuned.pt"                      # <-- EDIT ME
POSE_MODEL_PATH = TRAINED_MODELS_DIR / POSE_MODEL_NAME

# YOLO scale-bar detector.
SCALE_BAR_MODEL_PATH = PROJECT_ROOT.parent / "scale_bar_detection" / "best.pt"

# Folder of images to process, and where to write the CSV.
INPUT_FOLDER = PROJECT_ROOT.parent.parent / "andré_bees"   # <-- EDIT ME if needed
OUTPUT_CSV = INPUT_FOLDER / "results.csv"

# Crash safety: force the CSV to disk every N rows (file.flush + os.fsync), so a
# crash loses at most the last N rows. 1 = safest (durable write per image);
# larger = a little faster; 0 = only flush at the very end.
CSV_FLUSH_EVERY_N_ROWS = 20

# Export the raw keypoints of the measured instance as extra CSV columns:
#   '<kp> [kp_x]', '<kp> [kp_y]', '<kp> [kp_conf]'   (pixel coords + confidence)
# This adds 3 * NUM_KEYPOINTS columns and is what unlocks the keypoint-level
# error analysis (OKS, per-keypoint error vs confidence) in analyze_results.py.
EXPORT_KEYPOINTS = True

# Accepted image extensions (lower-case, with the dot).
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# --------------------------------------------------------------------------- #
# Dataset membership (train / val columns)
# --------------------------------------------------------------------------- #
IMAGES_SUBDIR = "images"        # "<dataset>/images/<split>"
TRAIN_SPLIT_DIRNAME = "train"
VAL_SPLIT_DIRNAME = "val"

# How a processed image is matched against the dataset files:
#   "name" -> exact file name incl. extension (e.g. "bee_001.jpg")   [requested]
#   "stem" -> file name without extension     (e.g. "bee_001")
MATCH_ON = "name"

# --------------------------------------------------------------------------- #
# Pose inference
# --------------------------------------------------------------------------- #
POSE_CONF_THRESHOLD = 0.25      # YOLO object-detection confidence threshold

# When several insects are detected on one image, which one do we measure?
#   "highest_conf" -> the instance with the highest detection score
#   "largest_box"  -> the instance with the largest bounding box
INSTANCE_SELECTION = "highest_conf"

# A keypoint whose confidence is below this threshold is considered "not
# reliably seen". This is used (a) to flag measurements and (b) optionally to
# drop them (see below). Give per-keypoint overrides for points that are known
# to be systematically hard (antenna tips, wing apex, tarsi, ...).
KEYPOINT_VISIBILITY_THRESHOLD = 0.50
PER_KEYPOINT_VISIBILITY_THRESHOLD = {
    # Example (uncomment / edit by hand):
    # "left-antenna-2":  0.30,
    # "right-antenna-2": 0.30,
    # "left-forewing-tip": 0.35,
}

# If True, a measurement is set to NaN (pixels AND mm) as soon as one of its
# keypoints is below its visibility threshold. If False, the measurement is
# still computed and it is only the *confidence* value that reflects the risk.
DROP_MEASUREMENT_IF_KP_BELOW_THRESHOLD = False

# --------------------------------------------------------------------------- #
# Confidence -- POSE (measurements)
# --------------------------------------------------------------------------- #
# Which signal fills the per-measurement confidence columns.
#   "keypoint" -> aggregated keypoint confidence (cheap, always available)
#   "tta"      -> test-time-augmentation dispersion (slower, needs ENABLE_TTA)
# Both signals are implemented in confidence.py; switch here once you have
# decided (on your val split) which one predicts the real error best.
MEASUREMENT_CONFIDENCE_SIGNAL = "keypoint"

# Aggregation used by the keypoint-based signal over the keypoints of one
# measurement: "min" | "geometric_mean" | "mean".
# "min" is the safest default: a distance is ruined as soon as ONE endpoint is
# wrong, so the weakest keypoint should drive the confidence.
KEYPOINT_AGGREGATION = "min"

# --------------------------------------------------------------------------- #
# Confidence -- POSE (test-time augmentation, TTA)
# --------------------------------------------------------------------------- #
# TTA re-runs inference on slightly perturbed copies of the image and measures
# how stable each measurement is. Low dispersion -> high confidence.
ENABLE_TTA = False                       # master switch (True is slower)

TTA_INCLUDE_IDENTITY = True              # include the un-augmented pass
TTA_USE_HFLIP = True                     # horizontal flip (swaps L/R keypoints)
TTA_ROTATION_DEGREES = [-4.0, 4.0]       # one extra pass per angle
TTA_BRIGHTNESS_FACTORS = [0.85, 1.15]    # one extra pass per factor (photometric)

# Confidence transform from the coefficient of variation (cv = std / mean):
#   confidence = exp(-TTA_CV_BETA * cv), clamped to [0, 1].
# Larger beta -> stricter (a small dispersion already lowers the confidence).
TTA_CV_BETA = 8.0

# --------------------------------------------------------------------------- #
# Confidence -- overall pose
# --------------------------------------------------------------------------- #
# Overall pose confidence = detection_conf * mean(keypoint_conf).
# The detection score says "this really is an insect"; the mean keypoint score
# says "and it is well articulated". Their product is an honest global value.
# Set to "min" to instead take min(detection_conf, mean_keypoint_conf).
OVERALL_POSE_METHOD = "det_x_kp"         # "det_x_kp" | "min"

# --------------------------------------------------------------------------- #
# Scale bar
# --------------------------------------------------------------------------- #
SCALE_BAR_CONF_THRESHOLD = 0.1
SCALE_BAR_PADDING = 20

# The scale-bar model is expected to detect TWO boxes: the bar and the text.
# Set the class ids below to match YOUR model. When the model is loaded, its
# class map (model.names) is printed so you can verify these values.
# If your model has a single class, set SCALE_BAR_TEXT_CLASS_ID = None; the text
# confidence then falls back to SCALE_BAR_MISSING_BOX_CONF and OCR is read from
# the (padded) bar crop.
SCALE_BAR_BAR_CLASS_ID = 0               # <-- VERIFY against model.names
SCALE_BAR_TEXT_CLASS_ID = 1              # <-- VERIFY against model.names (or None)
SCALE_BAR_MISSING_BOX_CONF = 1.0         # neutral value when a box is absent

# --------------------------------------------------------------------------- #
# Ruler (Fourier analysis)
# --------------------------------------------------------------------------- #
RULER_RATIO = 5                          # image sub-sampling factor
RULER_GRADUATION_MM = 1.0                # physical spacing of the ruler ticks
RULER_CONF_THRESHOLD = 0.03

HORIZONTAL_RULER_ONLY = True

# results_7 : threshold = 0.1
# results_8 : threshold = 0.05
# results_9 : threshold = 0.03
# results_10 : threshold = 0.02

# --------------------------------------------------------------------------- #
# Scale strategy
# --------------------------------------------------------------------------- #
USE_SCALE_BAR = True                     # try the scale bar first
USE_RULER_FALLBACK = True                # if the bar fails, try the ruler

# The ruler detector now runs twice per call (horizontal + vertical), so it is
# worth skipping entirely once the scale bar has already succeeded -- UNLESS
# we are evaluating against the manual scale annotations (set from
# --only_scale_annotated in process_folder.py), which needs the raw ruler
# confidence on every image, win or lose, for analyze_results.py's
# fig_scale_method_confidence / scale_*_confusion_matrix figures.
ONLY_SCALE_ANNOTATED = False

# --------------------------------------------------------------------------- #
# Converted-measurement confidence (millimetres)
# --------------------------------------------------------------------------- #
# A millimetre value depends on TWO reliable things: the pixel measurement and
# the scale. Combine their confidences with:
#   "min"     -> min(measurement_conf, scale_conf)      [conservative, simple]
#   "product" -> measurement_conf * scale_conf
CONVERTED_CONF_METHOD = "min"

# A detected scale that implies an unrealistic photographed extent is a
# scale-detection glitch rather than a real value: the mm conversion is
# skipped (mm -> NaN) for that image. scale_px_per_mm / scale_confidence
# themselves are left untouched (still diagnostic).
#
# Both bounds are expressed as a FRACTION of the image's largest dimension
# (image_size = max(width, height) in px), not a fixed absolute px/mm, so
# they scale with whatever resolution the photos actually are:
#   scale_px_per_mm > MAX_SCALE_PX_PER_MM_FRACTION * image_size
#       -> too zoomed IN (e.g. at 50%, one measured mm would already be half
#          the image -- implausible for a whole-insect photo)
#   scale_px_per_mm < MIN_SCALE_PX_PER_MM_FRACTION * image_size
#       -> too zoomed OUT (e.g. at 0.1%, one measured mm barely covers a
#          thousandth of the image -- implausibly far away / low-res)
MIN_SCALE_PX_PER_MM_FRACTION = 0.005     # 0.5% of image size
MAX_SCALE_PX_PER_MM_FRACTION = 0.5       # 50% of image size

# --------------------------------------------------------------------------- #
# Optional CSV columns
# --------------------------------------------------------------------------- #
# These are USEFUL additions that were proposed but are intentionally OFF by
# default (only the columns you asked for are written). Flip any of them to True
# to add the column; the plumbing already exists in csv_writer.py.
OPTIONAL_COLUMNS = {
    "scale_method":         True,   # "scale_bar" | "ruler" | "none"
    "detection_confidence": True,   # raw YOLO box score of the measured insect
    "image_width":          True,
    "image_height":         True,
    "needs_review":         True,   # derived boolean from confidence thresholds
    "scale_bar_confidence" :  True,
    "ruler_confidence" : True,
    "scale_bar_box":        True,   # scale-bar detection box, 'x1,y1,x2,y2'
    "scale_text_box":       True,   # scale-bar text box, 'x1,y1,x2,y2' (blank if no text class)
    "scale_ocr_text":       True,   # scale-bar raw OCR read
    "ruler_line":           True,   # ruler: row/col index the reading was taken at
    "ruler_orientation":    True,   # ruler: "horizontal" | "vertical"
}

# Threshold used only if OPTIONAL_COLUMNS["needs_review"] is True:
# a row is flagged when the overall pose confidence OR the scale confidence
# falls below this value.
NEEDS_REVIEW_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Measurement-validity classifier (pre-trained, inference only)
# --------------------------------------------------------------------------- #
# Per-measurement random forests (trained offline, see ../conf_classifier/
# measure_validity_classifiers.ipynb) that predict whether a measurement is
# trustworthy from the keypoint confidences of its anatomical neighbourhood
# plus the insect's taxonomic group. Nothing is trained here -- the saved
# models are only loaded and queried.
RUN_MEASUREMENT_CLASSIFIER = True

# Where each image's taxonomic group is looked up: DATABASE_DIR/<group>/...
# Used both as a classifier input and for the '<group>_one_hot' CSV columns.
DATABASE_DIR = PROJECT_ROOT.parent.parent / "databases" / "full databases"

MEASUREMENT_CLASSIFIER_DIR = TRAINED_MODELS_DIR / "measurement_classification_models"
MEASUREMENT_CLASSIFIER_METRICS_CSV = PROJECT_ROOT.parent / "conf_classifier" / "outputs" / "metrics.csv"

# --------------------------------------------------------------------------- #
# Per-stage processing time
# --------------------------------------------------------------------------- #
# Export the wall-clock time of every pipeline stage for one image, in seconds:
#   '<stage> time [s]'
EXPORT_TIMINGS = True
