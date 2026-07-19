"""
definitions.py
==============

Static description of the skeleton, transcribed from your two reference files
(`_keypoint_list.txt` and `_measurements_list.txt`):

    * KEYPOINT_NAMES   -> the canonical keypoint order. THIS ORDER MUST MATCH the
                          order the model was trained with, because YOLO returns
                          keypoints as an array indexed by this order.
    * KEYPOINT_COLORS  -> RGB colour of each keypoint (kept for future overlays).
    * MEASUREMENTS     -> for each measurement, the ordered list of keypoints
                          whose consecutive segments are summed to obtain the
                          measurement length (confirmed: sum of segments).
    * FLIP_INDEX       -> permutation that swaps left/right keypoints, used by
                          the horizontal-flip test-time augmentation.

Deriving indices and the flip map from these lists (instead of hard-coding
numbers) keeps everything consistent if the skeleton ever changes.
"""

# --------------------------------------------------------------------------- #
# Keypoints -- canonical order (index 0 .. N-1) and colours
# --------------------------------------------------------------------------- #
# The order below is exactly the order of `_keypoint_list.txt`.
KEYPOINT_COLORS = {
    "head-top":              [0,   0,   255],
    "head-left":             [75,  0,   225],
    "head-right":            [0,   75,  225],
    "neck":                  [0,   0,   200],
    "left-eye":              [220, 200, 200],
    "right-eye":             [200, 220, 200],
    "thorax-left":           [50,  150, 255],
    "thorax-right":          [0,   200, 255],
    "thorax-bottom":         [0,   255, 255],
    "body-left":             [100, 200, 100],
    "body-right":            [0,   255, 100],
    "body-tip":              [0,   255, 0],
    "left-antenna-0":        [255, 255, 0],
    "left-antenna-1":        [255, 255, 150],
    "left-antenna-2":        [255, 255, 200],
    "left-forewing-base":    [255, 0,   200],
    "left-forewing-tip":     [150, 0,   200],
    "left-forewing-front":   [200, 0,   255],
    "left-forewing-rear":    [175, 0,   150],
    "left-hindwing-base":    [150, 100, 100],
    "left-hindwing-tip":     [255, 100, 100],
    "left-hindwing-front":   [200, 100, 150],
    "left-hindwing-rear":    [200, 100, 50],
    "left-leg-0":            [255, 200, 0],
    "left-leg-1":            [255, 200, 125],
    "left-leg-2":            [255, 200, 175],
    "left-leg-3":            [255, 200, 255],
    "right-antenna-0":       [200, 255, 0],
    "right-antenna-1":       [200, 255, 150],
    "right-antenna-2":       [200, 255, 200],
    "right-forewing-base":   [150, 50,  200],
    "right-forewing-tip":    [150, 150, 200],
    "right-forewing-front":  [150, 100, 255],
    "right-forewing-rear":   [150, 100, 150],
    "right-hindwing-base":   [255, 150, 150],
    "right-hindwing-tip":    [150, 150, 150],
    "right-hindwing-front":  [200, 150, 200],
    "right-hindwing-rear":   [200, 150, 100],
    "right-leg-0":           [200, 200, 0],
    "right-leg-1":           [200, 200, 125],
    "right-leg-2":           [200, 200, 175],
    "right-leg-3":           [200, 200, 255],
}

# Ordered list of keypoint names (dict preserves insertion order in Py>=3.7).
KEYPOINT_NAMES = list(KEYPOINT_COLORS.keys())

# name -> index lookup (index in the model output array).
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

NUM_KEYPOINTS = len(KEYPOINT_NAMES)

# --------------------------------------------------------------------------- #
# Measurements -- ordered keypoint chains (value = sum of consecutive segments)
# --------------------------------------------------------------------------- #
MEASUREMENTS = {
    "total length":                ["head-top", "neck", "thorax-bottom", "body-tip"],
    "head width":                  ["head-left", "head-right"],
    "head length":                 ["head-top", "neck"],
    "inter ocular distance":       ["right-eye", "left-eye"],
    "right antenna length":        ["right-antenna-0", "right-antenna-1", "right-antenna-2"],
    "left antenna length":         ["left-antenna-0", "left-antenna-1", "left-antenna-2"],
    "thorax width":                ["thorax-left", "thorax-right"],
    "thorax length":               ["neck", "thorax-bottom"],
    "abdomen width":               ["body-left", "body-right"],
    "abdomen length":              ["thorax-bottom", "body-tip"],
    "intertegular distance":       ["left-hindwing-base", "right-hindwing-base"],
    "left hind wing length":       ["left-hindwing-base", "left-hindwing-tip"],
    "right hind wing length":      ["right-hindwing-base", "right-hindwing-tip"],
    "left hind wing width":        ["left-hindwing-front", "left-hindwing-rear"],
    "right hind wing width":       ["right-hindwing-front", "right-hindwing-rear"],
    "left fore wing length":       ["left-forewing-base", "left-forewing-tip"],
    "right fore wing length":      ["right-forewing-base", "right-forewing-tip"],
    "left fore wing width":        ["left-forewing-front", "left-forewing-rear"],
    "right fore wing width":       ["right-forewing-front", "right-forewing-rear"],
    "left hind leg length":        ["left-leg-0", "left-leg-1", "left-leg-2", "left-leg-3"],
    "left hind leg femur length":  ["left-leg-0", "left-leg-1"],
    "left hind leg tibia length":  ["left-leg-1", "left-leg-2"],
    "left hind leg tarsus length": ["left-leg-2", "left-leg-3"],
    "right hind leg length":       ["right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3"],
    "right hind leg femur length": ["right-leg-0", "right-leg-1"],
    "right hind leg tibia length": ["right-leg-1", "right-leg-2"],
    "right hind leg tarsus length":["right-leg-2", "right-leg-3"],
}

MEASUREMENT_NAMES = list(MEASUREMENTS.keys())

# Pre-computed keypoint indices for each measurement (used everywhere).
MEASUREMENT_INDICES = {
    name: [KEYPOINT_INDEX[kp] for kp in chain]
    for name, chain in MEASUREMENTS.items()
}

# Pairs of left/right measurements (handy later for a symmetry cross-check;
# not used by default but exposed here so it is easy to add).
SYMMETRIC_MEASUREMENT_PAIRS = [
    ("left antenna length",        "right antenna length"),
    ("left hind wing length",      "right hind wing length"),
    ("left hind wing width",       "right hind wing width"),
    ("left fore wing length",      "right fore wing length"),
    ("left fore wing width",       "right fore wing width"),
    ("left hind leg length",       "right hind leg length"),
    ("left hind leg femur length", "right hind leg femur length"),
    ("left hind leg tibia length", "right hind leg tibia length"),
    ("left hind leg tarsus length","right hind leg tarsus length"),
]


# --------------------------------------------------------------------------- #
# Left / right flip map (for the horizontal-flip TTA pass)
# --------------------------------------------------------------------------- #
def _mirror_name(name: str) -> str:
    """Return the left/right counterpart of a keypoint name.

    A horizontally flipped bee has its left and right parts swapped, so a model
    trained on normally-oriented images labels them the other way round. After a
    flip we therefore reorder the keypoints with this mapping. Midline keypoints
    (no 'left'/'right' token) map to themselves.
    """
    if "left" in name:
        return name.replace("left", "right")
    if "right" in name:
        return name.replace("right", "left")
    return name


# FLIP_INDEX[i] = index of the keypoint that keypoint i becomes after a flip.
# Used as a permutation:  keypoints_restored = keypoints_flipped[FLIP_INDEX]
FLIP_INDEX = [KEYPOINT_INDEX[_mirror_name(name)] for name in KEYPOINT_NAMES]
