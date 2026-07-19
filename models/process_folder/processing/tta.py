"""
tta.py
======

Test-Time Augmentation (TTA) for the "stability" measurement-confidence signal.

Idea
----
Run the pose model several times on slightly perturbed copies of the image,
map every prediction back into the ORIGINAL image frame, recompute the
measurements, and hand the per-pass values to confidence.measurement_confidence_tta.
A measurement that barely moves across augmentations is trustworthy.

Correctness notes
-----------------
* Geometric augmentations (flip, rotation) are applied to the image with an
  affine matrix M. The predicted keypoints live in the augmented frame, so we
  map them BACK with the inverse affine before measuring. Otherwise the values
  would not be comparable across passes.
* A horizontal flip additionally swaps the left/right keypoints, because a
  model trained on normal images labels the mirrored insect the other way
  round. We undo that with definitions.FLIP_INDEX.
* Photometric augmentations (brightness) do not move points, so their transform
  is the identity.

Everything here is optional and only runs when config.ENABLE_TTA is True (or
when the measurement signal is set to "tta").
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .definitions import FLIP_INDEX, MEASUREMENT_NAMES
from .measurements import compute_measurements
from .pose_inference import run_pose_on_array


# --------------------------------------------------------------------------- #
# Augmentation specification
# --------------------------------------------------------------------------- #
class _Augmentation:
    """One augmented pass: how to warp the image and how to undo it on points.

    Attributes
    ----------
    name       : label for logging.
    M          : 2x3 affine matrix mapping ORIGINAL -> AUGMENTED pixel coords
                 (None means no geometric change).
    brightness : multiplicative photometric factor (1.0 = unchanged).
    swap_lr    : whether to swap left/right keypoints after mapping back
                 (True only for the horizontal flip).
    """

    def __init__(self, name, M=None, brightness=1.0, swap_lr=False):
        self.name = name
        self.M = M
        self.brightness = brightness
        self.swap_lr = swap_lr

    # -- forward: build the augmented image ---------------------------------- #
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self.M is not None:
            h, w = img.shape[:2]
            out = cv2.warpAffine(out, self.M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        if self.brightness != 1.0:
            out = np.clip(out.astype(np.float32) * self.brightness, 0, 255).astype(np.uint8)
        return out

    # -- backward: map predicted keypoints back to the original frame -------- #
    def invert_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        kp = keypoints.copy()
        if self.M is not None:
            M_inv = cv2.invertAffineTransform(self.M)          # 2x3
            xy = kp[:, :2]
            ones = np.ones((xy.shape[0], 1), dtype=float)
            xy_h = np.concatenate([xy, ones], axis=1)          # (n, 3)
            kp[:, :2] = xy_h @ M_inv.T                         # (n, 2)
        if self.swap_lr:
            # Reorder rows so semantics match the un-flipped prediction.
            kp = kp[FLIP_INDEX]
        return kp


def _build_augmentations(image_shape) -> list[_Augmentation]:
    """Build the list of augmentations from the config toggles."""
    h, w = image_shape[:2]
    center = (w / 2.0, h / 2.0)
    augs: list[_Augmentation] = []

    if config.TTA_INCLUDE_IDENTITY:
        augs.append(_Augmentation("identity"))

    if config.TTA_USE_HFLIP:
        # Horizontal flip as an affine matrix: x' = w - 1 - x, y' = y.
        M_flip = np.array([[-1.0, 0.0, w - 1.0],
                           [0.0, 1.0, 0.0]], dtype=np.float32)
        augs.append(_Augmentation("hflip", M=M_flip, swap_lr=True))

    for angle in config.TTA_ROTATION_DEGREES:
        M_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0).astype(np.float32)
        augs.append(_Augmentation(f"rot{angle:+.0f}", M=M_rot))

    for factor in config.TTA_BRIGHTNESS_FACTORS:
        augs.append(_Augmentation(f"bright{factor:.2f}", brightness=float(factor)))

    return augs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def collect_tta_measurements(model, img_bgr: np.ndarray) -> dict[str, list[float]]:
    """Run all TTA passes and gather the per-pass value of each measurement.

    Returns
    -------
    dict {measurement_name: [value_pass_1, value_pass_2, ...]}
        Passes where no instance is detected are simply skipped, so a
        measurement may have fewer values than there are augmentations.
    """
    augs = _build_augmentations(img_bgr.shape)
    per_measure: dict[str, list[float]] = {name: [] for name in MEASUREMENT_NAMES}

    for aug in augs:
        aug_img = aug.apply_image(img_bgr)
        pose = run_pose_on_array(model, aug_img)
        if pose is None:
            continue
        kp_original_frame = aug.invert_keypoints(pose.keypoints)
        values = compute_measurements(kp_original_frame)
        for name, val in values.items():
            per_measure[name].append(val)

    return per_measure
