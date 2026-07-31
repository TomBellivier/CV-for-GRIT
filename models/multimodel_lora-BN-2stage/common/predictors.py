"""
Predictor abstraction.

The evaluation driver only needs two things from whatever is being evaluated:

    predict(image_path) -> boxes[N,4], kpts[N,K,3], scores[N], (h, w)
    native_val(data_yaml, imgsz, device) -> dict | None

Three implementations are provided:

* ``SingleModelPredictor``   -- one plain YOLO-pose model (baseline, and the
  shared base model of the LoRA / group-BN approaches).
* ``VariantPredictor``       -- one shared backbone whose per-group behaviour is
  switched by ``set_group``; used for both LoRA adapters and group BatchNorm.
* ``TwoStagePredictor``      -- detector on the full image, then pose model on
  each expanded crop, with keypoints mapped back to original coordinates.

``native_val`` returns ``None`` when Ultralytics' validator cannot score the
predictor as a whole; the driver then falls back to ``map_eval.MapCollector``.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class BasePredictor:
    def set_group(self, group_name):
        """Switch to a group-specific configuration. No-op by default."""

    def predict(self, image_path):
        raise NotImplementedError

    def native_val(self, data_yaml, imgsz, device):
        return None


def _unpack(result, n_kpts):
    """Turn one Ultralytics Result into plain numpy arrays."""
    img_h, img_w = result.orig_shape
    if result.boxes is None or len(result.boxes) == 0:
        return (np.zeros((0, 4)), np.zeros((0, n_kpts, 3)), np.zeros(0),
                (img_h, img_w))
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    if result.keypoints is None:
        kpts = np.zeros((len(boxes), n_kpts, 3))
    else:
        kpts = result.keypoints.data.cpu().numpy()
        if kpts.shape[-1] == 2:  # no visibility channel
            pad = np.ones((*kpts.shape[:2], 1))
            kpts = np.concatenate([kpts, pad], axis=-1)
    return boxes, kpts, scores, (img_h, img_w)


class SingleModelPredictor(BasePredictor):
    """A single YOLO-pose model applied to every group."""

    def __init__(self, model, n_kpts, conf=0.25, imgsz=640, device=None):
        self.model = model if isinstance(model, YOLO) else YOLO(str(model))
        self.n_kpts = n_kpts
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def predict(self, image_path):
        result = self.model.predict(str(image_path), conf=self.conf,
                                    imgsz=self.imgsz, device=self.device,
                                    verbose=False)[0]
        return _unpack(result, self.n_kpts)

    def native_val(self, data_yaml, imgsz, device):
        metrics = self.model.val(data=data_yaml, imgsz=imgsz, device=device,
                                 verbose=False)
        return {
            "pose_map": float(metrics.pose.map),
            "pose_map50": float(metrics.pose.map50),
            "pose_map75": float(metrics.pose.map75),
            "box_map": float(metrics.box.map),
            "box_map50": float(metrics.box.map50),
        }


class VariantPredictor(BasePredictor):
    """Shared weights + a per-group variant swapped in before inference.

    ``switch_fn(group_name)`` is supplied by the calling project: it flips the
    active LoRA adapter set, or the active BatchNorm bank, on the underlying
    torch module. The YOLO wrapper object is reused across groups, so the shared
    weights are held in memory exactly once -- which is the whole point of both
    approaches.
    """

    def __init__(self, yolo_model, switch_fn, n_kpts, conf=0.25, imgsz=640,
                 device=None):
        self.model = yolo_model
        self.switch_fn = switch_fn
        self.n_kpts = n_kpts
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.active_group = None

    def set_group(self, group_name):
        self.switch_fn(group_name)
        self.active_group = group_name

    def predict(self, image_path):
        result = self.model.predict(str(image_path), conf=self.conf,
                                    imgsz=self.imgsz, device=self.device,
                                    verbose=False)[0]
        return _unpack(result, self.n_kpts)

    def native_val(self, data_yaml, imgsz, device):
        # The active variant is already installed on self.model.model, so the
        # native validator scores the group-specific configuration.
        metrics = self.model.val(data=data_yaml, imgsz=imgsz, device=device,
                                 verbose=False)
        return {
            "pose_map": float(metrics.pose.map),
            "pose_map50": float(metrics.pose.map50),
            "pose_map75": float(metrics.pose.map75),
            "box_map": float(metrics.box.map),
            "box_map50": float(metrics.box.map50),
        }


def expand_box(box, img_w, img_h, margin):
    """Expand an xyxy box by ``margin`` around its centre, clamped to the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) * margin / 2.0
    half_h = (y2 - y1) * margin / 2.0
    nx1 = int(max(0, round(cx - half_w)))
    ny1 = int(max(0, round(cy - half_h)))
    nx2 = int(min(img_w, round(cx + half_w)))
    ny2 = int(min(img_h, round(cy + half_h)))
    if nx2 <= nx1:
        nx1, nx2 = max(0, nx1 - 1), min(img_w, nx1 + 1)
    if ny2 <= ny1:
        ny1, ny2 = max(0, ny1 - 1), min(img_h, ny1 + 1)
    return nx1, ny1, nx2, ny2


class TwoStagePredictor(BasePredictor):
    """Top-down pipeline: detect on the full image, estimate pose on each crop.

    The pose stage runs at a low confidence threshold and keeps the single
    highest-scoring instance inside each crop. That is the standard top-down
    protocol: the crop is known to contain exactly one insect, so suppressing it
    on confidence would only ever lose a keypoint set.

    Box coordinates reported back are the *detector's* boxes in original image
    coordinates; keypoints are the pose model's, translated back from crop space.
    """

    def __init__(self, det_model, pose_model, n_kpts, margin=1.25,
                 det_conf=0.25, pose_conf=0.01, det_imgsz=640, pose_imgsz=320,
                 device=None, max_instances=20):
        self.det = det_model if isinstance(det_model, YOLO) else YOLO(str(det_model))
        self.pose = pose_model if isinstance(pose_model, YOLO) else YOLO(str(pose_model))
        self.n_kpts = n_kpts
        self.margin = margin
        self.det_conf = det_conf
        self.pose_conf = pose_conf
        self.det_imgsz = det_imgsz
        self.pose_imgsz = pose_imgsz
        self.device = device
        self.max_instances = max_instances
        self.group_class = None  # optional: restrict detections to one class

    def set_group(self, group_name):
        """Optionally filter detections by predicted taxon.

        Left as a no-op unless ``group_class_map`` was installed by the caller;
        keeping every detection is the honest default, since the pipeline is
        supposed to work without being told the group in advance.
        """
        mapping = getattr(self, "group_class_map", None)
        self.group_class = mapping.get(group_name) if mapping else None

    def predict(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return (np.zeros((0, 4)), np.zeros((0, self.n_kpts, 3)),
                    np.zeros(0), (0, 0))
        img_h, img_w = image.shape[:2]

        det = self.det.predict(str(image_path), conf=self.det_conf,
                               imgsz=self.det_imgsz, device=self.device,
                               verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            return (np.zeros((0, 4)), np.zeros((0, self.n_kpts, 3)),
                    np.zeros(0), (img_h, img_w))

        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int)

        if self.group_class is not None:
            keep = classes == self.group_class
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        # Highest-scoring instances first, capped to keep runtime bounded.
        order = np.argsort(-scores)[:self.max_instances]
        boxes, scores = boxes[order], scores[order]

        out_boxes, out_kpts, out_scores = [], [], []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = expand_box(box, img_w, img_h, self.margin)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pose_res = self.pose.predict(crop, conf=self.pose_conf,
                                         imgsz=self.pose_imgsz,
                                         device=self.device, verbose=False)[0]
            if pose_res.boxes is None or len(pose_res.boxes) == 0 \
                    or pose_res.keypoints is None:
                continue

            best = int(np.argmax(pose_res.boxes.conf.cpu().numpy()))
            kpts = pose_res.keypoints.data.cpu().numpy()[best]
            if kpts.shape[-1] == 2:
                kpts = np.concatenate([kpts, np.ones((len(kpts), 1))], axis=-1)

            kpts = kpts.copy()
            kpts[:, 0] += x1
            kpts[:, 1] += y1

            out_boxes.append(box)
            out_kpts.append(kpts)
            out_scores.append(score)

        if not out_boxes:
            return (np.zeros((0, 4)), np.zeros((0, self.n_kpts, 3)),
                    np.zeros(0), (img_h, img_w))

        return (np.asarray(out_boxes), np.asarray(out_kpts),
                np.asarray(out_scores), (img_h, img_w))

    def native_val(self, data_yaml, imgsz, device):
        # No single Ultralytics model represents this pipeline, so the driver
        # must fall back to the custom mAP implementation.
        return None
