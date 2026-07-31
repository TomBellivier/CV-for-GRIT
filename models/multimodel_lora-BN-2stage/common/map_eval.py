"""
Self-contained COCO-style mAP, used when Ultralytics' native validator cannot
be applied to the predictor as a whole (that is: the two-stage pipeline, where
detection and pose live in two different models).

Conventions are deliberately aligned with ``ultralytics.utils.metrics``:

* similarity thresholds 0.50:0.05:0.95 (10 values);
* box similarity = IoU;
* pose similarity = OKS with ``e = d^2 / (8 * sigma^2 * area)``, matching
  Ultralytics' ``kpt_iou``;
* per-keypoint sigma defaults to ``1 / n_kpts``, which is exactly what
  Ultralytics uses whenever ``kpt_shape != [17, 3]`` -- i.e. what your 42-keypoint
  models were scored with in the native runs;
* object area = box area * 0.53, Ultralytics' approximation of segmentation
  area from a bounding box;
* AP computed by 101-point interpolation, single class.

Caveat worth keeping in mind: this reimplementation tracks the native validator
closely but is not guaranteed to be numerically identical to it. Within one
workbook the numbers are consistent; across workbooks, compare native-to-native
and custom-to-custom. The ``map_source`` row in the metadata sheet records which
was used.
"""

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
AREA_FACTOR = 0.53
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _pairwise_box_iou(gt_boxes, pred_boxes):
    """[G, 4] x [P, 4] -> [G, P] IoU matrix."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))
    gt = np.asarray(gt_boxes, dtype=float)[:, None, :]
    pr = np.asarray(pred_boxes, dtype=float)[None, :, :]

    inter_w = np.clip(np.minimum(gt[..., 2], pr[..., 2])
                      - np.maximum(gt[..., 0], pr[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(gt[..., 3], pr[..., 3])
                      - np.maximum(gt[..., 1], pr[..., 1]), 0, None)
    inter = inter_w * inter_h

    area_gt = np.clip(gt[..., 2] - gt[..., 0], 0, None) * \
        np.clip(gt[..., 3] - gt[..., 1], 0, None)
    area_pr = np.clip(pr[..., 2] - pr[..., 0], 0, None) * \
        np.clip(pr[..., 3] - pr[..., 1], 0, None)
    union = area_gt + area_pr - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _pairwise_oks(gt_kpts, gt_boxes, pred_kpts, sigmas):
    """[G, K, 3] x [P, K, 3] -> [G, P] OKS matrix (Ultralytics convention)."""
    n_gt, n_pred = len(gt_kpts), len(pred_kpts)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred))

    gt = np.asarray(gt_kpts, dtype=float)
    pr = np.asarray(pred_kpts, dtype=float)
    boxes = np.asarray(gt_boxes, dtype=float)

    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * \
        np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    areas = np.maximum(areas * AREA_FACTOR, 1e-9)

    d2 = ((gt[:, None, :, 0] - pr[None, :, :, 0]) ** 2
          + (gt[:, None, :, 1] - pr[None, :, :, 1]) ** 2)

    sigmas = np.asarray(sigmas, dtype=float)[None, None, :]
    denom = (2 * sigmas) ** 2 * areas[:, None, None] * 2
    exponent = np.exp(-d2 / np.maximum(denom, 1e-12))

    mask = (gt[..., 2] > 0)[:, None, :]
    n_visible = mask.sum(axis=2)
    return (exponent * mask).sum(axis=2) / np.maximum(n_visible, 1e-9)


def _average_precision(tp, scores, n_gt):
    """101-point interpolated AP for one similarity threshold."""
    if n_gt == 0:
        return np.nan
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-scores)
    tp = tp[order].astype(float)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Make precision monotonically decreasing, then sample at 101 recalls.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.interp(RECALL_POINTS, recall, precision, left=precision[0],
                           right=0.0).mean())


class MapCollector:
    """Accumulate GT and predictions image by image, then compute mAP.

    A single class is assumed (the evaluation is per group, and each group's
    dataset holds one object class). Predictions with no GT in their image are
    correctly counted as false positives.
    """

    def __init__(self, n_kpts, kpt_sigma=None):
        self.n_kpts = n_kpts
        if kpt_sigma is None:
            self.sigmas = np.ones(n_kpts) / n_kpts
        else:
            self.sigmas = np.full(n_kpts, float(kpt_sigma))
        self.images = []
        self.n_gt = 0

    def add(self, gt_boxes, gt_kpts, pred_boxes, pred_kpts, pred_scores):
        gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
        gt_kpts = np.asarray(gt_kpts, dtype=float).reshape(-1, self.n_kpts, 3)
        pred_kpts = np.asarray(pred_kpts, dtype=float).reshape(-1, self.n_kpts, 3)
        pred_scores = np.asarray(pred_scores, dtype=float).reshape(-1)

        self.n_gt += len(gt_boxes)
        self.images.append({
            "gt_boxes": gt_boxes,
            "gt_kpts": gt_kpts,
            "pred_boxes": pred_boxes,
            "pred_kpts": pred_kpts,
            "scores": pred_scores,
        })

    def _match_all(self, similarity_fn):
        """Return (tp [N, T], scores [N]) over the whole dataset."""
        tp_chunks, score_chunks = [], []
        n_thr = len(IOU_THRESHOLDS)

        for item in self.images:
            n_pred = len(item["pred_boxes"])
            if n_pred == 0:
                continue
            scores = item["scores"]
            sim = similarity_fn(item)  # [G, P]
            tp = np.zeros((n_pred, n_thr), dtype=bool)

            if sim.size:
                order = np.argsort(-scores)
                for ti, thr in enumerate(IOU_THRESHOLDS):
                    matched_gt = set()
                    for pi in order:
                        column = sim[:, pi]
                        best_gi, best_val = -1, thr
                        for gi in range(len(column)):
                            if gi in matched_gt:
                                continue
                            if column[gi] >= best_val:
                                best_val, best_gi = column[gi], gi
                        if best_gi >= 0:
                            matched_gt.add(best_gi)
                            tp[pi, ti] = True

            tp_chunks.append(tp)
            score_chunks.append(scores)

        if not tp_chunks:
            return np.zeros((0, n_thr), dtype=bool), np.zeros(0)
        return np.concatenate(tp_chunks, 0), np.concatenate(score_chunks, 0)

    def _compute(self, similarity_fn):
        tp, scores = self._match_all(similarity_fn)
        aps = [_average_precision(tp[:, ti], scores, self.n_gt)
               for ti in range(len(IOU_THRESHOLDS))]
        aps = np.asarray(aps, dtype=float)
        return {
            "map": float(np.nanmean(aps)) if len(aps) else float("nan"),
            "map50": float(aps[0]),
            "map75": float(aps[5]),
        }

    def results(self):
        box = self._compute(
            lambda item: _pairwise_box_iou(item["gt_boxes"], item["pred_boxes"]))
        pose = self._compute(
            lambda item: _pairwise_oks(item["gt_kpts"], item["gt_boxes"],
                                       item["pred_kpts"], self.sigmas))
        return {
            "pose_map": pose["map"],
            "pose_map50": pose["map50"],
            "pose_map75": pose["map75"],
            "box_map": box["map"],
            "box_map50": box["map50"],
        }
