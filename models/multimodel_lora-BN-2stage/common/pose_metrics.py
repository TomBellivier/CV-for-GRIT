"""
Keypoint metric primitives.

Everything in this module is a verbatim port of the corresponding code in the
original ``train_eval_pose.py``. It is kept byte-for-byte equivalent on purpose:
the numbers produced here must stay directly comparable with the workbooks that
were already generated with the original script.

Do not change PCK_THRESHOLDS, OKS_SIGMA or the accumulator formulas unless you
intend to re-evaluate every previous run as well.
"""

import math

import numpy as np
import pandas as pd

# PCK thresholds, expressed as a fraction of the ground-truth bbox diagonal.
PCK_THRESHOLDS = [0.05, 0.1]

# Constant per-keypoint sigma used for the custom OKS estimate. Without an
# established sigma table for these keypoints, a single value is a reasonable
# starting point and can be tuned later.
OKS_SIGMA = 0.05


def parse_label_file(label_path, img_w, img_h, n_kpts, kpt_dim):
    """Read one YOLO-pose label file into a list of GT instances (pixels).

    The per-keypoint dimension is inferred from the actual line width when it
    disagrees with kpt_dim, so a [N, 3] kpt_shape paired with x,y-only labels
    (or the reverse) is still parsed instead of being silently dropped.
    """
    instances = []
    if not label_path.exists():
        return instances

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < 5 + n_kpts * 2:
                continue
            values = [float(v) for v in tokens]
            n_fields = len(values) - 5
            dim = kpt_dim
            if n_fields != n_kpts * kpt_dim and n_fields % n_kpts == 0:
                dim = n_fields // n_kpts
            if dim not in (2, 3) or n_fields < n_kpts * dim:
                continue

            cx, cy, bw, bh = values[1:5]
            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h

            raw_kpts = np.array(values[5:5 + n_kpts * dim], dtype=float)
            raw_kpts = raw_kpts.reshape(n_kpts, dim)
            kpts = np.zeros((n_kpts, 3), dtype=float)
            kpts[:, 0] = raw_kpts[:, 0] * img_w
            kpts[:, 1] = raw_kpts[:, 1] * img_h
            kpts[:, 2] = raw_kpts[:, 2] if dim == 3 else 2.0

            instances.append({
                "cls": int(values[0]),
                "box": np.array([x1, y1, x2, y2]),
                "kpts": kpts,
            })
    return instances


def box_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_instances(gt_boxes, pred_boxes, iou_thr):
    """Greedy GT-to-prediction matching by IoU. Returns (gt_idx, pred_idx)."""
    pairs = []
    used = set()
    for gi, gt_box in enumerate(gt_boxes):
        best_pi, best_iou = -1, iou_thr
        for pi, pred_box in enumerate(pred_boxes):
            if pi in used:
                continue
            iou = box_iou(gt_box, pred_box)
            if iou >= best_iou:
                best_iou, best_pi = iou, pi
        if best_pi >= 0:
            pairs.append((gi, best_pi))
            used.add(best_pi)
    return pairs


class KeypointAccumulator:
    """Aggregate per-keypoint distances, PCK hits and OKS over all matches."""

    def __init__(self, n_kpts, thresholds):
        self.n_kpts = n_kpts
        self.thresholds = thresholds
        self.dist_px = np.zeros(n_kpts)
        self.dist_norm = np.zeros(n_kpts)
        self.conf_sum = np.zeros(n_kpts)
        self.count = np.zeros(n_kpts)
        self.pck_hits = {thr: np.zeros(n_kpts) for thr in thresholds}
        self.oks_values = []
        self.n_matched = 0

    def add(self, gt_kpts, pred_kpts, gt_box):
        bw = gt_box[2] - gt_box[0]
        bh = gt_box[3] - gt_box[1]
        diag = math.hypot(bw, bh)
        area = max(bw * bh, 1e-6)
        if diag <= 0:
            return
        self.n_matched += 1

        visible = gt_kpts[:, 2] > 0
        deltas = gt_kpts[:, :2] - pred_kpts[:, :2]
        dists = np.linalg.norm(deltas, axis=1)

        oks_terms = []
        for k in range(self.n_kpts):
            if not visible[k]:
                continue
            d = dists[k]
            self.dist_px[k] += d
            self.dist_norm[k] += d / diag
            self.conf_sum[k] += pred_kpts[k, 2]
            self.count[k] += 1
            for thr in self.thresholds:
                if d / diag <= thr:
                    self.pck_hits[thr][k] += 1
            oks_terms.append(math.exp(-(d ** 2) / (2 * area * OKS_SIGMA ** 2)))

        if oks_terms:
            self.oks_values.append(float(np.mean(oks_terms)))

    def per_keypoint_frame(self, kpt_names):
        rows = []
        for k in range(self.n_kpts):
            n = self.count[k]
            name = kpt_names[k] if k < len(kpt_names) else str(k)
            row = {
                "kpt_index": k,
                "kpt_name": name,
                "n_obs": int(n),
                "kpt_conf": self.conf_sum[k] / n if n > 0 else np.nan,
                "mpjpe_px": self.dist_px[k] / n if n > 0 else np.nan,
                "nmpjpe": self.dist_norm[k] / n if n > 0 else np.nan,
            }
            for thr in self.thresholds:
                row[f"pck_{thr}"] = self.pck_hits[thr][k] / n if n > 0 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self):
        total = self.count.sum()
        out = {
            "num_matched": self.n_matched,
            "mean_kpt_conf": self.conf_sum.sum() / total if total > 0 else np.nan,
            "mpjpe_px": self.dist_px.sum() / total if total > 0 else np.nan,
            "nmpjpe": self.dist_norm.sum() / total if total > 0 else np.nan,
            "mean_oks": float(np.mean(self.oks_values)) if self.oks_values
            else np.nan,
        }
        for thr in self.thresholds:
            hits = self.pck_hits[thr].sum()
            out[f"pck_{thr}"] = hits / total if total > 0 else np.nan
        return out
