"""Geometry: bboxes, affine transforms, OKS (§3.4, §9.3).

RULE: every coordinate handled here is in absolute pixels in the frame of the original
image, except when an affine matrix is explicitly applied. The crop -> image
back-projection is tested by a round trip (tests/test_geometry.py).
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def bbox_diag(bbox_xywh: Array) -> Array:
    """Diagonal of one (or N) bbox(es) in the xywh format."""
    b = np.atleast_2d(np.asarray(bbox_xywh, dtype=float))
    return np.sqrt(b[:, 2] ** 2 + b[:, 3] ** 2)


def bbox_area(bbox_xywh: Array) -> Array:
    """Area of one (or N) bbox(es) in the xywh format."""
    b = np.atleast_2d(np.asarray(bbox_xywh, dtype=float))
    return b[:, 2] * b[:, 3]


def xywh_to_xyxy(bbox_xywh: Array) -> Array:
    """xywh -> xyxy conversion (absolute pixels)."""
    b = np.atleast_2d(np.asarray(bbox_xywh, dtype=float)).copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    return b


def bbox_iou(a_xywh: Array, b_xywh: Array) -> Array:
    """IoU matrix (N_a, N_b) between two sets of xywh bboxes."""
    a = xywh_to_xyxy(a_xywh)
    b = xywh_to_xyxy(b_xywh)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def bbox_from_keypoints(kpts_xy: Array, kpts_vis: Array, margin: float = 0.05,
                        image_wh: tuple[int, int] | None = None) -> Array:
    """Bounding box of the visible keypoints, with a relative margin. xywh format.

    Only used for `bbox_source='derived'`: it is never a detection.
    """
    pts = np.asarray(kpts_xy, dtype=float).reshape(-1, 2)
    vis = np.asarray(kpts_vis).reshape(-1)
    sel = pts[vis > 0]
    if sel.size == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    x0, y0 = sel.min(axis=0)
    x1, y1 = sel.max(axis=0)
    w, h = x1 - x0, y1 - y0
    x0 -= margin * w
    y0 -= margin * h
    w *= 1 + 2 * margin
    h *= 1 + 2 * margin
    if image_wh is not None:
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        w = min(w, image_wh[0] - x0)
        h = min(h, image_wh[1] - y0)
    return np.array([x0, y0, w, h])


def jitter_bbox(bbox_xywh: Array, rng: np.random.Generator, scale_std: float = 0.15,
                shift_std: float = 0.10) -> Array:
    """Add noise to a GT bbox (§9.3): without this noise, a train/test shift is guaranteed."""
    x, y, w, h = np.asarray(bbox_xywh, dtype=float)
    s = float(np.exp(rng.normal(0.0, scale_std)))
    cx = x + w / 2 + rng.normal(0.0, shift_std) * w
    cy = y + h / 2 + rng.normal(0.0, shift_std) * h
    nw, nh = w * s, h * s
    return np.array([cx - nw / 2, cy - nh / 2, nw, nh])


def crop_affine(bbox_xywh: Array, out_wh: tuple[int, int], keep_aspect: bool = True) -> Array:
    """2x3 image -> crop affine matrix for a given bbox.

    To be kept in `meta.transform_matrix`: every prediction made in the frame of the
    crop MUST be back-projected with `invert_affine` before writing (§3.4).
    """
    x, y, w, h = np.asarray(bbox_xywh, dtype=float)
    ow, oh = out_wh
    if keep_aspect:
        s = min(ow / max(w, 1e-9), oh / max(h, 1e-9))
        sx = sy = s
    else:
        sx, sy = ow / max(w, 1e-9), oh / max(h, 1e-9)
    tx = ow / 2 - sx * (x + w / 2)
    ty = oh / 2 - sy * (y + h / 2)
    return np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=float)


def invert_affine(matrix: Array) -> Array:
    """Inverse of a 2x3 affine matrix."""
    m = np.asarray(matrix, dtype=float)
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)
    return inv[:2, :]


def apply_affine(matrix: Array, points_xy: Array) -> Array:
    """Apply a 2x3 affine to points (N, 2) or to a flat vector (2K,)."""
    pts = np.asarray(points_xy, dtype=float)
    flat = pts.ndim == 1
    p = pts.reshape(-1, 2)
    out = p @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]
    return out.reshape(-1) if flat else out


def oks_matrix(gt_kpts: Array, gt_vis: Array, pred_kpts: Array, sigmas: Array,
               gt_areas: Array, eps: float = 1e-9) -> Array:
    """OKS matrix (P, G) between the predictions and the ground truths of an image.

    gt_kpts: (G, K, 2) - pred_kpts: (P, K, 2) - gt_vis: (G, K) - sigmas: (K,)
    Points with `vis == 0` are excluded from the computation (never counted as a zero error).
    """
    g = np.asarray(gt_kpts, dtype=float)
    p = np.asarray(pred_kpts, dtype=float)
    vis = np.asarray(gt_vis) > 0
    if g.shape[0] == 0 or p.shape[0] == 0:
        return np.zeros((p.shape[0], g.shape[0]), dtype=float)
    k = 2.0 * np.asarray(sigmas, dtype=float)[None, None, :]
    d2 = ((p[:, None, :, :] - g[None, :, :, :]) ** 2).sum(axis=-1)  # (P, G, K)
    s2 = np.asarray(gt_areas, dtype=float)[None, :, None]
    e = d2 / (2.0 * np.maximum(s2, eps) * k**2 + eps)
    mask = vis[None, :, :]
    n_vis = mask.sum(axis=-1)
    scores = np.where(mask, np.exp(-e), 0.0).sum(axis=-1)
    return np.where(n_vis > 0, scores / np.maximum(n_vis, 1), 0.0)
