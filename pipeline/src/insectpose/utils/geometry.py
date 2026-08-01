"""Geometrie : bboxes, transformations affines, OKS (§3.4, §9.3).

REGLE : toute coordonnee manipulee ici est en pixels absolus dans le repere de
l'image d'origine, sauf quand une matrice affine est explicitement appliquee. La
retro-projection crop -> image est testee en aller-retour (tests/test_geometry.py).
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def bbox_diag(bbox_xywh: Array) -> Array:
    """Diagonale d'une (ou N) bbox(es) au format xywh."""
    b = np.atleast_2d(np.asarray(bbox_xywh, dtype=float))
    return np.sqrt(b[:, 2] ** 2 + b[:, 3] ** 2)


def bbox_area(bbox_xywh: Array) -> Array:
    """Aire d'une (ou N) bbox(es) au format xywh."""
    b = np.atleast_2d(np.asarray(bbox_xywh, dtype=float))
    return b[:, 2] * b[:, 3]


def xywh_to_xyxy(bbox_xywh: Array) -> Array:
    """Conversion xywh -> xyxy (pixels absolus)."""
    b = np.atleast_2d(np.asarray(bbox_xywh, dtype=float)).copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    return b


def bbox_iou(a_xywh: Array, b_xywh: Array) -> Array:
    """Matrice IoU (N_a, N_b) entre deux jeux de bboxes xywh."""
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
    """Bbox englobante des keypoints visibles, avec marge relative. Format xywh.

    Utilisee uniquement pour `bbox_source='derived'` : ce n'est jamais une detection.
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
    """Bruite une bbox GT (§9.3) : sans ce bruit, decalage train/test garanti."""
    x, y, w, h = np.asarray(bbox_xywh, dtype=float)
    s = float(np.exp(rng.normal(0.0, scale_std)))
    cx = x + w / 2 + rng.normal(0.0, shift_std) * w
    cy = y + h / 2 + rng.normal(0.0, shift_std) * h
    nw, nh = w * s, h * s
    return np.array([cx - nw / 2, cy - nh / 2, nw, nh])


def crop_affine(bbox_xywh: Array, out_wh: tuple[int, int], keep_aspect: bool = True) -> Array:
    """Matrice affine 2x3 image -> crop pour une bbox donnee.

    A conserver dans `meta.transform_matrix` : toute prediction faite dans le repere
    du crop DOIT etre retro-projetee avec `invert_affine` avant ecriture (§3.4).
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
    """Inverse d'une matrice affine 2x3."""
    m = np.asarray(matrix, dtype=float)
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)
    return inv[:2, :]


def apply_affine(matrix: Array, points_xy: Array) -> Array:
    """Applique une affine 2x3 a des points (N, 2) ou a un vecteur plat (2K,)."""
    pts = np.asarray(points_xy, dtype=float)
    flat = pts.ndim == 1
    p = pts.reshape(-1, 2)
    out = p @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]
    return out.reshape(-1) if flat else out


def oks_matrix(gt_kpts: Array, gt_vis: Array, pred_kpts: Array, sigmas: Array,
               gt_areas: Array, eps: float = 1e-9) -> Array:
    """Matrice OKS (P, G) entre predictions et verites terrain d'une image.

    gt_kpts : (G, K, 2) - pred_kpts : (P, K, 2) - gt_vis : (G, K) - sigmas : (K,)
    Les points `vis == 0` sont exclus du calcul (jamais comptes comme erreur nulle).
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
