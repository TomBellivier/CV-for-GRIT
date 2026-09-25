"""Tests unitaires de geometrie : la retro-projection est le bug classique du §9.3."""

from __future__ import annotations

import numpy as np

from insectpose.utils.geometry import (
    apply_affine,
    bbox_from_keypoints,
    bbox_iou,
    crop_affine,
    invert_affine,
    oks_matrix,
)


def test_crop_roundtrip_is_identity() -> None:
    rng = np.random.default_rng(0)
    bbox = np.array([30.0, 40.0, 120.0, 90.0])
    pts = rng.uniform(0, 200, size=(12, 2))
    matrix = crop_affine(bbox, (256, 256))
    back = apply_affine(invert_affine(matrix), apply_affine(matrix, pts))
    assert np.allclose(pts, back, atol=1e-6)


def test_crop_centers_bbox() -> None:
    bbox = np.array([10.0, 20.0, 40.0, 80.0])
    matrix = crop_affine(bbox, (100, 100))
    center = apply_affine(matrix, np.array([[30.0, 60.0]]))
    assert np.allclose(center, [[50.0, 50.0]], atol=1e-6)


def test_iou_identical_boxes_is_one() -> None:
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert np.isclose(bbox_iou(box, box)[0, 0], 1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    a = np.array([[0.0, 0.0, 5.0, 5.0]])
    b = np.array([[100.0, 100.0, 5.0, 5.0]])
    assert np.isclose(bbox_iou(a, b)[0, 0], 0.0)


def test_oks_is_one_for_perfect_prediction() -> None:
    gt = np.array([[[10.0, 10.0], [20.0, 20.0]]])
    vis = np.array([[2, 2]])
    sigmas = np.array([0.05, 0.05])
    areas = np.array([400.0])
    assert np.isclose(oks_matrix(gt, vis, gt, sigmas, areas)[0, 0], 1.0)


def test_oks_ignores_invisible_keypoints() -> None:
    gt = np.array([[[10.0, 10.0], [20.0, 20.0]]])
    pred = np.array([[[10.0, 10.0], [999.0, 999.0]]])
    vis = np.array([[2, 0]])
    value = oks_matrix(gt, vis, pred, np.array([0.05, 0.05]), np.array([400.0]))
    assert np.isclose(value[0, 0], 1.0)


def test_bbox_from_keypoints_ignores_absent_points() -> None:
    pts = np.array([[10.0, 10.0], [20.0, 30.0], [1000.0, 1000.0]])
    bbox = bbox_from_keypoints(pts, np.array([2, 2, 0]), margin=0.0)
    assert np.allclose(bbox, [10.0, 10.0, 10.0, 20.0])
