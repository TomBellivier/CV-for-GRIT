"""Tests des metriques sur cas calcules a la main (§10.2).

Une metrique dont on ne sait pas predire la valeur sur un cas trivial n'est pas
utilisable pour departager des approches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insectpose.data.keypoints import load_schema
from insectpose.evaluation.evaluator import evaluate_predictions
from insectpose.evaluation.matching import assign_greedy, average_precision

SCHEMA = "insect42_v1"


def _template(k: int) -> list[float]:
    """Instance jouet : points repartis, thorax-left/right ecartes de 20 px."""
    pts = np.stack([np.linspace(20, 80, k), np.linspace(20, 80, k)], axis=1)
    pts[6] = [40.0, 50.0]   # thorax-left
    pts[7] = [60.0, 50.0]   # thorax-right -> largeur de thorax = 20 px
    return [float(v) for v in pts.reshape(-1)]


def _gt_row(image_id: str, kpts: list[float], k: int) -> dict:
    return {
        "schema_version": 1, "dataset": "coleoptera", "image_id": image_id,
        "image_path": "raw/x.png", "image_width": 100, "image_height": 100,
        "instance_id": f"{image_id}#0", "group_id": image_id,
        "bbox_xywh": [0.0, 0.0, 100.0, 100.0], "kpts_xy": kpts,
        "kpts_vis": [2] * k, "area": 10000.0, "keypoint_schema": SCHEMA,
        "split_source": "unknown", "qc_flags": "",
    }


def _pred_row(image_id: str, kpts: list[float], k: int, score: float = 1.0) -> dict:
    return {
        "schema_version": 1, "run_id": "r", "fold": 0, "split": "test",
        "dataset": "coleoptera", "image_id": image_id, "pred_id": f"{image_id}#p",
        "bbox_xywh": [0.0, 0.0, 100.0, 100.0], "bbox_score": score, "kpts_xy": kpts,
        "kpts_score": [1.0] * k, "keypoint_schema": SCHEMA,
        "bbox_source": "predicted", "inference_ms": 1.0,
    }


@pytest.fixture()
def eval_cfg(cfg):
    return cfg.eval


def test_perfect_predictions_give_perfect_scores(project, eval_cfg) -> None:
    schema = load_schema(SCHEMA, project.configs)
    k = schema.n_keypoints
    kpts = _template(k)
    gt = pd.DataFrame([_gt_row("coleoptera/a", kpts, k)])
    pred = pd.DataFrame([_pred_row("coleoptera/a", kpts, k)])

    out = evaluate_predictions(pred, gt, {SCHEMA: schema}, eval_cfg)
    values = out.set_index(["scope", "metric"])["value"]
    assert values[("overall", "oks_ap")] == pytest.approx(1.0)
    assert values[("overall", "pck@0.25_thorax_width")] == pytest.approx(1.0)
    assert values[("overall", "nme_matched_only")] == pytest.approx(0.0, abs=1e-9)


def test_missed_instance_is_counted_as_failure(project, eval_cfg) -> None:
    """Une image non predite doit faire chuter le PCK, pas disparaitre du denominateur."""
    schema = load_schema(SCHEMA, project.configs)
    k = schema.n_keypoints
    kpts = _template(k)
    gt = pd.DataFrame([_gt_row("coleoptera/a", kpts, k), _gt_row("coleoptera/b", kpts, k)])
    pred = pd.DataFrame([_pred_row("coleoptera/a", kpts, k)])

    out = evaluate_predictions(pred, gt, {SCHEMA: schema}, eval_cfg)
    values = out.set_index(["scope", "metric"])["value"]
    assert values[("overall", "pck@0.25_thorax_width")] == pytest.approx(0.5)
    assert values[("overall", "kpt_coverage")] == pytest.approx(0.5)
    assert values[("overall", "oks_ar")] == pytest.approx(0.5)


def test_average_precision_hand_computed() -> None:
    scores = np.array([0.9, 0.8, 0.7])
    tp = np.array([True, False, True])
    # rappels 1/2 puis 1 ; precisions 1.0 et 2/3 -> AP 101 points
    value = average_precision(scores, tp, n_gt=2)
    assert 0.75 < value < 0.90


def test_average_precision_is_nan_without_gt() -> None:
    assert np.isnan(average_precision(np.array([0.9]), np.array([False]), n_gt=0))


def test_greedy_assignment_respects_score_order() -> None:
    similarity = np.array([[0.9, 0.4], [0.8, 0.3]])
    matched, _ = assign_greedy(similarity, np.array([0.5, 0.99]), threshold=0.5)
    assert matched[1] == 0   # la prediction la mieux notee prend le meilleur GT
    assert matched[0] == -1  # l'autre GT est sous le seuil


def test_prediction_in_wrong_schema_is_rejected(project, eval_cfg) -> None:
    schema = load_schema(SCHEMA, project.configs)
    k = schema.n_keypoints
    kpts = _template(k)
    gt = pd.DataFrame([_gt_row("coleoptera/a", kpts, k)])
    pred = pd.DataFrame([_pred_row("coleoptera/a", kpts, k)])
    pred["keypoint_schema"] = "insect42_v2"
    from insectpose.contracts import ContractError

    with pytest.raises(ContractError, match="schema"):
        evaluate_predictions(pred, gt, {SCHEMA: schema}, eval_cfg)
