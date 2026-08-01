"""Tests des decisions de protocole : mesures, normalisateur PCK, resolution commune."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.contracts import ContractError
from insectpose.data.keypoints import load_schema
from insectpose.data.measurements import load_measurements, measure_all, polyline_length
from insectpose.evaluation.evaluator import evaluate_predictions
from insectpose.evaluation.metrics.pose import compute_normalizer

SCHEMA = "insect42_v1"


# --- mesures morphometriques (ADR-0008) -------------------------------------
def test_measurement_definitions_match_schema(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    spec = load_measurements(project.configs / "measurements" / "insect42_v1.yaml")
    index = spec.indices(schema)
    assert len(index) == 27
    assert len(spec.symmetric_pairs) == 9
    expected = [schema.index("thorax-left"), schema.index("thorax-right")]
    assert list(index["thorax width"]) == expected


def test_polyline_length_is_sum_of_segments() -> None:
    pts = np.array([[[0.0, 0.0], [3.0, 4.0], [3.0, 9.0]]])
    assert polyline_length(pts)[0] == pytest.approx(10.0)


def test_measure_all_on_known_geometry(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    spec = load_measurements(project.configs / "measurements" / "insect42_v1.yaml")
    kpts = np.zeros((1, schema.n_keypoints, 2))
    kpts[0, schema.index("thorax-left")] = [10.0, 0.0]
    kpts[0, schema.index("thorax-right")] = [30.0, 0.0]
    values = measure_all(kpts, spec.indices(schema))
    assert values["thorax width"][0] == pytest.approx(20.0)


def test_measurement_error_is_zero_for_perfect_prediction(cfg, project) -> None:
    gt, pred = _instance_frames(project)
    schemas = {SCHEMA: load_schema(SCHEMA, project.configs)}
    out = evaluate_predictions(pred, gt, schemas, cfg.eval)
    values = out.set_index(["scope", "metric"])["value"]
    assert values[("overall", "measurement_mape_median")] == pytest.approx(0.0, abs=1e-9)
    assert ("measurement:thorax width", "mape_median") in values.index


def test_measurement_error_grows_with_displacement(cfg, project) -> None:
    """Un point deplace doit degrader la mesure qui l'utilise, et elle seule."""
    gt, pred = _instance_frames(project)
    schema = load_schema(SCHEMA, project.configs)
    kpts = np.asarray(pred.loc[0, "kpts_xy"], dtype=float).reshape(-1, 2)
    gt_width = float(np.linalg.norm(kpts[schema.index("thorax-left")]
                                    - kpts[schema.index("thorax-right")]))
    kpts[schema.index("thorax-right")] += [10.0, 0.0]
    pred.at[0, "kpts_xy"] = [float(v) for v in kpts.reshape(-1)]

    out = evaluate_predictions(pred, gt, {SCHEMA: schema}, cfg.eval)
    values = out.set_index(["scope", "metric"])["value"]
    assert values[("measurement:thorax width", "mape_median")] == pytest.approx(
        10.0 / gt_width, abs=1e-6
    )
    assert values[("measurement:head width", "mape_median")] == pytest.approx(0.0, abs=1e-9)


def test_symmetry_gap_detects_asymmetric_prediction(cfg, project) -> None:
    """Controle sans verite terrain : ecart gauche/droite des mesures predites."""
    gt, pred = _instance_frames(project)
    schema = load_schema(SCHEMA, project.configs)
    out = evaluate_predictions(pred, gt, {SCHEMA: schema}, cfg.eval)
    baseline = out.set_index(["scope", "metric"])["value"][("overall", "symmetry_gap_p90")]

    kpts = np.asarray(pred.loc[0, "kpts_xy"], dtype=float).reshape(-1, 2)
    kpts[schema.index("left-antenna-2")] += [50.0, 50.0]
    pred.at[0, "kpts_xy"] = [float(v) for v in kpts.reshape(-1)]
    out2 = evaluate_predictions(pred, gt, {SCHEMA: schema}, cfg.eval)
    degraded = out2.set_index(["scope", "metric"])["value"][("overall", "symmetry_gap_p90")]
    assert degraded > baseline


# --- normalisateur PCK (ADR-0009) -------------------------------------------
def test_normalizer_uses_thorax_width(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    kpts = np.zeros((1, schema.n_keypoints, 2))
    kpts[0, schema.index("thorax-left")] = [0.0, 0.0]
    kpts[0, schema.index("thorax-right")] = [40.0, 0.0]
    vis = np.ones((1, schema.n_keypoints), dtype=bool)
    spec = OmegaConf.create(
        {"name": "thorax_width", "type": "keypoint_distance",
         "keypoints": ["thorax-left", "thorax-right"], "fallback": "bbox_diag"}
    )
    bbox = np.array([[0.0, 0.0, 3.0, 4.0]])
    values, fell_back = compute_normalizer(spec, schema, kpts, vis, bbox)
    assert values[0] == pytest.approx(40.0)
    assert not fell_back[0]


def test_normalizer_falls_back_and_reports_it(project) -> None:
    """Un repli silencieux d'echelle fausserait la comparaison : il est compte."""
    schema = load_schema(SCHEMA, project.configs)
    kpts = np.zeros((1, schema.n_keypoints, 2))
    vis = np.ones((1, schema.n_keypoints), dtype=bool)
    vis[0, schema.index("thorax-left")] = False   # point de reference non annote
    spec = OmegaConf.create(
        {"name": "thorax_width", "type": "keypoint_distance",
         "keypoints": ["thorax-left", "thorax-right"], "fallback": "bbox_diag"}
    )
    bbox = np.array([[0.0, 0.0, 3.0, 4.0]])
    values, fell_back = compute_normalizer(spec, schema, kpts, vis, bbox)
    assert values[0] == pytest.approx(5.0)   # diagonale de bbox
    assert fell_back[0]


def test_fallback_rate_is_published(cfg, project) -> None:
    gt, pred = _instance_frames(project)
    gt.at[0, "kpts_vis"] = [0 if i == 6 else 2 for i in range(42)]
    out = evaluate_predictions(pred, gt, {SCHEMA: load_schema(SCHEMA, project.configs)}, cfg.eval)
    values = out.set_index(["scope", "metric"])["value"]
    assert values[("overall", "pck_normalizer_fallback_rate")] == pytest.approx(1.0)


# --- resolution commune (ADR-0013) ------------------------------------------
def test_divergent_image_size_is_refused(cfg, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(cfg)
    OmegaConf.update(cfg, "train.image_size", [512, 512])
    with pytest.raises(ContractError, match="resolution commune"):
        pipeline.cmd_train(cfg)


def test_divergent_image_size_allowed_when_guard_disabled(cfg, project) -> None:  # noqa: ARG001
    pipeline.cmd_split(cfg)
    OmegaConf.update(cfg, "train.image_size", [512, 512])
    OmegaConf.update(cfg, "strict.enforce_common_image_size", False)
    assert pipeline.cmd_train(cfg).run_id


# --- helpers ----------------------------------------------------------------
def _instance_frames(project) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une instance GT et sa prediction parfaite, geometrie anatomique jouet."""
    from insectpose.data.adapters.synthetic import _TEMPLATE_42

    schema = load_schema(SCHEMA, project.configs)
    pts = np.asarray(_TEMPLATE_42, dtype=float) * 40.0 + np.array([100.0, 100.0])
    flat = [float(v) for v in pts.reshape(-1)]
    gt = pd.DataFrame([{
        "schema_version": 1, "dataset": "coleoptera", "image_id": "coleoptera/a",
        "image_path": "raw/x.png", "image_width": 200, "image_height": 200,
        "instance_id": "coleoptera/a#0", "group_id": "coleoptera/a",
        "bbox_xywh": [0.0, 0.0, 200.0, 200.0], "kpts_xy": flat,
        "kpts_vis": [2] * schema.n_keypoints, "area": 40000.0, "keypoint_schema": SCHEMA,
        "split_source": "unknown", "qc_flags": "",
    }])
    pred = pd.DataFrame([{
        "schema_version": 1, "run_id": "r", "fold": 0, "split": "test",
        "dataset": "coleoptera", "image_id": "coleoptera/a", "pred_id": "p0",
        "bbox_xywh": [0.0, 0.0, 200.0, 200.0], "bbox_score": 1.0, "kpts_xy": flat,
        "kpts_score": [1.0] * schema.n_keypoints, "keypoint_schema": SCHEMA,
        "bbox_source": "predicted", "inference_ms": 1.0,
    }])
    return gt, pred
