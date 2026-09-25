"""Erreur sur les mesures morphometriques (ADR-0008).

Deux keypoints peuvent etre legerement decales sans que la mesure en souffre, et
inversement : cette metrique mesure ce que le projet produit vraiment en aval.

- `measurement_mape` : erreur relative absolue mediane/moyenne pred vs GT.
- `symmetry_gap` : ecart gauche/droite des mesures PREDITES. Calculable sans verite
  terrain, donc utilisable comme controle qualite sur des donnees non annotees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from insectpose.data.measurements import load_measurements, measure_all
from insectpose.evaluation.bundle import EvalBundle, record
from insectpose.registry import register_metric


def _instance_arrays(bundle: EvalBundle, pairs: list) -> dict[str, Any]:
    """Keypoints GT et predits des instances appariees, groupes par schema."""
    matched = bundle.matched_instances(pairs)
    if matched.empty:
        return {}
    out: dict[str, Any] = {}
    for schema_name, group in matched.groupby("keypoint_schema"):
        gt_rows = group["gt_row"].to_numpy()
        pred_rows = group["pred_row"].to_numpy()
        k = bundle.schemas[str(schema_name)].n_keypoints
        gt_vis = np.stack(
            bundle.gt.loc[gt_rows, "kpts_vis"].map(lambda v: np.asarray(v, int)).to_numpy()
        )
        out[str(schema_name)] = {
            "gt": bundle.gt_array("kpts_xy", gt_rows).reshape(-1, k, 2),
            "pred": bundle.pred_array("kpts_xy", pred_rows).reshape(-1, k, 2),
            "vis": gt_vis > 0,
        }
    return out


@register_metric("measurement_error")
def measurement_error(bundle: EvalBundle) -> list[dict[str, Any]]:
    """Erreur relative sur chaque mesure, plus une synthese par perimetre."""
    cfg = bundle.cfg.get("measurements")
    if cfg is None or not bool(cfg.enabled):
        return []
    spec = load_measurements(Path(str(cfg.file)))
    out: list[dict[str, Any]] = []

    for scope, pairs in bundle.scopes():
        per_measure: dict[str, list[float]] = {}
        for schema_name, arrays in _instance_arrays(bundle, pairs).items():
            index = spec.indices(bundle.schemas[schema_name])
            gt_values = measure_all(arrays["gt"], index)
            pred_values = measure_all(arrays["pred"], index)
            for measure, idx in index.items():
                # Mesure evaluable seulement si tous ses points sont annotes visibles.
                usable = arrays["vis"][:, idx].all(axis=1) & (gt_values[measure] > 1e-6)
                if not usable.any():
                    continue
                rel = np.abs(pred_values[measure][usable] - gt_values[measure][usable])
                per_measure.setdefault(measure, []).extend(
                    (rel / gt_values[measure][usable]).tolist()
                )

        if not per_measure:
            continue
        medians: list[float] = []
        for measure, errors in sorted(per_measure.items()):
            values = np.asarray(errors, dtype=float)
            medians.append(float(np.median(values)))
            if scope == "overall":
                out.append(
                    record(f"measurement:{measure}", "mape_median",
                           float(np.median(values)), int(values.size))
                )
        total = int(sum(len(v) for v in per_measure.values()))
        out.append(record(scope, "measurement_mape_median", float(np.median(medians)), total))
        out.append(record(scope, "measurement_mape_worst", float(np.max(medians)), total))
    return out


@register_metric("symmetry_gap")
def symmetry_gap(bundle: EvalBundle) -> list[dict[str, Any]]:
    """Ecart gauche/droite des mesures predites : controle sans verite terrain."""
    cfg = bundle.cfg.get("measurements")
    if cfg is None or not bool(cfg.enabled) or not bool(cfg.symmetry):
        return []
    spec = load_measurements(Path(str(cfg.file)))
    if not spec.symmetric_pairs:
        return []

    out: list[dict[str, Any]] = []
    for scope, pairs in bundle.scopes():
        gaps: list[float] = []
        for schema_name, arrays in _instance_arrays(bundle, pairs).items():
            index = spec.indices(bundle.schemas[schema_name])
            values = measure_all(arrays["pred"], index)
            for left, right in spec.symmetric_pairs:
                if left not in values or right not in values:
                    continue
                mean = (values[left] + values[right]) / 2.0
                usable = mean > 1e-6
                if usable.any():
                    gaps.extend(
                        (np.abs(values[left][usable] - values[right][usable]) / mean[usable]
                         ).tolist()
                    )
        if gaps:
            arr = np.asarray(gaps, dtype=float)
            # La mediane resume, le p90 detecte une asymetrie localisee sur une seule paire.
            out.append(record(scope, "symmetry_gap_median", float(np.median(arr)), int(arr.size)))
            out.append(record(scope, "symmetry_gap_p90", float(np.percentile(arr, 90)),
                              int(arr.size)))
    return out
