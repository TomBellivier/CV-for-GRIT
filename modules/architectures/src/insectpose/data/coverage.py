"""Couverture des keypoints et des mesures par dataset (ADR-0016).

Le schema est commun aux 4 ordres d'insectes, mais certains points n'existent pas
chez tous : ailes absentes, antennes non annotees, etc. Ces points ont `vis = 0`.

Consequences, toutes assumees explicitement plutot que subies :
- ils sont exclus de l'OKS et du PCK (jamais comptes comme erreur nulle) ;
- ils sont masques dans la loss des modeles poules, jamais remplaces par zero ;
- les mesures qui en dependent sont ininterpretables pour ce dataset ;
- leur PCK par keypoint est vide ou calcule sur trop peu d'instances pour etre lu.

Ce module produit l'artefact qui rend tout cela visible AVANT l'entrainement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insectpose.data.keypoints import KeypointSchema
from insectpose.data.measurements import MeasurementSet
from insectpose.utils.io import write_json, write_parquet
from insectpose.utils.logging import get_logger

log = get_logger("coverage")

ABSENT = "absent"
RARE = "rare"
PRESENT = "present"


def keypoint_coverage(annotations: pd.DataFrame, schemas: dict[str, KeypointSchema],
                      absent_max: float = 0.01, rare_max: float = 0.5) -> pd.DataFrame:
    """Taux d'annotation de chaque keypoint, par dataset. Aucun effet de bord.

    `rate` = part des instances ou le point est annote (vis > 0).
    `rate_visible` = part ou il est annote ET non occulte (vis == 2).
    """
    rows: list[dict[str, Any]] = []
    for (dataset, schema_name), group in annotations.groupby(["dataset", "keypoint_schema"]):
        schema = schemas[str(schema_name)]
        vis = np.stack(group["kpts_vis"].map(lambda v: np.asarray(v, int)).to_numpy())
        n = len(group)
        for k, name in enumerate(schema.names):
            rate = float((vis[:, k] > 0).mean())
            status = ABSENT if rate <= absent_max else (RARE if rate < rare_max else PRESENT)
            rows.append({
                "dataset": str(dataset), "keypoint_schema": str(schema_name),
                "keypoint_index": k, "keypoint": name, "n_instances": n,
                "n_annotated": int((vis[:, k] > 0).sum()), "rate": rate,
                "rate_visible": float((vis[:, k] == 2).mean()), "status": status,
            })
    return pd.DataFrame(rows)


def measurement_coverage(annotations: pd.DataFrame, schemas: dict[str, KeypointSchema],
                         spec: MeasurementSet, min_rate: float = 0.5) -> pd.DataFrame:
    """Part des instances ou une mesure est calculable (tous ses points annotes)."""
    rows: list[dict[str, Any]] = []
    for (dataset, schema_name), group in annotations.groupby(["dataset", "keypoint_schema"]):
        schema = schemas[str(schema_name)]
        index = spec.indices(schema)
        vis = np.stack(group["kpts_vis"].map(lambda v: np.asarray(v, int)).to_numpy()) > 0
        for measure, idx in index.items():
            rate = float(vis[:, idx].all(axis=1).mean())
            rows.append({
                "dataset": str(dataset), "measurement": measure, "n_instances": len(group),
                "rate": rate, "usable": bool(rate >= min_rate),
            })
    return pd.DataFrame(rows)


def summarize(kpt_cov: pd.DataFrame, meas_cov: pd.DataFrame | None = None) -> dict[str, Any]:
    """Synthese lisible : quels points et quelles mesures sont inexploitables et ou."""
    absent = kpt_cov[kpt_cov["status"] == ABSENT]
    rare = kpt_cov[kpt_cov["status"] == RARE]
    summary: dict[str, Any] = {
        "n_datasets": int(kpt_cov["dataset"].nunique()),
        "absent_by_dataset": {
            d: sorted(g["keypoint"]) for d, g in absent.groupby("dataset")
        },
        "rare_by_dataset": {
            d: {row.keypoint: round(row.rate, 3) for row in g.itertuples(index=False)}
            for d, g in rare.groupby("dataset")
        },
        # Points annotes dans AUCUN dataset : le modele les predirait sans supervision.
        "absent_everywhere": sorted(
            set(kpt_cov["keypoint"]) - set(kpt_cov.loc[kpt_cov["status"] != ABSENT, "keypoint"])
        ),
        # Points presents partout : socle comparable entre datasets.
        "present_everywhere": sorted(
            set(kpt_cov.loc[kpt_cov["status"] == PRESENT].groupby("keypoint")["dataset"].nunique()
                .pipe(lambda s: s[s == kpt_cov["dataset"].nunique()]).index)
        ),
    }
    if meas_cov is not None:
        summary["unusable_measurements_by_dataset"] = {
            d: sorted(g.loc[~g["usable"], "measurement"])
            for d, g in meas_cov.groupby("dataset")
            if (~g["usable"]).any()
        }
    return summary


def write_coverage(annotations: pd.DataFrame, schemas: dict[str, KeypointSchema],
                   out_dir: Path, spec: MeasurementSet | None = None,
                   absent_max: float = 0.01, rare_max: float = 0.5,
                   measurement_min_rate: float = 0.5) -> Path:
    """Ecrit le rapport de couverture et journalise ce qui est inexploitable.

    Effet de bord : ecrit <out_dir>/coverage_keypoints.parquet,
    coverage_measurements.parquet et coverage_summary.json.
    """
    kpt_cov = keypoint_coverage(annotations, schemas, absent_max, rare_max)
    out = write_parquet(out_dir / "coverage_keypoints.parquet", kpt_cov)
    meas_cov = None
    if spec is not None:
        meas_cov = measurement_coverage(annotations, schemas, spec, measurement_min_rate)
        write_parquet(out_dir / "coverage_measurements.parquet", meas_cov)

    summary = summarize(kpt_cov, meas_cov)
    write_json(out_dir / "coverage_summary.json", summary)

    for dataset, points in summary["absent_by_dataset"].items():
        if points:
            log.warning("[%s] %d keypoint(s) jamais annote(s) : %s", dataset, len(points),
                        ", ".join(points[:8]) + (" ..." if len(points) > 8 else ""))
    for dataset, points in summary["rare_by_dataset"].items():
        if points:
            log.warning("[%s] %d keypoint(s) rarement annote(s) : leur PCK par point sera "
                        "peu informatif : %s", dataset, len(points), list(points)[:6])
    if summary["absent_everywhere"]:
        log.warning("%d keypoint(s) absent(s) de TOUS les datasets : le modele les predirait "
                    "sans supervision. Envisager de les retirer du schema (nouvelle version) : %s",
                    len(summary["absent_everywhere"]), summary["absent_everywhere"])
    for dataset, measures in summary.get("unusable_measurements_by_dataset", {}).items():
        log.warning("[%s] %d mesure(s) non calculable(s) faute de points annotes : %s",
                    dataset, len(measures), measures[:6])
    return out
