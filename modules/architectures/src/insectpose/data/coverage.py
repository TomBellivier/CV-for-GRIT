"""Coverage of the keypoints and measurements per dataset (ADR-0016).

The schema is common to the 4 insect orders, but some points do not exist in all of
them: missing wings, antennae not annotated, etc. These points have `vis = 0`.

Consequences, all taken on explicitly rather than suffered:
- they are excluded from the OKS and the PCK (never counted as a zero error);
- they are masked in the loss of the pooled models, never replaced by zero;
- the measurements depending on them cannot be interpreted for that dataset;
- their per-keypoint PCK is empty or computed on too few instances to be read.

This module produces the artefact that makes all this visible BEFORE training.
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
    """Annotation rate of each keypoint, per dataset. No side effect.

    `rate` = share of the instances where the point is annotated (vis > 0).
    `rate_visible` = share where it is annotated AND not occluded (vis == 2).
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
    """Share of the instances where a measurement can be computed (all its points annotated)."""
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
    """Readable summary: which points and which measurements are unusable, and where."""
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
        # Points annotated in NO dataset: the model would predict them without supervision.
        "absent_everywhere": sorted(
            set(kpt_cov["keypoint"]) - set(kpt_cov.loc[kpt_cov["status"] != ABSENT, "keypoint"])
        ),
        # Points present everywhere: the base comparable across datasets.
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
    """Write the coverage report and log what is unusable.

    Side effect: writes <out_dir>/coverage_keypoints.parquet,
    coverage_measurements.parquet and coverage_summary.json.
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
            log.warning("[%s] %d keypoint(s) never annotated: %s", dataset, len(points),
                        ", ".join(points[:8]) + (" ..." if len(points) > 8 else ""))
    for dataset, points in summary["rare_by_dataset"].items():
        if points:
            log.warning("[%s] %d keypoint(s) rarely annotated: their per-point PCK will "
                        "not be very informative: %s", dataset, len(points), list(points)[:6])
    if summary["absent_everywhere"]:
        log.warning("%d keypoint(s) absent from ALL the datasets: the model would predict them "
                    "without supervision. Consider removing them from the schema (new version): %s",
                    len(summary["absent_everywhere"]), summary["absent_everywhere"])
    for dataset, measures in summary.get("unusable_measurements_by_dataset", {}).items():
        log.warning("[%s] %d measurement(s) not computable for lack of annotated points: %s",
                    dataset, len(measures), measures[:6])
    return out
